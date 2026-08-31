//! Command Code session parser
//!
//! Parses JSONL transcripts from `~/.commandcode/projects/<slug>/<session>.jsonl`.
//!
//! Command Code's on-disk transcript (version 3) is a stream of typed JSON
//! records, one per line:
//!
//! - `{"type":"session","version":3,"id":"<session>",...}` — the header, which
//!   carries the session id that every message line also repeats.
//! - `{"type":"message","id":...,"timestamp":...,"message":{"role":...,"content":[...]},"usage":{...},"model":"provider/model"}` —
//!   conversation entries. Assistant responses carry an authoritative `usage`
//!   block (`inputTokens`, `outputTokens`, `cacheReadTokens`,
//!   `cacheWriteTokens`, `costUsd`) plus the `model` in effect for that call.
//! - `{"type":"model_change","timestamp":...,"model":"provider/model"}` —
//!   mid-session model switches.
//!
//! Unlike earlier versions of this parser (which assumed Command Code never
//! persisted usage locally and estimated tokens at ~4 chars/token from message
//! text), the v3 transcript DOES persist per-request usage and cost. When a
//! message line carries a `usage` block, its token counts and `costUsd` are
//! used verbatim and marked provider-reported, so the cost is not overwritten
//! by tokscale's estimated pricing. `inputTokens` is CACHE-INCLUSIVE (it
//! already contains the `cacheReadTokens`/`cacheWriteTokens` buckets), so the
//! cache overlap is subtracted from the input bucket — the same
//! cache-inclusive normalization cline and zcode apply. Lines without `usage`
//! (user turns, tool results) contribute nothing themselves — the assistant
//! turn that follows them carries the full request accounting.
//!
//! Legacy flat-schema transcripts (`{"role":..., "content":..., ...}` with no
//! `type`/`message` nesting) are still parsed via the ~4 chars/token estimation
//! fallback, using the configured agent model from `~/.commandcode/config.json`.
//!
//! The model id is read from the line's own `model` field, falling back to the
//! most recent `model_change` event, then to `~/.commandcode/config.json` (the
//! configured agent model), then to "unknown". Gateway ids such as
//! `MiniMaxAI/MiniMax-M3-Free` are canonicalized by dropping the org prefix and
//! the `-Free` promo suffix so pricing resolves to the real model key, and the
//! provider hint carried in the id (e.g. `minimax`) is recovered the same way.

use super::utils::{
    estimate_tokens, file_modified_timestamp_ms, for_each_json_line, session_id_from_path,
    workspace_key_from_path,
};
use super::{workspace_label_from_key, UnifiedMessage};
use crate::TokenBreakdown;
use serde::Deserialize;
use std::path::Path;

const CLIENT_ID: &str = "commandcode";
const PROVIDER_ID: &str = "command-code";
const UNKNOWN_MODEL: &str = "unknown";

/// One JSONL record in a Command Code v3 transcript.
///
/// Only the fields this parser needs are modeled; everything else
/// (`parentId`, `meta`, `effort`, …) is ignored by serde. The top-level
/// `role`/`content` fields exist for legacy flat-schema transcripts (see
/// `parse_commandcode_file`).
#[derive(Debug, Deserialize)]
struct CommandCodeEntry {
    #[serde(rename = "type")]
    record_type: Option<String>,
    id: Option<String>,
    timestamp: Option<String>,
    /// The conversation entry, present on `type == "message"` records.
    message: Option<CommandCodeMessage>,
    /// Authoritative usage block, present on assistant response lines.
    usage: Option<CommandCodeUsage>,
    /// The model in effect for this record (message or model_change).
    model: Option<String>,
    /// Legacy flat-schema fields: `{"role":..., "content":..., "sessionId":...}`.
    role: Option<String>,
    content: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct CommandCodeMessage {
    role: Option<String>,
    content: Option<serde_json::Value>,
}

/// The camelCase usage block Command Code persists on assistant responses.
#[derive(Debug, Deserialize)]
struct CommandCodeUsage {
    #[serde(rename = "inputTokens")]
    input_tokens: Option<i64>,
    #[serde(rename = "outputTokens")]
    output_tokens: Option<i64>,
    #[serde(rename = "cacheReadTokens")]
    cache_read_tokens: Option<i64>,
    #[serde(rename = "cacheWriteTokens")]
    cache_write_tokens: Option<i64>,
    #[serde(rename = "costUsd")]
    cost_usd: Option<f64>,
}

impl CommandCodeUsage {
    /// Token breakdown with every field clamped at zero.
    ///
    /// Command Code's `inputTokens` is CACHE-INCLUSIVE: it already contains
    /// the `cacheReadTokens`/`cacheWriteTokens` buckets (verified against real
    /// transcripts — cache is ≥90% of input on most lines, and no line has
    /// input < cache). tokscale's `TokenBreakdown` expects non-overlapping
    /// buckets whose `total()` sums without double-counting, so the cache
    /// overlap is subtracted from `input`, mirroring `cline`'s
    /// `input - cache_read - cache_write` and `zcode`'s
    /// `normalize_zcode_input_and_output`.
    fn to_breakdown(&self) -> Option<TokenBreakdown> {
        // Provider-reported values come from an untrusted transcript; clamp to
        // a plausible ceiling so one tampered line cannot poison aggregates
        // (and so the sum below cannot overflow). ~1e12 tokens exceeds any
        // real session by orders of magnitude.
        const TOKEN_CEILING: i64 = 1_000_000_000_000;
        let raw_input = self.input_tokens.unwrap_or(0).max(0).min(TOKEN_CEILING);
        let output = self.output_tokens.unwrap_or(0).max(0).min(TOKEN_CEILING);
        let cache_read = self
            .cache_read_tokens
            .unwrap_or(0)
            .max(0)
            .min(TOKEN_CEILING);
        let cache_write = self
            .cache_write_tokens
            .unwrap_or(0)
            .max(0)
            .min(TOKEN_CEILING);
        let overlap = cache_read.saturating_add(cache_write);
        let input = raw_input.saturating_sub(overlap.min(raw_input));
        // Saturating sum: adversarial near-i64::MAX fields must not overflow
        // (panic in debug, wrap in release) — same discipline as
        // `TokenBreakdown::total`.
        if input
            .saturating_add(output)
            .saturating_add(cache_read)
            .saturating_add(cache_write)
            == 0
        {
            return None;
        }
        Some(TokenBreakdown {
            input,
            output,
            cache_read,
            cache_write,
            reasoning: 0,
        })
    }

    /// Authoritative cost in USD, or `None` when absent/non-finite/beyond a
    /// plausible ceiling.
    ///
    /// An explicit `0.0` IS a reported cost (free-promo models such as
    /// `MiniMax-M3-Free` bill nothing) and must stay provider-reported so the
    /// pricing lane does not reprice it at the paid rate — the same convention
    /// junie/cursor/fx pin. Negative costs are rejected (`-1` is not a real
    /// bill); so is anything above a $1M ceiling, which no single request can
    /// reach and which would otherwise flow verbatim into aggregates.
    fn reported_cost(&self) -> Option<f64> {
        const COST_CEILING: f64 = 1_000_000.0;
        self.cost_usd
            .filter(|cost| cost.is_finite() && *cost >= 0.0 && *cost <= COST_CEILING)
    }
}

#[derive(Debug, Deserialize)]
struct CommandCodeConfig {
    model: Option<String>,
}

pub fn parse_commandcode_file(path: &Path) -> Vec<UnifiedMessage> {
    // The `*.jsonl` glob also matches the per-session checkpoint log
    // (`<session>.checkpoints.jsonl`), which is a snapshot stream, not a
    // transcript. Skip it explicitly rather than relying on schema mismatch.
    if path
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.ends_with(".checkpoints.jsonl"))
    {
        return Vec::new();
    }

    let fallback_timestamp = file_modified_timestamp_ms(path);
    let session_id_from_path = session_id_from_path(path);
    let workspace_key = workspace_key_from_path(path);
    let workspace_label = workspace_key.as_deref().and_then(workspace_label_from_key);
    // Hoisted out of the loop: the legacy fallback path reads the configured
    // agent model once per file, not once per assistant line.
    let config_model = model_from_config(path);

    let mut messages = Vec::new();
    let mut session_id: Option<String> = None;
    let mut model_id: Option<String> = None;
    // Char count of the *new* context added since the previous assistant
    // response (the user prompt plus any tool results for this turn). Only used
    // for the estimation fallback on transcripts without a `usage` block.
    let mut turn_input_chars: usize = 0;
    // Tracks whether the most recent non-assistant entry started a user turn,
    // used to mark the first assistant response of each turn.
    let mut pending_turn_start = false;
    let mut assistant_index = 0usize;

    for_each_json_line(path, &mut |_index, trimmed| {
        let entry = match serde_json::from_str::<CommandCodeEntry>(trimmed) {
            Ok(entry) => entry,
            Err(_) => return,
        };

        // The session id comes from the `session` header record. A message
        // record's `id` is a per-message UUID, not a session id — never use it
        // as a fallback when the header is missing/corrupt.
        if entry.record_type.as_deref() == Some("session") {
            if let Some(id) = entry.id.as_deref().filter(|id| !id.is_empty()) {
                session_id = Some(id.to_string());
            }
        }

        // Track the most-recently-seen model: `model_change` records update it
        // without emitting anything, and message records carry the model in
        // effect for that call.
        if let Some(model) = entry.model.as_deref().filter(|model| !model.is_empty()) {
            model_id = Some(model.to_string());
        }

        match entry.record_type.as_deref() {
            // A `model_change` record updates the model in effect; nothing to
            // emit (handled above by the shared model tracking).
            Some("model_change") => {}
            // v3 `message` records, plus legacy flat-schema lines that carry no
            // `type` at all (`_` arm). Legacy lines put `role`/`content` at the
            // top level; v3 nests them under `message`.
            Some("message") | _ => {
                let message = match entry.message.as_ref() {
                    Some(message) => message,
                    None if entry.role.is_some() || entry.content.is_some() => {
                        &legacy_message(&entry)
                    }
                    None => return,
                };
                let Some(role) = message.role.as_deref() else {
                    return;
                };

                let chars = message.content.as_ref().map(content_chars).unwrap_or(0);

                if role == "assistant" {
                    // Prefer authoritative usage; fall back to estimating from
                    // the per-turn delta of message content when no usage block
                    // is present (legacy transcripts).
                    let breakdown = entry
                        .usage
                        .as_ref()
                        .and_then(CommandCodeUsage::to_breakdown)
                        .unwrap_or_else(|| {
                            let input = estimate_tokens(turn_input_chars);
                            let output = estimate_tokens(chars);
                            TokenBreakdown {
                                input,
                                output,
                                cache_read: 0,
                                cache_write: 0,
                                reasoning: 0,
                            }
                        });
                    turn_input_chars = 0;

                    if breakdown.total() == 0 {
                        // No message is emitted, so the next real assistant
                        // message in this turn must keep its is_turn_start
                        // marker — mirroring zcode's drop path.
                        return;
                    }

                    let resolved_session = session_id
                        .clone()
                        .unwrap_or_else(|| session_id_from_path.clone());
                    // Resolve the raw model: the line's own `model` (or the most
                    // recent `model_change`), else the configured agent model,
                    // else "unknown".
                    let raw_model = match model_id {
                        Some(ref model) => model.clone(),
                        None => config_model
                            .clone()
                            .unwrap_or_else(|| UNKNOWN_MODEL.to_string()),
                    };
                    let resolved_model = canonicalize_model(&raw_model);
                    // Recover the real provider from the model id (e.g.
                    // `MiniMaxAI/MiniMax-M3-Free` -> `minimax`) so pricing
                    // resolves to that provider's catalog. The client's own
                    // `command-code` provider is not a pricing provider, so
                    // without this a MiniMax model would never reach a
                    // `minimax/...` key. Falls back to `command-code` when
                    // nothing is inferred.
                    let provider_id =
                        provider_hint_for_model(&resolved_model).unwrap_or(PROVIDER_ID);
                    let timestamp = entry
                        .timestamp
                        .as_deref()
                        .and_then(parse_rfc3339_ms)
                        .unwrap_or(fallback_timestamp);

                    let cost = entry.usage.as_ref().and_then(|u| u.reported_cost());
                    let mut message = UnifiedMessage::new_with_dedup(
                        CLIENT_ID,
                        resolved_model,
                        provider_id,
                        resolved_session.clone(),
                        timestamp,
                        breakdown,
                        cost.unwrap_or(0.0),
                        Some(format!("{}:{}", resolved_session, assistant_index)),
                    );
                    message.message_count = 1;
                    message.is_turn_start = pending_turn_start;
                    message.set_workspace(workspace_key.clone(), workspace_label.clone());
                    if cost.is_some() {
                        message.mark_provider_reported_cost();
                    }
                    messages.push(message);

                    assistant_index += 1;
                    pending_turn_start = false;
                } else if role == "user" {
                    // A tool result is delivered under `role: "user"` but is a
                    // continuation of the current turn, not a new prompt — it
                    // must not start a new turn. Only a genuine user text
                    // prompt does.
                    if !is_tool_result(&message.content) {
                        pending_turn_start = true;
                    }
                    turn_input_chars += chars;
                } else {
                    // Tool results (and any other roles) are part of the new
                    // context the model sees on the next turn.
                    turn_input_chars += chars;
                }
            }
        }
    });

    messages
}

/// Adapt a legacy flat-schema record (`{"role":..., "content":..., ...}` with
/// no `type`/`message` nesting) to the nested-message shape the parser shares
/// with v3 records.
fn legacy_message(entry: &CommandCodeEntry) -> CommandCodeMessage {
    CommandCodeMessage {
        role: entry.role.clone(),
        content: entry.content.clone(),
    }
}

/// Whether a user-role content block is a tool result (a continuation of the
/// current turn) rather than a genuine new user prompt.
///
/// Command Code delivers tool results under `role: "user"` with content blocks
/// of `type: "tool_result"`, so a plain role check would start a new turn for
/// every tool result. The content block's type is decisive: a real prompt
/// carries `type: "text"` (or a string/object form in older transcripts).
fn is_tool_result(content: &Option<serde_json::Value>) -> bool {
    match content {
        Some(serde_json::Value::Array(items)) => items
            .iter()
            .filter_map(serde_json::Value::as_object)
            .any(|block| {
                block.get("type").and_then(serde_json::Value::as_str) == Some("tool_result")
            }),
        // A single bare `{"type":"tool_result",...}` object (not array-wrapped).
        Some(serde_json::Value::Object(map)) => {
            map.get("type").and_then(serde_json::Value::as_str) == Some("tool_result")
        }
        _ => false,
    }
}

/// Char count of a message's `content` for token estimation, measured from its
/// canonical JSON serialization. Counting the serialized form keeps every
/// prompt-bearing byte the model receives — object keys (`command`, `path`, …),
/// tool-call arguments, tool-result payloads, and numeric/boolean values — and
/// avoids guessing which fields are structural versus content.
///
/// Genuinely empty content (null, `""`, `[]`, `{}`) counts as zero so that
/// contentless turns are not charged for their structural brackets.
fn content_chars(content: &serde_json::Value) -> usize {
    match content {
        serde_json::Value::Null => 0,
        serde_json::Value::String(s) if s.is_empty() => 0,
        serde_json::Value::Array(items) if items.is_empty() => 0,
        serde_json::Value::Object(map) if map.is_empty() => 0,
        _ => serde_json::to_string(content)
            .map(|serialized| serialized.chars().count())
            .unwrap_or(0),
    }
}

/// Canonicalize the model id for pricing. Command Code reports gateway ids such
/// as `MiniMaxAI/MiniMax-M3-Free`; the `-Free` suffix is a temporary promo and
/// the org prefix is not a key tokscale's pricing resolver recognizes verbatim.
/// Dropping the org segment yields the real paid model (e.g. `MiniMax-M3`) so
/// output pricing resolves; the provider hint that the org segment carried
/// (e.g. `minimax`) is recovered separately by [`provider_hint_for_model`] and
/// applied to `provider_id`, so pricing keys like `minimax/minimax-m3` are
/// still reached.
fn canonicalize_model(model: &str) -> String {
    let base = model.rsplit('/').next().unwrap_or(model);
    // Char-safe, case-insensitive suffix strip. The original code byte-sliced
    // `base[base.len() - N..]` guarded only by a length check, which panics on a
    // non-ASCII model id from the untrusted `~/.commandcode/config.json` when
    // the byte index lands mid-codepoint. `-free` is pure ASCII, so when the
    // lowercased tail matches, the matched bytes are guaranteed ASCII and
    // `base.len() - PROMO_SUFFIX.len()` is a valid char boundary.
    const PROMO_SUFFIX: &str = "-free";
    if base.len() > PROMO_SUFFIX.len()
        && base
            .get(base.len() - PROMO_SUFFIX.len()..)
            .is_some_and(|tail| tail.eq_ignore_ascii_case(PROMO_SUFFIX))
    {
        base[..base.len() - PROMO_SUFFIX.len()].to_string()
    } else {
        base.to_string()
    }
}

/// Recover the provider hint that the model id carries (e.g.
/// `MiniMaxAI/MiniMax-M3-Free` -> `minimax`) so pricing resolves to the real
/// provider's catalog. Command Code's own `command-code` provider id is not a
/// pricing provider, so without this hint a MiniMax model would never reach a
/// `minimax/...` pricing key. Returns `None` when no known provider can be
/// inferred, leaving the default `command-code` provider in place.
fn provider_hint_for_model(model: &str) -> Option<&'static str> {
    crate::provider_identity::inferred_provider_from_model(model)
}

fn parse_rfc3339_ms(timestamp: &str) -> Option<i64> {
    chrono::DateTime::parse_from_rfc3339(timestamp)
        .ok()
        .map(|dt| dt.timestamp_millis())
}

/// Read the configured agent model from `~/.commandcode/config.json`.
///
/// `session_path` is normally `<root>/.commandcode/projects/<slug>/<session>.jsonl`,
/// but the scanner walks `projects` recursively, so a transcript can sit at any
/// depth. Count-parents guessing would land on a *different* `config.json` for
/// nested paths (silently misattributing the model). Instead, walk up until the
/// parent directory is named `projects` — the Command Code root is one level
/// above it — and read `<root>/config.json` from there. Returns `None` when no
/// ancestor named `projects` is found (so the caller's "unknown" fallback
/// applies).
fn model_from_config(session_path: &Path) -> Option<String> {
    let mut dir = session_path.parent()?;
    // Find the `projects` directory that anchors the Command Code root.
    loop {
        if dir.file_name().and_then(|name| name.to_str()) == Some("projects") {
            break;
        }
        dir = dir.parent()?;
    }
    let commandcode_root = dir.parent()?;
    let config_path = commandcode_root.join("config.json");
    let bytes = std::fs::read(config_path).ok()?;
    let config: CommandCodeConfig = serde_json::from_slice(&bytes).ok()?;
    config.model.filter(|model| !model.trim().is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::io::Write;
    use tempfile::TempDir;

    fn write_session(dir: &TempDir, slug: &str, session: &str, jsonl: &str) -> std::path::PathBuf {
        let project_dir = dir.path().join("projects").join(slug);
        std::fs::create_dir_all(&project_dir).unwrap();
        let path = project_dir.join(format!("{session}.jsonl"));
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(jsonl.as_bytes()).unwrap();
        path
    }

    fn write_config(dir: &TempDir, model: &str) {
        let path = dir.path().join("config.json");
        let mut file = std::fs::File::create(&path).unwrap();
        write!(file, r#"{{"provider":"command-code","model":"{model}"}}"#).unwrap();
    }

    /// Build a realistic v3 transcript line for an assistant response carrying
    /// authoritative usage and cost.
    fn assistant_line(
        id: &str,
        timestamp: &str,
        model: &str,
        text: &str,
        input: i64,
        output: i64,
        cache_read: i64,
        cost_usd: f64,
    ) -> String {
        json!({
            "type": "message",
            "id": id,
            "parentId": null,
            "timestamp": timestamp,
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": text}]
            },
            "usage": {
                "inputTokens": input,
                "outputTokens": output,
                "cacheReadTokens": cache_read,
                "cacheWriteTokens": 0,
                "costUsd": cost_usd
            },
            "model": model
        })
        .to_string()
    }

    fn user_line(id: &str, timestamp: &str, text: &str) -> String {
        json!({
            "type": "message",
            "id": id,
            "parentId": null,
            "timestamp": timestamp,
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": text}]
            }
        })
        .to_string()
    }

    /// A tool-result record, as Command Code writes it after a tool call.
    fn tool_result_line(id: &str, timestamp: &str, tool_use_id: &str, text: &str) -> String {
        json!({
            "type": "message",
            "id": id,
            "parentId": null,
            "timestamp": timestamp,
            "message": {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": [{"type": "text", "text": text}]
                }]
            }
        })
        .to_string()
    }

    /// Full realistic v3 session: header, model_change, user turn, tool call +
    /// tool result, then the assistant response carrying authoritative usage.
    #[test]
    fn test_parse_realistic_v3_session_with_tool_turns() {
        let dir = TempDir::new().unwrap();
        write_config(&dir, "deepseek/deepseek-v4-flash");
        let jsonl = format!(
            "{}\n{}\n{}\n{}\n{}\n{}",
            json!({"type": "session", "version": 3, "id": "sess-1", "timestamp": "2026-08-31T04:36:38.441Z", "cwd": "/home/al/learning"}),
            json!({"type": "model_change", "id": "m1", "parentId": null, "timestamp": "2026-08-31T04:37:00.000Z", "model": "deepseek/deepseek-v4-flash"}),
            user_line("u1", "2026-08-31T04:39:16.867Z", "list the repo"),
            json!({
                "type": "message",
                "id": "t1",
                "parentId": "u1",
                "timestamp": "2026-08-31T04:39:18.000Z",
                "message": {
                    "role": "assistant",
                    "content": [{
                        "type": "tool_use",
                        "id": "call_00_abc",
                        "name": "read_directory",
                        "input": {"path": "/home/al/learning"}
                    }],
                    "meta": {"source": "model"}
                },
                "usage": {
                    "inputTokens": 21000,
                    "outputTokens": 90,
                    "cacheReadTokens": 4000,
                    "cacheWriteTokens": 0,
                    "costUsd": 0.002
                },
                "model": "deepseek/deepseek-v4-flash"
            }),
            tool_result_line(
                "r1",
                "2026-08-31T04:39:19.000Z",
                "call_00_abc",
                "Found 28 items"
            ),
            assistant_line(
                "a1",
                "2026-08-31T04:39:20.351Z",
                "deepseek/deepseek-v4-flash",
                "There are 28 items.",
                28534,
                205,
                7424,
                0.006464748
            ),
        );
        let path = write_session(&dir, "home-al-learning", "sess-1", &jsonl);

        let messages = parse_commandcode_file(&path);

        // Two assistant records: the tool_use call (turn start) and the final
        // text response. Both carry authoritative usage.
        assert_eq!(messages.len(), 2);
        let tool_call = &messages[0];
        assert_eq!(tool_call.model_id, "deepseek-v4-flash");
        assert_eq!(tool_call.provider_id, "deepseek");
        // inputTokens 21000 is cache-inclusive: 17000 fresh + 4000 cache read.
        assert_eq!(tool_call.tokens.input, 17000);
        assert_eq!(tool_call.tokens.output, 90);
        assert_eq!(tool_call.tokens.cache_read, 4000);
        assert!((tool_call.cost - 0.002).abs() < 1e-9);
        assert!(tool_call.has_authoritative_cost());
        assert!(tool_call.is_turn_start);
        assert_eq!(tool_call.dedup_key.as_deref(), Some("sess-1:0"));

        let msg = &messages[1];
        assert_eq!(msg.model_id, "deepseek-v4-flash");
        assert_eq!(msg.provider_id, "deepseek");
        assert_eq!(msg.session_id, "sess-1");
        // Authoritative usage from the assistant line, not estimated from the
        // tool-result text (which is far larger than 205 output tokens).
        // 28534 inputTokens is cache-inclusive: 21110 fresh + 7424 cache read.
        assert_eq!(msg.tokens.input, 21110);
        assert_eq!(msg.tokens.output, 205);
        assert_eq!(msg.tokens.cache_read, 7424);
        assert!((msg.cost - 0.006464748).abs() < 1e-9);
        assert!(msg.has_authoritative_cost());
        assert!(!msg.is_turn_start);
        assert_eq!(msg.dedup_key.as_deref(), Some("sess-1:1"));
    }

    #[test]
    fn test_parse_v3_transcript_with_authoritative_usage() {
        let dir = TempDir::new().unwrap();
        write_config(&dir, "deepseek/deepseek-v4-flash");
        let jsonl = format!(
            "{}\n{}\n{}",
            json!({"type": "session", "version": 3, "id": "sess-1", "cwd": "/home/al/learning"}),
            user_line("u1", "2026-08-31T04:39:16.867Z", "hello"),
            assistant_line(
                "a1",
                "2026-08-31T04:39:20.351Z",
                "deepseek/deepseek-v4-flash",
                "hi there",
                28534,
                205,
                7424,
                0.006464748
            ),
        );
        let path = write_session(&dir, "home-al-learning", "sess-1", &jsonl);

        let messages = parse_commandcode_file(&path);

        assert_eq!(messages.len(), 1);
        let msg = &messages[0];
        assert_eq!(msg.client, "commandcode");
        assert_eq!(msg.provider_id, "deepseek");
        assert_eq!(msg.model_id, "deepseek-v4-flash");
        assert_eq!(msg.session_id, "sess-1");
        // Authoritative counts, not estimates. `inputTokens` is cache-inclusive
        // (28534 = 21110 fresh + 7424 cache read), so `input` is the fresh
        // remainder, exactly like cline/zcode normalize.
        assert_eq!(msg.tokens.input, 21110);
        assert_eq!(msg.tokens.output, 205);
        assert_eq!(msg.tokens.cache_read, 7424);
        assert_eq!(msg.tokens.cache_write, 0);
        assert_eq!(msg.tokens.reasoning, 0);
        // Non-overlapping buckets: total() must not double-count the cache.
        assert_eq!(msg.tokens.total(), 28739); // 21110 + 205 + 7424
                                               // Authoritative cost is embedded and marked provider-reported.
        assert!((msg.cost - 0.006464748).abs() < 1e-9);
        assert!(msg.has_authoritative_cost());
        assert!(msg.is_turn_start);
        assert_eq!(msg.message_count, 1);
        assert_eq!(msg.timestamp, 1788151160351); // 2026-08-31T04:39:20.351Z
        assert_eq!(msg.workspace_key.as_deref(), Some("home-al-learning"));
    }

    #[test]
    fn test_parse_v3_transcript_model_change_updates_model() {
        let dir = TempDir::new().unwrap();
        write_config(&dir, "deepseek/deepseek-v4-flash");
        let jsonl = format!(
            "{}\n{}\n{}\n{}\n{}",
            json!({"type": "session", "version": 3, "id": "sess-1"}),
            json!({"type": "model_change", "id": "m1", "timestamp": "2026-08-31T03:32:29.603Z", "model": "minimax/minimax-m3-free"}),
            user_line("u1", "2026-08-31T03:33:00.000Z", "first"),
            assistant_line(
                "a1",
                "2026-08-31T03:33:05.000Z",
                "minimax/minimax-m3-free",
                "resp one",
                100,
                10,
                0,
                0.001
            ),
            json!({"type": "model_change", "id": "m2", "timestamp": "2026-08-31T03:34:07.747Z", "model": "deepseek/deepseek-v4-flash"}),
        );
        let path = write_session(&dir, "proj", "sess-1", &jsonl);

        let messages = parse_commandcode_file(&path);

        assert_eq!(messages.len(), 1);
        // The model in effect at the assistant line is the model_change one.
        assert_eq!(messages[0].model_id, "minimax-m3");
        assert_eq!(messages[0].provider_id, "minimax");
    }

    #[test]
    fn test_canonicalize_model_strips_org_prefix_and_free_promo_suffix() {
        assert_eq!(
            canonicalize_model("MiniMaxAI/MiniMax-M3-Free"),
            "MiniMax-M3"
        );
        assert_eq!(
            canonicalize_model("minimaxai/minimax-m3-free"),
            "minimax-m3"
        );
        assert_eq!(canonicalize_model("MiniMaxAI/MiniMax-M2.5"), "MiniMax-M2.5");
        assert_eq!(canonicalize_model("taste-1"), "taste-1");
        assert_eq!(canonicalize_model("MiniMax-M3-FrEe"), "MiniMax-M3");
    }

    #[test]
    fn test_canonicalize_model_does_not_panic_on_non_ascii() {
        assert_eq!(canonicalize_model("vendor/modèle"), "modèle");
        assert_eq!(canonicalize_model("café-🚀"), "café-🚀");
        assert_eq!(canonicalize_model("café-free"), "café");
    }

    #[test]
    fn test_content_chars_counts_keys_numbers_and_nested_payloads() {
        assert!(content_chars(&json!([{"value": 12345}])) > 0);
        let small = content_chars(&json!([{"a": "x"}]));
        let large = content_chars(&json!([{"command": "run", "args": ["a", "b"], "n": 42}]));
        assert!(large > small);
    }

    #[test]
    fn test_checkpoint_files_are_skipped() {
        let dir = TempDir::new().unwrap();
        write_config(&dir, "model-x");
        let project_dir = dir.path().join("projects").join("proj");
        std::fs::create_dir_all(&project_dir).unwrap();
        let path = project_dir.join("s.checkpoints.jsonl");
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(
            br#"{"type":"checkpoint","messageId":"m","snapshot":"snap","isSnapshotUpdate":false}"#,
        )
        .unwrap();

        let messages = parse_commandcode_file(&path);
        assert!(messages.is_empty());
    }

    #[test]
    fn test_missing_config_and_model_falls_back_to_unknown_model() {
        let dir = TempDir::new().unwrap();
        let jsonl = format!(
            "{}\n{}\n{}",
            json!({"type": "session", "version": 3, "id": "s"}),
            user_line("u1", "2026-08-31T00:00:00Z", "hello"),
            assistant_line("a1", "2026-08-31T00:00:05Z", "", "world", 10, 5, 0, 0.0),
        );
        let path = write_session(&dir, "proj", "s", &jsonl);

        let messages = parse_commandcode_file(&path);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].model_id, "unknown");
        assert_eq!(messages[0].provider_id, "command-code");
    }

    #[test]
    fn test_skips_malformed_lines_without_panicking() {
        let dir = TempDir::new().unwrap();
        write_config(&dir, "model-x");
        let jsonl = concat!(
            "not valid json at all\n",
            r#"{"type":"session","version":3,"id":"s"}"#,
            "\n",
        );
        let path = write_session(&dir, "proj", "s", jsonl);

        let messages = parse_commandcode_file(&path);
        assert!(messages.is_empty());
    }

    #[test]
    fn test_estimated_fallback_when_usage_absent() {
        let dir = TempDir::new().unwrap();
        write_config(&dir, "model-x");
        let jsonl = format!(
            "{}\n{}\n{}",
            json!({"type": "session", "version": 3, "id": "s"}),
            json!({
                "type": "message",
                "id": "u1",
                "timestamp": "2026-08-31T00:00:00Z",
                "message": {"role": "user", "content": [{"type": "text", "text": "12345678"}]}
            }),
            json!({
                "type": "message",
                "id": "a1",
                "timestamp": "2026-08-31T00:00:05Z",
                "message": {"role": "assistant", "content": [{"type": "text", "text": "abcd"}]},
                "model": "model-x"
            }),
        );
        let path = write_session(&dir, "proj", "s", &jsonl);

        let messages = parse_commandcode_file(&path);
        assert_eq!(messages.len(), 1);
        let msg = &messages[0];
        // Estimated: input from the user turn's content, output from assistant.
        assert_eq!(
            msg.tokens.input,
            estimate_tokens(content_chars(
                &json!([{"type": "text", "text": "12345678"}])
            ))
        );
        assert_eq!(
            msg.tokens.output,
            estimate_tokens(content_chars(&json!([{"type": "text", "text": "abcd"}])))
        );
        assert_eq!(msg.cost, 0.0);
        assert!(!msg.has_authoritative_cost());
        assert!(msg.is_turn_start);
    }

    /// An explicit `costUsd: 0` is a real reported cost (free-promo models bill
    /// nothing) and must stay provider-reported — otherwise the pricing lane
    /// reprices the call at the paid rate. Mirrors the junie/cursor/fx
    /// convention.
    #[test]
    fn test_explicit_zero_cost_is_provider_reported() {
        let dir = TempDir::new().unwrap();
        write_config(&dir, "minimax/minimax-m3-free");
        let jsonl = format!(
            "{}\n{}\n{}",
            json!({"type": "session", "version": 3, "id": "s"}),
            user_line("u1", "2026-08-31T00:00:00Z", "hi"),
            json!({
                "type": "message",
                "id": "a1",
                "timestamp": "2026-08-31T00:00:05Z",
                "message": {"role": "assistant", "content": [{"type": "text", "text": "yo"}]},
                "usage": {
                    "inputTokens": 100,
                    "outputTokens": 10,
                    "cacheReadTokens": 0,
                    "cacheWriteTokens": 0,
                    "costUsd": 0
                },
                "model": "minimax/minimax-m3-free"
            }),
        );
        let path = write_session(&dir, "proj", "s", &jsonl);

        let messages = parse_commandcode_file(&path);
        assert_eq!(messages.len(), 1);
        let msg = &messages[0];
        assert_eq!(msg.cost, 0.0);
        assert!(
            msg.has_authoritative_cost(),
            "an explicit zero cost is provider-reported, not repriced"
        );
        // The free-promo model still canonicalizes (org + suffix stripped).
        assert_eq!(msg.model_id, "minimax-m3");
    }

    /// `inputTokens` is cache-inclusive, so the breakdown must subtract the
    /// cache buckets from input — otherwise `total()` double-counts cache.
    #[test]
    fn test_usage_input_is_cache_inclusive() {
        let dir = TempDir::new().unwrap();
        write_config(&dir, "model-x");
        let jsonl = format!(
            "{}\n{}\n{}",
            json!({"type": "session", "version": 3, "id": "s"}),
            user_line("u1", "2026-08-31T00:00:00Z", "hi"),
            json!({
                "type": "message",
                "id": "a1",
                "timestamp": "2026-08-31T00:00:05Z",
                "message": {"role": "assistant", "content": [{"type": "text", "text": "yo"}]},
                "usage": {
                    "inputTokens": 1000,
                    "outputTokens": 50,
                    "cacheReadTokens": 900,
                    "cacheWriteTokens": 10,
                    "costUsd": 0.01
                },
                "model": "model-x"
            }),
        );
        let path = write_session(&dir, "proj", "s", &jsonl);

        let messages = parse_commandcode_file(&path);
        assert_eq!(messages.len(), 1);
        let msg = &messages[0];
        // 1000 inputTokens = 90 fresh + 900 cache read + 10 cache write.
        assert_eq!(msg.tokens.input, 90);
        assert_eq!(msg.tokens.cache_read, 900);
        assert_eq!(msg.tokens.cache_write, 10);
        // Non-overlapping: total() is fresh + output + both caches, no double count.
        assert_eq!(msg.tokens.total(), 1050);
    }

    /// A dropped zero-total assistant line must not consume the turn-start
    /// marker: the next real assistant response in the same turn keeps
    /// `is_turn_start`, mirroring zcode's drop path.
    #[test]
    fn test_turn_start_survives_dropped_zero_assistant_line() {
        let dir = TempDir::new().unwrap();
        write_config(&dir, "model-x");
        // The first assistant line has NO preceding context and empty content,
        // so estimation yields 0 tokens and it is dropped. Its turn-start
        // marker must survive for the next response in the same turn.
        let jsonl = format!(
            "{}\n{}\n{}\n{}",
            json!({"type": "session", "version": 3, "id": "s"}),
            json!({
                "type": "message",
                "id": "a0",
                "timestamp": "2026-08-31T00:00:01Z",
                "message": {"role": "assistant", "content": []}
            }),
            user_line("u1", "2026-08-31T00:00:02Z", "hi"),
            assistant_line(
                "a1",
                "2026-08-31T00:00:05Z",
                "model-x",
                "response",
                10,
                5,
                0,
                0.0
            ),
        );
        let path = write_session(&dir, "proj", "s", &jsonl);

        let messages = parse_commandcode_file(&path);
        assert_eq!(messages.len(), 1);
        // The zero-total assistant line was dropped; the surviving response
        // belongs to the turn started by u1 and must be marked as its start.
        assert!(messages[0].is_turn_start);
    }

    /// Legacy flat-schema transcripts (`{"role":..., "content":...}` with no
    /// `type`/`message` nesting) still parse via estimation.
    #[test]
    fn test_legacy_flat_schema_still_parses_with_estimation() {
        let dir = TempDir::new().unwrap();
        write_config(&dir, "MiniMaxAI/MiniMax-M3-Free");
        let jsonl = concat!(
            r#"{"role":"user","sessionId":"sess-1","content":[{"type":"text","text":"12345678"}]}"#,
            "\n",
            r#"{"role":"assistant","sessionId":"sess-1","content":[{"type":"text","text":"abcd"}]}"#,
        );
        let path = write_session(&dir, "proj", "sess-1", &jsonl);

        let messages = parse_commandcode_file(&path);
        assert_eq!(messages.len(), 1);
        let msg = &messages[0];
        assert_eq!(msg.client, "commandcode");
        assert_eq!(msg.model_id, "MiniMax-M3");
        assert_eq!(msg.provider_id, "minimax");
        assert_eq!(msg.session_id, "sess-1");
        assert_eq!(
            msg.tokens.input,
            estimate_tokens(content_chars(
                &json!([{"type": "text", "text": "12345678"}])
            ))
        );
        assert_eq!(
            msg.tokens.output,
            estimate_tokens(content_chars(&json!([{"type": "text", "text": "abcd"}])))
        );
        assert!(!msg.has_authoritative_cost());
    }

    /// An empty `usage` block on an assistant line falls back to estimation.
    #[test]
    fn test_empty_usage_block_falls_back_to_estimation() {
        let dir = TempDir::new().unwrap();
        write_config(&dir, "model-x");
        let jsonl = format!(
            "{}\n{}\n{}",
            json!({"type": "session", "version": 3, "id": "s"}),
            user_line("u1", "2026-08-31T00:00:00Z", "12345678"),
            json!({
                "type": "message",
                "id": "a1",
                "timestamp": "2026-08-31T00:00:05Z",
                "message": {"role": "assistant", "content": [{"type": "text", "text": "abcd"}]},
                "usage": {},
                "model": "model-x"
            }),
        );
        let path = write_session(&dir, "proj", "s", &jsonl);

        let messages = parse_commandcode_file(&path);
        assert_eq!(messages.len(), 1);
        let msg = &messages[0];
        assert_eq!(
            msg.tokens.input,
            estimate_tokens(content_chars(
                &json!([{"type": "text", "text": "12345678"}])
            ))
        );
        assert_eq!(
            msg.tokens.output,
            estimate_tokens(content_chars(&json!([{"type": "text", "text": "abcd"}])))
        );
        assert!(!msg.has_authoritative_cost());
    }

    /// Adversarial near-i64::MAX token fields and an absurd cost are clamped to
    /// sane ceilings instead of overflowing or poisoning aggregates.
    #[test]
    fn test_adversarial_usage_values_are_clamped() {
        let dir = TempDir::new().unwrap();
        write_config(&dir, "model-x");
        let jsonl = format!(
            "{}\n{}\n{}",
            json!({"type": "session", "version": 3, "id": "s"}),
            user_line("u1", "2026-08-31T00:00:00Z", "hi"),
            json!({
                "type": "message",
                "id": "a1",
                "timestamp": "2026-08-31T00:00:05Z",
                "message": {"role": "assistant", "content": [{"type": "text", "text": "yo"}]},
                "usage": {
                    "inputTokens": 9223372036854775807i64,
                    "outputTokens": 9223372036854775807i64,
                    "cacheReadTokens": 9223372036854775807i64,
                    "cacheWriteTokens": 9223372036854775807i64,
                    "costUsd": 1.0e300
                },
                "model": "model-x"
            }),
        );
        let path = write_session(&dir, "proj", "s", &jsonl);

        let messages = parse_commandcode_file(&path);
        assert_eq!(messages.len(), 1);
        let msg = &messages[0];
        // Clamped to the ceiling; no overflow, no i64::MAX poisoning.
        assert_eq!(msg.tokens.input, 0); // 1e12 - (1e12 + 1e12) clamped to 0
        assert_eq!(msg.tokens.output, 1_000_000_000_000);
        assert_eq!(msg.tokens.cache_read, 1_000_000_000_000);
        assert_eq!(msg.tokens.total(), 3_000_000_000_000);
        // Absurd cost rejected: not provider-reported, so pricing applies.
        assert_eq!(msg.cost, 0.0);
        assert!(!msg.has_authoritative_cost());
    }
}
