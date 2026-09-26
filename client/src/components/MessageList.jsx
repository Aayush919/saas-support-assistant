import { useState } from "react";

/**
 * Renders conversation history, sources, not-found badge, and optional meta.
 */
export default function MessageList({ messages, loading, bottomRef }) {
  if (!messages.length && !loading) {
    return (
      <div className="empty">
        <p>No messages yet.</p>
        <p className="muted">Pick a sample on the left or type a question below.</p>
      </div>
    );
  }

  return (
    <div className="transcript" aria-live="polite">
      {messages.map((m, i) => (
        <Bubble key={`${m.role}-${i}`} message={m} />
      ))}
      {loading && (
        <div className="bubble assistant pending">
          <div className="pulse" />
          <span>Thinking…</span>
        </div>
      )}
      <div ref={bottomRef} />
    </div>
  );
}

function Bubble({ message }) {
  // Assistant bubbles carry the brief fields (answer, status, sources, confidence)
  // plus demo telemetry: latency, tokens, USD estimate, retrieval score, cache.
  const [copied, setCopied] = useState(false);
  const isUser = message.role === "user";
  const notFound = message.status === "not_found";

  async function copy() {
    try {
      await navigator.clipboard.writeText(message.content);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard may be denied — ignore quietly */
    }
  }

  return (
    <div className={`bubble ${isUser ? "user" : "assistant"} ${notFound ? "not-found" : ""}`}>
      <div className="bubble-role">{isUser ? "You" : "Assistant"}</div>
      <div className="bubble-body">{message.content}</div>

      {/* Brief requires a visible "not found" signal, separate from the answer text. */}
      {!isUser && notFound && <div className="badge">Not found in documentation</div>}

      {!isUser && message.sources?.length > 0 && (
        <div className="sources">
          <span>Sources</span>
          {message.sources.map((s) => (
            <code key={s}>{s}</code>
          ))}
        </div>
      )}

      {!isUser && (
        <div className="bubble-actions">
          <button type="button" className="linkish" onClick={copy}>
            {copied ? "Copied" : "Copy"}
          </button>
          {typeof message.confidence === "number" && (
            <span className="meta-chip">confidence {message.confidence.toFixed(2)}</span>
          )}
          {message.meta?.latency_ms != null && (
            <span className="meta-chip">{message.meta.latency_ms} ms</span>
          )}
          {/* 0 in / 0 out means this request did not call the model (gate or cache). */}
          {message.meta && (
            <span className="meta-chip">
              {message.meta.input_tokens ?? 0} in / {message.meta.output_tokens ?? 0} out
            </span>
          )}
          {message.meta?.estimated_cost_usd != null && (
            <span className="meta-chip">
              ${Number(message.meta.estimated_cost_usd).toFixed(6)}
            </span>
          )}
          {message.meta?.retrieval_score != null && (
            <span className="meta-chip">
              retrieval {Number(message.meta.retrieval_score).toFixed(2)}
            </span>
          )}
          {message.meta?.cached && <span className="meta-chip">cached</span>}
        </div>
      )}
    </div>
  );
}
