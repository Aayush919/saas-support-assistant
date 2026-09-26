/**
 * Thin HTTP client for the FastAPI backend.
 * All product knowledge / prompts stay on the server — this only shuttles JSON.
 */

const API_BASE = (import.meta.env.VITE_API_BASE_URL || "http://localhost:8000").replace(
  /\/$/,
  ""
);

/** Sidebar uses this to show Connected · sarvam, or a clear "start the server" hint. */
export async function fetchHealth(signal) {
  const res = await fetch(`${API_BASE}/api/health`, { signal });
  if (!res.ok) throw new Error(`Health check failed (${res.status})`);
  return res.json();
}

/**
 * @param {string} message
 * @param {{role: string, content: string}[]} conversation
 * @param {AbortSignal} [signal]
 */
export async function sendChat(message, conversation, signal) {
  const res = await fetch(`${API_BASE}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, conversation }),
    signal,
  });

  let data = null;
  try {
    data = await res.json();
  } catch {
    throw new Error("Server returned a non-JSON response.");
  }

  if (!res.ok) {
    // Backend always uses { error: "..." } — never show raw stacks.
    throw new Error(data?.error || `Request failed (${res.status})`);
  }
  return data;
}

export { API_BASE };
