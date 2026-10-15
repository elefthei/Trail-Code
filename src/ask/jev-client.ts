/**
 * Client for OpenRouter's typed-decision endpoint (`/alpha/decisions`).
 *
 * The model answers narrow, typed questions about a blob of state and returns
 * calibrated numbers rather than prose: a `noul` is a probability in [0, 1], a
 * `choice` carries the winning label plus the full distribution. That shape is
 * what makes it usable inside a retriever — a probability can be thresholded,
 * sorted and reported, and a sentence cannot.
 */

/** A yes/no question answered as a probability. */
export interface NoulQuestion {
  type: "noul";
  instructions: string;
  criteria?: { true: string; false: string };
}

/** A pick-one question answered with a label plus its distribution. */
export interface ChoiceQuestion {
  type: "choice";
  instructions: string;
  criteria: Record<string, string>;
}

export type JevQuestion = NoulQuestion | ChoiceQuestion;

export interface JevAnswer {
  noul?: number;
  choice?: string;
  probabilities?: Record<string, number>;
  score?: number;
}

export interface JevRequest {
  state: string;
  questions: Record<string, JevQuestion>;
}

export interface JevClientOptions {
  apiKey: string;
  model?: string;
  baseUrl?: string;
  /** Injectable for tests; defaults to the global fetch. */
  fetchImpl?: typeof fetch;
  /** Abort a single call after this many ms. `ask` is on the agent's hot path,
   * so a hung decision must degrade to the plain lexical pack, not hang. */
  timeoutMs?: number;
}

export const DEFAULT_JEV_MODEL = "typesafe/jev-1.13";
const DEFAULT_DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions";

/** Raised when the endpoint answers but not in the documented shape, so callers
 * can tell "jev said no" apart from "jev never really answered". */
export class JevError extends Error {}

/**
 * Ask one batch of typed questions about one state. Returns the answers keyed
 * exactly as the questions were.
 */
export async function decide(
  req: JevRequest,
  opts: JevClientOptions,
): Promise<Record<string, JevAnswer>> {
  const fetchImpl = opts.fetchImpl ?? globalThis.fetch;
  if (!fetchImpl) throw new JevError("no fetch implementation available");
  const url = opts.baseUrl ?? DEFAULT_DECISIONS_URL;

  const controller = new AbortController();
  const timer =
    opts.timeoutMs && opts.timeoutMs > 0
      ? setTimeout(() => controller.abort(), opts.timeoutMs)
      : null;
  let res: Response;
  try {
    res = await fetchImpl(url, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${opts.apiKey}`,
        "Content-Type": "application/json",
        "X-OpenRouter-Title": "graft",
      },
      body: JSON.stringify({
        model: opts.model ?? DEFAULT_JEV_MODEL,
        state: req.state,
        questions: req.questions,
      }),
      signal: controller.signal,
    });
  } catch (err) {
    throw new JevError(
      controller.signal.aborted
        ? `decision timed out after ${opts.timeoutMs}ms`
        : `decision request failed: ${(err as Error).message}`,
    );
  } finally {
    if (timer) clearTimeout(timer);
  }

  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new JevError(`${res.status} ${res.statusText}${body ? ` — ${body.slice(0, 200)}` : ""}`);
  }

  const json = (await res.json().catch(() => null)) as { answers?: unknown } | null;
  const answers = json?.answers;
  if (!answers || typeof answers !== "object") {
    throw new JevError("response carried no `answers` object");
  }
  return answers as Record<string, JevAnswer>;
}

/** The probability a noul answer carries, or null when the answer is missing or
 * malformed. Callers treat null as "unjudged" rather than as a low score, so a
 * dropped answer never silently deletes a candidate. */
export function noulOf(answer: JevAnswer | undefined): number | null {
  const n = answer?.noul;
  return typeof n === "number" && Number.isFinite(n) ? n : null;
}
