/**
 * `ask --loop`: the grep-and-read loop, with a decision model in the driver's
 * seat instead of a coding agent.
 *
 * A coding agent answers "how does X work" well, and expensively, by iterating:
 * look at something, notice a name it doesn't recognise, look that up, stop
 * when satisfied. Measured on honojs/hono that took 20 round trips, 80 seconds
 * and $1.09, and every one of those turns carried whole files through a large
 * model's context.
 *
 * The loop itself is cheap, though. Each turn is one decision — what do I not
 * yet understand, and what should I fetch next — and a typed decision model
 * answers exactly that shape of question for a fraction of a cent. So this runs
 * the same loop with jev deciding and graft's own primitives fetching, and
 * hands the agent one finished pack from a single tool call.
 *
 * Two findings shaped the menu it picks from:
 *
 *   - jev chooses well from a wide menu. Given 100 shuffled file paths and
 *     nothing else, it picked hono's real entry file 3 times out of 3; given
 *     the text of `#dispatch` plus 86 options, it picked `grep:compose` 3 out
 *     of 3. Narrowing the menu to 40 changed nothing, so width is not the
 *     constraint.
 *   - but that first round leaned on `hono-base.ts` being an honest filename.
 *     A repo of `util3.ts` would take that signal away, and symbol names
 *     survive bad file naming — `util3.ts` still defines `matchRoute()`. So the
 *     opening menu offers symbols first and paths second.
 */

import { resolve } from "node:path";
import { contextDirFor } from "../context/node-file.js";
import { loadGraphCached } from "../graph/load.js";
import type { GraphV1, NodeV1 } from "../graph/types.js";
import { grepGraph } from "../search/grep.js";
import { ask, sliceSpan, type AskHit, type AskResult } from "./ask.js";
import { decide, noulOf, type ChoiceQuestion, type JevQuestion } from "./jev-client.js";

export interface JevLoopOptions {
  contextDir?: string;
  limit?: number;
  /** Hard cap on rounds. Each round is one decision call. */
  maxRounds?: number;
  /** Stop outright once the decision model is at least this sure the pack
   * answers the question. */
  doneThreshold?: number;
  /** Rounds of no improvement tolerated before stopping. An absolute threshold
   * alone never fires — jev sits around 0.3 even on good coverage — so this
   * relative rule is what actually ends the loop.
   *
   * It is deliberately paired with {@link minRounds}, because `done` is noisy:
   * a measured run read `#dispatch`, the single most important span, on the
   * round AFTER the running best, and a bare patience rule stopped there and
   * discarded it. Wandering costs a few cheap calls; stopping early costs the
   * answer. */
  patience?: number;
  /** Rounds that always run before patience may fire. */
  minRounds?: number;
  /** Symbols offered in the opening menu (ranked lexically, for recall only). */
  openSymbols?: number;
  /** File paths offered under them. */
  openFiles?: number;
  /** Options carried into each round's menu. A wide menu measured fine; this
   * exists to bound tokens, not accuracy. */
  menuWidth?: number;
  /** Characters of source kept per fetch in the running state. */
  digestChars?: number;
  source?: boolean;
  in?: string;
  apiKey: string;
  model?: string;
  baseUrl?: string;
  fetchImpl?: typeof fetch;
  timeoutMs?: number;
}

export interface JevLoopStep {
  round: number;
  /** The label chosen, e.g. `grep:compose` or `read:src/compose.ts#compose`. */
  action: string;
  /** Its probability, and how much of the mass went elsewhere. */
  confidence: number;
  /** How sure the model was, before acting, that it could already answer. */
  done: number | null;
  /** What the fetch actually returned, in one line. */
  yielded: string;
  ms: number;
}

export interface JevLoopResult extends AskResult {
  loop: {
    rounds: number;
    steps: JevLoopStep[];
    /** Final confidence that the pack answers the question. */
    done: number | null;
    /** True when the loop hit its round cap rather than deciding it was done —
     * the pack is then a best effort, and says so. */
    ranOut: boolean;
    /** The highest completeness seen; the pack is the one from that round. */
    best: number | null;
    calls: number;
    ms: number;
  };
}

/** A fetched span, held for both the running state and the final pack. */
interface Fetched {
  node: NodeV1 | null;
  path: string;
  span?: string;
  label: string;
  text: string;
}

const IDENT = /\b[A-Za-z_#$][A-Za-z0-9_$]{3,}\b/g;
/** Words too common in any source file to be worth a round. */
const NOISE = new Set([
  "const", "return", "function", "export", "import", "this", "true", "false", "null",
  "undefined", "string", "number", "boolean", "void", "async", "await", "class",
  "extends", "implements", "interface", "type", "from", "else", "case", "break",
  "continue", "throw", "catch", "typeof", "instanceof", "public", "private",
  "static", "readonly", "unknown", "never", "Promise", "Record", "Array", "Object",
]);

/**
 * Supporting code answers "who exercises this", never "how does this work".
 * Tests, benchmarks, examples and fixtures all use the real API heavily, so
 * they win an in-degree ranking and then waste a round — measured: `grep:router`
 * landed on `benchmarks/fetch/bench.mts` twice in one run.
 */
function isSupporting(path: string): boolean {
  return (
    /(^|\/)(tests?|__tests__|benchmarks?|examples?|fixtures?|perf-measures|runtime-tests)\//.test(path) ||
    /\.(test|spec|bench)\.[cm]?[jt]sx?$/.test(path)
  );
}

function nodeLabel(n: NodeV1): string {
  return n.kind === "file" ? `skeleton:${n.path}` : `read:${n.id}`;
}

function nodeDesc(n: NodeV1): string {
  if (n.kind === "file") return `the API of ${n.path}`;
  return `${n.name} — ${n.signature ?? n.kind} — ${n.path}:${n.span}`;
}

/**
 * The opening menu: symbols first, paths second.
 *
 * The lexical rank is used purely to bound the list; it is not trusted to put
 * the answer near the top, because on the hono query it put the real answer at
 * rank 66. Being in the menu at all is the whole job.
 */
function openingMenu(
  hits: AskHit[],
  graph: GraphV1,
  nSymbols: number,
  nFiles: number,
): Map<string, string> {
  const menu = new Map<string, string>();
  const byPointer = (p: string) => {
    const m = p.match(/^(.*):L(\d+)-L(\d+)$/);
    if (!m) return graph.nodes.find((n) => n.kind === "file" && n.path === p) ?? null;
    return graph.nodes.find((n) => n.path === m[1] && n.span === `L${m[2]}-L${m[3]}`) ?? null;
  };
  let symbols = 0;
  let files = 0;
  for (const hit of hits) {
    const node = byPointer(hit.pointer);
    if (!node || isSupporting(node.path)) continue;
    if (node.kind !== "file" && symbols < nSymbols) {
      menu.set(nodeLabel(node), nodeDesc(node));
      symbols++;
    } else if (node.kind === "file" && files < nFiles) {
      menu.set(nodeLabel(node), nodeDesc(node));
      files++;
    }
  }
  return menu;
}

/** Everything a fetch just revealed that the loop could chase next. */
function frontierFrom(text: string, node: NodeV1 | null, graph: GraphV1): Map<string, string> {
  const next = new Map<string, string>();
  for (const raw of text.match(IDENT) ?? []) {
    // `L408-L468` is a span label from a skeleton listing, not a name in the
    // code — grepping it costs a whole round and returns nothing.
    if (NOISE.has(raw) || /^L\d+$/.test(raw)) continue;
    next.set(`grep:${raw}`, `every use of ${raw}`);
  }
  if (node) {
    const byId = new Map(graph.nodes.map((n) => [n.id, n]));
    for (const e of graph.edges) {
      // Downstream: what this thing uses.
      if (e.source === node.id || e.source === node.path) {
        const target = byId.get(e.target);
        if (target && !isSupporting(target.path)) next.set(nodeLabel(target), `${e.relation} → ${nodeDesc(target)}`);
      }
      // Upstream: what uses this thing. Without it the loop can only ever walk
      // deeper into the code it already found, and the entry point of a flow is
      // by definition upstream of everything — on hono that is `#dispatch`,
      // which calls `compose` and is otherwise unreachable from it.
      if (e.target === node.id || e.target === node.path) {
        const source = byId.get(e.source);
        if (source && !isSupporting(source.path)) next.set(nodeLabel(source), `${nodeDesc(source)} — ${e.relation} this`);
      }
    }
  }
  return next;
}

/** Run one chosen label, and report what it yielded. */
function perform(
  label: string,
  root: string,
  graph: GraphV1,
  digestChars: number,
): Fetched | null {
  if (label.startsWith("read:")) {
    const node = graph.nodes.find((n) => n.id === label.slice(5));
    if (!node) return null;
    const m = node.span.match(/^L(\d+)-L(\d+)$/);
    const text = m ? (sliceSpan(root, node.path, Number(m[1]), Number(m[2])) ?? "") : "";
    return { node, path: node.path, span: node.span, label, text: text.slice(0, digestChars) };
  }
  if (label.startsWith("skeleton:")) {
    const path = label.slice(9);
    const node = graph.nodes.find((n) => n.kind === "file" && n.path === path) ?? null;
    const lines = graph.nodes
      .filter((n) => n.kind !== "file" && n.path === path)
      .map((n) => `${n.span}  ${n.kind} ${n.name}  ${n.signature ?? ""}`)
      .join("\n");
    return { node, path, label, text: lines.slice(0, digestChars) };
  }
  if (label.startsWith("grep:")) {
    const pattern = label.slice(5);
    const r = grepGraph(graph, root, pattern, { maxHits: 40 });
    const groups = r.groups.filter((g) => !isSupporting(g.path));
    const lines = groups
      .slice(0, 8)
      .map((g) => `${g.symbol?.name ?? g.path}  ${g.path}\n  ${g.hits.map((h) => `L${h.line}: ${h.text}`).join("\n  ")}`)
      .join("\n");
    // The most-referenced symbol carrying the pattern is the one worth keeping
    // as a pack candidate; the rest of the groups are only navigation.
    const lead = groups[0];
    const node = lead?.symbol
      ? (graph.nodes.find((n) => n.path === lead.path && n.span === lead.symbol!.span) ?? null)
      : null;
    return { node, path: lead?.path ?? pattern, span: node?.span, label, text: lines.slice(0, digestChars) };
  }
  return null;
}

function stateOf(query: string, got: Fetched[]): string {
  const seen = got
    .map((f) => `--- ${f.label}  (${f.path}${f.span ? `:${f.span}` : ""})\n${f.text}`)
    .join("\n\n");
  return `QUESTION: ${query}\n\nWHAT HAS BEEN LOOKED AT SO FAR:\n${seen || "(nothing yet)"}`;
}

export async function askJevLoop(
  dir: string,
  query: string,
  opts: JevLoopOptions,
): Promise<JevLoopResult> {
  const root = resolve(dir);
  const maxRounds = opts.maxRounds ?? 12;
  const doneThreshold = opts.doneThreshold ?? 0.85;
  const patience = opts.patience ?? 3;
  const minRounds = opts.minRounds ?? 7;
  const menuWidth = opts.menuWidth ?? 90;
  const digestChars = opts.digestChars ?? 1400;
  const limit = opts.limit ?? 8;
  const client = {
    apiKey: opts.apiKey,
    model: opts.model,
    baseUrl: opts.baseUrl,
    fetchImpl: opts.fetchImpl,
    timeoutMs: opts.timeoutMs ?? 20_000,
  };
  const startedAt = Date.now();

  const seedPack = ask(dir, query, { limit: 150, contextDir: opts.contextDir, in: opts.in });
  const graph = loadGraphCached(contextDirFor(dir, opts.contextDir));
  if (!graph) {
    return {
      ...seedPack,
      loop: { rounds: 0, steps: [], done: null, ranOut: false, best: null, calls: 0, ms: 0 },
    };
  }

  // Menu entries carry the round they were discovered in, because the cap must
  // fall on the STALE end of the list. Slicing an insertion-ordered map keeps
  // the opening menu forever and silently hides everything the loop learns —
  // which is the whole point of looping.
  const menu = new Map<string, { desc: string; found: number }>();
  for (const [label, desc] of openingMenu(seedPack.hits, graph, opts.openSymbols ?? 60, opts.openFiles ?? 30)) {
    menu.set(label, { desc, found: 0 });
  }
  const got: Fetched[] = [];
  const steps: JevLoopStep[] = [];
  const spent = new Set<string>();
  let done: number | null = null;
  let calls = 0;
  // The pack is rolled back to its best round, so a wandering tail costs a few
  // decision calls but never degrades the answer.
  let best = { done: -1, size: 0, round: 0 };

  for (let round = 1; round <= maxRounds; round++) {
    // Freshest first: what the last fetch revealed is the most likely next
    // step, and it is also what the cap would otherwise throw away.
    const options: Array<[string, string]> = [...menu]
      .filter(([label]) => !spent.has(label))
      .sort((a, b) => b[1].found - a[1].found)
      .slice(0, menuWidth)
      .map(([label, v]) => [label, v.desc]);
    if (options.length === 0) break;

    const questions: Record<string, JevQuestion> = {
      done: {
        type: "noul",
        instructions: "Is what has been looked at so far enough to answer the question in full?",
        criteria: {
          true: "every step the question names has its code present above",
          false: "a step the question names has no code above yet",
        },
      },
    };
    if (options.length >= 2) {
      questions.next = {
        type: "choice",
        instructions: "What is the single most useful next thing to look at?",
        criteria: Object.fromEntries(options),
      } satisfies ChoiceQuestion;
    }

    const t0 = Date.now();
    const answers = await decide({ state: stateOf(query, got), questions }, client);
    calls++;
    done = noulOf(answers.done);

    // Asked BEFORE acting, so the loop never spends a round confirming what it
    // already knew — and never stops on round one before fetching anything.
    if (done !== null && done > best.done) best = { done, size: got.length, round };
    if (done !== null && done >= doneThreshold && got.length > 0) {
      steps.push({ round, action: "(stop)", confidence: done, done, yielded: "enough", ms: Date.now() - t0 });
      break;
    }
    if (got.length > 0 && round > minRounds && round - best.round >= patience) {
      steps.push({
        round,
        action: "(stop)",
        confidence: best.done,
        done,
        yielded: `no gain in ${patience} rounds (best was round ${best.round})`,
        ms: Date.now() - t0,
      });
      break;
    }

    const choice = answers.next?.choice;
    if (!choice) break;
    spent.add(choice);
    const fetched = perform(choice, root, graph, digestChars);
    if (!fetched) {
      steps.push({ round, action: choice, confidence: 0, done, yielded: "nothing", ms: Date.now() - t0 });
      continue;
    }
    got.push(fetched);
    for (const [label, desc] of frontierFrom(fetched.text, fetched.node, graph)) {
      if (!spent.has(label) && !menu.has(label)) menu.set(label, { desc, found: round });
    }
    steps.push({
      round,
      action: choice,
      confidence: answers.next?.probabilities?.[choice] ?? 0,
      done,
      yielded: `${fetched.path}${fetched.span ? `:${fetched.span}` : ""} (${fetched.text.length} chars)`,
      ms: Date.now() - t0,
    });
  }

  // The pack is what the loop actually opened, newest judgement first but
  // deduped by span — the order it looked things up IS the reading order.
  // Everything fetched is kept. Rolling the pack back to the best-scoring
  // round sounded principled and measured badly: `done` moves by ±0.08 between
  // adjacent rounds, so the rollback routinely discarded a span the loop had
  // just deliberately gone and fetched. `best` is reported, not enforced.
  const hits: AskHit[] = [];
  const seen = new Set<string>();
  for (const f of got) {
    const pointer = f.span ? `${f.path}:${f.span}` : f.path;
    if (seen.has(pointer)) continue;
    seen.add(pointer);
    hits.push({
      kind: "symbol",
      title: f.node ? `${f.node.name} · ${f.node.kind}` : f.label,
      pointer,
      snippet: f.node?.signature ?? f.label,
      score: 1,
      ...(opts.source ? { code: f.text } : {}),
    });
  }

  return {
    ...seedPack,
    hits: hits.slice(0, limit),
    loop: {
      rounds: steps.length,
      steps,
      done,
      ranOut: steps.length >= maxRounds && (done === null || done < doneThreshold),
      best: best.done < 0 ? null : best.done,
      calls,
      ms: Date.now() - startedAt,
    },
  };
}

export function formatJevLoop(r: JevLoopResult): string {
  const out: string[] = [`graft ask --loop — "${r.query}"\n`];
  for (const s of r.loop.steps) {
    const d = s.done === null ? "  ? " : s.done.toFixed(2);
    out.push(`${String(s.round).padStart(2)}. done=${d}  ${s.action}`);
    out.push(`     → ${s.yielded}  (${s.ms} ms)`);
  }
  out.push("");
  for (const h of r.hits) {
    out.push(`${h.title}  ${h.pointer}`);
    if (h.code) out.push("```\n" + h.code + "\n```");
  }
  out.push("");
  out.push(
    `${r.loop.rounds} rounds · ${r.loop.calls} decision calls · ${r.loop.ms} ms · best done ${r.loop.best?.toFixed(2) ?? "?"}`,
  );
  if (r.loop.ranOut) out.push("⚠ hit the round cap before it was satisfied — this is a best effort, not a complete answer");
  return out.join("\n") + "\n";
}
