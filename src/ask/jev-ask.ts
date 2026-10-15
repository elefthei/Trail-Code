/**
 * `ask --jev`: seed, walk, prune.
 *
 * Plain `ask` scores every symbol in isolation against the query's words and
 * returns the top N. Measured on honojs/hono, the file holding the answer
 * (`src/compose.ts`) sat at rank 66 and three more never appeared in 300
 * results at all — word overlap simply cannot see that a symbol is *on a code
 * path*, and no amount of extra depth fixes that.
 *
 * So this path splits the work by what each part is actually good at:
 *
 *   1. lexical retrieval, widened — recall only, no pretence of precision
 *   2. jev scores every candidate — the one judgment words cannot make
 *   3. the graph walks from the survivors — exact, free, and the step that
 *      reaches code the query never named
 *   4. jev prunes the fan-out, names the entry point, and says whether the
 *      pack is actually enough
 *
 * Step 4's completeness answer is the point of the whole design. Plain `ask`'s
 * real failure on that hono query was not being wrong; it was being wrong
 * quietly. A calibrated number lets the pack admit its own gap and hand over
 * the one command that closes it.
 */

import { resolve } from "node:path";
import { contextDirFor } from "../context/node-file.js";
import { loadGraphCached } from "../graph/load.js";
import type { GraphV1, NodeV1 } from "../graph/types.js";
import { edgeWalk } from "../graph/traverse.js";
import { ask, type AskHit, type AskResult } from "./ask.js";
import { decide, noulOf, type JevAnswer, type JevQuestion } from "./jev-client.js";

export interface JevAskOptions {
  contextDir?: string;
  /** Final pack size. */
  limit?: number;
  /** How many lexical candidates to hand jev. Measured: 50 misses `compose`
   * (rank 66) and `url.ts` (rank 65); 100 catches both for one more page of
   * cards. */
  netWidth?: number;
  /** Hops to walk out from each seed. */
  depth?: number;
  /** A candidate becomes a seed at or above this probability. */
  seedThreshold?: number;
  /** A walked node stays in the pack at or above this probability. */
  keepThreshold?: number;
  /** Below this completeness the pack reports its own gap. */
  completeThreshold?: number;
  /** Most seeds to walk from, highest-scoring first. Bounds the fan-out. */
  maxSeeds?: number;
  /** Most walked nodes to put in front of jev's second call. Bounds tokens. */
  maxWalked?: number;
  source?: boolean;
  full?: boolean;
  in?: string;
  apiKey: string;
  model?: string;
  baseUrl?: string;
  fetchImpl?: typeof fetch;
  timeoutMs?: number;
}

export interface JevSeed {
  id: string;
  title: string;
  pointer: string;
  noul: number;
}

export interface JevTrace {
  /** Candidates handed to the first call. */
  net: number;
  seeds: JevSeed[];
  /** Distinct nodes the walk reached, before pruning. */
  walked: number;
  /** Nodes kept after the prune. */
  kept: number;
  entry?: string;
  /** Probability the pack answers the question, or null if jev didn't say. */
  complete: number | null;
  /** A node the walk arrived at and could not leave, when the pack looks thin. */
  stalledAt?: string;
  /** The single command that would close the gap. */
  next?: string;
  /** Decision requests actually made (0 when the lexical net was empty). */
  calls: number;
  /** Set when the pack is the plain lexical list rather than a walked subgraph,
   * with the reason. Never left unsaid: a fallback that looks like a result is
   * exactly the quiet wrongness this path exists to remove. */
  fellBack?: string;
}

/** How a node was reached, and in how many hops — the hop count is what lets
 * the prune call see the closest of a large fan-out rather than an arbitrary
 * slice of it. */
interface Reach {
  via: string;
  hops: number;
}

export interface JevAskResult extends AskResult {
  jev: JevTrace;
}

/** One line per candidate — enough for jev to judge role, cheap enough to send
 * a hundred of. */
function card(index: number, hit: AskHit): string {
  const gist = hit.snippet.split("\n").find((l) => l.trim().length > 0)?.trim() ?? "";
  return `[${index}] ${hit.title} · ${hit.pointer}\n     ${gist.slice(0, 160)}`;
}

function nodeCard(index: number, node: NodeV1, via: string): string {
  const sig = node.signature ?? node.summary ?? "";
  return `[${index}] ${node.name} · ${node.kind} · ${node.path}:${node.span} (reached by ${via})\n     ${sig.slice(0, 160)}`;
}

/** The graph node a lexical hit points at. File hits carry no span, symbol hits do. */
function nodeForPointer(graph: GraphV1, pointer: string): NodeV1 | null {
  const m = pointer.match(/^(.*):L(\d+)-L(\d+)$/);
  if (!m) return graph.nodes.find((n) => n.kind === "file" && n.path === pointer) ?? null;
  const [, path, from, to] = m;
  const span = `L${from}-L${to}`;
  return graph.nodes.find((n) => n.path === path && n.span === span) ?? null;
}

/**
 * Everything reachable from one seed: its own file, the siblings defined beside
 * it, and the edges out of AND into both.
 *
 * The inbound direction is not symmetry for its own sake — it is where the
 * entry point lives. Measured on hono: jev correctly refuses `#addRoute` as a
 * seed (it registers routes, it does not dispatch them), so no seed lands in
 * `hono-base.ts` and an outbound-only walk never reaches `#dispatch` at all.
 * One inbound hop from `compose` lands on it exactly, because the caller of the
 * thing you found is usually the thing that starts the flow.
 */
function expand(graph: GraphV1, seed: NodeV1, depth: number): Map<string, Reach> {
  const reached = new Map<string, Reach>();
  const note = (id: string, via: string, hops: number) => {
    if (id === seed.id) return;
    const prev = reached.get(id);
    if (!prev || prev.hops > hops) reached.set(id, { via, hops });
  };

  const fileNode = graph.nodes.find((n) => n.kind === "file" && n.path === seed.path) ?? null;
  const roots: NodeV1[] = [seed];
  if (fileNode && fileNode.id !== seed.id) {
    roots.push(fileNode);
    note(fileNode.id, "its file", 0);
  }
  for (const sib of graph.nodes) {
    if (sib.kind === "file" || sib.path !== seed.path || sib.id === seed.id) continue;
    note(sib.id, `defined beside ${seed.name}`, 1);
    roots.push(sib);
  }

  for (const root of roots) {
    for (const hit of edgeWalk(graph, root, "out", depth)) {
      if (!hit.node) continue;
      note(hit.node.id, `${root.name} ${hit.relation}`, hit.depth);
    }
    // Inbound is deliberately shallower: callers fan out far faster than
    // callees, and one hop is all the entry point ever needs.
    for (const hit of edgeWalk(graph, root, "in", 1)) {
      if (!hit.node) continue;
      note(hit.node.id, `${hit.relation} ${root.name}`, hit.depth);
    }
  }
  return reached;
}

/**
 * A node the walk entered and could not leave, *and* that sits next to something
 * the pack kept — the shape a re-export barrel used to have.
 *
 * The adjacency condition is what makes this worth printing. Any large repo has
 * dead-end file nodes lying around; naming one that has nothing to do with the
 * answer is noise dressed up as diagnosis, which is worse than saying nothing.
 */
function deadEnd(graph: GraphV1, reached: Iterable<string>, kept: Set<string>): NodeV1 | null {
  const byId = new Map(graph.nodes.map((n) => [n.id, n]));
  const hasOut = new Set(graph.edges.map((e) => e.source));
  const nextToKept = new Set<string>();
  for (const e of graph.edges) {
    if (kept.has(e.source)) nextToKept.add(e.target);
    if (kept.has(e.target)) nextToKept.add(e.source);
  }
  for (const id of reached) {
    if (!nextToKept.has(id) || hasOut.has(id)) continue;
    const node = byId.get(id);
    if (node && node.kind === "file") return node;
  }
  return null;
}

/** Order the pack the way the code runs: breadth-first from the entry point
 * along outgoing edges, then anything the walk didn't connect. */
function callOrder(graph: GraphV1, ids: Set<string>, entry: string | undefined): string[] {
  if (!entry || !ids.has(entry)) return [...ids];
  const out = new Map<string, string[]>();
  for (const e of graph.edges) {
    if (!ids.has(e.source) || !ids.has(e.target)) continue;
    const list = out.get(e.source);
    if (list) list.push(e.target);
    else out.set(e.source, [e.target]);
  }
  const ordered: string[] = [];
  const seen = new Set<string>();
  const queue = [entry];
  while (queue.length) {
    const id = queue.shift()!;
    if (seen.has(id)) continue;
    seen.add(id);
    ordered.push(id);
    for (const next of out.get(id) ?? []) if (!seen.has(next)) queue.push(next);
  }
  for (const id of ids) if (!seen.has(id)) ordered.push(id);
  return ordered;
}

function hitFor(node: NodeV1, score: number, via: string): AskHit {
  return {
    kind: "symbol",
    title: `${node.name} · ${node.kind}`,
    pointer: node.kind === "file" ? node.path : `${node.path}:${node.span}`,
    snippet: node.summary ?? node.signature ?? via,
    score,
  };
}

export async function askJev(
  dir: string,
  query: string,
  opts: JevAskOptions,
): Promise<JevAskResult> {
  const netWidth = opts.netWidth ?? 100;
  const limit = opts.limit ?? 8;
  const depth = opts.depth ?? 2;
  const seedThreshold = opts.seedThreshold ?? 0.7;
  const keepThreshold = opts.keepThreshold ?? 0.5;
  const completeThreshold = opts.completeThreshold ?? 0.7;
  const maxSeeds = opts.maxSeeds ?? 6;
  const maxWalked = opts.maxWalked ?? 40;
  const client = {
    apiKey: opts.apiKey,
    model: opts.model,
    baseUrl: opts.baseUrl,
    fetchImpl: opts.fetchImpl,
    timeoutMs: opts.timeoutMs ?? 20_000,
  };

  const net = ask(dir, query, {
    limit: netWidth,
    contextDir: opts.contextDir,
    in: opts.in,
  });
  const fallback = (trace: Partial<JevTrace>): JevAskResult => ({
    ...net,
    hits: net.hits.slice(0, limit),
    jev: { net: net.hits.length, seeds: [], walked: 0, kept: 0, complete: null, calls: 0, ...trace },
  });
  if (net.hits.length === 0) return fallback({ fellBack: "the lexical net came back empty" });

  const graph = loadGraphCached(contextDirFor(dir, opts.contextDir));
  if (!graph) {
    return fallback({ fellBack: "no wiring graph on disk — run `graft build`" });
  }

  // ---- call 1: which candidates sit on the path the question asks about ----
  const questions: Record<string, JevQuestion> = {};
  const cards: string[] = [];
  net.hits.forEach((hit, i) => {
    cards.push(card(i, hit));
    questions[`c${i}`] = {
      type: "noul",
      instructions: `Is candidate ${i} part of the code path this question asks about?`,
      criteria: {
        true: "it implements, or is directly on, the path the question describes",
        false: "unrelated to that path, or it merely shares a word with the question",
      },
    };
  });
  const seedAnswers = await decide(
    { state: `QUESTION: ${query}\n\nCANDIDATES:\n${cards.join("\n")}`, questions },
    client,
  );

  const seeds: Array<JevSeed & { node: NodeV1 }> = [];
  net.hits.forEach((hit, i) => {
    const p = noulOf(seedAnswers[`c${i}`]);
    if (p === null || p < seedThreshold) return;
    const node = nodeForPointer(graph, hit.pointer);
    if (node) seeds.push({ id: node.id, title: hit.title, pointer: hit.pointer, noul: p, node });
  });
  seeds.sort((a, b) => b.noul - a.noul);
  const chosen = seeds.slice(0, maxSeeds);
  if (chosen.length === 0) {
    // Everything retrieved was judged off-path. Returning the lexical list is
    // still more useful than nothing, but it is NOT a walked answer and the
    // pack says so rather than passing it off as one.
    return fallback({
      calls: 1,
      fellBack: "no candidate cleared the seed bar — the list below is the plain lexical pack, which jev judged off-path",
    });
  }

  // ---- the walk: one traversal, all seeds unioned ----
  const reached = new Map<string, Reach>();
  for (const seed of chosen) {
    for (const [id, r] of expand(graph, seed.node, depth)) {
      const prev = reached.get(id);
      if (!prev || prev.hops > r.hops) reached.set(id, r);
    }
  }
  for (const seed of chosen) reached.delete(seed.id);

  const byId = new Map(graph.nodes.map((n) => [n.id, n]));
  // Nearest first. A wide fan-out routinely exceeds `maxWalked`, and truncating
  // in map order would decide the pack by extraction order — on hono that threw
  // away two thirds of the walk before jev ever saw it.
  const walkedIds = [...reached.entries()]
    .filter(([id]) => byId.get(id)?.kind !== "file")
    .sort((a, b) => a[1].hops - b[1].hops)
    .slice(0, maxWalked)
    .map(([id]) => id);

  // ---- call 2: prune, order, and say whether this is enough ----
  const pruneQuestions: Record<string, JevQuestion> = {};
  const walkedCards: string[] = [];
  walkedIds.forEach((id, i) => {
    const node = byId.get(id)!;
    walkedCards.push(nodeCard(i, node, reached.get(id)!.via));
    pruneQuestions[`w${i}`] = {
      type: "noul",
      instructions: `Does symbol ${i} belong in the answer to this question?`,
      criteria: {
        true: "a reader needs this span to understand the answer",
        false: "reached only incidentally — types, constants, unrelated helpers",
      },
    };
  });
  const entryCriteria: Record<string, string> = {};
  for (const seed of chosen) entryCriteria[seed.id] = `${seed.title} — ${seed.pointer}`;
  walkedIds.slice(0, 12).forEach((id) => {
    const node = byId.get(id)!;
    entryCriteria[id] = `${node.name} — ${node.path}:${node.span}`;
  });
  if (Object.keys(entryCriteria).length >= 2) {
    pruneQuestions.entry = {
      type: "choice",
      instructions: "Which of these is the entry point that the flow starts at?",
      criteria: entryCriteria,
    };
  }
  pruneQuestions.complete = {
    type: "noul",
    instructions: "Is this set of code enough to answer the question in full?",
    criteria: {
      true: "every step the question names has code present here",
      false: "a step the question names has no code in this set",
    },
  };

  const seedList = chosen
    .map((s) => `[seed] ${s.title} · ${s.pointer}`)
    .join("\n");
  const pruneAnswers: Record<string, JevAnswer> = await decide(
    {
      state:
        `QUESTION: ${query}\n\nALREADY IN THE PACK:\n${seedList}\n\n` +
        `REACHED BY WALKING THE CALL GRAPH:\n${walkedCards.join("\n")}`,
      questions: pruneQuestions,
    },
    client,
  );

  const kept = new Set<string>(chosen.map((s) => s.id));
  const scoreOf = new Map<string, number>(chosen.map((s) => [s.id, s.noul]));
  walkedIds.forEach((id, i) => {
    const p = noulOf(pruneAnswers[`w${i}`]);
    if (p !== null && p >= keepThreshold) {
      kept.add(id);
      scoreOf.set(id, p);
    }
  });

  const entry = pruneAnswers.entry?.choice;
  const ordered = callOrder(graph, kept, entry).slice(0, limit);
  const hits = ordered
    .map((id) => {
      const node = byId.get(id);
      return node ? hitFor(node, scoreOf.get(id) ?? 0, reached.get(id)?.via ?? "seed") : null;
    })
    .filter((h): h is AskHit => h !== null);

  const complete = noulOf(pruneAnswers.complete);
  const trace: JevTrace = {
    net: net.hits.length,
    seeds: chosen.map(({ id, title, pointer, noul }) => ({ id, title, pointer, noul })),
    walked: reached.size,
    kept: kept.size,
    entry,
    complete,
    calls: 2,
  };
  if (complete !== null && complete < completeThreshold) {
    const stalled = deadEnd(graph, reached.keys(), kept);
    if (stalled) {
      trace.stalledAt = stalled.path;
      // Point at the directory the walk could not enter, and let the agent pick
      // its own literal. Guessing a search term from the question's wording is
      // how the lexical ranker got this wrong in the first place.
      trace.next = `graft skeleton ${stalled.path}`;
    }
  }

  const result: JevAskResult = { ...net, hits, jev: trace };
  if (opts.source) {
    const { inlineSource } = await import("./ask.js");
    inlineSource(resolve(dir), result.hits, graph, opts.full ?? false);
  }
  return result;
}

/**
 * Render the pack. Deliberately shaped as a trace, not a ranked list: the whole
 * reason to walk the graph is that the answer is a path, and a numbered list
 * throws away the one thing the walk recovered. The trace line at the bottom is
 * not decoration — when `complete` is low, that is the pack telling the agent
 * it is short, and naming the command that fixes it.
 */
export function formatJevAsk(r: JevAskResult): string {
  const out: string[] = [];
  out.push(`graft ask --jev — "${r.query}"\n`);
  if (r.jev.fellBack) {
    out.push(`\n⚠ not a walked pack: ${r.jev.fellBack}\n`);
  }
  if (r.hits.length === 0) {
    out.push("\nno matching nodes — try different words, or `graft build` if graft/ is empty\n");
    return out.join("");
  }

  out.push("");
  r.hits.forEach((hit, i) => {
    const arrow = i === 0 ? "" : "└─ ";
    out.push(`${arrow}${hit.title}  ${hit.pointer}  (${hit.score.toFixed(2)})`);
    if (hit.snippet && hit.snippet !== hit.title) out.push(`   ${hit.snippet.split("\n")[0].slice(0, 120)}`);
    if (hit.code) out.push("\n```\n" + hit.code + "\n```\n");
  });

  const t = r.jev;
  out.push("");
  out.push(
    `${t.net} candidates → ${t.seeds.length} seeds → ${t.walked} walked → ${t.kept} kept · ${t.calls} decision call(s)`,
  );
  if (t.complete !== null) {
    out.push(`complete: ${t.complete.toFixed(2)}`);
    if (t.stalledAt) out.push(`walk stalled at ${t.stalledAt} — no outgoing edges indexed`);
    if (t.next) out.push(`next: ${t.next}`);
  }
  return out.join("\n") + "\n";
}
