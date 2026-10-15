import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { buildGraph } from "../src/graph/build.js";
import { askJev, formatJevAsk } from "../src/ask/jev-ask.js";
import { decide, noulOf, JevError } from "../src/ask/jev-client.js";

/**
 * A repo shaped like the failure this path exists for: the symbol a word-match
 * finds (`addRoute`) is NOT the answer, the answer (`dispatch` → `compose`) is
 * reachable only by walking, and a barrel sits between the entry and one leg of
 * the flow.
 */
function fixture(): string {
  const root = mkdtempSync(join(tmpdir(), "graft-jev-"));
  const write = (rel: string, src: string) => {
    const full = join(root, rel);
    mkdirSync(join(full, ".."), { recursive: true });
    writeFileSync(full, src);
  };
  write(
    "src/app.ts",
    `import { compose } from "./compose"\nimport { Router } from "./router"\n\nexport class App {\n  addRoute(method: string, path: string) { this.routes.push([method, path]) }\n  dispatch(request: Request) {\n    const matched = new Router().match(request)\n    return compose(matched)(request)\n  }\n}\n`,
  );
  write("src/compose.ts", `export function compose(handlers: unknown[]) {\n  return (req: unknown) => dispatchAt(handlers, 0, req)\n}\nexport function dispatchAt(hs: unknown[], i: number, req: unknown) { return hs[i] }\n`);
  write("src/router/index.ts", `export { Router } from "./router"\n`);
  write("src/router/router.ts", `import { Matcher } from "./matcher"\nexport class Router {\n  match(request: unknown) { return new Matcher().run(request) }\n}\n`);
  write("src/router/matcher.ts", `export class Matcher {\n  run(request: unknown) { return [] }\n}\n`);
  write("src/unrelated.ts", `export function runBenchmark() { return "route matched run middleware" }\n`);
  return root;
}

/** A fake endpoint that answers every noul with `yes` and names an entry point,
 * so the test exercises the seed/walk/prune wiring rather than the model. */
function stub(answers: (body: Record<string, unknown>) => Record<string, unknown>) {
  const calls: Array<Record<string, unknown>> = [];
  const fetchImpl = (async (_url: string, init: RequestInit) => {
    const body = JSON.parse(String(init.body)) as Record<string, unknown>;
    calls.push(body);
    return {
      ok: true,
      status: 200,
      statusText: "OK",
      json: async () => ({ answers: answers(body) }),
      text: async () => "",
    } as unknown as Response;
  }) as unknown as typeof fetch;
  return { fetchImpl, calls };
}

const yesToAll = (body: Record<string, unknown>) => {
  const qs = body.questions as Record<string, { type: string; criteria?: Record<string, string> }>;
  const out: Record<string, unknown> = {};
  for (const [key, q] of Object.entries(qs)) {
    if (q.type === "choice") {
      const first = Object.keys(q.criteria ?? {})[0];
      out[key] = { choice: first, probabilities: { [first]: 1 } };
    } else {
      out[key] = { noul: 0.95 };
    }
  }
  return out;
};

test("jev arm: the walk reaches code no lexical rank retrieved", async () => {
  const root = fixture();
  await buildGraph(root);
  const { fetchImpl, calls } = stub(yesToAll);

  const r = await askJev(root, "how does a request get matched to a route and run through middleware", {
    apiKey: "test", fetchImpl, netWidth: 50, limit: 12,
  });

  assert.equal(calls.length, 2, "exactly two decision calls: seed, then prune");
  assert.ok(r.jev.seeds.length > 0, "at least one seed cleared the bar");
  const paths = r.hits.map((h) => h.pointer.split(":")[0]);
  assert.ok(paths.includes("src/compose.ts"), `compose.ts reached by the walk (got ${paths.join(", ")})`);
  assert.ok(
    paths.includes("src/router/router.ts") || paths.includes("src/router/matcher.ts"),
    `past the barrel into the router (got ${paths.join(", ")})`,
  );
});

test("jev arm: the first call carries one question per candidate, the second names the entry point", async () => {
  const root = fixture();
  await buildGraph(root);
  const { fetchImpl, calls } = stub(yesToAll);
  await askJev(root, "how does a request reach a handler", { apiKey: "test", fetchImpl, netWidth: 20 });

  const seedQs = calls[0].questions as Record<string, { type: string }>;
  assert.ok(Object.keys(seedQs).length > 1, "a noul per candidate");
  assert.ok(Object.values(seedQs).every((q) => q.type === "noul"));

  const pruneQs = calls[1].questions as Record<string, { type: string }>;
  assert.equal(pruneQs.complete?.type, "noul", "the prune call always asks whether the pack is enough");
  assert.equal(pruneQs.entry?.type, "choice", "and which node the flow starts at");
});

test("jev arm: nothing clearing the seed bar degrades loudly, never silently", async () => {
  const root = fixture();
  await buildGraph(root);
  const noToAll = (body: Record<string, unknown>) => {
    const qs = body.questions as Record<string, unknown>;
    return Object.fromEntries(Object.keys(qs).map((k) => [k, { noul: 0.01 }]));
  };
  const { fetchImpl, calls } = stub(noToAll);
  const r = await askJev(root, "how does a request get matched", { apiKey: "test", fetchImpl });

  assert.equal(calls.length, 1, "no seeds, so the prune call is never made");
  assert.equal(r.jev.seeds.length, 0);
  assert.match(r.jev.fellBack ?? "", /no candidate cleared the seed bar/);
  // The lexical list is still returned — it beats nothing — but the pack is
  // labelled so the agent never reads it as a walked answer.
  assert.ok(r.hits.length > 0, "the lexical pack is still handed over");
  assert.match(formatJevAsk(r), /not a walked pack/);
});

test("jev arm: a low completeness score surfaces the gap instead of hiding it", async () => {
  const root = fixture();
  await buildGraph(root);
  const thin = (body: Record<string, unknown>) => {
    const out = yesToAll(body) as Record<string, { noul?: number }>;
    if ("complete" in out) out.complete = { noul: 0.2 };
    return out;
  };
  const { fetchImpl } = stub(thin);
  const r = await askJev(root, "how does a request get matched to a route", {
    apiKey: "test", fetchImpl, netWidth: 50,
  });
  assert.equal(r.jev.complete, 0.2);
  assert.match(formatJevAsk(r), /complete: 0\.20/);
});

test("jev client: a non-2xx answer is an error, not a silent empty result", async () => {
  const fetchImpl = (async () =>
    ({ ok: false, status: 401, statusText: "Unauthorized", text: async () => "User not found" }) as unknown as Response) as unknown as typeof fetch;
  await assert.rejects(
    () => decide({ state: "s", questions: {} }, { apiKey: "bad", fetchImpl }),
    (err: unknown) => err instanceof JevError && /401/.test((err as Error).message),
  );
});

test("jev client: a missing or malformed noul reads as unjudged, never as a zero", () => {
  assert.equal(noulOf({ noul: 0.4 }), 0.4);
  assert.equal(noulOf({}), null);
  assert.equal(noulOf(undefined), null);
  assert.equal(noulOf({ noul: Number.NaN }), null);
});
