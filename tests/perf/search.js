// Dev-loop latency probe for GET /search.
// Reads server-side spans from the `X-Perf-Spans` response header
// (added when PERF_TRACE=1 — see readme_docs/SEARCH_LATENCY_PLAN.md step 3).
//
// Usage:
//   k6 run -e OUTFILE=readme_docs/perf_baselines/baseline_step0_search.json tests/perf/search.js
// Or via Makefile:
//   make perf-search STEP=0

import http from 'k6/http';
import { check } from 'k6';
import { Trend } from 'k6/metrics';
import { textSummary } from 'https://jslib.k6.io/k6-summary/0.0.2/index.js';

const QUERIES = [
  'indemnification clause',
  'termination for convenience',
  'governing law',
  'limitation of liability',
  'assignment restrictions',
];

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';
const TOP_K = __ENV.TOP_K || '10';
const WARMUP = parseInt(__ENV.WARMUP || '2');
const RUNS = parseInt(__ENV.RUNS || '5');

const spans = {
  total: new Trend('span_total', true),
  embed: new Trend('span_embed', true),
  qdrant_query: new Trend('span_qdrant_query', true),
  rerank: new Trend('span_rerank', true),
  highlight_assemble: new Trend('span_highlight_assemble', true),
};

export const options = {
  scenarios: {
    sequential: {
      executor: 'per-vu-iterations',
      vus: 1,
      iterations: QUERIES.length * (WARMUP + RUNS),
      maxDuration: '10m',
    },
  },
  thresholds: {
    http_req_failed: ['rate==0'],
    // Starter absolute gates; tighten after step 1 lands.
    // For step-over-step gating, derive these from the prior baseline JSON
    // and export them as env vars in the Makefile target.
    'span_total{warmup:false}': [{ threshold: 'p(95)<30000', abortOnFail: false }],
    'span_rerank{warmup:false}': [{ threshold: 'p(95)<25000', abortOnFail: false }],
  },
};

export default function () {
  const perQ = WARMUP + RUNS;
  const qIdx = Math.floor(__ITER / perQ);
  const slot = __ITER % perQ;
  const isWarmup = slot < WARMUP;
  const q = QUERIES[qIdx];

  const url = `${BASE_URL}/search?q=${encodeURIComponent(q)}&top_k=${TOP_K}`;
  const r = http.get(url, {
    tags: { query: q, warmup: String(isWarmup) },
  });

  check(r, { 'status 200': (res) => res.status === 200 });
  if (isWarmup) return;

  const header = r.headers['X-Perf-Spans'];
  if (!header) return;
  let parsed;
  try {
    parsed = JSON.parse(header);
  } catch (e) {
    return;
  }
  for (const [k, v] of Object.entries(parsed)) {
    if (spans[k] && typeof v === 'number') {
      spans[k].add(v, { query: q, warmup: 'false' });
    }
  }
}

export function handleSummary(data) {
  const out = __ENV.OUTFILE || 'baseline_search.json';
  return {
    [out]: JSON.stringify(data, null, 2),
    stdout: textSummary(data, { indent: ' ', enableColors: true }),
  };
}
