// Dev-loop latency probe for POST /chat.
// Mirrors search.js but with the chat-side spans (retrieve / chat_completion / total).
//
// Usage:
//   k6 run -e OUTFILE=readme_docs/perf_baselines/baseline_step0_chat.json tests/perf/chat.js
// Or via Makefile:
//   make perf-chat STEP=0

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
const TOP_K = parseInt(__ENV.TOP_K || '10');
const WARMUP = parseInt(__ENV.WARMUP || '2');
const RUNS = parseInt(__ENV.RUNS || '5');

const spans = {
  total: new Trend('span_total', true),
  retrieve: new Trend('span_retrieve', true),
  chat_completion: new Trend('span_chat_completion', true),
};

export const options = {
  scenarios: {
    sequential: {
      executor: 'per-vu-iterations',
      vus: 1,
      iterations: QUERIES.length * (WARMUP + RUNS),
      maxDuration: '20m',
    },
  },
  thresholds: {
    http_req_failed: ['rate==0'],
    'span_total{warmup:false}': [{ threshold: 'p(95)<60000', abortOnFail: false }],
    'span_chat_completion{warmup:false}': [{ threshold: 'p(95)<45000', abortOnFail: false }],
  },
};

export default function () {
  const perQ = WARMUP + RUNS;
  const qIdx = Math.floor(__ITER / perQ);
  const slot = __ITER % perQ;
  const isWarmup = slot < WARMUP;
  const q = QUERIES[qIdx];

  const body = JSON.stringify({ query: q, top_k: TOP_K });
  const r = http.post(`${BASE_URL}/chat`, body, {
    headers: { 'Content-Type': 'application/json' },
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
  const out = __ENV.OUTFILE || 'baseline_chat.json';
  return {
    [out]: JSON.stringify(data, null, 2),
    stdout: textSummary(data, { indent: ' ', enableColors: true }),
  };
}
