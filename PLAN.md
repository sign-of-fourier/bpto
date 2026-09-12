# Plan: parallel BO through the batch-suggestion service

Status: planned, not started (2026-09-12). Contract: `CLIENT_GUIDE.md` (stateless `POST` with `X`, `y`,
`candidates`, `q` -> the `q` pool members that are the best *joint* next batch). Dataset: to be chosen by
the user before coding; everything below is dataset-agnostic.

## Why

`experiments/2026-09-12-compression-v2-gepa-vs-bo/` showed acquisition + surrogate child screen finding
shorter prompts than Pareto-weighted sampling at the targeted accuracy (3-5 tokens, 8-9/12 seeds). That BO
is sequential: 1 parent per round, top-1 of 3 children by a ranking. A joint batch of `q` is (a) how the
search parallelises - `q` expansions per round, wall time / q - and (b) a better batch than top-q of a
ranking, which picks near-duplicates. The service supplies exactly that and is stateless, which suits us:
our `y` is re-scored whenever the accuracy floor moves, so we simply resend it.

## What changes in `bpto/` (small)

1. `bpto/bo/remote.py`: `RemoteBatchSelector(embedder, value, q, url=None, timeout=180, retries=1)`.
   Selector only (`tree -> list[Node]`), peer of `BOSelector`; the tree and ops never see it.
   - `X` = embeddings of nodes with `value(node) is not None`; `y` = those values (higher = better; our
     values are already "gain", so no negation); `candidates` = embeddings of the nodes passed in;
     `q` = min(q, len(candidates)). Returns the `q` nodes by `index`.
   - Fewer than ~5 training rows: fall back to the caller-supplied warm-up selector (GEPA sampler), as
     the harness already does for the local GPR.
   - Records `mu`/`sigma` per returned node in `tree.meta["bo_fits"]` for analysis, not decisions.
   - URL from `MODAL_BO_API_URL` (`.env`); cold start 20-30 s -> timeout 180 s, one retry.
   - Same object serves both roles: parent choice (`value=best_children_gain`) and child pre-screen
     (`value=own gain`), two instances, two calls per round.
2. Tests: `tests/test_remote_bo.py` with a fake `urllib`/httpx transport returning canned indices; assert
   request shape (same column count in `X` and `candidates`, `y` length, `q`), fallback below 5 rows,
   retry-once on timeout. No network.
3. Nothing else in `bpto/` changes. `BOSelector` (local GPR) stays as the sequential baseline.

## What changes in the harness (`experiments/live_compare/compress.py`)

- `--q N` (parents per round; default 1 = current behaviour) and `--arms gepa bo bo-service`.
  - gepa: `pareto_sample(q, "weighted", metric=...)`, 1 child each -> q children/round.
  - bo (local): `BOSelector.rank(...)[:q]` parents, 3q children, `child_bo` keeps q.
  - bo-service: `RemoteBatchSelector(q)` parents, 3q children, second `RemoteBatchSelector(q)` keeps q.
- Control arm `gepa-3`: Pareto sample 1 parent, 3 children, keep 1 *at random* -> isolates "more
  reflector proposals" from "the surrogate picked the right one".
- Rollout accounting unchanged (task-model calls); add `reflect_calls` to the summary so the
  reflector-call asymmetry is visible in every table.
- Independent arm-seeds run in parallel (`asyncio.gather` over a semaphore of 3-4); the shared `Budget`
  is already concurrency-safe (reserve-before-await).

## Experiment

- Same design as v2 unless the new dataset dictates otherwise: train 100 / held-out 200, minibatch 5,
  2,000 rollouts, floor root - 0.15 -> root - 0.05 keyed on rollouts, 12 seeds, hard dollar cap.
- Arms: gepa-q4, gepa-3 (control), bo-local-q4, bo-service-q4. 48 runs; at v2 prices ~$0.85, cap $1.50.
  Wall time ~30 min with 4 runs in parallel.
- Readout: `front_plot.py` (pooled + per-seed fronts, mean ± SE, paired differences vs gepa-q4) and
  `analyze_compress.py` (threshold table, union-front regret, rollout accounting). Report the full front,
  never one row.
- Success = bo-service-q4 front left of gepa-q4 with the ±2 SE band clear of zero over the targeted
  region, *and* not worse than bo-local-q4 (joint batch >= sequential top-q). If bo-local-q4 == bo-service,
  the batch mechanism is not where the value is.

## Open questions for the service owner (from `CLIENT_GUIDE.md`: "ask; do not guess")

- 256-dim Titan embeddings with 15-40 training rows: does the service project / learn features, or fit
  in the raw space? (Our local GPR fits raw; same regime.)
- Randomness: is there a seed field? Without one runs are not exactly replayable (acceptable, noted).
- Is calling it twice per round with different `X`/`y` (parent model vs child model) the intended usage?
- Egress from this machine to the Modal URL (not yet verified).

## Follow-ups that stay in the backlog

Front-aware BO value (hypervolume gain instead of gain-at-the-floor) - the strict-end rows in v2 ask for
it; real extraction data; held-out fronts; Bedrock retry on transient `ResourceNotFoundException`.
