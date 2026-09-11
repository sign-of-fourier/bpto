# IFBench: GEPA vs BO at equal rollouts — seed 0 only, stopped (2026-09-11)

    python -m experiments.live_compare.compare --task ifbench --seeds 5 --rollouts 1200 --n-train 100 --holdout 200 --max-usd 1.00

**Hypothesis (stated before running):** a node's fitness is not its value as a *parent*. GEPA samples parents
by fitness (∝ examples won); BO with ancestor attribution predicts which node will produce good children
and trades exploration/exploitation via EI. Prediction: BO ≥ GEPA on held-out at equal rollouts.

**Arms** (identical task, split, reflective mutator, 3-example minibatch gate, rollout budget, held-out):
- `gepa` — faithful simplified GEPA: Pareto pool, sample ∝ wins, 1 parent/round, 1 child.
- `bo` — parent = argmax EI over all candidates, surrogate on `value = best child`, Titan V2 embeddings;
  3 children proposed, surrogate (`own_score`) keeps 1 for the minibatch. First 4 expansions use the GEPA
  sampler (no attribution data yet).

Models: Nova Micro rollouts, Nova Lite reflection (with a "make substantive changes" directive), Titan V2
embeddings, all at default temperature (0.7). 100 train / 200 held-out, seed-0 split.

## Result

| arm | rollouts | nodes | pool (full-evaluated) | best train `strict` | root held | best held | reflections | time |
|---|---|---|---|---|---|---|---|---|
| gepa | 1,216 | 56 | 12 | 0.350 (= root) | 0.365 | 0.365 | 58 | 8.7 min |
| bo | 1,201 | 216 | 11 | 0.350 (= root) | 0.365 | 0.365 | 81 | 12.9 min |

A tie **by construction**: in neither arm did any accepted child beat the root on the full train set, so
"best" is the root in both and held-out is identical. Pool scores: gepa 0.35, 0.34, 0.34, 0.34, 0.33, 0.32 …;
bo 0.35, 0.34, 0.34, 0.32, 0.32 …. Eleven children per arm passed the 3-example minibatch gate and then
scored below the root on 100 - the noisy-gate behaviour from the synthetic ladder's "noise 0.3" condition.

Spend: $0.16 total including the plumbing runs. Run stopped after seed 0; seeds 1-4 would have tied the
same way for ~$0.50.

## Why

1. **The mutator only paraphrases.** The second-best prompt in each arm is the root with one adjective
   changed ("careful" -> "meticulous"). Lite ignores the "substantive changes" directive. With near-clone
   children, selection strategy cannot matter - this is the ladder's "dumb mutator" corner.
2. **No headroom.** Micro's ceiling on this split looks like ~0.35 `strict`; IFBench barely moves for anyone
   (GEPA: +1.7 on Qwen-8B).
3. **Warm-up.** BO's first ~400 of 1,200 rollouts are GEPA by design (no attribution data before parents
   have children), so the arms could only differ in the last two-thirds.

## Plumbing fixed along the way (all in the library, all tested)

- Lite wraps rewrites in `<prompt>` tags (copied from the meta-prompt) -> stripped in `LLMExpander`.
- Lite invents placeholders (`{constraint_name: ...}`) -> variants must have *exactly* the parent's placeholders.
- One `Origin` object was shared by all siblings -> per-child `Origin`.
- A malformed structured reply from the reflector killed the run -> the expansion yields no children.
- `Budget(max_usd, prices, parent=)`: one dollar meter across task client, reflector and embedder.

## Follow-ups (ordered)

1. **Make the reflector rewrite, not paraphrase.** Forced-structure meta-prompt ("add at least two concrete
   rules and a step-by-step procedure; the result must differ from the original in more than wording"),
   temperature 1.0 on the reflector; check on one seed (~$0.05). If still timid, Nova Pro for reflection only
   (~$0.002/call, ~$1.40 for 2 seeds).
2. **Move to a task with headroom** (compression, then HotpotQA) - see BACKLOG for the compression design.
3. **Warm-up alternative:** bootstrap the parent surrogate from `own_score` until attribution data exists, so
   BO covers the whole budget; compare against the GEPA-sampler warm-up.
4. Harness: `rollouts` in results.jsonl includes held-out calls when they miss the cache (gepa s0 shows
   1,416) - record search calls separately; skip parents whose last expansion produced zero children
   (BO's deterministic argmax could otherwise re-select a persistently failing node).
5. Reduce the gate's noise: minibatch 3 is GEPA's setting, but with a 0.7-temperature task model and
   binary per-example scores it passes ~1 in 4 children that then lose on the full set. Options: larger
   minibatch, or a nested-sample rung (successive halving) before the full set.
