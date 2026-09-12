# Backlog (in order)

Goal: **prove that BO adds lift over GEPA's selection, and find out where** — before any of it becomes a
default in `bpto/`. GEPA (Agrawal et al. 2025) has no surrogate: parent choice is Pareto-pool +
stochastic sampling (∝ examples won), child acceptance is a minibatch gate, mutation is a reflection LM.
The honest baseline for BO is therefore *Pareto-weighted sampling*, not uniform random. Findings go in
`experiments/`.

Where BO can go (one variable at a time):
- (1) **which candidate to expand** — replace the stochastic Pareto sampler with acquisition over
  ancestor-attributed value (`BOSelector`), or augment: Pareto pool as candidate set, acquisition inside it.
- (2) **which proposed children get rollouts** — surrogate pre-screen before the minibatch gate
  (`BOSelector.top(k, among=unevaluated)` in front of `evaluate`). Pure rollout saving.
- later: minibatch example selection by surrogate variance; merge-partner choice. Never mutation itself.

## Steps

1. ~~**Simplified GEPA as a peer of `bpto.bo`**~~ — done (`bpto/gepa/`: `pareto_sample` with mode switch,
   `ReflectiveExpander`, `gepa()` schedule with minibatch gate; task feedback in `tasks/*/feedback.py`;
   `--strategy gepa` in `tasks/ifbench/run.py`). Not yet run live.
2. ~~**Synthetic ladder, offline, $0.**~~ done - see `experiments/2026-09-11-synthetic-ladder/NOTES.md`. Add a
   `best+screen` arm to the live harness. Extend the planted-optimum benchmark: arms = Pareto sampling /
   BO replaces (1) / BO within Pareto pool / Pareto + BO child-screen (2) / both. Anytime curves
   (best-so-far vs rollouts), 50 seeds, mean ± SE. Include the warm-up hybrid "Pareto until k expansions,
   then BO". If BO does not beat the Pareto sampler here, stop and rethink.
3. **Live embedder smoke**: `BedrockEmbedder` (Titan V2) on ≤ 10 texts; then one BO-selected round on the
   compression task with a small cap.
4. ~~Equal-budget harness~~ built (`experiments/live_compare/compare.py`); IFBench run stopped after seed 0
   (tie by construction - see `experiments/2026-09-11-ifbench-gepa-vs-bo/`). Before re-running:
   fix the reflector (forced-structure prompt, temperature 1.0; Pro if still timid), and switch task.

4b. ~~Compression comparison~~ two runs done: maiden (`experiments/2026-09-11-compression-gepa-vs-bo/`)
   and v2 (`experiments/2026-09-12-compression-v2-gepa-vs-bo/`, 12 seeds x 2,000 rollouts, train 100,
   minibatch 5). BO 3.5 tokens shorter at the targeted accuracy (8-9/12 seeds), GEPA better at stricter
   bars the search was not aimed at. Readout = front graph (`front_plot.py`). Next, in order:
   (i) front-aware BO value (hypervolume / EI over the front), same harness and seeds (~$0.40);
   (ii) real extraction data (CoNLL-2003 PER or Few-NERD person) into `tasks/compression/data/`;
   (iii) held-out fronts (score all candidates on held-out, ~$0.20); (iv) parallel arm-seeds in the harness;
   (v) Bedrock retry on transient `ResourceNotFoundException`.

4c. **Equal-budget harness on Nova Micro**, same arms as step 2, ≥ 3 seeds, held-out ≥ 150, equal `Budget`
   per arm; rollouts on Micro, expansion via `Task.expander_client` on Nova Lite so the mutator is not the
   ceiling. Tasks: compression first (real objective landscape, cheap), then HotpotQA-distractor
   (`tasks/hotpotqa/`, `{question, context}` → answer, EM + F1; ~1.5–2k input tokens/example).
   Not IFBench: flat for every published optimizer; +2 pts is undetectable at n=300.
5. **Decide the informed improvement** from 2+4: replace (1), augment (1), add (2), or a combination →
   make it the documented default `BOSelector` usage, numbers in `experiments/`.
6. Stronger models / cross-model transfer (prompts found on Micro evaluated on Lite/Pro/Claude).
7. Multi-prompt programs only if HoVer / PUPA become targets.
8. Housekeeping: live checks for `AnthropicClient`, `AzureOpenAIEmbedder`, `VoyageEmbedder`; real token
   counting for `OpenAICompatibleClient`; `Stop(no_improvement_rounds)` behaviour on flat landscapes.
