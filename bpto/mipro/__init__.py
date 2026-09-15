"""A simplified MIPROv2 (Opsahl-Ong et al. 2024) on top of the bpto tree - a peer of `bpto.gepa` and `bpto.bo`.

MIPROv2 = (1) propose N instruction candidates *once*, up front, with a grounded proposer (dataset summary +
a rotating "tip"), (2) bootstrap few-shot demo sets, (3) Bayesian optimisation with TPE over the *discrete*
choice of (instruction, demo set), scored on minibatches with periodic full evaluations of the incumbent.

Here: (1) is `GroundedProposer` (an `LLMExpander` whose directive carries the summary and a per-call tip),
(2) is dropped (0-shot MIPRO, which the paper also ablates - and the compression task is zero-shot), and (3)
is `CategoricalTPE` over the candidate index. Versus GEPA the candidates are never mutated again and the
proposer sees no execution feedback; versus bpto's BO the surrogate is a density ratio over a fixed set
rather than a GP over embeddings on a growing tree. The schedule (trial -> minibatch -> periodic full eval)
lives with the experiment harness because rollout accounting is what is being compared.
"""
from __future__ import annotations

import random as _random
from collections import Counter
from typing import Any, Sequence

from ..data import Dataset
from ..ops import LLMExpander
from ..tree import Node, Tree

# MIPROv2's proposer tips (dspy.propose.grounded_proposer), lightly paraphrased.
TIPS = {
    "none": "",
    "creative": "Don't be afraid to be creative when writing the instruction.",
    "simple": "Keep the instruction clear and concise.",
    "description": "Make sure the instruction is very informative and descriptive.",
    "high_stakes": "The instruction should include a high-stakes scenario in which the model must solve the task.",
    "persona": "Include a persona that is relevant to the task in the instruction (e.g. 'You are a ...').",
}

PROPOSE_PROMPT = (
    "You are an expert prompt engineer. Write instruction prompt templates for a language model that will "
    "perform this task: {description}\n"
    "Each template must keep these placeholders exactly, in curly braces: {placeholders}\n\n"
    "A starting template (do not just rephrase it; you may keep, restructure or replace its content):\n"
    "<prompt>\n{prompt}\n</prompt>\n\n"
    "{directive}\n"
    "Propose {n} distinct instruction templates. Each must be complete and usable on its own.{seed}"
)


def dataset_summary(dataset: Dataset, k: int = 3, seed: int = 0, max_chars: int = 400) -> str:
    """A few (inputs, answer) pairs, truncated - the proposer's view of the data (MIPRO summarises with an LM;
    showing rows is the cheaper equivalent)."""
    rows = dataset.sample(k, seed=seed)
    parts = []
    for ex in rows:
        inputs = "; ".join(f"{name}: {str(v)[:max_chars]}{'...' if len(str(v)) > max_chars else ''}" for name, v in ex.inputs.items())
        parts.append(f"- {inputs}\n  expected answer: {ex.answer!r}")
    return "\n".join(parts)


class GroundedProposer(LLMExpander):
    """One-shot fan-out of instruction candidates: `calls` structured calls of `n` each, every call grounded in
    a dataset summary and steered by a different tip (MIPRO's diversity mechanism, not temperature)."""
    name = "mipro_propose"

    def __init__(self, summary: str, n: int = 4, calls: int = 3, tips: Sequence[str] = tuple(TIPS), seed: int = 0, **kw):
        super().__init__(n=n, calls=calls, meta_prompt=kw.pop("meta_prompt", PROPOSE_PROMPT), **kw)
        self.summary, self.tips, self.seed = summary, list(tips), seed

    def tip_for(self, i: int) -> str:
        order = list(self.tips)
        _random.Random(self.seed).shuffle(order)
        return order[i % len(order)]

    def render_meta(self, tree: Tree, node: Node, seed: int) -> str:
        tip = TIPS.get(self.tip_for(seed), self.tip_for(seed))
        directive = f"Examples from the dataset:\n{self.summary}\n" + (f"\nTip: {tip}\n" if tip else "")
        return self.meta_prompt.format(
            n=self.n, prompt=node.prompt.template, description=tree.task.description or "(unspecified)",
            directive=directive, placeholders=", ".join("{%s}" % p for p in node.prompt.placeholders),
            seed=f" (batch {seed})" if self.calls > 1 else "",
        )

    def params(self) -> dict[str, Any]:
        p = super().params()
        p["directive"] = "grounded:" + ",".join(self.tips)
        return p


class CategoricalTPE:
    """Tree-structured Parzen estimator over one categorical variable (what Optuna's TPESampler reduces to for
    MIPRO's instruction index): split observed trials at the top `gamma` quantile into good/bad, model each
    side as a Laplace-smoothed histogram over candidates, and pick argmax l(c)/g(c). Each candidate is tried
    once first (startup trials)."""

    def __init__(self, n_candidates: int, gamma: float = 0.25, seed: int = 0):
        self.k, self.gamma, self.rnd = n_candidates, gamma, _random.Random(seed)
        self.trials: list[tuple[int, float]] = []

    def observe(self, candidate: int, score: float) -> None:
        self.trials.append((candidate, score))

    def scores(self) -> dict[int, float]:
        """Mean observed score per candidate."""
        by: dict[int, list[float]] = {}
        for c, s in self.trials:
            by.setdefault(c, []).append(s)
        return {c: sum(v) / len(v) for c, v in by.items()}

    def ratios(self) -> dict[int, float]:
        n_good = max(1, int(round(self.gamma * len(self.trials))))
        ordered = sorted(self.trials, key=lambda t: -t[1])
        good, bad = Counter(c for c, _ in ordered[:n_good]), Counter(c for c, _ in ordered[n_good:])
        ng, nb = sum(good.values()), sum(bad.values())
        return {c: ((good[c] + 1) / (ng + self.k)) / ((bad[c] + 1) / (nb + self.k)) for c in range(self.k)}

    def suggest(self) -> int:
        untried = [c for c in range(self.k) if all(t != c for t, _ in self.trials)]
        if untried:
            return self.rnd.choice(untried)
        r = self.ratios()
        top = max(r.values())
        return self.rnd.choice([c for c, v in r.items() if v == top])


__all__ = ["GroundedProposer", "CategoricalTPE", "dataset_summary", "TIPS", "PROPOSE_PROMPT"]
