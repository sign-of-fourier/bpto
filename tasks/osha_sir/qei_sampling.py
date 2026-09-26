"""bpto's q-EI in official GEPA's expand seat: a `SamplingStrategy` that picks q parents per iteration jointly.

Everything else is GEPA's own engine (minibatch, strict gate, reflection prompt and parser, Pareto pool, accounting),
so the arm differs from `IndependentSampling(q)` in one decision: *which* q parents. The surrogate is bpto's
`BOSelector`, unchanged, fed with stand-ins for GEPA's candidates:

  target  every proposal is one observation at its parent's input: the child's estimated full score = parent's full
          validation score + (child - parent) on the child's own minibatch rows (the gepa-ei rule; the bare gain
          favours weak parents). Read from `state.full_program_trace`, so it survives a resume.
  noise   the child's minibatch SE (sd of its per-row scores / sqrt(rows)), PIT-rescaled inside BOSelector.
  among   every program GEPA has accepted (all fully evaluated), as in the gepa-ei arm; EI does the pruning. GEPA's
          own sampler draws from the non-dominated subset, which on a heritable landscape is often one program.
  warmup  GEPA's Pareto sampler (q independent draws) until `warmup` candidates have observations.
  q       always q tasks: when the pool (or the joint pick) has fewer than q distinct parents, the picks repeat, each
          with its own minibatch - as IndependentSampling does - so every iteration proposes q children.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from statistics import pstdev

from gepa.gepa_utils import find_dominator_programs
from gepa.strategies.proposal_sampling import ProposalTask

from bpto.bo import EI, GPR, QEI, BOSelector


@dataclass
class _Prompt:
    template: str


class _Node:
    """What BOSelector reads from a node: id, prompt.template, embedding (filled once, then reused)."""

    def __init__(self, idx: int, text: str):
        self.id, self.prompt, self.embedding = str(idx), _Prompt(text), None
        self.obs: list[float] = []
        self.ses: list[float] = []


def observations(state) -> list[tuple[int, float, float]]:
    """(parent_idx, estimated child full score, SE) for every proposal whose child was minibatch-evaluated."""
    full = state.program_full_scores_val_set
    out = []
    for it in state.full_program_trace:
        for t in it.get("tasks") or []:
            old, new = t.get("subsample_scores"), t.get("new_subsample_scores")
            if not old or not new or len(old) != len(new):
                continue
            m = len(new)
            est = full[t["parent_idx"]] + (sum(new) - sum(old)) / m
            out.append((t["parent_idx"], est, pstdev(new) / m ** 0.5 if m > 1 else 0.0))
    return out


class QEISampling:
    def __init__(self, q: int, embedder, *, warmup: int = 2, pca: int | None = 4, seed: int = 0, batch=None):
        self.q, self.embedder, self.warmup = q, embedder, warmup
        self.bo = BOSelector(embedder, GPR(), EI(), value=self._value, noise=self._noise, transform="pit",
                             pca=pca, batch=batch or QEI(seed=seed))
        self.rng = random.Random(seed)
        self.nodes: dict[int, _Node] = {}
        self.log: list[dict] = []
        self.bridge = None

    def bind(self, bridge):  # run_official hands over its event loop (the embedder and BOSelector are async)
        self.bridge = bridge

    @staticmethod
    def _value(n, tree):
        return list(n.obs) or None

    @staticmethod
    def _noise(n, tree):
        return list(n.ses) or None

    def sample_tasks(self, state, candidate_selector, batch_sampler, trainset) -> list[ProposalTask]:
        for i, cand in enumerate(state.program_candidates):
            if i not in self.nodes:
                self.nodes[i] = _Node(i, cand["instruction"])
        for n in self.nodes.values():
            n.obs, n.ses = [], []
        for p, est, se in observations(state):
            self.nodes[p].obs.append(est)
            self.nodes[p].ses.append(se)
        pool = list(range(len(state.program_candidates)))
        entry_front = sorted(find_dominator_programs(state.get_pareto_front_mapping(), state.per_program_tracked_scores))
        trained = [i for i in pool if self.nodes[i].obs]
        entry = {"iteration": state.i + 1, "pool": len(pool), "pareto_front": entry_front, "n_trained": len(trained),
                 "n_obs": sum(len(n.obs) for n in self.nodes.values())}
        if len(trained) < min(self.warmup, len(pool)) or len(pool) == 1:
            parents = [candidate_selector.select_candidate_idx(state) for _ in range(self.q)]
            entry.update(mode="warmup", parents=parents)
        else:
            tree = [self.nodes[i] for i in range(len(state.program_candidates))]
            cands = [self.nodes[i] for i in pool]
            picks = self.bridge.run(self.bo.top(min(self.q, len(cands)), among=lambda t: cands)(tree))
            parents = [int(n.id) for n in picks]
            while len(parents) < self.q:           # pool smaller than q: repeat the joint picks
                parents.append(parents[len(parents) % len(picks)])
            fit = {k: v for k, v in self.bo.last_fit.items() if k != "pred"}
            entry.update(mode="qei", parents=parents, fit=fit,
                         pred={k: [float(x) for x in v] for k, v in self.bo.last_fit.get("pred", {}).items()})
        self.log.append(entry)
        tasks = []
        for p in parents:
            mb_ids = batch_sampler.next_minibatch_ids(trainset, state)
            tasks.append(ProposalTask(p, state.program_candidates[p], mb_ids, trainset.fetch(mb_ids)))
        return tasks
