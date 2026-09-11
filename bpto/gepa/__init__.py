"""A simplified GEPA (Agrawal et al. 2025, "GEPA: Reflective Prompt Evolution Can Outperform
Reinforcement Learning") on top of the bpto tree - a peer of `bpto.bo`. Same tree, ops, cache and
budget; only parent selection (Pareto pool + stochastic sampling) and mutation (reflection on
feedback) differ. Built to be taken apart: each ingredient is a switch or a selector so BO can
replace one at a time.
"""
from .loop import beats_parent, gepa, minibatch_for
from .reflect import REFLECT_PROMPT, Feedback, ReflectiveExpander, default_feedback
from .select import candidates, example_scores, pareto_pool, pareto_sample

__all__ = ["gepa", "beats_parent", "minibatch_for", "ReflectiveExpander", "REFLECT_PROMPT", "Feedback",
           "default_feedback", "pareto_pool", "pareto_sample", "candidates", "example_scores"]
