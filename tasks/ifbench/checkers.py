"""Adapter over the official IFBench constraint checkers (github.com/allenai/IFBench, not on PyPI).

`check(response, ids, kwargs)` reproduces `evaluation_lib.test_instruction_following_{strict,loose}`
for one example. Strict = the raw response passes; loose = any of the 8 relaxations passes
(markdown `*` stripped, first/last line dropped). Import of `ifbench` is deferred so the rest
of the task (and its tests) work without it installed.
"""
from __future__ import annotations

from typing import Callable, Sequence

Checker = Callable[[str, Sequence[str], Sequence[dict]], tuple[list[bool], list[bool]]]


def _relaxations(response: str) -> list[str]:
    lines = response.split("\n")
    variants = [response, "\n".join(lines[1:]).strip(), "\n".join(lines[:-1]).strip(), "\n".join(lines[1:-1]).strip()]
    return variants + [v.replace("*", "") for v in variants]


def _build(instruction_id: str, kwargs: dict, prompt: str):
    from ifbench import instructions_registry
    inst = instructions_registry.INSTRUCTION_DICT[instruction_id](instruction_id)
    inst.build_description(**{k: v for k, v in kwargs.items() if v is not None})
    args = inst.get_instruction_args()
    if args and "prompt" in args:
        inst.build_description(prompt=prompt)
    return inst


def check(response: str, ids: Sequence[str], kwargs: Sequence[dict], prompt: str = "") -> tuple[list[bool], list[bool]]:
    """Per-constraint (strict, loose) pass flags. Empty responses fail everything."""
    strict, loose = [], []
    if not response or not response.strip():
        return [False] * len(ids), [False] * len(ids)
    relaxed = _relaxations(response)
    for cid, kw in zip(ids, kwargs):
        inst = _build(cid, kw, prompt)
        strict.append(bool(inst.check_following(response)))
        loose.append(any(r.strip() and inst.check_following(r) for r in relaxed))
    return strict, loose


def available() -> bool:
    try:
        import ifbench  # noqa: F401
        return True
    except ImportError:
        return False
