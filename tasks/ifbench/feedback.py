"""Feedback text for reflective mutation: per constraint, its natural-language description (from the
official checker's `build_description`) and whether the response passed it."""
from __future__ import annotations

from bpto import Example
from bpto.tree import ExampleResult

from .checkers import _build, check


def describe(instruction_id: str, kwargs: dict, prompt: str = "") -> str:
    try:
        return _build(instruction_id, kwargs, prompt).build_description(**{k: v for k, v in kwargs.items() if v is not None})
    except Exception:
        return f"{instruction_id} {kwargs}"


def feedback(example: Example, result: ExampleResult) -> str:
    if result.error:
        return f"The response failed with an error: {result.error}"
    ids, kwargs = example.meta["instruction_id_list"], example.meta["kwargs"]
    strict, loose = check(result.output, ids, kwargs, example.inputs["instruction"])
    lines = []
    for cid, kw, s, l in zip(ids, kwargs, strict, loose):
        status = "PASSED" if s else ("passed only after stripping markdown/first-last line" if l else "FAILED")
        lines.append(f"- constraint `{cid}` ({describe(cid, kw, example.inputs['instruction'])}): {status}")
    return "\n".join(lines)
