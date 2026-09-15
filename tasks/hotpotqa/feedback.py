"""Feedback text for reflective mutation on HotpotQA: expected vs predicted answer, F1, and the two gold
paragraphs (titles) the answer had to be assembled from - the "supporting facts" signal GEPA's reflector gets."""
from __future__ import annotations

from bpto import Example
from bpto.tree import ExampleResult

from . import em_f1


def predicted(result: ExampleResult) -> str:
    p = (result.parsed or {}).get("answer") if isinstance(result.parsed, dict) else None
    return str(p) if p is not None else (result.output or "").strip()[-200:]


def passed(example: Example, result: ExampleResult) -> bool:
    return not result.error and em_f1(predicted(result), example.answer or "")[0] == 1.0


def feedback(example: Example, result: ExampleResult) -> str:
    if result.error:
        return f"The response failed with an error: {result.error}"
    pred = predicted(result)
    em, f1 = em_f1(pred, example.answer or "")
    parts = [f"expected answer: {example.answer!r}; model answered: {pred!r}; F1 {f1:.2f}"
             + (" (exact match)" if em else "")]
    titles = example.meta.get("supporting_titles") or []
    if titles:
        parts.append(f"the answer had to be assembled from the paragraphs titled {titles}")
    if not em and example.meta.get("type") == "comparison":
        parts.append("this is a comparison question: both entities must be looked up and compared")
    return "; ".join(parts)
