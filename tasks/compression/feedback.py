"""Feedback text for reflective mutation on the compression task: missed and spurious names."""
from __future__ import annotations

from bpto import Example
from bpto.tree import ExampleResult

from . import _norm


def feedback(example: Example, result: ExampleResult) -> str:
    if result.error:
        return f"The response failed with an error: {result.error}"
    got = [str(x) for x in ((result.parsed or {}).get("names") or [])]
    ref = list(example.answer or [])
    missed = [r for r in ref if _norm(r) not in {_norm(g) for g in got}]
    spurious = [g for g in got if _norm(g) not in {_norm(r) for r in ref}]
    parts = [f"expected names: {ref or 'none'}; extracted: {got or 'none'}"]
    if missed:
        parts.append(f"missed: {missed}")
    if spurious:
        parts.append(f"should not have been extracted: {spurious}")
    if not missed and not spurious:
        parts.append("correct")
    parts.append(f"template tokens: {result.metrics.get('template_tokens', '?')} (shorter is better)")
    return "; ".join(parts)
