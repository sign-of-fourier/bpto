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


# --- two-stage program (tasks.hotpotqa.program): feedback on the *selection*, which is what the searched prompt controls

def _selected(example: Example, result: ExampleResult) -> list[str]:
    from .program import select_titles
    return select_titles(result.parsed, example.inputs["context"])


def program_passed(example: Example, result: ExampleResult) -> bool:
    return not result.error and result.metrics.get("sel_recall", 0.0) == 1.0 and result.metrics.get("em", 0.0) == 1.0


def program_feedback(example: Example, result: ExampleResult) -> str:
    if result.error:
        return f"The response failed with an error: {result.error}"
    gold = list(example.meta.get("supporting_titles") or [])
    picked = _selected(example, result)
    norm = lambda t: " ".join(str(t).lower().split())
    missed = [t for t in gold if norm(t) not in {norm(p) for p in picked}]
    extra = [p for p in picked if norm(p) not in {norm(t) for t in gold}]
    m = result.metrics
    parts = [f"selected {picked}; the paragraphs actually needed were {gold}"]
    if missed:
        parts.append(f"MISSED {missed} - the answer could not be found without them")
    if extra:
        parts.append(f"unnecessary: {extra}")
    if not missed and not extra:
        parts.append("selection was exactly right")
    parts.append(f"answer stage (given only the selected paragraphs): expected {example.answer!r}, answer F1 {m.get('f1', 0.0):.2f}"
                 + (" (exact match)" if m.get("em", 0.0) == 1.0 else ""))
    if example.meta.get("type") == "comparison":
        parts.append("comparison question: the paragraphs for both compared entities are needed")
    return "; ".join(parts)


# --- answerer module (program with both modules under search): what stage B saw is in result.trace["answerer"]

def answerer_passed(example: Example, result: ExampleResult) -> bool:
    return not result.error and result.metrics.get("em", 0.0) == 1.0


def answerer_feedback(example: Example, result: ExampleResult) -> str:
    if result.error:
        return f"The response failed with an error: {result.error}"
    tr = (result.trace or {}).get("answerer") or {}
    pred = tr.get("answer", "")
    em, f1 = em_f1(str(pred), example.answer or "")
    parts = [f"expected answer: {example.answer!r}; model answered: {pred!r}; F1 {f1:.2f}" + (" (exact match)" if em else "")]
    if result.metrics.get("sel_recall", 1.0) < 1.0:
        parts.append("NOTE: the paragraphs given to this stage were incomplete (a needed one was not selected), so a "
                     "full answer may not have been possible from them - do not over-fit the prompt to this case")
    elif not em:
        parts.append("the paragraphs given contained the facts needed: the answer stage itself fell short "
                     "(wrong fact, wrong span, or extra words - F1 rewards the shortest exact span)")
    if example.meta.get("type") == "comparison":
        parts.append("comparison question: look up the attribute for both entities, then compare")
    return "; ".join(parts)


def answerer_trace_order(example: Example, result: ExampleResult) -> int:
    """Sort key for the answerer's minibatch: recoverable misses first (gold paragraphs present, answer wrong), then
    misses the selector caused, then passes."""
    if result.error:
        return 0
    if result.metrics.get("em", 0.0) == 1.0:
        return 2
    return 0 if result.metrics.get("sel_recall", 0.0) == 1.0 else 1


FEEDBACK = {"selector": program_feedback, "answerer": answerer_feedback}
PASSED = {"selector": program_passed, "answerer": answerer_trace_order}  # int key: recoverable misses first
