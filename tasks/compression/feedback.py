"""Feedback text and reflection prompt for reflective mutation on the compression task.

User-space design: bpto's `ReflectiveExpander` only asks for (example, result) -> text and a meta-prompt;
what a "failure" is and how the rewrite is steered ("shorter, same accuracy") is decided here.
"""
from __future__ import annotations

from bpto import Example
from bpto.tree import ExampleResult

from . import _norm

# GEPA-style reflection with a compression directive instead of "add rules".
COMPRESS_REFLECT_PROMPT = (
    "I gave an assistant the following prompt template to perform a task. The template is supposed to: "
    "{description}\nIt must keep these placeholders exactly, in curly braces: {placeholders}\n\n"
    "Current prompt template:\n<prompt>\n{prompt}\n</prompt>\n\n"
    "Here are examples of task inputs, the assistant's response under this prompt, and feedback on each:\n\n"
    "{directive}\n"
    "Goal: a SHORTER prompt template with the same or better extraction accuracy. Read the feedback: keep "
    "whatever rule prevents the misses and spurious names shown, cut everything else - filler, repetition, "
    "explanations the model does not need. If the feedback shows failures, fix them in as few words as "
    "possible. Then write {n} shorter prompt template(s). Each must be complete and usable on its own.{seed}"
)


def _split(example: Example, result: ExampleResult) -> tuple[list[str], list[str], list[str], list[str]]:
    got = [str(x) for x in ((result.parsed or {}).get("names") or [])]
    ref = list(example.answer or [])
    missed = [r for r in ref if _norm(r) not in {_norm(g) for g in got}]
    spurious = [g for g in got if _norm(g) not in {_norm(r) for r in ref}]
    return got, ref, missed, spurious


def passed(example: Example, result: ExampleResult) -> bool:
    """A trace is a success when the extracted set is exactly right (independent of the token objective)."""
    if result.error:
        return False
    _, _, missed, spurious = _split(example, result)
    return not missed and not spurious


def feedback(example: Example, result: ExampleResult) -> str:
    if result.error:
        return f"The response failed with an error: {result.error}"
    got, ref, missed, spurious = _split(example, result)
    parts = [f"expected names: {ref or 'none'}; extracted: {got or 'none'}"]
    if missed:
        parts.append(f"missed: {missed}")
    if spurious:
        parts.append(f"should not have been extracted: {spurious}")
    if not missed and not spurious:
        parts.append("correct")
    parts.append(f"template tokens: {result.metrics.get('template_tokens', '?')} (shorter is better)")
    return "; ".join(parts)
