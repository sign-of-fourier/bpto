"""What every arm shares: the {narrative} slot, the label parser, the feedback string."""
from __future__ import annotations

import re

SLOT = "{narrative}"


def render(instruction: str, narrative: str) -> tuple[str, bool]:
    """The task prompt for one row, and whether the slot had to be restored (a rewrite dropped it)."""
    if SLOT in instruction:
        return instruction.replace(SLOT, narrative), False
    return f"{instruction.rstrip()}\n\nNarrative: {narrative}", True


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def parse_label(text: str, labels: list[str]) -> str | None:
    """Lenient and identical for every prompt: the last line that names exactly one label wins, else the label named
    last anywhere in the text. Longest labels are matched first so 'Falls to lower level' beats 'Falls'."""
    by_len = sorted(labels, key=lambda l: -len(l))
    normed = [(l, _norm(l)) for l in by_len]
    for line in reversed([ln for ln in text.splitlines() if ln.strip()]):
        nl = f" {_norm(line)} "
        hits = {l for l, n in normed if f" {n} " in nl}
        # drop labels that are substrings of another hit ("Falls" inside "Falls to lower level")
        hits = {h for h in hits if not any(h != o and _norm(h) in _norm(o) for o in hits)}
        if len(hits) == 1:
            return hits.pop()
    t = f" {_norm(text)} "
    best, pos = None, -1
    for l, n in normed:
        i = t.rfind(f" {n} ")
        if i > pos:
            best, pos = l, i
    return best


def feedback(pred: str | None, gold: str, error: str | None = None) -> str:
    if error:
        return f"The call failed ({error}); gold was {gold}."
    if pred is None:
        return f"No category could be read from the answer; gold was {gold}. Answer with one category name only."
    if pred == gold:
        return f"Correct: {gold}."
    return f"Predicted {pred}; gold was {gold}."
