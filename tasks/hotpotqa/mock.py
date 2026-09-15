"""Offline stand-in for the QA model: answers correctly with a probability that grows with how many "helpful"
cue words the prompt template contains, so a search has something to find. Plumbing checks only."""
from __future__ import annotations

import hashlib
import re

from bpto import Dataset, MockClient
from bpto.ops import Variants

from . import Answer

CUES = ["two paragraphs", "step", "compare", "shortest", "copy", "yes / no", "distractor", "bridge"]


def _mock_client(dataset: Dataset, base: float = 0.35, per_cue: float = 0.08, **kw) -> MockClient:
    answers = {ex.inputs["question"]: ex.answer for ex in dataset}
    calls = {"variants": 0}

    def handler(prompt, cfg, schema):
        if schema is Variants:  # each call adds a different cue so successive rounds propose different children
            n = int(re.search(r"(?:Return|write|Propose) (\d+)", prompt).group(1))
            base_t = re.search(r"<prompt>\n(.*?)\n</prompt>", prompt, re.S).group(1)
            calls["variants"] += 1
            outs = []
            for i in range(n):
                j = (calls["variants"] * 3 + i) % len(CUES)
                outs.append(f"{CUES[j]}. {base_t}" if (calls["variants"] + i) % 2 else f"{base_t} ({CUES[j]}, v{calls['variants']})")
            return Variants(prompts=outs)
        q = re.search(r"Question: (.*)$", prompt.split("Respond with a single JSON")[0].strip(), re.S)
        question = q.group(1).strip() if q else ""
        template_cues = sum(c in prompt for c in CUES)
        p = min(0.95, base + per_cue * template_cues)
        h = int(hashlib.md5((question + str(template_cues)).encode()).hexdigest(), 16) % 1000 / 1000
        gold = answers.get(question, "unknown")
        return Answer(answer=gold if h < p else "something else")
    return MockClient(handler, **kw)
