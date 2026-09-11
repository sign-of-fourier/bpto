import re

import pytest
from pydantic import BaseModel

from bpto import (Dataset, LinearObjective, MockClient, Task, combine, exact_match, token_count)
from bpto.ops import Variants


class Names(BaseModel):
    name: str


RECORDS = [
    {"inputs": {"text": "There once was a man from Nantucket. His name was Charles."}, "answer": "Charles"},
    {"inputs": {"text": "Alice went to the market."}, "answer": "Alice"},
    {"inputs": {"text": "Nobody was there."}, "answer": ""},
    {"inputs": {"text": "Bob and his dog."}, "answer": "Bob"},
]


def handler(prompt, cfg, schema):
    """Mock model. Expansion: return n variants. Evaluation: extract a capitalised name;
    prompts containing 'ONLY' get the 'Nobody' case right, mimicking a better prompt."""
    if schema is Variants:
        n = int(re.search(r"Return (\d+) distinct", prompt).group(1))
        m = re.search(r"<prompt>\n(.*?)\n</prompt>", prompt, re.S)
        base = m.group(1)
        tag = "ONLY" if "succinct" in prompt else "v"
        return Variants(prompts=[f"{tag}{i} {base}" for i in range(n)])
    text = prompt.split(":", 1)[-1]
    known = {"Charles", "Alice"} | ({"Bob"} if "ONLY" in prompt else set())
    names = [w for w in re.findall(r"\b[A-Z][a-z]+\b", text) if w in known]
    return Names(name=names[0] if names else "")


@pytest.fixture
def client():
    return MockClient(handler, max_concurrency=4)


@pytest.fixture
def task(client):
    return Task(
        root="Extract the name of the person from the following text:\n{text}",
        description="extracts the name of the person mentioned in a passage",
        dataset=Dataset.from_records(RECORDS),
        schema=Names,
        scorer=combine(exact_match(field="name"), token_count()),
        objective=LinearObjective(accuracy=1.0, prompt_tokens=-0.001),
        client=client,
    )
