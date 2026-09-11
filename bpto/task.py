"""Task = everything problem-specific: root prompt, data, schema, scorer, objective, client."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from .data import Dataset
from .llm import ModelClient, ModelConfig
from .prompt import Prompt
from .scoring import Objective, Scorer


@dataclass
class Task:
    root: Prompt | str
    dataset: Dataset
    scorer: Scorer
    objective: Objective
    client: ModelClient
    description: str = ""                 # "extracts the names of people from a passage" - used by guided ops
    schema: type[BaseModel] | None = None # structured output for evaluation calls
    config: ModelConfig | None = None     # default per-evaluation model config (None -> client default)
    expander_client: ModelClient | None = None  # model used to rewrite prompts; defaults to `client`
    expander_config: ModelConfig | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.root, str):
            self.root = Prompt(template=self.root)
        if self.expander_client is None:
            self.expander_client = self.client
