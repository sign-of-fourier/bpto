"""Task = everything problem-specific: root prompt, data, schema, scorer, objective, client."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from .data import Dataset
from .llm import ModelClient, ModelConfig
from .prompt import Program, Prompt, as_prompt
from .scoring import Objective, Scorer


@dataclass
class Task:
    root: Prompt | Program | str | dict[str, str]
    dataset: Dataset
    scorer: Scorer
    objective: Objective
    client: ModelClient
    description: str | dict[str, str] = ""  # "extracts the names of people from a passage" - used by guided ops; per module for a Program
    schema: type[BaseModel] | None = None # structured output for evaluation calls
    config: ModelConfig | None = None     # default per-evaluation model config (None -> client default)
    expander_client: ModelClient | None = None  # model used to rewrite prompts; defaults to `client`
    expander_config: ModelConfig | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.root = as_prompt(self.root)
        if self.expander_client is None:
            self.expander_client = self.client
