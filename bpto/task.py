"""Task = everything problem-specific: root prompt, data, schema, scorer, objective, client."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel

from .data import Dataset, Example
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
    schema: type[BaseModel] | dict[str, type[BaseModel] | None] | None = None  # structured output for evaluation calls; per module for a Program
    config: ModelConfig | dict[str, ModelConfig | None] | None = None  # default per-evaluation model config (None -> client default); per module for a Program
    expander_client: ModelClient | None = None  # model used to rewrite prompts; defaults to `client`
    expander_config: ModelConfig | None = None
    # graph programs: code run before a module is rendered - `adapter(example, state) -> dict` of extra render inputs
    # for that call only (e.g. turn the titles a selector returned into the paragraphs the answerer needs); the
    # persisted state stays dataset inputs + mapped fields. Keyed by target module.
    adapters: dict[str, Callable[[Example, dict[str, Any]], dict[str, Any]]] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.root = as_prompt(self.root)
        if self.expander_client is None:
            self.expander_client = self.client

    def schema_for(self, module: str | None = None) -> type[BaseModel] | None:
        return self.schema.get(module) if isinstance(self.schema, dict) else self.schema

    def config_for(self, module: str | None = None) -> ModelConfig | None:
        return self.config.get(module) if isinstance(self.config, dict) else self.config
