"""A prompt template with named placeholders."""
from __future__ import annotations

import hashlib
import string

from pydantic import BaseModel, computed_field


class Prompt(BaseModel, frozen=True):
    template: str

    @computed_field
    @property
    def placeholders(self) -> tuple[str, ...]:
        names = []
        for _, field, _, _ in string.Formatter().parse(self.template):
            if field is None:
                continue
            if field == "" or field.isdigit():  # positional `{}` / `{0}`: unrenderable with named inputs
                raise ValueError(f"positional placeholder {{{field}}} in template")
            if field not in names:
                names.append(field)
        return tuple(names)

    def render(self, **inputs) -> str:
        return self.template.format(**inputs)

    @property
    def hash(self) -> str:
        return hashlib.sha256(self.template.encode()).hexdigest()[:16]

    def __str__(self) -> str:
        return self.template


class Program(BaseModel, frozen=True):
    """Several named prompts searched together (a multi-module pipeline). The `entry` module is what
    `evaluate` renders and sends; the scorer runs the other modules (it receives the Program) and records their
    traces. Delegating `template` / `placeholders` / `render` to the entry module keeps every single-prompt
    code path working on a program node; module-aware code (`LLMExpander(module=)`, `BOSelector`) looks at
    `modules` directly."""
    modules: dict[str, Prompt]
    entry: str

    @property
    def template(self) -> str:
        return self.modules[self.entry].template

    @property
    def placeholders(self) -> tuple[str, ...]:
        return self.modules[self.entry].placeholders

    def render(self, **inputs) -> str:
        return self.modules[self.entry].render(**inputs)

    @property
    def hash(self) -> str:
        h = hashlib.sha256()
        for name, p in self.modules.items():
            h.update(f"{name}\0{p.template}\0".encode())
        return h.hexdigest()[:16]

    def with_module(self, name: str, prompt: "Prompt | str") -> "Program":
        if name not in self.modules:
            raise KeyError(name)
        p = prompt if isinstance(prompt, Prompt) else Prompt(template=prompt)
        return Program(modules={**self.modules, name: p}, entry=self.entry)

    def __str__(self) -> str:
        return "\n\n".join(f"[{name}]\n{p.template}" for name, p in self.modules.items())


def as_prompt(p: "Prompt | Program | str | dict[str, str]") -> "Prompt | Program":
    """str -> Prompt; {module: template} -> Program with the first module as entry."""
    if isinstance(p, (Prompt, Program)):
        return p
    if isinstance(p, dict):
        return Program(modules={k: Prompt(template=v) for k, v in p.items()}, entry=next(iter(p)))
    return Prompt(template=p)
