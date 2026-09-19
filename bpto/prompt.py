"""A prompt template with named placeholders."""
from __future__ import annotations

import hashlib
import string

from pydantic import BaseModel, Field, computed_field


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


class Edge(BaseModel, frozen=True):
    """One outgoing edge of a module. `mapping` says which fields of the source module's parsed output feed which
    placeholders of the target (`{placeholder: field}`; the field `$text` is the raw completion text). A module with
    several edges is orchestrating: its output schema carries `next: enum[edge names]` and the executor follows the
    edge whose `name` matches; when `next` is missing or unknown the `default` edge (else the first) is taken and
    `parse_fail.<module>` is recorded."""
    to: str
    name: str = ""
    mapping: dict[str, str] = Field(default_factory=dict)
    default: bool = False


class Program(BaseModel, frozen=True):
    """Several named prompts searched together (a multi-module pipeline). Delegating `template` / `placeholders` /
    `render` to the entry module keeps every single-prompt code path working on a program node; module-aware code
    (`LLMExpander(module=)`, `BOSelector`) looks at `modules` directly.

    With `edges` the program is a graph and `evaluate` runs it (`bpto.executor`): start at `entry`, follow edges,
    stop at a module with no outgoing edge (a terminal) or at `max_steps` module calls / `max_visits` calls of one
    module. Without `edges` only the entry module is run and the scorer runs the rest (the pre-executor convention).
    Every module can read the example's own inputs; mapped fields accumulate along the path."""
    modules: dict[str, Prompt]
    entry: str
    edges: dict[str, list[Edge]] | None = None  # module -> outgoing edges; absent/empty = terminal
    max_steps: int = 8                           # per example, module calls (only with edges)
    max_visits: int | None = None                # per example, calls of one module (only with edges)

    @property
    def is_graph(self) -> bool:
        return bool(self.edges)

    def outgoing(self, module: str) -> list[Edge]:
        return list((self.edges or {}).get(module) or [])

    def is_orchestrating(self, module: str) -> bool:
        return len(self.outgoing(module)) > 1

    @property
    def terminals(self) -> list[str]:
        return [m for m in self.modules if not self.outgoing(m)]

    def graph_errors(self) -> list[str]:
        """Static checks; empty means well-formed. Schema checks (`next` enum) need the task and live in the executor."""
        errs: list[str] = []
        if self.entry not in self.modules:
            errs.append(f"entry {self.entry!r} is not a module")
        for src, edges in (self.edges or {}).items():
            if src not in self.modules:
                errs.append(f"edges from unknown module {src!r}")
            names = [e.name for e in edges]
            if len(edges) > 1 and (len(set(names)) != len(names) or "" in names):
                errs.append(f"module {src!r}: several edges need distinct non-empty names (got {names})")
            if sum(e.default for e in edges) > 1:
                errs.append(f"module {src!r}: more than one default edge")
            for e in edges:
                if e.to not in self.modules:
                    errs.append(f"edge {src!r} -> unknown module {e.to!r}")
        if not errs and self.edges:
            seen, todo = set(), [self.entry]
            while todo:
                m = todo.pop()
                if m in seen:
                    continue
                seen.add(m)
                todo.extend(e.to for e in self.outgoing(m))
            if not any(not self.outgoing(m) for m in seen):
                errs.append("no terminal reachable from entry: every path is a cycle (the step cap is the only stop)")
            unreachable = set(self.modules) - seen
            if unreachable:
                errs.append(f"modules unreachable from entry: {sorted(unreachable)}")
        return errs

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
        return self.model_copy(update={"modules": {**self.modules, name: p}})

    def __str__(self) -> str:
        return "\n\n".join(f"[{name}]\n{p.template}" for name, p in self.modules.items())


def as_prompt(p: "Prompt | Program | str | dict[str, str]") -> "Prompt | Program":
    """str -> Prompt; {module: template} -> Program with the first module as entry."""
    if isinstance(p, (Prompt, Program)):
        return p
    if isinstance(p, dict):
        return Program(modules={k: Prompt(template=v) for k, v in p.items()}, entry=next(iter(p)))
    return Prompt(template=p)
