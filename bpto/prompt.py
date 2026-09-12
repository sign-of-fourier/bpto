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
