"""Flat node store with parent/child adjacency and ancestry navigation."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable, Iterable, TYPE_CHECKING

from pydantic import BaseModel, Field

from .llm import ModelConfig
from .metrics import Metrics
from .prompt import Prompt

if TYPE_CHECKING:
    from .task import Task


class NodeState(StrEnum):
    PROPOSED = "proposed"
    EVALUATED = "evaluated"
    EXPANDED = "expanded"


class Origin(BaseModel):
    op: str
    params: dict[str, Any] = Field(default_factory=dict)


class ExampleResult(BaseModel):
    example_id: str
    output: str
    parsed: Any = None
    metrics: Metrics
    error: str | None = None


class Evaluation(BaseModel):
    per_example: list[ExampleResult]
    metrics: Metrics
    metrics_std: Metrics
    n: int
    score: float
    feasible: bool
    dataset_ids: list[str]


class Node(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:10])
    parent_id: str | None = None
    depth: int = 0
    prompt: Prompt
    config: ModelConfig | None = None  # per-node override of the task's default
    origin: Origin = Origin(op="root")
    state: NodeState = NodeState.PROPOSED
    evaluation: Evaluation | None = None
    embedding: list[float] | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def score(self) -> float | None:
        return self.evaluation.score if self.evaluation else None

    @property
    def evaluated(self) -> bool:
        return self.evaluation is not None


class Tree:
    def __init__(self, task: "Task", root: Prompt | str | None = None):
        self.task = task
        self.nodes: dict[str, Node] = {}
        self.children: dict[str, list[str]] = {}
        root = root if root is not None else task.root
        self.root = self._add(Node(prompt=self._as_prompt(root)))
        self.listeners: list[Callable[[str, Node], None]] = []
        self.meta: dict[str, Any] = {}  # run bookkeeping (round/step), persisted

    # ---- construction --------------------------------------------------------------
    @staticmethod
    def _as_prompt(p: Prompt | str) -> Prompt:
        return p if isinstance(p, Prompt) else Prompt(template=p)

    def _add(self, node: Node) -> Node:
        self.nodes[node.id] = node
        self.children.setdefault(node.id, [])
        if node.parent_id is not None:
            self.children[node.parent_id].append(node.id)
        return node

    def add_child(self, parent: Node, prompt: Prompt | str, origin: Origin, config: ModelConfig | None = None) -> Node:
        node = Node(parent_id=parent.id, depth=parent.depth + 1, prompt=self._as_prompt(prompt),
                    origin=origin, config=config if config is not None else parent.config)
        self._add(node)
        self._emit("proposed", node)
        return node

    def _emit(self, event: str, node: Node) -> None:
        for fn in self.listeners:
            fn(event, node)

    # ---- navigation ----------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.nodes)

    def __iter__(self):
        return iter(self.nodes.values())

    def get(self, node_id: str) -> Node:
        return self.nodes[node_id]

    def parent(self, node: Node) -> Node | None:
        return self.nodes[node.parent_id] if node.parent_id else None

    def ancestors(self, node: Node) -> list[Node]:
        """Nearest first, root last."""
        out, cur = [], self.parent(node)
        while cur is not None:
            out.append(cur)
            cur = self.parent(cur)
        return out

    def ancestor(self, node: Node, k: int) -> Node | None:
        """k-th ancestor (0 = node itself, 1 = parent, ...)."""
        cur: Node | None = node
        for _ in range(k):
            if cur is None:
                return None
            cur = self.parent(cur)
        return cur

    def child_nodes(self, node: Node) -> list[Node]:
        return [self.nodes[c] for c in self.children[node.id]]

    def descendants(self, node: Node, generations: int | None = None) -> list[Node]:
        """generations=k -> exactly the k-th generation below `node`; None -> whole subtree."""
        frontier, out = [node], []
        g = 0
        while frontier:
            if generations is not None and g == generations:
                return frontier if g > 0 else []
            nxt = [c for n in frontier for c in self.child_nodes(n)]
            if generations is None:
                out.extend(nxt)
            frontier, g = nxt, g + 1
        return out if generations is None else []

    @property
    def leaves(self) -> list[Node]:
        return [n for n in self if not self.children[n.id]]

    @property
    def max_depth(self) -> int:
        return max(n.depth for n in self)

    # ---- results -------------------------------------------------------------------
    def evaluated_nodes(self) -> list[Node]:
        return [n for n in self if n.evaluated]

    def best(self, k: int = 1, feasible_only: bool = True) -> Node | list[Node] | None:
        cands = [n for n in self.evaluated_nodes() if (n.evaluation.feasible or not feasible_only)]
        cands.sort(key=lambda n: n.score, reverse=True)
        if k == 1:
            return cands[0] if cands else None
        return cands[:k]

    def pareto(self, maximize: dict[str, bool]) -> list[Node]:
        from .scoring import pareto_front
        pts = {n.id: n.evaluation.metrics for n in self.evaluated_nodes()}
        return [self.nodes[i] for i in pareto_front(pts, maximize)]

    # ---- the single mutation entry point -------------------------------------------
    async def apply(self, op, select=None, chunk: int = 32, checkpoint: str | Path | None = None) -> list[Node]:
        """Run `op` over the nodes chosen by `select` (default: leaves), `chunk` nodes at a time.
        With `checkpoint`, the tree is saved after every chunk."""
        import inspect
        from .select import leaves as _leaves
        targets = (select or _leaves)(self)
        if inspect.isawaitable(targets):  # selectors may be async (e.g. BO ranking)
            targets = await targets
        after = (lambda: self.save(checkpoint)) if checkpoint else None
        return await op.apply(self, targets, chunk=chunk, after_chunk=after)

    # ---- persistence ---------------------------------------------------------------
    def to_dict(self) -> dict:
        return {"root": self.root.id, "meta": self.meta,
                "nodes": [json.loads(n.model_dump_json()) for n in self.nodes.values()]}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        tmp = path.with_suffix(path.suffix + ".tmp")  # atomic: never leave a half-written checkpoint
        tmp.write_text(json.dumps(self.to_dict(), indent=1, default=str))
        tmp.replace(path)

    @classmethod
    def load(cls, path: str | Path, task: "Task") -> "Tree":
        data = json.loads(Path(path).read_text())
        tree = cls.__new__(cls)
        tree.task, tree.nodes, tree.children, tree.listeners = task, {}, {}, []
        tree.meta = data.get("meta", {})
        for rec in data["nodes"]:
            tree._add(Node.model_validate(rec))
        tree.root = tree.nodes[data["root"]]
        return tree

    def __repr__(self) -> str:
        return f"Tree(nodes={len(self)}, depth={self.max_depth}, evaluated={len(self.evaluated_nodes())})"
