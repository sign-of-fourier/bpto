import pytest
from bpto import DescendantValue, ModelConfig, NodeState, Pipeline, SubtreeValue, Tree, evaluate, guided, random, select
from bpto.bo import GPR, EI, BOSelector, HashEmbedder


async def test_pipeline_is_one_expansion(task):
    tree = Tree(task)
    pipe = Pipeline([random(2), guided("make it more succinct", 2), evaluate()], name="rg")
    created = await tree.apply(pipe, select=select.root)
    assert tree.root.state == NodeState.EXPANDED
    assert len(created) == 2 + 4 and len(tree) == 7
    assert all(n.evaluated for n in created if n.depth == 2) and not any(n.evaluated for n in created if n.depth == 1)
    assert all(n.origin.params["pipeline"] == "rg" and n.origin.params["pipeline_root"] == tree.root.id for n in created)
    assert {n.origin.op for n in created} == {"random", "guided"}   # per-op origin kept
    # subtree value sees both generations; fixed-generation value sees only one
    assert SubtreeValue(max)(tree.root, tree) == max(n.score for n in created if n.evaluated)
    assert DescendantValue(1, max)(tree.root, tree) is None and DescendantValue(2, max)(tree.root, tree) is not None
    assert SubtreeValue(max)(created[-1], tree) is None


async def test_pipeline_only_attribution(task):
    tree = Tree(task)
    await tree.apply(Pipeline([random(2), evaluate()], name="p1"), select=select.root)
    a = tree.leaves[0]
    await tree.apply(Pipeline([random(1), evaluate()], name="p2"), select=lambda t: [a])
    # a's subtree under p2 only; root's pipeline_only value excludes p2's grandchild
    assert SubtreeValue(max, pipeline_only=True)(a, tree) == tree.child_nodes(a)[0].score
    assert SubtreeValue(max, pipeline_only=True)(tree.root, tree) == max(n.score for n in tree.child_nodes(tree.root))


async def test_bo_with_pipeline_expansion(task):
    tree = Tree(task)
    pipe = Pipeline([random(2), guided("make it more succinct", 1), evaluate()])
    await tree.apply(pipe, select=select.root)
    bo = BOSelector(HashEmbedder(64), GPR(), EI(), value=SubtreeValue(max, include_self=True))
    picked = await tree.apply(pipe, select=bo.top(k=1, among=select.unexpanded))
    assert len(picked) == 4 and bo.last_fit["n_train"] >= 1


async def test_expander_config_override(task):
    seen = []
    orig = task.expander_client._complete

    async def spy(prompt, cfg, schema):
        seen.append(cfg); return await orig(prompt, cfg, schema)
    task.expander_client._complete = spy
    tree = Tree(task)
    await tree.apply(guided("x", n=1, config=ModelConfig(temperature=1.0)), select=select.root)
    assert seen[0].temperature == 1.0
    assert tree.leaves[0].origin.params["config"] == {"temperature": 1.0}


async def test_expander_drops_invented_placeholders_and_strips_wrappers(task):
    from bpto import MockClient, Tree, select
    from bpto.ops import Variants, random
    def handler(prompt, cfg, schema):
        return Variants(prompts=["<prompt>\nGood {text}\n</prompt>", "```\n<prompt>Also good {text}</prompt>\n```",
                                 "Bad {text} {invented: x}", "Missing placeholder", "Unbalanced {text"])
    task.client = task.expander_client = MockClient(handler)
    tree = Tree(task)
    kids = await tree.apply(random(n=5), select.root)
    assert [k.prompt.template for k in kids] == ["Good {text}", "Also good {text}"]
    assert kids[0].origin is not kids[1].origin   # per-child origin


async def test_malformed_expansion_yields_no_children(task):
    from bpto import Budget, BudgetExceeded, MockClient, Tree, select
    from bpto.ops import random
    task.client = task.expander_client = MockClient(lambda p, c, s: "not json at all")
    tree = Tree(task)
    kids = await tree.apply(random(n=2), select.root)
    assert kids == [] and tree.root.state == "expanded"
    task.expander_client = MockClient(lambda p, c, s: "x", budget=Budget(max_calls=0))
    with pytest.raises(BudgetExceeded):
        await tree.apply(random(n=2), select.root)


def test_positional_placeholders_are_rejected():
    """A model-written template with a bare `{}` must fail at construction, not at render (IndexError mid-run)."""
    import pytest
    from bpto import Prompt
    for t in ["Names: {}\n{text}", "Names {0} {text}"]:
        with pytest.raises(ValueError):
            Prompt(template=t).placeholders
    assert Prompt(template="a {{b}} {text}").placeholders == ("text",)


def test_expander_splits_glued_templates_and_rejects_repeated_placeholders():
    import asyncio
    from bpto import Dataset, MockClient, Task, Tree, exact_match, select
    from bpto.ops import LLMExpander, Variants

    def handler(prompt, cfg, schema):
        return Variants(prompts=["A {x}\n<prompt>\nB {x}\n</prompt>", "C {x} and again {x}", "<prompt>D {x}</prompt>"])
    task = Task(root="root {x}", dataset=Dataset.from_records([{"inputs": {"x": 1}, "answer": 1}]), scorer=exact_match(),
                objective=lambda m, c: (0.0, True), client=MockClient(handler))
    tree = Tree(task)
    kids = asyncio.run(tree.apply(LLMExpander(n=4), select.root))
    assert [k.prompt.template for k in kids] == ["A {x}", "B {x}", "D {x}"]
