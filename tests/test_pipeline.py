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
