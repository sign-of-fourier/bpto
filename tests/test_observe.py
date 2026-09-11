import io

from bpto import EventLog, Progress, Tree, evaluate, guided, lineage, plot_tree, random, select, tree_dot, tree_text


async def test_event_log_and_progress(task, tmp_path):
    tree = Tree(task)
    log = EventLog(tmp_path / "events.jsonl", tree)
    buf = io.StringIO()
    Progress(tree, stream=buf)
    await tree.apply(random(n=2))
    await tree.apply(evaluate(), select=select.unevaluated)
    log.close()
    events = EventLog.read(tmp_path / "events.jsonl")
    kinds = [e["event"] for e in events]
    assert kinds.count("proposed") == 2 and kinds.count("expanded") == 1 and kinds.count("evaluated") == 3
    assert events[-1]["node"]["score"] is not None and events[-1]["usage"]["calls"] > 0
    out = buf.getvalue()
    assert "expanded" in out and out.count("eval #") == 3


async def test_text_dot_plot_lineage(task, tmp_path):
    tree = Tree(task)
    await tree.apply(random(n=2))
    await tree.apply(guided("shorter", n=1))
    await tree.apply(evaluate(), select=select.unevaluated)
    txt = tree_text(tree)
    assert txt.count("\n") == 4 and "└─" in txt and "[guided]" in txt
    dot = tree_dot(tree)
    assert dot.count("->") == 4 and "fillcolor" in dot
    assert lineage(tree, tree.best()).count("\n") == tree.best().depth
    p = plot_tree(tree, tmp_path / "tree.png")
    assert p.exists() and p.stat().st_size > 1000
