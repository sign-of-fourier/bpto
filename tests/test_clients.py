import pytest
import json

import httpx
from pydantic import BaseModel

from bpto import Dataset, LinearObjective, MockClient, OpenAICompatibleClient, Task, Tree, evaluate, llm_judge, select
from bpto.scoring import JudgeVerdict


class Out(BaseModel):
    answer: str


async def test_openai_compatible_client_parses_and_retries():
    seen = {"n": 0}

    def handler(req: httpx.Request):
        seen["n"] += 1
        body = json.loads(req.content)
        if seen["n"] == 1:
            return httpx.Response(500, text="boom")
        assert body["response_format"]["type"] == "json_schema"
        assert req.headers["authorization"] == "Bearer k"
        return httpx.Response(200, json={
            "model": "m", "choices": [{"message": {"content": '{"answer": "42"}'}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3}})

    c = OpenAICompatibleClient("m", base_url="http://x/v1", api_key="k", transport=httpx.MockTransport(handler))
    comp = await c.complete("q", schema=Out)
    assert comp.parsed == Out(answer="42") and comp.input_tokens == 7 and seen["n"] == 2
    comp2 = await c.complete("q", schema=Out)  # cached
    assert comp2.cached and seen["n"] == 2


async def test_llm_judge_uses_separate_client():
    def answerer(prompt, cfg, schema):
        return Out(answer="Paris is the capital.")

    def judge(prompt, cfg, schema):
        assert schema is JudgeVerdict and "Reference answer" in prompt
        return JudgeVerdict(score=0.9 if "Paris" in prompt else 0.0, reason="ok")

    task = Task(root="Q: {question}", dataset=Dataset.from_records([
        {"inputs": {"question": "capital of France?"}, "answer": "Paris"}]),
        schema=Out, scorer=llm_judge("Is the answer semantically equivalent?", client=MockClient(judge)),
        objective=LinearObjective(judge=1.0), client=MockClient(answerer))
    tree = Tree(task)
    await tree.apply(evaluate(), select=select.root)
    assert tree.root.evaluation.metrics == {"judge": 0.9}


async def test_bedrock_client_converse_shape():
    from bpto.llm.bedrock import BedrockClient, parse_json_reply

    class FakeRT:
        def __init__(self): self.calls = []
        def converse(self, **kw):
            self.calls.append(kw)
            return {"output": {"message": {"content": [{"text": '```json\n{"answer": "42"}\n```'}]}},
                    "usage": {"inputTokens": 5, "outputTokens": 2}, "stopReason": "end_turn"}

    c = BedrockClient.__new__(BedrockClient)
    ModelClientInit = type(c).__mro__[1].__init__
    ModelClientInit(c, default_config=__import__("bpto").ModelConfig(model="amazon.nova-micro-v1:0"))
    c._rt = FakeRT()
    comp = await c.complete("q", schema=Out)
    assert comp.parsed == Out(answer="42") and comp.input_tokens == 5
    sent = c._rt.calls[0]
    assert sent["modelId"] == "amazon.nova-micro-v1:0" and "JSON object" in sent["messages"][0]["content"][0]["text"]
    assert parse_json_reply('noise {"answer": "x"} trailing', Out) == Out(answer="x")


async def test_azure_embedder_url_and_headers():
    from bpto.bo import AzureOpenAIEmbedder
    seen = {}

    def handler(req):
        seen["url"], seen["key"] = str(req.url), req.headers.get("api-key")
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0]}]})

    e = AzureOpenAIEmbedder("emb-3", endpoint="https://x.openai.azure.com", api_key="k", transport=httpx.MockTransport(handler))
    assert await e.embed(["a"]) == [[1.0]]
    assert seen["url"] == "https://x.openai.azure.com/openai/deployments/emb-3/embeddings?api-version=2024-10-21"
    assert seen["key"] == "k"


async def test_budget_counts_in_flight_calls():
    import asyncio
    from bpto import Budget, BudgetExceeded, MockClient
    client = MockClient(lambda p, c, s: "ok", delay=0.01, max_concurrency=50, budget=Budget(max_calls=5))
    results = await asyncio.gather(*(client.complete(f"p{i}") for i in range(20)), return_exceptions=True)
    ok = [r for r in results if not isinstance(r, Exception)]
    assert len(ok) == 5 and client.usage.calls == 5
    assert all(isinstance(r, BudgetExceeded) for r in results if isinstance(r, Exception))


async def test_shared_dollar_budget_across_clients():
    import asyncio
    from bpto import Budget, BudgetExceeded, MockClient, ModelConfig
    from bpto.llm.mock import MockClient as MC
    budget = Budget(max_usd=0.001, prices={"mock-a": (1.0, 1.0), "mock-b": (100.0, 100.0)})  # $ per 1M tokens
    a = MC(lambda p, c, s: "x " * 50, default_config=ModelConfig(model="mock-a"), budget=budget)
    b = MC(lambda p, c, s: "x " * 50, default_config=ModelConfig(model="mock-b"), budget=budget)
    for i in range(5):
        await a.complete(f"a{i}")
    assert 0 < budget.spent_usd < 0.001 and budget.spent.calls == 5
    with pytest.raises(BudgetExceeded):
        for i in range(50):
            await b.complete(f"b{i}")
    assert budget.spent_usd >= 0.001 and a.usage.calls == 5 and b.usage.calls < 50


def test_price_lookup_handles_region_prefix():
    from bpto.llm import price_for
    assert price_for("us.amazon.nova-micro-v1:0") == price_for("amazon.nova-micro-v1:0")
    with pytest.raises(KeyError):
        price_for("nobody-knows-this-model")


async def test_budget_parent_meter():
    from bpto import Budget, BudgetExceeded, ModelConfig
    from bpto.llm.mock import MockClient as MC
    shared = Budget(max_usd=1.0, prices={"m": (1.0, 1.0)})
    local = Budget(max_calls=2, parent=shared)
    c = MC(lambda p, c, s: "hi", default_config=ModelConfig(model="m"), budget=local)
    await c.complete("a"); await c.complete("b")
    with pytest.raises(BudgetExceeded):
        await c.complete("c")
    assert shared.spent.calls == 2 and shared.spent_usd > 0 and local.spent.calls == 2


def test_bedrock_parse_accepts_raw_newlines_in_strings():
    from pydantic import BaseModel
    from bpto.llm.bedrock import parse_json_reply

    class V(BaseModel):
        prompts: list[str]
    out = parse_json_reply('Sure:\n{"prompts": ["line one\nline two {x}", "b"]}', V)
    assert out.prompts[0] == "line one\nline two {x}"


def _jev_reply(body):
    answers = {}
    for name, q in body["questions"].items():
        if q["type"] == "noul":
            answers[name] = {"type": "noul", "noul": 0.9}
        elif q["type"] == "score":
            answers[name] = {"type": "score", "score": 1.5, "confidence": 0.8,
                             "legend": {str(i): c for i, c in enumerate(q["criteria"])},
                             "probabilities": {"1": 0.5, "2": 0.5}}
        else:
            labels = list(q["criteria"])
            answers[name] = {"type": "choice", "choice": labels[0], "confidence": 0.7,
                             "probabilities": {labels[0]: 0.7, labels[1]: 0.3}}
    return {"model": "typesafe/jev-1.13-20260917", "usage": {"input_tokens": 40, "output_tokens": 9}, "answers": answers}


async def test_jev_client_request_shape_retry_and_cache():
    from bpto import JevClient
    from bpto.llm.jev import noul
    seen = {"n": 0, "bodies": []}

    def handler(req: httpx.Request):
        seen["n"] += 1
        if seen["n"] == 1:
            return httpx.Response(429, text="slow down")
        assert req.url.path == "/api/v1/systemone" and req.headers["authorization"] == "Bearer k"
        body = json.loads(req.content)
        seen["bodies"].append(body)
        return httpx.Response(200, json=_jev_reply(body))

    c = JevClient("jev-1.13", base_url="https://openrouter.ai/api", api_key="k", transport=httpx.MockTransport(handler))
    qs = {"same": noul("Same person?")}
    a = await c.ask({"a": "Ike", "b": "Dwight D. Eisenhower"}, qs)
    assert a["same"]["noul"] == 0.9 and seen["n"] == 2
    assert seen["bodies"][0] == {"state": {"a": "Ike", "b": "Dwight D. Eisenhower"}, "model": "jev-1.13",
                                 "questions": {"same": {"type": "noul", "instructions": "Same person?"}}}
    assert c.usage.input_tokens == 40 and c.usage.output_tokens == 9
    assert await c.ask({"a": "Ike", "b": "Dwight D. Eisenhower"}, qs) == a and seen["n"] == 2  # cached
    await c.ask({"b": "Dwight D. Eisenhower", "a": "Ike"}, qs)  # key order changes Jev's answer: sent as given, not cached
    assert seen["n"] == 3 and list(seen["bodies"][-1]["state"]) == ["b", "a"]


async def test_jev_client_rejects_text_prompts_and_bad_requests():
    from bpto import JevClient
    from bpto.llm.jev import choice, noul

    sent = []

    def handler(req):
        sent.append(req)
        return httpx.Response(400, json={"error": "bad question"})

    c = JevClient(transport=httpx.MockTransport(handler), max_retries=3)
    with pytest.raises(ValueError, match="use JevClient.ask"):
        await c.complete("free text")
    with pytest.raises(ValueError, match="noul, choice or score"):
        await c.ask("s", {"q": {"type": "essay"}})
    with pytest.raises(ValueError, match="1-255"):
        choice("pick", {})
    with pytest.raises(httpx.HTTPStatusError):  # 4xx other than 408/409/429 is not retried
        await c.ask("s", {"q": noul("?")})
    assert len(sent) == 1


async def test_jev_judge_metrics_per_question():
    from bpto import JevClient, jev_judge
    from bpto.llm.jev import choice, noul, score
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json=_jev_reply(seen["body"]))

    jev = JevClient(transport=httpx.MockTransport(handler))

    def answerer(prompt, cfg, schema):
        return Out(answer="Paris")

    scorer = jev_judge({"correct": noul("Same answer as the reference?"),
                        "grade": score("How close?", ["wrong", "partial", "right, verbose", "exact"]),
                        "kind": choice("What kind of answer?", {"city": None, "country": None})}, client=jev)
    task = Task(root="Q: {question}", dataset=Dataset.from_records([
        {"inputs": {"question": "capital of France?"}, "answer": "Paris"}]),
        schema=Out, scorer=scorer, objective=LinearObjective(correct=1.0), client=MockClient(answerer))
    tree = Tree(task)
    await tree.apply(evaluate(), select=select.root)
    assert tree.root.evaluation.metrics == pytest.approx({"correct": 0.9, "grade": 0.5, "kind.city": 0.7, "kind.country": 0.3})
    assert seen["body"]["state"]["reference_answer"] == "Paris" and seen["body"]["state"]["inputs"] == {"question": "capital of France?"}
