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
