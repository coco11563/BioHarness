# How the framework-eval QAClient maps to V14CascadeClient

`framework-eval` calls every method through the
`framework_eval.plugins.QAClient` protocol:

```python
class QAClient(Protocol):
    name: str
    async def generate(self, item: Item) -> Prediction: ...
    async def aclose(self) -> None: ...
```

`V14CascadeClient` implements that protocol. Its `generate` method is
the orchestration point for the seven-stage cascade described in the
`{framework}^χ` paper. The mapping looks like:

```text
Item                     ->  V14CascadeClient.generate
                              ├─ _retrieve              # dense + dual rerank
                              ├─ _fast_path             # constrained gen + logprob
                              ├─ _answer_is_grounded    # optional grounded gate
                              │   (skipped on yesno; agent over-analyses)
                              ├─ _agent                 # escalation only
                              └─ _rejudge               # constrained re-judgment
Prediction(answer=..., extras={stage, logprob, ...})
```

Each underscore-prefixed method is a stable hook. To plug in your own
agent (e.g. the full `BiomedicalRLMPipeline` that runs the REPL +
multi-tool dispatch from your infrastructure repository), subclass
`V14CascadeClient` and override `_agent`:

```python
from framework_chi.cascade.client import V14CascadeClient, AgentOutcome

class MyChi(V14CascadeClient):
    async def _agent(self, item, ctx, fast_path):
        # Call your in-house pipeline here
        text = await my_repl_pipeline.run(item, ctx.passages)
        return AgentOutcome(
            answer=text, response_text=text,
            iterations=self.options.max_agent_iterations,
            tool_calls=[],
        )
```

Register the subclass via your own pyproject entry point or use
`framework-eval run --method my_pkg.my_module:MyChi`.

## Stages in detail

### `_retrieve`
Embed the question, search Qdrant `paper-full`, optionally rerank with
the cross-encoder. Returns `RetrievalContext` with at most
`options.rerank_top_k` passages.

### `_fast_path`
Build the constrained-gen prompt (system + per-question-type
instruction + question + options + retrieved passages), call the LLM
with `max_tokens=4` and `temperature=0`, parse the routing logprob from
the response, run `extract_answer` to normalise.

### `_answer_is_grounded`
Substring-match the fast-path answer against retrieved passage text.
The `enable_grounded_gate` flag promotes this to a routing condition.

### `_agent`
Escalation. Default: longer LLM call with the same context.

### `_rejudge`
Constrained re-judgment. Wraps the agent's free-form answer in a
`max_tokens=4` call that emits the benchmark-compliant string.

### Yesno special case
The cascade always returns the fast-path answer for `yesno` items: the
paper §5.2 shows the agent over-analyses these and introduces a `no` bias.
