# How the framework-eval QAClient maps to PipelineCascadeClient

> Naming note: the in-tree class is ``PipelineCascadeClient`` and the
> registered entry-point id is ``pipeline``. The research method id
> ``v14-cascade-dual-rerank-grounded`` is not used by this package; it
> still appears in `paper_reproduction/`, whose run scripts use it.

`framework-eval` calls every method through the
`framework_eval.plugins.QAClient` protocol:

```python
class QAClient(Protocol):
    name: str
    async def generate(self, item: Item) -> Prediction: ...
    async def aclose(self) -> None: ...
```

`PipelineCascadeClient` implements that protocol. Its `generate` method is
the orchestration point for this package's re-implementation of the
BioHarness cascade (it does not reproduce the paper's Table 1; see the
README). The mapping looks like:

```text
Item                     ->  PipelineCascadeClient.generate
                              ├─ _retrieve              # dense + dual rerank
                              ├─ _fast_path             # constrained gen + logprob
                              ├─ _answer_is_grounded    # grounded gate (factoid/list)
                              │   (skipped on yesno; agent over-analyses)
                              ├─ _agent                 # escalation only
                              └─ _rejudge               # constrained re-judgment
Prediction(answer=..., extras={stage, logprob, ...})
```

Each underscore-prefixed method is a stable hook. To plug in your own
agent (e.g. the full `BiomedicalRLMPipeline` that runs the REPL +
multi-tool dispatch, in `paper_reproduction/src/rlm/pipeline.py`), subclass
`PipelineCascadeClient` and override `_agent`:

```python
from bioharness.cascade.client import PipelineCascadeClient, AgentOutcome

class MyPipeline(PipelineCascadeClient):
    async def _agent(self, item, ctx, fast_path):
        # Call your in-house pipeline here
        text = await my_repl_pipeline.run(item, ctx.passages)
        return AgentOutcome(
            answer=text, response_text=text,
            iterations=1,
            tool_calls=[],
        )
```

Register the subclass via your own pyproject entry point or use
`framework-eval run --method my_pkg.my_module:MyPipeline`.

## Stages in detail

### `_retrieve`
Embed the question (plus a pseudo-answer and, for yes/no items, a
negative-evidence query), search Qdrant `paper-full`, and rerank the
merged pool by the original question with the cross-encoder (title plus
text per passage). Returns `RetrievalContext` with at most `RERANK_TOP_K`
(20) passages.

### `_fast_path`
Build the constrained-gen prompt (system + per-question-type
instruction + question + options + retrieved passages), call the LLM
with the per-type `MAX_TOKENS` budget (`constrained.py`: yesno 4, mcq 4,
mcq_multi 16, factoid 32, list 128, summary 256, expression 300 via
`atlas.EXPRESSION_MAX_TOKENS`) and
`temperature=0.1`, parse the routing logprob from the response, and run
`extract_constrained_answer` to normalise.

### `_answer_is_grounded`
Substring-match the fast-path answer against retrieved passage text.
An ungrounded factoid or list answer always escalates.

### `_agent`
Escalation. Default: a single LLM call over the stage-1 passages plus
pre-called gene / genomics evidence. When `BIOHARNESS_PRODUCTION_SRC`
points at the research code, the hook delegates to its REPL agent.

### `_rejudge`
Constrained re-judgment. Wraps the agent's free-form answer (plus the
tool evidence and the original passages) in a constrained call with the
same per-type `MAX_TOKENS` budget and `temperature=0.1`, which emits the
benchmark-compliant string.

### Yesno and expression special case
The cascade always returns the fast-path answer for `yesno` and
`expression` items. For `yesno`, the agent over-analysed these items and
introduced a `no` bias in the paper's runs; for `expression`, the
fast-path prompt (with optional atlas context) is the complete answer.
