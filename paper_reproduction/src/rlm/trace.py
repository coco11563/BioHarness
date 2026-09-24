"""Opt-in per-item trace of every place the agent's evidence gets cut.

The V14 cascade truncates evidence at five points, and none of them was observable:
  stage 1     20 docs x XC_EVIDENCE_DOC_CHARS into the constrained prompt
  REPL        each code block's stdout is shown to the LM cut to 5000 chars
  history     only the last 2 iterations stay verbatim; older ones become ~300-char
              summaries; the whole history is capped at 12000 chars
  rejudge     agent text + REPL dump + original evidence, then context[:8000]
  fallback    optional no-context re-ask
This module records, per item, what was retrieved at each point, what the model
actually saw, and whether a gold-paper id survived each cut. It is enabled only
when XC_TRACE_DIR is set; with it unset the hooks are not installed at all, and
every recorder swallows its own exceptions so a trace bug can never alter a run.

Files: {XC_TRACE_DIR}/{dataset}/{item_id}.json with a flat list of events.
Gold ids come from XC_TRACE_GOLD_MAP: {item_id: {"ids": [pmid|pmcid|doi|uuid, ...]}}.
"""

from __future__ import annotations

import json
import os
import re
import time
from contextvars import ContextVar
from typing import Any

TRACE_DIR: str | None = os.environ.get("XC_TRACE_DIR") or None


def enabled() -> bool:
    return TRACE_DIR is not None


_ctx: ContextVar[dict | None] = ContextVar("_xc_trace", default=None)
_GOLD: dict | None = None

# Tools whose calls are recorded (name -> wrapped in the REPL globals).
TRACED_TOOLS = (
    "search_papers", "search_chunks", "search_section_aware_chunks", "search_fulltext_chunks",
    "hybrid_search", "iterative_search", "keyword_search", "search_by_entities",
    "get_paper_abstracts", "get_paper_sections", "expand_section_evidence", "list_sections",
    "rerank_papers", "rerank_chunks", "judge_evidence", "format_papers", "format_chunks",
    "get_paper_meta", "entity_expand", "llm_query",
)
_RERANK_TOOLS = {"rerank_papers", "rerank_chunks"}
_ID_KEYS = ("pmid", "paper_id", "pmcid", "pmc_id", "doi", "id", "uuid", "chunk_id")
_TXT_KEYS = ("text", "abstract", "content", "chunk_text", "title")


def _gold_map() -> dict:
    global _GOLD
    if _GOLD is None:
        path = os.environ.get("XC_TRACE_GOLD_MAP")
        try:
            _GOLD = json.load(open(path)) if path else {}
        except Exception:
            _GOLD = {}
    return _GOLD


# ----------------------------------------------------------------------------- lifecycle

def begin(item_key: tuple[str, str] | None, extra: dict | None = None):
    """Start a trace for one item; returns a token for end(). No-op when disabled."""
    if not enabled():
        return None
    try:
        dataset, item_id = item_key if item_key else ("unknown", "unknown")
        entry = _gold_map().get(str(item_id)) or {}
        gold = list(entry.get("ids") or [])
        rec = {"dataset": str(dataset), "item_id": str(item_id), "gold_ids": gold,
               "gold_shingles": _shingles(entry.get("key_passage") or ""),
               "t0": time.time(), "events": []}
        if extra:
            rec.update(extra)
        return _ctx.set(rec)
    except Exception:
        return None


def event(kind: str, **fields: Any) -> None:
    rec = _ctx.get()
    if rec is None:
        return
    try:
        fields["kind"] = kind
        fields["t"] = round(time.time() - rec["t0"], 3)
        rec["events"].append(fields)
    except Exception:
        pass


def end(token, **final: Any) -> None:
    rec = _ctx.get()
    if token is not None:
        try:
            _ctx.reset(token)
        except Exception:
            pass
    if rec is None or not enabled():
        return
    try:
        rec.update(final)
        rec["key_passage_shingles"] = len(rec.pop("gold_shingles", ()) or ())
        rec["elapsed_s"] = round(time.time() - rec["t0"], 1)
        d = os.path.join(TRACE_DIR, rec["dataset"])
        os.makedirs(d, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", rec["item_id"])
        with open(os.path.join(d, safe + ".json"), "w") as f:
            json.dump(rec, f, ensure_ascii=False, default=str)
    except Exception:
        pass


# ----------------------------------------------------------------------------- gold helpers

def _gold() -> list[str]:
    rec = _ctx.get()
    return (rec or {}).get("gold_ids") or []


_NORM_RE = re.compile(r"[^a-z0-9]+")


def _norm(text: str) -> str:
    return _NORM_RE.sub(" ", (text or "").lower()).strip()


def _shingles(passage: str, k: int = 8) -> tuple[str, ...]:
    """Word k-grams of the gold key passage, used to spot the passage text itself
    (full-text chunk results carry no pmid, so id matching alone misses them)."""
    words = _norm(passage).split()
    if not words:
        return ()
    if len(words) < k + 4:
        return (" ".join(words),)
    return tuple(" ".join(words[i:i + k]) for i in range(0, len(words) - k + 1))


KEY_PASSAGE = "KEY_PASSAGE"


def gold_in_text(text: str) -> list[str]:
    """Gold ids that occur verbatim (case-insensitive) in a text, plus the marker
    KEY_PASSAGE when any k-gram of the item's key passage occurs in it."""
    try:
        t = (text or "").lower()
        hits = [g for g in _gold() if g.lower() in t]
        sh = (_ctx.get() or {}).get("gold_shingles") or ()
        if sh:
            n = _norm(text)
            if any(s in n for s in sh):
                hits.append(KEY_PASSAGE)
        return hits
    except Exception:
        return []


def doc_ids(doc: Any) -> list[str]:
    """Every id-like field of one result dict (or the string itself)."""
    if isinstance(doc, str):
        return [doc]
    if not isinstance(doc, dict):
        return []
    out = []
    for k in _ID_KEYS:
        v = doc.get(k)
        if v not in (None, ""):
            out.append(str(v))
    return out


def _as_docs(obj: Any) -> list:
    """Normalise the shapes our tools return into a list of result dicts."""
    if isinstance(obj, dict) and obj and all(isinstance(v, dict) for v in obj.values()):
        return [{"pmid": k, **v} for k, v in obj.items()]          # get_paper_abstracts
    if isinstance(obj, (list, tuple)):
        return [d for d in obj if isinstance(d, dict)]
    if isinstance(obj, dict):
        for k in ("results", "papers", "chunks", "docs", "items"):
            if isinstance(obj.get(k), list):
                return [d for d in obj[k] if isinstance(d, dict)]
    return []


def summarize_docs(obj: Any) -> dict:
    """n docs, total text chars, 1-based rank of the first gold doc (None if absent)."""
    try:
        docs = _as_docs(obj)
        gold = {g.lower() for g in _gold()}
        sh = (_ctx.get() or {}).get("gold_shingles") or ()
        chars, rank, hits = 0, None, 0
        for i, d in enumerate(docs, 1):
            texts = [d.get(k) for k in _TXT_KEYS if isinstance(d.get(k), str)]
            chars += sum(len(v) for v in texts)
            ids = {x.lower() for x in doc_ids(d)}
            is_gold = bool(gold and (ids & gold))
            if not is_gold and sh and texts:
                n = _norm(" ".join(texts))
                is_gold = any(s in n for s in sh)
            if is_gold:
                hits += 1
                if rank is None:
                    rank = i
        return {"n": len(docs), "chars": chars, "gold_rank": rank, "gold_hits": hits,
                "ids": [doc_ids(d)[:1] for d in docs[:20]]}
    except Exception:
        return {"n": 0, "chars": 0, "gold_rank": None, "gold_hits": 0, "ids": []}


# ----------------------------------------------------------------------------- recorders

def wrap_tool(name: str, fn):
    """Return fn wrapped so each call logs args, size, gold rank and latency."""
    def _w(*a, **kw):
        t0 = time.perf_counter()
        try:
            res = fn(*a, **kw)
        except Exception as exc:  # noqa: BLE001 - re-raised unchanged
            event("tool", name=name, args=_args_summary(a, kw), error=f"{type(exc).__name__}: {exc}"[:300],
                  ms=round((time.perf_counter() - t0) * 1000))
            raise
        try:
            info = summarize_docs(res)
            if name in _RERANK_TOOLS:
                src = a[1] if len(a) > 1 else kw.get("papers", kw.get("chunks"))
                info["input"] = summarize_docs(src)
            event("tool", name=name, args=_args_summary(a, kw), ms=round((time.perf_counter() - t0) * 1000), **info)
        except Exception:
            pass
        return res
    _w.__name__ = getattr(fn, "__name__", name)
    _w.__doc__ = getattr(fn, "__doc__", None)
    return _w


def _args_summary(a, kw) -> dict:
    out = {}
    try:
        for i, v in enumerate(a[:3]):
            out[f"a{i}"] = v[:200] if isinstance(v, str) else (f"<{type(v).__name__} n={len(v)}>" if hasattr(v, "__len__") else str(v)[:60])
        for k, v in list(kw.items())[:6]:
            out[k] = v[:200] if isinstance(v, str) else (f"<{type(v).__name__} n={len(v)}>" if hasattr(v, "__len__") else str(v)[:60])
    except Exception:
        pass
    return out


def record_repl_block(code: str, result: Any, seconds: float) -> None:
    """One executed code block: what it printed, what the LM will be shown, gold survival."""
    if _ctx.get() is None:
        return
    try:
        out = getattr(result, "stdout", "") or ""
        err = getattr(result, "stderr", "") or ""
        cap = int(os.environ.get("XC_AGENT_OBS_CHARS", "5000"))
        shown = out[:cap]
        doc_vars = []
        loc = getattr(result, "locals", None) or {}
        for k, v in list(loc.items())[:200]:
            if k.startswith("_") or k in TRACED_TOOLS or callable(v):
                continue
            info = summarize_docs(v)
            if info["n"]:
                doc_vars.append({"var": k, "n": info["n"], "chars": info["chars"], "gold_rank": info["gold_rank"]})
        event("repl",
              code=code[:1500], calls=sorted({m for m in re.findall(r"\b([a-z_]+)\(", code) if m in TRACED_TOOLS}),
              stdout_chars=len(out), shown_chars=len(shown), truncated=len(out) > cap,
              stderr_head=err.strip()[:300], seconds=round(seconds, 2),
              gold_in_stdout=gold_in_text(out), gold_in_shown=gold_in_text(shown),
              doc_vars=doc_vars[:20])
    except Exception:
        pass


def _msg_chars(messages) -> int:
    try:
        return sum(len(m.get("content")) if isinstance(m.get("content"), str) else len(str(m.get("content")))
                   for m in messages if isinstance(m, dict))
    except Exception:
        return -1


def _msgs_text(messages) -> str:
    try:
        return "\n".join(m.get("content") if isinstance(m.get("content"), str) else str(m.get("content"))
                         for m in messages if isinstance(m, dict))
    except Exception:
        return ""


def record_turn(prompt, iteration) -> None:
    """One LM call: the context it actually saw and what it answered."""
    if _ctx.get() is None:
        return
    try:
        msgs = prompt if isinstance(prompt, list) else [{"content": str(prompt)}]
        resp = getattr(iteration, "response", "") or ""
        event("turn", prompt_msgs=len(msgs), prompt_chars=_msg_chars(msgs),
              gold_in_prompt=gold_in_text(_msgs_text(msgs)),
              response_chars=len(resp), response_head=resp[:400],
              code_blocks=len(getattr(iteration, "code_blocks", []) or []),
              # iteration.final_answer is assigned by the loop AFTER this hook runs,
              # so test the response text itself.
              final=("FINAL(" in resp) or ("FINAL_VAR(" in resp),
              seconds=round(getattr(iteration, "iteration_time", 0.0) or 0.0, 2))
    except Exception:
        pass


def record_iteration(iteration, new_messages, cap: int) -> None:
    """format_iteration: raw REPL output vs the message the LM gets next turn."""
    if _ctx.get() is None:
        return
    try:
        raw = sum(len((getattr(b, "result", None) and b.result.stdout) or "") for b in (getattr(iteration, "code_blocks", []) or []))
        event("format", cap=cap, raw_stdout_chars=raw, message_chars=_msg_chars(new_messages),
              gold_in_message=gold_in_text(_msgs_text(new_messages)))
    except Exception:
        pass


def record_prune(before, after, keep_recent: int, max_total: int) -> None:
    """_prune_message_history: what the summarisation step threw away."""
    if _ctx.get() is None:
        return
    try:
        event("prune", keep_recent=keep_recent, max_total_chars=max_total,
              msgs_before=len(before), msgs_after=len(after),
              chars_before=_msg_chars(before), chars_after=_msg_chars(after),
              gold_before=gold_in_text(_msgs_text(before)), gold_after=gold_in_text(_msgs_text(after)))
    except Exception:
        pass
