"""Repair-context case study on SciHorizon expression (offline: cached retrieval + Postgres).

Method = "repair context": retrieve literature (base context), and the atlas REPAIRS
it by adding structured tissue-expression context that literature lacks; the combined
context feeds the final generation. If the atlas returns nothing, +D degrades to the
literature-only answer.

For every expression item we record three variants — their CONTEXT and final ANSWER —
for case study:
  - atlas_only : HPA-thresholded tissue list (structured DB lookup, no LLM)
  - ours_-d    : final generation over the literature context only
  - ours (+d)  : final generation over literature + atlas-repair context
                 (== ours_-d when there is no atlas content)

Infra used: cached B6 retrieval (PMIDs) + Postgres :5432 (abstracts) + LLM :8004/5 +
atlas :8443. No embedding/rerank needed.
"""
from __future__ import annotations
import asyncio, json, os, re, statistics, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import asyncpg
import httpx

ROOT = Path(__file__).resolve().parent.parent
PG_DSN = None  # resolved from src/config.py (XC_PG_* environment variables) in main()
ATLAS_URL = os.environ.get("DISCO_SERVER_URL", "http://127.0.0.1:8443")  # atlas server (not released)
B6_CACHE = ROOT / ".cache/retrieval/B6/scihorizon_cache.jsonl"
TOP_K_DOCS = 20
OUT = ROOT / "output/case_study_repair_context.jsonl"

VOCAB = ['adrenal','appendix','bone marrow','brain','colon','duodenum','endometrium','esophagus',
         'fat','gall bladder','heart','kidney','liver','lung','lymph node','ovary','pancreas',
         'placenta','prostate','salivary gland','skin','small intestine','spleen','stomach',
         'testis','thyroid','urinary bladder']
VOCABSET = set(VOCAB)
BRAIN = {'cerebellum','cerebral cortex','basal ganglia','amygdala','hippocampal formation','hypothalamus',
         'midbrain','pons','medulla oblongata','spinal cord','substantia nigra','thalamus','white matter',
         'choroid plexus','retina','pituitary gland'}
EXPR_SYS = ("You are an expert on human gene/protein tissue expression. List ALL tissues from the "
            "ALLOWED list where the gene is expressed — be comprehensive. Choose ONLY from the allowed "
            "tissues. Output a JSON array of tissue names, nothing else.\nALLOWED TISSUES: " + ", ".join(VOCAB))


def tov(t):
    t = (t or '').lower().strip()
    return 'brain' if t in BRAIN else t.replace('adipose tissue','fat').replace('gallbladder','gall bladder').replace(' gland','').replace('heart muscle','heart')

def parse_tissues(txt):
    txt = (txt or '').strip()
    m = re.search(r'\[.*\]', txt, re.S)
    cand = []
    for c in ((m.group(0) if m else None), txt):
        if not c: continue
        try:
            a = json.loads(c)
            if isinstance(a, list): cand = [str(x).lower().strip() for x in a]; break
        except json.JSONDecodeError: continue
    if not cand:
        cand = [t.lower().strip() for t in re.split(r'[,\n;]+', re.sub(r'[\[\]"{}]',' ',txt)) if t.strip()]
    out, seen = [], set()
    for t in cand:
        if t in VOCABSET and t not in seen: seen.add(t); out.append(t)
    return out

def f1(pred, gt):
    gt = set(gt); pred = set(pred)
    if not gt: return None      # empty GT (Low-expression) — reported separately
    if not pred: return 0.0
    tp = len(pred & gt); P = tp/len(pred); R = tp/len(gt); return 2*P*R/(P+R) if P+R > 0 else 0.0

def parse_ans(a):
    try: return a if isinstance(a, dict) else json.loads(a)
    except: return {}
def gene_of(q):
    m = re.search(r'expression pattern of ([A-Za-z0-9\-\._]+) gene', q or ''); return m.group(1) if m else None


async def atlas_rows(client, gene):
    """HPA tissue rows mapped to vocab: 'tissue:nx, ...' or None."""
    try:
        r = await client.post(f"{ATLAS_URL}/primitives/hpa/get_tissue_expression", json={"gene": gene})
        if r.status_code != 200: return None, {}
        ent = r.json().get("entries") or []
    except Exception:
        return None, {}
    agg = {}
    for e in ent:
        v = tov(e.get("tissue")); nx = float(e.get("nx") or 0)
        if v in VOCABSET and nx > 0: agg[v] = max(agg.get(v, 0), nx)
    if not agg: return None, {}
    pairs = sorted(agg.items(), key=lambda x: -x[1])
    return ", ".join(f"{t}:{nx:.0f}" for t, nx in pairs), agg

def atlas_only_pred(agg, thr=1.0):
    return [t for t, nx in agg.items() if nx >= thr]

async def llm(client, servers, k, user):
    body = {"model": MODEL, "messages": [{"role": "system", "content": EXPR_SYS},
            {"role": "user", "content": user}], "max_tokens": 300, "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False}}
    for a in range(4):
        try:
            r = await client.post(f"{servers[k % len(servers)]}/chat/completions", json=body)
            return r.json()["choices"][0]["message"]["content"]
        except Exception:
            await asyncio.sleep(2 ** a)
    return "[]"

MODEL = None

async def main():
    from config import get_config
    global MODEL
    cfg = get_config(); MODEL = cfg.llm.model
    PG_DSN = cfg.postgres.pubmed_url
    # Probe configured LLM endpoints; keep only those that answer chat (some
    # ports currently serve embedding/rerank, not chat completions).
    cand = []
    for _ in range(8):
        s = cfg.llm.get_server()
        if s not in cand: cand.append(s)
    servers = []
    async with httpx.AsyncClient(timeout=20) as probe:
        for s in cand:
            try:
                r = await probe.post(f"{s}/chat/completions", json={
                    "model": MODEL, "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 1, "temperature": 0})
                if r.status_code == 200 and r.json().get("choices"):
                    servers.append(s)
            except Exception:
                pass
    if not servers:
        raise SystemExit("no working LLM chat endpoint among: " + ", ".join(cand))
    print(f"working LLM servers: {servers}", flush=True)

    items = []
    for line in open(ROOT / "benchmark/unified/scihorizon_hgkb.jsonl"):
        d = json.loads(line)
        if (d.get("question_type") or d.get("subtask")) == "expression":
            a = parse_ans(d.get("answer"))
            items.append({"id": d["id"], "q": d["question"], "gene": gene_of(d["question"]),
                          "gt": [t.lower().strip() for t in a.get("tissue_list", [])], "cat": a.get("category", "")})
    # B6 cached retrieval: qid -> top-k pmids
    cache = {}
    for line in open(B6_CACHE):
        try: d = json.loads(line)
        except: continue
        cache[d["question_id"]] = [r["pmid"] for r in (d.get("results") or [])[:TOP_K_DOCS]]
    print(f"items={len(items)}  B6-cache covers {sum(1 for it in items if it['id'] in cache)}/{len(items)}", flush=True)

    pg = await asyncpg.create_pool(PG_DSN, min_size=2, max_size=8)
    sem = asyncio.Semaphore(6)
    results = []

    async def fetch_lit(pmids):
        if not pmids: return ""
        # Cast the INPUT to bigint so the BIGINT pmid index is used; casting the
        # column (pmid::text) forces a full scan of 27.3M rows (~143s/query).
        int_pmids = [int(p) for p in pmids if str(p).isdigit()]
        if not int_pmids: return ""
        async with pg.acquire() as conn:
            rows = await conn.fetch(
                "SELECT pmid, title, abstract FROM articles "
                "WHERE pmid = ANY($1::bigint[])", int_pmids)
        by = {str(r["pmid"]): r for r in rows}
        parts, total = [], 0
        for i, pid in enumerate(pmids, 1):
            r = by.get(pid)
            if not r: continue
            block = f"[{i}] PMID:{pid}\nTitle: {r['title'] or ''}\n{(r['abstract'] or '')[:1200]}\n"
            if total + len(block) > 8000: break
            parts.append(block); total += len(block)
        return "\n".join(parts)

    async def one(client, idx, it):
        async with sem:
            pmids = cache.get(it["id"], [])
            lit = await fetch_lit(pmids)
            rows, agg = await atlas_rows(client, it["gene"]) if it["gene"] else (None, {})
            lit_block = f"Literature evidence:\n{lit}\n" if lit else "Literature evidence: (none retrieved)\n"
            # ours_-d : literature only
            user_d = lit_block + f"\nGene question: {it['q']}\nAnswer (JSON array):"
            ans_d = await llm(client, servers, idx, user_d)
            # ours (+d): literature + atlas repair; fallback to ours_-d if no atlas
            if rows:
                atlas_block = ("Reference tissue expression (HPA nTPM; higher = more expressed) — "
                               "use as supplementary evidence the literature does not contain:\n"
                               f"{it['gene']}: {rows}\n")
                user_p = lit_block + "\n" + atlas_block + f"\nGene question: {it['q']}\nAnswer (JSON array):"
                ans_p = await llm(client, servers, idx, user_p); fallback = False
            else:
                atlas_block = ""; user_p = user_d; ans_p = ans_d; fallback = True   # degrade to literature
            ao = atlas_only_pred(agg)
            rec = {
                "id": it["id"], "gene": it["gene"], "gt_tissue_list": it["gt"], "gt_category": it["cat"],
                "is_empty_gt": not it["gt"],
                "n_pmids": len(pmids),
                "context": {"literature": lit, "atlas_repair": rows or ""},
                "atlas_fallback_to_literature": fallback,
                "atlas_only":  {"answer": ao, "f1": f1(ao, it["gt"])},
                "ours_minusD": {"context_used": "literature", "raw": ans_d, "answer": parse_tissues(ans_d), "f1": f1(parse_tissues(ans_d), it["gt"])},
                "ours":        {"context_used": "literature" if fallback else "literature+atlas", "raw": ans_p, "answer": parse_tissues(ans_p), "f1": f1(parse_tissues(ans_p), it["gt"])},
            }
            results.append(rec)

    async with httpx.AsyncClient(timeout=60, limits=httpx.Limits(max_connections=16)) as client:
        tasks = [one(client, i, it) for i, it in enumerate(items)]
        for i in range(0, len(tasks), 30):
            await asyncio.gather(*tasks[i:i+30])
            print(f"  {min(i+30,len(tasks))}/{len(tasks)}", flush=True)
    await pg.close()

    results.sort(key=lambda r: r["id"])
    with open(OUT, "w") as f:
        for r in results: f.write(json.dumps(r, ensure_ascii=False) + "\n")

    ne = [r for r in results if not r["is_empty_gt"]]
    def mean(key): return statistics.mean(r[key]["f1"] for r in ne) * 100
    print(f"\n=== repair-context (non-empty GT n={len(ne)}) ===", flush=True)
    print(f"  atlas_only   F1 = {mean('atlas_only'):.1f}%", flush=True)
    print(f"  ours_-D      F1 = {mean('ours_minusD'):.1f}%   (literature only)", flush=True)
    print(f"  ours (+D)    F1 = {mean('ours'):.1f}%   (literature + atlas repair)", flush=True)
    up = sum(1 for r in ne if r['ours']['f1'] > r['ours_minusD']['f1'] + 1e-9)
    dn = sum(1 for r in ne if r['ours']['f1'] < r['ours_minusD']['f1'] - 1e-9)
    fb = sum(1 for r in ne if r['atlas_fallback_to_literature'])
    print(f"  +D vs -D: helps {up} / hurts {dn} / atlas-fallback(no atlas) {fb}", flush=True)
    print(f"saved -> {OUT}", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
