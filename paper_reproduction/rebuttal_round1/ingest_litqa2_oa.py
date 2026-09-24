"""Ingest LitQA2 open-access source papers into papergraph + Qdrant `chunks`.

Reconstructs the original BioC-based pipeline, verified against an already-ingested
paper (PMC10944506):
  papers.abstract  <- BioC `abstract` passages (not indexed as chunks)
  sections         <- one row per `title_N` passage; paragraphs + figure captions
                      under that heading form content_text; `ref`/`footnote` dropped
  chunks           <- <=512 tokens (tiktoken cl100k_base, verified ratio 1.000),
                      never crossing a section boundary
  Qdrant point.id  == chunks.id ; unnamed 1024-d cosine vector

Default is a dry run.  --commit writes, and records every inserted uuid in a
manifest so the whole batch can be deleted again.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from datetime import date, datetime, timezone
from pathlib import Path

import aiohttp
import asyncpg
import tiktoken

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from qdrant_client import AsyncQdrantClient, models  # noqa: E402
from src.config import get_config  # noqa: E402
from utils.clients import embed_client  # noqa: E402

BIOC = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{}/unicode"
MAX_TOKENS = 512
ENC = tiktoken.get_encoding("cl100k_base")

# BioC section_type -> the vocabulary already present in papergraph.sections
SECTION_TYPE = {
    "INTRO": "introduction", "RESULTS": "results", "DISCUSS": "discussion",
    "METHODS": "methods", "CONCL": "conclusions", "SUPPL": "supplementary-material",
    "CASE": "case", "APPENDIX": "appendix", "ABBR": "abbreviations",
}
BODY_TYPES = {"paragraph", "fig_caption", "fig_title_caption", "table_caption", "title"}
DROP_TYPES = {"ref", "footnote", "front", "abstract"}


def parse_bioc(doc: dict) -> dict | None:
    """BioC document -> {paper metadata, sections[]}.

    PMC BioC keeps article ids / year / authors on the *front* passage's infons,
    not on the document's (which carries only `license`), so metadata is read
    from there.
    """
    passages = doc.get("passages") or []
    doc_infons = doc.get("infons") or {}
    infons = dict(doc_infons)
    for p in passages:
        pi = p.get("infons") or {}
        if pi.get("type") == "front" or "article-id_pmid" in pi or "article-id_doi" in pi:
            infons = {**pi, **{k: v for k, v in doc_infons.items() if v}}
            break
    title, abstract_parts, sections = "", [], []
    cur = None

    for p in passages:
        i = p.get("infons") or {}
        ptype, stype, text = i.get("type") or "", i.get("section_type") or "", (p.get("text") or "").strip()
        if not text:
            continue
        if ptype == "front":
            title = title or text
        elif ptype == "abstract":
            abstract_parts.append(text)
        elif ptype.startswith("title_"):
            try:
                level = int(ptype.split("_", 1)[1])
            except ValueError:
                level = 1
            cur = {
                "title": text or (SECTION_TYPE.get(stype) or "body").title(),
                "hierarchy_level": level,
                "section_type": SECTION_TYPE.get(stype) if level == 1 else None,
                "parts": [],
            }
            sections.append(cur)
        elif ptype in BODY_TYPES:
            if cur is None:  # body text before any heading; sections.title is NOT NULL
                canon = SECTION_TYPE.get(stype)
                cur = {"title": (canon or "body").replace("-", " ").title(),
                       "hierarchy_level": 1, "section_type": canon, "parts": []}
                sections.append(cur)
            cur["parts"].append(text)
        elif ptype in DROP_TYPES:
            continue

    for idx, s in enumerate(sections):
        s["sequence_order"] = idx
        s["content_text"] = "\n\n".join(s.pop("parts"))
    if not any(s["content_text"] for s in sections):
        return None

    year = str(infons.get("year") or "")
    authors = [infons[k] for k in sorted(infons) if k.startswith("name_")]
    kwd = (infons.get("kwd") or "").strip()
    return {
        "pmcid": str(infons.get("article-id_pmc") or "").replace("PMC", ""),
        "pmid": str(infons.get("article-id_pmid") or "") or None,
        "doi": (infons.get("article-id_doi") or "").lower() or None,
        "title": title or f"PMC{infons.get('article-id_pmc') or ''}",
        "abstract": "\n\n".join(abstract_parts) or None,
        "journal": (infons.get("journal") or "")[:500] or None,
        "publication_date": date(int(year), 1, 1) if year.isdigit() else None,
        "license": (doc_infons.get("license") or infons.get("license") or "")[:200] or None,
        "authors": json.dumps(authors),
        "keywords": kwd.split() if kwd else None,
        "sections": sections,
    }


def chunk_section(text: str) -> list[dict]:
    """Split one section into <=MAX_TOKENS chunks on paragraph, then sentence, boundaries."""
    if not text.strip():
        return []
    out, buf, buf_start, cursor = [], "", 0, 0

    def flush(end: int) -> None:
        nonlocal buf, buf_start
        if buf.strip():
            out.append({"text_content": buf.strip(), "token_count": len(ENC.encode(buf.strip())),
                        "start_char_offset": buf_start, "end_char_offset": end})
        buf = ""

    for para in text.split("\n\n"):
        block = para + "\n\n"
        if len(ENC.encode(buf + block)) <= MAX_TOKENS:
            if not buf:
                buf_start = cursor
            buf += block
        else:
            flush(cursor)
            buf_start = cursor
            if len(ENC.encode(block)) <= MAX_TOKENS:
                buf = block
            else:  # oversized paragraph: hard-split on token windows
                toks = ENC.encode(block)
                for k in range(0, len(toks), MAX_TOKENS):
                    piece = ENC.decode(toks[k:k + MAX_TOKENS])
                    out.append({"text_content": piece.strip(),
                                "token_count": len(toks[k:k + MAX_TOKENS]),
                                "start_char_offset": cursor, "end_char_offset": cursor + len(piece)})
                buf = ""
        cursor += len(block)
    flush(cursor)
    for k, c in enumerate(out):
        c["sequence_order"] = k
    return [c for c in out if c["text_content"]]


async def fetch_all(pmcids: list[str]) -> dict[str, dict]:
    sem, got = asyncio.Semaphore(6), {}

    async def one(sess: aiohttp.ClientSession, pmcid: str) -> None:
        async with sem:
            for attempt in range(3):
                try:
                    async with sess.get(BIOC.format(pmcid), timeout=aiohttp.ClientTimeout(total=90)) as r:
                        if r.status == 200:
                            j = await r.json(content_type=None)
                            got[pmcid] = j[0]["documents"][0]
                            return
                except Exception:
                    pass
                await asyncio.sleep(1.5 * (attempt + 1))

    async with aiohttp.ClientSession() as sess:
        await asyncio.gather(*[one(sess, p) for p in pmcids])
    return got


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", action="store_true", help="actually write to papergraph and Qdrant")
    args = ap.parse_args()

    oa = json.loads((ROOT / "rebuttal_round1/litqa2_oa_fetchable.json").read_text())
    cov = json.loads((ROOT / "rebuttal_round1/litqa2_coverage.json").read_text())
    # DOI/PMID resolved by the NCBI ID converter -- authoritative fallback
    idmap = {p: (d, pm) for d, p, pm in cov["missing_with_pmcid"] if p}
    pmcids = [v["pmcid"] for v in oa.values()]
    print(f"待处理 {len(pmcids)} 篇 PMC OA 全文\n")

    docs = await fetch_all(pmcids)
    print(f"BioC 抓取成功 {len(docs)}/{len(pmcids)}")

    papers, n_sec, n_chunk, tok = [], 0, 0, []
    for pmcid, doc in docs.items():
        rec = parse_bioc(doc)
        if rec is None:
            print(f"  ! {pmcid} 解析后无正文,跳过")
            continue
        rec["pmcid"] = rec["pmcid"] or pmcid.replace("PMC", "")
        fb_doi, fb_pmid = idmap.get(pmcid, (None, None))
        rec["doi"] = rec["doi"] or (fb_doi or "").lower() or None
        rec["pmid"] = rec["pmid"] or (str(fb_pmid) if fb_pmid else None)
        for s in rec["sections"]:
            s["chunks"] = chunk_section(s["content_text"])
            n_chunk += len(s["chunks"])
            tok += [c["token_count"] for c in s["chunks"]]
        rec["sections"] = [s for s in rec["sections"] if s["chunks"] or s["content_text"]]
        n_sec += len(rec["sections"])
        papers.append(rec)

    miss = [(r["pmcid"], not r["pmid"], not r["doi"], not r["title"]) for r in papers]
    print(f"  元数据缺失: pmid {sum(m[1] for m in miss)} 篇 / doi {sum(m[2] for m in miss)} 篇 "
          f"/ title {sum(m[3] for m in miss)} 篇  (应为 0)")
    tok.sort()
    print(f"\n解析结果: {len(papers)} 篇 / {n_sec} sections / {n_chunk} chunks")
    if tok:
        print(f"  chunk token: p25={tok[len(tok)//4]} med={tok[len(tok)//2]} "
              f"p90={tok[int(len(tok)*.9)]} max={tok[-1]}   (库中基准 175/366/512/512)")
        print(f"  每篇平均 {n_sec/len(papers):.1f} sections, {n_chunk/n_sec:.2f} chunks/section (库中基准 1.50)")

    if not args.commit:
        print("\n[dry-run] 未写库。加 --commit 执行写入。")
        return

    cfg = get_config()
    pool = await asyncpg.create_pool(cfg.postgres.papergraph_url, min_size=2, max_size=6)
    qc = AsyncQdrantClient(cfg.qdrant.url, timeout=300)
    man = {"created_at": datetime.now(timezone.utc).isoformat(), "collection": "chunks",
           "papers": [], "sections": [], "chunks": []}

    for rec in papers:
        async with pool.acquire() as con:
            dup = await con.fetchval("select id from papers where pmcid=$1", rec["pmcid"])
            if dup:
                print(f"  跳过 PMC{rec['pmcid']}: 已存在")
                continue
            pid = uuid.uuid4()
            async with con.transaction():
                await con.execute(
                    """insert into papers (id,pmcid,pmid,title,abstract,publication_date,
                                             journal,doi,license,authors,keywords,
                                             created_at,updated_at)
                       values ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10::jsonb,$11,now(),now())""",
                    pid, rec["pmcid"], rec["pmid"], rec["title"], rec["abstract"],
                    rec["publication_date"], rec["journal"], rec["doi"], rec["license"],
                    rec["authors"], rec["keywords"])
                man["papers"].append(str(pid))
                for s in rec["sections"]:
                    sid = uuid.uuid4()
                    await con.execute(
                        """insert into sections (id,paper_id,title,section_type,hierarchy_level,
                                                 sequence_order,content_text,created_at)
                           values ($1,$2,$3,$4,$5,$6,$7,now())""",
                        sid, pid, s["title"], s["section_type"], s["hierarchy_level"],
                        s["sequence_order"], s["content_text"])
                    man["sections"].append(str(sid))
                    for c in s["chunks"]:
                        cid = uuid.uuid4()
                        await con.execute(
                            """insert into chunks (id,section_id,text_content,token_count,
                                                   sequence_order,start_char_offset,
                                                   end_char_offset,created_at)
                               values ($1,$2,$3,$4,$5,$6,$7,now())""",
                            cid, sid, c["text_content"], c["token_count"],
                            c["sequence_order"], c["start_char_offset"], c["end_char_offset"])
                        c["_id"], c["_sid"] = str(cid), str(sid)
                        man["chunks"].append(str(cid))

        flat = [c for s in rec["sections"] for c in s["chunks"] if "_id" in c]
        for k in range(0, len(flat), 32):
            batch = flat[k:k + 32]
            vecs = await embed_client.embed([c["text_content"] for c in batch])
            await qc.upsert(collection_name="chunks", points=[
                models.PointStruct(id=c["_id"], vector=v, payload={
                    "chunk_id": c["_id"], "pmcid": rec["pmcid"], "pmid": rec["pmid"] or "",
                    "section_id": c["_sid"], "token_count": c["token_count"],
                    "sequence_order": c["sequence_order"]})
                for c, v in zip(batch, vecs)])
        print(f"  PMC{rec['pmcid']}: {len(rec['sections'])} sections / {len(flat)} chunks 已入库")

    # Rebuild the rollback manifest from the database rather than from this run's
    # inserts alone, so a resumed run still lists every row of the whole batch.
    async with pool.acquire() as con:
        pids = [r["id"] for r in await con.fetch(
            "select id from papers where pmcid=any($1::text[])", [r["pmcid"] for r in papers])]
        sids = [r["id"] for r in await con.fetch(
            "select id from sections where paper_id=any($1::uuid[])", pids)] if pids else []
        cids = [r["id"] for r in await con.fetch(
            "select id from chunks where section_id=any($1::uuid[])", sids)] if sids else []
    man["papers"] = [str(x) for x in pids]
    man["sections"] = [str(x) for x in sids]
    man["chunks"] = [str(x) for x in cids]
    (ROOT / "rebuttal_round1/litqa2_ingest_manifest.json").write_text(json.dumps(man, indent=1))
    print(f"\n写入完成: {len(man['papers'])} papers / {len(man['sections'])} sections / {len(man['chunks'])} chunks")
    print("manifest -> rebuttal_round1/litqa2_ingest_manifest.json (可据此全量回滚)")
    await pool.close()
    await qc.close()


if __name__ == "__main__":
    asyncio.run(main())
