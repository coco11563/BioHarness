"""Multi-strategy retrieval and extraction cache for benchmark experiments.

Caches retrieval results (PMID list + scores) and entity/relation extractions
so all downstream models (KG-RAG, vanilla RAG, etc.) can share without
redundant Qdrant/DB/LLM calls.

Cache Structure:
    .cache/retrieval/{strategy}/{dataset}_cache.jsonl   # Retrieval results
    .cache/extraction/{extractor_id}/{prefix}/{pmid}.json  # Entity/Relation extractions

Usage - Retrieval Cache:
    from cache import RetrievalCache

    cache = RetrievalCache()
    pmids = await cache.load(question_id, strategy="C1", top_k=50)

Usage - Extraction Cache:
    from cache import ExtractionCache

    cache = ExtractionCache(extractor_id="v1_qwen_default")
    result = await cache.load(pmid)
    if result is None:
        entities, relations = await extract(doc)
        await cache.set(pmid, title, text, entities, relations)
    else:
        entities, relations = result

Available Retrieval Strategies (14 total):
    Group A (Filter-Only): A1, A2, A3, A4, A5, A6
    Group B (Filter+Vector): B1, B2, B3, B4, B5, B6
    Group C (Dense-Only): C1, C2
"""

from .retrieval_cache import (
    RetrievalCache,
    CacheMode,
    CachedDoc,
    CacheEntry,
    extract_pmids_from_metadata,
    extract_pmids_from_urls,
)

from .extraction_cache import (
    ExtractionCache,
    ExtractionEntry,
    ExtractionMeta,
    ExtractionData,
    CachedEntity,
    CachedRelation,
    compute_doc_hash,
    compute_extractor_id,
)

__all__ = [
    # Retrieval cache
    "RetrievalCache",
    "CacheMode",
    "CachedDoc",
    "CacheEntry",
    "extract_pmids_from_metadata",
    "extract_pmids_from_urls",
    # Extraction cache
    "ExtractionCache",
    "ExtractionEntry",
    "ExtractionMeta",
    "ExtractionData",
    "CachedEntity",
    "CachedRelation",
    "compute_doc_hash",
    "compute_extractor_id",
]
