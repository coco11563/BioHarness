"""LLM-based pre-router: classify which items would benefit from Disco scRNA atlas.

Scans all questions in the 8 unified benchmark datasets, asks Qwen "would this
question benefit from single-cell RNA atlas data?", outputs a YES/NO + entity
extraction per item.

Output: .cache/disco_router/router_predictions.jsonl
        each line: {dataset, id, question, question_type, route_disco: bool,
                    extracted_genes, extracted_tissues, extracted_celltypes, raw}

Then `--item-filter` flag in run_unified_benchmark.py can subset the run to only
items with route_disco=true for a paired V14 vs V14+disco comparison.
"""
import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmark" / "code" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUT_DIR = Path(__file__).resolve().parent.parent / '.cache' / 'disco_router'
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / 'router_predictions.jsonl'

ROUTER_PROMPT = """You are classifying biomedical questions to decide whether a single-cell RNA atlas (scRNA-seq) would help answer them.

A scRNA atlas provides:
- Per-cell-type gene expression statistics (avg expression, % cells expressing)
- Top markers for specific (tissue, cell type) pairs
- Quantitative tissue/cell-type distributions
- Comparative expression across cell types or disease vs healthy

Answer YES only if the question requires:
- Knowing in which tissues / cell types a SPECIFIC gene is expressed
- Identifying marker genes of a specific cell type
- Comparing gene expression between cell types or conditions
- Knowing cell type composition of a tissue/atlas

Answer NO for:
- Pure clinical/medical questions (diagnosis, treatment, drug)
- Gene location/chromosome/disease association (not expression)
- Pathway/GO/protein structure questions
- General medical knowledge

Question: {question}

Reply in this exact format:
ROUTE: YES|NO
GENES: <comma-separated gene symbols mentioned, or NONE>
TISSUES: <comma-separated tissues mentioned, or NONE>
CELLTYPES: <comma-separated cell types mentioned, or NONE>"""


def parse_router_response(text: str) -> dict:
    """Parse router LLM response."""
    out = {'route_disco': False, 'extracted_genes': [], 'extracted_tissues': [], 'extracted_celltypes': []}
    if not text:
        return out
    # ROUTE
    m = re.search(r'ROUTE:\s*(YES|NO)', text, re.IGNORECASE)
    if m:
        out['route_disco'] = m.group(1).upper() == 'YES'
    # GENES
    m = re.search(r'GENES:\s*([^\n]+)', text, re.IGNORECASE)
    if m:
        v = m.group(1).strip()
        if v.upper() != 'NONE':
            out['extracted_genes'] = [g.strip() for g in v.split(',') if g.strip()]
    # TISSUES
    m = re.search(r'TISSUES:\s*([^\n]+)', text, re.IGNORECASE)
    if m:
        v = m.group(1).strip()
        if v.upper() != 'NONE':
            out['extracted_tissues'] = [t.strip() for t in v.split(',') if t.strip()]
    # CELLTYPES
    m = re.search(r'CELLTYPES:\s*([^\n]+)', text, re.IGNORECASE)
    if m:
        v = m.group(1).strip()
        if v.upper() != 'NONE':
            out['extracted_celltypes'] = [c.strip() for c in v.split(',') if c.strip()]
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+',
                    default=['scihorizon_hgkb', 'bioasq', 'geneturing',
                             'pubmedqa_pqal_test', 'medmcqa',
                             'medqa_us', 'medqa_taiwan', 'medqa_mainland'])
    ap.add_argument('--limit', type=int, default=None,
                    help='Per-dataset cap (None = all)')
    ap.add_argument('--concurrency', type=int, default=10)
    ap.add_argument('--resume', action='store_true', help='Skip items already in output')
    args = ap.parse_args()

    from benchmark import BenchmarkLoader
    from benchmark.client_protocol import LLMClient
    from src.config import get_config

    config = get_config()
    loader = BenchmarkLoader(unified_dir=Path(__file__).parent.parent / "benchmark" / "unified")

    # Existing routes (resume)
    seen_keys = set()
    if args.resume and OUT_PATH.exists():
        with open(OUT_PATH) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    seen_keys.add((d['dataset'], d['id']))
                except Exception:
                    pass
        print(f'Resume: skipping {len(seen_keys)} already-routed items')

    # Load all items
    all_items = []
    for ds in args.datasets:
        items = list(loader.load_dataset(ds))
        if args.limit:
            items = items[:args.limit]
        for it in items:
            key = (ds, it.id)
            if key not in seen_keys:
                all_items.append((ds, it))
    print(f'To route: {len(all_items)} items across {len(args.datasets)} datasets')

    client = LLMClient(
        backends=[config.llm.get_server() for _ in range(2)],
        model=config.llm.model,
        max_tokens=128,
        temperature=0.0,
    )

    out_f = open(OUT_PATH, 'a')
    sem = asyncio.Semaphore(args.concurrency)
    n_done = [0]
    n_yes = [0]
    t0 = time.time()
    lock = asyncio.Lock()

    async def process(ds, item):
        async with sem:
            prompt = ROUTER_PROMPT.format(question=item.question)
            for attempt in range(3):
                try:
                    async with client:
                        resp = await client.generate(question=prompt, question_type='factoid')
                    raw = resp.response_text or ''
                    if raw.startswith('ERROR:'):
                        await asyncio.sleep(2 ** attempt); continue
                    parsed = parse_router_response(raw)
                    rec = {
                        'dataset': ds,
                        'id': item.id,
                        'question': item.question,
                        'question_type': item.question_type,
                        'route_disco': parsed['route_disco'],
                        'extracted_genes': parsed['extracted_genes'],
                        'extracted_tissues': parsed['extracted_tissues'],
                        'extracted_celltypes': parsed['extracted_celltypes'],
                        'raw': raw[:300],
                    }
                    async with lock:
                        out_f.write(json.dumps(rec) + '\n'); out_f.flush()
                        n_done[0] += 1
                        if parsed['route_disco']: n_yes[0] += 1
                        if n_done[0] % 50 == 0:
                            elapsed = time.time() - t0
                            rate = n_done[0] / elapsed
                            eta = (len(all_items) - n_done[0]) / rate
                            print(f'  [{n_done[0]}/{len(all_items)}] yes={n_yes[0]} ({100*n_yes[0]/n_done[0]:.1f}%) '
                                  f'rate={rate:.1f}/s eta={eta:.0f}s', flush=True)
                    break
                except Exception as e:
                    if attempt == 2:
                        async with lock:
                            out_f.write(json.dumps({
                                'dataset': ds, 'id': item.id,
                                'question': item.question,
                                'question_type': item.question_type,
                                'route_disco': False,
                                'error': str(e)[:200],
                            }) + '\n'); out_f.flush()
                            n_done[0] += 1
                    await asyncio.sleep(2 ** attempt)

    await asyncio.gather(*(process(ds, item) for ds, item in all_items))
    out_f.close()
    print(f'\nDone. Total: {n_done[0]}, YES: {n_yes[0]} ({100*n_yes[0]/max(n_done[0],1):.1f}%)')
    print(f'Output: {OUT_PATH}')


if __name__ == '__main__':
    asyncio.run(main())
