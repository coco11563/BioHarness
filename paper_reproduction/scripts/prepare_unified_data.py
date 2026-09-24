"""Materialise the benchmark files the research runner reads, from the HF release.

The runner (``scripts/run_unified_benchmark.py``) loads ``benchmark/unified/<name>.jsonl``.
The HF dataset ``Shaow/BioHarness_Eval`` hosts the same items; seven of its files are
byte-identical to the research files, and two differ only in packaging:

* ``scihorizon-gene.jsonl``  -> ``scihorizon_hgkb.jsonl``: field ``dataset`` is
  ``"scihorizon_hgkb"`` in the research file;
* ``medxpertqa_text.jsonl``: the research file carries ``"subtask": "mcq"`` and
  ``"context": ""`` (HF: no ``subtask``, ``context`` null) and a different key order.

LitQA2 is not hosted; build it with the HF repository's ``scripts/build_litqa2.py`` and
pass the result with ``--litqa2``.

Every output is checked against the SHA-256 of the file the paper runs used.

Usage::

    python scripts/prepare_unified_data.py --hf-dir <local copy of the HF data/ folder> \\
        [--litqa2 <output of build_litqa2.py>]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "benchmark" / "unified"

# sha256 of the research files in benchmark/unified/ that produced the paper's runs
EXPECTED = {
    "bioasq.jsonl": "1f2ea9afe2b8570c284c545e0c39060f83c3715d421bbf4847d0b66412fda1fd",
    "geneturing.jsonl": "85008a527c5b6755bfdd6fcc74c3579ebb2013ba83a350718e1d430f8723cf2d",
    "medmcqa.jsonl": "f14d29b32244c7e2d984b6108cc640e2fa7dc122fe65f9ca5ea7af29115e65ab",
    "medqa_mainland.jsonl": "3368e966ba596885e0bd93634272b50e3f3ec683b542c04fc9921e9e67ad33d1",
    "medqa_taiwan.jsonl": "5ae122938fa0635362b6f6ef2aa68698222634bbb95a95cd67275efe51ff48be",
    "medqa_us.jsonl": "5a8459ceb5da67a81ad23a5ad3e70e331d1dd9909e5a146f287c3ab58d5c00ca",
    "pubmedqa_pqal_test.jsonl": "b0e80d372ea243b16384267c8fe85de6deed5a0552680eeeb69415597fa4130a",
    "scihorizon_hgkb.jsonl": "8f4e76bc4ed870af725ab8120f36caf8d18f59a40ac7aad2a04b171d39073981",
    "medxpertqa_text.jsonl": "0553185673842f5906ec3cf3e7f4cf13f96c1d83a030b8da5d601cb8f256ee4b",
    "litqa2.jsonl": "66e2910584d7db59c9c148bb5ada45850825861ec67bc4e08c98c1739e7ec67d",
}
COPY_AS_IS = [
    "bioasq.jsonl", "geneturing.jsonl", "medmcqa.jsonl", "medqa_mainland.jsonl",
    "medqa_taiwan.jsonl", "medqa_us.jsonl", "pubmedqa_pqal_test.jsonl",
]
MX_KEYS = ["id", "dataset", "question", "options", "answer", "answer_type",
           "question_type", "subtask", "context", "metadata"]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--hf-dir", type=Path, required=True)
    ap.add_argument("--litqa2", type=Path, help="output of the HF build_litqa2.py")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    for name in COPY_AS_IS:
        shutil.copyfile(args.hf_dir / name, OUT / name)

    sci = [{**r, "dataset": "scihorizon_hgkb"}
           for r in _read_jsonl(args.hf_dir / "scihorizon-gene.jsonl")]
    _write_jsonl(OUT / "scihorizon_hgkb.jsonl", sci)

    mx = []
    for r in _read_jsonl(args.hf_dir / "medxpertqa_text.jsonl"):
        r = {**r, "subtask": "mcq", "context": r.get("context") or ""}
        mx.append({k: r[k] for k in MX_KEYS})
    _write_jsonl(OUT / "medxpertqa_text.jsonl", mx)

    if args.litqa2:
        shutil.copyfile(args.litqa2, OUT / "litqa2.jsonl")

    bad = 0
    for name, want in EXPECTED.items():
        path = OUT / name
        if not path.exists():
            print(f"missing  {name}")
            continue
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        ok = got == want
        bad += not ok
        print(f"{'ok      ' if ok else 'MISMATCH'} {name}")
    if bad:
        raise SystemExit(f"{bad} file(s) differ from the research copies")


if __name__ == "__main__":
    main()
