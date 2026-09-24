"""Convert MedXpertQA (Text subset) into the unified benchmark format.

Source: TsinghuaC3I/MedXpertQA, Text/test.jsonl (2,450 items), downloaded via
hf-mirror.com on 2026-08-26.

Format decisions (each one matters for comparability):
- `answer` stores the option TEXT, matching medqa_* in this benchmark (the
  evaluator resolves letters via `options`). Storing the letter instead would
  silently change scoring.
- The source `question` field already embeds an "Answer Choices:" block. We strip
  it, because `options` is passed separately and duplicating the choices inflates
  the prompt and changes the task.
- 10 options (A-J) per item => chance = 10%, versus 20-25% for the existing MCQ
  datasets. Any pooled accuracy across datasets must account for this.
- `subtask` is "mcq" so the existing constrained-generation path applies.
- Stratification fields (question_type / medical_task / body_system) are kept in
  `metadata` for per-slice reporting.
"""
import json, re, sys, os
from collections import Counter

SRC = "rebuttal_round1/data/medxpertqa_text_test.raw.jsonl"
DST = "benchmark/unified/medxpertqa_text.jsonl"
AUDIT = "rebuttal_round1/data/conversion_audit.json"

_CHOICES = re.compile(r"\n?\s*Answer Choices:\s*.*\Z", re.S | re.I)


def strip_choices(q: str) -> tuple[str, bool]:
    new = _CHOICES.sub("", q).strip()
    return (new, new != q.strip())


def main():
    rows = [json.loads(l) for l in open(SRC)]
    out, audit = [], {"n_in": len(rows), "stripped": 0, "skipped": [],
                      "n_options": Counter(), "label_in_options": 0}
    for r in rows:
        opts = r.get("options") or {}
        label = str(r.get("label", "")).strip()
        q, stripped = strip_choices(str(r.get("question", "")))
        audit["stripped"] += int(stripped)
        audit["n_options"][len(opts)] += 1
        if label not in opts:
            audit["skipped"].append({"id": r.get("id"), "why": "label not in options"})
            continue
        audit["label_in_options"] += 1
        out.append({
            "id": f"medxpertqa_{r.get('id')}",
            "dataset": "medxpertqa_text",
            "question": q,
            "options": opts,
            "answer": opts[label],          # option TEXT, as medqa_* does
            "answer_type": "label",
            "question_type": "mcq",
            "subtask": "mcq",
            "context": "",
            "metadata": {
                "source": "TsinghuaC3I/MedXpertQA Text/test",
                "gold_letter": label,
                "n_options": len(opts),
                "mx_question_type": r.get("question_type"),
                "medical_task": r.get("medical_task"),
                "body_system": r.get("body_system"),
            },
        })
    os.makedirs(os.path.dirname(DST), exist_ok=True)
    with open(DST, "w") as f:
        for x in out:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    audit["n_out"] = len(out)
    audit["n_options"] = dict(audit["n_options"])
    json.dump(audit, open(AUDIT, "w"), indent=1, ensure_ascii=False)

    print(f"in={audit['n_in']}  out={len(out)}  skipped={len(audit['skipped'])}")
    print(f"  stripped embedded 'Answer Choices': {audit['stripped']}")
    print(f"  option-count distribution: {audit['n_options']}")
    print(f"  -> {DST}")
    if out:
        s = out[0]
        print(f"\n  sample id={s['id']}  gold_letter={s['metadata']['gold_letter']}")
        print(f"    question[:90]: {s['question'][:90]!r}")
        print(f"    answer[:70]:   {s['answer'][:70]!r}")


if __name__ == "__main__":
    main()
