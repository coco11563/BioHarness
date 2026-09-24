"""Atlas component (D): structured tissue-expression knowledge as supplementary context.

The SciHorizon HGKB ``expression`` subtask (gene -> ``tissue_list``) is a
structured-fact task scored by exact-string set-F1 over a fixed ~27-tissue
vocabulary. The ``D`` component has two halves:

1. **Format / recall prompt (ships and runs by default).** A recall-oriented
   prompt that lists tissues from the fixed allowed vocabulary and returns a
   parseable JSON array. The upstream free-text / comma-separated expression
   prompt produced 0/210 parseable answers and ~6% set-F1 — almost entirely a
   formatting artifact, not a knowledge gap. This half is wired into the cascade
   (see ``constrained.py``) and is what the default ``-D`` path runs.

2. **Atlas-as-context (+D) — wired, opt-in.** The cascade fetches the gene's
   HPA bulk tissue expression via ``clients/atlas.py`` (``AtlasClient``) and
   injects it as a *supplementary* reference block, which the model reconciles
   with its parametric knowledge (it is NOT forced to copy the atlas). This is
   **off by default** (``-D``), so the package runs with no atlas backend;
   enabling it needs ``BIOHARNESS_ENABLE_ATLAS=1`` plus a reachable primitive
   server at ``BIOHARNESS_ATLAS_URL`` (e.g. ``scdata_primitive_server`` on
   :8443). ``build_expression_messages(question, atlas_rows=...)`` /
   ``atlas_rows_from_hpa(...)`` are the standalone helpers behind the wiring.

The paper reports atlas context as a post-hoc case study applied to
BioHarness alone (Table 1, footnote a); see ``paper_reproduction/README.md``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Any

# HPA consensus tissue vocabulary used by the SciHorizon expression GT (27 organs).
EXPRESSION_TISSUE_VOCAB: tuple[str, ...] = (
    "adrenal", "appendix", "bone marrow", "brain", "colon", "duodenum",
    "endometrium", "esophagus", "fat", "gall bladder", "heart", "kidney",
    "liver", "lung", "lymph node", "ovary", "pancreas", "placenta", "prostate",
    "salivary gland", "skin", "small intestine", "spleen", "stomach", "testis",
    "thyroid", "urinary bladder",
)

# Recall + fixed-vocabulary + parseable-JSON prompt (replaces the upstream
# free-text expression prompt). Format with ``vocab=...``.
EXPRESSION_SYSTEM_PROMPT: str = (
    "You are an expert on human gene/protein tissue expression. "
    "List ALL tissues from the ALLOWED list where the gene is expressed — be "
    "comprehensive, include every tissue with meaningful expression, not just "
    "the top one. Choose ONLY from the allowed tissues. Output a JSON array of "
    "tissue names, nothing else.\n"
    "ALLOWED TISSUES: {vocab}\n"
)

# Supplementary atlas reference block (+D) — the structured evidence that
# repairs the retrieved literature. Format with ``rows=...`` where rows is e.g.
# ``"TP53: liver:23, kidney:18, ..."`` (HPA nTPM, vocab-mapped).
ATLAS_CONTEXT_TEMPLATE: str = (
    "Reference tissue expression (HPA nTPM; higher = more expressed) — "
    "supplementary structured evidence not found in the retrieved literature:\n"
    "{rows}\n"
)

# Brain sub-regions and lab-style names collapse to the GT vocabulary.
_TISSUE_NORMALIZE: dict[str, str] = {
    "adipose tissue": "fat", "gallbladder": "gall bladder", "heart muscle": "heart",
    "adrenal gland": "adrenal", "thyroid gland": "thyroid",
    "amygdala": "brain", "basal ganglia": "brain", "cerebellum": "brain",
    "cerebral cortex": "brain", "choroid plexus": "brain", "hippocampal formation": "brain",
    "hypothalamus": "brain", "midbrain": "brain", "pons": "brain",
    "medulla oblongata": "brain", "spinal cord": "brain", "substantia nigra": "brain",
    "thalamus": "brain", "white matter": "brain", "retina": "brain", "pituitary gland": "brain",
}


def normalize_tissue(name: str) -> str:
    """Map an HPA/atlas tissue name to the GT vocabulary."""
    name = (name or "").lower().strip()
    return _TISSUE_NORMALIZE.get(name, name)


_GENE_RE = re.compile(r"expression pattern of ([A-Za-z0-9\-._]+) gene", re.IGNORECASE)


def gene_from_expression_question(question: str) -> str | None:
    """Extract the gene symbol from a SciHorizon expression question."""
    match = _GENE_RE.search(question or "")
    return match.group(1) if match else None


def atlas_rows_from_hpa(gene: str, entries: Iterable[dict[str, Any]]) -> str | None:
    """Render HPA ``get_tissue_expression`` entries as a reference block.

    ``entries`` are dicts with ``tissue`` and ``nx``; only tissues that map into
    the allowed vocabulary are kept, sorted by descending nTPM.
    """
    allowed = set(EXPRESSION_TISSUE_VOCAB)
    agg: dict[str, float] = {}
    for e in entries or []:
        v = normalize_tissue(e.get("tissue", ""))
        nx = float(e.get("nx") or 0.0)
        if v in allowed and nx > 0:
            agg[v] = max(agg.get(v, 0.0), nx)
    if not agg:
        return None
    pairs = sorted(agg.items(), key=lambda kv: -kv[1])
    return f"{gene}: " + ", ".join(f"{t}:{nx:.0f}" for t, nx in pairs)


def build_expression_messages(
    question: str, *, evidence: str = "", atlas_rows: str | None = None,
) -> list[dict[str, str]]:
    """Build the chat messages for an expression question (repair context).

    ``evidence`` is the retrieved literature, used as the base context. With
    ``atlas_rows`` provided (+D) the HPA reference is appended as the
    supplementary structured evidence that repairs the literature; without it
    (-D) the prompt degrades to the literature-only answer. The single ablation
    flag is atlas-in-context-or-not.
    """
    vocab = ", ".join(EXPRESSION_TISSUE_VOCAB)
    user = EXPRESSION_SYSTEM_PROMPT.format(vocab=vocab)
    user += "\nLiterature evidence:\n" + (evidence or "(none retrieved)") + "\n"
    if atlas_rows:
        user += "\n" + ATLAS_CONTEXT_TEMPLATE.format(rows=atlas_rows)
    user += f"\nGene question: {question}\nAnswer (JSON array):"
    return [{"role": "user", "content": user}]


def parse_expression_tissues(text: str) -> list[str]:
    """Parse a model answer into a deduped tissue list restricted to the vocab.

    Accepts a JSON array, a ``{"tissue_list": [...]}`` object, or a degenerate
    comma / newline / semicolon-separated fallback. A leading prose prefix
    before the array is tolerated (the first ``[...]`` span is tried first).
    ``"low expression"`` is allowed through as the HPA / GT no-expression
    sentinel (genes whose ground-truth tissue list is empty).
    """
    text = (text or "").strip()
    cand: list[str] = []
    # Try the first JSON array span (handles "Here: [...]"), then the whole text.
    match = re.search(r"\[.*\]", text, re.S)
    for candidate in ((match.group(0) if match else None), text):
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, list):
            cand = [str(x).lower().strip() for x in obj]
        elif isinstance(obj, dict) and "tissue_list" in obj:
            cand = [str(x).lower().strip() for x in obj["tissue_list"]]
        break
    if not cand:  # last resort: split on commas, newlines, or semicolons
        stripped = re.sub(r"[\[\]\"{}]", " ", text)
        cand = [t.lower().strip() for t in re.split(r"[,\n;]+", stripped) if t.strip()]
    allowed = set(EXPRESSION_TISSUE_VOCAB) | {"low expression"}
    out, seen = [], set()
    for t in cand:
        if t in allowed and t not in seen:
            seen.add(t)
            out.append(t)
    return out


EXPRESSION_MAX_TOKENS: int = 300
