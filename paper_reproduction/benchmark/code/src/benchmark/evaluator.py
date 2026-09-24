"""
Metrics evaluator for benchmark evaluation.

Provides type-specific evaluation methods that return comprehensive
raw metrics for detailed analysis and aggregation.
"""

import json
import re
from typing import Any

from rouge_score import rouge_scorer

from .metrics import (
    ClassificationMetrics,
    SetMetrics,
    RougeMetrics,
    RawMetrics,
    ScoreResult,
)


class MetricsEvaluator:
    """Type-specific evaluation with raw metrics collection.

    Supports evaluation of:
    - yesno: Binary/ternary classification
    - mcq: Single-choice multiple choice
    - mcq_multi: Multi-select multiple choice
    - factoid: Short factual answers (ROUGE-based)
    - list: List of items
    - summary: Long-form summarization
    - expression: Gene expression patterns

    Example:
        evaluator = MetricsEvaluator()
        result = evaluator.evaluate(
            predicted="yes",
            ground_truth="yes",
            question_type="yesno",
        )
        print(result.score)  # 1.0
    """

    def __init__(self):
        """Initialize evaluator with ROUGE scorer."""
        self._rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True,
        )

    def evaluate(
        self,
        predicted: str,
        ground_truth: str,
        question_type: str,
        options: dict[str, str] | None = None,
        thresholds: dict[str, float] | None = None,
    ) -> ScoreResult:
        """Unified evaluation interface.

        Args:
            predicted: Model's answer
            ground_truth: Reference answer
            question_type: Type of question (yesno, mcq, factoid, etc.)
            options: MCQ options dict (for mcq type)
            thresholds: Override default thresholds

        Returns:
            ScoreResult with score, correct flag, and raw metrics
        """
        if not ground_truth:
            return ScoreResult(
                score=0.0,
                correct=False,
                method="no_ground_truth",
                raw_metrics=RawMetrics(question_type=question_type),
            )

        method_map = {
            "yesno": self._evaluate_yesno,
            "mcq": lambda p, g: self._evaluate_mcq(p, g, options),
            "mcq_multi": self._evaluate_mcq_multi,
            "factoid": lambda p, g: self._evaluate_factoid(p, g, thresholds),
            "list": lambda p, g: self._evaluate_list(p, g, thresholds),
            "summary": lambda p, g: self._evaluate_summary(p, g, thresholds),
            "expression": lambda p, g: self._evaluate_expression(p, g, thresholds),
        }

        if question_type not in method_map:
            return ScoreResult(
                score=0.0,
                correct=False,
                method="unknown_type",
                raw_metrics=RawMetrics(question_type=question_type),
            )

        return method_map[question_type](predicted, ground_truth)

    # =========================================================================
    # Classification Types
    # =========================================================================

    def _evaluate_yesno(self, predicted: str, ground_truth: str) -> ScoreResult:
        """Evaluate yes/no/maybe classification."""
        pred_lower = predicted.lower().strip()
        gt_lower = ground_truth.lower().strip()
        correct = pred_lower == gt_lower

        raw_metrics = RawMetrics(
            question_type="yesno",
            classification=ClassificationMetrics(
                correct=correct,
                predicted=pred_lower,
                ground_truth=gt_lower,
            ),
        )

        return ScoreResult(
            score=1.0 if correct else 0.0,
            correct=correct,
            method="exact_match",
            raw_metrics=raw_metrics,
        )

    def _evaluate_mcq(
        self,
        predicted: str,
        ground_truth: str,
        options: dict[str, str] | None = None,
    ) -> ScoreResult:
        """Evaluate single-choice MCQ."""
        pred_upper = predicted.upper().strip()
        gt_stripped = ground_truth.strip()
        opt_letters = {k.upper() for k in (options or {})}

        # Case 1: Ground truth is already an option letter.
        # The valid letters come from `options`, not a fixed A-E: MedXpertQA has
        # ten (A-J), and a hard-coded "ABCDE" silently dropped F-J to the text
        # branches below.
        if len(gt_stripped) == 1 and gt_stripped.upper() in (opt_letters or set("ABCDE")):
            correct = pred_upper == gt_stripped.upper()
            method = "letter_match"
        # Case 2: Ground truth is option TEXT that resolves to exactly one option.
        # Resolve it to its letter and compare letters exactly. This must be tried
        # BEFORE any text comparison: the previous order resolved the *prediction*
        # to text and then accepted a bidirectional SUBSTRING match, which credited
        # a prediction whose option text is contained in the gold text -- e.g.
        # option "C7" against gold "C7-C8", or "Oral prednisone" against "Oral
        # prednisone and tocilizumab". That is one-directional inflation: a strictly
        # less complete answer scored as correct.
        elif options and (
            gold_letters := [k for k, v in options.items()
                             if str(v).strip().lower() == gt_stripped.lower()]
        ):
            correct = len(gold_letters) == 1 and pred_upper == gold_letters[0].upper()
            method = "reverse_lookup"
        # Case 3: Gold text matches no option exactly (malformed or paraphrased
        # reference). Fall back to comparing the predicted option's text, exact
        # first and then substring, and label it so such items can be audited.
        elif options and pred_upper in options:
            predicted_text = options[pred_upper].strip()
            correct = predicted_text.lower() == gt_stripped.lower()
            if not correct:
                correct = (
                    gt_stripped.lower() in predicted_text.lower()
                    or predicted_text.lower() in gt_stripped.lower()
                )
            method = "text_substring_fallback" if correct else "text_mismatch"
        else:
            correct = pred_upper == gt_stripped.upper()
            method = "direct_compare"

        raw_metrics = RawMetrics(
            question_type="mcq",
            classification=ClassificationMetrics(
                correct=correct,
                predicted=pred_upper,
                ground_truth=gt_stripped,
            ),
        )

        return ScoreResult(
            score=1.0 if correct else 0.0,
            correct=correct,
            method=method,
            raw_metrics=raw_metrics,
        )

    # =========================================================================
    # Set-Based Types
    # =========================================================================

    def _evaluate_mcq_multi(
        self,
        predicted: str,
        ground_truth: str,
    ) -> ScoreResult:
        """Evaluate multi-select MCQ.

        Both predicted and ground_truth should be JSON arrays of letters.
        E.g., '["A", "C", "D"]'
        """
        # Parse predicted
        try:
            pred_letters = set(json.loads(predicted))
        except json.JSONDecodeError:
            # Fallback: extract letters from string
            pred_letters = set(re.findall(r'[A-E]', predicted.upper()))

        # Parse ground truth
        try:
            gt_letters = set(json.loads(ground_truth))
        except json.JSONDecodeError:
            gt_letters = set(re.findall(r'[A-E]', ground_truth.upper()))

        if not gt_letters:
            set_metrics = SetMetrics(
                precision=1.0, recall=1.0, f1=1.0,
                true_positives=0, pred_count=len(pred_letters), gt_count=0,
            )
            return ScoreResult(
                score=1.0,
                correct=True,
                method="empty_ground_truth",
                raw_metrics=RawMetrics(question_type="mcq_multi", set_metrics=set_metrics),
            )

        if not pred_letters:
            set_metrics = SetMetrics(
                precision=0.0, recall=0.0, f1=0.0,
                true_positives=0, pred_count=0, gt_count=len(gt_letters),
            )
            return ScoreResult(
                score=0.0,
                correct=False,
                method="empty_prediction",
                raw_metrics=RawMetrics(question_type="mcq_multi", set_metrics=set_metrics),
            )

        true_positives = len(pred_letters & gt_letters)
        precision = true_positives / len(pred_letters)
        recall = true_positives / len(gt_letters)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        set_metrics = SetMetrics(
            precision=precision,
            recall=recall,
            f1=f1,
            true_positives=true_positives,
            pred_count=len(pred_letters),
            gt_count=len(gt_letters),
        )

        # Correct if exact match or F1 >= 0.5
        correct = pred_letters == gt_letters or f1 >= 0.5

        return ScoreResult(
            score=f1,
            correct=correct,
            method="multi_select_f1",
            raw_metrics=RawMetrics(question_type="mcq_multi", set_metrics=set_metrics),
        )

    def _evaluate_list(
        self,
        predicted: str,
        ground_truth: str,
        thresholds: dict[str, float] | None = None,
    ) -> ScoreResult:
        """Evaluate list-type answers using F1 score with synonym-group-aware matching."""
        threshold = (thresholds or {}).get("list_f1", 0.3)

        pred_items = self._parse_pred_items(predicted)

        try:
            gt_groups = self._parse_gt_groups(json.loads(ground_truth))
        except json.JSONDecodeError:
            gt_groups = [[s.lower().strip()] for s in ground_truth.split(',') if s.strip()]

        gt_count = len(gt_groups)
        pred_count = len(pred_items)

        if not gt_groups:
            set_metrics = SetMetrics(
                precision=1.0, recall=1.0, f1=1.0,
                true_positives=0, pred_count=pred_count, gt_count=0,
            )
            return ScoreResult(
                score=1.0, correct=True, method="empty_ground_truth",
                raw_metrics=RawMetrics(question_type="list", set_metrics=set_metrics),
            )

        if not pred_items:
            set_metrics = SetMetrics(
                precision=0.0, recall=0.0, f1=0.0,
                true_positives=0, pred_count=0, gt_count=gt_count,
            )
            return ScoreResult(
                score=0.0, correct=False, method="empty_prediction",
                raw_metrics=RawMetrics(question_type="list", set_metrics=set_metrics),
            )

        true_positives = self._match_with_groups(pred_items, gt_groups)
        precision = true_positives / pred_count
        recall = true_positives / gt_count
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        set_metrics = SetMetrics(
            precision=precision, recall=recall, f1=f1,
            true_positives=true_positives, pred_count=pred_count, gt_count=gt_count,
        )
        return ScoreResult(
            score=f1, correct=f1 >= threshold, method="list_f1",
            raw_metrics=RawMetrics(question_type="list", set_metrics=set_metrics),
        )

    def _evaluate_expression(
        self,
        predicted: str,
        ground_truth: str,
        thresholds: dict[str, float] | None = None,
    ) -> ScoreResult:
        """Evaluate gene expression pattern answers.

        Ground truth format: {"tissue_list": ["liver", ...], "category": "..."}
        Predicted: JSON array or comma-separated tissues
        """
        threshold = (thresholds or {}).get("expression_f1", 0.3)

        # Parse ground truth
        try:
            gt_data = json.loads(ground_truth)
            gt_tissues = set(t.lower().strip() for t in gt_data.get('tissue_list', []))
        except json.JSONDecodeError:
            gt_tissues = set(t.lower().strip() for t in ground_truth.split(',') if t.strip())

        # Parse predicted
        try:
            pred_data = json.loads(predicted)
            if isinstance(pred_data, dict) and 'tissue_list' in pred_data:
                pred_tissues = set(t.lower().strip() for t in pred_data.get('tissue_list', []))
            elif isinstance(pred_data, list):
                pred_tissues = set(t.lower().strip() for t in pred_data if isinstance(t, str))
            else:
                pred_tissues = set()
        except json.JSONDecodeError:
            pred_tissues = set(t.lower().strip() for t in predicted.split(',') if t.strip())

        if not gt_tissues:
            set_metrics = SetMetrics(
                precision=1.0, recall=1.0, f1=1.0,
                true_positives=0, pred_count=len(pred_tissues), gt_count=0,
            )
            return ScoreResult(
                score=1.0,
                correct=True,
                method="empty_ground_truth",
                raw_metrics=RawMetrics(question_type="expression", set_metrics=set_metrics),
            )

        if not pred_tissues:
            set_metrics = SetMetrics(
                precision=0.0, recall=0.0, f1=0.0,
                true_positives=0, pred_count=0, gt_count=len(gt_tissues),
            )
            return ScoreResult(
                score=0.0,
                correct=False,
                method="empty_prediction",
                raw_metrics=RawMetrics(question_type="expression", set_metrics=set_metrics),
            )

        true_positives = len(pred_tissues & gt_tissues)
        precision = true_positives / len(pred_tissues)
        recall = true_positives / len(gt_tissues)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        set_metrics = SetMetrics(
            precision=precision,
            recall=recall,
            f1=f1,
            true_positives=true_positives,
            pred_count=len(pred_tissues),
            gt_count=len(gt_tissues),
        )

        return ScoreResult(
            score=f1,
            correct=f1 >= threshold,
            method="expression_f1",
            raw_metrics=RawMetrics(question_type="expression", set_metrics=set_metrics),
        )

    # =========================================================================
    # Generative Types (ROUGE-based)
    # =========================================================================

    def _evaluate_factoid(
        self,
        predicted: str,
        ground_truth: str,
        thresholds: dict[str, float] | None = None,
    ) -> ScoreResult:
        """Evaluate factoid answers using ROUGE scores.

        Changed from fuzzy matching to ROUGE-1/2/L per user requirement.
        """
        threshold = (thresholds or {}).get("factoid_rouge_l", 0.2)
        if not predicted or not ground_truth:
            rouge_metrics = RougeMetrics(
                rouge_1={"precision": 0.0, "recall": 0.0, "fmeasure": 0.0},
                rouge_2={"precision": 0.0, "recall": 0.0, "fmeasure": 0.0},
                rouge_l={"precision": 0.0, "recall": 0.0, "fmeasure": 0.0},
            )
            return ScoreResult(
                score=0.0,
                correct=False,
                method="empty_input",
                raw_metrics=RawMetrics(question_type="factoid", rouge=rouge_metrics),
            )

        # Normalize before scoring (e.g., "chromosome 8" → "chr8", "3" → "chr3")
        predicted = self._normalize_factoid(predicted)
        ground_truth = self._normalize_factoid(ground_truth)

        scores = self._rouge_scorer.score(ground_truth, predicted)

        rouge_metrics = RougeMetrics(
            rouge_1={
                "precision": scores['rouge1'].precision,
                "recall": scores['rouge1'].recall,
                "fmeasure": scores['rouge1'].fmeasure,
            },
            rouge_2={
                "precision": scores['rouge2'].precision,
                "recall": scores['rouge2'].recall,
                "fmeasure": scores['rouge2'].fmeasure,
            },
            rouge_l={
                "precision": scores['rougeL'].precision,
                "recall": scores['rougeL'].recall,
                "fmeasure": scores['rougeL'].fmeasure,
            },
        )

        # Use ROUGE-L F1 as primary score
        score = scores['rougeL'].fmeasure
        correct = score >= threshold

        # Semantic fallback: if ROUGE fails, check embedding similarity
        if not correct:
            try:
                sem_match = self._embedding_similarity(predicted.lower(), ground_truth.lower(), threshold=0.85)
                if sem_match:
                    correct = True
                    score = max(score, 0.5)  # Give partial credit
            except Exception:
                pass

        return ScoreResult(
            score=score,
            correct=correct,
            method="rouge_factoid",
            raw_metrics=RawMetrics(question_type="factoid", rouge=rouge_metrics),
        )

    def _evaluate_summary(
        self,
        predicted: str,
        ground_truth: str,
        thresholds: dict[str, float] | None = None,
    ) -> ScoreResult:
        """Evaluate summary answers using ROUGE-1/2/L."""
        threshold = (thresholds or {}).get("summary_rouge_l", 0.1)

        if not predicted or not ground_truth:
            rouge_metrics = RougeMetrics(
                rouge_1={"precision": 0.0, "recall": 0.0, "fmeasure": 0.0},
                rouge_2={"precision": 0.0, "recall": 0.0, "fmeasure": 0.0},
                rouge_l={"precision": 0.0, "recall": 0.0, "fmeasure": 0.0},
            )
            return ScoreResult(
                score=0.0,
                correct=False,
                method="empty_input",
                raw_metrics=RawMetrics(question_type="summary", rouge=rouge_metrics),
            )

        scores = self._rouge_scorer.score(ground_truth, predicted)

        rouge_metrics = RougeMetrics(
            rouge_1={
                "precision": scores['rouge1'].precision,
                "recall": scores['rouge1'].recall,
                "fmeasure": scores['rouge1'].fmeasure,
            },
            rouge_2={
                "precision": scores['rouge2'].precision,
                "recall": scores['rouge2'].recall,
                "fmeasure": scores['rouge2'].fmeasure,
            },
            rouge_l={
                "precision": scores['rougeL'].precision,
                "recall": scores['rougeL'].recall,
                "fmeasure": scores['rougeL'].fmeasure,
            },
        )

        score = scores['rougeL'].fmeasure
        correct = score >= threshold

        return ScoreResult(
            score=score,
            correct=correct,
            method="rouge_summary",
            raw_metrics=RawMetrics(question_type="summary", rouge=rouge_metrics),
        )

    # =========================================================================
    # Utilities
    # =========================================================================

    @staticmethod
    def _parse_gt_groups(gt_data) -> list[list[str]]:
        """Parse BioASQ ground truth into synonym groups.

        BioASQ format: [["syn1a","syn1b"], ["syn2a"]] = 2 groups.
        Matching any synonym in a group counts as matching that group.
        Recall denominator = number of groups, not total synonym count.
        """
        groups = []
        for item in (gt_data if isinstance(gt_data, list) else []):
            if isinstance(item, list):
                syns = [str(s).lower().strip() for s in item if str(s).strip()]
                if syns:
                    groups.append(syns)
            else:
                s = str(item).lower().strip()
                if s:
                    groups.append([s])
        return groups

    @staticmethod
    def _parse_pred_items(text: str) -> list[str]:
        """Parse predicted text into a list of items."""
        text = text.strip()
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return [str(x).lower().strip() for x in data if str(x).strip()]
            if isinstance(data, str):
                # JSON parsed as single string (e.g. from FINAL("a, b, c"))
                return [s.strip() for s in data.lower().split(',') if s.strip()]
        except json.JSONDecodeError:
            pass
        # Fallback: comma split with bracket/quote cleanup
        items = []
        for s in text.split(','):
            s = s.strip().strip('"\'[]').strip()
            if s.startswith('and '):
                s = s[4:].strip()
            if s:
                items.append(s.lower())
        return items

    @staticmethod
    def _match_with_groups(
        pred_items: list[str],
        gt_groups: list[list[str]],
        embedding_threshold: float = 0.80,
    ) -> int:
        """Count how many GT groups are matched by predicted items.

        Three-pass matching:
        Pass 1: Exact match (pred ∈ group synonyms)
        Pass 2: Substring containment (pred ⊂ synonym or synonym ⊂ pred)
        Pass 3: Embedding cosine similarity (only for unmatched residual)
        """
        matched_groups: set[int] = set()
        matched_preds: set[int] = set()

        # Pass 1: exact match
        for pi, p in enumerate(pred_items):
            if pi in matched_preds:
                continue
            for gi, group in enumerate(gt_groups):
                if gi in matched_groups:
                    continue
                if p in group:
                    matched_groups.add(gi)
                    matched_preds.add(pi)
                    break

        # Pass 2: substring containment
        for pi, p in enumerate(pred_items):
            if pi in matched_preds:
                continue
            for gi, group in enumerate(gt_groups):
                if gi in matched_groups:
                    continue
                for syn in group:
                    if p in syn or syn in p:
                        matched_groups.add(gi)
                        matched_preds.add(pi)
                        break
                if pi in matched_preds:
                    break

        # Pass 3: embedding similarity (only for unmatched residual)
        if len(matched_groups) < len(gt_groups):
            unmatched_preds = [(pi, pred_items[pi]) for pi in range(len(pred_items)) if pi not in matched_preds]
            unmatched_groups = [(gi, gt_groups[gi]) for gi in range(len(gt_groups)) if gi not in matched_groups]

            if unmatched_preds and unmatched_groups:
                new_matches = MetricsEvaluator._embedding_match_groups(
                    unmatched_preds, unmatched_groups, embedding_threshold,
                )
                matched_groups.update(new_matches)

        return len(matched_groups)

    @staticmethod
    def _embedding_match_groups(
        unmatched_preds: list[tuple[int, str]],
        unmatched_groups: list[tuple[int, list[str]]],
        threshold: float,
    ) -> set[int]:
        """Match remaining pred items to GT groups via embedding cosine similarity."""
        import numpy as np

        # Collect all texts: pred items + first synonym of each group
        pred_texts = [text for _, text in unmatched_preds]
        gt_texts = [group[0] for _, group in unmatched_groups]
        all_texts = pred_texts + gt_texts

        try:
            import asyncio
            import sys
            sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[4] / 'src'))
            from utils.clients import embed_client

            async def _embed():
                return await embed_client.embed(all_texts)

            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        embeddings = pool.submit(asyncio.run, _embed()).result()
                else:
                    embeddings = asyncio.run(_embed())
            except RuntimeError:
                embeddings = asyncio.run(_embed())

        except Exception:
            return set()

        embs = np.array(embeddings)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        embs = embs / norms

        n_pred = len(pred_texts)
        sim = embs[:n_pred] @ embs[n_pred:].T

        # Greedy matching
        new_matches: set[int] = set()
        used_gt_idx: set[int] = set()
        for pi in range(n_pred):
            best_j = -1
            best_score = threshold
            for gj in range(len(gt_texts)):
                if gj in used_gt_idx:
                    continue
                if sim[pi][gj] > best_score:
                    best_score = sim[pi][gj]
                    best_j = gj
            if best_j >= 0:
                new_matches.add(unmatched_groups[best_j][0])  # original group index
                used_gt_idx.add(best_j)

        return new_matches

    @staticmethod
    @staticmethod
    def _normalize_factoid(text: str) -> str:
        """Normalize factoid answers for fairer comparison.

        Handles chromosome format variations:
          "chromosome 8", "Chromosome 8", "8" → "chr8"
          "8q13.1" → "chr8" (strip cytoband)
        """
        import re as _re
        t = text.strip().lower()
        # "chromosome 8" / "chromosome8" → "chr8"
        m = _re.match(r'^chromosome\s*(\d+|[xy])$', t)
        if m:
            return f'chr{m.group(1)}'
        # Bare number that looks like chromosome: "8", "21", "X"
        m = _re.match(r'^(\d{1,2}|[xy])$', t)
        if m:
            return f'chr{m.group(1)}'
        # Cytoband "8q13.1" → "chr8"
        m = _re.match(r'^(\d{1,2}|[xy])[pq]\d', t)
        if m:
            return f'chr{m.group(1)}'
        return text

    @staticmethod
    def _embedding_similarity(text_a: str, text_b: str, threshold: float = 0.85) -> bool:
        """Check if two texts are semantically similar via embedding cosine."""
        import numpy as np

        try:
            import asyncio
            import sys
            sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[4] / 'src'))
            from utils.clients import embed_client

            async def _embed():
                return await embed_client.embed([text_a, text_b])

            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        embeddings = pool.submit(asyncio.run, _embed()).result()
                else:
                    embeddings = asyncio.run(_embed())
            except RuntimeError:
                embeddings = asyncio.run(_embed())

        except Exception:
            return False

        embs = np.array(embeddings)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        embs = embs / norms
        return float(embs[0] @ embs[1]) >= threshold


# =============================================================================
# Aggregation Functions
# =============================================================================

def aggregate_subtask_results(
    results: list[ScoreResult],
    question_type: str,
) -> dict[str, float]:
    """Aggregate raw metrics for a subtask.

    Args:
        results: List of ScoreResult for this subtask
        question_type: Type of questions

    Returns:
        Aggregated metrics dict
    """
    if not results:
        return {}

    # Classification types: compute accuracy
    if question_type in ("yesno", "mcq"):
        correct_count = sum(1 for r in results if r.correct)
        return {
            "accuracy": correct_count / len(results),
            "correct": correct_count,
            "total": len(results),
        }

    # Set-based types: compute macro-average P/R/F1
    if question_type in ("list", "mcq_multi", "expression"):
        precisions = []
        recalls = []
        f1s = []
        for r in results:
            if r.raw_metrics.set_metrics:
                precisions.append(r.raw_metrics.set_metrics.precision)
                recalls.append(r.raw_metrics.set_metrics.recall)
                f1s.append(r.raw_metrics.set_metrics.f1)

        if not f1s:
            return {}

        return {
            "precision": sum(precisions) / len(precisions),
            "recall": sum(recalls) / len(recalls),
            "f1": sum(f1s) / len(f1s),
            "correct": sum(1 for r in results if r.correct),
            "total": len(results),
        }

    # Generative types: compute average ROUGE scores
    if question_type in ("summary", "factoid"):
        rouge1_f = []
        rouge2_f = []
        rougel_f = []
        for r in results:
            if r.raw_metrics.rouge:
                rouge1_f.append(r.raw_metrics.rouge.rouge_1.get("fmeasure", 0))
                rouge2_f.append(r.raw_metrics.rouge.rouge_2.get("fmeasure", 0))
                rougel_f.append(r.raw_metrics.rouge.rouge_l.get("fmeasure", 0))

        if not rougel_f:
            return {}

        return {
            "rouge_1": sum(rouge1_f) / len(rouge1_f),
            "rouge_2": sum(rouge2_f) / len(rouge2_f),
            "rouge_l": sum(rougel_f) / len(rougel_f),
            "correct": sum(1 for r in results if r.correct),
            "total": len(results),
        }

    return {}


def get_subtask_score(
    aggregated: dict[str, float],
    question_type: str,
) -> float:
    """Get primary score for a subtask from aggregated metrics.

    Args:
        aggregated: Aggregated metrics dict
        question_type: Type of questions

    Returns:
        Primary score (0.0 - 1.0)
    """
    if question_type in ("yesno", "mcq"):
        return aggregated.get("accuracy", 0.0)
    if question_type in ("list", "mcq_multi", "expression"):
        return aggregated.get("f1", 0.0)
    if question_type in ("summary", "factoid"):
        return aggregated.get("rouge_l", 0.0)
    return 0.0
