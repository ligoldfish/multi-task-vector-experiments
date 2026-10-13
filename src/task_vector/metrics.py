"""Task metrics with original units, cached sample evidence, and corpus rescoring."""
from __future__ import annotations

from collections import Counter
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from functools import lru_cache
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import unicodedata

from .code_execution import execute_code
from .prompts import canonical_task, unwrap_prediction

PRIMARY_METRICS = {"math": "accuracy", "logic": "accuracy", "translation": "chrF++", "paraphrase": "iBLEU", "code_generation": "pass_all", "code_repair": "pass_all", "summary": "rougeL", "joint": "joint_success"}


def _settings(config: dict, task: str) -> dict:
    settings = config.get("metrics", {})
    return {**settings, **settings.get(task, {})}


def _numeric(text: str) -> Fraction | None:
    text = text.strip().replace(",", "").replace("$", "").replace("−", "-")
    text = text.strip(" \\{}").rstrip(".")
    # GSM8K's official extractor scores the printed number, excluding a '%' unit.
    text = text.rstrip("%")
    try:
        if "/" in text:
            numerator, denominator = text.split("/", 1)
            return Fraction(Decimal(numerator.strip())) / Fraction(Decimal(denominator.strip()))
        return Fraction(Decimal(text))
    except (ValueError, ZeroDivisionError, InvalidOperation):
        return None


def extract_math_answer(text: str) -> Fraction | None:
    text = text.replace("−", "-")
    text = re.sub(r"\\(?:dfrac|tfrac|frac)\s*\{\s*([-+]?\d+(?:\.\d+)?)\s*\}\s*\{\s*([-+]?\d+(?:\.\d+)?)\s*\}", r"\1/\2", text)
    number = r"[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?(?:\s*/\s*[-+]?(?:\d+(?:\.\d+)?|\.\d+))?%?"
    if "####" in text:
        text = text.rsplit("####", 1)[-1]
        match = re.search(number, text)
        return _numeric(match.group()) if match else None
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        value = _numeric(boxed[-1])
        if value is not None:
            return value
    numbers = re.findall(number, text)
    return _numeric(numbers[-1]) if numbers else None


def extract_logic_answer(text: str) -> str | None:
    match = re.fullmatch(r"\s*(true|false|unknown)\s*[.!]?\s*", text, re.I)
    return match.group(1).capitalize() if match else None


def _normal_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


@lru_cache(maxsize=8)
def _semantic_file_hashes(path: str, state: tuple) -> dict:
    hashes = {}
    for name, size, modified in state:
        digest = hashlib.sha256()
        with (Path(path) / name).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        hashes[name] = digest.hexdigest()
    try:
        version = importlib.metadata.version("sentence-transformers")
    except importlib.metadata.PackageNotFoundError:
        version = None  # Hash inspection is valid before installing the inference dependency.
    return {"path": str(Path(path).resolve()), "files": hashes,
            "sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
            "sentence_transformers_version": version,
            "embedding_similarity": "L2_normalized_dot_product", "device": "cpu"}


@lru_cache(maxsize=4)
def _semantic_encoder(path: str, fingerprint: str):
    if not Path(path).exists():
        raise FileNotFoundError(f"Semantic-preservation model must be an available offline asset: {path}")
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(path, device="cpu", local_files_only=True)


def semantic_asset_metadata(config: dict) -> dict:
    """Content provenance for a required offline semantic metric asset.

    This performs no embedding inference. File stat tuples only invalidate the
    content-hash cache; the reported fingerprint itself is built from SHA256s.
    """
    settings = _settings(config, "paraphrase")
    path = settings.get("semantic_model_path") or os.environ.get("TV_SEMANTIC_MODEL_PATH") or os.environ.get("TV_SEMANTIC_MODEL")
    if not path and settings.get("semantic_model") and Path(str(settings["semantic_model"])).exists():
        path = settings["semantic_model"]
    if not path:
        raise ValueError("QQP semantic-preservation metric requires metrics.paraphrase.semantic_model_path / TV_SEMANTIC_MODEL; it cannot be omitted")
    directory = Path(path)
    if not directory.is_dir():
        raise FileNotFoundError(f"Semantic-preservation model must be an available offline directory: {path}")
    state = tuple((file.relative_to(directory).as_posix(), file.stat().st_size, file.stat().st_mtime_ns) for file in sorted(directory.rglob("*")) if file.is_file() and ".git" not in file.parts)
    if not state:
        raise ValueError("Semantic metric asset directory is empty")
    return _semantic_file_hashes(str(directory), state)


def semantic_asset_fingerprint(config: dict) -> str:
    return semantic_asset_metadata(config)["sha256"]


def _semantic_scores(rows: list[dict], config: dict) -> list[tuple[float, float]]:
    settings = _settings(config, "paraphrase")
    provenance = semantic_asset_metadata(config)
    encoder = _semantic_encoder(provenance["path"], provenance["sha256"])
    texts = [str(value) for row in rows for value in (row.get("metadata", {}).get("source_question", row["prompt"]), row["_metric_prediction"], row["target"])]
    embeddings = encoder.encode(texts, normalize_embeddings=True, convert_to_numpy=True, batch_size=int(settings.get("semantic_batch_size", 32)), show_progress_bar=False)
    for row in rows:
        row.setdefault("metric_provenance", {})["semantic_model"] = provenance
    return [(float(embeddings[3 * i] @ embeddings[3 * i + 1]), float(embeddings[3 * i + 2] @ embeddings[3 * i + 1])) for i in range(len(rows))]


def _text_metrics(config: dict, task: str):
    from sacrebleu.metrics import BLEU, CHRF
    settings = _settings(config, task)
    bleu = BLEU(tokenize=settings.get("bleu_tokenize", "13a"), lowercase=bool(settings.get("lowercase", False)), smooth_method=settings.get("bleu_smoothing", "exp"), effective_order=False)
    chrf = CHRF(char_order=6, word_order=2, beta=2, lowercase=bool(settings.get("lowercase", False)), eps_smoothing=False)
    return bleu, chrf


def _cache_sample_scores(task: str, rows: list[dict], config: dict):
    missing = [row for row in rows if PRIMARY_METRICS[task] not in row.get("scores", {})]
    semantic_missing = [row for row in rows if task == "paraphrase" and not {"semantic_source", "semantic_reference"} <= row.get("scores", {}).keys()]
    if task in {"translation", "paraphrase"}:
        bleu, chrf = _text_metrics(config, task)
    if task == "summary":
        from rouge_score.rouge_scorer import RougeScorer
        rouge = RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=bool(_settings(config, task).get("use_stemmer", True)))
    for row in missing:
        prediction, target = row["_metric_prediction"], row["target"]
        scores = dict(row.get("scores", {}))
        if task == "math":
            pred, gold = extract_math_answer(prediction), extract_math_answer(target)
            if gold is None:
                raise ValueError(f"Unparseable GSM8K gold answer: {row.get('id')}")
            scores["accuracy"] = float(pred is not None and pred == gold)
            row["answer_extraction"] = {"prediction": str(pred) if pred is not None else None, "target": str(gold)}
        elif task == "logic":
            pred = extract_logic_answer(prediction)
            gold = row.get("metadata", {}).get("label", target)
            if gold not in {"True", "False", "Unknown"}:
                raise ValueError(f"Invalid gold logic label {gold!r}")
            scores["accuracy"] = float(pred == gold)
            row["answer_extraction"] = {"prediction": pred, "target": gold}
        elif task == "translation":
            scores["chrF++"] = chrf.sentence_score(prediction, [target]).score
            scores["BLEU"] = bleu.sentence_score(prediction, [target]).score
        elif task == "paraphrase":
            source = row.get("metadata", {}).get("source_question", row["prompt"])
            alpha = float(_settings(config, task).get("ibleu_alpha", 0.9))
            if not 0 <= alpha <= 1:
                raise ValueError("iBLEU alpha must be in [0,1]")
            # effective-order sentence BLEU is explicit and used ONLY for scale/thresholds.
            from sacrebleu.metrics import BLEU
            sentence_bleu = BLEU(tokenize=_settings(config, task).get("bleu_tokenize", "13a"), smooth_method="exp", effective_order=True)
            ref_bleu = sentence_bleu.sentence_score(prediction, [target]).score
            source_bleu = sentence_bleu.sentence_score(prediction, [source]).score
            scores.update({"iBLEU": alpha * ref_bleu - (1 - alpha) * source_bleu, "reference_BLEU": ref_bleu, "source_BLEU": source_bleu,
                           "copy_rate": float(_normal_text(prediction) == _normal_text(source))})
            source_tokens, pred_tokens = _normal_text(source).split(), _normal_text(prediction).split()
            common = sum((Counter(source_tokens) & Counter(pred_tokens)).values())
            scores["copied_token_fraction"] = common / len(pred_tokens) if pred_tokens else 0.0
        elif task in {"code_generation", "code_repair"}:
            meta = row.get("metadata", {})
            if "test_list" not in meta:
                raise ValueError("MBPP row lacks official test_list")
            evidence = row.get("code_execution") or execute_code(prediction, meta["test_list"], meta.get("test_setup_code", ""), config)
            row["code_execution"] = evidence
            scores["pass_all"] = float(evidence["pass_all"])
        elif task == "summary":
            scores.update({key: value.fmeasure * 100 for key, value in rouge.score(target, prediction).items()})
        row["scores"] = scores
    if semantic_missing:
        for row, (source_score, reference_score) in zip(semantic_missing, _semantic_scores(semantic_missing, config)):
            row.setdefault("scores", {}).update({"semantic_source": source_score, "semantic_reference": reference_score})


def aggregate_predictions(task: str, rows: list[dict], config: dict | None = None) -> dict:
    """Aggregate cached rows; no code execution or embedding inference occurs here.

    Corpus BLEU/chrF/iBLEU are recomputed on the resampled paired rows. Their
    sentence scores are retained only for the baseline std and joint thresholds.
    """
    config = config or {}
    task = canonical_task(task)
    if task == "joint":
        from .joint import aggregate_joint_predictions
        return aggregate_joint_predictions(rows, config)
    if not rows:
        raise ValueError("Cannot score an empty evaluation panel")
    primary = PRIMARY_METRICS[task]
    if any(primary not in row.get("scores", {}) for row in rows):
        raise ValueError("aggregate_predictions requires cached rows from score_predictions")
    keys = set.intersection(*(set(row["scores"]) for row in rows))
    metrics = {key: sum(float(row["scores"][key]) for row in rows) / len(rows) for key in sorted(keys)}
    signatures = {}
    if task == "math":
        signatures["answer_extraction"] = "prefer_final_hash_marker_then_boxed_then_last_number; exact_rational; ignore_commas_currency_percent_units"
    elif task == "logic":
        signatures["answer_extraction"] = "one_complete_True_False_Unknown_label_case_insensitive_optional_terminal_punctuation"
    elif task == "summary":
        signatures["ROUGE"] = {
            "implementation": "rouge_score.rouge_scorer.RougeScorer",
            "version": importlib.metadata.version("rouge-score"),
            "metrics": ["rouge1", "rouge2", "rougeL"],
            "use_stemmer": bool(_settings(config, task).get("use_stemmer", True)),
            "statistic": "F1", "scale": 100, "aggregation": "mean_sample_F1",
        }
    if task in {"translation", "paraphrase"}:
        bleu, chrf = _text_metrics(config, task)
        predictions = [row.get("_metric_prediction", unwrap_prediction(row["prediction"], row.get("variant", "canonical"), task)) for row in rows]
        targets = [row["target"] for row in rows]
        reference_bleu = bleu.corpus_score(predictions, [targets]).score
        signatures["BLEU"] = str(bleu.get_signature())
        if task == "translation":
            metrics["BLEU"] = reference_bleu
            metrics["chrF++"] = chrf.corpus_score(predictions, [targets]).score
            signatures["chrF++"] = str(chrf.get_signature())
        else:
            sources = [row.get("metadata", {}).get("source_question", row["prompt"]) for row in rows]
            source_bleu = bleu.corpus_score(predictions, [sources]).score
            alpha = float(_settings(config, task).get("ibleu_alpha", 0.9))
            metrics.update({"iBLEU": alpha * reference_bleu - (1 - alpha) * source_bleu, "reference_BLEU": reference_bleu, "source_BLEU": source_bleu})
            signatures["iBLEU"] = {"alpha": alpha, "reference_and_copy": signatures["BLEU"], "aggregation": "corpus_BLEU_not_mean_sentence_BLEU"}
            semantic_provenance = rows[0].get("metric_provenance", {}).get("semantic_model")
            if semantic_provenance is not None:
                signatures["semantic_model"] = semantic_provenance
    units = {key: "fraction" if key in {"accuracy", "pass_all", "copy_rate", "copied_token_fraction"} else "cosine" if key.startswith("semantic_") else "0-100" for key in metrics}
    if "iBLEU" in units:
        units["iBLEU"] = "BLEU_points; negative values possible"
    return {"metrics": metrics, "primary_metric": primary, "sample_scores": [float(row["scores"][primary]) for row in rows], "rows": rows, "units": units, "signatures": signatures, "count": len(rows),
            "truncation_rate": sum(bool(row.get("truncated", False)) for row in rows) / len(rows)}


class PreparedBootstrapMetrics:
    """Exact corpus rescoring from SacreBLEU's cached sufficient statistics.

    The supported SacreBLEU version is pinned because its private sufficient-
    statistic API is used. Its official corpus computation still determines each
    replicate's BLEU/chrF score. Python's original row-order summation is retained
    for cached scalar metrics, including all semantic/copy/ROUGE metrics.
    """

    def __init__(self, task: str, rows: list[dict], config: dict):
        self.task = canonical_task(task)
        self.rows = rows
        self.config = config
        # Also validates nested joint subtask evidence and cached score fields.
        reference = aggregate_predictions(self.task, rows, config)["metrics"]
        self.names = list(reference)
        self.cached_names = [name for name in self.names
                             if all(name in row["scores"] for row in rows)]
        self.scalar_values = {
            name: tuple(row["scores"][name] if self.task == "joint" else float(row["scores"][name])
                        for row in rows)
            for name in self.cached_names
        }
        self.statistics = {}
        self.bleu = self.chrf = None
        if self.task in {"translation", "paraphrase"}:
            import numpy as np
            if importlib.metadata.version("sacrebleu") != "2.5.1":
                raise RuntimeError("Cached corpus bootstrap requires pinned sacrebleu==2.5.1")
            self.bleu, self.chrf = _text_metrics(config, self.task)
            predictions = [row.get("_metric_prediction", unwrap_prediction(row["prediction"], row.get("variant", "canonical"), self.task)) for row in rows]
            targets = [row["target"] for row in rows]
            def cache(metric, references):
                values = metric._extract_corpus_statistics(predictions, [references])
                if any(not isinstance(value, int) for vector in values for value in vector):
                    raise RuntimeError("Expected exact integer corpus sufficient statistics")
                return np.asarray(values, dtype=np.int64)
            self.statistics["reference_BLEU"] = cache(self.bleu, targets)
            if self.task == "translation":
                self.statistics["chrF++"] = cache(self.chrf, targets)
            else:
                sources = [row.get("metadata", {}).get("source_question", row["prompt"]) for row in rows]
                self.statistics["source_BLEU"] = cache(self.bleu, sources)
        actual = self.score(list(range(len(rows))))
        if actual != reference:
            raise RuntimeError("Cached bootstrap point metrics differ from official uncached corpus metrics")

    def score(self, indices: list[int]) -> dict:
        if not indices:
            raise ValueError("Cannot score an empty bootstrap replicate")
        if self.task == "joint":
            # paired_bootstrap consumes only aggregate['metrics']; per_task and
            # per_pair panels are separately reported, not part of this delta.
            return {name: sum(self.scalar_values[name][index] for index in indices) / len(indices)
                    for name in self.names}
        values = {name: sum(self.scalar_values[name][index] for index in indices) / len(indices)
                  for name in self.cached_names}
        if self.statistics:
            # Integer sums are order independent. Convert back to Python ints
            # before invoking the same official score function as corpus_score.
            def corpus(metric, key):
                stats = self.statistics[key][indices].sum(axis=0).tolist()
                return metric._compute_score_from_stats(stats).score
            reference = corpus(self.bleu, "reference_BLEU")
            if self.task == "translation":
                values["BLEU"] = reference
                values["chrF++"] = corpus(self.chrf, "chrF++")
            else:
                source = corpus(self.bleu, "source_BLEU")
                alpha = float(_settings(self.config, self.task).get("ibleu_alpha", .9))
                values.update({"reference_BLEU": reference, "source_BLEU": source,
                               "iBLEU": alpha * reference - (1 - alpha) * source})
        return values


def score_predictions(task: str, rows: list[dict], config: dict | None = None) -> dict:
    config = config or {}
    task = canonical_task(task)
    if task == "joint":
        from .joint import score_joint_predictions
        return score_joint_predictions(rows, config)
    prepared = [{**row, "metadata": dict(row.get("metadata", {})), "scores": dict(row.get("scores", {}))} for row in rows]
    for row in prepared:
        row["_metric_prediction"] = unwrap_prediction(str(row["prediction"]), row.get("variant", "canonical"), task)
    _cache_sample_scores(task, prepared, config)
    return aggregate_predictions(task, prepared, config)
