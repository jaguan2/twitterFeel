"""Local, independently trained emotion and sentiment classifiers.

Loading never downloads models or sends posts to an external service. Scores are
model support, not emotional intensity or established annotation accuracy.
"""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
import math
from pathlib import Path
import threading

from emotion_schema import file_digest
from label_schema import LABELS, SCHEMA_VERSION, interpret, text_hash, validate_thresholds

MODELS = {
    "emotion": {"id": "SamLowe/roberta-base-go_emotions", "directory": "roberta-go-emotions"},
    "sentiment": {"id": "cardiffnlp/twitter-roberta-base-sentiment-latest", "directory": "twitter-sentiment"},
}
EMOTION_MODELS = {
    "roberta": {**MODELS["emotion"], "name": "RoBERTa · current baseline"},
    "modernbert": {"id": "cirimus/modernbert-base-go-emotions", "directory": "modernbert-go-emotions",
                   "name": "ModernBERT · research candidate",
                   "revision": "cb5ed17bc26c8cd1f017fabe700a5d723f825287"},
}
SENTIMENT_LABELS = ("negative", "neutral", "positive")
MAX_LENGTH = 256
SENTIMENT_TRANSFORM = "cardiff-mention-url-placeholders-v1"


def sentiment_text(text):
    """The upstream model card's whitespace/mention/link preprocessing."""
    tokens = []
    for token in text.split(" "):
        if token.startswith("@") and len(token) > 1:
            token = "@user"
        if token.startswith("http"):
            token = "http"
        tokens.append(token)
    return " ".join(tokens)


def validate_tasks(tasks):
    if tasks is None:
        return ("emotion", "sentiment")
    if (not isinstance(tasks, (list, tuple)) or not tasks
            or any(not isinstance(task, str) for task in tasks) or len(set(tasks)) != len(tasks)):
        raise ValueError("Tasks must contain emotion and optionally sentiment")
    if "emotion" not in tasks or not set(tasks).issubset(MODELS):
        raise ValueError("Tasks must contain emotion and optionally sentiment")
    return tuple(task for task in MODELS if task in tasks)


class AnalysisEngine:
    def __init__(self, root, emotion_model="roberta"):
        if emotion_model not in EMOTION_MODELS:
            raise ValueError("Unknown emotion model")
        self.root = Path(root).resolve()
        self.emotion_model = emotion_model
        self._models = {}
        self._lock = threading.RLock()
        self._hash_cache = {}

    @property
    def emotion_model_id(self):
        return EMOTION_MODELS[self.emotion_model]["id"]

    def _spec(self, task):
        return EMOTION_MODELS[self.emotion_model] if task == "emotion" else MODELS[task]

    def _path(self, task):
        spec = self._spec(task)
        local = self.root / "runs/model_cache" / spec["directory"]
        if (local / "model.safetensors").is_file() and (local / "config.json").is_file():
            if spec.get("revision"):
                metadata = local / ".cache/huggingface/download/config.json.metadata"
                if not metadata.is_file() or metadata.read_text(encoding="utf-8").splitlines()[:1] != [spec["revision"]]:
                    return None
            return local
        # Keep support for the existing emotion download's standard HF cache.
        if task == "emotion" and self.emotion_model == "roberta":
            from huggingface_hub import try_to_load_from_cache
            cached = try_to_load_from_cache(MODELS[task]["id"], "model.safetensors")
            if isinstance(cached, str) and (Path(cached).parent / "config.json").is_file():
                return Path(cached).parent
        return None

    def available_models(self):
        return {task: self._path(task) is not None for task in MODELS}

    def _load(self, task):
        if task not in self._models:
            path = self._path(task)
            if path is None:
                raise ValueError(f"Local {task} model missing. Install {self._spec(task)['id']} "
                                 f"in runs/model_cache/{self._spec(task)['directory']} using the model setup instructions.")
            import torch
            from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
            torch.set_num_threads(min(4, torch.get_num_threads()))
            tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
            options = {"local_files_only": True, "use_safetensors": True, "trust_remote_code": False}
            if task == "emotion" and self.emotion_model == "modernbert":
                config = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=False)
                config.reference_compile = False
                options.update(config=config, attn_implementation="sdpa")
            model = AutoModelForSequenceClassification.from_pretrained(path, **options)
            names = {int(key): value for key, value in model.config.id2label.items()}
            if task == "emotion":
                if set(names) != set(range(len(LABELS))) or set(names.values()) != set(LABELS):
                    raise ValueError("Local emotion model has an incompatible label vocabulary")
                if model.config.problem_type != "multi_label_classification":
                    raise ValueError("Local emotion model must be a multi-label classifier")
            elif names != dict(enumerate(SENTIMENT_LABELS)):
                raise ValueError("Local sentiment model must map 0/1/2 to negative/neutral/positive")
            model.eval()
            self._models[task] = tokenizer, model, names
        return self._models[task]

    def _score(self, task, text):
        import torch
        tokenizer, model, names = self._load(task)
        truncated = len(tokenizer(text, truncation=False)["input_ids"]) > MAX_LENGTH
        tokens = tokenizer(text, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        with torch.inference_mode():
            logits = model(**tokens).logits
            values = logits.sigmoid() if task == "emotion" else logits.softmax(dim=-1)
            values = values.cpu().tolist()[0]
        if len(values) != len(names) or not all(math.isfinite(x) and 0 <= x <= 1 for x in values):
            raise ValueError(f"The {task} model returned invalid scores")
        return {names[index]: float(value) for index, value in enumerate(values)}, truncated

    def infer(self, text, threshold=.5, tasks=None, thresholds=None):
        tasks = validate_tasks(tasks)
        if isinstance(threshold, bool):
            raise ValueError("Threshold must be a finite number between zero and one")
        try:
            threshold = float(threshold)
        except (TypeError, ValueError):
            raise ValueError("Threshold must be a finite number between zero and one") from None
        if not math.isfinite(threshold) or not 0 < threshold < 1:
            raise ValueError("Threshold must be a finite number between zero and one")
        if thresholds is not None:
            thresholds = validate_thresholds(thresholds)
        if not isinstance(text, str):
            raise ValueError("Post text must be a string")
        result = {"text": text, "text_sha256": text_hash(text), "labels": [],
                  "status": "invalid", "scores": {}, "truncated": False,
                  "review_reason": "blank_text", "related_groups": [], "sentiment": None,
                  "tasks": list(tasks), "emotion_requested": True,
                  "sentiment_requested": "sentiment" in tasks}
        if not text.strip():
            return result
        with self._lock:
            scores, truncated = self._score("emotion", text)
            result.update(scores=scores, truncated=truncated, **interpret(scores, threshold, thresholds))
            if truncated:
                result["status"] = "review_needed"
                result["review_reason"] = ";".join(filter(None, [result["review_reason"], "text_truncated"]))
            if "sentiment" in tasks:
                scored_text = sentiment_text(text)
                scores, truncated = self._score("sentiment", scored_text)
                result["sentiment"] = {
                    "label": max(scores, key=scores.get), "scores": scores,
                    "truncated": truncated, "model": MODELS["sentiment"]["id"],
                    "scored_text": scored_text, "scored_text_sha256": text_hash(scored_text),
                    "preprocessing": SENTIMENT_TRANSFORM,
                }
        return result

    def _digest(self, path):
        stat = path.stat()
        key = (str(path), stat.st_mtime_ns, stat.st_size)
        if key not in self._hash_cache:
            self._hash_cache[key] = file_digest(path)
        return self._hash_cache[key]

    def provenance(self, tasks=None):
        tasks = validate_tasks(tasks)
        models = {}
        with self._lock:
            for task in tasks:
                path = self._path(task)
                if path is None:
                    raise ValueError(f"Local {task} model is missing; install {self._spec(task)['id']} first")
                files = {p.name: self._digest(p) for p in sorted(path.iterdir())
                         if p.is_file() and ((task == "emotion" and self.emotion_model == "modernbert") or
                                            p.suffix in (".json", ".txt", ".safetensors", ".bin"))}
                metadata = path / ".cache/huggingface/download/config.json.metadata"
                revision = metadata.read_text(encoding="utf-8").splitlines()[0] if metadata.is_file() else None
                if revision is None and path.parent.name == "snapshots":
                    revision = path.name
                models[task] = {"id": self._spec(task)["id"], "revision": revision,
                                "files_sha256": files, "runtime_format": "safetensors"}
            packages = {}
            for name in ("torch", "transformers", "tokenizers", "safetensors", "numpy"):
                try:
                    packages[name] = version(name)
                except PackageNotFoundError:
                    packages[name] = "not installed"
        return {"models": models, "packages": packages, "task_configs": {
            "emotion": {"schema": SCHEMA_VERSION, "activation": "sigmoid", "max_length": MAX_LENGTH,
                        "preprocessing": "original_text", "batch_size": 1,
                        **({"attention": "sdpa", "reference_compile": False}
                           if self.emotion_model == "modernbert" else {})},
            **({"sentiment": {"labels": list(SENTIMENT_LABELS), "activation": "softmax",
                               "max_length": MAX_LENGTH, "preprocessing": SENTIMENT_TRANSFORM}}
               if "sentiment" in tasks else {}),
        }, "inference": "local_cpu_only"}
