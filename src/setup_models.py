"""Install pinned local classifier files; reuse complete matching caches offline."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
PINS = {
    "emotion": {"id": "SamLowe/roberta-base-go_emotions", "directory": "roberta-go-emotions",
                "revision": "d75048347613a25d77de8cf6412eaae9fa7b26be", "weight": "model.safetensors"},
    "sentiment": {"id": "cardiffnlp/twitter-roberta-base-sentiment-latest", "directory": "twitter-sentiment",
                  "revision": "3216a57f2a0d9c45a2e6c20157c20c49fb4bf9c7", "weight": "pytorch_model.bin"},
}
TOKEN_FILES = ("config.json", "vocab.json", "merges.txt")
OPTIONAL_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")


def _digest(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024*1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _metadata(path, name):
    file = path / ".cache/huggingface/download" / (name + ".metadata")
    return file.read_text(encoding="utf-8").splitlines() if file.exists() else []


def setup_model(root, task, download=None):
    if task not in PINS:
        raise ValueError("Choose emotion or sentiment")
    spec = PINS[task]
    path = Path(root) / "runs/model_cache" / spec["directory"]
    path.mkdir(parents=True, exist_ok=True)
    needed = (*TOKEN_FILES, spec["weight"])
    for name in needed:
        metadata = _metadata(path, name)
        if (path / name).exists() and metadata and metadata[0] != spec["revision"]:
            raise ValueError(f"{path / name} belongs to another model revision. Move that cache aside before installing this pinned version.")
    reusable = all((path / name).is_file() and _metadata(path, name)[:1] == [spec["revision"]] for name in needed)
    if not reusable:
        if download is None:
            from huggingface_hub import snapshot_download
            download = snapshot_download
        download(repo_id=spec["id"], revision=spec["revision"], local_dir=str(path),
                 allow_patterns=list(needed + OPTIONAL_FILES))
    for name in needed:
        if not (path / name).is_file():
            raise ValueError("Model setup did not produce " + str(path / name))
        metadata = _metadata(path, name)
        if not metadata or metadata[0] != spec["revision"]:
            raise ValueError("Missing pinned source metadata for " + str(path / name))
        if len(metadata) > 1 and re.fullmatch(r"[a-f0-9]{64}", metadata[1]) and _digest(path / name) != metadata[1]:
            raise ValueError("Cached weight/content hash mismatch: " + str(path / name))
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    labels = {int(k): v for k, v in config.get("id2label", {}).items()}
    if task == "sentiment" and labels != {0: "negative", 1: "neutral", 2: "positive"}:
        raise ValueError("Sentiment model label mapping is incompatible")
    if task == "emotion":
        from label_schema import LABELS
        if set(labels) != set(range(len(LABELS))) or set(labels.values()) != set(LABELS) or config.get("problem_type") != "multi_label_classification":
            raise ValueError("Emotion model vocabulary or multi-label setting is incompatible")
    conversion = None
    runtime = path / "model.safetensors"
    previous_manifest = path / "setup_manifest.json"
    if previous_manifest.exists():
        previous = json.loads(previous_manifest.read_text(encoding="utf-8"))
        expected = previous.get("files_sha256", {}).get("model.safetensors")
        if expected and runtime.exists() and _digest(runtime) != expected:
            raise ValueError("Runtime weights changed since setup: " + str(runtime))
    if task == "sentiment":
        conversion = {"source": "pytorch_model.bin", "source_sha256": _digest(path / "pytorch_model.bin"),
                      "method": "torch.load(weights_only=True,map_location='cpu') -> safetensors",
                      "performed_this_setup": not runtime.exists()}
        if not runtime.exists():
            import torch
            from safetensors.torch import save_file
            weights = torch.load(path / "pytorch_model.bin", map_location="cpu", weights_only=True)
            if not isinstance(weights, dict) or not weights or any(not isinstance(v, torch.Tensor) for v in weights.values()):
                raise ValueError("Expected a tensor-only PyTorch state dictionary")
            temporary = path / "model.safetensors.tmp"
            save_file({key: value.contiguous() for key, value in weights.items()}, str(temporary),
                      metadata={"format": "pt", "source_repo": spec["id"], "source_revision": spec["revision"],
                                "source_sha256": conversion["source_sha256"]})
            os.replace(temporary, runtime)
    files = {item.name: _digest(item) for item in sorted(path.iterdir())
             if item.is_file() and item.name != "setup_manifest.json" and item.suffix in (".json", ".txt", ".bin", ".safetensors")}
    manifest = {"task": task, "model": spec["id"], "revision": spec["revision"],
                "runtime_format": "safetensors", "files_sha256": files, "conversion": conversion,
                "setup_at": datetime.now(timezone.utc).isoformat()}
    manifest_path = previous_manifest
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("revision") == manifest["revision"] and previous.get("files_sha256") == files:
            manifest = previous
    temporary = path / "setup_manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(temporary, manifest_path)
    return {"task": task, "path": str(path), "revision": spec["revision"],
            "reused_download": reusable, "manifest": str(manifest_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("emotion", "sentiment", "all"), default="all")
    args = parser.parse_args()
    try:
        for task in PINS if args.task == "all" else (args.task,):
            result = setup_model(ROOT, task)
            print(f"Ready: {task} ({result['revision']}); " + ("reused cached download" if result["reused_download"] else "downloaded pinned files"))
            print(result["path"])
    except Exception as exc:
        print("Model setup failed: " + str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
