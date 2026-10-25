"""Download an immutable public model snapshot into this checkout and verify it."""
import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen3-0.6B"


def sha256(file):
    digest = hashlib.sha256()
    with file.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL)
    parser.add_argument("--revision", help="Optional immutable model commit")
    args = parser.parse_args()
    model_id = args.model_id
    destination = ROOT / "models" / model_id.rsplit("/", 1)[-1]
    if destination.parent != ROOT / "models" or destination.name in (".", "..", ""):
        parser.error("Model id must have a valid repository name")
    manifest = destination / "experiment-model-manifest.json"
    revision = args.revision
    if manifest.exists():
        previous = json.loads(manifest.read_text(encoding="utf-8"))
        assert previous["model"] == model_id, "Destination belongs to another model"
        if revision is not None:
            assert revision == previous["revision"], "Use a separate destination for another revision"
        revision = previous["revision"]
    print(json.dumps({"phase": "resolve_revision", "model": model_id}), flush=True)
    info = HfApi().model_info(model_id, revision=revision, files_metadata=True, token=False)
    assert len(info.sha) == 40
    print(json.dumps({"phase": "download", "revision": info.sha}), flush=True)
    snapshot_download(model_id, revision=info.sha, local_dir=destination, token=False,
                      allow_patterns=["*.json", "*.safetensors", "merges.txt", "vocab.json",
                                      "tokenizer*", "LICENSE", "README.md"], max_workers=2)
    files = {}
    for sibling in info.siblings:
        file = destination / sibling.rfilename
        if not file.is_file():
            continue
        actual = sha256(file)
        expected = sibling.lfs.sha256 if sibling.lfs is not None else None
        if expected:
            assert actual == expected, f"SHA256 mismatch: {sibling.rfilename}"
        files[sibling.rfilename] = {"bytes": file.stat().st_size, "sha256": actual,
                                   "expected_lfs_sha256": expected}
    weights = [name for name in files if name.endswith(".safetensors")]
    assert weights, "No model weights downloaded"
    payload = {"model": model_id, "revision": info.sha, "local_path": str(destination.relative_to(ROOT)),
               "files": files, "status": "verified"}
    manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"phase": "verified", "revision": info.sha,
                      "weight_bytes": sum(files[name]["bytes"] for name in weights)}), flush=True)


if __name__ == "__main__":
    main()
