"""Download the three resources used by the documented ABC/YAM recipe."""
import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

RESOURCES = (
    ("GEAR-Dreams/DreamZero-AgiBot", "DreamZero-AgiBot", None),
    ("Wan-AI/Wan2.1-I2V-14B-480P", "Wan2.1-I2V-14B-480P", None),
    # DreamZero only uses this repository for its tokenizer.
    ("google/umt5-xxl", "umt5-xxl", ["*.json", "*.model", "*.txt"]),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    api = HfApi()
    plan = []
    for repo, directory, patterns in RESOURCES:
        info = api.model_info(repo, files_metadata=True)
        from fnmatch import fnmatch
        files = [f for f in info.siblings if patterns is None or
                 any(fnmatch(f.rfilename, p) for p in patterns)]
        item = dict(repo_id=repo, revision=info.sha,
                    destination=str(args.destination / directory),
                    allow_patterns=patterns, files=len(files),
                    bytes=sum(f.size or 0 for f in files))
        plan.append(item)
        print(json.dumps(item), flush=True)
    if args.plan:
        return
    args.destination.mkdir(parents=True, exist_ok=True)
    for item in plan:
        snapshot_download(repo_id=item["repo_id"], revision=item["revision"],
                          local_dir=item["destination"],
                          allow_patterns=item["allow_patterns"], max_workers=4)
        (Path(item["destination"]) / "dreamzero_download_manifest.json").write_text(
            json.dumps(item, indent=2) + "\n")
        print(f"COMPLETE: {item['destination']}", flush=True)


if __name__ == "__main__":
    main()
