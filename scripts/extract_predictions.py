#!/usr/bin/env python3
"""Extract patches from mini-swe-agent trajectory files into predictions JSONL."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Extract patches from traj files -> predictions JSONL")
    parser.add_argument("--model", default="deepseek-v4-flash", help="Model name for predictions")
    parser.add_argument("--output", help="Output .jsonl path (default: runs/mini-swe/predictions/<model>__SWE-bench_Verified__test.jsonl)")
    parser.add_argument("--base-dir", default=".", help="Directory containing instance_id/traj.json files")
    parser.add_argument("--skip-missing", action="store_true", help="Skip instances with no patch (instead of error)")
    args = parser.parse_args()

    base = Path(args.base_dir).resolve()
    output = args.output or f"runs/mini-swe/predictions/{args.model}__SWE-bench_Verified__test.jsonl"

    predictions = []
    skipped = 0

    for traj_path in sorted(base.glob("*/*.traj.json")):
        instance_id = traj_path.parent.name
        if traj_path.name != f"{instance_id}.traj.json":
            continue

        with open(traj_path) as f:
            data = json.load(f)

        patch = data.get("info", {}).get("submission", "")
        if not patch:
            print(f"[SKIP] {instance_id}: no submission/patch found in traj")
            skipped += 1
            continue

        predictions.append({
            "instance_id": instance_id,
            "model_name_or_path": args.model,
            "model_patch": patch,
        })
        print(f"[OK] {instance_id}")

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")

    print(f"\nDone: {len(predictions)} instances -> {output}")
    if skipped:
        print(f"Skipped: {skipped} instances with no patch")


if __name__ == "__main__":
    main()
