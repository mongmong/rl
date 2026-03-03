#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def has_saved_artifact(run_dir: Path) -> bool:
    model_zip = run_dir / "model.zip"
    if model_zip.exists():
        return True
    checkpoint_dir = run_dir / "checkpoints"
    if not checkpoint_dir.exists():
        return False
    return any(checkpoint_dir.glob("*.zip"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Remove model run folders that do not contain either "
            "a saved model.zip or any checkpoint .zip."
        )
    )
    parser.add_argument("--models-dir", default="models", help="Root folder containing model runs.")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Actually delete folders. Without this flag, runs in dry-run mode.",
    )
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    if not models_dir.exists():
        print(f"Models directory not found: {models_dir}")
        return

    candidates: list[Path] = []
    for entry in sorted(models_dir.iterdir()):
        if not entry.is_dir():
            continue
        if has_saved_artifact(entry):
            continue
        candidates.append(entry)

    if not candidates:
        print("No empty model run folders found.")
        return

    mode = "APPLY" if args.yes else "DRY-RUN"
    print(f"[{mode}] Found {len(candidates)} folder(s) to remove:")
    for path in candidates:
        print(f"- {path}")

    if not args.yes:
        print("No folders were removed. Re-run with --yes to delete.")
        return

    removed = 0
    for path in candidates:
        shutil.rmtree(path, ignore_errors=False)
        removed += 1
    print(f"Removed {removed} folder(s).")


if __name__ == "__main__":
    main()
