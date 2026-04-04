#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

CANON_PART = re.compile(r"(cell_\d+_\d+__r\d+_c\d+)", re.IGNORECASE)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="Directory containing .txt labels")
    ap.add_argument(
        "--delete-collisions",
        action="store_true",
        help="Delete duplicate prefixed label if canonical destination already exists",
    )
    args = ap.parse_args()

    root = Path(args.dir).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"Directory does not exist: {root}")

    renamed = 0
    skipped = 0
    collisions = 0
    deleted = 0

    for p in sorted(root.glob("*.txt")):
        m = CANON_PART.search(p.stem)
        if not m:
            skipped += 1
            continue

        new_stem = m.group(1).lower()
        new_path = p.with_name(new_stem + ".txt")

        if p.name == new_path.name:
            continue

        if new_path.exists():
            collisions += 1
            if args.delete_collisions:
                print("DELETE DUPLICATE:", p.name, "->", new_path.name)
                p.unlink()
                deleted += 1
            else:
                print("COLLISION:", new_path)
            continue

        p.rename(new_path)
        renamed += 1
        print("RENAMED:", p.name, "->", new_path.name)

    print()
    print("SUMMARY")
    print("renamed:", renamed)
    print("skipped:", skipped)
    print("collisions:", collisions)
    print("deleted:", deleted)

if __name__ == "__main__":
    main()