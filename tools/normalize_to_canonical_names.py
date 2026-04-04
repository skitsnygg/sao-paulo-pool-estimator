#!/usr/bin/env python3
import re
from pathlib import Path

# match canonical portion starting at cell_
CANON_PART = re.compile(r"(cell_\d+_\d+__r\d+_c\d+)", re.IGNORECASE)

def normalize_dir(root: Path):
    renamed = 0
    skipped = 0
    collisions = 0

    for p in sorted(root.glob("*")):
        if not p.is_file():
            continue

        m = CANON_PART.search(p.stem)
        if not m:
            skipped += 1
            continue

        new_stem = m.group(1).lower()
        new_path = p.with_name(new_stem + p.suffix.lower())

        if p.name == new_path.name:
            continue

        if new_path.exists():
            collisions += 1
            print("DELETE DUPLICATE:", p.name, "->", new_path.name)
            p.unlink()
            continue

        p.rename(new_path)
        renamed += 1
        print("RENAMED:", p.name, "->", new_path.name)

    print()
    print("SUMMARY")
    print("renamed:", renamed)
    print("skipped:", skipped)
    print("collisions:", collisions)


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="Directory to normalize")
    args = ap.parse_args()

    root = Path(args.dir).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"Directory does not exist: {root}")

    normalize_dir(root)


if __name__ == "__main__":
    main()