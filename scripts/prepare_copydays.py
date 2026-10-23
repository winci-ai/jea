"""Select disjoint COCO whitening/distractor pools and save reproducible manifests."""

import argparse
from pathlib import Path
import random
import shutil


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--coco-root", required=True, type=Path)
    p.add_argument("--copydays-root", required=True, type=Path)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--copy",
        action="store_true",
        help="Copy images instead of creating symbolic links",
    )
    args = p.parse_args()
    images = sorted(
        f.resolve()
        for f in args.coco_root.iterdir()
        if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if len(images) < 30000:
        p.error("At least 30,000 COCO images are needed")
    random.Random(args.seed).shuffle(images)
    selections = [
        ("whitening_images", images[:20000]),
        ("distractors_images", images[20000:30000]),
    ]
    for name, _ in selections:
        if (args.copydays_root / name).exists():
            p.error(f"Refusing to overwrite existing pool: {args.copydays_root/name}")
    for name, selection in selections:
        directory = args.copydays_root / name
        directory.mkdir(parents=True)
        for source in selection:
            target = directory / source.name
            if args.copy:
                shutil.copy2(source, target)
            else:
                target.symlink_to(source)
        (directory / "manifest.txt").write_text(
            "\n".join(f.name for f in selection) + "\n", encoding="utf-8"
        )
        print(f"{name}: {len(selection)} images, seed={args.seed}")


if __name__ == "__main__":
    main()
