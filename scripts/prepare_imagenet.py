"""Build indexed ImageNet metadata (TRAIN and VAL only)."""
import argparse
from pathlib import Path
from jea.data.datasets import ImageNet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--extra", required=True, type=Path)
    args = parser.parse_args()
    if not (args.root / "labels.txt").is_file():
        parser.error("root/labels.txt must contain ImageNet synset,class_name CSV rows in ImageFolder class order")
    args.extra.mkdir(parents=True, exist_ok=True)
    for split in (ImageNet.Split.TRAIN, ImageNet.Split.VAL):
        ImageNet(split=split, root=str(args.root), extra=str(args.extra)).dump_extra()


if __name__ == "__main__":
    main()
