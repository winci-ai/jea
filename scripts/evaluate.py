"""Paper evaluation presets over selected exported EMA checkpoints (Linux/Bash)."""

import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
TASKS = (
    "knn",
    "segmentation",
    "object_detection",
    "image_retrieval",
    "copy_detection",
    "davis_video_segmentation",
    "optical_flow",
    "cal_metrics",
    "local_detail_decodability",
    "local_global_correspondence",
)
LAYERS = {
    "vit_small": "2,5,8,11",
    "vit_base": "2,5,8,11",
    "vit_large": "5,11,17,23",
    "vit_huge": "7,15,23,31",
}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("task", choices=TASKS)
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument(
        "--epochs",
        default="300",
        help="Comma-separated epochs or 'all'; optical-flow paper table uses 180/260/300/270 for S/B/L/H",
    )
    p.add_argument(
        "--epoch-length",
        type=int,
        help="Override checkpoint step conversion for non-paper batch sizes",
    )
    p.add_argument("--gpu", default="0,1,2,3")
    p.add_argument("--port", default="29501")
    p.add_argument("--imagenet-root", default="datasets/ImageNet")
    p.add_argument("--imagenet-extra", default="datasets/ImageNet")
    p.add_argument(
        "--data-path",
        help="Copydays, DAVIS, or revisited Paris/Oxford root; dense tasks use their dataset configs",
    )
    p.add_argument(
        "--layers",
        default="final",
        help="final, paper (quarter-depth sweep), or comma-separated zero-based blocks",
    )
    p.add_argument(
        "--crop-ratios", default="0.1", help="Use 0.1,0.2,0.3,0.5 for ratio comparisons"
    )
    p.add_argument("--dry-run", action="store_true")
    return p


@contextmanager
def selected_exports(run, epochs, epoch_length, dry_run=False):
    if epochs == "all":
        yield run / "eval"
        return
    values = [int(e) for e in epochs.split(",")]
    if not values or any(e <= 0 for e in values):
        raise ValueError("Epochs must be positive integers")
    names = [f"training_{e * epoch_length - 1}" for e in values]
    if dry_run:
        print("Selected exports:", ", ".join(names))
        yield Path("<selected-checkpoints>")
        return
    # Real directories let existing run_all.sh discovery retain its original behavior.
    with tempfile.TemporaryDirectory(prefix="jea-eval-") as directory:
        for name in names:
            source = run / "eval" / name / "teacher_checkpoint.pth"
            if not source.is_file():
                raise FileNotFoundError(source)
            target = Path(directory) / name
            target.mkdir()
            (target / source.name).symlink_to(source)
        yield Path(directory)


def commands(args, cfg, eval_root):
    task, arch = args.task, cfg.student.arch
    output = args.run_dir / f"{task}_all"
    common = [
        "bash",
        str(ROOT / "scripts" / task / "run_all.sh"),
        "--eval-root",
        str(eval_root),
        "--output-root",
        str(output),
        "--csv-out",
        str(output / "summary.csv"),
        "--gpu",
        args.gpu,
        "--port",
        args.port,
    ]
    if task in {"segmentation", "object_detection", "optical_flow"}:
        if task == "segmentation":
            name = f"linear/{arch}_512_voc2012_10k.py"
        elif task == "object_detection":
            name = f"cascade_rcnn/{arch}_froze_giou_4conv1f_voc_3x.py"
        else:
            name = f"pwcnet/{arch}_froze_sintel.py"
        nproc = len(args.gpu.split(","))
        if 16 % nproc:
            raise ValueError(
                "The dense-evaluation batch size 16 must be divisible by GPU count"
            )
        common += [
            "--train-config",
            str(ROOT / "jea/eval" / task / "configs" / name),
            "--train-batch-size",
            str(16 // nproc),
            "--lr",
            "5e-4" if task == "segmentation" else "1e-4",
        ]
    else:
        common += ["--config-file", str(args.run_dir / "config.yaml")]
    train = (
        f"ImageNet:split=TRAIN:root={args.imagenet_root}:extra={args.imagenet_extra}"
    )
    if task == "knn":
        common += [
            "--train-dataset",
            train,
            "--val-dataset",
            train.replace("split=TRAIN", "split=VAL"),
        ]
    elif task == "cal_metrics":
        common += ["--dataset", train, "--num-images", "10000", "--subset-seed", "0"]
    elif task == "local_detail_decodability":
        layers = LAYERS[arch] if args.layers == "paper" else args.layers
        common += [
            "--dataset",
            train,
            "--train-images",
            "10000",
            "--val-images",
            "10000",
            "--subset-seed",
            "0",
            "--run-seeds",
            "0",
            "--patches-per-image",
            "0",
            "--layers",
            layers,
            "--ridge-alpha",
            "1e-3",
        ]
    elif task == "local_global_correspondence":
        layers = LAYERS[arch] if args.layers == "paper" else args.layers
        common += [
            "--data-path",
            args.data_path or "datasets/copydays",
            "--layers",
            layers,
            "--crop-ratios",
            args.crop_ratios,
            "--run-seeds",
            "0",
            "--max-images",
            "157",
            "--crops-per-image",
            "4",
        ]
    elif task == "copy_detection":
        root = args.data_path or "datasets/copydays"
        common += [
            "--data-path",
            root,
            "--whitening-path",
            str(Path(root) / "whitening_images"),
            "--distractors-path",
            str(Path(root) / "distractors_images"),
        ]
    elif task == "davis_video_segmentation":
        common += [
            "--data-path",
            args.data_path or "datasets/davis",
            "--set",
            "val",
            "--task",
            "semi-supervised",
        ]
    elif task == "image_retrieval":
        common += ["--data-path", args.data_path or "datasets/revisited_paris_oxford"]
        result = []
        for dataset in ("rparis6k", "roxford5k"):
            command = common.copy()
            command[command.index("--output-root") + 1] = str(output / dataset)
            command[command.index("--csv-out") + 1] = str(
                output / dataset / "summary.csv"
            )
            result.append(command + ["--dataset", dataset])
        return result
    return [common]


def main(argv=None):
    p = parser()
    argv = list(sys.argv[1:] if argv is None else argv)
    separator = argv.index("--") if "--" in argv else len(argv)
    extra = argv[separator + 1 :]
    args = p.parse_args(argv[:separator])
    args.run_dir = args.run_dir.resolve()
    cfg = OmegaConf.load(args.run_dir / "config.yaml")
    arch = cfg.student.arch
    if arch not in LAYERS:
        p.error(f"Unsupported paper architecture: {arch}")
    epoch_length = args.epoch_length or cfg.train.get("OFFICIAL_EPOCH_LENGTH", -1)
    if epoch_length <= 0:
        epoch_length = 2502 if arch == "vit_huge" else 1251
    if args.task == "optical_flow" and not args.dry_run:
        try:
            import mmflow.compat  # noqa: F401
        except ImportError as exc:
            raise SystemExit(
                "Optical flow needs the modified MMFlow; see README Section 1.3"
            ) from exc
    env = os.environ.copy()
    env["TORCHRUN_NPROC_PER_NODE"] = str(len(args.gpu.split(",")))
    with selected_exports(
        args.run_dir, args.epochs, epoch_length, args.dry_run
    ) as selected:
        for command in commands(args, cfg, selected):
            command += extra
            print(shlex.join(command), flush=True)
            if not args.dry_run:
                subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
