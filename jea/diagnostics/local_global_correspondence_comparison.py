# Modified for the locality-tradeoff research codebase; see NOTICE.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""Compare correspondence results, rendering missing model rows for selected crops only."""

import argparse
import csv
import json
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from jea.diagnostics.visualization_utils import (
    ensure_dir,
    make_unlabeled_panel_grid,
    safe_name,
    save_visualization_manifest,
    save_visualization_pdf,
)


def get_args_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-roots", nargs="+", required=True,
                        help="Model directories in row order; names are derived from directory names.")
    parser.add_argument("--checkpoint-steps", default="",
                        help="Comma-separated steps; defaults to all evaluated checkpoints of the first model.")
    parser.add_argument("--layers", default="2,5,8,11")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gap", type=int, default=4)
    parser.add_argument("--visualization-images", type=int, default=16)
    parser.add_argument("--visualization-samples", type=int, default=16)
    parser.add_argument("--visualization-max-per-image", type=int, default=1)
    parser.add_argument("--visualization-max-match-lines", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-dir", default="", help="Optional source-image directory override for rendering missing rows.")
    return parser


def model_display_names(roots):
    names = []
    for root in roots:
        base = root.name
        name = base
        suffix = 2
        while name in names:
            name = f"{base}_{suffix}"
            suffix += 1
        names.append(name)
    return names


def query_key(row):
    # Query indices and filename ranks are not sufficient to identify a crop.
    return (
        str(Path(row["image_path"])),
        row["image_id"],
        int(row["crop_index"]),
        float(row["crop_ratio"]),
        tuple(json.loads(row["box_xyxy"])),
        tuple(json.loads(row["box_norm_xyxy"])),
    )


class MissingVisualizationError(FileNotFoundError):
    """Saved rows are absent or incomplete and need to be rendered."""


def load_visualization_rows(vis_dir):
    manifest = vis_dir / "manifest_pdf.csv"
    if not manifest.is_file():
        raise MissingVisualizationError(f"Missing correspondence visualization manifest: {manifest}")
    with manifest.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise MissingVisualizationError(f"No visualization rows in {manifest}")
    indexed = OrderedDict()
    for row in rows:
        key = query_key(row)
        if key in indexed:
            raise ValueError(f"Duplicate crop in {manifest}: {row['image_id']}, crop {row['crop_index']}")
        value = row.get("visualization_png", "")
        if not value:
            raise MissingVisualizationError(f"Missing PNG row in {manifest}")
        path = Path(value)
        # Also support result directories moved together with their manifest.
        local_path = vis_dir / path.name
        if local_path.is_file():
            path = local_path
        if not path.is_file():
            raise MissingVisualizationError(f"Missing visualization PNG: {path}")
        indexed[key] = (row, path)
    return indexed


def select_comparison_rows(rows, args):
    selected = []
    counts = {}
    for row in sorted(rows, key=lambda row: int(row["query_index"])):
        image = row["image_path"]
        count = counts.get(image, 0)
        limit = max(1, args.visualization_max_per_image) if args.visualization_images > 0 else args.visualization_max_per_image
        if limit > 0 and count >= limit:
            continue
        if args.visualization_images > 0:
            if image not in counts and len(counts) >= args.visualization_images:
                continue
        elif len(selected) >= args.visualization_samples:
            break
        selected.append(row)
        counts[image] = count + 1
    if not selected:
        raise ValueError("No crops selected for comparison; set a positive visualization image/sample budget.")
    return selected


def load_evaluated_rows(result_dir):
    path = result_dir / "per_local_global_correspondence.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing per-crop results needed to render visualizations: {path}")
    with path.open(newline="", encoding="utf-8") as f:
        # Evaluation visualizations use the first repeat; retain the same crop identities.
        rows = [row for row in csv.DictReader(f) if int(row.get("run_index") or 0) == 0]
    indexed = OrderedDict()
    for row in rows:
        key = query_key(row)
        if key in indexed:
            raise ValueError(f"Duplicate first-run crop in {path}: {row['image_id']}")
        indexed[key] = row
    if not indexed:
        raise ValueError(f"No first-run crop results in {path}")
    return indexed


def load_visualization_model(root, checkpoint):
    config_path = root / "config.yaml"
    weights_path = root / "eval" / checkpoint / "teacher_checkpoint.pth"
    for path in (config_path, weights_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing file required to render correspondence visualizations: {path}")
    # Load inference config directly: no evaluation setup, distributed launch, or metrics writes.
    from jea.eval.setup import build_model_for_eval, get_autocast_dtype
    from jea.utils.config import get_cfg_from_args

    config = get_cfg_from_args(SimpleNamespace(config_file=str(config_path), opts=[]))
    return build_model_for_eval(config, str(weights_path)), get_autocast_dtype(config)


def render_missing_rows(root, checkpoint, layer, rows, vis_dir, args):
    """Run feature extraction only for the requested visualizations, never full evaluation."""
    result_dir = root / "local_global_correspondence_all" / checkpoint / f"layer_{layer}"
    with (result_dir / "results_local_global_correspondence.json").open(encoding="utf-8") as f:
        settings = json.load(f)
    specs = []
    for row in rows:
        image_path = Path(args.image_dir) / Path(row["image_path"]).name if args.image_dir else Path(row["image_path"])
        if not image_path.is_file():
            raise FileNotFoundError(f"Source image needed for visualization not found: {image_path}; set --image-dir if images moved.")
        specs.append(dict(
            image_path=str(image_path), box_xyxy=json.loads(row["box_xyxy"]),
            box_norm_xyxy=json.loads(row["box_norm_xyxy"]), crop_ratio=float(row["crop_ratio"]),
        ))

    import torch
    from torch.utils.data import DataLoader
    from jea.diagnostics.local_global_correspondence import (
        LocalGlobalCorrespondenceDataset, evaluate_one_crop, save_crop_visualizations,
    )
    from jea.diagnostics.local_detail_decodability import extract_patch_tokens, get_patch_size, parse_layer

    print(f"Rendering {len(rows)} selected crops for {root.name}/{checkpoint}/layer_{layer}")
    model, autocast_dtype = load_visualization_model(root, checkpoint)
    try:
        model.eval()
        device = next(model.parameters()).device
        imsize = int(settings["imsize"])
        patch_size = get_patch_size(model)
        if imsize % patch_size:
            raise ValueError(f"Saved imsize {imsize} is not divisible by patch size {patch_size}")
        grid_size = imsize // patch_size
        loader = DataLoader(LocalGlobalCorrespondenceDataset(specs, imsize=imsize), batch_size=args.batch_size,
                            num_workers=args.num_workers, shuffle=False)
        payloads = []
        with torch.inference_mode():
            for batch in loader:
                source = batch["source"].to(device)
                crop = batch["crop"].to(device)
                with torch.cuda.amp.autocast(enabled=device.type == "cuda", dtype=autocast_dtype):
                    tokens = extract_patch_tokens(model, torch.cat([source, crop]), parse_layer(layer))
                for position, index in enumerate(batch["index"].tolist()):
                    metrics, artifacts = evaluate_one_crop(
                        tokens[position], tokens[source.shape[0] + position], batch["box_norm_xyxy"][position],
                        grid_size, grid_size, int(settings["topk"]), int(settings["ransac_iters"]),
                        float(settings["geometric_tolerance_patches"]), return_artifacts=True,
                    )
                    row = OrderedDict(rows[index])
                    row.update(metrics)
                    row["visualization_recomputed"] = True
                    payloads.append(dict(row=row, artifacts=artifacts,
                                         source_tensor=source[position].cpu(), crop_tensor=crop[position].cpu()))
        render_args = SimpleNamespace(
            save_visualizations=True, visualization_dir=str(vis_dir), visualization_images=0,
            visualization_samples=len(payloads), visualization_max_per_image=0,
            visualization_max_match_lines=args.visualization_max_match_lines,
        )
        save_crop_visualizations(str(vis_dir), payloads, render_args, grid_size, grid_size)
    finally:
        del model
    return load_visualization_rows(vis_dir)


def prepare_comparison_rows(roots, checkpoint, layer, args):
    relative = Path(checkpoint) / f"layer_{layer}"
    indexed_models = []
    cache_dirs = []
    for index, root in enumerate(roots):
        cache_dir = Path(args.output_dir) / relative / "_model_rows" / f"model_{index:03d}"
        cache_dirs.append(cache_dir)
        indexed = OrderedDict()
        # Original saved rows take precedence; a cache is tied to its model root below.
        try:
            indexed.update(load_visualization_rows(root / "local_global_correspondence_all" / relative / "visualizations/local_global_correspondence"))
        except MissingVisualizationError:
            pass
        provenance_path = cache_dir / "model.json"
        if provenance_path.is_file():
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            if provenance == cache_provenance(root, args):
                try:
                    for key, value in load_visualization_rows(cache_dir).items():
                        indexed.setdefault(key, value)
                except MissingVisualizationError:
                    pass
        indexed_models.append(indexed)

    available = next((indexed for indexed in indexed_models if indexed), None)
    reference_dir = roots[0] / "local_global_correspondence_all" / relative
    if (reference_dir / "per_local_global_correspondence.csv").is_file():
        reference_rows = list(load_evaluated_rows(reference_dir).values())
    elif available is not None:
        reference_rows = [row for row, path in available.values()]
    else:
        reference_rows = list(load_evaluated_rows(roots[0] / "local_global_correspondence_all" / relative).values())
    selected = select_comparison_rows(reference_rows, args)
    keys = [query_key(row) for row in selected]
    for index, (root, indexed) in enumerate(zip(roots, indexed_models)):
        missing = [key for key in keys if key not in indexed]
        if missing:
            evaluated = load_evaluated_rows(root / "local_global_correspondence_all" / relative)
            if any(key not in evaluated for key in missing):
                raise ValueError(f"Selected crops do not match evaluated crops for {root}; use the same crop specs for all models.")
            # Include cached rows when repairing the cache so its manifest remains complete.
            render_keys = [key for key in keys if key in missing or indexed[key][1].parent == cache_dirs[index]]
            rendered = render_missing_rows(root, checkpoint, layer, [evaluated[key] for key in render_keys], cache_dirs[index], args)
            indexed.update(rendered)
            (cache_dirs[index] / "model.json").write_text(
                json.dumps(cache_provenance(root, args), indent=2), encoding="utf-8",
            )
        indexed_models[index] = OrderedDict((key, indexed[key]) for key in keys)
    return indexed_models


def cache_provenance(root, args):
    return {
        "model_root": str(root.resolve()),
        "visualization_max_match_lines": args.visualization_max_match_lines,
        "image_dir": str(Path(args.image_dir).resolve()) if args.image_dir else "",
    }


def save_comparison(roots, names, checkpoint, layer, output_dir, gap=4, indexed_models=None):
    relative_dir = Path(checkpoint) / f"layer_{layer}" / "visualizations" / "local_global_correspondence"
    indexed_models = indexed_models if indexed_models is not None else [
        load_visualization_rows(root / "local_global_correspondence_all" / relative_dir)
        for root in roots
    ]
    reference = indexed_models[0]
    for name, indexed in zip(names[1:], indexed_models[1:]):
        if set(indexed) != set(reference):
            raise ValueError(
                f"Visualization crops do not match for {name} at {checkpoint}/layer_{layer}: "
                f"{len(set(reference) - set(indexed))} missing, "
                f"{len(set(indexed) - set(reference))} extra. "
                "Use the same crop specs and visualization settings for every model, then refresh visualizations."
            )
    destination = ensure_dir(Path(output_dir) / checkpoint / f"layer_{layer}")
    manifest_rows = []
    for rank, key in enumerate(reference):
        row_images = []
        source_rows = []
        reference_size = None
        for name, indexed in zip(names, indexed_models):
            row, path = indexed[key]
            with Image.open(path) as image:
                panel_row = image.convert("RGB")
            if reference_size is None:
                reference_size = panel_row.size
            elif panel_row.size != reference_size:
                raise ValueError(f"Panel sizes differ for {name}: {panel_row.size} vs {reference_size}; use the same --imsize.")
            row_images.append(panel_row)
            source_rows.append(dict(model=name, **row))
        grid = make_unlabeled_panel_grid(row_images, columns=1, gap=gap)
        reference_row = reference[key][0]
        stem = (
            f"{rank:04d}_{safe_name(reference_row['image_id'])}"
            f"_ratio{safe_name(reference_row['crop_ratio'])}_crop{reference_row['crop_index']}"
        )
        pdf_path = save_visualization_pdf(grid, destination / f"{stem}.pdf")
        png_path = destination / f"{stem}.png"
        grid.save(png_path)
        metadata = OrderedDict([
            ("checkpoint", checkpoint),
            ("layer", layer),
            ("row_order", names),
            ("model_roots", [str(root) for root in roots]),
            ("panels", ["source", "crop", "GT/predicted boxes", "heatmap", "patch matches"]),
            ("models", source_rows),
        ])
        with (destination / f"{stem}.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        manifest_rows.append(OrderedDict([
            ("image_id", reference_row["image_id"]),
            ("query_index", reference_row["query_index"]),
            ("crop_ratio", reference_row["crop_ratio"]),
            ("crop_index", reference_row["crop_index"]),
            ("row_order", json.dumps(names, ensure_ascii=False)),
            ("visualization_path", str(pdf_path)),
            ("visualization_png", str(png_path)),
        ]))
    save_visualization_manifest(destination, manifest_rows)
    print(f"Wrote {len(manifest_rows)} correspondence comparisons ({len(roots)} model rows) to {destination}")


def main(args):
    roots = [Path(root) for root in args.model_roots]
    if len({root.resolve() for root in roots}) != len(roots):
        raise ValueError("--model-roots must contain distinct model directories.")
    if args.gap < 0:
        raise ValueError("--gap must be non-negative.")
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("--batch-size must be positive and --num-workers non-negative.")
    names = model_display_names(roots)
    layers = [value.strip() for value in args.layers.split(",") if value.strip()]
    if not layers or any(value != "final" and not value.isdigit() for value in layers):
        raise ValueError("--layers must contain final or non-negative layer indices.")
    steps = [value.strip() for value in args.checkpoint_steps.split(",") if value.strip()]
    if steps:
        if any(not value.isdigit() for value in steps):
            raise ValueError("--checkpoint-steps must contain integer checkpoint steps.")
        checkpoints = [f"training_{step}" for step in steps]
    else:
        checkpoints = sorted(
            path.name for path in (roots[0] / "local_global_correspondence_all").glob("training_*")
            if path.is_dir()
        )
    if not checkpoints:
        raise ValueError(f"No evaluated checkpoints found under {roots[0]}")
    for checkpoint in checkpoints:
        for layer in layers:
            indexed = prepare_comparison_rows(roots, checkpoint, layer, args)
            save_comparison(roots, names, checkpoint, layer, args.output_dir, args.gap, indexed_models=indexed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(get_args_parser().parse_args()))
