# Modified for the locality-tradeoff research codebase; see NOTICE.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import argparse
import json
import logging
import sys
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.backends.cudnn as cudnn
from PIL import Image, ImageDraw, ImageFont

import jea.distributed as distributed
from jea.eval.setup import build_model_for_eval, get_autocast_dtype, setup_and_build_model
from jea.diagnostics.local_detail_decodability import (
    LocalDetailTargetBuilder,
    assert_token_target_alignment,
    collect_probe_stats,
    extract_patch_tokens,
    fit_raw_patch_pca,
    get_patch_size,
    parse_layer,
    parse_targets,
    parse_visualization_targets,
    predict_targets_from_tokens,
    prepare_dataloaders,
    solve_ridge_probe,
    target_group_to_grid_image,
)
from jea.diagnostics.visualization_utils import (
    ensure_dir,
    load_visualization_indices,
    make_unlabeled_panel_grid,
    safe_name,
    save_manifest_csv,
    save_visualization_indices,
    save_visualization_pdf,
    tensor_to_pil_image,
)
from jea.utils.config import apply_scaling_rules_to_cfg, get_cfg_from_args, write_config


logger = logging.getLogger("JEA")


def get_args_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Render local-detail decodability comparison grids. Each metric row is: "
            "image, GT, then one prediction column per model in model_roots order."
        )
    )
    parser.add_argument("--model-roots", nargs="+", help="Model directories, in prediction-column order.")
    parser.add_argument("--checkpoint-step", default="375299", type=str)
    parser.add_argument("--model-names", nargs="+", help="Optional names, one per root; defaults to directory names.")
    parser.add_argument("--model-weights", nargs="+", help="Optional checkpoint paths, one per root.")
    parser.add_argument("--model-probe-files", nargs="+", help="Optional probe paths, one per root; use an empty string to fit a probe.")
    parser.add_argument("--left-config-file", type=str)
    parser.add_argument("--left-pretrained-weights", type=str)
    parser.add_argument("--left-name", default="dino", type=str)
    parser.add_argument("--left-probe-file", default="", type=str)
    parser.add_argument("--right-config-file", type=str)
    parser.add_argument("--right-pretrained-weights", type=str)
    parser.add_argument("--right-name", default="dino_ibot", type=str)
    parser.add_argument("--right-probe-file", default="", type=str)
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--dataset", dest="dataset_str", default="ImageNet:split=TRAIN", type=str)
    parser.add_argument("--train-images", default=10000, type=int)
    parser.add_argument("--val-images", default=10000, type=int)
    parser.add_argument("--subset-seed", default=0, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--num-workers", default=5, type=int)
    parser.add_argument("--patches-per-image", default=0, type=int)
    parser.add_argument("--layer", default="final", type=str)
    parser.add_argument("--targets", default="rgb_mean,rgb_hist,hog,dct_energy,lbp_hist,raw_pca", type=str)
    parser.add_argument("--ridge-alpha", default=1e-3, type=float)
    parser.add_argument("--standardize-eps", default=1e-6, type=float)
    parser.add_argument("--resize-size", default=256, type=int)
    parser.add_argument("--crop-size", default=224, type=int)
    parser.add_argument("--rgb-hist-bins", default=8, type=int)
    parser.add_argument("--hog-bins", default=9, type=int)
    parser.add_argument("--dct-bands", default=6, type=int)
    parser.add_argument("--lbp-bins", default=16, type=int)
    parser.add_argument("--raw-pca-dim", default=64, type=int)
    parser.add_argument("--log-every", default=20, type=int)
    parser.add_argument("--visualization-samples", default=8, type=int)
    parser.add_argument("--visualization-targets", default="rgb_mean,hog,dct_energy,raw_pca", type=str)
    parser.add_argument("--visualization-indices-file", default="", type=str)
    parser.add_argument("--write-visualization-indices-file", default="", type=str)
    parser.add_argument("--gap", default=8, type=int)
    parser.add_argument("--sample-gap", default=18, type=int)
    parser.add_argument("--metric-label-width", default=120, type=int)
    parser.add_argument(
        "--save-probes",
        action="store_true",
        help="Save comparison probes to output_dir for reuse with --model-probe-files.",
    )
    parser.add_argument(
        "--left-opts",
        default=[],
        nargs="*",
        help="Optional config overrides for the left model. Place after all normal options if used.",
    )
    parser.add_argument(
        "--right-opts",
        default=[],
        nargs="*",
        help="Optional config overrides for the right model. Place after all normal options if used.",
    )
    return parser


def resolve_models(args) -> List[Dict]:
    legacy_fields = (args.left_config_file, args.left_pretrained_weights, args.right_config_file,
                     args.right_pretrained_weights, args.left_probe_file, args.right_probe_file,
                     args.left_opts, args.right_opts)
    if args.model_roots:
        if any(legacy_fields):
            raise ValueError("Use --model-roots with --model-weights/--model-probe-files, or legacy left/right inputs, not both.")
        roots = [Path(value) for value in args.model_roots]
        for option, values in (("--model-names", args.model_names), ("--model-weights", args.model_weights),
                               ("--model-probe-files", args.model_probe_files)):
            if values is not None and len(values) != len(roots):
                raise ValueError(f"{option} must have one entry per model root.")
        models = []
        used_names = set()
        for index, root in enumerate(roots):
            name = args.model_names[index].strip() if args.model_names else root.name
            if not args.model_names:
                suffix = 2
                while name in used_names:
                    name = f"{root.name}_{suffix}"
                    suffix += 1
            used_names.add(name)
            models.append(dict(
                name=name, root=str(root), config_file=str(root / "config.yaml"),
                pretrained_weights=(args.model_weights[index] if args.model_weights else
                                    str(root / "eval" / f"training_{args.checkpoint_step}" / "teacher_checkpoint.pth")),
                probe_file=args.model_probe_files[index] if args.model_probe_files else "", opts=[],
            ))
    else:
        if any(value is not None for value in (args.model_names, args.model_weights, args.model_probe_files)):
            raise ValueError("--model-names/--model-weights/--model-probe-files require --model-roots.")
        if not all((args.left_config_file, args.left_pretrained_weights, args.right_config_file, args.right_pretrained_weights)):
            raise ValueError("Provide --model-roots, or both legacy model configs and weights.")
        models = [dict(name=getattr(args, f"{side}_name"), config_file=getattr(args, f"{side}_config_file"),
                       pretrained_weights=getattr(args, f"{side}_pretrained_weights"),
                       probe_file=getattr(args, f"{side}_probe_file"), opts=getattr(args, f"{side}_opts"))
                  for side in ("left", "right")]
    names = [model["name"] for model in models]
    if any(not name.strip() for name in names) or len(set(names)) != len(names):
        raise ValueError("Model names must be non-empty and unique.")
    for model in models:
        for key in ("config_file", "pretrained_weights", "probe_file"):
            if key == "probe_file" and not model[key]:
                continue
            if not Path(model[key]).is_file():
                raise FileNotFoundError(f"{model['name']} {key} not found: {model[key]}")
    return models


def build_model(
    config_file: str,
    weights: str,
    output_dir: Path,
    opts: Optional[List[str]] = None,
    initialize_distributed: bool = False,
):
    model_args = SimpleNamespace(
        config_file=config_file,
        pretrained_weights=weights,
        output_dir=str(output_dir),
        opts=opts or [],
    )
    if initialize_distributed:
        model, autocast_dtype = setup_and_build_model(model_args)
    else:
        cudnn.benchmark = True
        output_dir.mkdir(parents=True, exist_ok=True)
        config = get_cfg_from_args(model_args)
        config.train.output_dir = str(output_dir)
        apply_scaling_rules_to_cfg(config)
        write_config(config, str(output_dir))
        model = build_model_for_eval(config, weights)
        autocast_dtype = get_autocast_dtype(config)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, autocast_dtype, get_patch_size(model)


def make_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def move_tensor_dict_to_device(values: Dict, device: torch.device) -> Dict:
    moved = {}
    for key, value in values.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device=device)
        else:
            moved[key] = value
    return moved


def load_probe_payload(path: str) -> Dict:
    payload = torch.load(path, map_location="cpu")
    if "probe" not in payload:
        raise RuntimeError(f"Probe file does not contain a 'probe' entry: {path}")
    return payload


def probe_target_slices(payload: Optional[Dict]) -> Optional[Dict[str, Tuple[int, int]]]:
    if not payload or "target_slices" not in payload:
        return None
    return {name: tuple(int(v) for v in span) for name, span in payload["target_slices"].items()}


def validate_probe(probe: Dict, target_builder: LocalDetailTargetBuilder, name: str):
    target_dim = int(probe.get("target_dim", probe["weight"].shape[1]))
    if target_dim != target_builder.target_dim:
        raise RuntimeError(
            f"{name} probe target_dim={target_dim} does not match requested targets "
            f"({target_builder.target_dim}). Use probes trained with the same --targets and raw target settings."
        )


def validate_probe_slices(payload: Optional[Dict], target_builder: LocalDetailTargetBuilder, name: str):
    slices = probe_target_slices(payload)
    if slices is None:
        return
    expected = {key: tuple(value) for key, value in target_builder.slices.items()}
    if slices != expected:
        raise RuntimeError(
            f"{name} probe target slices do not match requested targets. "
            "Use matching probes or omit probe files so this script fits new probes."
        )


def raw_pca_from_payload(payload: Optional[Dict]) -> Optional[Dict]:
    if payload and payload.get("raw_pca") is not None:
        return payload["raw_pca"]
    return None


def validate_raw_pca_compatible(left_pca: Optional[Dict], right_pca: Optional[Dict]):
    if left_pca is None or right_pca is None:
        return
    for key in ("mean", "components"):
        left_value = left_pca.get(key)
        right_value = right_pca.get(key)
        if left_value is None or right_value is None:
            continue
        if tuple(left_value.shape) != tuple(right_value.shape):
            raise RuntimeError(f"Loaded probe raw_pca {key} shapes do not match.")
        if not torch.allclose(left_value.float(), right_value.float(), rtol=1e-4, atol=1e-5):
            raise RuntimeError(
                "Loaded probes use different raw_pca bases. "
                "Use probes trained with the same subset/settings or omit probe files so this script fits shared probes."
            )


def resolve_raw_pca_info(
    targets: Sequence[str],
    train_loader,
    patch_size: int,
    args,
    payloads: Sequence[Optional[Dict]],
):
    if "raw_pca" not in targets:
        return None, None, None, args.raw_pca_dim

    pca_info = None
    for payload in payloads:
        model_pca = raw_pca_from_payload(payload)
        validate_raw_pca_compatible(pca_info, model_pca)
        if pca_info is None and model_pca is not None:
            pca_info = model_pca
    if pca_info is None:
        logger.info("Fitting shared raw-patch PCA for local-detail comparison.")
        pca_info = fit_raw_patch_pca(
            train_loader=train_loader,
            patch_size=patch_size,
            patches_per_image=args.patches_per_image,
            raw_pca_dim=args.raw_pca_dim,
            log_every=args.log_every,
            patch_generator=make_generator(args.subset_seed + 17),
        )
    raw_pca_mean = pca_info["mean"]
    raw_pca_components = pca_info["components"]
    raw_pca_dim = int(pca_info["pca_dim"])
    return pca_info, raw_pca_mean, raw_pca_components, raw_pca_dim


def fit_or_load_probe(
    name: str,
    model,
    autocast_dtype: torch.dtype,
    train_loader,
    target_builder: LocalDetailTargetBuilder,
    layer,
    args,
    probe_file: str,
    seed_offset: int,
) -> Dict:
    device = torch.device("cuda", torch.cuda.current_device())
    if probe_file:
        logger.info(f"Loading {name} local-detail probe from {probe_file}.")
        payload = load_probe_payload(probe_file)
        validate_probe_slices(payload, target_builder, name)
        probe = move_tensor_dict_to_device(payload["probe"], device)
        validate_probe(probe, target_builder, name)
        return probe

    logger.info(f"Collecting ridge sufficient statistics for {name}.")
    stats = collect_probe_stats(
        model=model,
        train_loader=train_loader,
        target_builder=target_builder,
        layer=layer,
        patches_per_image=args.patches_per_image,
        autocast_dtype=autocast_dtype,
        log_every=args.log_every,
        patch_generator=make_generator(args.subset_seed + seed_offset),
    )
    probe = solve_ridge_probe(stats, ridge_alpha=args.ridge_alpha, eps=args.standardize_eps)
    validate_probe(probe, target_builder, name)
    return probe


def save_probe(path: Path, probe: Dict, target_builder: LocalDetailTargetBuilder, pca_info: Optional[Dict], args):
    payload = {
        "probe": {key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in probe.items()},
        "target_slices": dict(target_builder.slices),
        "args": vars(args),
    }
    if pca_info is not None:
        payload["raw_pca"] = {
            key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in pca_info.items()
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def save_png(image: Image.Image, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path)
    return path


def text_size(draw: ImageDraw.ImageDraw, text: str, font) -> Tuple[int, int]:
    if hasattr(draw, "textbbox"):
        try:
            box = draw.textbbox((0, 0), text, font=font)
            return box[2] - box[0], box[3] - box[1]
        except ValueError:
            # Older Pillow exposes textbbox but does not support bitmap fonts there.
            pass
    return draw.textsize(text, font=font)


def draw_centered_text(draw: ImageDraw.ImageDraw, box: Tuple[int, int, int, int], text: str, font, fill):
    x0, y0, x1, y1 = box
    w, h = text_size(draw, text, font)
    available_width = max(0, x1 - x0 - 4)
    if w > available_width:
        shortened = text
        while shortened and text_size(draw, shortened + "...", font)[0] > available_width:
            shortened = shortened[:-1]
        text = shortened + "..." if shortened else ""
        w, h = text_size(draw, text, font)
    draw.text((x0 + (x1 - x0 - w) / 2, y0 + (y1 - y0 - h) / 2), text, font=font, fill=fill)


def make_metric_comparison_grid(
    base: Image.Image,
    targets: torch.Tensor,
    predictions: Sequence[torch.Tensor],
    target_builder: LocalDetailTargetBuilder,
    target_names: Sequence[str],
    model_names: Sequence[str],
    grid_h: int,
    grid_w: int,
    gap: int,
    metric_label_width: int,
) -> Image.Image:
    if not predictions or len(predictions) != len(model_names):
        raise ValueError("Provide one prediction tensor per model name.")
    font = ImageFont.load_default()
    header_h = 26
    metric_w = int(metric_label_width)
    panel_w, panel_h = base.size
    num_columns = 2 + len(predictions)
    width = metric_w + panel_w * num_columns + gap * num_columns
    height = header_h + len(target_names) * panel_h + gap * max(len(target_names), 1)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    headers = ["image", "GT"] + [f"{name} pred" for name in model_names]
    draw.rectangle((0, 0, metric_w, header_h), fill=(245, 245, 245))
    draw_centered_text(draw, (0, 0, metric_w, header_h), "metric", font, (20, 20, 20))
    x = metric_w + gap
    for header in headers:
        draw.rectangle((x, 0, x + panel_w, header_h), fill=(245, 245, 245))
        draw_centered_text(draw, (x, 0, x + panel_w, header_h), header, font, (20, 20, 20))
        x += panel_w + gap

    for row_idx, target_name in enumerate(target_names):
        y = header_h + gap + row_idx * (panel_h + gap)
        draw.rectangle((0, y, metric_w, y + panel_h), fill=(250, 250, 250))
        draw_centered_text(draw, (0, y, metric_w, y + panel_h), target_name, font, (20, 20, 20))

        start, end = target_builder.slices[target_name]
        gt_values = targets[:, start:end]
        panels = [
            base,
            target_group_to_grid_image(gt_values, target_name, grid_h, grid_w, base.size),
        ]
        panels.extend(target_group_to_grid_image(pred[:, start:end], target_name, grid_h, grid_w, base.size)
                      for pred in predictions)
        x = metric_w + gap
        for panel in panels:
            canvas.paste(panel.convert("RGB"), (x, y))
            x += panel_w + gap
    return canvas


@torch.inference_mode()
def save_comparison_visualizations(
    models: Sequence[Dict],
    val_loader,
    target_builder: LocalDetailTargetBuilder,
    layer,
    output_dir: Path,
    args,
):
    locked_indices = []
    if args.visualization_indices_file:
        locked_indices = load_visualization_indices(args.visualization_indices_file)
    locked_index_set = set(int(idx) for idx in locked_indices)
    target_sample_count = len(locked_index_set) if locked_index_set else int(args.visualization_samples)
    if target_sample_count <= 0:
        return [], []

    write_outputs = distributed.is_main_process()
    rows_dir = ensure_dir(output_dir / "rows") if write_outputs else None
    vis_targets = parse_visualization_targets(args.visualization_targets, target_builder.slices.keys())
    if not vis_targets:
        raise RuntimeError("No requested local-detail visualization targets are available in this run.")

    manifest_rows = []
    sample_grids = []
    saved_sample_indices = []
    saved = 0
    for _step, (samples, indices) in enumerate(val_loader):
        images = samples["image"].cuda(non_blocking=True)
        raw = samples["raw"].cuda(non_blocking=True)
        targets = target_builder.compute(raw).float()
        predictions = []
        for model in models:
            with torch.cuda.amp.autocast(dtype=model["dtype"]):
                tokens = extract_patch_tokens(model["model"], images, layer).float()
            assert_token_target_alignment(tokens, targets)
            pred = predict_targets_from_tokens(tokens.reshape(-1, tokens.shape[-1]), model["probe"]).float()
            predictions.append(pred.view(tokens.shape[0], tokens.shape[1], -1))

        grid_h = raw.shape[-2] // target_builder.patch_size
        grid_w = raw.shape[-1] // target_builder.patch_size
        for batch_idx in range(raw.shape[0]):
            if saved >= target_sample_count:
                break
            sample_id = int(indices[batch_idx].item()) if torch.is_tensor(indices) else saved
            if locked_index_set and sample_id not in locked_index_set:
                continue
            if write_outputs:
                base = tensor_to_pil_image(raw[batch_idx])
                grid = make_metric_comparison_grid(
                    base=base,
                    targets=targets[batch_idx],
                    predictions=[pred[batch_idx] for pred in predictions],
                    target_builder=target_builder,
                    target_names=vis_targets,
                    model_names=[model["name"] for model in models],
                    grid_h=grid_h,
                    grid_w=grid_w,
                    gap=args.gap,
                    metric_label_width=args.metric_label_width,
                )
                row_stem = f"{saved:04d}_sample{sample_id}"
                png_path = save_png(grid, rows_dir / f"{row_stem}.png")
                pdf_path = save_visualization_pdf(grid, rows_dir / f"{row_stem}.pdf")
                sample_grids.append(grid)
                saved_sample_indices.append(sample_id)
                manifest_rows.append(
                    OrderedDict(
                        [
                            ("rank", saved),
                            ("sample_index", sample_id),
                            ("png_path", str(png_path)),
                            ("pdf_path", str(pdf_path)),
                            ("targets", ",".join(vis_targets)),
                            ("model_names", json.dumps([model["name"] for model in models], ensure_ascii=False)),
                        ]
                    )
                )
            saved += 1
        if saved >= target_sample_count:
            break

    if write_outputs and sample_grids:
        combined = make_unlabeled_panel_grid(sample_grids, columns=1, gap=args.sample_gap)
        combined_png = save_png(combined, output_dir / "local_detail_decodability_comparison.png")
        combined_pdf = save_visualization_pdf(combined, output_dir / "local_detail_decodability_comparison.pdf")
        save_manifest_csv(output_dir / "manifest.csv", manifest_rows)
        if args.write_visualization_indices_file:
            save_visualization_indices(args.write_visualization_indices_file, saved_sample_indices)
        logger.info(f"Wrote {len(sample_grids)} local-detail comparison grids to {combined_pdf}")
        return manifest_rows, [str(combined_png), str(combined_pdf)]

    return manifest_rows, []


def main(args):
    model_specs = resolve_models(args)
    output_dir = ensure_dir(args.output_dir)
    setup_dir = ensure_dir(output_dir / "model_setup")
    layer = parse_layer(args.layer)
    targets = parse_targets(args.targets)

    payloads = [load_probe_payload(spec["probe_file"]) if spec["probe_file"] else None for spec in model_specs]
    models = []
    patch_size = None
    for index, spec in enumerate(model_specs):
        model, dtype, model_patch_size = build_model(
            spec["config_file"], spec["pretrained_weights"],
            setup_dir / f"{index:02d}_{safe_name(spec['name'])}", spec["opts"],
            initialize_distributed=index == 0,
        )
        if patch_size is not None and patch_size != model_patch_size:
            raise RuntimeError(f"Patch sizes must match, got {patch_size} and {model_patch_size} for {spec['name']}.")
        patch_size = model_patch_size
        models.append(dict(name=spec["name"], model=model, dtype=dtype))

    train_loader, val_loader, subset_indices = prepare_dataloaders(args)
    pca_info, raw_pca_mean, raw_pca_components, raw_pca_dim = resolve_raw_pca_info(
        targets=targets,
        train_loader=train_loader,
        patch_size=patch_size,
        args=args,
        payloads=payloads,
    )
    target_builder = LocalDetailTargetBuilder(
        target_names=targets,
        patch_size=patch_size,
        rgb_hist_bins=args.rgb_hist_bins,
        hog_bins=args.hog_bins,
        dct_bands=args.dct_bands,
        lbp_bins=args.lbp_bins,
        raw_pca_dim=raw_pca_dim,
        raw_pca_mean=raw_pca_mean,
        raw_pca_components=raw_pca_components,
    )
    for spec, model, payload in zip(model_specs, models, payloads):
        validate_probe_slices(payload, target_builder, spec["name"])
        model["probe"] = fit_or_load_probe(
            name=spec["name"], model=model["model"], autocast_dtype=model["dtype"],
            train_loader=train_loader, target_builder=target_builder, layer=layer,
            args=args, probe_file=spec["probe_file"], seed_offset=101,
        )

    if distributed.is_main_process():
        torch.save(subset_indices.cpu(), output_dir / "sampled_indices.pt")
        if args.save_probes:
            for index, model in enumerate(models):
                save_probe(output_dir / f"{index:02d}_{safe_name(model['name'])}_ridge_probe.pt",
                           model["probe"], target_builder, pca_info, args)

    manifest_rows, combined_outputs = save_comparison_visualizations(
        models=models,
        val_loader=val_loader,
        target_builder=target_builder,
        layer=layer,
        output_dir=output_dir,
        args=args,
    )

    if distributed.is_main_process():
        summary = OrderedDict()
        summary["models"] = model_specs
        summary["column_order"] = ["image", "GT"] + [spec["name"] for spec in model_specs]
        if len(model_specs) == 2:
            for side, spec in zip(("left", "right"), model_specs):
                for key in ("name", "config_file", "pretrained_weights", "probe_file"):
                    summary[f"{side}_{key}"] = spec[key]
        summary["dataset"] = args.dataset_str
        summary["train_images"] = args.train_images
        summary["val_images"] = args.val_images
        summary["subset_seed"] = args.subset_seed
        summary["layer"] = args.layer
        summary["targets"] = targets
        summary["visualization_targets"] = parse_visualization_targets(args.visualization_targets, target_builder.slices.keys())
        summary["target_slices"] = {name: [start, end] for name, (start, end) in target_builder.slices.items()}
        summary["outputs"] = combined_outputs
        summary["num_samples"] = len(manifest_rows)
        with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    if distributed.is_enabled():
        torch.distributed.barrier()
    return 0


if __name__ == "__main__":
    parsed_args = get_args_parser().parse_args()
    sys.exit(main(parsed_args))
