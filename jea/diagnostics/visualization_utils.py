# Modified for the locality-tradeoff research codebase; see NOTICE.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import csv
import json
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import torch
from PIL import Image, ImageDraw, ImageFont


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def ensure_dir(path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(value))


def tensor_to_pil_image(tensor: torch.Tensor, denormalize: bool = False) -> Image.Image:
    tensor = tensor.detach().cpu().float()
    if tensor.ndim != 3:
        raise ValueError(f"Expected CHW tensor, got shape={tuple(tensor.shape)}")
    if denormalize:
        tensor = tensor * IMAGENET_STD + IMAGENET_MEAN
    tensor = tensor.clamp(0.0, 1.0)
    array = (tensor.permute(1, 2, 0).numpy() * 255.0 + 0.5).astype("uint8")
    return Image.fromarray(array, mode="RGB")


def normalize_map(values: torch.Tensor, vmin: Optional[float] = None, vmax: Optional[float] = None) -> torch.Tensor:
    values = values.detach().cpu().float()
    if vmin is None:
        vmin = float(values.min().item())
    if vmax is None:
        vmax = float(values.max().item())
    scale = max(vmax - vmin, 1e-12)
    return ((values - vmin) / scale).clamp(0.0, 1.0)


def heatmap_pil(values: torch.Tensor, size: Optional[Tuple[int, int]] = None, vmin=None, vmax=None) -> Image.Image:
    """Small blue-red heatmap without matplotlib dependency."""
    values = normalize_map(values, vmin=vmin, vmax=vmax)
    r = torch.clamp(1.5 * values - 0.25, 0.0, 1.0)
    g = torch.clamp(1.5 - 3.0 * torch.abs(values - 0.5), 0.0, 1.0)
    b = torch.clamp(1.25 - 1.5 * values, 0.0, 1.0)
    rgb = torch.stack([r, g, b], dim=-1)
    array = (rgb.numpy() * 255.0 + 0.5).astype("uint8")
    image = Image.fromarray(array, mode="RGB")
    if size is not None:
        image = image.resize(size, Image.BICUBIC)
    return image


def overlay_heatmap(base: Image.Image, values: torch.Tensor, alpha: float = 0.45, vmin=None, vmax=None) -> Image.Image:
    heat = heatmap_pil(values, size=base.size, vmin=vmin, vmax=vmax)
    return Image.blend(base.convert("RGB"), heat, alpha)


def draw_label(image: Image.Image, text: str, pad: int = 5) -> Image.Image:
    image = image.convert("RGB")
    label_h = 22
    canvas = Image.new("RGB", (image.width, image.height + label_h), (255, 255, 255))
    canvas.paste(image, (0, label_h))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, image.width, label_h), fill=(245, 245, 245))
    draw.text((pad, 4), str(text), fill=(20, 20, 20))
    return canvas


def make_panel_grid(images: Sequence[Image.Image], labels: Sequence[str], columns: int = 3, gap: int = 8) -> Image.Image:
    if not images:
        raise ValueError("No images were provided.")
    labeled = [draw_label(img, label) for img, label in zip(images, labels)]
    w = max(img.width for img in labeled)
    h = max(img.height for img in labeled)
    rows = (len(labeled) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * w + (columns - 1) * gap, rows * h + (rows - 1) * gap), (255, 255, 255))
    for idx, img in enumerate(labeled):
        row = idx // columns
        col = idx % columns
        x = col * (w + gap)
        y = row * (h + gap)
        canvas.paste(img, (x, y))
    return canvas


def make_unlabeled_panel_grid(images: Sequence[Image.Image], columns: int = 3, gap: int = 8) -> Image.Image:
    if not images:
        raise ValueError("No images were provided.")
    converted = [img.convert("RGB") for img in images]
    w = max(img.width for img in converted)
    h = max(img.height for img in converted)
    rows = (len(converted) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * w + (columns - 1) * gap, rows * h + (rows - 1) * gap), (255, 255, 255))
    for idx, img in enumerate(converted):
        row = idx // columns
        col = idx % columns
        x = col * (w + gap)
        y = row * (h + gap)
        canvas.paste(img, (x, y))
    return canvas


def draw_norm_box(image: Image.Image, box_norm_xyxy: Sequence[float], color=(0, 255, 0), width: int = 3) -> Image.Image:
    image = image.convert("RGB").copy()
    x0, y0, x1, y1 = box_norm_xyxy
    box = [x0 * image.width, y0 * image.height, x1 * image.width, y1 * image.height]
    draw = ImageDraw.Draw(image)
    for offset in range(width):
        draw.rectangle(
            [box[0] - offset, box[1] - offset, box[2] + offset, box[3] + offset],
            outline=color,
        )
    return image


def patch_grid_point(index: int, grid_w: int, size: Tuple[int, int], grid_h: Optional[int] = None) -> Tuple[float, float]:
    grid_h = grid_w if grid_h is None else grid_h
    row = int(index) // grid_w
    col = int(index) % grid_w
    return ((col + 0.5) / grid_w * size[0], (row + 0.5) / grid_h * size[1])


def draw_patch_points(
    image: Image.Image,
    indices: Iterable[int],
    grid_h: int,
    grid_w: int,
    color=(255, 0, 0),
    radius: int = 4,
) -> Image.Image:
    image = image.convert("RGB").copy()
    draw = ImageDraw.Draw(image)
    for idx in indices:
        row = int(idx) // grid_w
        col = int(idx) % grid_w
        x = (col + 0.5) / grid_w * image.width
        y = (row + 0.5) / grid_h * image.height
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=(255, 255, 255))
    return image


def patch_values_to_image(values: torch.Tensor, grid_h: int, grid_w: int, size: Tuple[int, int]) -> Image.Image:
    values = values.detach().cpu().float().view(grid_h, grid_w)
    return heatmap_pil(values, size=size)


def save_manifest_csv(path: Path, rows):
    if not rows:
        return
    import csv
    from collections import OrderedDict

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(OrderedDict((key, None) for row in rows for key in row.keys()).keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_visualization_pdf(image: Image.Image, path, scale: int = 3, resolution: float = 300.0) -> Path:
    path = Path(path)
    if path.suffix.lower() != ".pdf":
        path = path.with_suffix(".pdf")
    path.parent.mkdir(parents=True, exist_ok=True)
    image = image.convert("RGB")
    if scale > 1:
        image = image.resize((image.width * scale, image.height * scale), Image.BICUBIC)
    image.save(path, "PDF", resolution=resolution)
    return path


def save_visualization_manifest(directory, rows):
    directory = Path(directory)
    save_manifest_csv(directory / "manifest.csv", rows)
    save_manifest_csv(directory / "manifest_pdf.csv", rows)


def load_visualization_indices(path) -> Sequence[int]:
    """Load locked visualization sample indices from txt, json, csv, or pt.

    CSV files may use a `sample_index` column, which matches the manifests
    written by these visualization helpers. Plain text files accept one integer
    per line. JSON files accept either a list of integers or a list of objects
    with `sample_index`.
    """
    path = Path(path)
    if not path:
        return []
    suffix = path.suffix.lower()
    if suffix in (".pt", ".pth"):
        data = torch.load(path, map_location="cpu")
        if torch.is_tensor(data):
            return [int(v) for v in data.flatten().tolist()]
        if isinstance(data, dict):
            for key in ("sample_index", "sample_indices", "indices"):
                if key in data:
                    values = data[key]
                    if torch.is_tensor(values):
                        values = values.flatten().tolist()
                    return [int(v) for v in values]
        return [int(v) for v in data]
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            for key in ("sample_index", "sample_indices", "indices", "rows"):
                if key in data:
                    data = data[key]
                    break
        if data and isinstance(data[0], dict):
            return [int(row["sample_index"]) for row in data]
        return [int(v) for v in data]
    if suffix == ".csv":
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                if "sample_index" in reader.fieldnames:
                    return [int(row["sample_index"]) for row in reader if row.get("sample_index", "") != ""]
                first_col = reader.fieldnames[0]
                return [int(row[first_col]) for row in reader if row.get(first_col, "") != ""]

    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        values.append(int(line.split(",")[0]))
    return values


def save_visualization_indices(path, indices: Sequence[int]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    unique = []
    seen = set()
    for value in indices:
        value = int(value)
        if value not in seen:
            unique.append(value)
            seen.add(value)
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(unique, indent=2) + "\n", encoding="utf-8")
    else:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["sample_index"])
            writer.writeheader()
            for value in unique:
                writer.writerow({"sample_index": value})
    return path
