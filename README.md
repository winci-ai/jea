# The Locality Trade-off in Masked Patch Self-Distillation

**NeurIPS 2026**

Xi Weng<sup>1</sup>, Zhaoyu Liu<sup>1</sup>, Lianyu Hu<sup>2</sup>, Jin Song Dong<sup>1</sup>

<sup>1</sup>School of Computing, National University of Singapore &nbsp;&nbsp; <sup>2</sup>Nanyang Technological University

Official PyTorch implementation of **"The Locality Trade-off in Masked Patch Self-Distillation"**, published at NeurIPS 2026.

Masked patch self-distillation (iBOT) is a core ingredient of DINOv2 and DINOv3. This repository contains everything needed to study what it does to *local* visual information:

- **Controlled pretraining** of **DINO** and **DINO+iBOT** (ViT-S/16, B/16, L/16, H/16; ImageNet-1K, 300 epochs). Within each scale, the only difference is the iBOT branch and its student masking.
- **Gram anchoring** experiments (standard Gram and **DINO-Gram**) on ViT-B/16 and ViT-L/16.
- **Eight downstream benchmarks**: ImageNet k-NN, RParis/ROxford retrieval, VOC2012 segmentation, VOC2007 detection, Copydays copy detection, DAVIS-2017 tracking, Sintel optical flow, plus patch effective rank.
- **Two annotation-free diagnostics**:
  - **Local Appearance Decodability**: how much pixel and texture evidence a linear probe can recover from frozen patch tokens.
  - **Local-to-Global Correspondence**: whether patches of a crop can be matched to, and used to localize, the crop inside its source image.
- **Visualization tools** for the paper's qualitative figures (DAVIS masks, decoded appearance, correspondence matches).

### Key findings

| | ViT-S/16, ViT-B/16 | ViT-L/16, ViT-H/16 |
|---|---|---|
| Semantic tasks (k-NN, VOC seg./det.) | iBOT helps | iBOT helps (more) |
| DAVIS tracking | iBOT hurts (B/16: 61.8 → 55.5 J&F) | iBOT slightly helps |
| Local appearance decodability | Largely preserved | **Drops in deep layers** |
| Local-to-global correspondence | **Drops substantially** | Preserved or improved |

Scaling changes the *form* of the locality cost instead of removing it. Gram anchoring shows that preserving patch relations does not restore both properties at once: for ViT-B/16, DINO-Gram mainly restores correspondence; for ViT-L/16, it mainly restores deep-layer appearance decodability.

---

## Contents

- [Code map](#code-map)
- [1. Installation](#1-installation)
- [2. Data preparation](#2-data-preparation)
- [3. Pretraining: DINO vs. DINO+iBOT](#3-pretraining-dino-vs-dinoibot)
- [4. Gram anchoring](#4-gram-anchoring)
- [5. Downstream evaluation](#5-downstream-evaluation)
- [6. Local Appearance Decodability](#6-local-appearance-decodability)
- [7. Local-to-Global Correspondence](#7-local-to-global-correspondence)
- [8. Qualitative visualizations](#8-qualitative-visualizations)
- [9. Collecting results](#9-collecting-results)
- [10. Reference results](#10-reference-results)
- [11. Reproducibility notes and limitations](#11-reproducibility-notes-and-limitations)
- [Acknowledgments, license and citation](#acknowledgments)

## Code map

| Paper | Code |
|---|---|
| DINO class-token loss with Sinkhorn-Knopp targets (Sec. 2.1, App. A) | [`jea/loss/dino_clstoken_loss.py`](jea/loss/dino_clstoken_loss.py) |
| iBOT masked patch cross-entropy loss (Sec. 2.2) | [`jea/loss/ibot_patch_loss.py`](jea/loss/ibot_patch_loss.py) |
| KoLeo regularizer | [`jea/loss/koleo_loss.py`](jea/loss/koleo_loss.py) |
| Gram anchoring loss (Sec. 5, App. B.2.3) | [`jea/loss/gram_loss.py`](jea/loss/gram_loss.py) |
| Student/teacher, heads, loss assembly, EMA, Gram teacher | [`jea/train/ssl_meta_arch.py`](jea/train/ssl_meta_arch.py) |
| Training loop and schedules | [`jea/train/train.py`](jea/train/train.py) |
| iBOT block masking | [`jea/data/masking.py`](jea/data/masking.py), [`jea/data/collate.py`](jea/data/collate.py) |
| Pretraining configs (App. B.2.1) | [`jea/configs/`](jea/configs) |
| Downstream benchmarks (Sec. 3, App. B.2.2) | [`jea/eval/`](jea/eval) |
| Patch effective rank | [`jea/eval/cal_metrics.py`](jea/eval/cal_metrics.py) |
| Local Appearance Decodability (Sec. 4.1, App. F) | [`jea/diagnostics/local_detail_decodability.py`](jea/diagnostics/local_detail_decodability.py) |
| Local-to-Global Correspondence (Sec. 4.2, App. G) | [`jea/diagnostics/local_global_correspondence.py`](jea/diagnostics/local_global_correspondence.py) |
| Qualitative comparisons (Figs. 3, 6 and App. E–G) | [`jea/diagnostics/*_comparison.py`](jea/diagnostics), [`*_compare_visualization.py`](jea/diagnostics) |

```text
jea/
  configs/ssl_default_config.yaml     shared defaults
  configs/train/dino/                 DINO        vit_{s,b,l,h}16.yaml
  configs/train/dino_ibot/            DINO+iBOT   vit_{s,b,l,h}16.yaml
                                      Gram        vit_{b,l}16_gram.yaml, vit_{b,l}16_dino_gram.yaml
  data/  layers/  models/             data pipeline, ViT with registers, projection heads
  loss/  train/  fsdp/                losses, meta-architecture, FSDP2 runtime and checkpoints
  eval/                               downstream benchmarks and patch effective rank
  diagnostics/                        the two diagnostics and comparison visualizations
scripts/
  train.sh, train_gram.sh             pretraining and Gram continuation launchers
  evaluate.py                         paper evaluation presets over exported checkpoints
  <task>/run_all.sh                   per-task checkpoint sweeps with CSV aggregation
  *_comparison.sh, *_visualization.sh qualitative side-by-side figures
  prepare_imagenet.py, prepare_copydays.py, setup_dense_eval.sh, collect_summaries.py
```

The Python package is named `jea`. In configs and checkpoints, `jea` / `jea_head` denote the **DINO class-token branch** and `ibot` / `ibot_head` denote the **iBOT patch branch**.

---

## 1. Installation

Experiments were run on Linux with CUDA. Run all commands from the repository root.

### 1.1 Training, diagnostics, k-NN, retrieval, copy detection, DAVIS, patch rank

```bash
conda create -n jea python=3.10 -y
conda activate jea
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -e .
```

Python ≥ 3.10 and PyTorch 2.7 are required: the trainer uses FSDP2 (`fully_shard`) and native scaled-dot-product attention. xFormers is not needed. Weights & Biases logging is disabled by default; enable it with `wandb.mode=online`.

### 1.2 VOC segmentation and detection (separate environment)

The OpenMMLab stack needs an older PyTorch, so use a **separate** environment:

```bash
conda create -n locality-dense python=3.10 -y
conda activate locality-dense
bash scripts/setup_dense_eval.sh
```

This installs PyTorch 2.1.2 / torchvision 0.16.2 (CUDA 11.8), MMEngine 0.10.7, MMCV 2.1.0, MMSegmentation 1.2.2 and MMDetection 3.3.0, then installs this package with `--no-deps`. Do not run pretraining in this environment.

### 1.3 Optical flow

The Sintel experiments use a modified MMFlow. Install it in the dense-evaluation environment:

```bash
conda activate locality-dense
git clone https://winci-ai@bitbucket.org/winci88/mmflow.git
cd mmflow
pip install -e .
```

---

## 2. Data preparation

Download each dataset from its official provider under its own terms, then arrange it as below. Symbolic links are fine.

```text
datasets/
  ImageNet/                       ImageNet-1K (pretraining, k-NN, patch rank, decodability probes)
    train/<synset>/*.JPEG
    val/<synset>/*.JPEG           validation images sorted into class folders
    labels.txt                    "synset,class_name" per line, sorted class order
    entries-*.npy, ...            generated by prepare_imagenet.py
  VOCdevkit/
    VOC2007/{JPEGImages,Annotations,ImageSets/Main}/
    VOC2012/{JPEGImages,SegmentationClass,ImageSets/Segmentation}/
  revisited_paris_oxford/
    rparis6k/{jpg/,gnd_rparis6k.pkl}
    roxford5k/{jpg/,gnd_roxford5k.pkl}
  copydays/                       copy detection and the correspondence diagnostic
    original/  strong/  jpegqual/{3,...,75}/  crops/{10,...,80}/
    whitening_images/             20,000 COCO images
    distractors_images/           10,000 COCO images
  davis/
    JPEGImages/480p/<sequence>/
    Annotations/480p/<sequence>/
    ImageSets/2017/val.txt
  sintel/
    training/{clean,final,flow,occlusions,invalid}/
```

**ImageNet metadata.** Generate it once:

```bash
python scripts/prepare_imagenet.py --root datasets/ImageNet --extra datasets/ImageNet
```

Images and metadata may live in different places. Datasets are specified as `ImageNet:split=TRAIN:root=/data/ImageNet:extra=/data/ImageNet-meta`.

**Copydays distractors and whitening set.** The paper uses 10,000 COCO distractors and 20,000 COCO whitening images. To build two disjoint, deterministic pools and save their file lists:

```bash
python scripts/prepare_copydays.py --coco-root /data/coco/train2017 --copydays-root datasets/copydays
```

The exact COCO lists used in the paper are not distributed, so copy-detection scores can differ slightly. Use the same pools for all models you compare.

**Dense-task paths** are set in `jea/eval/{segmentation,object_detection,optical_flow}/configs/datasets/`. Edit them if your layout differs.

---

## 3. Pretraining: DINO vs. DINO+iBOT

*Paper: Section 3 and Appendix B.2.1.*

### 3.1 Objectives

Both models use the DINO class-token objective with **Sinkhorn-Knopp** teacher targets (as in SwAV and DINOv2) and a KoLeo regularizer on the class token. **DINO+iBOT** adds the iBOT masked patch objective:

- The student sees masked global crops. Masked patch embeddings are replaced by a learnable mask token. The teacher sees the unmasked crops. Local crops are never masked.
- At each masked position, the student patch distribution is trained to match the teacher's Sinkhorn-Knopp normalized patch distribution with cross-entropy. The loss is averaged over masked patches within each crop, then over global crops and images.
- Following DINOv2, class tokens and patch tokens use **separate projection heads**.

For DINO+iBOT, the class-token loss uses the same masked student forward pass. The intervention therefore consists of patch supervision *and* student masking.

```text
L_DINO        = L_DINO-cls + 0.1 * L_KoLeo
L_DINO+iBOT   = L_DINO-cls + 1.0 * L_iBOT + 0.1 * L_KoLeo
```

### 3.2 Settings

| | ViT-S/16 | ViT-B/16 | ViT-L/16 | ViT-H/16 |
|---|---:|---:|---:|---:|
| Blocks / width | 12 / 384 | 12 / 768 | 24 / 1024 | 32 / 1280 |
| Global batch (4 GPUs) | 1024 | 1024 | 1024 | 512 |
| Batch per GPU | 256 | 256 | 256 | 128 |
| Peak LR | 4e-3 | 4e-3 | 4e-3 | ≈2.83e-3 |
| Drop path (uniform) | 0.3 | 0.3 | 0.4 | 0.4 |

Shared settings:

| Component | Setting |
|---|---|
| Architecture | Patch size 16, SwiGLU FFN, 4 register tokens, LayerScale 1e-5 |
| Heads (DINO and iBOT) | 3-layer MLP, hidden 2048, bottleneck 256, **131,072 prototypes**; last layer frozen for the first epoch |
| Temperatures | Student 0.1; teacher warmed up linearly 0.04 → 0.07 over 30 epochs |
| iBOT | Loss weight **1.0 throughout training**; masking probability 0.5; masking ratio sampled in [0.1, 0.5] |
| KoLeo | Weight 0.1, on class tokens |
| EMA teacher | Momentum 0.994 → 1.0, cosine schedule |
| Optimizer | AdamW, β = (0.9, 0.999), 300 epochs, 10-epoch warmup, cosine decay to 1e-6 |
| Learning rate | `4e-3 * sqrt(global_batch / 1024)` |
| Weight decay | 0.04 → 0.4, cosine schedule |
| Other | Gradient clip 3.0, patch-embedding LR multiplier 0.2, layer-wise LR decay 0.9 |
| Multi-crop | 2 global crops 224², scale [0.32, 1.0]; 8 local crops 96², scale [0.05, 0.32]; DINO's asymmetric color/blur/solarize augmentation |
| Precision | bf16 mixed precision, FSDP |

The paired configs in `jea/configs/train/dino/` and `jea/configs/train/dino_ibot/` differ **only** in `ibot.use_loss` (and run names).

### 3.3 Launching the eight runs

The paper uses one node with four GPUs per run (NVIDIA H200):

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
bash scripts/train.sh dino      s
bash scripts/train.sh dino_ibot s
bash scripts/train.sh dino      b
bash scripts/train.sh dino_ibot b
bash scripts/train.sh dino      l
bash scripts/train.sh dino_ibot l
bash scripts/train.sh dino      h
bash scripts/train.sh dino_ibot h
```

Runs are written to `out/dino_vit_{s,b,l,h}16/` and `out/dino_ibot_vit_{s,b,l,h}16/`. Extra arguments are OmegaConf `key=value` overrides:

```bash
bash scripts/train.sh dino_ibot b \
  train.dataset_path=ImageNet:split=TRAIN:root=/data/ImageNet:extra=/data/ImageNet \
  train.output_dir=/checkpoints wandb.mode=offline
```

The number of processes follows `CUDA_VISIBLE_DEVICES`. If you use a different GPU count, set `train.batch_size_per_gpu` so the global batch stays the same. No gradient accumulation is applied. For multi-node jobs, call the module directly:

```bash
torchrun --nproc_per_node=4 -m jea.train.train \
  --config-file jea/configs/train/dino_ibot/vit_b16.yaml --output-dir out/my_run
```

### 3.4 Checkpoints

- `out/<run>/config.yaml`: the fully merged config.
- `out/<run>/eval/training_<step>/teacher_checkpoint.pth`: the **EMA teacher**, exported every 10 epochs (`{"teacher": state_dict}`). All evaluations use these files.
- `out/<run>/model_epoch_0260`: full FSDP2 training checkpoint (student, EMA teacher, optimizer) at epoch 260, from which the Gram runs continue. The Gram anchor itself is the epoch-250 teacher export above (`eval/training_312749/teacher_checkpoint.pth` for S/B/L), not a full checkpoint. Periodic training checkpoints are tracked by `last_checkpoint`, and runs resume from it automatically.

Step indices are zero-based: `step = epoch * epoch_length - 1`, where `epoch_length = floor(1,281,167 / global_batch)`.

| Backbone | Steps / epoch | Epoch 250 | Epoch 260 | Epoch 300 |
|---|---:|---:|---:|---:|
| S/16, B/16, L/16 | 1251 | `training_312749` | `training_325259` | `training_375299` |
| H/16 | 2502 | `training_625499` | `training_650519` | `training_750599` |

### 3.5 Compute

Approximate per-GPU peak memory and wall-clock time of the original 300-epoch runs (4 × H200, bf16). The eight runs total about 3,964 GPU-hours.

| Backbone | DINO | DINO+iBOT |
|---|---|---|
| ViT-S/16 | 28 GB, 1 d 7 h | 100 GB, 2 d 5 h |
| ViT-B/16 | 47 GB, 3 d 1 h | 125 GB, 3 d 8 h |
| ViT-L/16 | 100 GB, 5 d 15 h | 129 GB, 6 d 5 h |
| ViT-H/16 (batch 512) | 86 GB, 9 d 8 h | 134 GB, 10 d 6 h |

Most of iBOT's extra memory comes from the 131,072-way patch logits and targets for masked patches.

---

## 4. Gram anchoring

*Paper: Section 5 and Appendix B.2.3.*

We compare three ViT-B/16 and ViT-L/16 models: DINO+iBOT, DINO+iBOT with standard Gram anchoring, and DINO+iBOT with **DINO-Gram**.

| | Standard Gram | DINO-Gram |
|---|---|---|
| Initial Gram teacher | DINO+iBOT EMA teacher, epoch 250 | DINO EMA teacher, epoch 250 |
| Student, EMA, optimizer | Continued from DINO+iBOT at epoch 260 | Same |
| Gram weight | 0 before epoch 260, cosine ramp 0 → 2 over one epoch, then 2 | Same |
| Gram-teacher refresh | Copy of the current EMA teacher at epochs 270, 280, 290 | Same |

Gram loss on backbone patch features (not head outputs) of each global crop:

```text
L_Gram = (1/N²) * || Ẑ_s Ẑ_sᵀ − Ẑ_g Ẑ_gᵀ ||_F²        (Ẑ = row-wise l2-normalized patch features)
L      = L_DINO-cls + L_iBOT + 0.1 * L_KoLeo + λ_Gram(t) * L_Gram
```

The student uses the masked global-crop forward pass, and the Gram teacher sees the unmasked crops. All patch positions enter both Gram matrices. Features are not centered and negative similarities are not clipped. Only the student receives gradients.

After the two baselines have reached epoch 260:

```bash
# ViT-B/16
bash scripts/train_gram.sh b gram \
  out/dino_ibot_vit_b16/model_epoch_0260 \
  out/dino_ibot_vit_b16/eval/training_312749/teacher_checkpoint.pth
bash scripts/train_gram.sh b dino_gram \
  out/dino_ibot_vit_b16/model_epoch_0260 \
  out/dino_vit_b16/eval/training_312749/teacher_checkpoint.pth

# ViT-L/16
bash scripts/train_gram.sh l gram \
  out/dino_ibot_vit_l16/model_epoch_0260 \
  out/dino_ibot_vit_l16/eval/training_312749/teacher_checkpoint.pth
bash scripts/train_gram.sh l dino_gram \
  out/dino_ibot_vit_l16/model_epoch_0260 \
  out/dino_vit_l16/eval/training_312749/teacher_checkpoint.pth
```

Arguments: backbone, variant, the DINO+iBOT epoch-260 FSDP2 checkpoint to continue from, and the epoch-250 teacher used as the initial anchor (exported through the `GRAM_TEACHER_CHECKPOINT` variable read by the config). Outputs go to `out/dino_ibot_vit_{b,l}16_{gram,dino_gram}/`, which export teachers at epochs 270–300. Restarts reload the run's own saved Gram teacher.

For late-training trajectories (epochs 250–300), combine epochs 250/260 of the DINO+iBOT baseline with epochs 270–300 of each Gram run.

---

## 5. Downstream evaluation

*Paper: Section 3 and Appendix B.2.2.* All evaluations use the frozen EMA teacher.

[`scripts/evaluate.py`](scripts/evaluate.py) selects exported checkpoints by epoch, reads the architecture from `<run>/config.yaml`, applies the paper's settings and runs `scripts/<task>/run_all.sh`.

```bash
RUN=out/dino_ibot_vit_b16

# Training environment
python scripts/evaluate.py knn                      --run-dir $RUN
python scripts/evaluate.py image_retrieval          --run-dir $RUN   # RParis6k and ROxford5k
python scripts/evaluate.py copy_detection           --run-dir $RUN
python scripts/evaluate.py davis_video_segmentation --run-dir $RUN
python scripts/evaluate.py cal_metrics              --run-dir $RUN   # patch effective rank

# Dense-evaluation environment (locality-dense)
python scripts/evaluate.py segmentation             --run-dir $RUN
python scripts/evaluate.py object_detection         --run-dir $RUN
python scripts/evaluate.py optical_flow             --run-dir $RUN --epochs 260   # needs MMFlow (Sec. 1.3)
```

Common options:

- `--epochs 300` (default), `--epochs 250,260,300` or `--epochs all` (all exports, for trajectories).
- `--gpu 0,1,2,3` (default). Dense tasks keep the total batch at 16 by setting the per-GPU batch to `16 / #GPUs`.
- `--imagenet-root`, `--imagenet-extra`, `--data-path`: dataset locations.
- `--dry-run`: print the commands without running them.
- Arguments after `--` are passed to `run_all.sh` (see `bash scripts/<task>/run_all.sh --help`).

Each task writes per-checkpoint outputs and `out/<run>/<task>_all/summary.csv`. Retrieval writes `rparis6k/` and `roxford5k/` subfolders.

### Protocols

| Benchmark | Metric | Protocol |
|---|---|---|
| ImageNet k-NN | Top-1 | Class token; weighted k-NN with cosine similarity, k = 15, temperature 0.07; train → val; resize 256, center crop 224 |
| RParis6k / ROxford5k | mAP, Hard | 224² input; descriptor = [CLS, GeM(p = 4) over patch tokens]; no whitening or fine-tuning |
| VOC2012 segmentation | mIoU | Frozen backbone at 512²; layer-normed outputs of 4 evenly spaced block chunks concatenated (1536/3072/4096/5120 channels for S/B/L/H); linear FCN head; 10k iterations, AdamW, LR 5e-4, WD 0, batch 16, 200-iteration warmup, poly decay; scale jitter [0.5, 2], 512² crops, flip, photometric distortion; sliding-window test 512 / stride 341; train → val |
| VOC2007 detection | mAP (11-point) | Frozen backbone, FPN on 4 block-chunk outputs; Cascade R-CNN with 4conv-1fc GIoU heads; 1000 × 600 resize, flip; AdamW, LR 1e-4, WD 0.05, batch 16, 500-iteration warmup; 12 epochs, decay at 8 and 11; train → val |
| Copydays | mAP, strong | 224² input; [CLS, GeM(4)] descriptor; centering + PCA whitening (exponent 0.5) fitted on 20k COCO images; 10k COCO distractors; cosine retrieval |
| DAVIS-2017 | J&F | DINO label propagation: first frame + 7 preceding frames; neighborhood radius 12 patches; top-5; temperature 0.1; short side 480, long side rounded to a multiple of 64; official DAVIS evaluation on val |
| Sintel | Clean EPE | Frozen backbone + FPN (channels 32/64/96/128 at levels 2–5) + PWCNet decoder; 12 epochs on clean+final training passes; 384 × 768 crops; AdamW, LR 1e-4, WD 0.05, batch 16; evaluated on the clean pass **of the same training videos** |
| Patch effective rank | Normalized rank × 100 | For each of 10k ImageNet-train images (subset seed 0): singular values of the uncentered patch-token matrix, normalized to sum to one; `exp(entropy) / #patches`; mean over images |

Notes:

- **Optical flow** measures how well a flow head fits frozen features on its training videos, not held-out generalization. The paper's paired table uses the latest common checkpoint with finite clean EPE ≤ 100: epochs **180 / 260 / 300 / 270** for S / B / L / H. All other benchmarks use epoch 300.
- The dense adapters (segmentation, detection, flow) feed CLS + patch tokens from four block chunks and do not insert the pretraining register tokens. Keep this unchanged to match the paper's protocol.
- RParis and ROxford are reported separately because they respond differently to iBOT.

---

## 6. Local Appearance Decodability

*Paper: Section 4.1 and Appendix F.*

How much local appearance can a **ridge-regression probe** recover from frozen patch tokens?

**Data.** 20,000 ImageNet-1K *training* images (subset seed 0) are split at the image level into 10,000 probe-training and 10,000 held-out images. Each image is resized (shorter side 256, bicubic) and center-cropped to 224. The same crop feeds the backbone (ImageNet-normalized) and the target computation. All 196 patches per image are used, giving 1.96 M token–target pairs per split.

**Targets.** Six descriptor groups per 16×16 patch (122 dimensions in total):

| Group | Dim. | Description |
|---|---:|---|
| `rgb_mean` | 3 | Mean R, G, B |
| `rgb_hist` | 24 | 8-bin histogram per channel |
| `hog` | 9 | Magnitude-weighted unsigned orientation histogram, Sobel gradients on the full image, pooled per patch |
| `dct_energy` | 6 | Log mean energy in 6 radial bands of the orthonormal 2-D DCT (DC excluded) |
| `lbp_hist` | 16 | 8-neighbor LBP codes on the full image, coarsened to 16 bins, pooled per patch |
| `raw_pca` | 64 | Projection of the flattened RGB patch onto the top 64 PCA components fitted on probe-training patches |

**Probe.** Features are standardized and targets centered, using probe-training statistics only. A closed-form ridge decoder with `ridge_alpha = 1e-3` is fitted per backbone and layer. Scores are held-out R², computed per target dimension and then averaged within each group (`R²_g,mean`). Negative values are kept.

**Layers.** 1-based blocks 3/6/9/12 (S, B), 6/12/18/24 (L) and 8/16/24/32 (H). In code these are zero-based: `2,5,8,11`, `5,11,17,23`, `7,15,23,31`. `--layers paper` selects them.

```bash
# Layer sweep at epoch 300 (Fig. 4)
python scripts/evaluate.py local_detail_decodability --run-dir out/dino_ibot_vit_b16 --layers paper

# Final-layer training trajectory (App. F.5)
python scripts/evaluate.py local_detail_decodability --run-dir out/dino_ibot_vit_b16 --epochs all --layers final

# Also save the fitted probe and decoded visualizations
python scripts/evaluate.py local_detail_decodability --run-dir out/dino_ibot_vit_b16 --layers final \
  -- --save-probe --save-visualizations
```

R² measures *linear accessibility*. A lower score does not prove that the information is gone, and a higher score does not imply better matching. The MAE and SimMIM reference curves in Fig. 4 come from separately pretrained ViT-B/16 checkpoints evaluated with the same probe. Their checkpoints are not included here.

---

## 7. Local-to-Global Correspondence

*Paper: Section 4.2 and Appendix G.*

Can patch similarities locate a cropped region inside its known source image?

**Queries.** Four crops from each of the 157 Copydays source images (628 queries, seed 0). Each crop covers about ρ of the image area (ρ = 0.1 by default; 0.1/0.2/0.3/0.5 for the ratio study) with aspect ratio log-uniform in [0.75, 1.25] and a uniformly random position. The crop and the full image are each resized to 224², and patch tokens are taken from a frozen layer. Crop specifications are fixed by the seed, so every model sees identical queries.

**Metrics** (computed per query from the cosine similarity matrix between crop and source patches, then averaged):

| Metric | Definition |
|---|---|
| Patch matching accuracy | Fraction of query patches whose top-1 source match lies inside the true crop region (region membership, not exact point correspondence) |
| Region IoU | Rank source patches by their maximum similarity to any query patch, take as many as the true region contains, and compute the IoU with that region |
| RANSAC box IoU | Each top-1 match votes for the crop's top-left corner given the known crop size. The vote with the largest consensus wins (≤ 128 candidates, L∞ tolerance 1.5 patches), and the resulting box is compared with the true box |
| IoU₅₀ success | Fraction of queries with box IoU ≥ 0.5 |

```bash
# Layer sweep at epoch 300, ratio 0.1 (Fig. 5)
python scripts/evaluate.py local_global_correspondence --run-dir out/dino_ibot_vit_b16 --layers paper

# Crop-ratio comparison (App. G.4)
python scripts/evaluate.py local_global_correspondence --run-dir out/dino_ibot_vit_b16 --layers final \
  --crop-ratios 0.1,0.2,0.3,0.5

# Training trajectory with saved match visualizations
python scripts/evaluate.py local_global_correspondence --run-dir out/dino_ibot_vit_b16 --epochs all --layers final \
  -- --save-visualizations
```

To lock queries explicitly, use `--write-crop-specs-json` and `--selected-crop-specs-json` (pass them after `--`). The paper reports a single sampling run (seed 0).

---

## 8. Qualitative visualizations

These scripts build side-by-side figures from evaluated runs. Defaults target ViT-B/16 at epoch 300 (`375299`). For ViT-H/16, set `checkpoint_step=750599` / `checkpoint_steps=750599`.

```bash
# DAVIS tracking masks (Fig. 3); run the DAVIS evaluation first
model_roots=out/dino_vit_b16,out/dino_ibot_vit_b16 video=dogs-jump frame_indices=0,5,10,15,20 \
  bash scripts/davis_video_segmentation_mask_comparison.sh

# Decoded RGB mean / HOG / DCT / raw-PCA maps (App. F.6)
model_roots=out/dino_vit_b16,out/dino_ibot_vit_b16 \
  bash scripts/local_detail_decodability_compare_visualization.sh

# Correspondence boxes, heatmaps and matches (Fig. 6); uses saved local_global_correspondence results
model_roots=out/dino_vit_b16,out/dino_ibot_vit_b16 \
  bash scripts/local_global_correspondence_comparison.sh
# For L/16 or H/16, set the last-block index, e.g. layers=23 (L) or layers=31 (H).
```

Non-RGB decoding maps show per-patch descriptor norms, each min-max normalized within its panel. They are not RGB reconstructions.

---

## 9. Collecting results

```bash
python scripts/collect_summaries.py --source out --output summaries   # copy every summary.csv
python scripts/collect_summaries.py --source out --check-only         # report missing/invalid summaries
```

CSV files store raw units. The paper multiplies retrieval/copy-detection mAP, detection mAP, DAVIS J&F, normalized patch rank and the diagnostic scores by 100. Segmentation mIoU and k-NN accuracy are already percentages, and EPE is in pixels.

---

## 10. Reference results

Paper results at epoch 300 (Sintel EPE at epochs 180/260/300/270 for S/B/L/H). Δ = DINO+iBOT − DINO.

| Metric | S: DINO | S: +iBOT | B: DINO | B: +iBOT | L: DINO | L: +iBOT | H: DINO | H: +iBOT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ImageNet k-NN top-1 ↑ | 70.88 | 71.27 | 76.36 | 76.97 | 79.36 | 80.44 | 74.45 | 78.11 |
| VOC2012 seg. mIoU ↑ | 53.13 | 56.02 | 65.61 | 67.88 | 71.93 | 79.63 | 72.88 | 81.95 |
| VOC2007 det. mAP ↑ | 42.88 | 43.96 | 52.64 | 54.29 | 59.71 | 69.08 | 58.89 | 68.94 |
| RParis Hard mAP ↑ | 29.75 | 30.78 | 37.24 | 34.87 | 40.81 | 44.11 | 35.10 | 38.96 |
| ROxford Hard mAP ↑ | 13.34 | 12.03 | 14.60 | 13.65 | 12.86 | 11.95 | 9.05 | 9.07 |
| Copydays strong mAP ↑ | 84.14 | 83.69 | 87.61 | 87.13 | 89.74 | 90.62 | 88.74 | 88.26 |
| DAVIS J&F ↑ | 60.10 | 59.30 | 61.80 | 55.50 | 62.90 | 63.10 | 62.20 | 62.60 |
| Sintel clean EPE ↓ | 18.19 | 18.26 | 24.44 | 24.04 | 22.84 | 23.57 | 24.73 | 26.66 |
| Normalized patch rank | 69.10 | 62.85 | 70.57 | 62.44 | 66.42 | 58.83 | 63.88 | 54.60 |

**Correspondence** (final block, ρ = 0.1): adding iBOT changes patch matching accuracy by about −11.8 (S), −16.9 (B), +3.3 (L) and +4.0 (H) points.

**Gram anchoring, ViT-B/16, epoch 300:**

| | DINO+iBOT | + Gram | + DINO-Gram |
|---|---:|---:|---:|
| DAVIS J&F | 55.5 | 57.6 | 61.4 |
| Final-layer patch matching accuracy (ρ = 0.1) | 24.8 | 30.1 | 40.8 |

See the paper for layer sweeps, trajectories and the ViT-L/16 Gram results.

---

## 11. Reproducibility notes and limitations

- **Runtime.** The paper's models were trained with an earlier FSDP1 version of this codebase. This release implements the same recipe on **PyTorch FSDP2 with native SDPA**. Kernel choice, floating-point order, activation checkpointing and checkpoint format differ, so numbers will not be bit-identical, and the resource figures above were measured on the original runtime.
- **Old checkpoints.** FSDP1 sharded checkpoints cannot be resumed here. Exported `teacher_checkpoint.pth` files (full weights) work for all evaluations and as Gram anchors. `student.pretrained_weights` initializes only the backbone, which is not equivalent to resuming training.
- **Single runs.** Pretraining and correspondence sampling were not repeated across seeds. Small differences should not be read as statistically established.
- **Scope.** Results cover ImageNet-1K, a 300-epoch budget, a fixed iBOT weight of 1.0, these four architectures and these readouts. Drop path differs between S/B and L/H, and H/16 uses a smaller batch.
- **Optical flow** measures training-set fitting of the flow head, not held-out generalization.
- **Data and weights.** Datasets and pretrained checkpoints are not included. External data and weights carry their own licenses.

---

<a id="acknowledgments"></a>
## Acknowledgments

This code builds on [DINO](https://github.com/facebookresearch/dino), [DINOv2](https://github.com/facebookresearch/dinov2), [iBOT](https://github.com/bytedance/ibot), [OpenMMLab](https://github.com/open-mmlab) (MMSegmentation, MMDetection, MMFlow) and the [DAVIS 2017 evaluation toolkit](https://github.com/davisvideochallenge/davis2017-evaluation). Gram anchoring follows [DINOv3](https://github.com/facebookresearch/dinov3).

## License

Released under the Apache License 2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE). The bundled DAVIS toolkit keeps its [BSD-3-Clause license](jea/eval/davis2017_evaluation/LICENSE). [`jea/loss/gram_loss.py`](jea/loss/gram_loss.py) is adapted from DINOv3 and remains subject to the DINOv3 License Agreement stated in its header.

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{weng2026locality,
  title     = {The Locality Trade-off in Masked Patch Self-Distillation},
  author    = {Weng, Xi and Liu, Zhaoyu and Hu, Lianyu and Dong, Jin Song},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```
