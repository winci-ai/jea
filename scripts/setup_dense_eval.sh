#!/usr/bin/env bash
# Install in a NEW Python 3.10 environment, not the FSDP2 training environment.
set -euo pipefail
python -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
python -m pip install 'numpy<2' 'omegaconf>=2.3,<3' regex scikit-learn scikit-image \
  'opencv-python<4.12' openmim==0.3.9 ftfy tensorboard fvcore iopath torchmetrics einops wandb matplotlib pandas
mim install mmengine==0.10.7 mmcv==2.1.0 mmsegmentation==1.2.2 mmdet==3.3.0
python -m pip install --no-deps -e .
echo 'VOC segmentation/detection dependencies installed. For optical flow, also install MMFlow (README Section 1.3).'
