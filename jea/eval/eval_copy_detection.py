# Modified for the locality-tradeoff research codebase; see NOTICE.
# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Copy-paste from DINO library:
https://github.com/facebookresearch/dino
"""

import os
import sys
import argparse
import json
import logging
import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import numpy as np
import utils
from typing import List, Optional
from PIL import Image
from torch import nn
from torchvision import transforms as pth_transforms
import jea.distributed as distributed
from jea.eval.setup import get_args_parser as get_setup_args_parser
from jea.eval.setup import setup_and_build_model


logger = logging.getLogger("JEA")


def get_args_parser(
    description: Optional[str] = None,
    parents: Optional[List[argparse.ArgumentParser]] = None,
    add_help: bool = True,
):
    parents = parents or []
    setup_args_parser = get_setup_args_parser(parents=parents, add_help=False)
    parents = [setup_args_parser]
    parser = argparse.ArgumentParser(
        description=description,
        parents=parents,
        add_help=add_help,
    )
    parser.add_argument('--data_path', default='datasets/copydays/', type=str,
        help="See https://lear.inrialpes.fr/~jegou/data.php.html#copydays")
    parser.add_argument('--whitening_path', default='datasets/copydays/whitening_images/', type=str,
        help="""Path to directory with images used for computing the whitening operator.
        In our paper, we use 20k random images from coco2017.""")
    parser.add_argument('--distractors_path', default='datasets/copydays/distractors_images/', type=str,
        help="Path to directory with distractors images. In our paper, we use 10k random images from coco2017.")
    parser.add_argument('--imsize', default=224, type=int, help='Image size (square image)')
    parser.add_argument('--batch_size_per_gpu', default=16, type=int, help='Per-GPU batch-size')
    parser.add_argument('--use_cuda', default=True, type=bool)
    parser.add_argument('--num_workers', default=4, type=int, help='Number of data loading workers per GPU.')
    return parser

class CopydaysDataset():
    def __init__(self, basedir):
        self.basedir = basedir
        self.block_names = (
            ['original', 'strong'] +
            ['jpegqual/%d' % i for i in
             [3, 5, 8, 10, 15, 20, 30, 50, 75]] +
            ['crops/%d' % i for i in
             [10, 15, 20, 30, 40, 50, 60, 70, 80]])
        self.nblocks = len(self.block_names)

        self.query_blocks = range(self.nblocks)
        self.q_block_sizes = np.ones(self.nblocks, dtype=int) * 157
        self.q_block_sizes[1] = 229
        # search only among originals
        self.database_blocks = [0]

    def get_block(self, i):
        dirname = self.basedir + '/' + self.block_names[i]
        fnames = [dirname + '/' + fname
                  for fname in sorted(os.listdir(dirname))
                  if fname.endswith('.jpg')]
        return fnames

    def get_block_filenames(self, subdir_name):
        dirname = self.basedir + '/' + subdir_name
        return [fname
                for fname in sorted(os.listdir(dirname))
                if fname.endswith('.jpg')]

    def eval_result(self, ids, distances):
        results = {}
        j0 = 0
        for i in range(self.nblocks):
            j1 = j0 + self.q_block_sizes[i]
            block_name = self.block_names[i]
            I = ids[j0:j1]   # block size
            sum_AP = 0
            if block_name != 'strong':
                # 1:1 mapping of files to names
                positives_per_query = [[i] for i in range(j1 - j0)]
            else:
                originals = self.get_block_filenames('original')
                strongs = self.get_block_filenames('strong')

                # check if prefixes match
                positives_per_query = [
                    [j for j, bname in enumerate(originals)
                     if bname[:4] == qname[:4]]
                    for qname in strongs]

            for qno, Iline in enumerate(I):
                positives = positives_per_query[qno]
                ranks = []
                for rank, bno in enumerate(Iline):
                    if bno in positives:
                        ranks.append(rank)
                sum_AP += score_ap_from_ranks_1(ranks, len(positives))

            block_map = sum_AP / (j1 - j0)
            print("eval on %s mAP=%.3f" % (block_name, block_map))
            results[f"{block_name}_mAP"] = block_map
            j0 = j1
        return results


# from the Holidays evaluation package
def score_ap_from_ranks_1(ranks, nres):
    """ Compute the average precision of one search.
    ranks = ordered list of ranks of true positives
    nres  = total number of positives in dataset
    """

    # accumulate trapezoids in PR-plot
    ap = 0.0

    # All have an x-size of:
    recall_step = 1.0 / nres

    for ntp, rank in enumerate(ranks):

        # y-size on left side of trapezoid:
        # ntp = nb of true positives so far
        # rank = nb of retrieved items so far
        if rank == 0:
            precision_0 = 1.0
        else:
            precision_0 = ntp / float(rank)

        # y-size on right side of trapezoid:
        # ntp and rank are increased by one
        precision_1 = (ntp + 1) / float(rank + 1)

        ap += (precision_1 + precision_0) * recall_step / 2.0

    return ap


class ImgListDataset(torch.utils.data.Dataset):
    def __init__(self, img_list, transform=None):
        self.samples = img_list
        self.transform = transform

    def __getitem__(self, i):
        with open(self.samples[i], 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, i

    def __len__(self):
        return len(self.samples)


def is_image_file(s):
    ext = s.split(".")[-1].lower()
    if ext in ['jpg', 'jpeg', 'png', 'ppm', 'bmp', 'pgm', 'tif', 'tiff', 'webp']:
        return True
    return False


@torch.no_grad()
def extract_features(image_list, model, args, autocast_dtype):
    transform = pth_transforms.Compose([
        pth_transforms.Resize((args.imsize, args.imsize), interpolation=3),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    tempdataset = ImgListDataset(image_list, transform=transform)
    data_loader = torch.utils.data.DataLoader(tempdataset, batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers, drop_last=False,
        sampler=torch.utils.data.DistributedSampler(tempdataset, shuffle=False))
    features = None
    patch_size = model.patch_embed.patch_size[0]
    for samples, index in utils.MetricLogger(delimiter="  ").log_every(data_loader, 10):
        samples, index = samples.cuda(non_blocking=True), index.cuda(non_blocking=True)
        with torch.cuda.amp.autocast(dtype=autocast_dtype):
            feats, cls_output_token = model.get_intermediate_layers(samples, n=1, return_class_token=True)[0]
        feats = feats.float()
        cls_output_token = cls_output_token.float()

        #cls_output_token = feats[:, 0, :]  #  [CLS] token
        # GeM with exponent 4 for output patch tokens
        b, h, w, d = len(samples), int(samples.shape[-2] / patch_size), int(samples.shape[-1] / patch_size), feats.shape[-1]
        #feats = feats[:, 1:, :].reshape(b, h, w, d)
        feats = feats.reshape(b, h, w, d)
        feats = feats.clamp(min=1e-6).permute(0, 3, 1, 2)
        feats = nn.functional.avg_pool2d(feats.pow(4), (h, w)).pow(1. / 4).reshape(b, -1)
        # concatenate [CLS] token and GeM pooled patch tokens
        feats = torch.cat((cls_output_token, feats), dim=1)

        # init storage feature matrix
        if distributed.is_main_process() and features is None:
            features = torch.zeros(len(data_loader.dataset), feats.shape[-1])
            if args.use_cuda:
                features = features.cuda(non_blocking=True)

        # get indexes from all processes
        y_all = torch.empty(dist.get_world_size(), index.size(0), dtype=index.dtype, device=index.device)
        y_l = list(y_all.unbind(0))
        y_all_reduce = torch.distributed.all_gather(y_l, index, async_op=True)
        y_all_reduce.wait()
        index_all = torch.cat(y_l)

        # share features between processes
        feats_all = torch.empty(dist.get_world_size(), feats.size(0), feats.size(1),
                                dtype=feats.dtype, device=feats.device)
        output_l = list(feats_all.unbind(0))
        output_all_reduce = torch.distributed.all_gather(output_l, feats, async_op=True)
        output_all_reduce.wait()

        # update storage feature matrix
        if distributed.is_main_process():
            if args.use_cuda:
                features.index_copy_(0, index_all, torch.cat(output_l))
            else:
                features.index_copy_(0, index_all.cpu(), torch.cat(output_l).cpu())
    return features  # features is still None for every rank which is not 0 (main)

def sorted_image_list(dirname):
    return [os.path.join(dirname, s) for s in sorted(os.listdir(dirname)) if is_image_file(s)]


def CopyDetection(model, dataset, args, autocast_dtype):
    # ============ Extract features ... ============
    # extract features for queries
    queries = []
    for q in dataset.query_blocks:
        queries.append(extract_features(dataset.get_block(q), model, args, autocast_dtype))
    if distributed.is_main_process():
        queries = torch.cat(queries)
        print(f"Extraction of queries features done. Shape: {queries.shape}")

    # extract features for database
    database = []
    for b in dataset.database_blocks:
        database.append(extract_features(dataset.get_block(b), model, args, autocast_dtype))

    # extract features for distractors
    if os.path.isdir(args.distractors_path):
        print("Using distractors...")
        list_distractors = sorted_image_list(args.distractors_path)
        database.append(extract_features(list_distractors, model, args, autocast_dtype))
    if distributed.is_main_process():
        database = torch.cat(database)
        print(f"Extraction of database and distractors features done. Shape: {database.shape}")

    # ============ Whitening ... ============
    if os.path.isdir(args.whitening_path):
        print(f"Extracting features on images from {args.whitening_path} for learning the whitening operator.")
        list_whit = sorted_image_list(args.whitening_path)
        features_for_whitening = extract_features(list_whit, model, args, autocast_dtype)
        if distributed.is_main_process():
            # center
            mean_feature = torch.mean(features_for_whitening, dim=0)
            database -= mean_feature
            queries -= mean_feature
            pca = utils.PCA(dim=database.shape[-1], whit=0.5)
            # compute covariance
            cov = torch.mm(features_for_whitening.T, features_for_whitening) / features_for_whitening.shape[0]
            pca.train_pca(cov.float().cpu().numpy())
            database = pca.apply(database)
            queries = pca.apply(queries)

    # ============ Copy detection ... ============
    if distributed.is_main_process():
        # l2 normalize the features
        database = nn.functional.normalize(database, dim=1, p=2)
        queries = nn.functional.normalize(queries, dim=1, p=2)

        # similarity
        similarity = torch.mm(queries, database.T)
        distances, indices = similarity.topk(20, largest=True, sorted=True)
        
        # evaluate
        results_dict = dataset.eval_result(indices, distances)
        metrics_file_path = os.path.join(args.output_dir, "results_eval_copy_detection.json")
        with open(metrics_file_path, "w") as f:
            json.dump(results_dict, f, indent=2)
        metadata = {
            "config_file": args.config_file,
            "pretrained_weights": args.pretrained_weights,
            "data_path": args.data_path,
            "whitening_path": args.whitening_path,
            "distractors_path": args.distractors_path,
            "imsize": args.imsize,
            "batch_size_per_gpu": args.batch_size_per_gpu,
            "num_workers": args.num_workers,
            "world_size": dist.get_world_size(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cudnn_version": cudnn.version(),
            "device_name": torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else "cpu",
        }
        metadata_file_path = os.path.join(args.output_dir, "metadata_eval_copy_detection.json")
        with open(metadata_file_path, "w") as f:
            json.dump(metadata, f, indent=2)
        for key, value in results_dict.items():
            logger.info(f"{key}: {value}")
    
    dist.barrier()


def main(args):
    model, autocast_dtype = setup_and_build_model(args)
    cudnn.benchmark = False
    cudnn.deterministic = True
    cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    dataset = CopydaysDataset(args.data_path)

    CopyDetection(model, dataset, args, autocast_dtype)

if __name__ == "__main__":
    description = "Copy detection evaluation"
    args_parser = get_args_parser(description=description)
    args = args_parser.parse_args()
    sys.exit(main(args))
