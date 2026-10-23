# Modified for the locality-tradeoff research codebase; see NOTICE.
# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Copy-paste from DINO library.
https://github.com/facebookresearch/dino
"""

import os
import sys
import json
import pickle
import argparse
import logging
import torch
import numpy as np
import utils
from typing import List, Optional
from torch import nn
from PIL import Image, ImageFile
from torchvision import transforms as pth_transforms

from jea.eval.utils import extract_features
from jea.eval.setup import get_args_parser as get_setup_args_parser
from jea.eval.setup import setup_and_build_model
import jea.distributed as distributed


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
    parser.add_argument('--data_path', default='datasets/revisited_paris_oxford/', type=str)
    parser.add_argument('--dataset', default='roxford5k', type=str, choices=['roxford5k', 'rparis6k'])
    parser.add_argument('--imsize', default=224, type=int, help='Image size')
    parser.add_argument('--gather_on_cpu', action="store_true")
    parser.add_argument('--num_workers', default=4, type=int, help='Number of data loading workers per GPU.')
    return parser


class OxfordParisDataset(torch.utils.data.Dataset):
    def __init__(self, dir_main, dataset, split, transform=None, imsize=None):
        if dataset not in ['roxford5k', 'rparis6k']:
            raise ValueError('Unknown dataset: {}!'.format(dataset))

        # loading imlist, qimlist, and gnd, in cfg as a dict
        gnd_fname = os.path.join(dir_main, dataset, 'gnd_{}.pkl'.format(dataset))
        with open(gnd_fname, 'rb') as f:
            cfg = pickle.load(f)
        cfg['gnd_fname'] = gnd_fname
        cfg['ext'] = '.jpg'
        cfg['qext'] = '.jpg'
        cfg['dir_data'] = os.path.join(dir_main, dataset)
        cfg['dir_images'] = os.path.join(cfg['dir_data'], 'jpg')
        cfg['n'] = len(cfg['imlist'])
        cfg['nq'] = len(cfg['qimlist'])
        cfg['im_fname'] = config_imname
        cfg['qim_fname'] = config_qimname
        cfg['dataset'] = dataset
        self.cfg = cfg

        self.samples = cfg["qimlist"] if split == "query" else cfg["imlist"]
        self.transform = transform
        self.imsize = imsize

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path = os.path.join(self.cfg["dir_images"], self.samples[index] + ".jpg")
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        if self.imsize is not None:
            #img.thumbnail((self.imsize, self.imsize), Image.LANCZOS)
            img = img.resize((self.imsize, self.imsize), Image.LANCZOS)
        if self.transform is not None:
            img = self.transform(img)
        # img, (index, label)
        return img, 0


def config_imname(cfg, i):
    return os.path.join(cfg['dir_images'], cfg['imlist'][i] + cfg['ext'])


def config_qimname(cfg, i):
    return os.path.join(cfg['dir_images'], cfg['qimlist'][i] + cfg['qext'])

def image_retrieval(model, dataset_train, dataset_query, args):

    train_features, _ = extract_features(
        model, dataset_train, 1, args.num_workers, gather_on_cpu=args.gather_on_cpu
    )
    query_features, _ = extract_features(
        model, dataset_query, 1, args.num_workers, gather_on_cpu=args.gather_on_cpu
    )

    if distributed.is_main_process():  # only rank 0 will work from now on
        # normalize features
        train_features = nn.functional.normalize(train_features, dim=1, p=2)
        query_features = nn.functional.normalize(query_features, dim=1, p=2)

        ############################################################################
        # Step 2: similarity
        sim = torch.mm(train_features, query_features.T)
        ranks = torch.argsort(-sim, dim=0).cpu().numpy()

        ############################################################################
        # Step 3: evaluate
        gnd = dataset_train.cfg['gnd']
        # evaluate ranks
        ks = [1, 5, 10]
        # search for easy & hard
        gnd_t = []
        for i in range(len(gnd)):
            g = {}
            g['ok'] = np.concatenate([gnd[i]['easy'], gnd[i]['hard']])
            g['junk'] = np.concatenate([gnd[i]['junk']])
            gnd_t.append(g)
        mapM, apsM, mprM, prsM = utils.compute_map(ranks, gnd_t, ks)
        # search for hard
        gnd_t = []
        for i in range(len(gnd)):
            g = {}
            g['ok'] = np.concatenate([gnd[i]['hard']])
            g['junk'] = np.concatenate([gnd[i]['junk'], gnd[i]['easy']])
            gnd_t.append(g)
        mapH, apsH, mprH, prsH = utils.compute_map(ranks, gnd_t, ks)
        print('>> {}: mAP M: {}, H: {}'.format(args.dataset, np.around(mapM*100, decimals=2), np.around(mapH*100, decimals=2)))
        print('>> {}: mP@k{} M: {}, H: {}'.format(args.dataset, np.array(ks), np.around(mprM*100, decimals=2), np.around(mprH*100, decimals=2)))

        results_dict = {
            "dataset": args.dataset,
            "imsize": args.imsize,
            "map_medium": float(mapM),
            "map_hard": float(mapH),
        }
        for k, value in zip(ks, mprM):
            results_dict[f"mp_{k}_medium"] = float(value)
        for k, value in zip(ks, mprH):
            results_dict[f"mp_{k}_hard"] = float(value)

        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            metrics_file_path = os.path.join(args.output_dir, "results_eval_image_retrieval.json")
            with open(metrics_file_path, "w") as f:
                json.dump(results_dict, f, indent=2)
            logger.info(f"Saved image retrieval metrics to: {metrics_file_path}")
            for key, value in results_dict.items():
                logger.info(f"{key}: {value}")

    if distributed.is_enabled():
        torch.distributed.barrier()

def main(args):
    
    # building network
    model, autocast_dtype = setup_and_build_model(args)

    # ============ preparing data ... ============
    transform = pth_transforms.Compose([
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    dataset_train = OxfordParisDataset(args.data_path, args.dataset, split="train", transform=transform, imsize=args.imsize)
    dataset_query = OxfordParisDataset(args.data_path, args.dataset, split="query", transform=transform, imsize=args.imsize)

    ############################################################################
    # Step 1: extract features
    with torch.cuda.amp.autocast(dtype=autocast_dtype):
        image_retrieval(model, dataset_train, dataset_query, args)

if __name__ == "__main__":
    description = "Image retrieval evaluation"
    args_parser = get_args_parser(description=description)
    args = args_parser.parse_args()
    sys.exit(main(args))
