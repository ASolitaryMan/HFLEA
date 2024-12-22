#!/usr/bin/env python3
"""This is the preprocessing script for HuBERT model training.
The script includes:
    - File list creation
    - MFCC/HuBERT feature extraction
    - KMeans clustering model training
    - Pseudo-label generation
"""
import logging
from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path

import torch
from utils import create_tsv, dump_features, get_km_label, learn_kmeans
from lightning_fabric.utilities.seed import seed_everything


def _init_logger(debug=False):
    message_fmt = "%(levelname)5s: %(funcName)10s: %(message)s" if debug else "%(message)s"
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format=f"%(asctime)s: {message_fmt}",
    )


def _parse_args():
    parser = ArgumentParser(
        description=__doc__,
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug log")
    parser.add_argument("--dataset", default="IEMOCAP", type=str, choices=["IEMOCAP"])
    parser.add_argument(
        "--root_dir",
        type=Path,
        help="The path to the directory where the directory ``LibriSpeech`` or ``LibriLight`` is stored.",
    )
    parser.add_argument("--num_rank", default=5, type=int)
    parser.add_argument("--feat_type", default="mfcc", choices=["mfcc", "hubert"], type=str)
    parser.add_argument(
        "--layer_index",
        default=6,
        type=int,
        help="The layer index in HuBERT model for feature extraction. (``1`` means the first layer output)",
    )
    parser.add_argument(
        "--hubert_base",
        default=None,
        type=Path,
        help="The model checkpoint of hubert_pretrain_base model.",
    )
    parser.add_argument(
        "--checkpoint_path",
        default=None,
        type=Path,
        help="The model checkpoint of hubert_pretrain_base model.",
    )
    parser.add_argument("--use_gpu", default=True, type=bool)
    parser.add_argument("--gpu_id", default=0, type=int)
    parser.add_argument(
        "--exp_dir",
        type=Path,
        help="The directory to store the experiment outputs.",
    )
    parser.add_argument(
        "--num_cluster",
        default=100,
        type=int,
        help="The number of clusters for KMeans clustering.",
    )
    parser.add_argument(
        "--percent",
        default=-1,
        type=float,
        help="The percent of data for KMeans clustering. If negative, use all data. (Default: -1)",
    )
    parser.add_argument(
        "--iskmean",
        default=True,
        type=bool,
        help="Whether to do kmean clustering. If false, you need provide the path of pre-trian kmean model",
    )
    parser.add_argument(
        "--pretrain_kmeans",
        default=None,
        type=Path,
        help="The path of pre-trian kmean model",
    )
    args = parser.parse_args()
    return args


def main(args):
    _init_logger(args.debug)
    # seed_everything(1337)
    if not args.exp_dir.exists():
        args.exp_dir.mkdir()
    if args.feat_type == "mfcc":
        data_dir = args.exp_dir / "data" / "mfcc"
    else:
        data_dir = args.exp_dir / "data" / f"{args.feat_type}_{args.layer_index}"
    data_dir.mkdir(parents=True, exist_ok=True)

    tsv_dir = data_dir / "tsv"
    feat_dir = data_dir / "feat"
    km_dir = data_dir / "km_model"
    label_dir = data_dir / "label"

    if args.use_gpu:
        device = torch.device("cuda:" + str(args.gpu_id) if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    # Create file lists for training and validation (optional)
    create_tsv(args.root_dir, tsv_dir)

    # Extract features for KMeans clustering
    if not feat_dir.exists():
        feat_dir.mkdir()

    for split in ["train", "valid"]:
        for rank in range(1, args.num_rank + 1):
            dump_features(
                tsv_dir / f"{args.dataset}_{split}.tsv",
                feat_dir,
                split,
                rank,
                args.num_rank,
                device,
                args.feat_type,
                args.layer_index,
                args.hubert_base,
                args.checkpoint_path,
                16_000,
                args.num_cluster
            )

    # Fit KMeans clustering model
    if args.iskmean is False and args.pretrain_kmeans is not None:
        km_dir = args.pretrain_kmeans
    else:
        learn_kmeans(
            feat_dir,
            "train",
            args.num_rank,
            km_dir,
            args.num_cluster,
            args.percent,
        )

    # Predict labels for MFCC or HuBERT features
    for split in ["train", "valid"]:
        get_km_label(
            feat_dir,
            km_dir,
            label_dir,
            split,
            args.num_rank,
            device,
        )


if __name__ == "__main__":
    main(_parse_args())
