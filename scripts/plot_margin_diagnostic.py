#!/usr/bin/env python3
"""Compute the positive--hard negative score margin diagnostic.

This script evaluates a trained host text-based person search model on a
selected split, computes

    s_i^+ = max_{j: y_j = y_i} s(q_i, x_j)
    s_i^- = max_{j: y_j != y_i} s(q_i, x_j)
    Delta_i = s_i^+ - s_i^-

and saves a CSV, a summary JSON file, a paper-sized histogram, and a LaTeX
include snippet.

Usage example:

python scripts/plot_margin_diagnostic.py \
--dataset_root /path/to/RSTPReid \
--dataset_name RSTPReid \
--split test \
--checkpoint /path/to/best.pth \
--output_dir outputs/margin_diagnostic/rstpreid_host \
--batch_size 128 \
--device cuda
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.bases import ImageDataset, TextDataset
from datasets.build import build_transforms
from datasets.cuhkpedes import CUHKPEDES
from datasets.icfgpedes import ICFGPEDES
from datasets.rstpreid import RSTPReid
from model import build_model
from utils.iotools import load_train_configs


DATASET_FACTORIES = {
    "CUHK-PEDES": CUHKPEDES,
    "ICFG-PEDES": ICFGPEDES,
    "RSTPReid": RSTPReid,
}

GENERIC_DATASET_CONFIGS = {
    "CUHK-PEDES": {
        "dataset_dir": "CUHK-PEDES",
        "annotation_files": ["reid_raw.json"],
        "image_dirs": ["imgs"],
        "path_keys": ["file_path"],
    },
    "ICFG-PEDES": {
        "dataset_dir": "ICFG-PEDES",
        "annotation_files": ["ICFG-PEDES.json"],
        "image_dirs": ["imgs"],
        "path_keys": ["file_path"],
    },
    "RSTPReid": {
        "dataset_dir": "RSTPReid",
        "annotation_files": ["data_captions.json"],
        "image_dirs": ["imgs"],
        "path_keys": ["img_path"],
    },
    "PAB": {
        "dataset_dir": "PAB",
        "annotation_files": ["PAB.json", "pab.json", "data_captions.json", "annotations.json"],
        "image_dirs": ["imgs", "images"],
        "path_keys": ["img_path", "file_path", "image_path", "path"],
    },
}

DEFAULT_ANNOTATION_FILES = [
    "data_captions.json",
    "reid_raw.json",
    "ICFG-PEDES.json",
    "annotations.json",
    "annotation.json",
]
DEFAULT_IMAGE_DIRS = ["imgs", "images", "image", ""]
DEFAULT_PATH_KEYS = ["img_path", "file_path", "image_path", "path", "filename", "image"]
DEFAULT_PID_KEYS = ["id", "pid", "person_id", "identity", "identity_id", "label"]
DEFAULT_CAPTION_KEYS = ["captions", "caption", "text", "description"]

LATEX_SNIPPET = r"""\begin{figure}[t]
\centering
\includegraphics[width=0.95\linewidth]{fig_margin_diagnostic.pdf}
\caption{
Positive--hard negative score margin diagnostic.
For each query $q_i$, we compute
$s_i^+ = \max_{j:y_j=y_i} s(q_i,x_j)$,
$s_i^- = \max_{j:y_j\ne y_i} s(q_i,x_j)$,
and $\Delta_i = s_i^+ - s_i^-$.
A near-zero or negative margin indicates that the strongest identity-wrong image is scored close to, or higher than, the best positive image.
}
\label{fig:margin_diagnostic}
\end{figure}
"""


@dataclass
class SplitData:
    image_pids: List[int]
    img_paths: List[str]
    caption_pids: List[int]
    captions: List[str]
    num_train_ids: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the positive--hard negative score margin diagnostic for a host TBPS model."
    )
    parser.add_argument("--dataset_root", required=True, help="Dataset folder or parent folder containing the dataset.")
    parser.add_argument("--checkpoint", required=True, help="Path to the trained host model checkpoint.")
    parser.add_argument("--dataset_name", required=True, help="Dataset name, e.g. RSTPReid, CUHK-PEDES, ICFG-PEDES, PAB.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Split to evaluate.")
    parser.add_argument("--output_dir", default="outputs/margin_diagnostic", help="Directory to save outputs.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for feature extraction.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers.")
    parser.add_argument("--device", default="cuda", help='Device, e.g. "cuda" or "cpu".')
    parser.add_argument("--config", default=None, help="Optional path to the saved model/config YAML file.")
    parser.add_argument("--max_queries", type=int, default=None, help="Optional query limit for debugging.")
    parser.add_argument("--bins", type=int, default=35, help="Histogram bin count.")
    parser.add_argument("--xmin", type=float, default=-0.25, help="Histogram x-axis lower bound.")
    parser.add_argument("--xmax", type=float, default=0.65, help="Histogram x-axis upper bound.")
    return parser.parse_args()


def resolve_path(path: str | Path, base: Path = REPO_ROOT) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = base / resolved
    return resolved.resolve()


def parse_img_size(value: Any) -> Tuple[int, int]:
    if value is None:
        return (384, 128)
    if isinstance(value, int):
        return (value, value)
    if isinstance(value, str):
        cleaned = value.strip().strip("()[]")
        parts = [part.strip() for part in cleaned.split(",") if part.strip()]
        if len(parts) != 2:
            raise ValueError(f"Could not parse img_size from config value: {value!r}")
        return (int(parts[0]), int(parts[1]))
    if isinstance(value, Sequence) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    raise ValueError(f"Could not parse img_size from config value: {value!r}")


def default_model_args() -> Dict[str, Any]:
    # Defaults mirror prototype-regu-irra/utils/options.py when no saved config is supplied.
    return {
        "local_rank": 0,
        "name": "baseline",
        "output_dir": "logs",
        "run_time": "",
        "seed": 1,
        "deterministic": True,
        "log_period": 100,
        "eval_period": 1,
        "eval_after_epoch": 0,
        "val_dataset": "test",
        "resume": False,
        "resume_ckpt_file": "",
        "finetune": "",
        "finetune_clip": "",
        "pretrain": "",
        "nohup": False,
        "nohup_log_dir": "logs",
        "wandb": False,
        "wandb_project": "prototype-regu-irra",
        "wandb_entity": None,
        "wandb_name": "",
        "wandb_mode": "disabled",
        "wandb_tags": [],
        "pretrain_choice": "ViT-B/16",
        "temperature": 0.02,
        "img_aug": False,
        "txt_aug": False,
        "cmt_depth": 4,
        "masked_token_rate": 0.8,
        "masked_token_unchanged_rate": 0.1,
        "MLM": False,
        "loss_names": "sdm+id+mlm",
        "mlm_loss_weight": 1.0,
        "id_loss_weight": 1.0,
        "prototype": False,
        "use_loss_id": False,
        "no_pbt": False,
        "prototype_feature": "auto",
        "prototype_projector": "default",
        "prototype_residual_scale": 0.1,
        "prototype_per_id": 2,
        "prototype_dim": 512,
        "prototype_kmeans_iters": 20,
        "prototype_warmup_epochs": 0,
        "prototype_tau": 0.05,
        "prototype_hard_k": 16,
        "prototype_id_weight": 1.0,
        "prototype_momentum": 0.2,
        "prototype_lr": None,
        "img_size": (384, 128),
        "stride_size": 16,
        "text_length": 77,
        "vocab_size": 49408,
        "optimizer": "Adam",
        "lr": 1e-5,
        "bias_lr_factor": 2.0,
        "lr_factor": 5.0,
        "momentum": 0.9,
        "weight_decay": 4e-5,
        "weight_decay_bias": 0.0,
        "alpha": 0.9,
        "beta": 0.999,
        "num_epoch": 60,
        "lr_total_epochs": None,
        "milestones": (20, 50),
        "gamma": 0.1,
        "warmup_factor": 0.1,
        "warmup_epochs": 5,
        "warmup_method": "linear",
        "lrscheduler": "cosine",
        "target_lr": 0,
        "power": 0.9,
        "early_stop_patience": 0,
        "early_stop_min_delta": 0.0,
        "dataset_name": "CUHK-PEDES",
        "sampler": "random",
        "num_instance": 4,
        "root_dir": "./data",
        "batch_size": 128,
        "test_batch_size": 512,
        "num_workers": 8,
        "training": False,
        "distributed": False,
        "only_global": False,
        "return_all": False,
        "topk_type": "mean",
        "layer_index": -1,
        "average_attn_weights": True,
        "modify_k": False,
        "track_train_diagnostics": False,
        "tau": 0.015,
        "select_ratio": 0.4,
        "margin": 0.1,
        "noisy_rate": 0.0,
        "noisy_file": "",
        "num_experts": 6,
        "topk": 2,
        "reduction": 8,
    }


def config_to_dict(config: Any) -> Dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, Mapping):
        return dict(config)
    if hasattr(config, "__dict__"):
        return dict(vars(config))
    return {key: getattr(config, key) for key in dir(config) if not key.startswith("_")}


def build_model_args(cli_args: argparse.Namespace) -> SimpleNamespace:
    cfg = default_model_args()
    if cli_args.config:
        config_path = resolve_path(cli_args.config)
        if not config_path.is_file():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        cfg.update(config_to_dict(load_train_configs(str(config_path))))

    cfg["dataset_name"] = cli_args.dataset_name
    cfg["root_dir"] = str(resolve_path(cli_args.dataset_root))
    cfg["training"] = False
    cfg["batch_size"] = cli_args.batch_size
    cfg["test_batch_size"] = cli_args.batch_size
    cfg["num_workers"] = cli_args.num_workers
    cfg["img_size"] = parse_img_size(cfg.get("img_size"))
    return SimpleNamespace(**cfg)


def dataset_root_for_repo_class(factory: type, dataset_root: Path) -> Path:
    dataset_dir = getattr(factory, "dataset_dir", None)
    if dataset_dir and dataset_root.name.lower() == str(dataset_dir).lower():
        return dataset_root.parent
    return dataset_root


def validate_split_data(split_data: SplitData, split: str) -> None:
    if len(split_data.image_pids) != len(split_data.img_paths):
        raise ValueError(
            f"Image labels and image paths differ in length for split {split}: "
            f"{len(split_data.image_pids)} labels vs {len(split_data.img_paths)} paths."
        )
    if len(split_data.caption_pids) != len(split_data.captions):
        raise ValueError(
            f"Caption labels and captions differ in length for split {split}: "
            f"{len(split_data.caption_pids)} labels vs {len(split_data.captions)} captions."
        )
    if not split_data.image_pids:
        raise ValueError(f"No gallery images found for split {split}.")
    if not split_data.caption_pids:
        raise ValueError(f"No text queries found for split {split}.")
    if any(pid is None for pid in split_data.image_pids) or any(pid is None for pid in split_data.caption_pids):
        raise ValueError(
            f"Identity labels are required to compute the margin diagnostic, but split {split} "
            "contains missing image or caption labels."
        )


def split_data_from_repo_dataset(dataset: Any, split: str) -> SplitData:
    train_ids = getattr(dataset, "train_id_container", set())
    num_train_ids = len(train_ids)

    if split == "train":
        if not hasattr(dataset, "train"):
            raise ValueError("The dataset object has no train split.")
        image_key_to_index: Dict[Tuple[int, str], int] = {}
        image_pids: List[int] = []
        img_paths: List[str] = []
        caption_pids: List[int] = []
        captions: List[str] = []

        for row in dataset.train:
            if len(row) != 4:
                raise ValueError("Expected train rows to be (pid, image_id, img_path, caption).")
            pid, _image_id, img_path, caption = row
            pid = int(pid)
            img_path = str(img_path)
            key = (pid, img_path)
            if key not in image_key_to_index:
                image_key_to_index[key] = len(img_paths)
                image_pids.append(pid)
                img_paths.append(img_path)
            caption_pids.append(pid)
            captions.append(str(caption))

        return SplitData(image_pids, img_paths, caption_pids, captions, num_train_ids)

    if not hasattr(dataset, split):
        raise ValueError(f"The dataset object has no {split!r} split.")
    split_obj = getattr(dataset, split)
    required = ["image_pids", "img_paths", "caption_pids", "captions"]
    missing = [key for key in required if key not in split_obj]
    if missing:
        raise ValueError(f"Split {split} is missing required fields: {missing}")

    return SplitData(
        image_pids=[int(pid) for pid in split_obj["image_pids"]],
        img_paths=[str(path) for path in split_obj["img_paths"]],
        caption_pids=[int(pid) for pid in split_obj["caption_pids"]],
        captions=[str(caption) for caption in split_obj["captions"]],
        num_train_ids=num_train_ids,
    )


def locate_generic_dataset_dir(dataset_name: str, dataset_root: Path) -> Path:
    cfg = GENERIC_DATASET_CONFIGS.get(dataset_name, {})
    dataset_dir = cfg.get("dataset_dir", dataset_name)
    if dataset_root.name.lower() == str(dataset_dir).lower():
        return dataset_root
    candidate = dataset_root / str(dataset_dir)
    if candidate.is_dir():
        return candidate
    return dataset_root


def locate_first_existing(base_dir: Path, names: Iterable[str], kind: str) -> Path:
    for name in names:
        candidate = base_dir / name if name else base_dir
        if kind == "file" and candidate.is_file():
            return candidate
        if kind == "dir" and candidate.is_dir():
            return candidate
    candidates = ", ".join(str(base_dir / name) for name in names if name)
    raise FileNotFoundError(f"Could not find expected {kind} under {base_dir}. Tried: {candidates}")


def annotation_pid(dataset_name: str, split: str, anno: Mapping[str, Any]) -> int:
    for key in DEFAULT_PID_KEYS:
        if key in anno and anno[key] is not None:
            pid = int(anno[key])
            if dataset_name == "CUHK-PEDES" and split == "train":
                pid -= 1
            return pid
    raise ValueError("Identity labels are required to compute this diagnostic, but an annotation has no pid/id field.")


def annotation_captions(anno: Mapping[str, Any]) -> List[str]:
    for key in DEFAULT_CAPTION_KEYS:
        if key not in anno:
            continue
        value = anno[key]
        if isinstance(value, str):
            return [value]
        if isinstance(value, Sequence):
            return [str(item) for item in value]
    raise ValueError("A text query field is required, but an annotation has no captions/caption/text field.")


def annotation_image_path(anno: Mapping[str, Any], dataset_dir: Path, image_dir: Path, path_keys: Sequence[str]) -> str:
    rel_path: Optional[str] = None
    for key in path_keys:
        if key in anno and anno[key]:
            rel_path = str(anno[key])
            break
    if rel_path is None:
        raise ValueError(f"An image path field is required, but an annotation has none of: {list(path_keys)}")

    candidate = Path(rel_path).expanduser()
    if candidate.is_absolute():
        return str(candidate.resolve())

    img_candidate = image_dir / candidate
    if img_candidate.exists():
        return str(img_candidate.resolve())

    dataset_candidate = dataset_dir / candidate
    if dataset_candidate.exists():
        return str(dataset_candidate.resolve())

    return str(img_candidate.resolve())


def split_annotations(raw: Any, split: str) -> List[Mapping[str, Any]]:
    if isinstance(raw, Mapping):
        if split in raw and isinstance(raw[split], Sequence):
            return list(raw[split])
        for key in ("annotations", "annos", "data"):
            if key in raw and isinstance(raw[key], Sequence):
                raw = raw[key]
                break
    if not isinstance(raw, Sequence):
        raise ValueError("Annotation JSON must be a list or a dict containing split lists.")

    annos = []
    for anno in raw:
        if not isinstance(anno, Mapping):
            continue
        anno_split = anno.get("split")
        if anno_split == split:
            annos.append(anno)
        elif split == "val" and anno_split not in ("train", "test"):
            annos.append(anno)
    return annos


def generic_split_data(dataset_name: str, dataset_root: Path, split: str) -> SplitData:
    cfg = GENERIC_DATASET_CONFIGS.get(dataset_name, {})
    dataset_dir = locate_generic_dataset_dir(dataset_name, dataset_root)
    annotation_files = list(cfg.get("annotation_files", [])) + DEFAULT_ANNOTATION_FILES
    image_dirs = list(cfg.get("image_dirs", [])) + DEFAULT_IMAGE_DIRS
    path_keys = list(cfg.get("path_keys", [])) + DEFAULT_PATH_KEYS

    annotation_path = locate_first_existing(dataset_dir, annotation_files, "file")
    image_dir = locate_first_existing(dataset_dir, image_dirs, "dir")

    with annotation_path.open("r", encoding="utf-8") as file:
        raw_annos = json.load(file)

    selected_annos = split_annotations(raw_annos, split)
    if not selected_annos:
        raise ValueError(f"No annotations found for split {split} in {annotation_path}.")

    all_annos_for_train = split_annotations(raw_annos, "train")
    train_pids = {
        annotation_pid(dataset_name, "train", anno)
        for anno in all_annos_for_train
        if isinstance(anno, Mapping)
    }

    image_key_to_index: Dict[Tuple[int, str], int] = {}
    image_pids: List[int] = []
    img_paths: List[str] = []
    caption_pids: List[int] = []
    captions: List[str] = []

    for anno in selected_annos:
        pid = annotation_pid(dataset_name, split, anno)
        img_path = annotation_image_path(anno, dataset_dir, image_dir, path_keys)
        image_key = (pid, img_path)
        if image_key not in image_key_to_index:
            image_key_to_index[image_key] = len(img_paths)
            image_pids.append(pid)
            img_paths.append(img_path)
        for caption in annotation_captions(anno):
            caption_pids.append(pid)
            captions.append(caption)

    num_train_ids = len(train_pids) if train_pids else len(set(image_pids) | set(caption_pids))
    return SplitData(image_pids, img_paths, caption_pids, captions, num_train_ids)


def load_split_data(dataset_name: str, dataset_root: Path, split: str) -> SplitData:
    factory = DATASET_FACTORIES.get(dataset_name)
    if factory is not None:
        root_for_factory = dataset_root_for_repo_class(factory, dataset_root)
        dataset = factory(root=str(root_for_factory), verbose=False)
        split_data = split_data_from_repo_dataset(dataset, split)
    else:
        split_data = generic_split_data(dataset_name, dataset_root, split)

    validate_split_data(split_data, split)
    return split_data


def limit_queries(split_data: SplitData, max_queries: Optional[int]) -> SplitData:
    if max_queries is None:
        return split_data
    if max_queries <= 0:
        raise ValueError("--max_queries must be a positive integer when provided.")
    return SplitData(
        image_pids=split_data.image_pids,
        img_paths=split_data.img_paths,
        caption_pids=split_data.caption_pids[:max_queries],
        captions=split_data.captions[:max_queries],
        num_train_ids=split_data.num_train_ids,
    )


def checkpoint_state_dict(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(checkpoint, Mapping):
        for key in ("state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, Mapping):
                return value
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Checkpoint must be a mapping or contain a 'state_dict'/'model' mapping.")
    return checkpoint


def strip_module_prefix(key: str) -> str:
    return key[7:] if key.startswith("module.") else key


def candidate_state_keys(key: str) -> List[str]:
    stripped = strip_module_prefix(key)
    keys = [stripped]
    if stripped.startswith("model."):
        keys.append(stripped[len("model."):])
    if not stripped.startswith("base_model."):
        keys.append(f"base_model.{stripped}")
    return keys


def load_checkpoint_for_inference(model: torch.nn.Module, checkpoint_path: Path) -> Dict[str, int]:
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    loaded_state = checkpoint_state_dict(checkpoint)
    model_state = model.state_dict()
    update_state: MutableMapping[str, torch.Tensor] = {}
    skipped_missing = 0
    skipped_shape = 0
    skipped_non_tensor = 0

    for raw_key, value in loaded_state.items():
        if not torch.is_tensor(value):
            skipped_non_tensor += 1
            continue

        target_key = None
        for candidate in candidate_state_keys(str(raw_key)):
            if candidate in model_state:
                target_key = candidate
                break

        if target_key is None:
            skipped_missing += 1
            continue
        if model_state[target_key].shape != value.shape:
            skipped_shape += 1
            continue
        update_state[target_key] = value.detach().clone()

    if not update_state:
        raise RuntimeError(f"No compatible tensors found in checkpoint: {checkpoint_path}")

    model_state.update(update_state)
    model.load_state_dict(model_state, strict=True)
    return {
        "loaded": len(update_state),
        "skipped_missing": skipped_missing,
        "skipped_shape": skipped_shape,
        "skipped_non_tensor": skipped_non_tensor,
    }


def resolve_device(device_arg: str) -> torch.device:
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA was requested but is not available; using CPU.")
        return torch.device("cpu")
    return device


def call_model_encoder(encoder: Any, tensor: torch.Tensor) -> torch.Tensor:
    params = inspect.signature(encoder).parameters
    features = encoder(tensor, 0) if len(params) >= 2 else encoder(tensor)
    if isinstance(features, (tuple, list)):
        features = features[0]
    return features


@torch.inference_mode()
def extract_text_features(
    model: torch.nn.Module,
    split_data: SplitData,
    text_length: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    text_set = TextDataset(split_data.caption_pids, split_data.captions, text_length=text_length)
    loader = DataLoader(
        text_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    features: List[torch.Tensor] = []
    pids: List[torch.Tensor] = []

    model.eval()
    for pid, tokens in tqdm(loader, desc="Extracting text features"):
        tokens = tokens.to(device, non_blocking=True)
        feats = call_model_encoder(model.encode_text, tokens).float()
        features.append(feats.cpu())
        pids.append(pid.view(-1).cpu().long())

    if not features:
        raise RuntimeError("No text features were extracted.")

    text_features = F.normalize(torch.cat(features, dim=0), p=2, dim=1)
    text_pids = torch.cat(pids, dim=0).long()
    return text_features, text_pids


@torch.inference_mode()
def extract_image_features(
    model: torch.nn.Module,
    split_data: SplitData,
    img_size: Tuple[int, int],
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    transform = build_transforms(img_size=img_size, is_train=False)
    image_set = ImageDataset(split_data.image_pids, split_data.img_paths, transform=transform)
    loader = DataLoader(
        image_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    features: List[torch.Tensor] = []
    pids: List[torch.Tensor] = []

    model.eval()
    for pid, images in tqdm(loader, desc="Extracting image features"):
        images = images.to(device, non_blocking=True)
        feats = call_model_encoder(model.encode_image, images).float()
        features.append(feats.cpu())
        pids.append(pid.view(-1).cpu().long())

    if not features:
        raise RuntimeError("No image features were extracted.")

    image_features = F.normalize(torch.cat(features, dim=0), p=2, dim=1)
    image_pids = torch.cat(pids, dim=0).long()
    return image_features, image_pids


def compute_margin_rows(
    sim: torch.Tensor,
    query_pids: torch.Tensor,
    gallery_pids: torch.Tensor,
) -> Tuple[List[Dict[str, Any]], int]:
    rows: List[Dict[str, Any]] = []
    skipped = 0

    for query_index in tqdm(range(sim.shape[0]), desc="Computing margins"):
        query_pid = int(query_pids[query_index].item())
        scores = sim[query_index]
        pos_mask = gallery_pids.eq(query_pid)
        neg_mask = ~pos_mask

        if not bool(pos_mask.any()) or not bool(neg_mask.any()):
            skipped += 1
            continue

        pos_scores = scores.masked_fill(~pos_mask, float("-inf"))
        neg_scores = scores.masked_fill(~neg_mask, float("-inf"))
        best_pos_score, best_pos_index = torch.max(pos_scores, dim=0)
        hard_neg_score, hard_neg_index = torch.max(neg_scores, dim=0)
        margin = best_pos_score - hard_neg_score

        rows.append(
            {
                "query_index": int(query_index),
                "query_pid": query_pid,
                "best_pos_index": int(best_pos_index.item()),
                "best_pos_pid": int(gallery_pids[best_pos_index].item()),
                "best_pos_score": float(best_pos_score.item()),
                "hard_neg_index": int(hard_neg_index.item()),
                "hard_neg_pid": int(gallery_pids[hard_neg_index].item()),
                "hard_neg_score": float(hard_neg_score.item()),
                "margin": float(margin.item()),
            }
        )

    return rows, skipped


def summarize_margins(rows: Sequence[Mapping[str, Any]], total_queries: int, skipped: int) -> Dict[str, Any]:
    if not rows:
        raise RuntimeError("No usable queries remained after filtering queries without positives/negatives.")

    margins = np.array([float(row["margin"]) for row in rows], dtype=np.float64)
    return {
        "num_queries_total": int(total_queries),
        "num_queries_used": int(len(rows)),
        "num_queries_skipped": int(skipped),
        "mean_margin": float(np.mean(margins)),
        "median_margin": float(np.median(margins)),
        "std_margin": float(np.std(margins)),
        "min_margin": float(np.min(margins)),
        "max_margin": float(np.max(margins)),
        "percent_margin_negative": float(np.mean(margins < 0.0) * 100.0),
        "percent_margin_below_0_01": float(np.mean(margins < 0.01) * 100.0),
        "percent_margin_below_0_05": float(np.mean(margins < 0.05) * 100.0),
    }


def save_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    columns = [
        "query_index",
        "query_pid",
        "best_pos_index",
        "best_pos_pid",
        "best_pos_score",
        "hard_neg_index",
        "hard_neg_pid",
        "hard_neg_score",
        "margin",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def save_summary(summary: Mapping[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
        file.write("\n")


def plot_margins(rows: Sequence[Mapping[str, Any]], args: argparse.Namespace, pdf_path: Path, png_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to plot the margin diagnostic.") from exc

    margins = np.array([float(row["margin"]) for row in rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(3.4, 2.0))
    ax.hist(
        margins,
        bins=args.bins,
        density=True,
        color="#4C78A8",
        edgecolor="white",
        linewidth=0.45,
        alpha=0.9,
    )
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel(r"Positive--hard negative score margin $\Delta$")
    ax.set_ylabel("Density")
    ax.set_xlim(args.xmin, args.xmax)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    fig.tight_layout(pad=0.4)
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_latex_snippet(path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        file.write(LATEX_SNIPPET)
        file.write("\n")


def print_summary(paths: Mapping[str, Path], summary: Mapping[str, Any]) -> None:
    print("\nOutputs:")
    for label, path in paths.items():
        print(f"  {label}: {path}")
    print("\nMargin diagnostic:")
    print(
        f"  used/skipped queries: {summary['num_queries_used']}/{summary['num_queries_skipped']} "
        f"(total={summary['num_queries_total']})"
    )
    print(f"  mean margin: {summary['mean_margin']:.6f}")
    print(f"  median margin: {summary['median_margin']:.6f}")
    print(f"  negative margins: {summary['percent_margin_negative']:.2f}%")
    print(f"  margins below 0.01: {summary['percent_margin_below_0_01']:.2f}%")
    print(f"  margins below 0.05: {summary['percent_margin_below_0_05']:.2f}%")


def main() -> None:
    args = parse_args()
    if args.bins <= 0:
        raise ValueError("--bins must be positive.")
    if args.xmin >= args.xmax:
        raise ValueError("--xmin must be less than --xmax.")

    dataset_root = resolve_path(args.dataset_root)
    checkpoint_path = resolve_path(args.checkpoint)
    output_dir = resolve_path(args.output_dir)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)

    # Load model/config arguments and the selected dataset split.
    model_args = build_model_args(args)
    split_data = load_split_data(args.dataset_name, dataset_root, args.split)
    split_data = limit_queries(split_data, args.max_queries)
    validate_split_data(split_data, args.split)

    print(
        f"[{args.split}] images={len(split_data.img_paths)} "
        f"queries={len(split_data.captions)} identities={len(set(split_data.image_pids))}"
    )

    # Build the host retrieval model and load checkpoint weights for inference only.
    num_classes = max(int(split_data.num_train_ids), 1)
    model = build_model(model_args, num_classes=num_classes)
    load_stats = load_checkpoint_for_inference(model, checkpoint_path)
    print(
        "Loaded checkpoint tensors: "
        f"{load_stats['loaded']} loaded, {load_stats['skipped_missing']} missing, "
        f"{load_stats['skipped_shape']} shape-mismatch, {load_stats['skipped_non_tensor']} non-tensor skipped."
    )
    model.to(device)
    if device.type == "cpu":
        model.float()
    model.eval()

    # Extract L2-normalized query text and gallery image features.
    text_features, query_pids = extract_text_features(
        model,
        split_data,
        text_length=int(model_args.text_length),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    image_features, gallery_pids = extract_image_features(
        model,
        split_data,
        img_size=parse_img_size(model_args.img_size),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    # Compute the full cosine similarity matrix and per-query margins.
    sim = text_features @ image_features.t()
    rows, skipped = compute_margin_rows(sim, query_pids, gallery_pids)
    if skipped:
        print(
            f"Warning: skipped {skipped} queries because they had no positive gallery image "
            "or no negative gallery image in the selected split."
        )
    summary = summarize_margins(rows, total_queries=len(query_pids), skipped=skipped)

    csv_path = output_dir / "margin_host.csv"
    summary_path = output_dir / "margin_summary.json"
    pdf_path = output_dir / "fig_margin_diagnostic.pdf"
    png_path = output_dir / "fig_margin_diagnostic.png"
    latex_path = output_dir / "latex_include_figure.txt"

    # Save tabular diagnostics, summary statistics, the figure, and the LaTeX snippet.
    save_csv(rows, csv_path)
    save_summary(summary, summary_path)
    plot_margins(rows, args, pdf_path, png_path)
    write_latex_snippet(latex_path)

    print_summary(
        {
            "CSV": csv_path,
            "summary JSON": summary_path,
            "PDF figure": pdf_path,
            "PNG figure": png_path,
            "LaTeX snippet": latex_path,
        },
        summary,
    )


if __name__ == "__main__":
    main()
