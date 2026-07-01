#!/usr/bin/env python3
"""Compare Rank-1 ambiguity rates for two TBPS/IAPR checkpoints.

For each text query q_i and gallery image x_j, the script computes cosine
similarity from L2-normalized standard model features, then measures the
positive--hard negative score margin

    s_i^+ = max_{j: y_j = y_i} sim(q_i, x_j)
    s_i^- = max_{j: y_j != y_i} sim(q_i, x_j)
    margin_i = s_i^+ - s_i^-

A query is ambiguous at threshold epsilon when margin_i < epsilon. The script
evaluates a baseline checkpoint and an IAPR/ours checkpoint on the same split,
saves per-query margins, saves a merged paired table, computes ambiguity-rate
statistics, and writes a compact paper-friendly comparison plot.

Example:

python scripts/plot_ambiguity_rate_compare.py \
--dataset_root /path/to/RSTPReid \
--dataset_name RSTPReid \
--split test \
--baseline_checkpoint /path/to/host_best.pth \
--ours_checkpoint /path/to/iapr_best.pth \
--baseline_name Host \
--ours_name "Host + IAPR" \
--output_dir outputs/ambiguity_rate/rstpreid \
--thresholds 0,0.01,0.05 \
--batch_size 128 \
--device cuda
"""

from __future__ import annotations

import argparse
import csv
import gc
import inspect
import json
import math
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

_DATASET_FACTORIES: Optional[Dict[str, type]] = None


def runtime_dependency_error(context: str, exc: ModuleNotFoundError) -> RuntimeError:
    error = RuntimeError(
        f"Could not import project runtime dependency while {context}. "
        "Install the repository dependencies before running inference."
    )
    error.__cause__ = exc
    return error


def dataset_factories() -> Dict[str, type]:
    global _DATASET_FACTORIES
    if _DATASET_FACTORIES is None:
        try:
            from datasets.cuhkpedes import CUHKPEDES
            from datasets.icfgpedes import ICFGPEDES
            from datasets.rstpreid import RSTPReid
        except ModuleNotFoundError as exc:
            raise runtime_dependency_error("loading dataset classes", exc)
        _DATASET_FACTORIES = {
            "CUHK-PEDES": CUHKPEDES,
            "ICFG-PEDES": ICFGPEDES,
            "RSTPReid": RSTPReid,
        }
    return _DATASET_FACTORIES


def text_dataset_class() -> type:
    try:
        from datasets.bases import TextDataset
    except ModuleNotFoundError as exc:
        raise runtime_dependency_error("loading the text dataset wrapper", exc)
    return TextDataset


def image_dataset_class() -> type:
    try:
        from datasets.bases import ImageDataset
    except ModuleNotFoundError as exc:
        raise runtime_dependency_error("loading the image dataset wrapper", exc)
    return ImageDataset


def build_eval_transforms(img_size: Tuple[int, int]) -> Any:
    try:
        from datasets.build import build_transforms
    except ModuleNotFoundError as exc:
        raise runtime_dependency_error("loading image transforms", exc)
    return build_transforms(img_size=img_size, is_train=False)


def build_repo_model(args: SimpleNamespace, num_classes: int) -> torch.nn.Module:
    try:
        from model import build_model
    except ModuleNotFoundError as exc:
        raise runtime_dependency_error("building the model", exc)
    return build_model(args, num_classes=num_classes)


def evaluator_class() -> type:
    try:
        from utils.metrics import Evaluator
    except ModuleNotFoundError as exc:
        raise runtime_dependency_error("loading the standard retrieval evaluator", exc)
    return Evaluator

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
\includegraphics[width=0.95\linewidth]{fig_ambiguity_rate_compare.pdf}
\caption{
Identity-level ambiguity diagnostic. For each text query, we compute the positive--hard negative score margin
$\Delta=s^+-s^-$, where $s^+$ is the highest score among same-identity gallery images and $s^-$ is the highest score among identity-wrong gallery images. A query is counted as ambiguous when $\Delta < \epsilon$. Lower values indicate fewer near-tie or hard-negative-confused queries.
}
\label{fig:ambiguity_rate}
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
        description="Compare ambiguity-rate curves for two trained TBPS/IAPR checkpoints."
    )
    parser.add_argument("--dataset_root", required=True, help="Dataset folder or parent folder containing the dataset.")
    parser.add_argument(
        "--dataset_name",
        required=True,
        choices=sorted(GENERIC_DATASET_CONFIGS),
        help="Dataset name.",
    )
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Dataset split to evaluate.")
    parser.add_argument("--baseline_checkpoint", required=True, help="Path to the baseline/host checkpoint.")
    parser.add_argument("--ours_checkpoint", required=True, help="Path to the ours/IAPR checkpoint.")
    parser.add_argument("--baseline_name", default="Host", help="Legend/display name for the baseline checkpoint.")
    parser.add_argument("--ours_name", default="IAPR", help="Legend/display name for the ours checkpoint.")
    parser.add_argument("--output_dir", default="outputs/ambiguity_rate_compare", help="Directory to save outputs.")
    parser.add_argument("--thresholds", default="0,0.01,0.02,0.05,0.10", help="Comma-separated margin thresholds.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for feature extraction.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers.")
    parser.add_argument("--device", default="cuda", help='Device, e.g. "cuda" or "cpu".')
    parser.add_argument("--max_queries", type=int, default=None, help="Optional query limit for debugging.")
    parser.add_argument("--img_size", default="384,128", help='Input image size as "height,width".')
    parser.add_argument("--text_length", type=int, default=77, help="Tokenized text length.")
    parser.add_argument("--pretrain_choice", default="ViT-B/16", help="CLIP backbone choice used by build_model(...).")
    parser.add_argument("--plot_type", default="bar", choices=["bar", "curve"], help="Plot style.")
    parser.add_argument("--dpi", type=int, default=300, help="PNG output DPI.")
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
            raise ValueError(f"Could not parse --img_size from value: {value!r}")
        return (int(parts[0]), int(parts[1]))
    if isinstance(value, Sequence) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    raise ValueError(f"Could not parse --img_size from value: {value!r}")


def parse_thresholds(value: str) -> List[float]:
    thresholds: List[float] = []
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            continue
        try:
            threshold = float(item)
        except ValueError as exc:
            raise ValueError(f"Could not parse threshold {item!r} from --thresholds={value!r}.") from exc
        if not math.isfinite(threshold):
            raise ValueError(f"Threshold must be finite, got {item!r}.")
        thresholds.append(threshold)
    if not thresholds:
        raise ValueError("--thresholds must contain at least one numeric threshold.")
    return thresholds


def threshold_label(threshold: float) -> str:
    if abs(threshold) < 1e-12:
        return "0"
    if abs(threshold) < 1.0:
        return f"{threshold:.2f}"
    return f"{threshold:g}"


def default_model_args() -> Dict[str, Any]:
    return {
        "tau": 0.015,
        "select_ratio": 0.4,
        "margin": 0.1,
        "lambda1_weight": 0.5,
        "lambda2_weight": 3.5,
        "local_rank": 0,
        "output_dir": "logs",
        "name": "baseline",
        "run_time": "",
        "seed": 1,
        "deterministic": True,
        "log_period": 100,
        "eval_period": 1,
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
        "wandb_entity": "",
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
        "img_size": (384, 128),
        "stride_size": 16,
        "text_length": 77,
        "vocab_size": 49408,
        "optimizer": "Adam",
        "lr": 1e-5,
        "bias_lr_factor": 2.0,
        "lr_factor": 5.0,
        "prototype_lr": None,
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
        "num_experts": 6,
        "topk": 2,
        "reduction": 8,
    }


def build_model_args(cli_args: argparse.Namespace) -> SimpleNamespace:
    cfg = default_model_args()
    cfg["dataset_name"] = cli_args.dataset_name
    cfg["root_dir"] = str(resolve_path(cli_args.dataset_root))
    cfg["training"] = False
    cfg["batch_size"] = cli_args.batch_size
    cfg["test_batch_size"] = cli_args.batch_size
    cfg["num_workers"] = cli_args.num_workers
    cfg["img_size"] = parse_img_size(cli_args.img_size)
    cfg["text_length"] = int(cli_args.text_length)
    cfg["pretrain_choice"] = cli_args.pretrain_choice
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
            f"Identity labels are required to compute ambiguity rates, but split {split} "
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
    raise ValueError("Identity labels are required, but an annotation has no pid/id field.")


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
    factory = dataset_factories().get(dataset_name)
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
        for key in ("state_dict", "model", "model_state_dict", "net", "network", "module"):
            value = checkpoint.get(key)
            if isinstance(value, Mapping):
                return value
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Checkpoint must be a mapping or contain a model state mapping.")
    return checkpoint


def strip_repeated_prefixes(key: str, prefixes: Sequence[str]) -> str:
    stripped = key
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if stripped.startswith(prefix):
                stripped = stripped[len(prefix):]
                changed = True
    return stripped


def candidate_state_keys(key: str) -> List[str]:
    raw = str(key)
    candidates: List[str] = []

    def add(candidate: str) -> None:
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    add(raw)
    no_module = strip_repeated_prefixes(raw, ("module.",))
    add(no_module)
    no_wrapper = strip_repeated_prefixes(no_module, ("model.", "net.", "network."))
    add(no_wrapper)
    normalized = strip_repeated_prefixes(raw, ("module.", "model.", "net.", "network."))
    add(normalized)
    for candidate in (no_wrapper, normalized):
        if candidate.startswith("base_model."):
            add(candidate[len("base_model."):])
        else:
            add(f"base_model.{candidate}")
    if no_module.startswith("model."):
        add(no_module[len("model."):])
    return candidates


def torch_load_checkpoint(checkpoint_path: Path) -> Any:
    try:
        return torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(checkpoint_path), map_location="cpu")


def load_checkpoint_for_inference(model: torch.nn.Module, checkpoint_path: Path) -> Dict[str, int]:
    checkpoint = torch_load_checkpoint(checkpoint_path)
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


def feature_tensor(output: Any, kind: str) -> torch.Tensor:
    if isinstance(output, (tuple, list)):
        output = output[0]
    if not torch.is_tensor(output):
        raise RuntimeError(f"model.encode_{kind}(...) returned {type(output)!r}, expected a tensor.")
    return output


def call_model_encoder(encoder: Any, tensor: torch.Tensor, kind: str) -> torch.Tensor:
    params = inspect.signature(encoder).parameters
    output = encoder(tensor, 0) if len(params) >= 2 else encoder(tensor)
    return feature_tensor(output, kind)


@torch.inference_mode()
def extract_text_features(
    model: torch.nn.Module,
    split_data: SplitData,
    text_length: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    TextDataset = text_dataset_class()
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
        feats = call_model_encoder(model.encode_text, tokens, "text").float()
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
    ImageDataset = image_dataset_class()
    transform = build_eval_transforms(img_size)
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
        feats = call_model_encoder(model.encode_image, images, "image").float()
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


def load_model_for_checkpoint(
    checkpoint_path: str | Path,
    model_args: SimpleNamespace,
    split_data: SplitData,
    device: torch.device,
) -> Tuple[torch.nn.Module, SimpleNamespace, Dict[str, int], Path]:
    checkpoint = resolve_path(checkpoint_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint}")

    run_args = SimpleNamespace(**vars(model_args))
    num_classes = max(int(split_data.num_train_ids), 1)
    model = build_repo_model(run_args, num_classes=num_classes)
    load_stats = load_checkpoint_for_inference(model, checkpoint)

    model.to(device)
    if device.type == "cpu":
        model.float()
    model.eval()
    return model, run_args, load_stats, checkpoint


def compute_margins_for_checkpoint(
    checkpoint_path: str | Path,
    model_args: SimpleNamespace,
    split_data: SplitData,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    model: Optional[torch.nn.Module] = None
    text_features: Optional[torch.Tensor] = None
    image_features: Optional[torch.Tensor] = None
    query_pids: Optional[torch.Tensor] = None
    gallery_pids: Optional[torch.Tensor] = None
    sim: Optional[torch.Tensor] = None

    try:
        model, run_args, load_stats, checkpoint = load_model_for_checkpoint(
            checkpoint_path,
            model_args,
            split_data,
            device,
        )
        text_features, query_pids = extract_text_features(
            model,
            split_data,
            text_length=int(run_args.text_length),
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
        )
        image_features, gallery_pids = extract_image_features(
            model,
            split_data,
            img_size=parse_img_size(run_args.img_size),
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
        )

        sim = text_features @ image_features.t()
        rows, skipped = compute_margin_rows(sim, query_pids, gallery_pids)
        margin_summary = summarize_margins(rows, total_queries=len(query_pids), skipped=skipped)
        metadata: Dict[str, Any] = {
            "checkpoint": str(checkpoint),
            "load_stats": load_stats,
            "margin_summary": margin_summary,
        }
        return rows, metadata
    finally:
        model = None
        text_features = None
        image_features = None
        query_pids = None
        gallery_pids = None
        sim = None
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def build_standard_eval_loaders(
    split_data: SplitData,
    model_args: SimpleNamespace,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> Tuple[DataLoader, DataLoader]:
    ImageDataset = image_dataset_class()
    TextDataset = text_dataset_class()
    transform = build_eval_transforms(parse_img_size(model_args.img_size))
    image_set = ImageDataset(split_data.image_pids, split_data.img_paths, transform=transform)
    text_set = TextDataset(split_data.caption_pids, split_data.captions, text_length=int(model_args.text_length))
    image_loader = DataLoader(
        image_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    text_loader = DataLoader(
        text_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    return image_loader, text_loader


def build_standard_evaluator(image_loader: DataLoader, text_loader: DataLoader, model_args: SimpleNamespace) -> Any:
    Evaluator = evaluator_class()
    params = inspect.signature(Evaluator).parameters
    if "args" in params or len(params) >= 3:
        return Evaluator(image_loader, text_loader, model_args)
    return Evaluator(image_loader, text_loader)


def numeric_mapping(mapping: Mapping[str, Any]) -> Dict[str, float]:
    numeric: Dict[str, float] = {}
    for key, value in mapping.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            numeric[str(key)] = float(value)
    return numeric


def run_standard_retrieval_eval(evaluator: Any, model: torch.nn.Module) -> Tuple[Dict[str, float], Dict[str, Any]]:
    if hasattr(evaluator, "eval_metrics"):
        metrics = evaluator.eval_metrics(model)
        return numeric_mapping(metrics), {}

    eval_params = inspect.signature(evaluator.eval).parameters
    kwargs: Dict[str, Any] = {}
    if "return_metrics" in eval_params:
        kwargs["return_metrics"] = True
    result = evaluator.eval(model, **kwargs)

    if isinstance(result, tuple):
        top1 = result[0]
        metrics = result[1] if len(result) > 1 and isinstance(result[1], Mapping) else {"R1": top1}
        best_metrics = result[2] if len(result) > 2 and isinstance(result[2], Mapping) else {}
        return numeric_mapping(metrics), dict(best_metrics)

    return {"R1": float(result)}, {}


def split_metric_key(key: str) -> Tuple[str, str]:
    tail = key.rsplit("/", 1)[-1]
    if "_" in tail:
        task, metric = key.rsplit("_", 1)
        return task, metric
    if "/" in key:
        task, metric = key.rsplit("/", 1)
        return task, metric
    return "retrieval", key


def grouped_retrieval_metrics(metrics: Mapping[str, float]) -> List[Tuple[str, Dict[str, float]]]:
    wanted = {"R1", "R5", "R10", "mAP", "mINP", "rSum"}
    groups: Dict[str, Dict[str, float]] = {}
    order: List[str] = []
    for key, value in metrics.items():
        task, metric = split_metric_key(str(key))
        if metric not in wanted:
            continue
        if task not in groups:
            groups[task] = {}
            order.append(task)
        groups[task][metric] = float(value)
    return [(task, groups[task]) for task in order]


def print_retrieval_metrics(label: str, metrics: Mapping[str, float], best_metrics: Mapping[str, Any]) -> None:
    print(f"[{label}] Standard test-set retrieval metrics:")
    metric_order = [("R1", "R@1"), ("R5", "R@5"), ("R10", "R@10"), ("mAP", "mAP"), ("mINP", "mINP"), ("rSum", "rSum")]
    if best_metrics:
        best_parts = []
        for key, display in metric_order:
            if key in best_metrics:
                best_parts.append(f"{display}={float(best_metrics[key]):.2f}")
        task = best_metrics.get("task", "best")
        if best_parts:
            print(f"  best ({task}): " + ", ".join(best_parts))

    groups = grouped_retrieval_metrics(metrics)
    if not groups:
        print(f"  No standard R@/mAP metrics were returned. Raw metric keys: {sorted(metrics)}")
        return

    for task, values in groups:
        parts = [f"{display}={values[key]:.2f}" for key, display in metric_order if key in values]
        print(f"  {task}: " + ", ".join(parts))


def evaluate_checkpoint_on_test_split(
    label: str,
    checkpoint_path: str | Path,
    model_args: SimpleNamespace,
    test_split_data: SplitData,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Dict[str, Any]:
    model: Optional[torch.nn.Module] = None
    try:
        model, run_args, load_stats, checkpoint = load_model_for_checkpoint(
            checkpoint_path,
            model_args,
            test_split_data,
            device,
        )
        print_load_stats(label, {"load_stats": load_stats})
        image_loader, text_loader = build_standard_eval_loaders(
            test_split_data,
            run_args,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
        )
        evaluator = build_standard_evaluator(image_loader, text_loader, run_args)
        metrics, best_metrics = run_standard_retrieval_eval(evaluator, model)
        print_retrieval_metrics(label, metrics, best_metrics)
        return {
            "checkpoint": str(checkpoint),
            "load_stats": load_stats,
            "retrieval_metrics": metrics,
            "best_retrieval_metrics": dict(best_metrics),
        }
    finally:
        model = None
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


MARGIN_COLUMNS = [
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

MERGED_COLUMNS = [
    "query_index",
    "query_pid",
    "margin_baseline",
    "margin_ours",
    "delta_margin",
    "baseline_best_pos_index",
    "baseline_best_pos_pid",
    "baseline_best_pos_score",
    "baseline_hard_neg_index",
    "baseline_hard_neg_pid",
    "baseline_hard_neg_score",
    "ours_best_pos_index",
    "ours_best_pos_pid",
    "ours_best_pos_score",
    "ours_hard_neg_index",
    "ours_hard_neg_pid",
    "ours_hard_neg_score",
]

AMBIGUITY_RATE_COLUMNS = [
    "threshold",
    "baseline_count",
    "ours_count",
    "baseline_percent",
    "ours_percent",
    "delta_percent_point",
    "relative_reduction_percent",
]


def save_csv(rows: Sequence[Mapping[str, Any]], path: Path, columns: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_json(data: Mapping[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
        file.write("\n")


def write_latex_snippet(path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        file.write(LATEX_SNIPPET)
        file.write("\n")


def rows_by_query_key(rows: Sequence[Mapping[str, Any]], label: str) -> Dict[Tuple[int, int], Mapping[str, Any]]:
    result: Dict[Tuple[int, int], Mapping[str, Any]] = {}
    for row in rows:
        key = (int(row["query_index"]), int(row["query_pid"]))
        if key in result:
            raise ValueError(f"{label} rows contain duplicate query key {key}.")
        result[key] = row
    return result


def merge_margin_rows(
    baseline_rows: Sequence[Mapping[str, Any]],
    ours_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    baseline_map = rows_by_query_key(baseline_rows, "baseline")
    ours_map = rows_by_query_key(ours_rows, "ours")
    baseline_keys = set(baseline_map)
    ours_keys = set(ours_map)
    if baseline_keys != ours_keys:
        baseline_only = sorted(baseline_keys - ours_keys)[:5]
        ours_only = sorted(ours_keys - baseline_keys)[:5]
        raise ValueError(
            "The two checkpoint runs produced different usable query sets. "
            f"baseline-only examples={baseline_only}; ours-only examples={ours_only}."
        )

    merged: List[Dict[str, Any]] = []
    for key in [(int(row["query_index"]), int(row["query_pid"])) for row in baseline_rows]:
        baseline = baseline_map[key]
        ours = ours_map[key]
        baseline_margin = float(baseline["margin"])
        ours_margin = float(ours["margin"])
        merged.append(
            {
                "query_index": int(key[0]),
                "query_pid": int(key[1]),
                "margin_baseline": baseline_margin,
                "margin_ours": ours_margin,
                "delta_margin": ours_margin - baseline_margin,
                "baseline_best_pos_index": int(baseline["best_pos_index"]),
                "baseline_best_pos_pid": int(baseline["best_pos_pid"]),
                "baseline_best_pos_score": float(baseline["best_pos_score"]),
                "baseline_hard_neg_index": int(baseline["hard_neg_index"]),
                "baseline_hard_neg_pid": int(baseline["hard_neg_pid"]),
                "baseline_hard_neg_score": float(baseline["hard_neg_score"]),
                "ours_best_pos_index": int(ours["best_pos_index"]),
                "ours_best_pos_pid": int(ours["best_pos_pid"]),
                "ours_best_pos_score": float(ours["best_pos_score"]),
                "ours_hard_neg_index": int(ours["hard_neg_index"]),
                "ours_hard_neg_pid": int(ours["hard_neg_pid"]),
                "ours_hard_neg_score": float(ours["hard_neg_score"]),
            }
        )
    return merged


def compute_ambiguity_rates(
    merged_rows: Sequence[Mapping[str, Any]],
    thresholds: Sequence[float],
) -> List[Dict[str, Any]]:
    if not merged_rows:
        raise RuntimeError("No merged query rows are available for ambiguity-rate computation.")

    baseline_margins = np.array([float(row["margin_baseline"]) for row in merged_rows], dtype=np.float64)
    ours_margins = np.array([float(row["margin_ours"]) for row in merged_rows], dtype=np.float64)
    num_queries = int(len(merged_rows))
    rows: List[Dict[str, Any]] = []

    for threshold in thresholds:
        baseline_count = int(np.sum(baseline_margins < threshold))
        ours_count = int(np.sum(ours_margins < threshold))
        baseline_percent = float(baseline_count / num_queries * 100.0)
        ours_percent = float(ours_count / num_queries * 100.0)
        delta_percent_point = float(ours_percent - baseline_percent)
        if baseline_percent > 0.0:
            relative_reduction_percent = float((baseline_percent - ours_percent) / baseline_percent * 100.0)
        else:
            relative_reduction_percent = 0.0

        rows.append(
            {
                "threshold": float(threshold),
                "baseline_count": baseline_count,
                "ours_count": ours_count,
                "baseline_percent": baseline_percent,
                "ours_percent": ours_percent,
                "delta_percent_point": delta_percent_point,
                "relative_reduction_percent": relative_reduction_percent,
            }
        )

    return rows


def paired_delta_summary(merged_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not merged_rows:
        raise RuntimeError("No merged query rows are available for paired-delta summary.")

    baseline = np.array([float(row["margin_baseline"]) for row in merged_rows], dtype=np.float64)
    ours = np.array([float(row["margin_ours"]) for row in merged_rows], dtype=np.float64)
    delta = ours - baseline
    return {
        "mean_delta_margin": float(np.mean(delta)),
        "median_delta_margin": float(np.median(delta)),
        "percent_delta_positive": float(np.mean(delta > 0.0) * 100.0),
        "percent_delta_negative": float(np.mean(delta < 0.0) * 100.0),
        "percent_negative_to_positive": float(np.mean((baseline < 0.0) & (ours >= 0.0)) * 100.0),
        "percent_positive_to_negative": float(np.mean((baseline >= 0.0) & (ours < 0.0)) * 100.0),
    }


def plot_ambiguity_rates(
    ambiguity_rates: Sequence[Mapping[str, Any]],
    baseline_name: str,
    ours_name: str,
    plot_type: str,
    dpi: int,
    pdf_path: Path,
    png_path: Path,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to plot the ambiguity-rate diagnostic.") from exc

    labels = [threshold_label(float(row["threshold"])) for row in ambiguity_rates]
    baseline_values = np.array([float(row["baseline_percent"]) for row in ambiguity_rates], dtype=np.float64)
    ours_values = np.array([float(row["ours_percent"]) for row in ambiguity_rates], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    colors = ("#4C78A8", "#F58518")

    if plot_type == "bar":
        x = np.arange(len(labels))
        width = 0.36
        baseline_bars = ax.bar(x - width / 2, baseline_values, width, label=baseline_name, color=colors[0])
        ours_bars = ax.bar(x + width / 2, ours_values, width, label=ours_name, color=colors[1])
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ymax = float(max(np.max(baseline_values), np.max(ours_values), 1.0))
        offset = max(ymax * 0.035, 0.35)
        for bars in (baseline_bars, ours_bars):
            for bar in bars:
                height = float(bar.get_height())
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + offset,
                    f"{height:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
        ax.set_ylim(0.0, ymax + 3.0 * offset)
    elif plot_type == "curve":
        x = np.array([float(row["threshold"]) for row in ambiguity_rates], dtype=np.float64)
        ax.plot(x, baseline_values, marker="o", linewidth=1.4, markersize=3.5, label=baseline_name, color=colors[0])
        ax.plot(x, ours_values, marker="s", linewidth=1.4, markersize=3.5, label=ours_name, color=colors[1])
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ymax = float(max(np.max(baseline_values), np.max(ours_values), 1.0))
        ax.set_ylim(0.0, ymax * 1.12 + 0.5)
    else:
        raise ValueError(f"Unsupported plot_type: {plot_type!r}")

    ax.set_xlabel("Margin threshold ε")
    ax.set_ylabel("Ambiguous queries (%)")
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def print_load_stats(label: str, metadata: Mapping[str, Any]) -> None:
    stats = metadata.get("load_stats", {})
    print(
        f"[{label}] Loaded checkpoint tensors: "
        f"{stats.get('loaded', 0)} loaded, "
        f"{stats.get('skipped_missing', 0)} missing/extra keys, "
        f"{stats.get('skipped_shape', 0)} shape-mismatch, "
        f"{stats.get('skipped_non_tensor', 0)} non-tensor entries skipped."
    )


def print_terminal_summary(
    dataset_name: str,
    split: str,
    split_data: SplitData,
    baseline_summary: Mapping[str, Any],
    ours_summary: Mapping[str, Any],
    ambiguity_rates: Sequence[Mapping[str, Any]],
    delta_summary: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> None:
    print(
        f"[Dataset] {dataset_name} split={split} "
        f"queries={len(split_data.captions)} gallery={len(split_data.img_paths)}"
    )
    print(
        f"[Baseline] mean={baseline_summary['mean_margin']:.6f}, "
        f"median={baseline_summary['median_margin']:.6f}, "
        f"negative={baseline_summary['percent_margin_negative']:.2f}%, "
        f"<0.01={baseline_summary['percent_margin_below_0_01']:.2f}%, "
        f"<0.05={baseline_summary['percent_margin_below_0_05']:.2f}%"
    )
    print(
        f"[Ours]     mean={ours_summary['mean_margin']:.6f}, "
        f"median={ours_summary['median_margin']:.6f}, "
        f"negative={ours_summary['percent_margin_negative']:.2f}%, "
        f"<0.01={ours_summary['percent_margin_below_0_01']:.2f}%, "
        f"<0.05={ours_summary['percent_margin_below_0_05']:.2f}%"
    )

    print("\nAmbiguity rate:")
    print("threshold | baseline % | ours % | delta pp | relative reduction %")
    for row in ambiguity_rates:
        print(
            f"{threshold_label(float(row['threshold'])):<9} | "
            f"{row['baseline_percent']:10.2f} | "
            f"{row['ours_percent']:6.2f} | "
            f"{row['delta_percent_point']:8.2f} | "
            f"{row['relative_reduction_percent']:20.2f}"
        )

    print("\nPaired delta:")
    print(f"mean delta margin: {delta_summary['mean_delta_margin']:.6f}")
    print(f"median delta margin: {delta_summary['median_delta_margin']:.6f}")
    print(f"delta > 0: {delta_summary['percent_delta_positive']:.2f}%")
    print(f"negative -> positive: {delta_summary['percent_negative_to_positive']:.2f}%")
    print(f"positive -> negative: {delta_summary['percent_positive_to_negative']:.2f}%")

    print("\nOutputs:")
    for label, path in paths.items():
        print(f"  {label}: {path}")


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    if args.num_workers < 0:
        raise ValueError("--num_workers must be non-negative.")
    if args.text_length <= 0:
        raise ValueError("--text_length must be positive.")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive.")

    thresholds = parse_thresholds(args.thresholds)
    dataset_root = resolve_path(args.dataset_root)
    baseline_checkpoint = resolve_path(args.baseline_checkpoint)
    ours_checkpoint = resolve_path(args.ours_checkpoint)
    output_dir = resolve_path(args.output_dir)
    if not baseline_checkpoint.is_file():
        raise FileNotFoundError(f"Baseline checkpoint file not found: {baseline_checkpoint}")
    if not ours_checkpoint.is_file():
        raise FileNotFoundError(f"Ours checkpoint file not found: {ours_checkpoint}")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    model_args = build_model_args(args)
    test_split_data = load_split_data(args.dataset_name, dataset_root, "test")
    validate_split_data(test_split_data, "test")

    print(
        f"[Verification] Running standard retrieval evaluation on test split "
        f"for {args.dataset_name}: queries={len(test_split_data.captions)} "
        f"gallery={len(test_split_data.img_paths)}"
    )
    print(f"[Baseline] Verifying {baseline_checkpoint}")
    baseline_eval_meta = evaluate_checkpoint_on_test_split(
        "Baseline",
        baseline_checkpoint,
        model_args,
        test_split_data,
        device,
        args.batch_size,
        args.num_workers,
    )
    print(f"[Ours] Verifying {ours_checkpoint}")
    ours_eval_meta = evaluate_checkpoint_on_test_split(
        "Ours",
        ours_checkpoint,
        model_args,
        test_split_data,
        device,
        args.batch_size,
        args.num_workers,
    )

    split_data = test_split_data if args.split == "test" else load_split_data(args.dataset_name, dataset_root, args.split)
    split_data = limit_queries(split_data, args.max_queries)
    validate_split_data(split_data, args.split)

    print(
        f"[Ambiguity] {args.dataset_name} split={args.split} "
        f"queries={len(split_data.captions)} gallery={len(split_data.img_paths)}"
    )
    print(f"[Baseline] Computing ambiguity margins from {baseline_checkpoint}")
    baseline_rows, baseline_meta = compute_margins_for_checkpoint(
        baseline_checkpoint,
        model_args,
        split_data,
        device,
        args.batch_size,
        args.num_workers,
    )
    print_load_stats("Baseline", baseline_meta)

    print(f"[Ours] Computing ambiguity margins from {ours_checkpoint}")
    ours_rows, ours_meta = compute_margins_for_checkpoint(
        ours_checkpoint,
        model_args,
        split_data,
        device,
        args.batch_size,
        args.num_workers,
    )
    print_load_stats("Ours", ours_meta)

    merged_rows = merge_margin_rows(baseline_rows, ours_rows)
    ambiguity_rates = compute_ambiguity_rates(merged_rows, thresholds)
    delta_summary = paired_delta_summary(merged_rows)
    baseline_summary = baseline_meta["margin_summary"]
    ours_summary = ours_meta["margin_summary"]

    summary = {
        "dataset_name": args.dataset_name,
        "split": args.split,
        "baseline_checkpoint": str(baseline_checkpoint),
        "ours_checkpoint": str(ours_checkpoint),
        "baseline_name": args.baseline_name,
        "ours_name": args.ours_name,
        "num_queries_total": int(len(split_data.captions)),
        "num_queries_used": int(len(merged_rows)),
        "thresholds": [float(threshold) for threshold in thresholds],
        "baseline_margin_summary": baseline_summary,
        "ours_margin_summary": ours_summary,
        "ambiguity_rates": ambiguity_rates,
        "paired_delta_summary": delta_summary,
        "baseline_load_stats": baseline_meta.get("load_stats", {}),
        "ours_load_stats": ours_meta.get("load_stats", {}),
        "test_retrieval_verification": {
            "baseline": baseline_eval_meta,
            "ours": ours_eval_meta,
        },
    }

    paths = {
        "baseline_margins.csv": output_dir / "baseline_margins.csv",
        "ours_margins.csv": output_dir / "ours_margins.csv",
        "merged_margins.csv": output_dir / "merged_margins.csv",
        "ambiguity_rates.csv": output_dir / "ambiguity_rates.csv",
        "summary.json": output_dir / "summary.json",
        "fig_ambiguity_rate_compare.pdf": output_dir / "fig_ambiguity_rate_compare.pdf",
        "fig_ambiguity_rate_compare.png": output_dir / "fig_ambiguity_rate_compare.png",
        "latex_include_figure.txt": output_dir / "latex_include_figure.txt",
    }

    save_csv(baseline_rows, paths["baseline_margins.csv"], MARGIN_COLUMNS)
    save_csv(ours_rows, paths["ours_margins.csv"], MARGIN_COLUMNS)
    save_csv(merged_rows, paths["merged_margins.csv"], MERGED_COLUMNS)
    save_csv(ambiguity_rates, paths["ambiguity_rates.csv"], AMBIGUITY_RATE_COLUMNS)
    save_json(summary, paths["summary.json"])
    plot_ambiguity_rates(
        ambiguity_rates,
        baseline_name=args.baseline_name,
        ours_name=args.ours_name,
        plot_type=args.plot_type,
        dpi=args.dpi,
        pdf_path=paths["fig_ambiguity_rate_compare.pdf"],
        png_path=paths["fig_ambiguity_rate_compare.png"],
    )
    write_latex_snippet(paths["latex_include_figure.txt"])

    print_terminal_summary(
        args.dataset_name,
        args.split,
        split_data,
        baseline_summary,
        ours_summary,
        ambiguity_rates,
        delta_summary,
        paths,
    )


if __name__ == "__main__":
    main()





