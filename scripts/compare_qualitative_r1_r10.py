#!/usr/bin/env python3
"""Compare qualitative text-to-image R1-R10 retrieval results for two Regu-IRRA checkpoints."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import inspect
import json
import math
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODEL_FAMILY = "Regu-IRRA"
RETRIEVAL_MODES = ("auto", "global")
SORT_MODES = (
    "best_r1_green_gap",
    "strict_showcase",
    "best_r1_baseline_not_r1",
    "best_r1_baseline_not_r10",
    "rank_improvement",
    "rr_improvement",
    "best_r1",
    "baseline_fail_best_success",
    "all",
)
DATASET_DIR_NAMES = {
    "CUHK-PEDES": "CUHK-PEDES",
    "ICFG-PEDES": "ICFG-PEDES",
    "RSTPReid": "RSTPReid",
}

Image = None
ImageDraw = None
ImageFont = None
ImageOps = None
RESAMPLE_LANCZOS = None

CSV_FIELD_ORDER = [
    "query_index",
    "query_pid",
    "caption",
    "baseline_top1_identity",
    "best_top1_identity",
    "baseline_top1_index",
    "best_top1_index",
    "baseline_top1_score",
    "best_top1_score",
    "baseline_r1_correct",
    "best_r1_correct",
    "baseline_z_topK",
    "best_z_topK",
    "baseline_green_count@K",
    "best_green_count@K",
    "green_gap",
    "baseline_first_correct_rank",
    "best_first_correct_rank",
    "rank_gain",
    "baseline_has_correct_top10",
    "best_has_correct_top10",
    "rank_improvement",
    "reciprocal_rank_improvement",
    "baseline_discounted_green_score@K",
    "best_discounted_green_score@K",
    "early_green_gap",
    "showcase_score",
    "baseline_top10_positive_count",
    "best_top10_positive_count",
    "baseline_top_indices",
    "best_top_indices",
    "baseline_top_pids",
    "best_top_pids",
    "baseline_top_scores",
    "best_top_scores",
    "baseline_top_image_paths",
    "best_top_image_paths",
]


@dataclass
class SplitMeta:
    captions: List[str]
    caption_pids: List[int]
    img_paths: List[str]
    image_pids: List[int]


@dataclass
class EmbeddingBundle:
    text_features: torch.Tensor
    image_features: torch.Tensor
    query_pids: torch.Tensor
    gallery_pids: torch.Tensor
    metadata: Dict[str, Any]
    load_stats: Dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate two Regu-IRRA checkpoints on the full test split, rank every caption "
            "query against the same gallery, and save R1-R10 qualitative comparison figures."
        )
    )
    parser.add_argument("--config", "--config_file", dest="config", default=None, help="Optional saved configs.yaml.")
    parser.add_argument("--dataset_name", default=None, help="Dataset name: CUHK-PEDES, ICFG-PEDES, or RSTPReid.")
    parser.add_argument("--data_root", "--root_dir", dest="data_root", default=None, help="Dataset parent directory.")
    parser.add_argument("--baseline_ckpt", "--baseline_checkpoint", dest="baseline_ckpt", required=True)
    parser.add_argument("--best_ckpt", "--best_checkpoint", dest="best_ckpt", required=True)
    parser.add_argument("--baseline_name", default="Baseline")
    parser.add_argument("--best_name", default="Prototype/IAPR")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_figs", type=int, default=20)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda", help="Torch device, e.g. cuda, cuda:0, or cpu.")
    parser.add_argument("--gpu_id", type=int, default=None, help="Optional CUDA device index; overrides --device cuda.")
    parser.add_argument("--sort_mode", choices=SORT_MODES, default="best_r1_green_gap")
    parser.add_argument(
        "--min_best_green",
        type=int,
        default=None,
        help="Minimum prototype/IAPR correct identity images in top-K for strict_showcase.",
    )
    parser.add_argument(
        "--max_baseline_green",
        type=int,
        default=None,
        help="Maximum baseline correct identity images in top-K for strict_showcase.",
    )
    parser.add_argument("--save_csv", dest="save_csv", action="store_true", default=True)
    parser.add_argument("--no_save_csv", dest="save_csv", action="store_false")
    parser.add_argument("--save_json", action="store_true", help="Also save full and selected row JSON files.")
    parser.add_argument("--cache_embeddings", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--retrieval_mode",
        choices=RETRIEVAL_MODES,
        default="auto",
        help="Regu-IRRA official retrieval mode. auto resolves to global.",
    )
    parser.add_argument("--batch_size", type=int, default=None, help="Override test batch size.")
    parser.add_argument("--num_workers", type=int, default=None, help="Override dataloader workers.")
    parser.add_argument("--img_size", default=None, help='Override image size as "height,width".')
    parser.add_argument("--text_length", type=int, default=None, help="Override tokenized text length.")
    parser.add_argument("--pretrain_choice", default=None, help="Override CLIP backbone choice.")
    parser.add_argument("--prototype", dest="prototype", action="store_true", default=None)
    parser.add_argument("--no_prototype", dest="prototype", action="store_false")
    parser.add_argument("--use_loss_id", dest="use_loss_id", action="store_true", default=None)
    parser.add_argument("--no_use_loss_id", dest="use_loss_id", action="store_false")
    parser.add_argument("--dpi", type=int, default=150, help="PNG DPI metadata.")
    return parser.parse_args()


def resolve_path(path: str | Path, base: Path = REPO_ROOT) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = base / resolved
    return resolved.resolve()


def config_to_dict(config: Any) -> Dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, Mapping):
        return dict(config)
    if hasattr(config, "__dict__"):
        return dict(vars(config))
    return {key: getattr(config, key) for key in dir(config) if not key.startswith("_")}


def parse_img_size(value: Any) -> Tuple[int, int]:
    if value is None:
        return (384, 128)
    if isinstance(value, int):
        return (value, value)
    if isinstance(value, str):
        cleaned = value.strip().strip("()[]")
        parts = [part.strip() for part in cleaned.split(",") if part.strip()]
        if len(parts) != 2:
            raise ValueError(f"Could not parse image size from value: {value!r}")
        return (int(parts[0]), int(parts[1]))
    if isinstance(value, Sequence) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    raise ValueError(f"Could not parse image size from value: {value!r}")


def normalize_dataset_root(dataset_name: str, root_dir: str | Path) -> Path:
    root = resolve_path(root_dir)
    dataset_dir_name = DATASET_DIR_NAMES.get(dataset_name)
    if dataset_dir_name and root.name.lower() == dataset_dir_name.lower():
        return root.parent
    return root


def default_model_args() -> Dict[str, Any]:
    return {
        "local_rank": 0,
        "name": "baseline",
        "output_dir": "logs",
        "log_period": 100,
        "eval_period": 1,
        "eval_after_epoch": 0,
        "val_dataset": "test",
        "resume": False,
        "resume_ckpt_file": "",
        "seed": 1,
        "deterministic": True,
        "wandb": False,
        "wandb_project": "prototype-regu-irra",
        "wandb_entity": None,
        "wandb_name": "",
        "wandb_tags": "",
        "wandb_mode": "disabled",
        "pretrain_choice": "ViT-B/16",
        "temperature": 0.02,
        "img_aug": False,
        "cmt_depth": 4,
        "masked_token_rate": 0.8,
        "masked_token_unchanged_rate": 0.1,
        "lr_factor": 5.0,
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
        "momentum": 0.9,
        "weight_decay": 4e-5,
        "weight_decay_bias": 0.0,
        "alpha": 0.9,
        "beta": 0.999,
        "num_epoch": 60,
        "milestones": (20, 50),
        "gamma": 0.1,
        "warmup_factor": 0.1,
        "warmup_epochs": 5,
        "warmup_method": "linear",
        "lrscheduler": "cosine",
        "target_lr": 0,
        "power": 0.9,
        "dataset_name": "CUHK-PEDES",
        "sampler": "random",
        "num_instance": 4,
        "root_dir": "./data",
        "batch_size": 128,
        "test_batch_size": 512,
        "num_workers": 8,
        "training": False,
        "distributed": False,
    }


def build_eval_args(cli_args: argparse.Namespace) -> SimpleNamespace:
    from utils.iotools import load_train_configs

    cfg = default_model_args()
    if cli_args.config:
        config_path = resolve_path(cli_args.config)
        if not config_path.is_file():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        cfg.update(config_to_dict(load_train_configs(str(config_path))))

    dataset_name = cli_args.dataset_name or cfg.get("dataset_name")
    if not dataset_name:
        raise ValueError("--dataset_name is required when the config does not provide dataset_name.")
    root_value = cli_args.data_root or cfg.get("root_dir") or cfg.get("data_root")
    if not root_value:
        raise ValueError("--data_root is required when the config does not provide root_dir.")

    cfg["dataset_name"] = str(dataset_name)
    cfg["root_dir"] = str(normalize_dataset_root(str(dataset_name), root_value))
    cfg["training"] = False
    cfg["distributed"] = False
    cfg["seed"] = int(cli_args.seed)

    if cli_args.batch_size is not None:
        if cli_args.batch_size <= 0:
            raise ValueError("--batch_size must be positive.")
        cfg["batch_size"] = int(cli_args.batch_size)
        cfg["test_batch_size"] = int(cli_args.batch_size)
    elif "test_batch_size" not in cfg or cfg["test_batch_size"] is None:
        cfg["test_batch_size"] = int(cfg.get("batch_size", 512))

    if cli_args.num_workers is not None:
        if cli_args.num_workers < 0:
            raise ValueError("--num_workers must be non-negative.")
        cfg["num_workers"] = int(cli_args.num_workers)

    if cli_args.img_size is not None:
        cfg["img_size"] = parse_img_size(cli_args.img_size)
    else:
        cfg["img_size"] = parse_img_size(cfg.get("img_size", (384, 128)))

    if cli_args.text_length is not None:
        if cli_args.text_length <= 0:
            raise ValueError("--text_length must be positive.")
        cfg["text_length"] = int(cli_args.text_length)
    if cli_args.pretrain_choice is not None:
        cfg["pretrain_choice"] = cli_args.pretrain_choice
    if cli_args.prototype is not None:
        cfg["prototype"] = bool(cli_args.prototype)
    if cli_args.use_loss_id is not None:
        cfg["use_loss_id"] = bool(cli_args.use_loss_id)

    cfg.setdefault("prototype", False)
    cfg.setdefault("use_loss_id", False)
    cfg.setdefault("no_pbt", False)
    cfg.setdefault("prototype_feature", "auto")
    cfg.setdefault("prototype_projector", "default")
    cfg.setdefault("prototype_residual_scale", 0.1)
    cfg.setdefault("prototype_per_id", 2)
    cfg.setdefault("prototype_dim", 512)
    cfg.setdefault("prototype_kmeans_iters", 20)
    cfg.setdefault("prototype_warmup_epochs", 0)
    cfg.setdefault("prototype_tau", 0.05)
    cfg.setdefault("prototype_hard_k", 16)
    cfg.setdefault("prototype_id_weight", 0.2)
    cfg.setdefault("prototype_momentum", 0.2)
    cfg.setdefault("prototype_lr", None)
    return SimpleNamespace(**cfg)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_arg_with_gpu(device_arg: str, gpu_id: Optional[int]) -> str:
    if gpu_id is None:
        return device_arg
    device = torch.device(device_arg)
    if device.type != "cuda":
        return device_arg
    return f"cuda:{gpu_id}"


def resolve_device(device_arg: str) -> torch.device:
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA was requested but is not available; using CPU.")
        return torch.device("cpu")
    return device


def resolve_retrieval_mode(mode: str) -> str:
    if mode == "auto":
        return "global"
    if mode != "global":
        raise ValueError(f"Unsupported {MODEL_FAMILY} retrieval mode: {mode!r}")
    return mode


def split_meta_from_loaders(test_img_loader: Any, test_txt_loader: Any) -> SplitMeta:
    img_set = getattr(test_img_loader, "test_img_set", getattr(test_img_loader, "dataset", None))
    txt_set = getattr(test_txt_loader, "test_txt_set", getattr(test_txt_loader, "dataset", None))
    required_img = ("image_pids", "img_paths")
    required_txt = ("caption_pids", "captions")
    if img_set is None or any(not hasattr(img_set, key) for key in required_img):
        raise RuntimeError("The test image loader does not expose image_pids/img_paths.")
    if txt_set is None or any(not hasattr(txt_set, key) for key in required_txt):
        raise RuntimeError("The test text loader does not expose caption_pids/captions.")

    return SplitMeta(
        captions=[str(caption) for caption in txt_set.captions],
        caption_pids=[int(pid) for pid in txt_set.caption_pids],
        img_paths=[str(path) for path in img_set.img_paths],
        image_pids=[int(pid) for pid in img_set.image_pids],
    )


def infer_num_classes(model_args: SimpleNamespace, default: int = 11003) -> int:
    try:
        import datasets.build as dataset_build

        factory = getattr(dataset_build, "__factory")[model_args.dataset_name]
        dataset = factory(root=model_args.root_dir, verbose=False)
        return int(len(dataset.train_id_container))
    except Exception as exc:
        print(f"Warning: could not infer train identity count from dataset; using {default}: {exc}")
        return int(default)


def build_test_loaders(model_args: SimpleNamespace) -> Tuple[Any, Any, int]:
    from datasets import build_dataloader

    loader_result = build_dataloader(model_args)
    if not isinstance(loader_result, tuple):
        raise RuntimeError("build_dataloader(args) did not return a tuple.")
    if len(loader_result) == 2:
        test_img_loader, test_txt_loader = loader_result
        num_classes = infer_num_classes(model_args)
        return test_img_loader, test_txt_loader, num_classes
    if len(loader_result) == 3:
        test_img_loader, test_txt_loader, num_classes = loader_result
        return test_img_loader, test_txt_loader, int(num_classes)
    raise RuntimeError(f"Unexpected build_dataloader(args) return length: {len(loader_result)}")


def torch_load_checkpoint(checkpoint_path: Path) -> Any:
    try:
        return torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(checkpoint_path), map_location="cpu")


def checkpoint_state_dict(checkpoint: Any) -> Tuple[Mapping[str, torch.Tensor], str]:
    if isinstance(checkpoint, Mapping):
        for key in ("state_dict", "model", "model_state_dict", "net", "network", "module"):
            value = checkpoint.get(key)
            if isinstance(value, Mapping):
                return value, key
        return checkpoint, "<root>"
    raise ValueError("Checkpoint must be a mapping or contain a model state mapping.")


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


def load_checkpoint_for_inference(model: torch.nn.Module, checkpoint_path: Path) -> Dict[str, Any]:
    checkpoint = torch_load_checkpoint(checkpoint_path)
    loaded_state, state_key = checkpoint_state_dict(checkpoint)
    model_state = model.state_dict()
    update_state: MutableMapping[str, torch.Tensor] = {}
    missing_examples: List[str] = []
    shape_examples: List[Dict[str, Any]] = []
    non_tensor_examples: List[str] = []
    skipped_missing = 0
    skipped_shape = 0
    skipped_non_tensor = 0

    for raw_key, value in loaded_state.items():
        if not torch.is_tensor(value):
            skipped_non_tensor += 1
            if len(non_tensor_examples) < 8:
                non_tensor_examples.append(str(raw_key))
            continue

        target_key = None
        for candidate in candidate_state_keys(str(raw_key)):
            if candidate in model_state:
                target_key = candidate
                break

        if target_key is None:
            skipped_missing += 1
            if len(missing_examples) < 8:
                missing_examples.append(str(raw_key))
            continue
        if tuple(model_state[target_key].shape) != tuple(value.shape):
            skipped_shape += 1
            if len(shape_examples) < 8:
                shape_examples.append(
                    {
                        "checkpoint_key": str(raw_key),
                        "model_key": target_key,
                        "checkpoint_shape": list(value.shape),
                        "model_shape": list(model_state[target_key].shape),
                    }
                )
            continue
        update_state[target_key] = value.detach().clone()

    if not update_state:
        raise RuntimeError(f"No compatible tensors found in checkpoint: {checkpoint_path}")

    model_state.update(update_state)
    model.load_state_dict(model_state, strict=True)
    return {
        "state_key": state_key,
        "checkpoint_tensors": int(sum(1 for value in loaded_state.values() if torch.is_tensor(value))),
        "loaded": len(update_state),
        "skipped_missing": skipped_missing,
        "skipped_shape": skipped_shape,
        "skipped_non_tensor": skipped_non_tensor,
        "missing_key_examples": missing_examples,
        "shape_mismatch_examples": shape_examples,
        "non_tensor_examples": non_tensor_examples,
    }


def feature_tensor(output: Any, kind: str) -> torch.Tensor:
    if isinstance(output, (tuple, list)):
        output = output[0]
    if not torch.is_tensor(output):
        raise RuntimeError(f"{kind} encoder returned {type(output)!r}, expected a tensor.")
    return output


def call_model_encoder(encoder: Any, tensor: torch.Tensor, kind: str) -> torch.Tensor:
    params = inspect.signature(encoder).parameters
    output = encoder(tensor, 0) if len(params) >= 2 else encoder(tensor)
    return feature_tensor(output, kind)


@torch.inference_mode()
def extract_text_features_from_encoder(
    model: torch.nn.Module,
    txt_loader: Any,
    device: torch.device,
    encoder_name: str,
    desc: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    encoder = getattr(model, encoder_name, None)
    if encoder is None:
        raise RuntimeError(f"Model does not provide {encoder_name}.")

    features: List[torch.Tensor] = []
    pids: List[torch.Tensor] = []
    model.eval()
    for pid, caption in tqdm(txt_loader, desc=desc):
        caption = caption.to(device, non_blocking=True)
        feats = call_model_encoder(encoder, caption, encoder_name).float()
        features.append(F.normalize(feats, p=2, dim=1).cpu())
        pids.append(pid.view(-1).cpu().long())

    if not features:
        raise RuntimeError(f"No text features were extracted from {encoder_name}.")
    return torch.cat(features, dim=0), torch.cat(pids, dim=0).long()


@torch.inference_mode()
def extract_image_features_from_encoder(
    model: torch.nn.Module,
    img_loader: Any,
    device: torch.device,
    encoder_name: str,
    desc: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    encoder = getattr(model, encoder_name, None)
    if encoder is None:
        raise RuntimeError(f"Model does not provide {encoder_name}.")

    features: List[torch.Tensor] = []
    pids: List[torch.Tensor] = []
    model.eval()
    for pid, image in tqdm(img_loader, desc=desc):
        image = image.to(device, non_blocking=True)
        feats = call_model_encoder(encoder, image, encoder_name).float()
        features.append(F.normalize(feats, p=2, dim=1).cpu())
        pids.append(pid.view(-1).cpu().long())

    if not features:
        raise RuntimeError(f"No image features were extracted from {encoder_name}.")
    return torch.cat(features, dim=0), torch.cat(pids, dim=0).long()


def ensure_same_tensor(left: torch.Tensor, right: torch.Tensor, label: str) -> None:
    if left.shape != right.shape or not torch.equal(left.cpu(), right.cpu()):
        raise RuntimeError(f"{label} ordering differs between extraction passes.")


def digest_sequence(items: Sequence[Any]) -> str:
    digest = hashlib.sha1()
    for item in items:
        digest.update(str(item).encode("utf-8", errors="replace"))
        digest.update(b"\0")
    return digest.hexdigest()


def stable_json_digest(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def sanitize_for_filename(value: str) -> str:
    allowed = []
    for char in str(value):
        if char.isalnum() or char in ("-", "_", "."):
            allowed.append(char)
        elif char.isspace():
            allowed.append("_")
    return "".join(allowed).strip("._") or "item"


def embedding_cache_metadata(
    label: str,
    checkpoint_path: Path,
    split_meta: SplitMeta,
    model_args: SimpleNamespace,
    retrieval_mode: str,
) -> Dict[str, Any]:
    stat = checkpoint_path.stat()
    return {
        "model_family": MODEL_FAMILY,
        "label": label,
        "checkpoint": str(checkpoint_path),
        "checkpoint_size": int(stat.st_size),
        "checkpoint_mtime_ns": int(stat.st_mtime_ns),
        "dataset_name": str(model_args.dataset_name),
        "root_dir": str(model_args.root_dir),
        "num_queries": len(split_meta.captions),
        "num_gallery": len(split_meta.img_paths),
        "caption_pid_digest": digest_sequence(split_meta.caption_pids),
        "image_pid_digest": digest_sequence(split_meta.image_pids),
        "caption_digest": digest_sequence(split_meta.captions),
        "image_path_digest": digest_sequence(split_meta.img_paths),
        "img_size": list(parse_img_size(model_args.img_size)),
        "text_length": int(model_args.text_length),
        "pretrain_choice": str(model_args.pretrain_choice),
        "retrieval_mode": retrieval_mode,
        "seed": int(getattr(model_args, "seed", 1)),
        "prototype": bool(getattr(model_args, "prototype", False)),
        "use_loss_id": bool(getattr(model_args, "use_loss_id", False)),
    }


def cache_path_for(output_dir: Path, label: str, checkpoint_path: Path, metadata: Mapping[str, Any]) -> Path:
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = stable_json_digest(metadata)[:16]
    safe_label = sanitize_for_filename(label)
    safe_stem = sanitize_for_filename(checkpoint_path.stem)
    return cache_dir / f"{safe_label}_{safe_stem}_{digest}.pt"


def try_load_embedding_cache(cache_path: Path, metadata: Mapping[str, Any]) -> Optional[EmbeddingBundle]:
    if not cache_path.is_file():
        return None
    payload = torch.load(str(cache_path), map_location="cpu")
    if payload.get("metadata") != dict(metadata):
        return None
    return EmbeddingBundle(
        text_features=payload["text_features"].float(),
        image_features=payload["image_features"].float(),
        query_pids=payload["query_pids"].long(),
        gallery_pids=payload["gallery_pids"].long(),
        metadata=dict(payload["metadata"]),
        load_stats={"cache": "hit", "cache_path": str(cache_path)},
    )


def save_embedding_cache(cache_path: Path, bundle: EmbeddingBundle) -> None:
    torch.save(
        {
            "metadata": bundle.metadata,
            "text_features": bundle.text_features.cpu(),
            "image_features": bundle.image_features.cpu(),
            "query_pids": bundle.query_pids.cpu(),
            "gallery_pids": bundle.gallery_pids.cpu(),
        },
        str(cache_path),
    )


def print_load_stats(label: str, stats: Mapping[str, Any]) -> None:
    print(
        f"[{label}] Loaded checkpoint tensors from {stats.get('state_key')}: "
        f"{stats.get('loaded')} loaded, {stats.get('skipped_missing')} missing, "
        f"{stats.get('skipped_shape')} shape-mismatch, {stats.get('skipped_non_tensor')} non-tensor skipped."
    )
    if stats.get("missing_key_examples"):
        print(f"[{label}] Unmatched checkpoint key examples: {stats['missing_key_examples']}")
    if stats.get("shape_mismatch_examples"):
        print(f"[{label}] Shape mismatch examples: {stats['shape_mismatch_examples']}")


def load_or_extract_embeddings(
    label: str,
    checkpoint_path: Path,
    model_args: SimpleNamespace,
    num_classes: int,
    test_img_loader: Any,
    test_txt_loader: Any,
    split_meta: SplitMeta,
    device: torch.device,
    retrieval_mode: str,
    output_dir: Path,
    use_cache: bool,
) -> EmbeddingBundle:
    from model import build_model

    metadata = embedding_cache_metadata(label, checkpoint_path, split_meta, model_args, retrieval_mode)
    cache_path = cache_path_for(output_dir, label, checkpoint_path, metadata) if use_cache else None
    if cache_path is not None:
        cached = try_load_embedding_cache(cache_path, metadata)
        if cached is not None:
            print(f"[{label}] Loaded cached embeddings: {cache_path}")
            return cached

    run_args = SimpleNamespace(**vars(model_args))
    model = build_model(run_args, num_classes=num_classes)
    load_stats = load_checkpoint_for_inference(model, checkpoint_path)
    print_load_stats(label, load_stats)
    model.to(device)
    if device.type == "cpu":
        model.float()
    model.eval()

    text_features, query_pids = extract_text_features_from_encoder(
        model, test_txt_loader, device, "encode_text", f"{label} text global"
    )
    image_features, gallery_pids = extract_image_features_from_encoder(
        model, test_img_loader, device, "encode_image", f"{label} image global"
    )

    bundle = EmbeddingBundle(
        text_features=text_features.cpu().float(),
        image_features=image_features.cpu().float(),
        query_pids=query_pids.cpu().long(),
        gallery_pids=gallery_pids.cpu().long(),
        metadata=metadata,
        load_stats=load_stats,
    )

    if cache_path is not None:
        save_embedding_cache(cache_path, bundle)
        print(f"[{label}] Saved embeddings cache: {cache_path}")

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return bundle


def top_list_values(
    order: torch.Tensor,
    scores: torch.Tensor,
    gallery_pids: torch.Tensor,
    img_paths: Sequence[str],
    k: int,
) -> Tuple[List[int], List[int], List[float], List[str]]:
    top_indices = [int(index) for index in order[:k].tolist()]
    top_pids = [int(gallery_pids[index].item()) for index in top_indices]
    top_scores = [float(scores[index].item()) for index in top_indices]
    top_paths = [str(img_paths[index]) for index in top_indices]
    return top_indices, top_pids, top_scores, top_paths


def green_vector_and_scores(matches: torch.Tensor, top_k: int) -> Tuple[List[int], int, float]:
    top_matches = matches[: min(top_k, matches.numel())]
    z_topk = [int(value) for value in top_matches.to(dtype=torch.int64).tolist()]
    green_count = int(sum(z_topk))
    discounted = 0.0
    for rank, is_green in enumerate(z_topk, start=1):
        if is_green:
            discounted += 1.0 / math.log2(rank + 1.0)
    return z_topk, green_count, float(discounted)


def showcase_score_from_values(
    best_r1_correct: bool,
    baseline_r1_correct: bool,
    best_green_count: int,
    baseline_green_count: int,
    best_first_correct_rank: int,
    rank_gain: int,
    early_green_gap: float,
) -> float:
    green_gap = best_green_count - baseline_green_count
    baseline_r1_wrong_bonus = 1000.0 if best_r1_correct and not baseline_r1_correct else 0.0
    best_r1_bonus = 100.0 if best_r1_correct else 0.0
    early_rank_bonus = 10.0 / max(float(best_first_correct_rank), 1.0)
    return float(
        best_r1_bonus
        + baseline_r1_wrong_bonus
        + 120.0 * green_gap
        + 35.0 * best_green_count
        - 25.0 * baseline_green_count
        + 30.0 * early_green_gap
        + 2.0 * rank_gain
        + early_rank_bonus
    )


def row_stats_for_model(
    scores: torch.Tensor,
    order: torch.Tensor,
    query_pid: int,
    gallery_pids: torch.Tensor,
    img_paths: Sequence[str],
    display_k: int,
    recall_k: int = 10,
) -> Dict[str, Any]:
    ranked_pids = gallery_pids[order.cpu()]
    matches = ranked_pids.eq(int(query_pid))
    positive_positions = matches.nonzero(as_tuple=False).view(-1)
    if positive_positions.numel() == 0:
        raise RuntimeError("row_stats_for_model called for a query with no positive gallery image.")

    first_correct_rank = int(positive_positions[0].item()) + 1
    top1_index = int(order[0].item())
    top_indices, top_pids, top_scores, top_paths = top_list_values(
        order.cpu(), scores.cpu(), gallery_pids.cpu(), img_paths, display_k
    )
    top10 = matches[: min(recall_k, matches.numel())]
    z_topk, green_count, discounted_green_score = green_vector_and_scores(matches, display_k)
    return {
        "top1_identity": int(gallery_pids[top1_index].item()),
        "top1_index": top1_index,
        "top1_score": float(scores[top1_index].item()),
        "r1_correct": first_correct_rank == 1,
        "first_correct_rank": first_correct_rank,
        "has_correct_top10": bool(top10.any().item()),
        "top10_positive_count": int(top10.sum().item()),
        "z_topK": z_topk,
        "green_count_at_k": green_count,
        "discounted_green_score_at_k": discounted_green_score,
        "top_indices": top_indices,
        "top_pids": top_pids,
        "top_scores": top_scores,
        "top_image_paths": top_paths,
    }


def compute_query_rows(
    baseline_sim: torch.Tensor,
    best_sim: torch.Tensor,
    captions: Sequence[str],
    query_pids: torch.Tensor,
    gallery_pids: torch.Tensor,
    img_paths: Sequence[str],
    display_k: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if baseline_sim.shape != best_sim.shape:
        raise ValueError("Baseline and prototype/IAPR similarity matrices must have the same shape.")
    if baseline_sim.shape[0] != len(captions) or baseline_sim.shape[0] != query_pids.numel():
        raise ValueError("Similarity row count does not match caption/query pid count.")
    if baseline_sim.shape[1] != gallery_pids.numel() or baseline_sim.shape[1] != len(img_paths):
        raise ValueError("Similarity column count does not match gallery pid/path count.")

    display_k = min(display_k, baseline_sim.shape[1])
    baseline_indices = torch.argsort(baseline_sim, dim=1, descending=True)
    best_indices = torch.argsort(best_sim, dim=1, descending=True)
    rows: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    for query_index in tqdm(range(baseline_sim.shape[0]), desc="Computing per-query rankings"):
        query_pid = int(query_pids[query_index].item())
        if not bool(gallery_pids.eq(query_pid).any().item()):
            skipped.append(
                {
                    "query_index": int(query_index),
                    "query_pid": query_pid,
                    "caption": str(captions[query_index]),
                    "reason": "query identity not found in gallery",
                }
            )
            continue

        baseline_stats = row_stats_for_model(
            baseline_sim[query_index], baseline_indices[query_index], query_pid, gallery_pids, img_paths, display_k
        )
        best_stats = row_stats_for_model(
            best_sim[query_index], best_indices[query_index], query_pid, gallery_pids, img_paths, display_k
        )

        baseline_rank = int(baseline_stats["first_correct_rank"])
        best_rank = int(best_stats["first_correct_rank"])
        rank_improvement = baseline_rank - best_rank
        rr_improvement = (1.0 / best_rank) - (1.0 / baseline_rank)
        baseline_green_count = int(baseline_stats["green_count_at_k"])
        best_green_count = int(best_stats["green_count_at_k"])
        green_gap = best_green_count - baseline_green_count
        baseline_discounted = float(baseline_stats["discounted_green_score_at_k"])
        best_discounted = float(best_stats["discounted_green_score_at_k"])
        early_green_gap = best_discounted - baseline_discounted
        showcase_score = showcase_score_from_values(
            best_r1_correct=bool(best_stats["r1_correct"]),
            baseline_r1_correct=bool(baseline_stats["r1_correct"]),
            best_green_count=best_green_count,
            baseline_green_count=baseline_green_count,
            best_first_correct_rank=best_rank,
            rank_gain=rank_improvement,
            early_green_gap=early_green_gap,
        )

        rows.append(
            {
                "query_index": int(query_index),
                "query_pid": query_pid,
                "caption": str(captions[query_index]),
                "baseline_top1_identity": baseline_stats["top1_identity"],
                "best_top1_identity": best_stats["top1_identity"],
                "baseline_top1_index": baseline_stats["top1_index"],
                "best_top1_index": best_stats["top1_index"],
                "baseline_top1_score": baseline_stats["top1_score"],
                "best_top1_score": best_stats["top1_score"],
                "baseline_r1_correct": bool(baseline_stats["r1_correct"]),
                "best_r1_correct": bool(best_stats["r1_correct"]),
                "baseline_z_topK": baseline_stats["z_topK"],
                "best_z_topK": best_stats["z_topK"],
                "baseline_green_count@K": baseline_green_count,
                "best_green_count@K": best_green_count,
                "green_gap": int(green_gap),
                "baseline_first_correct_rank": baseline_rank,
                "best_first_correct_rank": best_rank,
                "rank_gain": int(rank_improvement),
                "baseline_has_correct_top10": bool(baseline_stats["has_correct_top10"]),
                "best_has_correct_top10": bool(best_stats["has_correct_top10"]),
                "rank_improvement": int(rank_improvement),
                "reciprocal_rank_improvement": float(rr_improvement),
                "baseline_discounted_green_score@K": baseline_discounted,
                "best_discounted_green_score@K": best_discounted,
                "early_green_gap": float(early_green_gap),
                "showcase_score": showcase_score,
                "baseline_top10_positive_count": int(baseline_stats["top10_positive_count"]),
                "best_top10_positive_count": int(best_stats["top10_positive_count"]),
                "baseline_top_indices": baseline_stats["top_indices"],
                "best_top_indices": best_stats["top_indices"],
                "baseline_top_pids": baseline_stats["top_pids"],
                "best_top_pids": best_stats["top_pids"],
                "baseline_top_scores": baseline_stats["top_scores"],
                "best_top_scores": best_stats["top_scores"],
                "baseline_top_image_paths": baseline_stats["top_image_paths"],
                "best_top_image_paths": best_stats["top_image_paths"],
            }
        )

    return rows, skipped


def showcase_sort_key(row: Mapping[str, Any]) -> Tuple[float, int, int, int, int, float, int, int, int]:
    return (
        float(row["showcase_score"]),
        int(bool(row["best_r1_correct"] and not row["baseline_r1_correct"])),
        int(row["green_gap"]),
        int(row["best_green_count@K"]),
        -int(row["baseline_green_count@K"]),
        float(row["early_green_gap"]),
        int(row["rank_gain"]),
        -int(row["best_first_correct_rank"]),
        int(row["baseline_first_correct_rank"]),
    )


def sort_query_rows(
    rows: Sequence[Mapping[str, Any]],
    sort_mode: str,
    top_k: int,
    min_best_green: int,
    max_baseline_green: int,
) -> List[Mapping[str, Any]]:
    if sort_mode == "best_r1_green_gap":
        candidates = [
            row
            for row in rows
            if row["best_r1_correct"] and int(row["best_green_count@K"]) > int(row["baseline_green_count@K"])
        ]
        return sorted(candidates, key=showcase_sort_key, reverse=True)
    if sort_mode == "strict_showcase":
        candidates = [
            row
            for row in rows
            if row["best_r1_correct"]
            and not row["baseline_r1_correct"]
            and int(row["best_green_count@K"]) >= min_best_green
            and int(row["baseline_green_count@K"]) <= max_baseline_green
        ]
        return sorted(candidates, key=showcase_sort_key, reverse=True)
    if sort_mode == "best_r1_baseline_not_r1":
        candidates = [row for row in rows if row["best_r1_correct"] and not row["baseline_r1_correct"]]
        return sorted(
            candidates,
            key=lambda row: (
                row["rank_improvement"],
                row["reciprocal_rank_improvement"],
                row["baseline_first_correct_rank"],
            ),
            reverse=True,
        )
    if sort_mode == "best_r1_baseline_not_r10":
        candidates = [row for row in rows if row["best_r1_correct"] and not row["baseline_has_correct_top10"]]
        return sorted(candidates, key=lambda row: (row["rank_improvement"], row["reciprocal_rank_improvement"]), reverse=True)
    if sort_mode == "rank_improvement":
        return sorted(rows, key=lambda row: (row["rank_improvement"], row["reciprocal_rank_improvement"]), reverse=True)
    if sort_mode == "rr_improvement":
        return sorted(rows, key=lambda row: (row["reciprocal_rank_improvement"], row["rank_improvement"]), reverse=True)
    if sort_mode == "best_r1":
        candidates = [row for row in rows if row["best_r1_correct"]]
        return sorted(candidates, key=lambda row: row["baseline_first_correct_rank"], reverse=True)
    if sort_mode == "baseline_fail_best_success":
        candidates = [
            row
            for row in rows
            if row["baseline_first_correct_rank"] > top_k and row["best_first_correct_rank"] <= top_k
        ]
        return sorted(
            candidates,
            key=lambda row: (
                row["rank_improvement"],
                row["reciprocal_rank_improvement"],
                row["baseline_first_correct_rank"],
            ),
            reverse=True,
        )
    if sort_mode == "all":
        return sorted(rows, key=lambda row: (row["rank_improvement"], row["reciprocal_rank_improvement"]), reverse=True)
    raise ValueError(f"Unsupported sort mode: {sort_mode!r}")


def mean_or_none(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def recall_from_rows(rows: Sequence[Mapping[str, Any]], prefix: str, rank: int) -> float:
    if not rows:
        return 0.0
    hits = sum(1 for row in rows if int(row[f"{prefix}_first_correct_rank"]) <= rank)
    return float(hits * 100.0 / len(rows))


def build_summary(
    rows: Sequence[Mapping[str, Any]],
    skipped: Sequence[Mapping[str, Any]],
    total_queries: int,
    sort_mode: str,
    min_best_green: int,
    max_baseline_green: int,
    saved_figures: int,
    retrieval_mode: str,
    baseline_ckpt: Path,
    best_ckpt: Path,
    baseline_name: str,
    best_name: str,
    model_args: SimpleNamespace,
    baseline_load_stats: Mapping[str, Any],
    best_load_stats: Mapping[str, Any],
) -> Dict[str, Any]:
    rank_improvements = [float(row["rank_improvement"]) for row in rows]
    summary = {
        "model_family": MODEL_FAMILY,
        "dataset_name": str(model_args.dataset_name),
        "data_root": str(model_args.root_dir),
        "baseline_checkpoint": str(baseline_ckpt),
        "best_checkpoint": str(best_ckpt),
        "baseline_name": baseline_name,
        "best_name": best_name,
        "retrieval_mode": retrieval_mode,
        "global_weight": 1.0,
        "retrieval_formula": "s_global",
        "num_test_caption_queries": int(total_queries),
        "num_queries_used": int(len(rows)),
        "num_queries_skipped_no_gallery_positive": int(len(skipped)),
        "baseline_R@1": recall_from_rows(rows, "baseline", 1),
        "baseline_R@5": recall_from_rows(rows, "baseline", 5),
        "baseline_R@10": recall_from_rows(rows, "baseline", 10),
        "best_R@1": recall_from_rows(rows, "best", 1),
        "best_R@5": recall_from_rows(rows, "best", 5),
        "best_R@10": recall_from_rows(rows, "best", 10),
        "best_r1_correct_baseline_r1_wrong": int(
            sum(1 for row in rows if row["best_r1_correct"] and not row["baseline_r1_correct"])
        ),
        "baseline_r1_correct_best_r1_wrong": int(
            sum(1 for row in rows if row["baseline_r1_correct"] and not row["best_r1_correct"])
        ),
        "num_queries_improved": int(sum(1 for row in rows if row["rank_improvement"] > 0)),
        "num_queries_degraded": int(sum(1 for row in rows if row["rank_improvement"] < 0)),
        "mean_first_correct_rank_improvement": mean_or_none(rank_improvements),
        "selected_sort_mode": sort_mode,
        "min_best_green": int(min_best_green),
        "max_baseline_green": int(max_baseline_green),
        "showcase_score_definition": (
            "100*best_r1 + 1000*(best_r1 and not baseline_r1) + 120*green_gap "
            "+ 35*best_green_count@K - 25*baseline_green_count@K + 30*early_green_gap "
            "+ 2*rank_gain + 10/best_first_correct_rank"
        ),
        "discounted_green_score_definition": "sum_{rank=1..K} z[rank] / log2(rank + 1)",
        "number_of_saved_figures": int(saved_figures),
        "img_size": list(parse_img_size(model_args.img_size)),
        "text_length": int(model_args.text_length),
        "seed": int(getattr(model_args, "seed", 1)),
        "prototype": bool(getattr(model_args, "prototype", False)),
        "use_loss_id": bool(getattr(model_args, "use_loss_id", False)),
        "baseline_load_stats": dict(baseline_load_stats),
        "best_load_stats": dict(best_load_stats),
    }
    if skipped:
        summary["skipped_queries_preview"] = list(skipped[:20])
    return summary


def csv_value(value: Any) -> Any:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def save_rows_csv(rows: Sequence[Mapping[str, Any]], path: Path, field_order: Sequence[str]) -> None:
    all_fields = list(field_order)
    for row in rows:
        for key in row.keys():
            if key not in all_fields:
                all_fields.append(key)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key, "")) for key in all_fields})


def save_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)


def require_pillow() -> None:
    global Image, ImageDraw, ImageFont, ImageOps, RESAMPLE_LANCZOS
    if Image is not None:
        return
    try:
        from PIL import Image as pil_image
        from PIL import ImageDraw as pil_image_draw
        from PIL import ImageFont as pil_image_font
        from PIL import ImageOps as pil_image_ops
    except ModuleNotFoundError as exc:
        raise RuntimeError("Pillow is required to render qualitative figures. Install pillow and rerun.") from exc

    Image = pil_image
    ImageDraw = pil_image_draw
    ImageFont = pil_image_font
    ImageOps = pil_image_ops
    try:
        RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
    except AttributeError:
        RESAMPLE_LANCZOS = Image.LANCZOS


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    require_pillow()
    candidates = []
    if bold:
        candidates.extend(["arialbd.ttf", "Arial Bold.ttf"])
    candidates.extend(
        [
            "arial.ttf",
            "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def safe_text(text: Any) -> str:
    return str(text).encode("ascii", errors="replace").decode("ascii")


def text_width(draw: ImageDraw.ImageDraw, text: str, used_font: ImageFont.ImageFont) -> int:
    try:
        bbox = draw.textbbox((0, 0), text, font=used_font)
    except UnicodeEncodeError:
        bbox = draw.textbbox((0, 0), safe_text(text), font=used_font)
    return int(bbox[2] - bbox[0])


def draw_text(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[int, int],
    text: str,
    used_font: ImageFont.ImageFont,
    fill: Tuple[int, int, int] | str,
    anchor: Optional[str] = None,
) -> None:
    try:
        draw.text(xy, text, font=used_font, fill=fill, anchor=anchor)
    except UnicodeEncodeError:
        draw.text(xy, safe_text(text), font=used_font, fill=fill, anchor=anchor)


def wrap_text(draw: ImageDraw.ImageDraw, text: str, used_font: ImageFont.ImageFont, max_width: int) -> List[str]:
    words = str(text).split()
    if not words:
        return [""]
    lines: List[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if text_width(draw, candidate, used_font) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def load_thumbnail(path: str, size: Tuple[int, int]) -> Image.Image:
    require_pillow()
    width, height = size
    canvas = Image.new("RGB", size, (245, 247, 250))
    try:
        with Image.open(path) as image:
            image = image.convert("RGB")
            image = ImageOps.contain(image, size, method=RESAMPLE_LANCZOS)
            x = (width - image.width) // 2
            y = (height - image.height) // 2
            canvas.paste(image, (x, y))
    except Exception:
        draw = ImageDraw.Draw(canvas)
        small = font(13)
        draw_text(draw, (width // 2, height // 2 - 8), "image", small, (120, 126, 138), anchor="mm")
        draw_text(draw, (width // 2, height // 2 + 10), "missing", small, (120, 126, 138), anchor="mm")
    return canvas


def draw_retrieval_row(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    row_name: str,
    top_paths: Sequence[str],
    top_pids: Sequence[int],
    top_scores: Sequence[float],
    query_pid: int,
    y: int,
    left: int,
    grid_left: int,
    thumb_w: int,
    thumb_h: int,
    gap: int,
) -> None:
    label_font = font(20, bold=True)
    small_font = font(14)
    rank_font = font(16, bold=True)
    draw_text(draw, (left, y + 36), row_name, label_font, (17, 24, 39))

    for idx, path in enumerate(top_paths):
        x = grid_left + idx * (thumb_w + gap)
        rank_label = f"R{idx + 1}"
        draw_text(draw, (x + thumb_w // 2, y), rank_label, rank_font, (17, 24, 39), anchor="mt")
        image_y = y + 24
        thumb = load_thumbnail(path, (thumb_w, thumb_h))
        canvas.paste(thumb, (x, image_y))

        correct = int(top_pids[idx]) == int(query_pid)
        color = (22, 163, 74) if correct else (220, 38, 38)
        for offset in range(4):
            draw.rectangle(
                [x - offset, image_y - offset, x + thumb_w + offset, image_y + thumb_h + offset],
                outline=color,
            )
        draw_text(
            draw,
            (x + thumb_w // 2, image_y + thumb_h + 10),
            f"sim {float(top_scores[idx]):.4f}",
            small_font,
            (55, 65, 81),
            anchor="mt",
        )
        draw_text(
            draw,
            (x + thumb_w // 2, image_y + thumb_h + 30),
            f"pid {int(top_pids[idx])}",
            small_font,
            (55, 65, 81),
            anchor="mt",
        )


def render_query_figure(
    row: Mapping[str, Any],
    out_path: Path,
    baseline_name: str,
    best_name: str,
    sort_mode: str,
    dpi: int,
) -> None:
    display_k = len(row["baseline_top_image_paths"])
    thumb_w = 140
    thumb_h = 210
    gap = 14
    left = 28
    row_label_w = 150
    grid_left = left + row_label_w
    right = 28
    width = grid_left + display_k * thumb_w + max(display_k - 1, 0) * gap + right

    title_font = font(22, bold=True)
    caption_font = font(17)
    meta_font = font(16)
    temp = Image.new("RGB", (width, 200), "white")
    temp_draw = ImageDraw.Draw(temp)
    caption_lines = wrap_text(temp_draw, f"Caption: {row['caption']}", caption_font, width - 2 * left)[:4]
    top_h = 116 + len(caption_lines) * 24
    row_h = thumb_h + 70
    height = top_h + 2 * row_h + 36

    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    title = (
        f"Baseline rank {row['baseline_first_correct_rank']} | "
        f"Best rank {row['best_first_correct_rank']} | "
        f"gain {row['rank_gain']}"
    )
    score_line = (
        f"green {row['baseline_green_count@K']}->{row['best_green_count@K']} "
        f"(gap {row['green_gap']}) | score {float(row.get('showcase_score', 0.0)):.1f} | {sort_mode}"
    )
    draw_text(draw, (left, 18), title, title_font, (17, 24, 39))
    draw_text(draw, (left, 50), score_line, meta_font, (55, 65, 81))
    draw_text(draw, (left, 74), f"Query index {row['query_index']} | query pid {row['query_pid']}", meta_font, (75, 85, 99))

    y_text = 100
    for line in caption_lines:
        draw_text(draw, (left, y_text), line, caption_font, (31, 41, 55))
        y_text += 24

    baseline_y = top_h
    best_y = top_h + row_h
    draw.line((left, baseline_y - 14, width - right, baseline_y - 14), fill=(229, 231, 235), width=1)
    draw_retrieval_row(
        canvas,
        draw,
        baseline_name,
        row["baseline_top_image_paths"],
        row["baseline_top_pids"],
        row["baseline_top_scores"],
        int(row["query_pid"]),
        baseline_y,
        left,
        grid_left,
        thumb_w,
        thumb_h,
        gap,
    )
    draw_retrieval_row(
        canvas,
        draw,
        best_name,
        row["best_top_image_paths"],
        row["best_top_pids"],
        row["best_top_scores"],
        int(row["query_pid"]),
        best_y,
        left,
        grid_left,
        thumb_w,
        thumb_h,
        gap,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(str(out_path), "PNG", dpi=(dpi, dpi))


def prepare_output_dir(output_dir: Path, overwrite: bool, save_csv: bool, save_json_rows: bool) -> Path:
    generated: List[Path] = [output_dir / "summary.json"]
    if save_csv:
        generated.extend([output_dir / "ranking_results.csv", output_dir / "selected_results.csv"])
    if save_json_rows:
        generated.extend([output_dir / "ranking_results.json", output_dir / "selected_results.json"])

    figs_dir = output_dir / "figs"
    collisions = [path for path in generated if path.exists()]
    if figs_dir.is_dir():
        collisions.extend(figs_dir.glob("*.png"))
    if collisions and not overwrite:
        preview = "\n".join(f"  {path}" for path in collisions[:10])
        raise FileExistsError("Output files already exist. Pass --overwrite to replace generated outputs:\n" + preview)

    output_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in generated:
            if path.is_file():
                path.unlink()
        if figs_dir.is_dir():
            shutil.rmtree(figs_dir)
    figs_dir.mkdir(parents=True, exist_ok=True)
    return figs_dir


def validate_cli_args(args: argparse.Namespace) -> None:
    if args.num_figs < 0:
        raise ValueError("--num_figs must be non-negative.")
    if args.top_k <= 0:
        raise ValueError("--top_k must be positive.")
    if args.min_best_green is not None and args.min_best_green < 0:
        raise ValueError("--min_best_green must be non-negative.")
    if args.max_baseline_green is not None and args.max_baseline_green < 0:
        raise ValueError("--max_baseline_green must be non-negative.")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive.")


def resolve_showcase_thresholds(args: argparse.Namespace) -> Tuple[int, int]:
    default_min_best_green = 3 if args.top_k <= 5 else 5
    default_max_baseline_green = 2 if args.top_k <= 5 else 3
    min_best_green = default_min_best_green if args.min_best_green is None else int(args.min_best_green)
    max_baseline_green = default_max_baseline_green if args.max_baseline_green is None else int(args.max_baseline_green)
    return min_best_green, max_baseline_green


def main() -> None:
    args = parse_args()
    validate_cli_args(args)
    set_seed(args.seed)
    min_best_green, max_baseline_green = resolve_showcase_thresholds(args)

    baseline_ckpt = resolve_path(args.baseline_ckpt)
    best_ckpt = resolve_path(args.best_ckpt)
    output_dir = resolve_path(args.output_dir)
    if not baseline_ckpt.is_file():
        raise FileNotFoundError(f"Baseline checkpoint not found: {baseline_ckpt}")
    if not best_ckpt.is_file():
        raise FileNotFoundError(f"Prototype/IAPR checkpoint not found: {best_ckpt}")

    model_args = build_eval_args(args)
    retrieval_mode = resolve_retrieval_mode(args.retrieval_mode)
    figs_dir = prepare_output_dir(output_dir, args.overwrite, args.save_csv, args.save_json)
    device = resolve_device(device_arg_with_gpu(args.device, args.gpu_id))

    print(
        f"[Dataset] Loading {model_args.dataset_name} test split from {model_args.root_dir} "
        f"with img_size={parse_img_size(model_args.img_size)} text_length={model_args.text_length}"
    )
    test_img_loader, test_txt_loader, num_classes = build_test_loaders(model_args)
    split_meta = split_meta_from_loaders(test_img_loader, test_txt_loader)
    print(
        f"[Dataset] test queries={len(split_meta.captions)} gallery={len(split_meta.img_paths)} "
        f"train_ids={num_classes}"
    )
    print(f"[Retrieval] {MODEL_FAMILY} mode={retrieval_mode} formula=s_global")

    baseline_bundle = load_or_extract_embeddings(
        args.baseline_name,
        baseline_ckpt,
        model_args,
        num_classes,
        test_img_loader,
        test_txt_loader,
        split_meta,
        device,
        retrieval_mode,
        output_dir,
        args.cache_embeddings,
    )
    best_bundle = load_or_extract_embeddings(
        args.best_name,
        best_ckpt,
        model_args,
        num_classes,
        test_img_loader,
        test_txt_loader,
        split_meta,
        device,
        retrieval_mode,
        output_dir,
        args.cache_embeddings,
    )

    ensure_same_tensor(baseline_bundle.query_pids, best_bundle.query_pids, "Baseline/prototype query pid")
    ensure_same_tensor(baseline_bundle.gallery_pids, best_bundle.gallery_pids, "Baseline/prototype gallery pid")
    ensure_same_tensor(
        baseline_bundle.query_pids,
        torch.tensor(split_meta.caption_pids, dtype=torch.long),
        "Loader/query metadata pid",
    )
    ensure_same_tensor(
        baseline_bundle.gallery_pids,
        torch.tensor(split_meta.image_pids, dtype=torch.long),
        "Loader/gallery metadata pid",
    )

    print("[Similarity] Computing full text-to-image similarity matrices.")
    baseline_sim = baseline_bundle.text_features @ baseline_bundle.image_features.t()
    best_sim = best_bundle.text_features @ best_bundle.image_features.t()

    rows, skipped = compute_query_rows(
        baseline_sim,
        best_sim,
        split_meta.captions,
        baseline_bundle.query_pids,
        baseline_bundle.gallery_pids,
        split_meta.img_paths,
        args.top_k,
    )
    if skipped:
        print(f"Warning: skipped {len(skipped)} queries with no same-identity gallery image.")

    sorted_rows = sort_query_rows(rows, args.sort_mode, args.top_k, min_best_green, max_baseline_green)
    selected_rows = list(sorted_rows[: args.num_figs])
    print(
        f"[Selection] sort_mode={args.sort_mode} candidates={len(sorted_rows)} "
        f"saved_figures={len(selected_rows)} min_best_green={min_best_green} "
        f"max_baseline_green={max_baseline_green}"
    )

    ranking_csv = output_dir / "ranking_results.csv"
    selected_csv = output_dir / "selected_results.csv"
    if args.save_csv:
        save_rows_csv(rows, ranking_csv, CSV_FIELD_ORDER)
        save_rows_csv(selected_rows, selected_csv, CSV_FIELD_ORDER)

    if args.save_json:
        save_json(rows, output_dir / "ranking_results.json")
        save_json(selected_rows, output_dir / "selected_results.json")

    saved_figures = 0
    for selected_rank, row in enumerate(tqdm(selected_rows, desc="Saving figures"), start=1):
        filename = (
            f"{selected_rank:04d}_q{int(row['query_index']):06d}_"
            f"pid{int(row['query_pid'])}_improve{int(row['rank_improvement']):+d}.png"
        )
        render_query_figure(row, figs_dir / filename, args.baseline_name, args.best_name, args.sort_mode, args.dpi)
        saved_figures += 1

    summary = build_summary(
        rows,
        skipped,
        total_queries=len(split_meta.captions),
        sort_mode=args.sort_mode,
        min_best_green=min_best_green,
        max_baseline_green=max_baseline_green,
        saved_figures=saved_figures,
        retrieval_mode=retrieval_mode,
        baseline_ckpt=baseline_ckpt,
        best_ckpt=best_ckpt,
        baseline_name=args.baseline_name,
        best_name=args.best_name,
        model_args=model_args,
        baseline_load_stats=baseline_bundle.load_stats,
        best_load_stats=best_bundle.load_stats,
    )
    save_json(summary, output_dir / "summary.json")

    print("\nDone.")
    if args.save_csv:
        print(f"  all results CSV:      {ranking_csv}")
        print(f"  selected results CSV: {selected_csv}")
    print(f"  summary JSON:         {output_dir / 'summary.json'}")
    print(f"  figures:              {figs_dir}")


if __name__ == "__main__":
    main()
