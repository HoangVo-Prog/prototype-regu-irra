#!/usr/bin/env python3
"""Visualize host-fixed similarity shift decomposition for retrieval embeddings.

This script intentionally reuses the local loading/evaluation flow from
scripts/plot_ambiguity_rate_compare.py, then adds the shift decomposition
analysis:

    m_host = s_pos_host - s_neg_host
    delta_s_pos = s_pos_iapr - s_pos_host
    delta_s_neg = s_neg_iapr - s_neg_host
    delta_m = delta_s_pos - delta_s_neg

The positive image and identity-wrong hard negative image are selected once
with the host checkpoint and then scored under both checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import inspect
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

POINT_COLUMNS = [
    "query_index",
    "query_pid",
    "pos_gallery_index_host_fixed",
    "neg_gallery_index_host_fixed",
    "pos_gallery_pid",
    "neg_gallery_pid",
    "s_pos_host",
    "s_neg_host",
    "s_pos_iapr",
    "s_neg_iapr",
    "m_host",
    "delta_s_pos",
    "delta_s_neg",
    "delta_m",
    "scale_host",
    "scale_iapr",
    "margin_scale",
    "m_host_norm",
    "delta_s_pos_norm",
    "delta_s_neg_norm",
    "delta_m_norm",
    "selected_by_threshold",
    "margin_improved",
    "positive_attraction",
    "hard_negative_suppression",
    "attract_and_suppress",
]

BOOL_COLUMNS = {
    "selected_by_threshold",
    "margin_improved",
    "positive_attraction",
    "hard_negative_suppression",
    "attract_and_suppress",
}

STRING_COLUMNS = {
    "margin_scale",
}

INT_COLUMNS = {
    "query_index",
    "query_pid",
    "pos_gallery_index_host_fixed",
    "neg_gallery_index_host_fixed",
    "pos_gallery_pid",
    "neg_gallery_pid",
}


@dataclass
class FeatureBundle:
    text_features: torch.Tensor
    image_features: torch.Tensor
    query_pids: torch.Tensor
    gallery_pids: torch.Tensor
    inference: Dict[str, Any]
    text_grab_features: Optional[torch.Tensor] = None
    image_grab_features: Optional[torch.Tensor] = None


def load_ambiguity_flow() -> ModuleType:
    flow_path = Path(__file__).with_name("plot_ambiguity_rate_compare.py")
    if not flow_path.is_file():
        raise FileNotFoundError(f"Could not find local ambiguity script: {flow_path}")
    spec = importlib.util.spec_from_file_location("_local_plot_ambiguity_rate_compare", flow_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import local ambiguity script: {flow_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot host-fixed decomposed similarity shifts between a baseline and IAPR checkpoint."
    )
    parser.add_argument("--dataset_root", "--root_dir", dest="dataset_root", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--baseline_ckpt", "--baseline_checkpoint", dest="baseline_ckpt", required=True)
    parser.add_argument("--iapr_ckpt", "--ours_checkpoint", dest="iapr_ckpt", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--margin_scale", default="iqr", choices=["none", "iqr", "std"])
    parser.add_argument("--scale_eta", type=float, default=1e-12)
    parser.add_argument("--scale_sample_size", type=int, default=2_000_000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--fig_width", type=float, default=5.0)
    parser.add_argument("--fig_height", type=float, default=5.0)
    parser.add_argument("--alpha", type=float, default=0.35)
    parser.add_argument("--marker_size", type=float, default=10.0)
    parser.add_argument("--max_points", type=int, default=0)
    parser.add_argument("--plot_title", default="")
    parser.add_argument("--save_csv", action="store_true")
    parser.add_argument("--save_npz", action="store_true")
    parser.add_argument("--cache_dir", default="")
    parser.add_argument("--use_cache", action="store_true")
    parser.add_argument("--no_eval", "--skip_test_eval", dest="no_eval", action="store_true")
    parser.add_argument("--pair_selection", default="host_fixed", choices=["host_fixed"])
    parser.add_argument("--plot_format", default="png", choices=["png", "pdf"])
    parser.add_argument("--img_size", default="384,128")
    parser.add_argument("--text_length", type=int, default=77)
    parser.add_argument("--pretrain_choice", default="ViT-B/16")
    parser.add_argument("--model_type", default="itself", choices=["clip", "itself"])
    parser.add_argument("--baseline_model_type", default=None, choices=["clip", "itself"])
    parser.add_argument("--iapr_model_type", "--ours_model_type", dest="iapr_model_type", default=None, choices=["clip", "itself"])
    parser.add_argument("--num_experts", type=int, default=6)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--reduction", type=int, default=8)
    return parser.parse_args()


def resolve_path(path: str | Path, base: Path = REPO_ROOT) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = base / resolved
    return resolved.resolve()


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    if args.num_workers < 0:
        raise ValueError("--num_workers must be non-negative.")
    if args.text_length <= 0:
        raise ValueError("--text_length must be positive.")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive.")
    if args.fig_width <= 0 or args.fig_height <= 0:
        raise ValueError("--fig_width and --fig_height must be positive.")
    if not (0.0 < args.alpha <= 1.0):
        raise ValueError("--alpha must be in (0, 1].")
    if args.marker_size <= 0:
        raise ValueError("--marker_size must be positive.")
    if args.max_points < 0:
        raise ValueError("--max_points must be non-negative.")
    if not math.isfinite(args.threshold):
        raise ValueError("--threshold must be finite.")
    if not math.isfinite(args.scale_eta) or args.scale_eta < 0.0:
        raise ValueError("--scale_eta must be a finite non-negative value.")
    if args.scale_sample_size < 0:
        raise ValueError("--scale_sample_size must be non-negative.")


def build_model_args(flow: ModuleType, args: argparse.Namespace, model_type: Optional[str]) -> SimpleNamespace:
    params = inspect.signature(flow.build_model_args).parameters
    if "model_type" in params:
        return flow.build_model_args(args, model_type=model_type or args.model_type)
    return flow.build_model_args(args)


def metadata_from_standard_eval(meta: Mapping[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    if meta.get("retrieval_metrics"):
        result.update(dict(meta["retrieval_metrics"]))
    if meta.get("best_retrieval_metrics"):
        result["best_retrieval_metrics"] = dict(meta["best_retrieval_metrics"])
    if meta.get("selected_retrieval_metrics"):
        result["selected_retrieval_metrics"] = dict(meta["selected_retrieval_metrics"])
    return result


def ensure_same_tensor(left: torch.Tensor, right: torch.Tensor, label: str) -> None:
    if left.shape != right.shape or not torch.equal(left.cpu(), right.cpu()):
        raise RuntimeError(f"{label} order differs between baseline and IAPR feature extraction.")


def ensure_same_pid_order(flow: ModuleType, left: torch.Tensor, right: torch.Tensor, label: str) -> None:
    if hasattr(flow, "ensure_same_pids"):
        flow.ensure_same_pids(left, right, label)
    else:
        ensure_same_tensor(left, right, label)


def bundle_global_weight(bundle: FeatureBundle, override: Optional[float] = None) -> float:
    if override is not None:
        return float(override)
    return float(bundle.inference.get("global_weight", 1.0))


def similarity_chunk(
    bundle: FeatureBundle,
    start: int,
    end: int,
    global_weight: Optional[float] = None,
) -> torch.Tensor:
    sim = bundle.text_features[start:end].float() @ bundle.image_features.t().float()
    if bundle.text_grab_features is not None and bundle.image_grab_features is not None:
        weight = bundle_global_weight(bundle, global_weight)
        sim_grab = bundle.text_grab_features[start:end].float() @ bundle.image_grab_features.t().float()
        sim = weight * sim + (1.0 - weight) * sim_grab
    return sim.cpu()


def score_pairs(bundle: FeatureBundle, query_indices: Sequence[int], gallery_indices: Sequence[int]) -> np.ndarray:
    if not query_indices:
        return np.zeros((0,), dtype=np.float64)
    q = torch.as_tensor(query_indices, dtype=torch.long)
    g = torch.as_tensor(gallery_indices, dtype=torch.long)
    scores = (bundle.text_features[q].float() * bundle.image_features[g].float()).sum(dim=1)
    if bundle.text_grab_features is not None and bundle.image_grab_features is not None:
        weight = bundle_global_weight(bundle)
        grab_scores = (bundle.text_grab_features[q].float() * bundle.image_grab_features[g].float()).sum(dim=1)
        scores = weight * scores + (1.0 - weight) * grab_scores
    return scores.cpu().numpy().astype(np.float64)


def compute_bundle_scale_metadata(
    flow: ModuleType,
    bundle: FeatureBundle,
    margin_scale: str,
    scale_eta: float,
    scale_sample_size: int,
    seed: int,
    chunk_size: int = 512,
) -> Dict[str, Any]:
    if margin_scale == "none":
        return {
            "margin_scale": margin_scale,
            "scale_value": 1.0,
            "scale_eta": float(scale_eta),
            "scale_score_count_total": int(bundle.query_pids.numel() * bundle.gallery_pids.numel()),
            "scale_score_count_used": 0,
            "scale_sampling_seed": int(seed),
            "scale_is_exact": True,
        }
    if not all(hasattr(flow, name) for name in ("init_scale_sampling", "collect_scale_scores", "finalize_scale_metadata")):
        raise RuntimeError("The local ambiguity script does not expose score-scale helpers.")

    num_queries = int(bundle.query_pids.numel())
    num_gallery = int(bundle.gallery_pids.numel())
    state = flow.init_scale_sampling(margin_scale, num_queries, num_gallery, scale_sample_size, seed)
    for start in tqdm(range(0, num_queries, max(1, int(chunk_size))), desc="Estimating score scale"):
        end = min(start + max(1, int(chunk_size)), num_queries)
        sim = similarity_chunk(bundle, start, end)
        flow.collect_scale_scores(state, sim, start, num_gallery)
    return flow.finalize_scale_metadata(state, margin_scale, scale_eta, seed)


def retrieval_metrics_from_bundle(
    bundle: FeatureBundle,
    global_weight: Optional[float] = None,
    chunk_size: int = 256,
    desc: str = "Computing retrieval metrics",
) -> Dict[str, float]:
    gallery_pids = bundle.gallery_pids.cpu().long()
    query_pids = bundle.query_pids.cpu().long()
    valid_count = 0
    r1_hits = 0
    r5_hits = 0
    r10_hits = 0
    ap_sum = 0.0
    minp_sum = 0.0

    ranks = torch.arange(1, gallery_pids.numel() + 1, dtype=torch.float32)
    for start in tqdm(range(0, query_pids.numel(), chunk_size), desc=desc):
        end = min(start + chunk_size, query_pids.numel())
        sim = similarity_chunk(bundle, start, end, global_weight=global_weight)
        indices = torch.argsort(sim, dim=1, descending=True)
        pred_labels = gallery_pids[indices]
        matches = pred_labels.eq(query_pids[start:end].view(-1, 1))
        num_rel = matches.sum(dim=1)
        for row_idx in range(matches.shape[0]):
            rel = int(num_rel[row_idx].item())
            if rel <= 0:
                continue
            row = matches[row_idx]
            valid_count += 1
            r1_hits += int(bool(row[:1].any()))
            r5_hits += int(bool(row[:5].any()))
            r10_hits += int(bool(row[:10].any()))
            cumulative = row.cumsum(0).float()
            precision = cumulative / ranks
            ap_sum += float((precision * row.float()).sum().item() / rel)
            positive_positions = row.nonzero(as_tuple=False).view(-1)
            last_pos = int(positive_positions[-1].item())
            minp_sum += float(cumulative[last_pos].item() / (last_pos + 1.0))

    if valid_count == 0:
        raise RuntimeError("No query has a positive gallery match; cannot compute retrieval metrics.")
    r1 = r1_hits / valid_count * 100.0
    r5 = r5_hits / valid_count * 100.0
    r10 = r10_hits / valid_count * 100.0
    return {
        "R1": float(r1),
        "R5": float(r5),
        "R10": float(r10),
        "mAP": float(ap_sum / valid_count * 100.0),
        "mINP": float(minp_sum / valid_count * 100.0),
        "rSum": float(r1 + r5 + r10),
        "num_queries_used": int(valid_count),
    }


def select_itself_inference(flow: ModuleType, bundle: FeatureBundle) -> Dict[str, Any]:
    candidates = getattr(flow, "ITSELF_ABLATION_CANDIDATES", [("global", 1.0), ("grab", 0.0), ("global+grab(0.5)", 0.5)])
    best_name = ""
    best_weight = 1.0
    best_metrics: Dict[str, float] = {}
    best_r1 = float("-inf")

    for name, weight in candidates:
        metrics = retrieval_metrics_from_bundle(
            bundle,
            global_weight=float(weight),
            desc=f"Selecting ITSELF shift inference: {name}",
        )
        if metrics["R1"] >= best_r1:
            best_name = str(name)
            best_weight = float(weight)
            best_metrics = metrics
            best_r1 = metrics["R1"]

    if hasattr(flow, "itself_inference_metadata"):
        return flow.itself_inference_metadata(
            best_name,
            best_weight,
            selection_source="shift_decomposition_local_sweep_best_r1",
            selected_metrics=best_metrics,
        )
    return {
        "model_type": "itself",
        "inference_mode": "itself_best_ablation",
        "ablation_task": f"{best_name}-t2i",
        "global_weight": best_weight,
        "grab_weight": 1.0 - best_weight,
        "selection_source": "shift_decomposition_local_sweep_best_r1",
        "selected_metrics": best_metrics,
    }


def extract_feature_bundle(
    flow: ModuleType,
    model: torch.nn.Module,
    split_data: Any,
    model_args: SimpleNamespace,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> FeatureBundle:
    text_features, query_pids = flow.extract_text_features(
        model,
        split_data,
        text_length=int(model_args.text_length),
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )
    image_features, gallery_pids = flow.extract_image_features(
        model,
        split_data,
        img_size=flow.parse_img_size(model_args.img_size),
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )

    inference: Dict[str, Any] = {
        "model_type": str(getattr(model_args, "model_type", "standard")),
        "inference_mode": "standard_global",
        "global_weight": 1.0,
        "grab_weight": 0.0,
    }
    bundle = FeatureBundle(
        text_features=text_features.cpu(),
        image_features=image_features.cpu(),
        query_pids=query_pids.cpu().long(),
        gallery_pids=gallery_pids.cpu().long(),
        inference=inference,
    )

    model_type = getattr(model_args, "model_type", None)
    if model_type != "itself":
        if model_type == "clip" and hasattr(flow, "clip_inference_metadata"):
            bundle.inference = flow.clip_inference_metadata()
        return bundle

    if not hasattr(flow, "extract_text_features_from_encoder"):
        return bundle
    if not hasattr(model, "encode_text_grab") or not hasattr(model, "encode_image_grab"):
        raise RuntimeError("ITSELF inference requires encode_text_grab/encode_image_grab; use --model_type clip for CLIP-only checkpoints.")

    text_grab, grab_query_pids = flow.extract_text_features_from_encoder(
        model,
        split_data,
        text_length=int(model_args.text_length),
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        encoder_name="encode_text_grab",
        desc="Extracting GRAB text features",
    )
    image_grab, grab_gallery_pids = flow.extract_image_features_from_encoder(
        model,
        split_data,
        img_size=flow.parse_img_size(model_args.img_size),
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        encoder_name="encode_image_grab",
        desc="Extracting GRAB image features",
    )
    ensure_same_pid_order(flow, bundle.query_pids, grab_query_pids.cpu().long(), "Query")
    ensure_same_pid_order(flow, bundle.gallery_pids, grab_gallery_pids.cpu().long(), "Gallery")
    bundle.text_grab_features = text_grab.cpu()
    bundle.image_grab_features = image_grab.cpu()

    metadata = None
    if hasattr(flow, "selected_itself_inference_from_args"):
        metadata = flow.selected_itself_inference_from_args(model_args)
    if metadata is None:
        print("[Shift] Warning: no ITSELF selected_inference found; selecting best combo on this split.")
        metadata = select_itself_inference(flow, bundle)
    bundle.inference = dict(metadata)
    return bundle


def load_bundle_for_checkpoint(
    label: str,
    flow: ModuleType,
    checkpoint_path: Path,
    model_args: SimpleNamespace,
    split_data: Any,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Tuple[FeatureBundle, Dict[str, Any]]:
    model: Optional[torch.nn.Module] = None
    try:
        model, run_args, load_stats, checkpoint = flow.load_model_for_checkpoint(
            checkpoint_path,
            model_args,
            split_data,
            device,
        )
        if hasattr(flow, "print_load_stats"):
            flow.print_load_stats(label, {"load_stats": load_stats})
        else:
            print(f"[{label}] Loaded checkpoint tensors: {load_stats}")
        bundle = extract_feature_bundle(
            flow,
            model,
            split_data,
            run_args,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
        )
        return bundle, {"checkpoint": str(checkpoint), "load_stats": load_stats, "inference": bundle.inference}
    finally:
        model = None
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def compute_shift_rows(
    host: FeatureBundle,
    iapr: FeatureBundle,
    threshold: float,
    margin_scale: str,
    host_scale: float,
    iapr_scale: float,
    scale_eta: float,
    chunk_size: int = 512,
) -> Tuple[List[Dict[str, Any]], int]:
    ensure_same_tensor(host.query_pids, iapr.query_pids, "Query identity")
    ensure_same_tensor(host.gallery_pids, iapr.gallery_pids, "Gallery identity")
    gallery_pids = host.gallery_pids.cpu().long()
    query_pids = host.query_pids.cpu().long()
    host_denom = 1.0 if margin_scale == "none" else float(host_scale) + float(scale_eta)
    if host_denom == 0.0:
        raise ValueError("Host score-scale denominator is zero; increase --scale_eta.")

    base_rows: List[Dict[str, Any]] = []
    skipped = 0
    for start in tqdm(range(0, query_pids.numel(), chunk_size), desc="Selecting host-fixed pairs"):
        end = min(start + chunk_size, query_pids.numel())
        sim = similarity_chunk(host, start, end)
        for local_index in range(sim.shape[0]):
            query_index = start + local_index
            query_pid = int(query_pids[query_index].item())
            scores = sim[local_index]
            pos_mask = gallery_pids.eq(query_pid)
            neg_mask = ~pos_mask
            if not bool(pos_mask.any()):
                skipped += 1
                print(f"[Shift] Warning: query_index={query_index} pid={query_pid} has no positive gallery image; skipping.")
                continue
            if not bool(neg_mask.any()):
                skipped += 1
                print(f"[Shift] Warning: query_index={query_index} pid={query_pid} has no negative gallery image; skipping.")
                continue

            pos_scores = scores.masked_fill(~pos_mask, float("-inf"))
            neg_scores = scores.masked_fill(~neg_mask, float("-inf"))
            s_pos_host, pos_index = torch.max(pos_scores, dim=0)
            s_neg_host, neg_index = torch.max(neg_scores, dim=0)
            base_rows.append(
                {
                    "query_index": int(query_index),
                    "query_pid": query_pid,
                    "pos_gallery_index_host_fixed": int(pos_index.item()),
                    "neg_gallery_index_host_fixed": int(neg_index.item()),
                    "pos_gallery_pid": int(gallery_pids[pos_index].item()),
                    "neg_gallery_pid": int(gallery_pids[neg_index].item()),
                    "s_pos_host": float(s_pos_host.item()),
                    "s_neg_host": float(s_neg_host.item()),
                }
            )

    query_indices = [int(row["query_index"]) for row in base_rows]
    pos_indices = [int(row["pos_gallery_index_host_fixed"]) for row in base_rows]
    neg_indices = [int(row["neg_gallery_index_host_fixed"]) for row in base_rows]
    s_pos_iapr = score_pairs(iapr, query_indices, pos_indices)
    s_neg_iapr = score_pairs(iapr, query_indices, neg_indices)

    rows: List[Dict[str, Any]] = []
    for row, pos_score_iapr, neg_score_iapr in zip(base_rows, s_pos_iapr, s_neg_iapr):
        s_pos_host = float(row["s_pos_host"])
        s_neg_host = float(row["s_neg_host"])
        m_host = s_pos_host - s_neg_host
        delta_s_pos = float(pos_score_iapr) - s_pos_host
        delta_s_neg = float(neg_score_iapr) - s_neg_host
        delta_m = delta_s_pos - delta_s_neg
        m_host_norm = m_host / host_denom
        delta_s_pos_norm = delta_s_pos / host_denom
        delta_s_neg_norm = delta_s_neg / host_denom
        delta_m_norm = delta_m / host_denom
        selected = bool(m_host < threshold) if margin_scale == "none" else bool(m_host_norm < threshold)
        row.update(
            {
                "s_pos_iapr": float(pos_score_iapr),
                "s_neg_iapr": float(neg_score_iapr),
                "m_host": float(m_host),
                "delta_s_pos": float(delta_s_pos),
                "delta_s_neg": float(delta_s_neg),
                "delta_m": float(delta_m),
                "scale_host": float(host_scale),
                "scale_iapr": float(iapr_scale),
                "margin_scale": str(margin_scale),
                "m_host_norm": float(m_host_norm),
                "delta_s_pos_norm": float(delta_s_pos_norm),
                "delta_s_neg_norm": float(delta_s_neg_norm),
                "delta_m_norm": float(delta_m_norm),
                "selected_by_threshold": selected,
                "margin_improved": bool(delta_m > 0.0),
                "positive_attraction": bool(delta_s_pos > 0.0),
                "hard_negative_suppression": bool(delta_s_neg < 0.0),
                "attract_and_suppress": bool(delta_s_pos > 0.0 and delta_s_neg < 0.0),
            }
        )
        rows.append(row)
    return rows, skipped


def stats_for_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Optional[float]]:
    keys = [
        "pct_delta_m_positive",
        "pct_delta_s_pos_positive",
        "pct_delta_s_neg_negative",
        "pct_attract_and_suppress",
        "mean_delta_s_pos",
        "mean_delta_s_neg",
        "mean_delta_m",
        "median_delta_s_pos",
        "median_delta_s_neg",
        "median_delta_m",
        "mean_delta_s_pos_norm",
        "mean_delta_s_neg_norm",
        "mean_delta_m_norm",
        "median_delta_s_pos_norm",
        "median_delta_s_neg_norm",
        "median_delta_m_norm",
    ]
    if not rows:
        return {key: None for key in keys}
    delta_s_pos = np.array([float(row["delta_s_pos"]) for row in rows], dtype=np.float64)
    delta_s_neg = np.array([float(row["delta_s_neg"]) for row in rows], dtype=np.float64)
    delta_m = np.array([float(row["delta_m"]) for row in rows], dtype=np.float64)
    delta_s_pos_norm = np.array([float(row["delta_s_pos_norm"]) for row in rows], dtype=np.float64)
    delta_s_neg_norm = np.array([float(row["delta_s_neg_norm"]) for row in rows], dtype=np.float64)
    delta_m_norm = np.array([float(row["delta_m_norm"]) for row in rows], dtype=np.float64)
    return {
        "pct_delta_m_positive": float(np.mean(delta_m > 0.0) * 100.0),
        "pct_delta_s_pos_positive": float(np.mean(delta_s_pos > 0.0) * 100.0),
        "pct_delta_s_neg_negative": float(np.mean(delta_s_neg < 0.0) * 100.0),
        "pct_attract_and_suppress": float(np.mean((delta_s_pos > 0.0) & (delta_s_neg < 0.0)) * 100.0),
        "mean_delta_s_pos": float(np.mean(delta_s_pos)),
        "mean_delta_s_neg": float(np.mean(delta_s_neg)),
        "mean_delta_m": float(np.mean(delta_m)),
        "median_delta_s_pos": float(np.median(delta_s_pos)),
        "median_delta_s_neg": float(np.median(delta_s_neg)),
        "median_delta_m": float(np.median(delta_m)),
        "mean_delta_s_pos_norm": float(np.mean(delta_s_pos_norm)),
        "mean_delta_s_neg_norm": float(np.mean(delta_s_neg_norm)),
        "mean_delta_m_norm": float(np.mean(delta_m_norm)),
        "median_delta_s_pos_norm": float(np.median(delta_s_pos_norm)),
        "median_delta_s_neg_norm": float(np.median(delta_s_neg_norm)),
        "median_delta_m_norm": float(np.median(delta_m_norm)),
    }


def format_stat(value: Optional[float], suffix: str = "") -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}{suffix}"


def selected_plot_rows(rows: Sequence[Mapping[str, Any]], max_points: int, seed: int) -> List[Mapping[str, Any]]:
    selected = [row for row in rows if bool(row["selected_by_threshold"])]
    if max_points <= 0 or len(selected) <= max_points:
        return selected
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(selected), size=max_points, replace=False)
    return [selected[int(index)] for index in indices]


def plot_shift_decomposition(
    rows: Sequence[Mapping[str, Any]],
    selected_stats: Mapping[str, Optional[float]],
    args: argparse.Namespace,
    dataset_name: str,
    split: str,
    output_dir: Path,
) -> Dict[str, Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to plot shift decomposition.") from exc

    plot_rows = selected_plot_rows(rows, args.max_points, args.seed)
    normalized = args.margin_scale != "none"
    x_key = "delta_s_neg_norm" if normalized else "delta_s_neg"
    y_key = "delta_s_pos_norm" if normalized else "delta_s_pos"
    x = np.array([float(row[x_key]) for row in plot_rows], dtype=np.float64)
    y = np.array([float(row[y_key]) for row in plot_rows], dtype=np.float64)
    improved = np.array([bool(row["margin_improved"]) for row in plot_rows], dtype=bool)

    selected_count = int(sum(bool(row["selected_by_threshold"]) for row in rows))
    fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))
    if len(plot_rows):
        ax.scatter(
            x[~improved],
            y[~improved],
            s=args.marker_size,
            c="#9A665F",
            alpha=args.alpha,
            linewidths=0,
            label="Delta m <= 0",
        )
        ax.scatter(
            x[improved],
            y[improved],
            s=args.marker_size,
            c="#4C78A8",
            alpha=args.alpha,
            linewidths=0,
            label="Delta m > 0",
        )
        values = np.concatenate([x, y, np.array([0.0])])
    else:
        ax.text(0.5, 0.5, "No queries selected", transform=ax.transAxes, ha="center", va="center", fontsize=10)
        values = np.array([-0.05, 0.05, 0.0], dtype=np.float64)

    low = float(np.min(values))
    high = float(np.max(values))
    if math.isclose(low, high):
        low -= 0.05
        high += 0.05
    pad = max((high - low) * 0.08, 0.01)
    limits = (low - pad, high + pad)

    ax.plot(limits, limits, color="#333333", linestyle="--", linewidth=1.0, label="Delta s+ = Delta s-")
    ax.axvline(0.0, color="#777777", linestyle=":", linewidth=0.9)
    ax.axhline(0.0, color="#777777", linestyle=":", linewidth=0.9)
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_aspect("equal", adjustable="box")
    if normalized:
        ax.set_xlabel(r"$\Delta s^- / c_{\mathrm{Host}}$")
        ax.set_ylabel(r"$\Delta s^+ / c_{\mathrm{Host}}$")
    else:
        ax.set_xlabel(r"$\Delta s^-$")
        ax.set_ylabel(r"$\Delta s^+$")
    title = args.plot_title.strip() or f"{dataset_name} {split} shift decomposition"
    ax.set_title(title)
    ax.grid(True, alpha=0.22, linewidth=0.6)
    ax.legend(frameon=False, fontsize=8, loc="best")

    threshold_line = (r"$\rho$ = " if normalized else "threshold = ") + f"{args.threshold:g}"
    scale_label = {"iqr": "IQR_Host", "std": "STD_Host"}.get(args.margin_scale, "raw")
    box_lines = [
        f"N = {selected_count}",
        threshold_line,
        f"scale = {scale_label}" if normalized else "scale = raw",
        r"% $\Delta m > 0$ = " + format_stat(selected_stats["pct_delta_m_positive"], "%"),
        r"% $\Delta s^+ > 0$ = " + format_stat(selected_stats["pct_delta_s_pos_positive"], "%"),
        r"% $\Delta s^- < 0$ = " + format_stat(selected_stats["pct_delta_s_neg_negative"], "%"),
        "% both = " + format_stat(selected_stats["pct_attract_and_suppress"], "%"),
        (r"median $\Delta m/c_H$ = " + format_stat(selected_stats["median_delta_m_norm"]))
        if normalized
        else (r"median $\Delta m$ = " + format_stat(selected_stats["median_delta_m"])),
    ]
    ax.text(
        0.03,
        0.97,
        "\n".join(box_lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#CCCCCC", "alpha": 0.92},
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {"png": output_dir / "shift_decomposition.png"}
    fig.tight_layout()
    fig.savefig(paths["png"], dpi=args.dpi, bbox_inches="tight")
    if args.plot_format == "pdf":
        paths["pdf"] = output_dir / "shift_decomposition.pdf"
        fig.savefig(paths["pdf"], bbox_inches="tight")
    plt.close(fig)
    return paths


def save_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=POINT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_npz(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    arrays: Dict[str, np.ndarray] = {}
    for column in POINT_COLUMNS:
        values = [row[column] for row in rows]
        if column in BOOL_COLUMNS:
            arrays[column] = np.asarray(values, dtype=bool)
        elif column in STRING_COLUMNS:
            arrays[column] = np.asarray(values, dtype=str)
        elif column in INT_COLUMNS:
            arrays[column] = np.asarray(values, dtype=np.int64)
        else:
            arrays[column] = np.asarray(values, dtype=np.float64)
    np.savez_compressed(path, **arrays)


def json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def save_json(data: Mapping[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(json_ready(data), file, indent=2)
        file.write("\n")


def print_shift_summary(rows: Sequence[Mapping[str, Any]], selected_rows: Sequence[Mapping[str, Any]], stats: Mapping[str, Any]) -> None:
    print(
        f"[Shift] usable_queries={len(rows)} selected_queries={len(selected_rows)} "
        f"threshold={stats['threshold']} rule={stats['selection_rule']}"
    )
    if not selected_rows:
        print(f"[Shift] Warning: no query satisfies {stats['selection_rule']}; saved empty selected plot and JSON stats.")
        return
    selected = stats["selected"]
    print(
        "[Shift] Selected subset: "
        f"pct_delta_m_positive={format_stat(selected['pct_delta_m_positive'], '%')}, "
        f"pct_delta_s_pos_positive={format_stat(selected['pct_delta_s_pos_positive'], '%')}, "
        f"pct_delta_s_neg_negative={format_stat(selected['pct_delta_s_neg_negative'], '%')}, "
        f"pct_attract_and_suppress={format_stat(selected['pct_attract_and_suppress'], '%')}, "
        f"median_delta_m={format_stat(selected['median_delta_m'])}, "
        f"median_delta_m_norm={format_stat(selected['median_delta_m_norm'])}"
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    flow = load_ambiguity_flow()
    if args.use_cache:
        print("[Cache] --use_cache is accepted for CLI compatibility; recomputing shift data to preserve checkpoint verification.")

    dataset_root = resolve_path(args.dataset_root)
    baseline_ckpt = resolve_path(args.baseline_ckpt)
    iapr_ckpt = resolve_path(args.iapr_ckpt)
    output_dir = resolve_path(args.output_dir)
    if not baseline_ckpt.is_file():
        raise FileNotFoundError(f"Baseline checkpoint file not found: {baseline_ckpt}")
    if not iapr_ckpt.is_file():
        raise FileNotFoundError(f"IAPR checkpoint file not found: {iapr_ckpt}")

    device = flow.resolve_device(args.device) if hasattr(flow, "resolve_device") else torch.device(args.device)
    baseline_model_args = build_model_args(flow, args, args.baseline_model_type)
    iapr_model_args = build_model_args(flow, args, args.iapr_model_type)

    split_data = flow.load_split_data(args.dataset_name, dataset_root, args.split)
    test_split_data = split_data if args.split == "test" else flow.load_split_data(args.dataset_name, dataset_root, "test")
    print(
        f"[Dataset] {args.dataset_name} split={args.split} "
        f"queries={len(split_data.captions)} gallery={len(split_data.img_paths)}"
    )

    baseline_eval_meta: Dict[str, Any] = {}
    iapr_eval_meta: Dict[str, Any] = {}
    if not args.no_eval:
        print(f"[Baseline] Verifying {baseline_ckpt}")
        baseline_eval_meta = flow.evaluate_checkpoint_on_test_split(
            "Baseline",
            baseline_ckpt,
            baseline_model_args,
            test_split_data,
            device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        if baseline_eval_meta.get("selected_inference") is not None:
            setattr(baseline_model_args, "selected_inference", baseline_eval_meta["selected_inference"])

        print(f"[IAPR] Verifying {iapr_ckpt}")
        iapr_eval_meta = flow.evaluate_checkpoint_on_test_split(
            "IAPR",
            iapr_ckpt,
            iapr_model_args,
            test_split_data,
            device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        if iapr_eval_meta.get("selected_inference") is not None:
            setattr(iapr_model_args, "selected_inference", iapr_eval_meta["selected_inference"])

    print(f"[Baseline] Extracting shift features from {baseline_ckpt}")
    host_bundle, baseline_meta = load_bundle_for_checkpoint(
        "Baseline",
        flow,
        baseline_ckpt,
        baseline_model_args,
        split_data,
        device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"[IAPR] Extracting shift features from {iapr_ckpt}")
    iapr_bundle, iapr_meta = load_bundle_for_checkpoint(
        "IAPR",
        flow,
        iapr_ckpt,
        iapr_model_args,
        split_data,
        device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print("[Baseline] Estimating host score scale")
    host_scale_metadata = compute_bundle_scale_metadata(
        flow,
        host_bundle,
        args.margin_scale,
        args.scale_eta,
        args.scale_sample_size,
        args.seed,
        chunk_size=args.batch_size,
    )
    print("[IAPR] Estimating IAPR score scale")
    iapr_scale_metadata = compute_bundle_scale_metadata(
        flow,
        iapr_bundle,
        args.margin_scale,
        args.scale_eta,
        args.scale_sample_size,
        args.seed,
        chunk_size=args.batch_size,
    )

    rows, skipped = compute_shift_rows(
        host_bundle,
        iapr_bundle,
        threshold=args.threshold,
        margin_scale=args.margin_scale,
        host_scale=float(host_scale_metadata["scale_value"]),
        iapr_scale=float(iapr_scale_metadata["scale_value"]),
        scale_eta=args.scale_eta,
    )
    selected_rows = [row for row in rows if bool(row["selected_by_threshold"])]
    full_stats = stats_for_rows(rows)
    selected_stats = stats_for_rows(selected_rows)

    stats: Dict[str, Any] = {
        "dataset_name": args.dataset_name,
        "split": args.split,
        "baseline_checkpoint": str(baseline_ckpt),
        "iapr_checkpoint": str(iapr_ckpt),
        "num_queries_total": int(host_bundle.query_pids.numel()),
        "num_queries_usable": int(len(rows)),
        "num_queries_skipped": int(skipped),
        "num_queries_selected": int(len(selected_rows)),
        "margin_scale": args.margin_scale,
        "scale_eta": float(args.scale_eta),
        "scale_sample_size": int(args.scale_sample_size),
        "host_scale_metadata": host_scale_metadata,
        "iapr_scale_metadata": iapr_scale_metadata,
        "threshold": float(args.threshold),
        "threshold_type": "raw" if args.margin_scale == "none" else "normalized",
        "selection_rule": "m_host < threshold" if args.margin_scale == "none" else "m_host_norm < threshold",
        "pair_selection": "host_fixed",
        "full": full_stats,
        "selected": selected_stats,
        "baseline_load_stats": baseline_meta.get("load_stats", {}),
        "iapr_load_stats": iapr_meta.get("load_stats", {}),
        "baseline_inference": baseline_meta.get("inference", {}),
        "iapr_inference": iapr_meta.get("inference", {}),
    }
    if baseline_eval_meta:
        stats["baseline_retrieval"] = metadata_from_standard_eval(baseline_eval_meta)
    if iapr_eval_meta:
        stats["iapr_retrieval"] = metadata_from_standard_eval(iapr_eval_meta)

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = plot_shift_decomposition(rows, selected_stats, args, args.dataset_name, args.split, output_dir)
    stats_path = output_dir / "shift_decomposition_stats.json"
    save_json(stats, stats_path)

    output_paths: MutableMapping[str, Path] = {"stats": stats_path}
    output_paths.update({f"plot_{key}": path for key, path in plot_paths.items()})
    if args.save_csv:
        csv_path = output_dir / "shift_decomposition_points.csv"
        save_csv(rows, csv_path)
        output_paths["csv"] = csv_path
    if args.save_npz:
        npz_path = output_dir / "shift_decomposition_points.npz"
        save_npz(rows, npz_path)
        output_paths["npz"] = npz_path

    print_shift_summary(rows, selected_rows, stats)
    print("[Output] Saved shift decomposition files:")
    for label, path in output_paths.items():
        print(f"  {label}: {path}")


if __name__ == "__main__":
    main()
