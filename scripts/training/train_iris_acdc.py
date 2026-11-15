from __future__ import annotations

import argparse
import json
import logging
import math
import random
import shutil
import sys
import time
from functools import partial
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from iris.data import DatasetSplit, build_dataset 
from iris.model import IrisModel 
from iris.model.tuning import DiceCrossEntropyLoss  
from iris.training.lamb import Lamb  


DEFAULT_DATASETS = ("acdc", "msd_brain_tumor")
AVAILABLE_DATASETS = ("acdc", "msd_brain_tumor","msd_pancreas","msd_spleen")
CLASS_NAMES = {
    "acdc": {1: "cardiac_foreground_union"},
    "msd_brain_tumor": {1: "whole_tumor_union"},
}


MEMORY_TASK_IDS = {
    "acdc": 101,
    "msd_brain_tumor": 201,
}
MEMORY_BANK_NAMESPACE_VERSION = 1


def setup_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("iris_multidataset_clean")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(output_dir / "train.log", encoding="utf-8")
    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def save_json_atomic(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
    tmp.replace(path)


def export_memory_bank_state(model) -> Dict[int, torch.Tensor]:
    if getattr(model, "memory_bank", None) is None:
        raise RuntimeError("EMA memory bank is not enabled on this IrisModel.")
    return {
        int(task_id): embedding.detach().float().cpu().clone()
        for task_id, embedding in model.memory_bank.storage.items()
    }


def restore_memory_bank_state(model, state: Dict, device: torch.device) -> None:
    if getattr(model, "memory_bank", None) is None:
        raise RuntimeError("Cannot restore EMA memory: IrisModel memory bank is disabled.")
    if not isinstance(state, dict):
        raise TypeError("Checkpoint memory_bank_state must be a dictionary.")
    model.memory_bank.clear()
    model.memory_bank.device = None
    model.memory_bank.dtype = None
    expected_shape = (
        int(model.task_encoder.num_query_tokens) + 1,
        int(model.task_encoder.query_tokens.shape[-1]),
    )
    for task_id, embedding in state.items():
        if not torch.is_tensor(embedding) or embedding.ndim != 2:
            raise ValueError(
                f"Invalid memory entry for key {task_id}: expected a 2D tensor (tokens, channels)."
            )
        if tuple(embedding.shape) != expected_shape:
            raise ValueError(
                f"Memory entry {task_id} has shape {tuple(embedding.shape)}; "
                f"this model expects {expected_shape}. Do not resume across incompatible model shapes."
            )
        model.memory_bank.update(int(task_id), embedding.to(device=device, dtype=torch.float32))


def save_memory_bank_files(
    output_dir: Path,
    model,
    dataset_names: Sequence[str],
    update_counts: Dict[str, int],
    iteration: int,
    tag: str,
) -> None:
    
    bank_dir = output_dir / "memory_banks" / tag
    bank_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "namespace_version": MEMORY_BANK_NAMESPACE_VERSION,
        "iteration": int(iteration),
        "momentum": float(model.memory_bank.momentum),
        "entries": {},
        "update_counts": {name: int(update_counts.get(name, 0)) for name in dataset_names},
    }
    for dataset_name in dataset_names:
        task_id = MEMORY_TASK_IDS[dataset_name]
        embedding = model.memory_bank.storage.get(task_id)
        if embedding is None:
            manifest["entries"][dataset_name] = {"task_id": task_id, "initialized": False}
            continue
        payload = {
            "dataset_name": dataset_name,
            "task_id": task_id,
            "segmentation_label_id": 1,
            "task_name": CLASS_NAMES[dataset_name][1],
            "momentum": float(model.memory_bank.momentum),
            "update_count": int(update_counts.get(dataset_name, 0)),
            "last_saved_iteration": int(iteration),
            "embedding": embedding.detach().float().cpu().clone(),
        }
        file_path = bank_dir / f"{dataset_name}_ema_memory.pt"
        temp_path = file_path.with_suffix(file_path.suffix + ".tmp")
        torch.save(payload, temp_path)
        temp_path.replace(file_path)
        manifest["entries"][dataset_name] = {
            "task_id": task_id,
            "task_name": CLASS_NAMES[dataset_name][1],
            "file": file_path.name,
            "shape": list(payload["embedding"].shape),
            "initialized": True,
        }
    save_json_atomic(bank_dir / "manifest.json", manifest)


def restore_bank_from_checkpoint(model, checkpoint: Dict, dataset_names: Sequence[str], device: torch.device):
    if checkpoint.get("memory_bank_namespace_version") != MEMORY_BANK_NAMESPACE_VERSION:
        raise ValueError(
            "Checkpoint does not contain the compatible dataset-namespaced EMA bank. "
            "A non-memory-bank v1/v2 checkpoint cannot be resumed into this experiment."
        )
    saved_map = checkpoint.get("memory_task_ids", {})
    expected_map = {name: MEMORY_TASK_IDS[name] for name in dataset_names}
    if saved_map != expected_map:
        raise ValueError(f"Checkpoint memory task IDs {saved_map} do not match {expected_map}.")
    if "memory_bank_state" not in checkpoint:
        raise ValueError("Checkpoint is missing memory_bank_state; refusing to resume with an empty bank.")
    restore_memory_bank_state(model, checkpoint["memory_bank_state"], device)
    counts = checkpoint.get("memory_bank_update_counts", {})
    return {name: int(counts.get(name, 0)) for name in dataset_names}


def resolve_device(require_cuda: bool, logger: logging.Logger) -> torch.device:
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but PyTorch cannot access a CUDA GPU. "
            "Check your PyTorch/CUDA installation and `torch.cuda.is_available()`."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        logger.info(
            "GPU: %s | VRAM total %.2f GiB | free at start %.2f GiB",
            props.name,
            props.total_memory / (1024 ** 3),
            free_bytes / (1024 ** 3),
        )
        
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        if free_bytes < 2.5 * (1024 ** 3):
            logger.warning(
                "Less than 2.5 GiB of VRAM is free. Use --resolution 48 if the smoke test OOMs."
            )
    return device


def choose_resolution(requested: str, device: torch.device, smoke_test: bool) -> int:
    if smoke_test and requested == "auto":
        return 32
    if requested != "auto":
        return int(requested)
    if device.type != "cuda":
        return 48
    free_bytes, _ = torch.cuda.mem_get_info(device)
    return 64 if free_bytes >= 3.0 * (1024 ** 3) else 48


def _patient_group_id(dataset_name: str, subject_id: str) -> str:
    if dataset_name == "acdc":
        return subject_id.split("_frame", 1)[0]
    return subject_id


def _make_split_strategy(dataset_name: str, seed: int):
    def split_strategy(records):
        records = sorted(records, key=lambda record: record.subject_id)
        groups = {}
        for record in records:
            group_id = _patient_group_id(dataset_name, record.subject_id)
            groups.setdefault(group_id, []).append(record)
        group_ids = sorted(groups)
        if len(group_ids) < 3:
            raise ValueError(
                f"{dataset_name} needs at least 3 independent subjects/groups; found {len(group_ids)}."
            )
        rng = np.random.default_rng(seed)
        rng.shuffle(group_ids)
        n_groups = len(group_ids)
        n_train = max(1, int(round(0.70 * n_groups)))
        n_val = max(1, int(round(0.15 * n_groups)))
        if n_train + n_val >= n_groups:
            n_train = max(1, n_groups - 2)
            n_val = 1
        train_ids = set(group_ids[:n_train])
        val_ids = set(group_ids[n_train:n_train + n_val])
        test_ids = set(group_ids[n_train + n_val:])
        split_map = {
            DatasetSplit.TRAIN: [],
            DatasetSplit.VALID: [],
            DatasetSplit.TEST: [],
        }
        for record in records:
            gid = _patient_group_id(dataset_name, record.subject_id)
            if gid in train_ids:
                split_map[DatasetSplit.TRAIN].append(record)
            elif gid in val_ids:
                split_map[DatasetSplit.VALID].append(record)
            elif gid in test_ids:
                split_map[DatasetSplit.TEST].append(record)
        return split_map
    return split_strategy


def build_splits(dataset_root: Path, dataset_names: Sequence[str], resolution: int, seed: int, cache_data: bool, logger):
    result = {}
    for name in dataset_names:
        root = dataset_root / name
        if not root.exists():
            raise FileNotFoundError(
                f"Missing dataset directory: {root}\n"
                f"Expected a labelled dataset layout under {root}."
            )
        result[name] = {}
        for split in (DatasetSplit.TRAIN, DatasetSplit.VALID, DatasetSplit.TEST):
            ds = build_dataset(
                name,
                root=root,
                split=split,
                target_size=(resolution, resolution, resolution),
                random_seed=seed,
                split_strategy=_make_split_strategy(name, seed),
                cache_data=cache_data,
            )
            result[name][split.value] = ds
        tr = result[name][DatasetSplit.TRAIN.value]
        va = result[name][DatasetSplit.VALID.value]
        te = result[name][DatasetSplit.TEST.value]
        tr_ids = {_patient_group_id(name, r.subject_id) for r in tr.records}
        va_ids = {_patient_group_id(name, r.subject_id) for r in va.records}
        te_ids = {_patient_group_id(name, r.subject_id) for r in te.records}
        if (tr_ids & va_ids) or (tr_ids & te_ids) or (va_ids & te_ids):
            raise RuntimeError(f"Subject overlap detected in {name} split.")
        if min(len(tr), len(va), len(te)) < 2:
            raise RuntimeError(
                f"{name} needs at least 2 records in each train/validation/test split; "
                f"got train={len(tr)}, val={len(va)}, test={len(te)}."
            )
        logger.info(
            "%s split sizes: train=%d records/%d groups, val=%d records/%d groups, test=%d records/%d groups | group overlap=none",
            name, len(tr), len(tr_ids), len(va), len(va_ids), len(te), len(te_ids),
        )
    return result


def as_image(sample: Dict, device: torch.device) -> torch.Tensor:
    image = sample["image"].float()
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError(
            f"Expected image [1,D,H,W], got {tuple(image.shape)}. "
            "Check the dataset loader and preprocessing."
        )
    return image.unsqueeze(0).to(device, non_blocking=True)


def as_mask(sample: Dict) -> torch.Tensor:
    mask = sample.get("mask")
    if mask is None:
        raise ValueError("A segmentation mask is missing from a dataset sample.")
    mask = mask.to(torch.int64)
    if mask.ndim == 4 and mask.shape[0] == 1:
        mask = mask.squeeze(0)
    if mask.ndim != 3:
        raise ValueError(f"Expected mask [D,H,W], got {tuple(mask.shape)}.")
    return mask


def augment_episode_pair(support_items, query_image: torch.Tensor, query_mask: torch.Tensor, rng: random.Random):

    support_items = [(im.clone(), ma.clone()) for im, ma in support_items]
    query_image = query_image.clone()
    query_mask = query_mask.clone()

    for axis in range(3):
        if rng.random() < 0.5:
            for i, (im, ma) in enumerate(support_items):
                support_items[i] = (torch.flip(im, dims=[axis + 2]), torch.flip(ma, dims=[axis]))
            query_image = torch.flip(query_image, dims=[axis + 2])
            query_mask = torch.flip(query_mask, dims=[axis])

    augmented_supports = []
    for im, ma in support_items:
        scale = rng.uniform(0.90, 1.10)
        shift = rng.uniform(-0.10, 0.10)
        im = (im * scale + shift).clamp(-3.0, 3.0)
        augmented_supports.append((im, ma))

    scale = rng.uniform(0.90, 1.10)
    shift = rng.uniform(-0.10, 0.10)
    query_image = (query_image * scale + shift).clamp(-3.0, 3.0)
    return augmented_supports, query_image, query_mask


def get_classes(mask: torch.Tensor) -> List[int]:
    return sorted(int(v) for v in torch.unique(mask).tolist() if int(v) > 0)


def make_binary_class_masks(mask: torch.Tensor, class_ids: Sequence[int], device: torch.device):
    if not class_ids:
        raise ValueError("Cannot make task masks: support mask has no foreground classes.")
    tensor = torch.stack([(mask == c).float() for c in class_ids], dim=0)
    return tensor.unsqueeze(0).to(device, non_blocking=True)


def dice_per_class(logits: torch.Tensor, targets: torch.Tensor) -> List[float | None]:
    pred = torch.sigmoid(logits.float()) >= 0.5
    target = targets > 0.5
    dims = tuple(range(2, pred.ndim))
    intersection = (pred & target).sum(dim=dims).float()
    pred_sum = pred.sum(dim=dims).float()
    target_sum = target.sum(dim=dims).float()
    denominator = pred_sum + target_sum
    scores = (2.0 * intersection + 1e-6) / (denominator + 1e-6)
    out: List[float | None] = []
    for score, ps, ts in zip(scores.detach().cpu().reshape(-1), pred_sum.detach().cpu().reshape(-1), target_sum.detach().cpu().reshape(-1)):
        if ts.item() == 0 and ps.item() == 0:
            out.append(None)  
        elif ts.item() == 0:
            out.append(0.0)   
        else:
            out.append(float(score.item()))
    return out


def sample_episode(
    dataset,
    dataset_name: str,
    rng: random.Random,
    device: torch.device,
    training: bool,
    n_supports: int = 1,
    max_tries: int = 50,
):
    
    if n_supports < 1:
        raise ValueError("n_supports must be at least 1.")
    if len(dataset) < n_supports + 1:
        raise ValueError(
            f"{dataset_name} split has {len(dataset)} records; need at least "
            f"{n_supports + 1} records for {n_supports}-support evaluation."
        )

    for _ in range(max_tries):
        query_idx = rng.randrange(len(dataset))
        query_sample = dataset[query_idx]
        query_id = str(query_sample.get("meta", {}).get("subject_id", query_idx))
        query_group = _patient_group_id(dataset_name, query_id)
        query_mask_raw = (as_mask(query_sample) > 0).to(torch.int64)
        if not bool(query_mask_raw.any()):
            continue

        candidate_indices = [i for i in range(len(dataset)) if i != query_idx]
        rng.shuffle(candidate_indices)
        support_items_raw = []
        support_ids = []
        seen_groups = {query_group}
        for support_idx in candidate_indices:
            support_sample = dataset[support_idx]
            support_id = str(support_sample.get("meta", {}).get("subject_id", support_idx))
            support_group = _patient_group_id(dataset_name, support_id)
            if support_group in seen_groups:
                continue

            support_mask_raw = (as_mask(support_sample) > 0).to(torch.int64)
            if not bool(support_mask_raw.any()):
                continue

            support_image = as_image(support_sample, device)
            support_items_raw.append((support_image, support_mask_raw))
            support_ids.append(support_id)
            seen_groups.add(support_group)
            if len(support_items_raw) == n_supports:
                break

        if len(support_items_raw) != n_supports:
            continue

        query_image = as_image(query_sample, device)
        if training:
            support_items_raw, query_image, query_mask_raw = augment_episode_pair(
                support_items_raw, query_image, query_mask_raw, rng
            )
            query_image = (query_image + torch.randn_like(query_image) * 0.02).clamp(-3.0, 3.0)

        
        class_ids = [1]
        support_images = [im for im, _ in support_items_raw]
        support_masks = [make_binary_class_masks(ma, class_ids, device) for _, ma in support_items_raw]
        query_masks = make_binary_class_masks(query_mask_raw, class_ids, device)

        
        for support_image in support_images:
            if support_image.shape[-3:] != query_image.shape[-3:]:
                raise ValueError(f"{dataset_name}: support/query image spatial shapes differ.")

        return {
            "support_images": support_images,
            "support_masks_list": support_masks,
            "query_image": query_image,
            "query_masks": query_masks,
            "class_ids": class_ids,
            "support_ids": support_ids,
            "query_id": query_id,
            "n_supports": n_supports,
        }

    raise RuntimeError(
        f"Could not sample a valid {n_supports}-support episode from {dataset_name} "
        f"after {max_tries} attempts. Check masks, split size, and subject IDs."
    )


def run_episode(
    model, episode, criterion, device, optimizer=None, scaler=None,
    training=False, use_amp=True, memory_task_id: int | None = None,
):
    support_images = episode["support_images"]
    support_masks_list = episode["support_masks_list"]
    query_image = episode["query_image"]
    query_masks = episode["query_masks"]

    if len(support_images) != len(support_masks_list) or not support_images:
        raise ValueError("Episode must contain matching, non-empty support image/mask lists.")
    if training and len(support_images) != 1:
        raise ValueError("Training must remain one-support so the bank update has one unambiguous task context.")
    if training and memory_task_id is None:
        raise ValueError("memory_task_id is required for every training episode when EMA memory is enabled.")
    for support_image, support_masks in zip(support_images, support_masks_list):
        if support_image.shape[-3:] != query_image.shape[-3:]:
            raise ValueError("Support/query shapes differ after preprocessing.")
        if support_masks.shape[-3:] != support_image.shape[-3:]:
            raise ValueError("Support mask/image shapes differ.")
    if query_masks.shape[-3:] != query_image.shape[-3:]:
        raise ValueError("Query mask/image shapes differ.")

    amp = device.type == "cuda" and use_amp
    bank_updated = False
    optimizer_step_succeeded = True
    grad_context = torch.enable_grad() if training else torch.no_grad()
    with grad_context:
        support_embeddings = []
        for support_image, support_masks in zip(support_images, support_masks_list):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                support_out = model.encode_support(support_image, support_masks)
            support_embeddings.append(support_out["task_embeddings"])

        base_shape = support_embeddings[0].shape
        if any(emb.shape != base_shape for emb in support_embeddings[1:]):
            raise ValueError("Support task-embedding shapes differ; cannot average multi-support context.")
        task_embeddings = torch.stack(support_embeddings, dim=0).mean(dim=0)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            outputs = model(query_image, task_embeddings)
            logits = outputs["logits"]
        loss = criterion(logits.float(), query_masks.float())
        if not bool(torch.isfinite(loss.detach())):
            raise FloatingPointError("Non-finite segmentation loss; stopping before optimizer or EMA-bank update.")
        scores = dice_per_class(logits, query_masks)

        if training:
            if optimizer is None:
                raise ValueError("optimizer is required when training=True")
            if scaler is not None and amp:
                previous_scale = scaler.get_scale()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
               
                optimizer_step_succeeded = scaler.get_scale() >= previous_scale
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()


            if optimizer_step_succeeded:
                bank_embedding = support_embeddings[0].detach().float()
                if not bool(torch.isfinite(bank_embedding).all()):
                    raise FloatingPointError("Non-finite support task embedding; refusing to corrupt the EMA memory bank.")
                model.update_memory_bank(bank_embedding, [[int(memory_task_id)]])
                bank_updated = True

    return float(loss.detach().cpu()), scores, logits.detach(), bank_updated


def run_memory_bank_query(model, episode, criterion, device, use_amp: bool, memory_task_id: int):
    if model.memory_bank is None:
        raise RuntimeError("Memory-bank inference requested, but IrisModel has no memory bank.")
    try:
        task_embeddings = model.retrieve_memory_embeddings(
            [int(memory_task_id)], device=device, dtype=torch.float32
        ).unsqueeze(0)
    except KeyError as exc:
        raise RuntimeError(
            f"EMA memory entry {memory_task_id} is missing. This task must receive at least one TRAIN update before evaluation."
        ) from exc

    amp = device.type == "cuda" and use_amp
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            outputs = model(episode["query_image"], task_embeddings)
            logits = outputs["logits"]
        loss = criterion(logits.float(), episode["query_masks"].float())
        scores = dice_per_class(logits, episode["query_masks"])
    return float(loss.detach().cpu()), scores, logits.detach()

def evaluate_dataset(
    model, dataset, dataset_name: str, episodes: int, rng: random.Random,
    device, criterion, use_amp: bool, eval_shots: int = 3,
    memory_task_id: int | None = None,
    prediction_dir: Path | None = None,
):
    if memory_task_id is None:
        memory_task_id = MEMORY_TASK_IDS[dataset_name]
    model.eval()
    bank_dice: List[float] = []
    ensemble_dice: List[float] = []
    one_shot_dice: List[float] = []
    bank_losses: List[float] = []
    ensemble_losses: List[float] = []
    one_shot_losses: List[float] = []
    with torch.no_grad():
        for episode_idx in range(episodes):
            ep = sample_episode(
                dataset, dataset_name, rng, device, training=False, n_supports=eval_shots
            )

            loss_ensemble, scores_ensemble, logits_ensemble, _ = run_episode(
                model, ep, criterion, device, training=False, use_amp=use_amp
            )
            if eval_shots == 1:
                loss_one, scores_one, logits_one = loss_ensemble, scores_ensemble, logits_ensemble
            else:
                ep_one = dict(ep)
                ep_one["support_images"] = ep["support_images"][:1]
                ep_one["support_masks_list"] = ep["support_masks_list"][:1]
                ep_one["support_ids"] = ep["support_ids"][:1]
                ep_one["n_supports"] = 1
                loss_one, scores_one, logits_one, _ = run_episode(
                    model, ep_one, criterion, device, training=False, use_amp=use_amp
                )

            loss_bank, scores_bank, logits_bank = run_memory_bank_query(
                model, ep, criterion, device, use_amp, int(memory_task_id)
            )

            usable_bank = [v for v in scores_bank if v is not None]
            usable_ensemble = [v for v in scores_ensemble if v is not None]
            usable_one = [v for v in scores_one if v is not None]
            if usable_bank:
                bank_dice.append(float(np.mean(usable_bank)))
            if usable_ensemble:
                ensemble_dice.append(float(np.mean(usable_ensemble)))
            if usable_one:
                one_shot_dice.append(float(np.mean(usable_one)))
            bank_losses.append(float(loss_bank))
            ensemble_losses.append(float(loss_ensemble))
            one_shot_losses.append(float(loss_one))

            if prediction_dir is not None and episode_idx == 0:
                prediction_dir.mkdir(parents=True, exist_ok=True)
                prediction_ensemble = (torch.sigmoid(logits_ensemble[0].float()) >= 0.5).detach().cpu().numpy().astype(np.uint8)
                prediction_one = (torch.sigmoid(logits_one[0].float()) >= 0.5).detach().cpu().numpy().astype(np.uint8)
                prediction_bank = (torch.sigmoid(logits_bank[0].float()) >= 0.5).detach().cpu().numpy().astype(np.uint8)
                target = ep["query_masks"][0].detach().cpu().numpy().astype(np.uint8)
                query_image = ep["query_image"][0, 0].detach().cpu().numpy().astype(np.float32)
                np.savez_compressed(
                    prediction_dir / "episode_000.npz",
                   
                    prediction=prediction_bank,
                    prediction_memory_bank=prediction_bank,
                    prediction_1shot=prediction_one,
                    prediction_ensemble=prediction_ensemble,
                    target=target,
                    query_image=query_image,
                    class_ids=np.asarray(ep["class_ids"], dtype=np.int16),
                    support_subject=np.asarray(",".join(ep["support_ids"])),
                    support_subjects=np.asarray(ep["support_ids"]),
                    query_subject=np.asarray(ep["query_id"]),
                    eval_shots=np.asarray(eval_shots, dtype=np.int16),
                    memory_task_id=np.asarray(int(memory_task_id), dtype=np.int32),
                    memory_bank_was_used=np.asarray(True),
                )

    return {
        "dataset": dataset_name,
        "episodes_requested": episodes,
        "episodes_evaluated": len(bank_dice),
        "dice_mean": float(np.mean(ensemble_dice)) if ensemble_dice else 0.0,
        "dice_std": float(np.std(ensemble_dice)) if ensemble_dice else 0.0,
        "dice_mean_ensemble": float(np.mean(ensemble_dice)) if ensemble_dice else 0.0,
        "dice_std_ensemble": float(np.std(ensemble_dice)) if ensemble_dice else 0.0,
        "dice_mean_1shot": float(np.mean(one_shot_dice)) if one_shot_dice else 0.0,
        "dice_std_1shot": float(np.std(one_shot_dice)) if one_shot_dice else 0.0,
        "dice_mean_memory_bank": float(np.mean(bank_dice)) if bank_dice else 0.0,
        "dice_std_memory_bank": float(np.std(bank_dice)) if bank_dice else 0.0,
        "loss_mean": float(np.mean(ensemble_losses)) if ensemble_losses else 0.0,
        "loss_mean_1shot": float(np.mean(one_shot_losses)) if one_shot_losses else 0.0,
        "loss_mean_memory_bank": float(np.mean(bank_losses)) if bank_losses else 0.0,
        "ensemble_delta_dice": (
            float(np.mean(ensemble_dice) - np.mean(one_shot_dice))
            if ensemble_dice and one_shot_dice else 0.0
        ),
        "memory_bank_delta_vs_1shot": (
            float(np.mean(bank_dice) - np.mean(one_shot_dice))
            if bank_dice and one_shot_dice else 0.0
        ),
        "memory_bank_delta_vs_ensemble": (
            float(np.mean(bank_dice) - np.mean(ensemble_dice))
            if bank_dice and ensemble_dice else 0.0
        ),
        "memory_task_id": int(memory_task_id),
        "evaluation_protocol": (
            "binary foreground; training updates a dataset-namespaced EMA bank using TRAIN support embeddings only; "
            f"validation/test compare support-free EMA retrieval, one-support, and {eval_shots}-support embedding averaging "
            "on the same query episodes; validation/test never update the bank; support and query subjects are distinct"
        ),
    }


def save_checkpoint(
    path: Path, model, optimizer, scaler, scheduler, iteration: int,
    config: Dict, best_val: float, history: Dict, train_rng: random.Random,
    memory_update_counts: Dict[str, int], dataset_names: Sequence[str],
):
    path.parent.mkdir(parents=True, exist_ok=True)
    bank_state = export_memory_bank_state(model)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": int(iteration),
        "best_validation_macro_dice": float(best_val),
        "config": config,
        "history": history,
        "train_rng_state": train_rng.getstate(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "memory_bank_namespace_version": MEMORY_BANK_NAMESPACE_VERSION,
        "memory_task_ids": {name: MEMORY_TASK_IDS[name] for name in dataset_names},
        "memory_bank_state": bank_state,
        "memory_bank_update_counts": {name: int(memory_update_counts.get(name, 0)) for name in dataset_names},
        "memory_bank_momentum": float(model.memory_bank.momentum),
        "resume_note": "Restores model, optimizer, scaler, RNG, namespaced EMA entries and update counts. Scheduler uses a fresh cosine tail over remaining requested iterations.",
    }
    if torch.cuda.is_available():
        payload["torch_cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()

    temp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp_path)
    temp_path.replace(path)

    
    save_memory_bank_files(
        path.parent.parent, model, dataset_names, memory_update_counts,
        iteration=int(iteration), tag=path.stem,
    )

def load_resume_history(checkpoint: Dict, checkpoint_path: Path, start_iteration: int) -> Dict:
    history = checkpoint.get("history")
    if history is None:
        history_path = checkpoint_path.resolve().parent.parent / "metrics" / "history.json"
        if history_path.exists():
            try:
                history = json.loads(history_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                history = None
    if not isinstance(history, dict):
        history = {"train": [], "validation": []}
    history.setdefault("train", [])
    history.setdefault("validation", [])
    history["train"] = [r for r in history["train"] if int(r.get("iteration", 0)) <= start_iteration]
    history["validation"] = [r for r in history["validation"] if int(r.get("iteration", 0)) <= start_iteration]
    return history


def write_experiment_report(output_dir: Path, config: Dict, result: Dict) -> None:
    rows = []
    for name, metrics in result["test_results_by_dataset"].items():
        rows.append(
            f"| {name} | {metrics['dice_mean_memory_bank']:.4f} | {metrics['dice_std_memory_bank']:.4f} | "
            f"{metrics['dice_mean_1shot']:.4f} | {metrics['dice_mean_ensemble']:.4f} | "
            f"{metrics['memory_bank_delta_vs_1shot']:+.5f} | {metrics['episodes_evaluated']} |"
        )
    report = [
        "# IRIS-inspired binary episodic segmentation with dataset-namespaced EMA memory",
        "",
        f"- Resolution: {config['resolution'][0]}³",
        f"- Training iterations in this run: {config['iterations']}",
        f"- EMA momentum: {config['memory_bank_momentum']}",
        f"- Training supports per episode: {config['training_support_shots']}",
        f"- Context ensemble supports at validation/test: {config['evaluation_support_shots']}",
        f"- Best validation checkpoint: iteration {result['best_validation_iteration']}",
        f"- Best validation macro Dice (EMA bank): {result['best_validation_macro_dice']:.4f}",
        "",
        "## Test results",
        "",
        "| Dataset | EMA bank Dice | Std. dev. | One-support Dice | Context-ensemble Dice | Bank − one-support | Episodes |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *rows,
        "",
        f"**Equal-weight macro test Dice (EMA bank): {result['test_macro_dice_memory_bank_across_datasets']:.4f}**",
        f"**Equal-weight macro test Dice (one support): {result['test_macro_dice_1shot_across_datasets']:.4f}**",
        f"**Equal-weight macro test Dice (context ensemble): {result['test_macro_dice_ensemble_across_datasets']:.4f}**",
        "",
        "## Memory-bank protocol",
        "",
        "The model uses the repository's ClassMemoryBank with EMA updates. The binary foreground tasks have dataset-specific memory IDs: ACDC=101 and MSD Brain Tumour=201. These IDs are deliberately distinct even though both segmentation masks use local foreground label 1.",
        "",
        "Each training episode encodes one labelled support volume. After the optimizer update, its detached support task embedding updates only that dataset's EMA entry. Validation and test never update the bank. Bank inference retrieves the stored task embedding and predicts the query without encoding any query-specific support pair.",
        "",
        "Separate per-dataset files are written under `memory_banks/`; each checkpoint also stores the complete bank snapshot and update counters so model weights and bank state can be restored together.",
        "",
        "## Metric definition and limitations",
        "",
        result["macro_dice_definition"],
        "",
        result["protocol_note"],
        "",
        "The memory-bank result measures retrieval for tasks represented during training. It is not an unseen-task or unseen-dataset generalization result. The ACDC binary target merges RV, myocardium and LV; the Brain Tumour binary target merges all nonzero tumour labels.",
        "",
    ]
    (output_dir / "EXPERIMENT_REPORT.md").write_text("\n".join(report), encoding="utf-8")

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=AVAILABLE_DATASETS, default=list(DEFAULT_DATASETS),
                        help="Datasets to train jointly. Default: acdc msd_brain_tumor.")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--resume-from", type=Path, default=None,
                        help="Resume from a compatible checkpoint (e.g. checkpoints/latest.pt or best_validation.pt). --iterations is the target total iteration, not additional steps.")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--val-episodes", type=int, default=20)
    parser.add_argument("--test-episodes", type=int, default=30)
    parser.add_argument("--eval-shots", type=int, default=3, help="Number of distinct support subjects used for context-ensemble validation/test; training remains one-shot.")
    parser.add_argument("--memory-momentum", type=float, default=0.999,
                        help="EMA momentum for the dataset-specific task memory bank (paper-style default: 0.999).")
    parser.add_argument("--resolution", choices=["auto", "32", "48", "64", "80", "96", "128"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-cache", action="store_true", help="Disable per-dataset CPU caching to reduce RAM use.")
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA mixed precision (uses more VRAM).")
    parser.add_argument("--smoke-test", action="store_true", help="Run a 2-iteration wiring check; uses 32^3 unless --resolution is specified.")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("ERROR: This runner is configured for CUDA use, but torch.cuda.is_available() is False.")
        print("Check PyTorch/CUDA in the active environment; the script will not silently fall back to CPU.")
        return 2

    dataset_names = tuple(args.datasets)
    if len(set(dataset_names)) != len(dataset_names):
        print("ERROR: --datasets must not contain duplicate dataset names.")
        return 2
    resolution = choose_resolution(args.resolution, device, args.smoke_test)
    iterations = 2 if args.smoke_test else max(1, args.iterations)
    eval_every = 1 if args.smoke_test else max(1, args.eval_every)
    val_episodes = 1 if args.smoke_test else max(1, args.val_episodes)
    test_episodes = 1 if args.smoke_test else max(1, args.test_episodes)
    use_amp = not args.no_amp
    if args.eval_shots < 1:
        print("ERROR: --eval-shots must be at least 1.")
        return 2
    if not (0.0 <= args.memory_momentum < 1.0):
        print("ERROR: --memory-momentum must be in [0, 1).")
        return 2

    dataset_tag = "_".join(dataset_names)
    default_name = (
        f"iris_memory_bank_{dataset_tag}_smoke_{resolution}"
        if args.smoke_test
        else f"iris_memory_bank_{dataset_tag}_{resolution}"
    )
    output_dir = (args.output_dir or (ROOT_DIR / "outputs" / default_name)).resolve()
    logger = setup_logger(output_dir)

    logger.info("IRIS-inspired binary ACDC + MSD Brain Tumour experiment with dataset-namespaced EMA memory bank")
    logger.info("Datasets: %s", ", ".join(dataset_names))
    logger.info("Resolution: %d^3 | iterations=%d | AMP=%s | train_shots=1 | eval_shots=%d", resolution, iterations, use_amp, args.eval_shots)
    logger.info("Output directory: %s", output_dir)
    props = torch.cuda.get_device_properties(device)
    free_bytes, _ = torch.cuda.mem_get_info(device)
    logger.info("GPU: %s | total VRAM %.2f GiB | initial free %.2f GiB", props.name, props.total_memory / (1024**3), free_bytes / (1024**3))

    config = {
        "datasets": list(dataset_names),
        "resolution": [resolution, resolution, resolution],
        "base_channels": 16,
        "num_query_tokens": 8,
        "optimizer": "LAMB",
        "learning_rate": 1e-4,
        "iterations": iterations,
        "validation_every": eval_every,
        "validation_episodes_per_dataset": val_episodes,
        "test_episodes_per_dataset": test_episodes,
        "training_support_shots": 1,
        "evaluation_support_shots": args.eval_shots,
        "mask_protocol": "binary: all labels > 0 converted to foreground 1",
        "validation_protocol": "deterministic episodes regenerated from fixed seed at every checkpoint",
        "augmentation_protocol": "same spatial flips applied across all support(s) and query per episode; intensity scale/shift independently sampled",
        "split": {"train": 0.70, "validation": 0.15, "test": 0.15, "seed": args.seed},
        "amp": use_amp,
        "memory_bank": True,
        "memory_bank_type": "ClassMemoryBank EMA",
        "memory_bank_momentum": float(args.memory_momentum),
        "memory_bank_namespace_version": MEMORY_BANK_NAMESPACE_VERSION,
        "memory_task_ids": {name: MEMORY_TASK_IDS[name] for name in dataset_names},
        "memory_bank_update_source": "detached task embeddings from labelled TRAIN supports only; no validation/test updates",
        "memory_bank_inference": "retrieve stored dataset-specific EMA embedding without query-specific support encoding",
        "contrastive_loss": False,
        "protocol": "binary episodic training; one support during training; dataset-namespaced EMA memory retrieval plus one-support/context-ensemble baselines at validation/test; patient-grouped split; no unseen-dataset claim",
        "seed": args.seed,
        "resume_from": str(args.resume_from.resolve()) if args.resume_from else None,
        "resume_schedule_policy": "fresh cosine schedule over remaining target iterations, initialized at checkpoint optimizer LR",
    }
    save_json_atomic(output_dir / "run_config.json", config)

    try:
        dataset_root = args.dataset_root
        if not dataset_root.is_absolute():
            dataset_root = ROOT_DIR / dataset_root
        split_data = build_splits(dataset_root.resolve(), dataset_names, resolution, args.seed, not args.no_cache, logger)
        train_sets = [split_data[name]["train"] for name in dataset_names]
        val_sets = {name: split_data[name]["val"] for name in dataset_names}
        test_sets = {name: split_data[name]["test"] for name in dataset_names}

        
        for name in dataset_names:
            sample = train_sets[dataset_names.index(name)][0]
            img_shape = tuple(sample["image"].shape)
            mask_values = torch.unique(sample["mask"]).tolist()
            logger.info("%s preflight | image_shape=%s | sample_mask_labels=%s (training target will be binarized)", name, img_shape, mask_values)
            if len(img_shape) != 4 or img_shape[0] != 1:
                raise ValueError(
                    f"{name} adapter returned image shape {img_shape}; this trainer requires [1,D,H,W]. "
                    "Fix the loader explicitly rather than silently discarding MRI modalities."
                )
            for split_name, ds in (("val", val_sets[name]), ("test", test_sets[name])):
                groups = {_patient_group_id(name, r.subject_id) for r in ds.records}
                if len(groups) < args.eval_shots + 1:
                    raise ValueError(
                        f"{name} {split_name} split has only {len(groups)} independent subjects; "
                        f"--eval-shots {args.eval_shots} requires at least {args.eval_shots + 1}."
                    )

        model = IrisModel(
            in_channels=1,
            base_channels=16,
            num_query_tokens=8,
            volume_shape=(resolution, resolution, resolution),
            use_memory_bank=True,
            memory_momentum=args.memory_momentum,
        ).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info("Model parameters: %.3f million", total_params / 1e6)
        logger.info(
            "EMA memory bank ENABLED | momentum=%.4f | task IDs=%s | entries are saved separately per dataset",
            args.memory_momentum, {name: MEMORY_TASK_IDS[name] for name in dataset_names},
        )
        criterion = DiceCrossEntropyLoss()
        optimizer = Lamb(model.parameters(), lr=1e-4, weight_decay=1e-5, betas=(0.9, 0.999), eps=1e-6)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        history = {"train": [], "validation": []}
        best_val = -math.inf
        start_iteration = 0
        train_rng = random.Random(args.seed + 100)
        memory_update_counts = {name: 0 for name in dataset_names}
        if args.resume_from is not None:
            resume_path = args.resume_from.expanduser().resolve()
            if not resume_path.is_file():
                raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
            checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
            saved_config = checkpoint.get("config", {})
            saved_datasets = tuple(saved_config.get("datasets", dataset_names))
            saved_resolution = tuple(saved_config.get("resolution", (resolution,) * 3))
            if saved_datasets != dataset_names:
                raise ValueError(f"Checkpoint datasets {saved_datasets} do not match requested {dataset_names}.")
            if saved_resolution != (resolution, resolution, resolution):
                raise ValueError(f"Checkpoint resolution {saved_resolution} does not match requested {(resolution,) * 3}.")
            saved_seed = int(saved_config.get("seed", args.seed))
            if saved_seed != args.seed:
                raise ValueError(f"Checkpoint split seed {saved_seed} does not match requested --seed {args.seed}.")
            if not saved_config.get("memory_bank", False):
                raise ValueError("Cannot resume a checkpoint that was trained without the EMA memory bank. Start this memory-bank experiment fresh.")
            if float(saved_config.get("memory_bank_momentum", -1.0)) != float(args.memory_momentum):
                raise ValueError("Checkpoint memory-bank momentum does not match --memory-momentum.")
            model.load_state_dict(checkpoint["model_state_dict"])
            memory_update_counts = restore_bank_from_checkpoint(model, checkpoint, dataset_names, device)
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if scaler.is_enabled() and "scaler_state_dict" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
            start_iteration = int(checkpoint.get("iteration", 0))
            if start_iteration >= iterations:
                raise ValueError(
                    f"Checkpoint is at iteration {start_iteration}; --iterations must be a larger target total (got {iterations})."
                )
            history = load_resume_history(checkpoint, resume_path, start_iteration)
            best_val = float(checkpoint.get("best_validation_macro_dice", -math.inf))
            if "train_rng_state" in checkpoint:
                train_rng.setstate(checkpoint["train_rng_state"])
            else:
                logger.warning("Legacy checkpoint has no training RNG state; sampling RNG restarts from seed %d.", args.seed + 100)
            if "python_rng_state" in checkpoint:
                random.setstate(checkpoint["python_rng_state"])
            if "numpy_rng_state" in checkpoint:
                np.random.set_state(checkpoint["numpy_rng_state"])
            if "torch_rng_state" in checkpoint:
                torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
            if torch.cuda.is_available() and "torch_cuda_rng_state_all" in checkpoint:
                torch.cuda.set_rng_state_all(checkpoint["torch_cuda_rng_state_all"])
            resume_best_source_path = resume_path
            if resume_path.name.lower() != "best_validation.pt":
                sibling_best = resume_path.with_name("best_validation.pt")
                if sibling_best.is_file():
                    candidate = torch.load(sibling_best, map_location="cpu", weights_only=False)
                    candidate_iteration = int(candidate.get("iteration", -1))
                    candidate_config = candidate.get("config", {})
                    candidate_datasets = tuple(candidate_config.get("datasets", dataset_names))
                    candidate_resolution = tuple(candidate_config.get("resolution", (resolution,) * 3))
                    if (candidate_datasets == dataset_names
                            and candidate_resolution == (resolution, resolution, resolution)
                            and 0 <= candidate_iteration <= start_iteration
                            and candidate.get("memory_bank_namespace_version") == MEMORY_BANK_NAMESPACE_VERSION
                            and "memory_bank_state" in candidate
                            and float(candidate.get("memory_bank_momentum", -1.0)) == float(args.memory_momentum)
                            and candidate.get("memory_task_ids", {}) == {name: MEMORY_TASK_IDS[name] for name in dataset_names}):
                        resume_best_source_path = sibling_best
                    else:
                        logger.warning(
                            "Sibling best_validation.pt did not match the resume experiment; "
                            "using the supplied checkpoint as the best candidate."
                        )
                else:
                    logger.warning(
                        "No sibling best_validation.pt found; the supplied resume checkpoint "
                        "will also be the initial best candidate."
                    )
            logger.info("Resumed checkpoint: %s | checkpoint iteration=%d | target total=%d | memory updates=%s", resume_path, start_iteration, iterations, memory_update_counts)
            logger.info("Resume restores model/optimizer/scaler and available RNG/history; cosine LR is recalculated over the remaining %d updates.", iterations - start_iteration)

        
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, iterations - start_iteration), eta_min=1e-6
        )

        best_path = output_dir / "checkpoints" / "best_validation.pt"
        latest_path = output_dir / "checkpoints" / "latest.pt"
        test_rngs = {name: random.Random(args.seed + 300 + i) for i, name in enumerate(dataset_names)}
        started = time.time()
        status = "completed"
        if args.resume_from is not None:
            shutil.copy2(resume_best_source_path, best_path)
            save_checkpoint(latest_path, model, optimizer, scaler, scheduler, start_iteration, config, best_val, history, train_rng, memory_update_counts, dataset_names)

        try:
            for iteration in range(start_iteration + 1, iterations + 1):
                model.train()
                dataset_index = (
                    (iteration - start_iteration - 1) % len(train_sets)
                    if args.smoke_test else train_rng.randrange(len(train_sets))
                )
                dataset = train_sets[dataset_index]
                dataset_name = dataset_names[dataset_index]
                episode = sample_episode(dataset, dataset_name, train_rng, device, training=True, n_supports=1)
                optimizer.zero_grad(set_to_none=True)
                train_loss, scores, _, bank_updated = run_episode(
                    model, episode, criterion, device, optimizer=optimizer,
                    scaler=scaler, training=True, use_amp=use_amp,
                    memory_task_id=MEMORY_TASK_IDS[dataset_name],
                )
                if bank_updated:
                    memory_update_counts[dataset_name] += 1
                else:
                    logger.warning("Optimizer step was skipped by AMP; no EMA update was applied for %s at iteration %d.", dataset_name, iteration)
                scheduler.step()
                usable_scores = [score for score in scores if score is not None]
                train_dice = float(np.mean(usable_scores)) if usable_scores else 0.0
                item = {
                    "iteration": iteration,
                    "dataset": dataset_name,
                    "loss": train_loss,
                    "query_dice_macro": train_dice,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "support_subjects": episode["support_ids"],
                    "query_subject": episode["query_id"],
                    "memory_task_id": MEMORY_TASK_IDS[dataset_name],
                    "memory_updates_for_dataset": memory_update_counts[dataset_name],
                }
                history["train"].append(item)
                if iteration % 10 == 0 or iteration == 1:
                    logger.info(
                        "Iter %d/%d | %s | loss=%.4f | query-Dice=%.4f | EMA-updates=%d | lr=%.2e",
                        iteration, iterations, dataset_name, train_loss, train_dice,
                        memory_update_counts[dataset_name], optimizer.param_groups[0]["lr"],
                    )

                banks_ready = all(
                    MEMORY_TASK_IDS[name] in model.memory_bank
                    for name in dataset_names
                )
                if (iteration % eval_every == 0 or iteration == iterations) and banks_ready:
                    val_results = {}
                    for val_index, name in enumerate(dataset_names):
                        fixed_val_rng = random.Random(args.seed + 200 + val_index)
                        val_results[name] = evaluate_dataset(
                            model, val_sets[name], name, val_episodes, fixed_val_rng,
                            device, criterion, use_amp, eval_shots=args.eval_shots,
                            memory_task_id=MEMORY_TASK_IDS[name],
                        )
                    macro_val = float(np.mean([val_results[name]["dice_mean_memory_bank"] for name in dataset_names]))
                    macro_ensemble_val = float(np.mean([val_results[name]["dice_mean_ensemble"] for name in dataset_names]))
                    validation_record = {
                        "iteration": iteration,
                        "macro_dice_across_datasets": macro_val,
                        "macro_dice_memory_bank_across_datasets": macro_val,
                        "macro_dice_ensemble_across_datasets": macro_ensemble_val,
                        "by_dataset": val_results,
                    }
                    history["validation"].append(validation_record)
                    dataset_summary = " | ".join(
                        f"{name} BANK={val_results[name]['dice_mean_memory_bank']:.4f} "
                        f"| 1-shot={val_results[name]['dice_mean_1shot']:.4f} "
                        f"| ensemble={val_results[name]['dice_mean_ensemble']:.4f} "
                        f"| bank-vs-1={val_results[name]['memory_bank_delta_vs_1shot']:+.4f}"
                        for name in dataset_names
                    )
                    logger.info("VAL @ %d | %s | EMA-bank macro=%.4f | ensemble macro=%.4f", iteration, dataset_summary, macro_val, macro_ensemble_val)
                    save_json_atomic(output_dir / "metrics" / "history.json", history)
                    save_json_atomic(output_dir / "metrics" / "latest_validation.json", validation_record)
                    if macro_val > best_val:
                        best_val = macro_val
                        save_checkpoint(best_path, model, optimizer, scaler, scheduler, iteration, config, best_val, history, train_rng, memory_update_counts, dataset_names)
                        logger.info("Saved best validation checkpoint: %s", best_path)
                elif iteration % eval_every == 0 or iteration == iterations:
                    missing = [name for name in dataset_names if MEMORY_TASK_IDS[name] not in model.memory_bank]
                    logger.warning("Skipping validation until each dataset has an initialized TRAIN memory entry. Missing: %s", missing)

                if iteration % 50 == 0 or iteration == iterations:
                    save_checkpoint(latest_path, model, optimizer, scaler, scheduler, iteration, config, best_val, history, train_rng, memory_update_counts, dataset_names)
                    save_json_atomic(output_dir / "metrics" / "history.json", history)

        except torch.cuda.OutOfMemoryError:
            status = "stopped_cleanly_cuda_oom"
            logger.error("CUDA out of memory. Saving emergency weights and metrics before stopping.")
            torch.cuda.empty_cache()
            last_iteration = max((int(item.get("iteration", 0)) for item in history.get("train", [])), default=start_iteration)
            emergency = output_dir / "checkpoints" / "emergency_oom.pt"
            save_checkpoint(emergency, model, optimizer, scaler, scheduler, last_iteration, config, best_val, history, train_rng, memory_update_counts, dataset_names)
            save_json_atomic(output_dir / "metrics" / "history.json", history)
            save_json_atomic(output_dir / "RUN_STATUS.json", {
                "status": status,
                "last_completed_iteration": len(history["train"]),
                "message": "Run stopped on CUDA OOM. Retry with --resolution 48 or --no-cache. No test metrics were produced.",
            })
            logger.error("Run stopped cleanly. Retry with --resolution 48 or --no-cache; no valid test result was produced.")
            return 3
        except KeyboardInterrupt:
            status = "interrupted_by_user"
            logger.warning("Interrupted by user; saving latest weights and metrics.")
            last_iteration = max((int(item.get("iteration", 0)) for item in history.get("train", [])), default=start_iteration)
            save_checkpoint(latest_path, model, optimizer, scaler, scheduler, last_iteration, config, best_val, history, train_rng, memory_update_counts, dataset_names)
            save_json_atomic(output_dir / "metrics" / "history.json", history)
            save_json_atomic(output_dir / "RUN_STATUS.json", {"status": status, "last_completed_iteration": len(history["train"])})
            return 130

        if not best_path.exists():
            save_checkpoint(best_path, model, optimizer, scaler, scheduler, iterations, config, best_val, history, train_rng, memory_update_counts, dataset_names)

        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        best_bank_counts = restore_bank_from_checkpoint(model, ckpt, dataset_names, device)
        save_memory_bank_files(output_dir, model, dataset_names, best_bank_counts, int(ckpt["iteration"]), tag="best_validation_for_test")
        model.eval()
        test_results = {}
        for name in dataset_names:
            test_results[name] = evaluate_dataset(
                model, test_sets[name], name, test_episodes, test_rngs[name], device, criterion, use_amp,
                eval_shots=args.eval_shots,
                memory_task_id=MEMORY_TASK_IDS[name],
                prediction_dir=output_dir / "predictions" / name,
            )
            logger.info(
                "TEST | %s | EMA-bank=%.4f ± %.4f | 1-shot=%.4f ± %.4f | ensemble=%.4f ± %.4f | bank-vs-1=%+.4f | episodes=%d",
                name, test_results[name]["dice_mean_memory_bank"], test_results[name]["dice_std_memory_bank"],
                test_results[name]["dice_mean_1shot"], test_results[name]["dice_std_1shot"],
                test_results[name]["dice_mean_ensemble"], test_results[name]["dice_std_ensemble"],
                test_results[name]["memory_bank_delta_vs_1shot"], test_results[name]["episodes_evaluated"],
            )
        macro_bank_test = float(np.mean([test_results[n]["dice_mean_memory_bank"] for n in dataset_names]))
        macro_one_test = float(np.mean([test_results[n]["dice_mean_1shot"] for n in dataset_names]))
        macro_ensemble_test = float(np.mean([test_results[n]["dice_mean_ensemble"] for n in dataset_names]))
        result_payload = {
            "status": status,
            "best_validation_iteration": ckpt["iteration"],
            "best_validation_macro_dice": best_val,
            "best_validation_macro_memory_bank_dice": best_val,
            "test_results_by_dataset": test_results,
            "test_macro_dice_memory_bank_across_datasets": macro_bank_test,
            "test_macro_dice_1shot_across_datasets": macro_one_test,
            "test_macro_dice_ensemble_across_datasets": macro_ensemble_test,
            "test_macro_memory_bank_delta_vs_1shot": macro_bank_test - macro_one_test,
            "test_macro_memory_bank_delta_vs_ensemble": macro_bank_test - macro_ensemble_test,
            "macro_dice_definition": "For each dataset, average query-episode Dice first, then take the arithmetic mean across datasets with equal dataset weight. This is not pooled voxel-wise Dice.",
            "memory_bank": {
                "enabled": True,
                "type": "ClassMemoryBank EMA",
                "momentum": float(model.memory_bank.momentum),
                "task_ids": {name: MEMORY_TASK_IDS[name] for name in dataset_names},
                "update_counts_at_selected_checkpoint": best_bank_counts,
                "updates_from": "TRAIN support embeddings only",
                "validation_test_updates": False,
                "bank_only_inference": True,
            },
            "protocol_note": (
                "Binary foreground segmentation. During training, one labelled support/query episode is sampled "
                "from one dataset per iteration; the EMA bank is updated only from the detached TRAIN support "
                "task embedding under a dataset-specific synthetic task ID. At validation/test, three methods "
                "are evaluated on the same held-out query episodes: one support, averaged multi-support context, "
                "and support-free EMA bank retrieval. Validation/test data never update the bank. Supports and "
                "queries are distinct subjects; ACDC frames are grouped by patient in the split. Both datasets "
                "are represented during training, so this is not unseen-dataset generalization. ACDC merges RV, "
                "myocardium and LV into one binary cardiac foreground; Brain Tumour merges all nonzero labels "
                "into whole-tumour foreground."
            ),
            "elapsed_seconds": time.time() - started,
        }
        save_json_atomic(output_dir / "metrics" / "test_metrics.json", result_payload)
        save_json_atomic(output_dir / "metrics" / "history.json", history)
        save_memory_bank_files(output_dir, model, dataset_names, best_bank_counts, int(ckpt["iteration"]), tag="best_validation_for_test")
        write_experiment_report(output_dir, config, result_payload)
        save_json_atomic(output_dir / "RUN_STATUS.json", {"status": status, "last_completed_iteration": iterations, "test_metrics": "metrics/test_metrics.json", "experiment_report": "EXPERIMENT_REPORT.md"})
        logger.info("Finished in %.1f minutes. EMA-bank test macro=%.4f | one-shot macro=%.4f | context-ensemble macro=%.4f | test metrics: %s", (time.time() - started) / 60.0, macro_bank_test, macro_one_test, macro_ensemble_test, output_dir / "metrics" / "test_metrics.json")
        return 0

    except Exception as exc:
        logger.exception("Run failed before completion: %s", exc)
        save_json_atomic(output_dir / "RUN_STATUS.json", {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
