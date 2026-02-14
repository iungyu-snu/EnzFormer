import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
from torch.optim.lr_scheduler import LambdaLR
from sklearn.model_selection import StratifiedKFold
import matplotlib
matplotlib.use('Agg') # Set the backend BEFORE importing pyplot
import matplotlib.pyplot as plt
import os
import json
import csv
import time
from model_250416 import ECT
import numpy as np
import logging
from focal_loss import FocalLoss
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    confusion_matrix,
    precision_recall_fscore_support,
    average_precision_score,
    precision_recall_curve,
)
from tqdm import tqdm

CALIBRATION_N_BINS = 10
SCREENING_K = [10, 50, 100]
SCREENING_EF_PCTS = [0.5, 1, 5, 10]

LABEL_ID_TO_NAME = {
    0: "high_activity",
    1: "low_activity",
    2: "No_activity",
    -1: "reject",
}

# One-vs-rest calibration/screening target.
HIGH_ACTIVITY_LABEL = 0


def label_id_to_name(label: int) -> str:
    if label is None:
        return "unknown"
    try:
        label_i = int(label)
    except Exception:
        return str(label)
    return LABEL_ID_TO_NAME.get(label_i, str(label_i))


def compute_brier_multiclass(probs: np.ndarray, y_true: np.ndarray, n_classes: int) -> float:
    """Multiclass Brier score: mean_i sum_k (p_ik - y_ik)^2."""
    if probs.ndim != 2:
        raise ValueError(f"probs must be 2D [N,C], got shape={probs.shape}")
    if probs.shape[1] != n_classes:
        raise ValueError(f"probs second dim must be n_classes={n_classes}, got shape={probs.shape}")

    y_true = np.asarray(y_true).astype(int)
    if y_true.ndim != 1 or y_true.shape[0] != probs.shape[0]:
        raise ValueError(f"y_true must be 1D [N], got shape={y_true.shape}")

    valid = (y_true >= 0) & (y_true < n_classes)
    if not np.all(valid):
        probs = probs[valid]
        y_true = y_true[valid]
    if probs.shape[0] == 0:
        raise ValueError("No valid samples to compute multiclass Brier score.")

    onehot = np.eye(n_classes, dtype=np.float32)[y_true]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def compute_calibration_bins(conf: np.ndarray, y: np.ndarray, n_bins: int = CALIBRATION_N_BINS) -> dict:
    """
    Computes calibration bins + ECE/MCE.
    - conf: confidence/probability in [0,1]
    - y: binary target in {0,1} (e.g., correctness or y_pos)
    """
    conf = np.asarray(conf, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if conf.ndim != 1 or y.ndim != 1 or conf.shape[0] != y.shape[0]:
        raise ValueError(f"conf and y must be 1D with same length, got conf={conf.shape}, y={y.shape}")
    if conf.shape[0] == 0:
        raise ValueError("No samples to compute calibration bins.")

    conf = np.clip(conf, 0.0, 1.0)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    # Equal-width bins in [0,1]; conf==1.0 goes to the last bin.
    bin_ids = np.floor(conf * n_bins).astype(int)
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    bin_count = np.zeros(n_bins, dtype=np.int64)
    bin_confidence = np.full(n_bins, np.nan, dtype=np.float64)
    bin_accuracy = np.full(n_bins, np.nan, dtype=np.float64)

    for b in range(n_bins):
        mask = bin_ids == b
        cnt = int(mask.sum())
        bin_count[b] = cnt
        if cnt > 0:
            bin_confidence[b] = float(conf[mask].mean())
            bin_accuracy[b] = float(y[mask].mean())

    n = float(conf.shape[0])
    gaps = np.abs(bin_accuracy - bin_confidence)
    ece = float(np.nansum(gaps * (bin_count / n)))
    mce = float(np.nanmax(gaps)) if np.any(bin_count > 0) else 0.0

    def _f(x: float):
        return None if np.isnan(x) else float(x)

    return {
        "ece": float(ece),
        "mce": float(mce),
        "bin_edges": [float(x) for x in bin_edges],
        "bin_count": [int(x) for x in bin_count],
        "bin_confidence": [_f(x) for x in bin_confidence],
        "bin_accuracy": [_f(x) for x in bin_accuracy],
    }


def compute_screening_metrics(
    p_pos: np.ndarray,
    y_pos: np.ndarray,
    k_list: list[int] = SCREENING_K,
    ef_pcts: list[float] = SCREENING_EF_PCTS,
) -> dict:
    """Computes ranking-based metrics for screening using p_pos as the ranking score."""
    p_pos = np.asarray(p_pos, dtype=np.float64)
    y_pos = np.asarray(y_pos, dtype=np.int64)
    if p_pos.ndim != 1 or y_pos.ndim != 1 or p_pos.shape[0] != y_pos.shape[0]:
        raise ValueError(f"p_pos and y_pos must be 1D with same length, got p_pos={p_pos.shape}, y_pos={y_pos.shape}")

    n = int(p_pos.shape[0])
    n_pos = int(y_pos.sum())
    prevalence = float(n_pos / n) if n > 0 else 0.0

    order = np.argsort(-p_pos, kind="mergesort")  # stable, deterministic

    def pct_key(pct) -> str:
        try:
            return format(float(pct), "g")
        except Exception:
            return str(pct)

    no_pos = n_pos == 0
    if no_pos:
        logging.warning("Screening metrics: no positive samples for the focus class.")

    # PR-AUC / Average Precision for high_activity vs rest (binary one-vs-rest)
    pr_auc = {"ap": None, "auprc_trapz": None}
    if not no_pos:
        try:
            pr_auc["ap"] = float(average_precision_score(y_pos, p_pos))
        except Exception as e:
            logging.warning(f"Failed to compute Average Precision(AP): {e}")

        try:
            precision, recall, _ = precision_recall_curve(y_pos, p_pos)
            # sklearn returns recall in decreasing order; reverse for trapezoid integration.
            pr_auc["auprc_trapz"] = float(np.trapezoid(precision[::-1], recall[::-1]))
        except Exception as e:
            logging.warning(f"Failed to compute PR-AUC (trapz): {e}")

    precision_at_k = {}
    recall_at_k = {}
    for k in k_list:
        k_eff = int(min(int(k), n))
        if k_eff <= 0:
            precision_at_k[str(int(k))] = None
            recall_at_k[str(int(k))] = None
            continue
        top = order[:k_eff]
        pos_in_top = int(y_pos[top].sum())
        precision_at_k[str(int(k))] = float(pos_in_top / k_eff)
        if no_pos:
            recall_at_k[str(int(k))] = None
        else:
            recall_at_k[str(int(k))] = float(pos_in_top / n_pos)

    precision_at_percent = {}
    recall_at_percent = {}

    ef_at_percent = {}
    for pct in ef_pcts:
        pct_f = float(pct)
        key = pct_key(pct_f)
        n_top = int(max(1, int(np.ceil(n * (pct_f / 100.0)))))
        top = order[:n_top]
        pos_in_top = int(y_pos[top].sum())
        top_rate = float(pos_in_top / n_top)
        precision_at_percent[key] = float(top_rate)
        if no_pos:
            recall_at_percent[key] = None
        else:
            recall_at_percent[key] = float(pos_in_top / n_pos)
        if prevalence == 0.0:
            ef_at_percent[key] = None
        else:
            ef_at_percent[key] = float(top_rate / prevalence)

    return {
        "n_samples": int(n),
        "n_positive": int(n_pos),
        "prevalence": float(prevalence),
        "pr_auc": pr_auc,
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
        "precision_at_percent": precision_at_percent,
        "recall_at_percent": recall_at_percent,
        "ef_at_percent": ef_at_percent,
    }


def plot_reliability_diagram(
    bin_confidence: list,
    bin_accuracy: list,
    bin_count: list,
    save_path: str,
    title: str,
    y_label: str = "Accuracy",
) -> None:
    """Plots and saves a reliability diagram from precomputed bins."""
    try:
        xs = []
        ys = []
        for c, a, n in zip(bin_confidence, bin_accuracy, bin_count):
            if n is None or int(n) <= 0:
                continue
            if c is None or a is None:
                continue
            xs.append(float(c))
            ys.append(float(a))

        if not xs:
            logging.warning(f"Skipping reliability plot (no non-empty bins): {save_path}")
            return

        plt.figure(figsize=(6, 6))
        plt.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="gray", linewidth=1)
        plt.plot(xs, ys, marker="o", linestyle="-", color="tab:blue")
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.0)
        plt.xlabel("Confidence")
        plt.ylabel(y_label)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        logging.info(f"Reliability plot saved to {save_path}")
    except Exception as e:
        logging.error(f"Failed to save reliability plot to {save_path}: {e}")


def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training", total=len(dataloader))):
        if batch is None:
            logging.warning(f"Skipping empty batch at index {batch_idx}")
            continue
        fasta_embeds, annotations = batch  # Data first, labels second

        fasta_embeds = fasta_embeds.to(device)
        annotations = annotations.to(device)
        # gradient를 초기화 하는 거지 파라미터를 초기화 하진 않음
        optimizer.zero_grad()
        outputs = model(fasta_embeds)
        loss = criterion(outputs, annotations)
        loss.backward()
        optimizer.step()
        # loss.item() is typically mean over the batch; weight by batch size for global mean.
        bs = annotations.size(0)
        total_loss += loss.item() * bs
        total_samples += bs

    if total_samples == 0:
        raise RuntimeError("No valid training samples: all batches were empty/filtered.")

    return total_loss / total_samples


def validate(model, dataloader, criterion, device, threshold):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    correct_predictions = 0
    total_samples = 0

    # Measure wall-time for a full pass over the validation dataloader (inference loop only).
    if device.type == "cuda":
        torch.cuda.synchronize()
    infer_t0 = time.perf_counter()

    with torch.no_grad(): # Disable gradient calculations
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Validation", total=len(dataloader))):
            if batch is None:
                 logging.warning(f"Skipping empty validation batch at index {batch_idx}")
                 continue

            fasta_embeds, annotations = batch
            fasta_embeds = fasta_embeds.to(device)
            annotations = annotations.to(device)

            outputs = model(fasta_embeds)
            loss = criterion(outputs, annotations)
            bs = annotations.size(0)
            total_loss += loss.item() * bs
            # outputs shape (Batch,outputdim)
            probabilities = torch.softmax(outputs, dim=1)
            all_probs.append(probabilities.detach().cpu().numpy())
            # max_probs 해당 차원에서 가장 큰 값, preds는 가장 큰 값이 있는 인덱스
            max_probs, preds = torch.max(probabilities, dim=1)

            # Apply threshold
            preds_thresholded = preds.clone()
            preds_thresholded[max_probs < threshold] = -1  # Assign -1 to predictions below threshold
            # list에 하나씩 원소 풀어서 extend
            all_preds.extend(preds_thresholded.cpu().numpy())
            all_labels.extend(annotations.cpu().numpy())

            total_samples += bs
            # 예측값과 정답이 같으면 1, 다르면 0으로 카운트 그걸 sum 하면 스칼라 tensor가 나옴
            correct_predictions += (preds_thresholded == annotations).sum().item()

    if device.type == "cuda":
        torch.cuda.synchronize()
    inference_seconds = float(time.perf_counter() - infer_t0)

    if total_samples == 0:
        raise RuntimeError("No valid validation samples: all batches were empty/filtered.")

    all_preds_array = np.array(all_preds)
    all_labels_array = np.array(all_labels)

    probs = np.concatenate(all_probs, axis=0)
    if probs.shape[0] != all_labels_array.shape[0]:
        raise RuntimeError(
            f"Probability/label length mismatch: probs={probs.shape[0]} labels={all_labels_array.shape[0]}"
        )

    # Extra metrics
    balanced_acc = balanced_accuracy_score(all_labels_array, all_preds_array)
    try:
        mcc = matthews_corrcoef(all_labels_array, all_preds_array)
    except Exception as e:
        logging.warning(f"Failed to compute MCC: {e}")
        mcc = 0.0

    # Calculate metrics
    # Use labels=np.arange(output_dim) to include all potential classes in macro averaging.
    # Use zero_division=0 to handle cases where a class has no true samples or no predictions.
    labels_for_metrics = np.arange(model.output_dim)

    # ===== Calibration (computed pre-threshold, from softmax probabilities) =====
    y_true = all_labels_array.astype(int)
    n_classes = int(model.output_dim)
    if probs.ndim != 2 or probs.shape[1] != n_classes:
        raise RuntimeError(f"Unexpected probs shape={probs.shape}, expected [N,{n_classes}]")

    conf = probs.max(axis=1)
    pred_raw = probs.argmax(axis=1)
    correct_raw = (pred_raw == y_true).astype(int)
    brier_mc = compute_brier_multiclass(probs, y_true, n_classes=n_classes)
    cal_mc = compute_calibration_bins(conf, correct_raw, n_bins=CALIBRATION_N_BINS)
    cal_mc["brier"] = float(brier_mc)

    pos_label = int(HIGH_ACTIVITY_LABEL)
    if not (0 <= pos_label < n_classes):
        pos_label = n_classes - 1
    p_pos = probs[:, pos_label]
    y_pos = (y_true == pos_label).astype(int)
    brier_high = float(np.mean((p_pos - y_pos) ** 2))
    cal_high = compute_calibration_bins(p_pos, y_pos, n_bins=CALIBRATION_N_BINS)
    cal_high["brier"] = float(brier_high)
    cal_high["positive_label"] = int(pos_label)
    cal_high["positive_label_name"] = label_id_to_name(pos_label)

    calibration = {
        "n_bins": int(CALIBRATION_N_BINS),
        "multiclass": cal_mc,
        "high_class": cal_high,
    }

    # ===== Screening metrics (high-class, one-vs-rest) =====
    screening = compute_screening_metrics(
        p_pos=p_pos,
        y_pos=y_pos,
        k_list=SCREENING_K,
        ef_pcts=SCREENING_EF_PCTS,
    )
    screening["positive_label"] = int(pos_label)
    screening["positive_label_name"] = label_id_to_name(pos_label)

    precision = precision_score(
        all_labels_array,
        all_preds_array,
        labels=labels_for_metrics,
        average="macro",
        zero_division=0,
    )
    recall = recall_score(
        all_labels_array,
        all_preds_array,
        labels=labels_for_metrics,
        average="macro",
        zero_division=0,
    )
    f1 = f1_score(
        all_labels_array,
        all_preds_array,
        labels=labels_for_metrics,
        average="macro",
        zero_division=0,
    )

    per_cls_p, per_cls_r, per_cls_f1, per_cls_support = precision_recall_fscore_support(
        all_labels_array,
        all_preds_array,
        labels=labels_for_metrics,
        average=None,
        zero_division=0,
    )
    per_class_metrics = {}
    for i, label in enumerate(labels_for_metrics):
        per_class_metrics[int(label)] = {
            "name": label_id_to_name(label),
            "precision": float(per_cls_p[i]),
            "recall": float(per_cls_r[i]),
            "f1": float(per_cls_f1[i]),
            "support": int(per_cls_support[i]),
        }

    cm_labels = labels_for_metrics.tolist()
    if np.any(all_preds_array == -1) or np.any(all_labels_array == -1):
        # Put reject label (-1) at the end for readability.
        cm_labels = cm_labels + [-1]
    cm = confusion_matrix(all_labels_array, all_preds_array, labels=cm_labels)

    accuracy = correct_predictions / total_samples
    average_loss = total_loss / total_samples

    inference = {
        "seconds_total": float(inference_seconds),
        "seconds_per_sample": float(inference_seconds / total_samples) if total_samples else None,
        "samples_per_sec": float(total_samples / inference_seconds) if inference_seconds > 0 else None,
        "n_samples": int(total_samples),
        "device": str(device.type),
    }

    return (
        average_loss,
        accuracy,
        float(balanced_acc),
        float(mcc),
        precision,
        recall,
        f1,
        per_class_metrics,
        cm,
        cm_labels,
        calibration,
        screening,
        inference,
    )


def pad_collate_fn(batch):
    """
    Pads sequences in the batch to the maximum length in the batch.
    Filters out samples with zero-dimensional data tensors.
    Assumes that each item in the batch is a tuple (data_tensor, label_tensor).
    """
    # Filter out potential None items if any were yielded by the dataset
    batch = [
        item
        for item in batch
        if item is not None and item[0] is not None and item[1] is not None
    ]
    if not batch:
        return None # Return None if the batch is empty after filtering

    # Keep (data, label) pairs so we can drop them together.
    pairs = []
    for data, label in batch:
        if not torch.is_tensor(data) or data.ndim < 2:
            logging.error(
                f"Invalid data tensor (ndim < 2). Dropping sample. type={type(data)}"
            )
            continue
        if not torch.is_tensor(label):
            label = torch.tensor(label, dtype=torch.long)
        pairs.append((data, label))

    if not pairs:
        return None

    # Choose the most common embed_dim in this batch (robust to occasional bad samples).
    embed_dims = [data.size(1) for data, _ in pairs]
    target_embed_dim = max(set(embed_dims), key=embed_dims.count)

    filtered = [(data, label) for data, label in pairs if data.size(1) == target_embed_dim]
    dropped = len(pairs) - len(filtered)
    if dropped:
        logging.warning(
            f"Dropped {dropped} samples due to embed_dim mismatch. target_embed_dim={target_embed_dim}"
        )

    if not filtered:
        return None

    max_length = max(data.size(0) for data, _ in filtered)

    padded_data = []
    padded_labels = []
    for data, label in filtered:
        padding_length = max_length - data.size(0)
        if padding_length > 0:
            padding = torch.zeros(
                (padding_length, target_embed_dim),
                dtype=data.dtype,
                device=data.device,
            )
            data = torch.cat((data, padding), dim=0)
        padded_data.append(data)
        padded_labels.append(label)

    # Stack tensors
    try:
        batch_data = torch.stack(padded_data, dim=0)
        batch_labels = torch.stack(padded_labels, dim=0)
    except Exception as e:
        logging.error(
            f"Error stacking tensors: {e}. data={[p.shape for p in padded_data]}"
        )
        return None

    return batch_data, batch_labels


class EmbedDataset(Dataset):
    def __init__(self, data_dir):
        self.samples = []
        self.labels = []  # Store labels to be used for stratification

        for filename in os.listdir(data_dir):
            if filename.endswith(".npy") and not filename.endswith("_dm.npy"):
                base_name = os.path.splitext(filename)[0]
                npy_path = os.path.join(data_dir, filename)
                header_path = os.path.join(data_dir, f"{base_name}_header.txt")

                if not os.path.exists(header_path):
                    logging.warning(f"Header file {header_path} not found for {npy_path}, skipping sample.")
                    continue

                try:
                    with open(header_path, "r") as f:
                        annotation = f.read().strip()
                    if not annotation:
                        logging.warning(f"Empty annotation in {header_path}, skipping sample.")
                        continue
                    
                    label = int(annotation)
                    self.samples.append((npy_path, header_path))
                    self.labels.append(label)

                except Exception as e:
                    logging.error(f"Error processing label for {header_path}, skipping sample: {e}")
                    continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npy_path, header_path = self.samples[idx]
        label = self.labels[idx]

        # Load the numpy arrays
        data = np.load(npy_path)

        # Convert to tensors
        data_tensor = torch.from_numpy(data).float()
        label_tensor = torch.tensor(label, dtype=torch.long)

        return data_tensor, label_tensor

    def get_labels(self):
        return self.labels


def plot_fold_metrics(fold_idx, train_losses, val_losses, val_f1_scores, model_name, save_dir, plot_filename_base):
    """Plots and saves the training/validation metrics for a single fold."""
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, marker='o', linestyle='-', label=f"Fold {fold_idx+1} Training Loss")
    plt.plot(epochs, val_losses, marker='s', linestyle='--', label=f"Fold {fold_idx+1} Validation Loss")
    plt.plot(epochs, val_f1_scores, marker='^', linestyle=':', label=f"Fold {fold_idx+1} Validation F1 Score")
    plt.title(f"Training & Validation Metrics - {model_name} - Fold {fold_idx+1}")
    plt.xlabel("Epochs")
    plt.ylabel("Metrics")
    plt.legend()
    plt.grid(True)

    plot_save_path = os.path.join(save_dir, f"{plot_filename_base}_fold{fold_idx+1}.png")
    try:
        plt.savefig(plot_save_path)
        plt.close()
        logging.info(f"Metrics plot for fold {fold_idx+1} saved to {plot_save_path}")
    except Exception as e:
        logging.error(f"Failed to save plot for fold {fold_idx+1} to {plot_save_path}: {e}")

def plot_average_metrics(k_folds, train_losses_all, val_losses_all, val_f1_scores_all, model_name, save_dir, plot_filename_base):
    """Plots and saves the average training/validation metrics across all folds."""
    if k_folds <= 1:
        return # No average plot needed for a single fold

    # Handle potentially different lengths due to early stopping
    min_epochs = min(len(losses) for losses in train_losses_all)
    if min_epochs == 0:
        logging.warning("Cannot plot average metrics: no epochs completed in at least one fold.")
        return

    # Truncate all results to the minimum number of epochs completed across folds
    train_losses_trunc = [losses[:min_epochs] for losses in train_losses_all]
    val_losses_trunc = [losses[:min_epochs] for losses in val_losses_all]
    val_f1_scores_trunc = [scores[:min_epochs] for scores in val_f1_scores_all]

    # Calculate averages
    avg_train_losses = np.mean(train_losses_trunc, axis=0)
    avg_val_losses = np.mean(val_losses_trunc, axis=0)
    avg_val_f1_scores = np.mean(val_f1_scores_trunc, axis=0)

    epochs = range(1, min_epochs + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, avg_train_losses, marker='o', linestyle='-', label="Average Training Loss")
    plt.plot(epochs, avg_val_losses, marker='s', linestyle='--', label="Average Validation Loss")
    plt.plot(epochs, avg_val_f1_scores, marker='^', linestyle=':', label="Average Validation F1 Score")
    plt.title(f"Average Training & Validation Metrics across {k_folds} Folds - {model_name}")
    plt.xlabel(f"Epochs (up to min completed: {min_epochs})")
    plt.ylabel("Average Metrics")
    plt.legend()
    plt.grid(True)

    aggregate_plot_save_path = os.path.join(save_dir, f"{plot_filename_base}_average.png")
    try:
        plt.savefig(aggregate_plot_save_path)
        plt.close()
        logging.info(f"Aggregate metrics plot saved to {aggregate_plot_save_path}")
    except Exception as e:
        logging.error(f"Failed to save aggregate plot to {aggregate_plot_save_path}: {e}")

def plot_confusion_matrix(cm, labels, save_path, title):
    """Plots and saves a confusion matrix."""
    if cm is None or labels is None:
        return

    try:
        plt.figure(figsize=(6, 6))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title(title)
        plt.colorbar()

        tick_labels = [label_id_to_name(l) for l in labels]
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, tick_labels, rotation=45, ha="right")
        plt.yticks(tick_marks, tick_labels)

        # Annotate cells
        thresh = cm.max() / 2.0 if cm.size else 0.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(
                    j,
                    i,
                    str(int(cm[i, j])),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                )

        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        logging.info(f"Confusion matrix plot saved to {save_path}")
    except Exception as e:
        logging.error(f"Failed to save confusion matrix plot to {save_path}: {e}")


def count_parameters(model) -> tuple[int, int]:
    """Returns (total_params, trainable_params)."""
    total = 0
    trainable = 0
    for p in model.parameters():
        n = int(p.numel())
        total += n
        if p.requires_grad:
            trainable += n
    return int(total), int(trainable)


def align_confusion_matrix(cm: np.ndarray, cm_labels: list[int], target_labels: list[int]) -> np.ndarray:
    """Aligns confusion matrix to target_labels order; missing labels are filled with zeros."""
    n = int(len(target_labels))
    aligned = np.zeros((n, n), dtype=np.int64)
    if cm is None or cm_labels is None:
        return aligned
    try:
        cm_arr = np.asarray(cm)
        if cm_arr.ndim != 2:
            return aligned
        if cm_arr.shape[0] != len(cm_labels) or cm_arr.shape[1] != len(cm_labels):
            return aligned
        label_to_idx = {int(l): int(i) for i, l in enumerate(cm_labels)}
        for i, true_l in enumerate(target_labels):
            ti = label_to_idx.get(int(true_l))
            if ti is None:
                continue
            for j, pred_l in enumerate(target_labels):
                pj = label_to_idx.get(int(pred_l))
                if pj is None:
                    continue
                aligned[i, j] = int(cm_arr[ti, pj])
    except Exception as e:
        logging.warning(f"Failed to align confusion matrix: {e}")
    return aligned


def sanitize_name(s: str) -> str:
    """Sanitize strings for CSV column names."""
    if s is None:
        return "unknown"
    s = str(s)
    out = []
    for ch in s:
        if ch.isalnum() or ch == "_":
            out.append(ch)
        else:
            out.append("_")
    cleaned = "".join(out)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    cleaned = cleaned.strip("_")
    return cleaned if cleaned else "unknown"


def make_fold_row(
    *,
    run_identifier: str,
    fold: int | str,
    best_epoch: int | None,
    best_val_loss,
    best_accuracy,
    best_balanced_acc,
    best_mcc,
    best_precision,
    best_recall,
    best_macro_f1,
    per_class_metrics: dict | None,
    confusion_matrix,
    confusion_labels,
    calibration: dict | None,
    screening: dict | None,
    inference: dict | None,
    param_total: int | None,
    param_trainable: int | None,
) -> dict:
    """Creates a flat dict row for fold-level summary CSV."""
    row = {
        "run_identifier": run_identifier,
        "fold": fold,
        "best_epoch": best_epoch,
        "val_loss": float(best_val_loss) if best_val_loss is not None else None,
        "accuracy": float(best_accuracy) if best_accuracy is not None else None,
        "balanced_accuracy": float(best_balanced_acc) if best_balanced_acc is not None else None,
        "mcc": float(best_mcc) if best_mcc is not None else None,
        "macro_precision": float(best_precision) if best_precision is not None else None,
        "macro_recall": float(best_recall) if best_recall is not None else None,
        "macro_f1": float(best_macro_f1) if best_macro_f1 is not None else None,
        "model_param_count_total": int(param_total) if param_total is not None else None,
        "model_param_count_trainable": int(param_trainable) if param_trainable is not None else None,
    }

    # Inference timing
    if isinstance(inference, dict):
        row["inference__seconds_total"] = inference.get("seconds_total")
        row["inference__seconds_per_sample"] = inference.get("seconds_per_sample")
        row["inference__samples_per_sec"] = inference.get("samples_per_sec")
        row["inference__n_samples"] = inference.get("n_samples")

    # Per-class metrics (fixed labels 0/1/2)
    if isinstance(per_class_metrics, dict):
        for label in (0, 1, 2):
            name = sanitize_name(label_id_to_name(label))
            m = per_class_metrics.get(label)
            if m is None:
                m = per_class_metrics.get(int(label)) if not isinstance(label, bool) else None
            if not isinstance(m, dict):
                row[f"per_class__{name}__precision"] = None
                row[f"per_class__{name}__recall"] = None
                row[f"per_class__{name}__f1"] = None
                row[f"per_class__{name}__support"] = None
            else:
                row[f"per_class__{name}__precision"] = m.get("precision")
                row[f"per_class__{name}__recall"] = m.get("recall")
                row[f"per_class__{name}__f1"] = m.get("f1")
                row[f"per_class__{name}__support"] = m.get("support")
    else:
        for label in (0, 1, 2):
            name = sanitize_name(label_id_to_name(label))
            row[f"per_class__{name}__precision"] = None
            row[f"per_class__{name}__recall"] = None
            row[f"per_class__{name}__f1"] = None
            row[f"per_class__{name}__support"] = None

    # Calibration scalars
    if isinstance(calibration, dict):
        mc = calibration.get("multiclass", {})
        hc = calibration.get("high_class", {})
        if isinstance(mc, dict):
            row["calibration__multiclass__ece"] = mc.get("ece")
            row["calibration__multiclass__mce"] = mc.get("mce")
            row["calibration__multiclass__brier"] = mc.get("brier")
        if isinstance(hc, dict):
            row["calibration__high_class__ece"] = hc.get("ece")
            row["calibration__high_class__mce"] = hc.get("mce")
            row["calibration__high_class__brier"] = hc.get("brier")

    # Screening scalars
    if isinstance(screening, dict):
        pr = screening.get("pr_auc", {})
        if isinstance(pr, dict):
            row["screening__pr_auc__ap"] = pr.get("ap")
            row["screening__pr_auc__auprc_trapz"] = pr.get("auprc_trapz")

        row["screening__n_samples"] = screening.get("n_samples")
        row["screening__n_positive"] = screening.get("n_positive")
        row["screening__prevalence"] = screening.get("prevalence")

        p_at_k = screening.get("precision_at_k", {})
        r_at_k = screening.get("recall_at_k", {})
        if isinstance(p_at_k, dict):
            for k in ("10", "50", "100"):
                row[f"screening__precision_at_k__{k}"] = p_at_k.get(k)
        if isinstance(r_at_k, dict):
            for k in ("10", "50", "100"):
                row[f"screening__recall_at_k__{k}"] = r_at_k.get(k)

        p_at_pct = screening.get("precision_at_percent", {})
        r_at_pct = screening.get("recall_at_percent", {})
        ef_at_pct = screening.get("ef_at_percent", {})
        for pct in ("0.5", "1", "5", "10"):
            if isinstance(p_at_pct, dict):
                row[f"screening__precision_at_percent__{pct}"] = p_at_pct.get(pct)
            if isinstance(r_at_pct, dict):
                row[f"screening__recall_at_percent__{pct}"] = r_at_pct.get(pct)
            if isinstance(ef_at_pct, dict):
                row[f"screening__ef_at_percent__{pct}"] = ef_at_pct.get(pct)

    # Confusion matrix flatten (fixed labels 0/1/2/-1)
    target_labels = [0, 1, 2, -1]
    aligned_cm = align_confusion_matrix(confusion_matrix, confusion_labels, target_labels)
    for i, true_l in enumerate(target_labels):
        true_name = sanitize_name(label_id_to_name(true_l))
        for j, pred_l in enumerate(target_labels):
            pred_name = sanitize_name(label_id_to_name(pred_l))
            row[f"cm__true_{true_name}__pred_{pred_name}"] = int(aligned_cm[i, j])

    return row


def compute_mean_std_rows(fold_rows: list[dict]) -> tuple[dict, dict]:
    """Computes mean/std rows over numeric scalar columns (ignoring None)."""
    mean_row: dict = {"fold": "mean"}
    std_row: dict = {"fold": "std"}
    if not fold_rows:
        return mean_row, std_row

    run_id = fold_rows[0].get("run_identifier")
    mean_row["run_identifier"] = run_id
    std_row["run_identifier"] = run_id

    keys = set()
    for r in fold_rows:
        keys.update(r.keys())

    for k in sorted(keys):
        if k in ("fold", "run_identifier"):
            continue

        # Determine if this column is numeric by inspecting non-None values.
        is_numeric = False
        for r in fold_rows:
            v = r.get(k)
            if v is None:
                continue
            if isinstance(v, bool):
                is_numeric = False
                break
            if isinstance(v, (int, float, np.integer, np.floating)):
                is_numeric = True
                break
            try:
                float(v)
                is_numeric = True
                break
            except Exception:
                is_numeric = False
                break

        if not is_numeric:
            continue

        arr = []
        for r in fold_rows:
            v = r.get(k)
            if v is None or v == "":
                arr.append(np.nan)
            else:
                try:
                    arr.append(float(v))
                except Exception:
                    arr.append(np.nan)
        arr = np.asarray(arr, dtype=np.float64)
        if np.all(np.isnan(arr)):
            mean_row[k] = None
            std_row[k] = None
            continue

        mean_row[k] = float(np.nanmean(arr))
        n_non_nan = int(np.sum(~np.isnan(arr)))
        if n_non_nan < 2:
            std_row[k] = None
        else:
            std_row[k] = float(np.nanstd(arr, ddof=1))

    return mean_row, std_row


def write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        logging.warning(f"No rows to write: {path}")
        return

    keys = set()
    for r in rows:
        keys.update(r.keys())

    # Put identifying columns first for readability.
    rest = sorted([k for k in keys if k not in ("run_identifier", "fold")])
    fieldnames = ["run_identifier", "fold"] + rest

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            out = {}
            for k in fieldnames:
                v = r.get(k)
                out[k] = "" if v is None else v
            w.writerow(out)


def main():
    parser = argparse.ArgumentParser(description="Train your model with specific data")
    parser.add_argument(
        "model_name",
        type=str,
        help="The name of the ESM BERT model: [8M,35M,150M,650M,3B,15B]",
    )
    parser.add_argument(
        "save_dir", type=str, help="Directory for saving model checkpoints and plots"
    )
    parser.add_argument(
        "output_dim",
        type=int,
        help="Number of groups to classify (1 for binary, >1 for multi-class)",
    )
    parser.add_argument("num_blocks", type=int, help="Number of linear blocks to use")
    parser.add_argument("batch_size", type=int, help="Batch size to use")
    parser.add_argument("learning_rate", type=float, help="Learning rate to use")
    parser.add_argument("num_epochs", type=int, help="Number of epochs")
    parser.add_argument("n_head", type=int, help="Number of heads for attention")
    parser.add_argument(
        "threshold", type=float, help="Threshold for precision, recall, F1"
    )
    parser.add_argument(
        "optimizer",
        type=str,
        choices=["Adam", "AdamW"],
        help="Choose 'Adam' or 'AdamW' as the optimizer",
    )
    parser.add_argument(
        "--dropout_rate", type=float, default=0.1, help="Dropout rate to use"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.00001, help="Weight decay value"
    )
    parser.add_argument("--nogpu", action="store_true", help="Use CPU instead of GPU")

    args = parser.parse_args()

    # Print arguments for verification
    print(f"Model name: {args.model_name}")
    print(f"Save directory: {args.save_dir}")
    print(f"Output dimension: {args.output_dim}")
    print(f"Number of blocks: {args.num_blocks}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Number of epochs: {args.num_epochs}")
    print(f"Dropout rate: {args.dropout_rate}")
    print(f"Weight Decay: {args.weight_decay}")
    print(f"Number of heads for Attention: {args.n_head}")
    print(f"Threshold for metrics: {args.threshold}")
    print(f"Optimizer: {args.optimizer}")
    print(f"Use GPU: {'Yes' if not args.nogpu and torch.cuda.is_available() else 'No'}")

    ########## Training Parameters #################
    output_dim = args.output_dim
    num_blocks = args.num_blocks
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    dropout_rate = args.dropout_rate
    weight_decay = args.weight_decay
    save_dir = args.save_dir
    optimizer_name = args.optimizer # Use a different name to avoid conflict with optimizer instance
    model_name = args.model_name
    n_head = args.n_head
    threshold = args.threshold
    data_dir = f"/ssddata/data"
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.nogpu else "cpu"
    )
    k_folds = 5
    os.makedirs(save_dir, exist_ok=True)

    # =========
    # Data processing
    logging.info(f"Building Dataset from {data_dir}...")
    try:
        dataset = EmbedDataset(data_dir)
        if len(dataset) == 0:
            logging.error(f"No valid samples found in {data_dir}. Exiting.")
            return # Exit if no data
    except Exception as e:
        logging.error(f"Failed to load dataset from {data_dir}: {e}")
        return
    logging.info(f"Dataset built successfully with {len(dataset)} samples.")

    ################ 5-Cross Validation #############

    kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    all_fold_best_f1 = []
    all_fold_best_balanced_acc = []
    all_fold_best_mcc = []
    all_fold_rows = []
    train_losses_all_folds = []
    val_losses_all_folds = []
    val_f1_scores_all_folds = []

    # Base filename for saving models and plots, incorporating hyperparameters
    run_identifier = f"{model_name}_blocks{num_blocks}_bs{batch_size}_lr{learning_rate}_do{dropout_rate}_wd{weight_decay}_nh{n_head}_opt{optimizer_name}"
    
    dataset_indices = np.arange(len(dataset))
    dataset_labels = dataset.get_labels()

    for fold, (train_indices, val_indices) in enumerate(kf.split(dataset_indices, dataset_labels)):
        logging.info(f"===== FOLD {fold+1}/{k_folds} =====")

        # Remove or adjust this line if you want to train all folds
        # if fold != 0:
        #      logging.info("Skipping fold based on conditional check.")
        #      continue

        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)

        # Use num_workers for potentially faster data loading
        num_workers = min(4, os.cpu_count() // 2 if os.cpu_count() else 1)
        logging.debug(f"Using {num_workers} workers for DataLoaders.")

        train_dataloader = DataLoader(
            train_subset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn, num_workers=num_workers, pin_memory=True if device.type == 'cuda' else False
        )
        val_dataloader = DataLoader(
            val_subset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_fn, num_workers=num_workers, pin_memory=True if device.type == 'cuda' else False
        )

        logging.info(f"Fold {fold+1}: Train samples={len(train_subset)}, Validation samples={len(val_subset)}")

        model = ECT(
            model_name=model_name,
            output_dim=output_dim,
            num_blocks=num_blocks,
            n_head=n_head,
            dropout_rate=dropout_rate,
        ).to(device)

        param_total, param_trainable = count_parameters(model)
        logging.info(
            f"Model parameters (fold {fold+1}): total={param_total} trainable={param_trainable}"
        )

        # Criterion
        if output_dim == 1:
            criterion = nn.BCEWithLogitsLoss().to(device)
        elif output_dim > 1:
            # Dynamically calculate class counts from the training subset for this fold
            train_labels = [dataset.labels[i] for i in train_indices]
            counts = np.bincount(train_labels, minlength=output_dim)

            # --- Consider moving class counts/weights calculation outside the fold loop if static --- 
            if len(counts) > output_dim:
                logging.warning(f"Found more classes in data ({len(counts)}) than expected ({output_dim}). Truncating.")
                counts = counts[:output_dim]
            
            # Use calculated counts for Focal Loss weights
            total_samples_in_fold = counts.sum()
            if total_samples_in_fold > 0:
                class_weights = total_samples_in_fold / (output_dim * counts + 1e-9) # Add epsilon
                alpha = torch.tensor(class_weights, dtype=torch.float32)
            else:
                logging.warning(f"Fold {fold+1} has no training samples. Using uniform weights.")
                alpha = torch.ones(output_dim, dtype=torch.float32)

            alpha = alpha.to(device)
            logging.info(f"Using Focal Loss with dynamically calculated alpha for Fold {fold+1}: {alpha.cpu().numpy()}")
            criterion = FocalLoss(gamma=2, alpha=alpha).to(device)
            # --- End of class weights section --- 
        else:
            logging.error("Invalid output_dim. Must be 1 or > 1.")
            raise ValueError("Invalid output_dim")

        # Optimizer
        if optimizer_name == "Adam":
            optimizer = optim.Adam(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
        elif optimizer_name == "AdamW":
            optimizer = optim.AdamW(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
        else:
            logging.error(f"Unsupported optimizer: {optimizer_name}")
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        # Scheduler
        warmup_steps = 5 # Consider making this an argument
        def lr_lambda(current_step):
             # Linear warmup
            if current_step < warmup_steps:
                return float(current_step + 1) / float(max(1.0, warmup_steps))
            # Exponential decay after warmup (adjust decay factor 0.95 if needed)
            return 0.95 ** (current_step - warmup_steps + 1)

        scheduler = LambdaLR(optimizer, lr_lambda)

        # Early stopping setup
        early_stopping_patience = 5 # Consider making this an argument
        max_val_f1 = -np.Inf
        patience_counter = 0
        best_model_state = None
        best_epoch = -1
        best_val_loss = None
        best_accuracy = None
        best_balanced_acc = None
        best_mcc = None
        best_precision = None
        best_recall = None
        best_per_class_metrics = None
        best_confusion_matrix = None
        best_confusion_labels = None
        best_calibration = None
        best_screening = None
        best_inference = None

        fold_train_losses = []
        fold_val_losses = []
        fold_val_f1_scores = []

        logging.info(f"Starting training for fold {fold+1}...")
        for epoch in range(num_epochs):
            avg_train_loss = train(model, train_dataloader, criterion, optimizer, device)
            fold_train_losses.append(avg_train_loss)

            # Validation
            (
                val_loss,
                accuracy,
                balanced_acc,
                mcc,
                precision,
                recall,
                f1,
                per_class_metrics,
                cm,
                cm_labels,
                calibration,
                screening,
                inference,
            ) = validate(
                model, val_dataloader, criterion, device, threshold
            )
            fold_val_losses.append(val_loss)
            fold_val_f1_scores.append(f1)

            current_lr = scheduler.get_last_lr()[0]
            # Short screening/calibration summary for epoch logging.
            high_ece = None
            high_brier = None
            p_at_10 = None
            ef_at_1 = None
            ap = None
            try:
                high_ece = calibration.get("high_class", {}).get("ece")
                high_brier = calibration.get("high_class", {}).get("brier")
                p_at_10 = screening.get("precision_at_k", {}).get("10")
                ef_at_1 = screening.get("ef_at_percent", {}).get("1")
                ap = screening.get("pr_auc", {}).get("ap")
            except Exception:
                pass

            cal_str = (
                f"HighCal(ECE={high_ece:.4f},Brier={high_brier:.4f})"
                if isinstance(high_ece, (float, int)) and isinstance(high_brier, (float, int))
                else "HighCal(ECE=NA,Brier=NA)"
            )
            p10_str = f"P@10={p_at_10:.4f}" if isinstance(p_at_10, (float, int)) else "P@10=NA"
            ef1_str = f"EF@1%={ef_at_1:.2f}" if isinstance(ef_at_1, (float, int)) else "EF@1%=NA"
            ap_str = f"AP={ap:.4f}" if isinstance(ap, (float, int)) else "AP=NA"
            logging.info(
                f"Epoch [{epoch+1}/{num_epochs}] | LR: {current_lr:.2e} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Accuracy: {accuracy:.4f} | Balanced Acc: {balanced_acc:.4f} | MCC: {mcc:.4f} | "
                f"Precision: {precision:.4f} | Recall: {recall:.4f} | Macro-F1: {f1:.4f} | "
                f"{cal_str} | {ap_str} | {p10_str} | {ef1_str}"
            )

            # Early Stopping Check
            if f1 > max_val_f1:
                max_val_f1 = f1
                patience_counter = 0
                # Deep copy snapshot to avoid later mutation; store on CPU for safe serialization.
                best_model_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                best_epoch = epoch + 1
                best_val_loss = val_loss
                best_accuracy = accuracy
                best_balanced_acc = balanced_acc
                best_mcc = mcc
                best_precision = precision
                best_recall = recall
                best_per_class_metrics = per_class_metrics
                best_confusion_matrix = cm
                best_confusion_labels = cm_labels
                best_calibration = calibration
                best_screening = screening
                best_inference = inference

                if isinstance(best_per_class_metrics, dict):
                    for cls in sorted(best_per_class_metrics.keys()):
                        m = best_per_class_metrics[cls]
                        cls_name = m.get("name", label_id_to_name(cls)) if isinstance(m, dict) else label_id_to_name(cls)
                        logging.info(
                            f"Per-class (label={cls} name={cls_name}) | Precision: {m['precision']:.4f} | "
                            f"Recall: {m['recall']:.4f} | F1: {m['f1']:.4f} | Support: {m['support']}"
                        )

                high_class_label = int(HIGH_ACTIVITY_LABEL)
                if isinstance(best_per_class_metrics, dict) and high_class_label in best_per_class_metrics:
                    hc = best_per_class_metrics[high_class_label]
                    hc_name = hc.get("name", label_id_to_name(high_class_label)) if isinstance(hc, dict) else label_id_to_name(high_class_label)
                    logging.info(
                        f"Focus-class (label={high_class_label} name={hc_name}) | Precision: {hc['precision']:.4f} | "
                        f"Recall: {hc['recall']:.4f} | F1: {hc['f1']:.4f} | Support: {hc['support']}"
                    )
                logging.debug(f"New best F1 score ({max_val_f1:.4f}) at epoch {best_epoch}. Saving model state.")
            else:
                patience_counter += 1
                logging.debug(f"F1 did not improve. Early stopping counter: {patience_counter}/{early_stopping_patience}")
                if patience_counter >= early_stopping_patience:
                    logging.info(f"Early stopping triggered at epoch {epoch+1} due to lack of F1 improvement for {early_stopping_patience} epochs.")
                    break # Stop training this fold

            # Step the scheduler after validation
            scheduler.step()

        # --- End of Epoch Loop for Fold --- 

        all_fold_best_f1.append(max_val_f1 if best_epoch != -1 else -1)
        all_fold_best_balanced_acc.append(best_balanced_acc if best_epoch != -1 else None)
        all_fold_best_mcc.append(best_mcc if best_epoch != -1 else None)
        logging.info(f"Fold {fold+1} completed. Best Validation F1 Score = {max_val_f1:.4f} at epoch {best_epoch}")
        if best_confusion_matrix is not None and best_confusion_labels is not None:
            logging.info(
                f"Best confusion matrix labels (fold {fold+1}): {best_confusion_labels} -> "
                f"{[label_id_to_name(l) for l in best_confusion_labels]}"
            )
            logging.info(f"Best confusion matrix (fold {fold+1}):\n{best_confusion_matrix}")

        # Collect fold row for 5-fold summary CSV (best epoch for this fold)
        try:
            fold_row = make_fold_row(
                run_identifier=run_identifier,
                fold=int(fold + 1),
                best_epoch=int(best_epoch) if best_epoch != -1 else None,
                best_val_loss=best_val_loss,
                best_accuracy=best_accuracy,
                best_balanced_acc=best_balanced_acc,
                best_mcc=best_mcc,
                best_precision=best_precision,
                best_recall=best_recall,
                best_macro_f1=float(max_val_f1) if best_epoch != -1 else None,
                per_class_metrics=best_per_class_metrics,
                confusion_matrix=best_confusion_matrix,
                confusion_labels=best_confusion_labels,
                calibration=best_calibration,
                screening=best_screening,
                inference=best_inference,
                param_total=param_total,
                param_trainable=param_trainable,
            )
            all_fold_rows.append(fold_row)
        except Exception as e:
            logging.error(f"Failed to build summary row for fold {fold+1}: {e}")

        # Save the best model for this fold
        if best_model_state is not None:
            model_save_path = os.path.join(save_dir, f"{run_identifier}_fold{fold+1}_best_f1_epoch{best_epoch}.pth")
            try:
                torch.save(best_model_state, model_save_path)
                logging.info(f"Best model for fold {fold+1} saved to {model_save_path}")
            except Exception as e:
                logging.error(f"Failed to save best model for fold {fold+1} to {model_save_path}: {e}")

            # Save metrics snapshot (best epoch for this fold)
            metrics_save_path = os.path.join(
                save_dir,
                f"{run_identifier}_fold{fold+1}_best_f1_epoch{best_epoch}_metrics.json",
            )
            try:
                payload = {
                    "fold": int(fold + 1),
                    "best_epoch": int(best_epoch),
                    "label_names": {
                        **{str(i): label_id_to_name(i) for i in range(int(output_dim))},
                        "-1": label_id_to_name(-1),
                    },
                    "val_loss": float(best_val_loss) if best_val_loss is not None else None,
                    "accuracy": float(best_accuracy) if best_accuracy is not None else None,
                    "balanced_accuracy": float(best_balanced_acc) if best_balanced_acc is not None else None,
                    "mcc": float(best_mcc) if best_mcc is not None else None,
                    "macro_precision": float(best_precision) if best_precision is not None else None,
                    "macro_recall": float(best_recall) if best_recall is not None else None,
                    "macro_f1": float(max_val_f1) if best_epoch != -1 else None,
                    "model_param_count_total": int(param_total) if param_total is not None else None,
                    "model_param_count_trainable": int(param_trainable) if param_trainable is not None else None,
                    "inference": best_inference,
                    "per_class": best_per_class_metrics,
                    "confusion_matrix": {
                        "labels": [int(l) for l in best_confusion_labels]
                        if best_confusion_labels is not None
                        else None,
                        "label_names": [label_id_to_name(l) for l in best_confusion_labels]
                        if best_confusion_labels is not None
                        else None,
                        "matrix": best_confusion_matrix.tolist()
                        if best_confusion_matrix is not None
                        else None,
                    },
                    "calibration": best_calibration,
                    "screening": best_screening,
                }
                with open(metrics_save_path, "w") as f:
                    json.dump(payload, f, indent=2)
                logging.info(
                    f"Best metrics for fold {fold+1} saved to {metrics_save_path}"
                )
            except Exception as e:
                logging.error(f"Failed to save metrics for fold {fold+1} to {metrics_save_path}: {e}")

            # Save confusion matrix plot (best epoch for this fold)
            if best_confusion_matrix is not None and best_confusion_labels is not None:
                cm_plot_path = os.path.join(
                    save_dir,
                    f"{run_identifier}_fold{fold+1}_best_f1_epoch{best_epoch}_confusion_matrix.png",
                )
                plot_confusion_matrix(
                    best_confusion_matrix,
                    best_confusion_labels,
                    cm_plot_path,
                    title=f"Confusion Matrix - {model_name} - Fold {fold+1} (Best epoch {best_epoch})",
                )

            # Save reliability plots (best epoch for this fold)
            if isinstance(best_calibration, dict):
                mc = best_calibration.get("multiclass", {})
                hc = best_calibration.get("high_class", {})

                if isinstance(mc, dict):
                    mc_path = os.path.join(
                        save_dir,
                        f"{run_identifier}_fold{fold+1}_best_f1_epoch{best_epoch}_reliability_multiclass.png",
                    )
                    mc_ece = mc.get("ece")
                    mc_brier = mc.get("brier")
                    mc_title = (
                        f"Reliability (Multiclass) - {model_name} - Fold {fold+1} (Best epoch {best_epoch})\n"
                        f"ECE={mc_ece:.4f} Brier={mc_brier:.4f}"
                        if isinstance(mc_ece, (float, int)) and isinstance(mc_brier, (float, int))
                        else f"Reliability (Multiclass) - {model_name} - Fold {fold+1} (Best epoch {best_epoch})"
                    )
                    plot_reliability_diagram(
                        mc.get("bin_confidence", []),
                        mc.get("bin_accuracy", []),
                        mc.get("bin_count", []),
                        mc_path,
                        title=mc_title,
                        y_label="Accuracy",
                    )

                if isinstance(hc, dict):
                    hc_path = os.path.join(
                        save_dir,
                        f"{run_identifier}_fold{fold+1}_best_f1_epoch{best_epoch}_reliability_high_class.png",
                    )
                    hc_ece = hc.get("ece")
                    hc_brier = hc.get("brier")
                    pos_label = hc.get("positive_label")
                    pos_label_name = (
                        hc.get("positive_label_name", label_id_to_name(pos_label))
                        if pos_label is not None
                        else "unknown"
                    )
                    hc_title = (
                        f"Reliability (Focus={pos_label_name}, label={pos_label}) - {model_name} - Fold {fold+1} (Best epoch {best_epoch})\n"
                        f"ECE={hc_ece:.4f} Brier={hc_brier:.4f}"
                        if isinstance(hc_ece, (float, int))
                        and isinstance(hc_brier, (float, int))
                        else f"Reliability (Focus class) - {model_name} - Fold {fold+1} (Best epoch {best_epoch})"
                    )
                    plot_reliability_diagram(
                        hc.get("bin_confidence", []),
                        hc.get("bin_accuracy", []),
                        hc.get("bin_count", []),
                        hc_path,
                        title=hc_title,
                        y_label="Fraction positive",
                    )
        else:
            logging.warning(f"No improvement found for fold {fold+1}, no best model saved.")

        train_losses_all_folds.append(fold_train_losses)
        val_losses_all_folds.append(fold_val_losses)
        val_f1_scores_all_folds.append(fold_val_f1_scores)

        # Plot metrics for this fold using the helper function
        plot_fold_metrics(fold, fold_train_losses, fold_val_losses, fold_val_f1_scores, model_name, save_dir, run_identifier)

        logging.info(f"---------------- End of Fold {fold+1} ----------------")


    # Aggregate results and plot average metrics using the helper function
    plot_average_metrics(k_folds, train_losses_all_folds, val_losses_all_folds, val_f1_scores_all_folds, model_name, save_dir, run_identifier)

    # Write 5-fold summary CSV (fold rows + mean/std).
    if all_fold_rows:
        try:
            mean_row, std_row = compute_mean_std_rows(all_fold_rows)
            summary_rows = list(all_fold_rows) + [mean_row, std_row]
            summary_csv_path = os.path.join(
                save_dir, f"{run_identifier}_cv{k_folds}_summary.csv"
            )
            write_csv(summary_csv_path, summary_rows)
            logging.info(f"CV summary CSV saved to {summary_csv_path}")
        except Exception as e:
            logging.error(f"Failed to write CV summary CSV: {e}")
    else:
        logging.warning("No fold summary rows collected; CV summary CSV not written.")

    # Log final results
    valid_f1_scores = [f1 for f1 in all_fold_best_f1 if f1 != -1]
    if valid_f1_scores:
        avg_best_f1 = np.mean(valid_f1_scores)
        std_best_f1 = np.std(valid_f1_scores)
        logging.info(f"Average Best Validation F1 Score across {len(valid_f1_scores)} folds: {avg_best_f1:.4f} +/- {std_best_f1:.4f}")
    else:
        logging.warning("No valid best F1 scores recorded across folds.")

    valid_bal_acc = [x for x in all_fold_best_balanced_acc if x is not None and not np.isnan(x)]
    if valid_bal_acc:
        avg_bal_acc = np.mean(valid_bal_acc)
        std_bal_acc = np.std(valid_bal_acc)
        logging.info(
            f"Average Best Balanced Accuracy across {len(valid_bal_acc)} folds: {avg_bal_acc:.4f} +/- {std_bal_acc:.4f}"
        )
    else:
        logging.warning("No valid best balanced accuracy recorded across folds.")

    valid_mcc = [x for x in all_fold_best_mcc if x is not None and not np.isnan(x)]
    if valid_mcc:
        avg_mcc = np.mean(valid_mcc)
        std_mcc = np.std(valid_mcc)
        logging.info(
            f"Average Best MCC across {len(valid_mcc)} folds: {avg_mcc:.4f} +/- {std_mcc:.4f}"
        )
    else:
        logging.warning("No valid best MCC recorded across folds.")


if __name__ == "__main__":
    main()
