import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from transformers import Trainer

try:
    from data_prep import FEATURE_GROUPS_LABELS, GROUP_OFFSETS, GROUP_SIZES
except ImportError:  # pragma: no cover - allows standalone use
    FEATURE_GROUPS_LABELS = None
    GROUP_OFFSETS = None
    GROUP_SIZES = None


def _default_feature_names() -> Optional[List[str]]:
    if FEATURE_GROUPS_LABELS is None:
        return None

    names: List[str] = []
    for group in FEATURE_GROUPS_LABELS:
        for label in group.labels:
            names.append(f"{group.name}:{label}")
    return names


def compute_concept_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    feature_names: Optional[List[str]] = None,
    print_feature_counts: bool = False,
    print_six_ones_count: bool = False,
) -> Dict[str, float]:
    logits_flat = logits.reshape(-1, logits.shape[-1])
    labels_flat = labels.reshape(-1, labels.shape[-1]).astype(np.int32)    # Labels flat shape before filtering valid mask: (452288, 29)

    print("Labels flat before filtering:", labels_flat.shape) 

    valid_mask = (labels_flat != -100).all(axis=1)
    logits_flat = logits_flat[valid_mask]
    labels_flat = labels_flat[valid_mask]           # Shape: (258511, 29)

    probs = 1 / (1 + np.exp(-logits_flat))
    predictions = (probs > 0.5).astype(np.int32)

    if print_six_ones_count:
        ones_per_row = predictions.sum(axis=1)
        total = int(ones_per_row.shape[0])
        six_count = int((ones_per_row == 6).sum())
        pct = (six_count / total) if total else 0.0
        print(f"Predictions with exactly six ones: {six_count}/{total} ({pct:.2%})")

    macro_f1 = f1_score(labels_flat, predictions, average="macro", zero_division=0)
    micro_f1 = f1_score(labels_flat, predictions, average="micro", zero_division=0)
    macro_precision = precision_score(labels_flat, predictions, average="macro", zero_division=0)
    macro_recall = recall_score(labels_flat, predictions, average="macro", zero_division=0)
    per_feature_f1 = f1_score(labels_flat, predictions, average=None, zero_division=0)
    vect_accuracy = accuracy_score(labels_flat, predictions)
    element_wise_acc = (predictions == labels_flat).mean()

    positive_mask = labels_flat == 1
    correct_positive = ((predictions == 1) & positive_mask).sum(axis=0)
    total_positive = positive_mask.sum(axis=0)
    incorrect_positive = total_positive - correct_positive

    if feature_names is None:
        feature_names = _default_feature_names()
    if feature_names is None:
        feature_names = [f"feature_{idx}" for idx in range(labels_flat.shape[1])]
    if len(feature_names) != labels_flat.shape[1]:
        raise ValueError(
            "feature_names length does not match number of features: "
            f"{len(feature_names)} vs {labels_flat.shape[1]}"
        )

    feature_counts = {
        feature_names[idx]: {
            "correct": int(correct_positive[idx]),
            "incorrect": int(incorrect_positive[idx]),
            "f1": float(per_feature_f1[idx]),
        }
        for idx in range(labels_flat.shape[1])
    }

    if print_feature_counts:
        print("Per-feature correct/incorrect counts and F1:")
        for name, counts in feature_counts.items():
            print(
                f"  {name}: correct={counts['correct']}, "
                f"incorrect={counts['incorrect']}"
            )

        for name, counts in feature_counts.items():
            print(f"{name}: f1={counts['f1']:.4f}")

    return {
        "concept_macro_f1": float(macro_f1),
        "concept_micro_f1": float(micro_f1),
        "concept_macro_precision": float(macro_precision),
        "concept_macro_recall": float(macro_recall),
        "concept_vect_accuracy": float(vect_accuracy),
        "concept_element_wise_accuracy": float(element_wise_acc),
    }


def save_feature_group_confusions(
    logits: np.ndarray,
    labels: np.ndarray,
    output_dir: str | Path,
) -> None:
    if FEATURE_GROUPS_LABELS is None or GROUP_OFFSETS is None or GROUP_SIZES is None:
        raise ValueError("Feature group metadata is not available for confusion matrices.")

    logits_flat = logits.reshape(-1, logits.shape[-1])
    labels_flat = labels.reshape(-1, labels.shape[-1]).astype(np.int32)

    valid_mask = (labels_flat != -100).all(axis=1)
    logits_flat = logits_flat[valid_mask]
    labels_flat = labels_flat[valid_mask]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for group_idx, group in enumerate(FEATURE_GROUPS_LABELS):
        start = GROUP_OFFSETS[group_idx]
        end = start + GROUP_SIZES[group_idx]
        label_slice = labels_flat[:, start:end]
        logits_slice = logits_flat[:, start:end]

        valid_group_mask = label_slice.sum(axis=1) > 0
        if not np.any(valid_group_mask):
            matrix = [[0 for _ in range(GROUP_SIZES[group_idx])] for _ in range(GROUP_SIZES[group_idx])]
        else:
            true_idx = np.argmax(label_slice[valid_group_mask], axis=1)
            pred_idx = np.argmax(logits_slice[valid_group_mask], axis=1)
            size = GROUP_SIZES[group_idx]
            matrix = np.zeros((size, size), dtype=np.int64)
            np.add.at(matrix, (true_idx, pred_idx), 1)
            matrix = matrix.tolist()

        payload = {
            "group": group.name,
            "labels": list(group.labels),
            "matrix": matrix,
        }

        file_path = output_path / f"{group.name}_feature_confusion.json"
        with file_path.open("w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2)


def evaluate_concept_layer(
    model: torch.nn.Module,
    dataset,
    data_collator,
    batch_size: int,
    device: torch.device,
    feature_names: Optional[List[str]] = None,
    print_feature_counts: bool = True,
    confusion_output_dir: str | Path | None = None,
) -> Dict[str, float]:
    model.eval()

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
    )

    all_logits = []
    all_labels = []

    for batch in dataloader:
        input_values = batch["input_values"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        concept_labels = batch["concept_labels"].to(device)

        with torch.no_grad():
            outputs = model(
                input_values=input_values,
                attention_mask=attention_mask,
                concept_labels=concept_labels,
                task_labels=None,
                return_dict=True,
            )

        concept_logits = outputs["concept_logits"]
        time_dim = concept_logits.size(1)
        label_time_dim = concept_labels.size(1)
        usable_length = min(time_dim, label_time_dim)

        concept_logits = concept_logits[:, :usable_length, :]
        concept_labels = concept_labels[:, :usable_length, :]

        concept_logits_np = concept_logits.cpu().numpy()
        concept_labels_np = concept_labels.cpu().numpy()

        all_logits.append(concept_logits_np.reshape(-1, concept_logits_np.shape[-1]))
        all_labels.append(concept_labels_np.reshape(-1, concept_labels_np.shape[-1]))

    if not all_logits:
        return {}

    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    if confusion_output_dir is not None:
        save_feature_group_confusions(logits, labels, confusion_output_dir)
    return compute_concept_metrics(
        logits,
        labels,
        feature_names=feature_names,
        print_feature_counts=print_feature_counts,
        print_six_ones_count=True,
    )