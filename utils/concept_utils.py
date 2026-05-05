import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from transformers import Trainer
from typing import Dict


def compute_concept_metrics(logits: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    logits_flat = logits.reshape(-1, logits.shape[-1])
    labels_flat = labels.reshape(-1, labels.shape[-1]).astype(np.int32)

    valid_mask = (labels_flat != -100).all(axis=1)
    logits_flat = logits_flat[valid_mask]
    labels_flat = labels_flat[valid_mask]

    probs = 1 / (1 + np.exp(-logits_flat))
    predictions = (probs > 0.5).astype(np.int32)

    macro_f1 = f1_score(labels_flat, predictions, average="macro", zero_division=0)
    micro_f1 = f1_score(labels_flat, predictions, average="micro", zero_division=0)
    macro_precision = precision_score(labels_flat, predictions, average="macro", zero_division=0)
    macro_recall = recall_score(labels_flat, predictions, average="macro", zero_division=0)
    vect_accuracy = accuracy_score(labels_flat, predictions)
    element_wise_acc = (predictions == labels_flat).mean()

    return {
        "concept_macro_f1": float(macro_f1),
        "concept_micro_f1": float(micro_f1),
        "concept_macro_precision": float(macro_precision),
        "concept_macro_recall": float(macro_recall),
        "concept_vect_accuracy": float(vect_accuracy),
        "concept_element_wise_accuracy": float(element_wise_acc),
    }


def evaluate_concept_layer(model: torch.nn.Module, dataset, data_collator, batch_size: int, device: torch.device,) -> Dict[str, float]:
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
    return compute_concept_metrics(logits, labels)