"""Layer-wise probing script for wav2vec2 phonetic concept representations."""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import yaml
from datasets import DatasetDict, load_from_disk
from sklearn.metrics import f1_score
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

from data_prep import (
    BINARY_FEATURE_DIM,
    extract_framewise_binfeatures,
    load_timit_dataset,
    prepare_audio_samples,
)

PROBE_CONFIG_PATH = Path("probe_config.yml")


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe wav2vec2 layers for articulatory features")
    parser.add_argument("--probe_config", type=Path, default=PROBE_CONFIG_PATH, help="Path to probe config YAML")
    parser.add_argument(
        "--probe_layer_idx",
        type=int,
        default=None,
        help="Transformer layer index to probe/fine-tune (0-based, wav2vec2-base: 0..11).",
    )
    return parser.parse_args()


@dataclass
class ArticulatoryFeatureDataCollator:
    """Pad raw waveform inputs and frame-level binary labels."""

    label_dim: int
    padding_value: float = 0.0

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        batch_size = len(features)

        input_lengths = [len(feature["input_values"]) for feature in features]
        max_input_length = max(input_lengths)

        input_values = torch.full((batch_size, max_input_length), self.padding_value, dtype=torch.float32)
        attention_mask = torch.zeros((batch_size, max_input_length), dtype=torch.long)

        for idx, feature in enumerate(features):
            values = torch.tensor(feature["input_values"], dtype=torch.float32)
            seq_len = values.size(0)
            input_values[idx, :seq_len] = values
            attention_mask[idx, :seq_len] = 1

        label_lengths = [len(feature["labels"]) for feature in features]
        max_label_length = max(label_lengths)

        labels = torch.full((batch_size, max_label_length, self.label_dim), -100.0, dtype=torch.float32)
        for idx, feature in enumerate(features):
            label_tensor = torch.tensor(feature["labels"], dtype=torch.float32)
            seq_len = label_tensor.size(0)
            labels[idx, :seq_len, :] = label_tensor

        return {
            "input_values": input_values,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def load_probe_config(path: Path = PROBE_CONFIG_PATH) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Probe config file not found at '{path}'. "
            "Create it (e.g., probe_config.yml) before running probe.py."
        )
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_dataset_cl(config: Dict[str, Any], feature_extractor: Wav2Vec2FeatureExtractor) -> DatasetDict:
    dataset_path = config.get("processed_dataset_path", "./datasets/processed_timit_dataset-conceptprobe")
    dataset_path = Path(dataset_path)
    os.makedirs(dataset_path.parent, exist_ok=True)

    if config.get("load_processed_dataset", True) and dataset_path.exists():
        print(f"Loading processed TIMIT dataset from {dataset_path}...")
        dataset = load_from_disk(str(dataset_path))
    else:
        print("Loading and processing TIMIT dataset...")
        dataset = load_timit_dataset(config["sample_validation_set"])

        dataset = dataset.map(
            extract_framewise_binfeatures,
            desc="Extracting binary articulatory features for frames",
            load_from_cache_file=False,
        )

        dataset = dataset.map(
            prepare_audio_samples,
            batched=True,
            fn_kwargs={"feature_extractor": feature_extractor},
            desc="Extracting wav2vec2 inputs",
            load_from_cache_file=False,
        )

        keep_columns = {"input_values", "labels"}
        for split in dataset.keys():
            remove_columns = [column for column in dataset[split].column_names if column not in keep_columns]
            if remove_columns:
                dataset[split] = dataset[split].remove_columns(remove_columns)

        print(f"Saving processed dataset to: {dataset_path}")
        dataset.save_to_disk(str(dataset_path))

    return dataset


def compute_output_mask(
    wav2vec2_model: Wav2Vec2Model,
    attention_mask: torch.Tensor,
    target_length: int,
    device: torch.device,
) -> torch.Tensor:
    output_lengths = wav2vec2_model._get_feat_extract_output_lengths(attention_mask.sum(dim=1))  # type: ignore[attr-defined]
    output_lengths = output_lengths.to(device)
    arange = torch.arange(target_length, device=device)
    return arange.unsqueeze(0) < output_lengths.unsqueeze(1)


def masked_bce_loss(logits: torch.Tensor, labels: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
    valid_frame_mask = frame_mask & (labels != -100).all(dim=-1)
    safe_labels = torch.where(labels == -100, torch.zeros_like(labels), labels)

    raw_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, safe_labels, reduction="none")
    frame_loss = raw_loss.mean(dim=-1)

    denom = valid_frame_mask.float().sum().clamp(min=1.0)
    return (frame_loss * valid_frame_mask.float()).sum() / denom


def evaluate_probes(wav2vec2_model: Wav2Vec2Model, probes: nn.ModuleList, dataloader: DataLoader, device: torch.device, max_eval_batches: int | None = None,) -> List[Dict[str, float]]:
    wav2vec2_model.eval()
    for probe in probes:
        probe.eval()

    logits_store: List[List[np.ndarray]] = [[] for _ in range(len(probes))]
    labels_store: List[List[np.ndarray]] = [[] for _ in range(len(probes))]

    with torch.no_grad():
        for step, batch in enumerate(tqdm(dataloader, desc="Evaluating probes", leave=False)):
            if max_eval_batches is not None and step >= max_eval_batches:
                break

            input_values = batch["input_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = wav2vec2_model(
                input_values=input_values,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_states = outputs.hidden_states
            max_time_steps = hidden_states[0].size(1)
            frame_mask = compute_output_mask(wav2vec2_model, attention_mask, max_time_steps, device)

            for layer_idx, probe in enumerate(probes):
                layer_hidden = hidden_states[layer_idx]
                usable_length = min(layer_hidden.size(1), labels.size(1))

                layer_logits = probe(layer_hidden[:, :usable_length, :])
                layer_labels = labels[:, :usable_length, :]
                layer_mask = frame_mask[:, :usable_length]
                valid_mask = layer_mask & (layer_labels != -100).all(dim=-1)

                if valid_mask.any():
                    valid_logits = layer_logits[valid_mask].cpu().numpy()
                    valid_labels = layer_labels[valid_mask].cpu().numpy().astype(np.int32)
                    logits_store[layer_idx].append(valid_logits)
                    labels_store[layer_idx].append(valid_labels)

    results: List[Dict[str, float]] = []
    for layer_idx in range(len(probes)):
        if not logits_store[layer_idx]:
            results.append(
                {
                    "layer": layer_idx,
                    "num_frames": 0,
                    "vect_accuracy": float("nan"),
                    "element_wise_accuracy": float("nan"),
                    "macro_f1": float("nan"),
                    "micro_f1": float("nan"),
                }
            )
            continue

        all_logits = np.concatenate(logits_store[layer_idx], axis=0)
        all_labels = np.concatenate(labels_store[layer_idx], axis=0)

        probs = 1.0 / (1.0 + np.exp(-all_logits))
        predictions = (probs > 0.5).astype(np.int32)

        vect_accuracy = np.all(predictions == all_labels, axis=1).mean().item()
        element_wise_accuracy = (predictions == all_labels).mean().item()
        macro_f1 = f1_score(all_labels, predictions, average="macro", zero_division=0)
        micro_f1 = f1_score(all_labels, predictions, average="micro", zero_division=0)

        results.append(
            {
                "layer": layer_idx,
                "num_frames": int(all_labels.shape[0]),
                "vect_accuracy": float(vect_accuracy),
                "element_wise_accuracy": float(element_wise_accuracy),
                "macro_f1": float(macro_f1),
                "micro_f1": float(micro_f1),
            }
        )

    return results


def train_layerwise_probes(wav2vec2_model: Wav2Vec2Model, probes: nn.ModuleList, dataloader: DataLoader, device: torch.device, epochs: int, optimizer: torch.optim.Optimizer, max_train_batches: int | None = None) -> None:
    wav2vec2_model.eval()
    for probe in probes:
        probe.train()

    for epoch in range(epochs):
        running_loss = 0.0
        steps = 0

        for step, batch in enumerate(tqdm(dataloader, desc=f"Probe training epoch {epoch + 1}/{epochs}", leave=False)):
            if max_train_batches is not None and step >= max_train_batches:
                break

            input_values = batch["input_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.no_grad():
                outputs = wav2vec2_model(
                    input_values=input_values,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )

            hidden_states = outputs.hidden_states
            max_time_steps = hidden_states[0].size(1)
            frame_mask = compute_output_mask(wav2vec2_model, attention_mask, max_time_steps, device)

            layer_losses: List[torch.Tensor] = []
            for layer_idx, probe in enumerate(probes):
                layer_hidden = hidden_states[layer_idx]
                usable_length = min(layer_hidden.size(1), labels.size(1))

                layer_logits = probe(layer_hidden[:, :usable_length, :])
                layer_labels = labels[:, :usable_length, :]
                layer_mask = frame_mask[:, :usable_length]

                layer_loss = masked_bce_loss(layer_logits, layer_labels, layer_mask)
                layer_losses.append(layer_loss)

            loss = torch.stack(layer_losses).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            steps += 1

        avg_loss = running_loss / max(steps, 1)
        print(f"Epoch {epoch + 1}/{epochs} - mean probe loss: {avg_loss:.6f}")


def _set_preceding_layers_trainable(wav2vec2_model: Wav2Vec2Model, transformer_layer_idx: int) -> int:
    """Unfreeze transformer layers up to and including the probed transformer layer.

    hidden_states index mapping:
    - 0: feature projection output (before transformer layers)
    - 1..N: outputs after transformer layers 0..N-1
    """
    wav2vec2_model.requires_grad_(False)

    total_encoder_layers = len(wav2vec2_model.encoder.layers)
    num_to_unfreeze = min(transformer_layer_idx + 1, total_encoder_layers)

    for layer_idx in range(num_to_unfreeze):
        for parameter in wav2vec2_model.encoder.layers[layer_idx].parameters():
            parameter.requires_grad = True

    return num_to_unfreeze


def train_and_evaluate_finetuned_probe(
    model_checkpoint: str,
    probe_layer_idx: int,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    device: torch.device,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    max_train_batches: int | None = None,
    max_eval_batches: int | None = None,
) -> Dict[str, float]:
    wav2vec2_model = Wav2Vec2Model.from_pretrained(model_checkpoint).to(device)
    probe = nn.Linear(wav2vec2_model.config.hidden_size, BINARY_FEATURE_DIM).to(device)

    hidden_state_idx = probe_layer_idx + 1
    num_unfrozen_layers = _set_preceding_layers_trainable(wav2vec2_model, probe_layer_idx)

    trainable_params = [param for param in wav2vec2_model.parameters() if param.requires_grad]
    trainable_params.extend(list(probe.parameters()))

    optimizer = AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)

    for epoch in range(epochs):
        running_loss = 0.0
        steps = 0

        for step, batch in enumerate(tqdm(train_loader, desc=f"FT layer {probe_layer_idx} epoch {epoch + 1}/{epochs}", leave=False)):
            if max_train_batches is not None and step >= max_train_batches:
                break

            input_values = batch["input_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = wav2vec2_model(
                input_values=input_values,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

            max_time_steps = outputs.hidden_states[0].size(1)
            frame_mask = compute_output_mask(wav2vec2_model, attention_mask, max_time_steps, device)

            selected_hidden = outputs.hidden_states[hidden_state_idx]
            usable_length = min(selected_hidden.size(1), labels.size(1))

            logits = probe(selected_hidden[:, :usable_length, :])
            curr_labels = labels[:, :usable_length, :]
            curr_mask = frame_mask[:, :usable_length]

            loss = masked_bce_loss(logits, curr_labels, curr_mask)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            
            # torch.nn.utils.clip_grad_norm_(trainable_params, 1.0) 
            
            optimizer.step()

            running_loss += loss.item()
            steps += 1

        avg_loss = running_loss / max(steps, 1)
        print(f"FT layer {probe_layer_idx} | epoch {epoch + 1}/{epochs} - mean loss: {avg_loss:.6f}")

    wav2vec2_model.eval()
    probe.eval()

    logits_store: List[np.ndarray] = []
    labels_store: List[np.ndarray] = []

    with torch.no_grad():
        for step, batch in enumerate(tqdm(eval_loader, desc=f"Evaluating FT layer {probe_layer_idx}", leave=False)):
            if max_eval_batches is not None and step >= max_eval_batches:
                break

            input_values = batch["input_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = wav2vec2_model(
                input_values=input_values,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

            hidden_states = outputs.hidden_states
            selected_hidden = hidden_states[hidden_state_idx]
            usable_length = min(selected_hidden.size(1), labels.size(1))

            selected_logits = probe(selected_hidden[:, :usable_length, :])
            selected_labels = labels[:, :usable_length, :]

            frame_mask = compute_output_mask(wav2vec2_model, attention_mask, hidden_states[0].size(1), device)
            valid_mask = frame_mask[:, :usable_length] & (selected_labels != -100).all(dim=-1)

            if valid_mask.any():
                logits_store.append(selected_logits[valid_mask].cpu().numpy())
                labels_store.append(selected_labels[valid_mask].cpu().numpy().astype(np.int32))

    if not logits_store:
        return {
            "layer": probe_layer_idx,
            "num_unfrozen_preceding_layers": num_unfrozen_layers,
            "num_frames": 0,
            "vect_accuracy": float("nan"),
            "element_wise_accuracy": float("nan"),
            "macro_f1": float("nan"),
            "micro_f1": float("nan"),
        }

    all_logits = np.concatenate(logits_store, axis=0)
    all_labels = np.concatenate(labels_store, axis=0)

    probabilities = 1.0 / (1.0 + np.exp(-all_logits))
    predictions = (probabilities > 0.5).astype(np.int32)

    return {
        "layer": probe_layer_idx,
        "hidden_state_index": hidden_state_idx,
        "num_unfrozen_preceding_layers": num_unfrozen_layers,
        "num_frames": int(all_labels.shape[0]),
        "vect_accuracy": float(np.all(predictions == all_labels, axis=1).mean().item()),
        "element_wise_accuracy": float((predictions == all_labels).mean().item()),
        "macro_f1": float(f1_score(all_labels, predictions, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(all_labels, predictions, average="micro", zero_division=0)),
    }


def print_results_table(results: List[Dict[str, float]]) -> None:
    print("\n" + "=" * 95)
    print("Layer-wise probe results (higher is better)")
    print("=" * 95)
    print(
        f"{'layer':>7} | {'frames':>9} | {'vect_acc':>10} | {'elem_acc':>10} | {'macro_f1':>10} | {'micro_f1':>10}"
    )
    print("-" * 95)

    for item in results:
        print(
            f"{item['layer']:>7} | {item['num_frames']:>9} | {item['vect_accuracy']:>10.4f} | "
            f"{item['element_wise_accuracy']:>10.4f} | {item['macro_f1']:>10.4f} | {item['micro_f1']:>10.4f}"
        )

    best_vect = max(results, key=lambda x: x["vect_accuracy"])
    best_macro_f1 = max(results, key=lambda x: x["macro_f1"])
    print("-" * 95)
    print(
        f"Best vector accuracy: layer {best_vect['layer']} ({best_vect['vect_accuracy']:.4f}) | "
        f"Best macro F1: layer {best_macro_f1['layer']} ({best_macro_f1['macro_f1']:.4f})"
    )
    print("=" * 95 + "\n")



    


if __name__ == "__main__":

    cli_args = parse_cli_args()
    probe_config = load_probe_config(cli_args.probe_config)
    set_seed(int(probe_config.get("seed", 42)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_checkpoint = probe_config["model_checkpoint"]

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_checkpoint)
    dataset = prepare_dataset_cl(probe_config, feature_extractor)

    batch_size = probe_config["batch_size"]

    collator = ArticulatoryFeatureDataCollator(label_dim=BINARY_FEATURE_DIM)

    train_loader = DataLoader(
        dataset["train"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=probe_config["num_workers"]
    )

    eval_loader = DataLoader(
        dataset["test"],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=probe_config["num_workers"]
    )

    wav2vec2_model = Wav2Vec2Model.from_pretrained(model_checkpoint)
    wav2vec2_model.to(device)
    wav2vec2_model.requires_grad_(False)
    wav2vec2_model.eval()

    num_transformer_layers = wav2vec2_model.config.num_hidden_layers
    num_probe_layers = num_transformer_layers + 1
    selected_layer_idx = cli_args.probe_layer_idx
    if selected_layer_idx is not None:
        selected_layer_idx = int(selected_layer_idx)
        if not (0 <= selected_layer_idx < num_transformer_layers):
            raise ValueError(
                f"probe_layer_idx={selected_layer_idx} is out of range. "
                f"Expected an integer in [0, {num_transformer_layers - 1}]"
            )

    hidden_size = wav2vec2_model.config.hidden_size
    probes = nn.ModuleList([nn.Linear(hidden_size, BINARY_FEATURE_DIM) for _ in range(num_probe_layers)]).to(device)

    optimizer = AdamW(
        probes.parameters(),
        lr=float(probe_config["lr"]),
        weight_decay=float(probe_config["weight_decay"]),
    )

    if probe_config["fine_tune_preceding_layers"]:
        if selected_layer_idx is None:
            raise ValueError(
                "fine_tune_preceding_layers=true requires a single layer index. "
                "Set probe_layer_idx in probe_config.yml or pass --probe_layer_idx."
            )

        print(
            f"Fine-tuning preceding transformer layers is enabled. "
            f"Training a probe on layer {selected_layer_idx} and fine-tunining preceding wav2vec2 layers for {probe_config['epochs']} epochs."
        )

        layer_result = train_and_evaluate_finetuned_probe(
            model_checkpoint=model_checkpoint,
            probe_layer_idx=selected_layer_idx,
            train_loader=train_loader,
            eval_loader=eval_loader,
            device=device,
            learning_rate=probe_config["lr"],
            weight_decay=probe_config["weight_decay"],
            epochs=probe_config["epochs"]
        )
        results = [layer_result]
        print(f"Completed fine-tuning probe for requested layer index: {selected_layer_idx}")
    else:
        if selected_layer_idx is not None:
            raise ValueError(
                "probe_layer_idx is only supported when fine_tune_preceding_layers=true. "
                "Set fine_tune_preceding_layers to true or remove probe_layer_idx."
            )

        train_layerwise_probes(
            wav2vec2_model=wav2vec2_model,
            probes=probes,
            dataloader=train_loader,
            device=device,
            epochs=int(probe_config["epochs"]),
            optimizer=optimizer
        )

        results = evaluate_probes(
            wav2vec2_model=wav2vec2_model,
            probes=probes,
            dataloader=eval_loader,
            device=device
        )

    print_results_table(results)

    output_json = Path(probe_config.get("output_json", "outputs/probe_results.json"))
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as fp:
        json.dump(
            {
                "checkpoint": model_checkpoint,
                "train_split": "train",
                "eval_split": "test",
                "epochs": int(probe_config.get("epochs", 5)),
                "batch_size": batch_size,
                "results": results,
            },
            fp,
            indent=2,
        )
    print(f"Saved probe results to: {output_json}")
