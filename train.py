"""Training entry point for wav2vec2 articulatory feature prediction."""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List
from dotenv import load_dotenv

import torch
import yaml
import wandb
import os
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from transformers import (
    Trainer,
    TrainingArguments,
    Wav2Vec2FeatureExtractor
)

from datasets import DatasetDict, load_from_disk

from callbacks import UnfreezingCallback, unfreeze_encoder_layers
from data_prep import (
    BINARY_FEATURE_DIM,
    TARGET_SAMPLING_RATE,
    load_timit_dataset,
    prepare_dataset,
    compute_inputs,
    FEATURE_GROUPS_LABELS
)
from model import Wav2Vec2ForArticulatoryFeatures

CONFIG_PATH = Path("config.yml")


@dataclass      # dataclass decorator allows for cleaner class definition without args and init
class ArticulatoryFeatureDataCollator:
    """Data Collator is used to pad the input values and labels to be the same length in the batch,
       and make the corresponding attention mask. This class is used as input argument for the Trainer,
       and can be used on the fly. Since audio files are highly variable in length, it is more efficient
       to perform padding at runtime for the batch, rather than for the entire dataset beforehand."""

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


def load_training_config(path: Path = CONFIG_PATH) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)
    

def compute_metrics(eval_pred):
    logits = eval_pred.predictions
    labels = eval_pred.label_ids

    logits_flat = logits.reshape(-1, logits.shape[-1])                      # Reshape to (total_frames, 29)
    labels_flat = labels.reshape(-1, labels.shape[-1]).astype(np.int32)
    
    # Only calculate metrics on 
    valid_mask = (labels_flat != -100).all(axis=1)
    logits_flat = logits_flat[valid_mask]
    labels_flat = labels_flat[valid_mask]
    
    # Convert logits to probabilities, then to 0/1 predictions
    probs = 1 / (1 + np.exp(-logits_flat))
    predictions = (probs > 0.5).astype(np.int32)
    
    # Macro F1: Averages the F1 of each of the 29 features
    macro_f1 = f1_score(labels_flat, predictions, average='macro', zero_division=0)
    micro_f1 = f1_score(labels_flat, predictions, average='micro', zero_division=0)

    # Precision & Recall (Macro)
    macro_precision = precision_score(labels_flat, predictions, average='macro', zero_division=0)
    macro_recall = recall_score(labels_flat, predictions, average='macro', zero_division=0)
    
    # Subset Accuracy: Strict (All 29 features must match)
    subset_accuracy = accuracy_score(labels_flat, predictions)
    
    # Element-wise Accuracy: Treat every one of the 29 decisions independently
    element_wise_acc = (predictions == labels_flat).mean()

    return {
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "subset_accuracy": subset_accuracy,
        "element_wise_accuracy": element_wise_acc
    }

def print_dataset_statistics(dataset: DatasetDict):
    """Calculates and prints the distribution of binary features in the dataset."""
    # feature_labels = [label for group in FEATURE_GROUPS_LABELS for label in group.labels]
    feature_labels = [f"{group.name}_{label}" for group in FEATURE_GROUPS_LABELS for label in group.labels]
    print("feature labels: ", feature_labels)
    
    print("\n" + "="*80)
    print("DATASET FEATURE DISTRIBUTION")
    print("="*80)

    for split in dataset.keys():
        print(f"\n--- Split: {split} ---")
        split_data = dataset[split]
        
        # Initialize counters
        feature_counts = np.zeros(BINARY_FEATURE_DIM, dtype=np.int64)
        for item in split_data:
            
            labels = np.array(item['labels'])
            
            # Sum down the frame axis (axis 0)
            feature_counts += labels.sum(axis=0).astype(np.int64)

        split_counts = {label: int(feature_counts[i]) for i, label in enumerate(feature_labels)}
        print(split_counts)
            
    print("="*80 + "\n")

def print_per_feature_statistics(test_results):
    # Calculate per-feature accuracy
    test_logits = test_results.predictions
    test_labels = test_results.label_ids

    # Flatten batch and sequence dimensions
    flat_logits = test_logits.reshape(-1, test_logits.shape[-1])
    flat_labels = test_labels.reshape(-1, test_labels.shape[-1])

    # Filter out padded tokens (where labels are -100)
    valid_mask = (flat_labels != -100).all(axis=1)
    flat_logits = flat_logits[valid_mask]
    flat_labels = flat_labels[valid_mask]

    # Convert logits to binary predictions
    probs = 1 / (1 + np.exp(-flat_logits))
    predictions = (probs > 0.5).astype(int)

    # Calculate accuracy per feature (axis 0 is the sample dimension now)
    per_feature_accuracy = (predictions == flat_labels).mean(axis=0)
    
    # feature_labels = [label for group in FEATURE_GROUPS_LABELS for label in group.labels]
    feature_labels = [f"{group.name}_{label}" for group in FEATURE_GROUPS_LABELS for label in group.labels]
    
    assert len(per_feature_accuracy) == len(feature_labels), "Number of features in accuracy does not match number of feature labels."
    print("Len per_feature_accuracy:", len(per_feature_accuracy))
    print("Len feature_labels:", len(feature_labels))
    
    print("\n=== Per-Feature Accuracy ===")
    for i, acc in enumerate(per_feature_accuracy):
        print(f"Feature {i} ({feature_labels[i]}): {acc:.4f}")

    per_feature_acc_dict = {f"{feature_labels[i]}": float(per_feature_accuracy[i]) for i in range(len(feature_labels))}
    print(per_feature_acc_dict)

def train_preprocessing(config, feature_extractor):
    dataset_path = config.get("processed_dataset_path_cl", "./datasets/processed_timit_dataset-")
    os.makedirs(Path(dataset_path).parent, exist_ok=True)
    
    if config["load_processed_dataset"] and Path(dataset_path).exists():
        print(f"Loading processed TIMIT dataset from {dataset_path}...")
        dataset = load_from_disk(dataset_path)
    else:
        print("Loading and processing TIMIT dataset...")
        dataset = load_timit_dataset(config["sample_validation_set"], config.get("sample_validation_size", 0.10))

        # timit_subset = dataset["train"].select(range(200))
        # dataset = DatasetDict({"train": timit_subset})
        
        dataset = dataset.map(
            prepare_dataset,
            desc="Aligning articulatory features",
            load_from_cache_file=False
        )
        
        dataset = dataset.map(
            compute_inputs,
            batched=True,       # Defaults to standard batch size=1000 for dataset library
            fn_kwargs={"feature_extractor": feature_extractor},
            desc="Extracting wav2vec2 inputs",
            load_from_cache_file=False
        )

        keep_columns = {"input_values", "labels"}
        for split in dataset.keys():
            remove_columns = [
                column for column in dataset[split].column_names if column not in keep_columns
            ]
            if remove_columns:
                dataset[split] = dataset[split].remove_columns(remove_columns)
            
        print(f"Saving processed dataset to {dataset_path}...")
        dataset.save_to_disk(dataset_path)
    
    num_labels = BINARY_FEATURE_DIM
    model_checkpoint = config["model_checkpoint"]

    model = Wav2Vec2ForArticulatoryFeatures.from_pretrained(            # from_pretrained will load default config.json from HuggingFace
        model_checkpoint,
        num_labels=num_labels,                                          # num_labels is updated in the config
        use_safetensors=True
    )

    initial_unfreeze = config.get("use_initial_unfreeze", False)
    if initial_unfreeze:
        assert config.get("unfreeze_layers") is not None, "unfreeze_layers cannot be None if use_initial_unfreeze is True"
        unfreeze_layers = config.get("unfreeze_layers")
        for layer_idx in unfreeze_layers:
            assert 0 <= layer_idx < len(model.wav2vec2.encoder.layers), f"Layer index {layer_idx} is out of bounds for wav2vec2 encoder layers."
        unfreeze_encoder_layers(model, unfreeze_layers)
        print(f"Initially unfroze encoder layers {unfreeze_layers}.")
    else:
        print("Using Default: keeping all wav2vec2 encoder layers frozen at the start of training.")
    
    return dataset, model




if __name__ == "__main__":
    
    config = load_training_config()

    load_dotenv()
    api_key = os.getenv("WANDB_API_KEY")
    wandb.login(key=api_key)
    
    wandb_run = wandb.init(
        project=config["wandb_project"],
        name=config["run_name"],
        config=config,
    )
    
    model_checkpoint = config["model_checkpoint"]
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_checkpoint)
    
    dataset, model = train_preprocessing(config, feature_extractor)

    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        eval_strategy=config["eval_strategy"],
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        num_train_epochs=config["num_train_epochs"],
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        eval_steps=config["eval_steps"],
        warmup_steps=config["warmup_steps"],
        save_total_limit=config["save_total_limit"],
        fp16=config["use_fp16"],
        report_to="wandb"
    )

    num_labels = BINARY_FEATURE_DIM
    data_collator = ArticulatoryFeatureDataCollator(label_dim=num_labels)

    eval_split = "validation" if "validation" in dataset else "test"

    print(f"Evaluation split: {eval_split}")

    # Define callbacks list
    callbacks = []
    if config.get("use_callback_unfreeze", False):              # Defaults to False if not in config, meaning all wav2vec2 layers will be kept frozen
        callbacks.append(UnfreezingCallback(model, 
                                            thaw_step=config["thaw_step"], 
                                            unfreeze_layers=config.get("unfreeze_layers", None)))

    trainer = Trainer(      # Trainer handles device placement
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset[eval_split],
        data_collator=data_collator,
        tokenizer=feature_extractor,
        callbacks=callbacks,
        compute_metrics=compute_metrics
    )

    trainer.train()

    test_results = trainer.predict(dataset["test"])
    print(test_results.metrics)

    # print_per_feature_statistics(test_results)

    wandb_run.finish()