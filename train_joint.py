"""Training entry point for joint concept + task optimization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import os

import numpy as np
import torch
import wandb
import yaml
from dotenv import load_dotenv
from datasets import load_from_disk
from jiwer import wer
from transformers import Trainer, TrainingArguments, Wav2Vec2FeatureExtractor

from data_prep import (
    BINARY_FEATURE_DIM,
    load_timit_dataset,
    extract_framewise_binfeatures,
    prepare_audio_samples,
    phoneme_token_to_id,
    format_for_joint,
)
from model import Wav2Vec2ForJointBottleneck

CONFIG_PATH = Path("config.yml")

CTC_BLANK_ID = 0
PHONEME_TOKEN_TO_ID = phoneme_token_to_id()
ID_TO_PHONEME = {idx: token for token, idx in PHONEME_TOKEN_TO_ID.items()}


@dataclass
class JointDataCollator:
    """Pad audio inputs, concept labels, and task labels for joint training."""

    concept_label_dim: int
    input_padding_value: float = 0.0
    label_padding_value: int = -100

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        batch_size = len(features)
        if batch_size == 0:
            raise ValueError("JointDataCollator received an empty batch")

        input_lengths = [len(feature["input_values"]) for feature in features]
        max_input_length = max(input_lengths)

        input_values = torch.full(
            (batch_size, max_input_length),
            self.input_padding_value,
            dtype=torch.float32,
        )
        attention_mask = torch.zeros((batch_size, max_input_length), dtype=torch.long)

        for idx, feature in enumerate(features):
            values = torch.tensor(feature["input_values"], dtype=torch.float32)
            seq_len = values.size(0)
            input_values[idx, :seq_len] = values
            attention_mask[idx, :seq_len] = 1

        concept_lengths = [len(feature["concept_labels"]) for feature in features]
        max_concept_len = max(concept_lengths)

        concept_labels = torch.full(
            (batch_size, max_concept_len, self.concept_label_dim),
            float(self.label_padding_value),
            dtype=torch.float32,
        )

        for idx, feature in enumerate(features):
            label_tensor = torch.tensor(feature["concept_labels"], dtype=torch.float32)
            seq_len = label_tensor.size(0)
            concept_labels[idx, :seq_len, :] = label_tensor

        task_lengths = [len(feature["task_labels"]) for feature in features]
        max_task_len = max(task_lengths)

        task_labels = torch.full(
            (batch_size, max_task_len),
            self.label_padding_value,
            dtype=torch.long,
        )

        for idx, feature in enumerate(features):
            task_tensor = torch.tensor(feature["task_labels"], dtype=torch.long)
            seq_len = task_tensor.size(0)
            task_labels[idx, :seq_len] = task_tensor

        return {
            "input_values": input_values,
            "attention_mask": attention_mask,
            "concept_labels": concept_labels,
            "task_labels": task_labels,
        }


def load_training_config(path: Path = CONFIG_PATH) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def prepare_dataset_joint(config: Dict[str, Any], feature_extractor: Wav2Vec2FeatureExtractor):
    dataset_path = Path(
        config.get("processed_dataset_path_joint", "./datasets/processed_timit_dataset-joint")
    )
    os.makedirs(dataset_path.parent, exist_ok=True)

    if config["load_processed_dataset"] and dataset_path.exists():
        print(f"Loading processed TIMIT dataset from: {dataset_path}")
        dataset = load_from_disk(str(dataset_path))
    else:
        print("Loading and processing TIMIT dataset...")
        dataset = load_timit_dataset(config["sample_validation_set"], config.get("sample_validation_size", 0.10))

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

        dataset = dataset.map(
            format_for_joint,
            desc="Formatting for joint training",
            load_from_cache_file=False,
        )

        keep_columns = {"input_values", "concept_labels", "task_labels"}
        for split in dataset.keys():
            remove_columns = [
                column for column in dataset[split].column_names if column not in keep_columns
            ]
            if remove_columns:
                dataset[split] = dataset[split].remove_columns(remove_columns)

        print(f"Saving processed dataset to: {dataset_path}")
        dataset.save_to_disk(str(dataset_path))

    return dataset


def _collapse_ctc_predictions(sequence: np.ndarray, blank_id: int = 0) -> List[int]:
    """Remove blank tokens and repeated predictions for a single sequence."""
    collapsed: List[int] = []
    prev_token = None
    for token in sequence.tolist():
        if token == blank_id:
            prev_token = None
            continue
        if token != prev_token:
            collapsed.append(int(token))
        prev_token = token
    return collapsed


def _ids_to_phonemes(sequence: List[int]) -> List[str]:
    """Convert a sequence of phoneme IDs to their string tokens."""
    tokens: List[str] = []
    for idx in sequence:
        if idx == CTC_BLANK_ID:
            continue
        token = ID_TO_PHONEME.get(int(idx))
        if token is not None and token != "<pad>":
            tokens.append(token)
    return tokens


def _get_task_label_ids(label_ids):
    if isinstance(label_ids, (tuple, list)) and len(label_ids) > 1:
        return label_ids[1]
    return label_ids


def compute_metrics(pred) -> Dict[str, float]:
    """Calculate phoneme error rate (PER) for CTC predictions."""
    logits = pred.predictions
    label_ids = _get_task_label_ids(pred.label_ids)

    pred_ids = np.argmax(logits, axis=-1)
    print(f"Percentage of blanks predicted: {(pred_ids == 0).mean():.2%}")
    pred_sequences = [_collapse_ctc_predictions(seq, blank_id=CTC_BLANK_ID) for seq in pred_ids]

    label_sequences: List[List[int]] = []
    for label_seq in label_ids:
        if isinstance(label_seq, torch.Tensor):
            label_seq = label_seq.cpu().numpy()
        filtered = [int(idx) for idx in label_seq.tolist() if idx != -100]
        label_sequences.append(filtered)

    np.random.seed(101)
    ids = np.random.randint(0, len(pred_sequences), size=5)
    for idx in ids:
        print(f"Sample {idx}:")
        print(f"prediction: {_ids_to_phonemes(pred_sequences[idx])}")
        print(f"labels: {_ids_to_phonemes(label_sequences[idx])} \n")

    references: List[str] = []
    predictions: List[str] = []

    for prediction, reference in zip(pred_sequences, label_sequences):
        reference_tokens = _ids_to_phonemes(reference)
        references.append(" ".join(reference_tokens))
        prediction_tokens = _ids_to_phonemes(prediction)
        predictions.append(" ".join(prediction_tokens))

    phoneme_error_rate = wer(references, predictions)

    return {"phoneme_error_rate": float(phoneme_error_rate)}


def unfreeze_encoder_layers(model, layer_indices: List[int]) -> None:
    """Unfreeze specific wav2vec2 encoder transformer layers by index."""
    if layer_indices is None:
        return

    encoder_layers = model.wav2vec2.encoder.layers

    for layer_idx in layer_indices:
        for param in encoder_layers[layer_idx].parameters():
            param.requires_grad = True


if __name__ == "__main__":
    
    config = load_training_config()

    model_checkpoint = config["model_checkpoint"]
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_checkpoint)

    dataset = prepare_dataset_joint(config, feature_extractor)
    eval_split = "validation" if "validation" in dataset else "test"
    print("eval split:", eval_split)

    vocab_size = len(PHONEME_TOKEN_TO_ID)
    print(f"Phoneme vocabulary size: {vocab_size}")

    model = Wav2Vec2ForJointBottleneck.from_pretrained(     # Use from_pretrained so trained weights are used
        model_checkpoint,
        num_concepts=BINARY_FEATURE_DIM,
        vocab_size=vocab_size,
        joint_lambda=config.get("joint_lambda", 1.0),
        use_safetensors=True
    )

    if config.get("use_initial_unfreeze"):
        unfreeze_encoder_layers(model, config.get("unfreeze_layers", []))

    load_dotenv()
    api_key = os.getenv("WANDB_API_KEY")
    wandb.login(key=api_key)

    wandb_run = wandb.init(
        project=config["wandb_project"],
        name=config["run_name"],
        config=config,
    )

    training_args = TrainingArguments(
        output_dir=config.get("output_dir_joint", config["output_dir"]),
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
        report_to="wandb",
    )

    data_collator = JointDataCollator(concept_label_dim=BINARY_FEATURE_DIM)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset[eval_split],
        data_collator=data_collator,
        tokenizer=feature_extractor,
        compute_metrics=compute_metrics,
        label_names=["concept_labels", "task_labels"],
    )

    trainer.train()

    test_results = trainer.predict(dataset["test"])
    print(test_results.metrics)

    wandb_run.finish()
