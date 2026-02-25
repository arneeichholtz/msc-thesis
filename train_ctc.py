"""Training entry point for the CTC phoneme prediction head."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import os
import wandb
from dotenv import load_dotenv

import torch
import numpy as np
import yaml
from transformers import Trainer, TrainingArguments

from datasets import DatasetDict, load_from_disk
from jiwer import wer

from data_prep import (
    BINARY_FEATURE_DIM, 
    load_timit_dataset, 
    prepare_dataset, 
    get_phoneme_vocab, 
    format_for_ctc
)
from model import LinearCTCModel

CONFIG_PATH = Path("config.yml")

CTC_BLANK_ID = 0
PHONEME_VOCAB = get_phoneme_vocab()
ID_TO_PHONEME = {idx: token for token, idx in PHONEME_VOCAB.items()}


@dataclass
class CTCDataCollator:
    """Pad frame-level articulatory inputs and phoneme targets for CTC."""

    feature_dim: int = BINARY_FEATURE_DIM
    input_padding_value: float = 0.0
    label_padding_value: int = -100

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        batch_size = len(features)

        frame_lengths = [len(f["input_values"]) for f in features]
        max_frames = max(frame_lengths)

        input_values = torch.full(
            (batch_size, max_frames, self.feature_dim),
            self.input_padding_value,
            dtype=torch.float32,
        )
        attention_mask = torch.zeros((batch_size, max_frames), dtype=torch.long)

        for idx, feature in enumerate(features):
            values = torch.tensor(feature["input_values"], dtype=torch.float32)
            seq_len = values.size(0)
            input_values[idx, :seq_len] = values
            attention_mask[idx, :seq_len] = 1

        label_lengths = [len(f["labels"]) for f in features]
        max_label_len = max(label_lengths)
        labels = torch.full(
            (batch_size, max_label_len),
            self.label_padding_value,
            dtype=torch.long,
        )

        for idx, feature in enumerate(features):
            label_tensor = torch.tensor(feature["labels"], dtype=torch.long)
            seq_len = label_tensor.size(0)
            labels[idx, :seq_len] = label_tensor

        return {
            "input_values": input_values,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def load_training_config(path: Path = CONFIG_PATH) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def prepare_ctc_dataset(config: Dict[str, Any]) -> Dict[str, Any]:
    dataset_path = config.get("processed_dataset_path_tl", "./datasets/processed_timit_dataset-tasklayer")
    os.makedirs(Path(dataset_path).parent, exist_ok=True)

    if config["load_processed_dataset"] and Path(dataset_path).exists():
        print(f"Loading processed TIMIT dataset from {dataset_path}...")
        dataset = load_from_disk(dataset_path)
    else:
        print("Loading and processing TIMIT dataset...")
        dataset = load_timit_dataset(config["sample_validation_set"], config.get("sample_validation_size", 0.10))
        
        timit_subset = dataset["train"].select(range(1000))
        dataset = DatasetDict({"train": timit_subset})
    
        dataset = dataset.map(
            prepare_dataset,
            desc="Aligning articulatory features",
            load_from_cache_file=False,
        )

        dataset = dataset.map(
            format_for_ctc,
            desc="Formatting for CTC",
            load_from_cache_file=False,
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


def compute_metrics(eval_pred) -> Dict[str, float]:
    """Calculate phoneme error rate (PER) for CTC predictions."""
    logits = eval_pred.predictions
    # if isinstance(logits, tuple):
    #     logits = logits[0]

    label_ids = eval_pred.label_ids

    pred_ids = np.argmax(logits, axis=-1)
    pred_sequences = [_collapse_ctc_predictions(seq, blank_id=CTC_BLANK_ID) for seq in pred_ids]

    label_sequences: List[List[int]] = []
    for label_seq in label_ids:
        if isinstance(label_seq, torch.Tensor):
            label_seq = label_seq.cpu().numpy()
        filtered = [int(idx) for idx in label_seq.tolist() if idx != -100]
        label_sequences.append(filtered)

    references: List[str] = []
    predictions: List[str] = []

    for prediction, reference in zip(pred_sequences, label_sequences):
        reference_tokens = _ids_to_phonemes(reference)
        if not reference_tokens:
            continue
        references.append(" ".join(reference_tokens))
        prediction_tokens = _ids_to_phonemes(prediction)
        predictions.append(" ".join(prediction_tokens))

    if not references:
        phoneme_error_rate = 0.0
    else:
        phoneme_error_rate = wer(references, predictions)

    return {"phoneme_error_rate": float(phoneme_error_rate)}


if __name__ == "__main__":
    config = load_training_config()

    # load_dotenv()
    # api_key = os.getenv("WANDB_API_KEY")
    # wandb.login(key=api_key)
    
    # wandb_run = wandb.init(
    #     project=config["wandb_project"],
    #     name=config["run_name"],
    #     config=config,
    # )

    dataset = prepare_ctc_dataset(config)
    eval_split = "validation" if "validation" in dataset else "test"

    print("eval split:", eval_split)

    vocab_size = len(PHONEME_VOCAB)

    print(f"Phoneme vocabulary size: {vocab_size}")

    model = LinearCTCModel(input_dim=BINARY_FEATURE_DIM, output_dim=vocab_size)

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
        report_to="wandb",
    )

    data_collator = CTCDataCollator(feature_dim=BINARY_FEATURE_DIM)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset[eval_split],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    test_results = trainer.predict(dataset["test"])
    print(test_results.metrics)

    # print_per_feature_statistics(test_results)

    # wandb_run.finish()
