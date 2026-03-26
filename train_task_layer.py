"""Training entry point for the CTC phoneme prediction head."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import os
import wandb
from dotenv import load_dotenv

import torch
import numpy as np
import yaml
from transformers import Trainer, TrainingArguments, Wav2Vec2FeatureExtractor, Wav2Vec2Model
from transformers.trainer_utils import get_last_checkpoint

from datasets import DatasetDict, load_from_disk
from jiwer import wer

from data_prep import (
    BINARY_FEATURE_DIM, 
    load_timit_dataset, 
    extract_framewise_binfeatures, 
    phoneme_token_to_id, 
    format_for_ctc,
    format_for_ctc_framewiselabels,
    compute_wav2vec2_hidden_states,
    compute_concept_logits,
)

from model import LinearCTCModel, FrameLevelPhonemeModel, Wav2Vec2ForArticulatoryFeatures

CONFIG_PATH = Path("config.yml")

CTC_BLANK_ID = 0
PHONEME_TOKEN_TO_ID = phoneme_token_to_id()
ID_TO_PHONEME = {idx: token for token, idx in PHONEME_TOKEN_TO_ID.items()}


@dataclass
class CTCDataCollator:
    """Pad frame-level articulatory inputs and phoneme targets for CTC."""

    feature_dim: Optional[int] = BINARY_FEATURE_DIM
    input_padding_value: float = 0.0
    label_padding_value: int = -100

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        batch_size = len(features)
        if batch_size == 0:
            raise ValueError("CTCDataCollator received an empty batch")

        frame_lengths = [len(f["input_values"]) for f in features]
        max_frames = max(frame_lengths)
        effective_dim = self.feature_dim
        if effective_dim is None:
            first_sequence = features[0]["input_values"]
            if not first_sequence:
                raise ValueError("Unable to infer feature dimension from empty input sequence")
            effective_dim = len(first_sequence[0])

        input_values = torch.full(
            (batch_size, max_frames, effective_dim),
            self.input_padding_value,
            dtype=torch.float32,
        )       # Shape: (batch_size, max_frames, 29)
        attention_mask = torch.zeros((batch_size, max_frames), dtype=torch.long)        # Attn mask starts out as all zeros

        for idx, feature in enumerate(features):
            values = torch.tensor(feature["input_values"], dtype=torch.float32)
            seq_len = values.size(0)
            input_values[idx, :seq_len] = values        # Assign values like this since torch.stack already requires same size
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


def _get_input_field(config: Dict[str, Any]) -> str:
    tl_input_representation = config.get("tl_input_representation", "binary_concepts")
    if tl_input_representation == "binary_concepts":
        return "labels"         # Labels since the input for the task layer are the labels from the concept layer
    if tl_input_representation == "concept_logits":
        return "concept_logits"
    if tl_input_representation == "w2v2_features":
        return "w2v2_features"
    raise ValueError(
        "Unsupported tl_input_representation='"
        f"{tl_input_representation}'. "
        "Choose one of: 'binary_concepts', 'concept_logits', 'w2v2_features'."
    )


def prepare_dataset_tl(config: Dict[str, Any]) -> Dict[str, Any]:
    tl_input_representation = config.get("tl_input_representation", "binary_concepts")      # Task layer input representation
    dataset_path = config.get("processed_dataset_path_tl")
    dataset_path = Path(dataset_path)
    dataset_path = dataset_path.with_name(dataset_path.stem + f"-{tl_input_representation}")
    if config.get("framewise_labels"):
        dataset_path = dataset_path.with_name(dataset_path.stem + "-framewise_labels")

    print(f"Dataset path: {dataset_path}")
    
    os.makedirs(dataset_path.parent, exist_ok=True)

    if config["load_processed_dataset"] and dataset_path.exists():
        print(f"Loading processed TIMIT dataset from: {dataset_path}")
        dataset = load_from_disk(str(dataset_path))
    else:
        print("Loading and processing TIMIT dataset...")
        dataset = load_timit_dataset(config["sample_validation_set"], config.get("sample_validation_size", 0.10))
        
        # timit_subset = dataset["train"].select(range(1))
        # dataset = DatasetDict({"train": timit_subset})
    
        dataset = dataset.map(
            extract_framewise_binfeatures,
            desc="Extracting binary articulatory features for frames",
            load_from_cache_file=False,
        )

        input_field = _get_input_field(config)

        if input_field == "concept_logits":
            concept_checkpoint_folder = config.get("concept_checkpoint_folder")
            
            if concept_checkpoint_folder is None:
                raise ValueError(
                    "tl_input_representation='concept_logits' requires 'concept_checkpoint_folder' "
                    "pointing to a trained checkpoint from train.py"
                )

            if os.path.isdir(concept_checkpoint_folder):
                concept_checkpoint = get_last_checkpoint(concept_checkpoint_folder)

            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(config.get("model_checkpoint"))
            concept_model = Wav2Vec2ForArticulatoryFeatures.from_pretrained(
                concept_checkpoint,
                num_labels=BINARY_FEATURE_DIM,
            )
            concept_model.eval()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            concept_model.to(device)

            batch_size = config.get("concept_logits_batch_size", 4)
            dataset = dataset.map(
                compute_concept_logits,
                batched=True,
                batch_size=batch_size,
                fn_kwargs={
                    "feature_extractor": feature_extractor,
                    "concept_model": concept_model,
                    "device": device,
                },
                desc="Extracting concept logits from trained concept model",
                load_from_cache_file=False,
            )
            concept_model.to("cpu")

        if input_field == "w2v2_features":
            checkpoint = config.get("wav2vec2_feature_checkpoint", config["model_checkpoint"])
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(checkpoint)
            wav2vec2_model = Wav2Vec2Model.from_pretrained(checkpoint)
            wav2vec2_model.eval()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            wav2vec2_model.to(device)
            batch_size = config.get("wav2vec2_feature_batch_size", 4)
            dataset = dataset.map(
                compute_wav2vec2_hidden_states,
                batched=True,
                batch_size=batch_size,
                fn_kwargs={
                    "feature_extractor": feature_extractor,
                    "wav2vec2_model": wav2vec2_model,
                    "device": device,
                },
                desc="Extracting wav2vec2 hidden states",
                load_from_cache_file=False,
            )
            wav2vec2_model.to("cpu")

        format_kwargs = {"input_field": input_field}

        if config.get("framewise_labels"):      # Use framewise labels
            dataset = dataset.map(
                format_for_ctc_framewiselabels,
                fn_kwargs=format_kwargs,
                desc="Formatting for CTC using framewise labels",
                load_from_cache_file=False,
            )
        else:
            dataset = dataset.map(
                format_for_ctc,
                fn_kwargs=format_kwargs,
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


def compute_metrics(pred) -> Dict[str, float]:
    """Calculate phoneme error rate (PER) for CTC predictions."""
    logits = pred.predictions       # logits shape: (eval_size, max_input_len, 40) = (462, 389, 40)
    label_ids = pred.label_ids      # label_ids shape: (eval_size, max_label_len) = (462, 73)

    pred_ids = np.argmax(logits, axis=-1)       # pred_ids shape: (eval_size, max_input_len) = (462, 389)
    print(f"Percentage of blanks predicted: {(pred_ids == 0).mean():.2%}")
    pred_sequences = [_collapse_ctc_predictions(seq, blank_id=CTC_BLANK_ID) for seq in pred_ids]        # Remove blank and repeated tokens

    label_sequences: List[List[int]] = []       # Save only non-padding labels
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
        references.append(" ".join(reference_tokens))       # Separate by space, this is what WER calculation expects
        prediction_tokens = _ids_to_phonemes(prediction)
        predictions.append(" ".join(prediction_tokens))

    phoneme_error_rate = wer(references, predictions)

    return {"phoneme_error_rate": float(phoneme_error_rate)}


def compute_metrics_frame_level(pred) -> Dict[str, float]:
    """Calculate frame-level phoneme accuracy, ignoring padding."""
    logits = pred.predictions
    label_ids = pred.label_ids

    # Get the predicted IDs by taking the argmax over the vocabulary dimension
    pred_ids = np.argmax(logits, axis=-1)

    # Flatten the arrays to compute metrics easily
    pred_ids_flat = pred_ids.flatten()
    label_ids_flat = label_ids.flatten()

    # Create a mask to ignore the padding tokens (-100 is standard CrossEntropy ignore_index)
    mask = label_ids_flat != -100

    valid_predictions = pred_ids_flat[mask]
    valid_labels = label_ids_flat[mask]

    # Calculate simple accuracy
    accuracy = (valid_predictions == valid_labels).mean()

    return {"accuracy": float(accuracy)}


if __name__ == "__main__":
    
    config = load_training_config()

    dataset = prepare_dataset_tl(config)
    eval_split = "validation" if "validation" in dataset else "test"
    print("eval split:", eval_split)

    vocab_size = len(PHONEME_TOKEN_TO_ID)
    print(f"Phoneme vocabulary size: {vocab_size}")

    sample_feature = dataset["train"][0]["input_values"]
    input_feature_dim = len(sample_feature[0])
    print(f"CTC input feature dimension: {input_feature_dim}")

    if config.get("framewise_labels"):
        print("Using frame-level phoneme labels for training.")
        model = FrameLevelPhonemeModel(input_dim=input_feature_dim, output_dim=vocab_size)
        compute_metrics_fn = compute_metrics_frame_level
    else:
        print("Using phoneme sequence labels for CTC training. (standard setup)")
        model = LinearCTCModel(input_dim=input_feature_dim, output_dim=vocab_size)
        compute_metrics_fn = compute_metrics

    load_dotenv()
    api_key = os.getenv("WANDB_API_KEY")
    wandb.login(key=api_key)
    
    wandb_run = wandb.init(
        project=config["wandb_project"],
        name=config["run_name"],
        config=config
    )
    
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

    data_collator = CTCDataCollator(feature_dim=input_feature_dim)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset[eval_split],
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn
    )

    trainer.train()

    test_results = trainer.predict(dataset["test"])
    print(test_results.metrics)

    # print_per_feature_statistics(test_results)

    wandb_run.finish()
