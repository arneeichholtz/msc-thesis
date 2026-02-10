"""Training entry point for wav2vec2 articulatory feature prediction."""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
import yaml
from transformers import (
    Trainer,
    TrainingArguments,
    Wav2Vec2FeatureExtractor,
)

from callbacks import FreezingCallback, freeze_encoder
from data_prep import (
    BINARY_FEATURE_DIM,
    TARGET_SAMPLING_RATE,
    filter_short_audio,
    load_timit_dataset,
    prepare_dataset,
)
from model import Wav2Vec2ForArticulatoryFeatures

CONFIG_PATH = Path("config.yml")


@dataclass
class ArticulatoryFeatureDataCollator:
    """Pad variable-length audio and frame-aligned label sequences."""

    label_dim: int
    padding_value: float = 0.0

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        batch_size = len(features)

        input_lengths = [len(feature["input_values"]) for feature in features]
        max_input_length = max(input_lengths)

        input_values = torch.full(
            (batch_size, max_input_length), self.padding_value, dtype=torch.float32
        )
        attention_mask = torch.zeros((batch_size, max_input_length), dtype=torch.long)

        for idx, feature in enumerate(features):
            values = torch.tensor(feature["input_values"], dtype=torch.float32)
            seq_len = values.size(0)
            input_values[idx, :seq_len] = values
            attention_mask[idx, :seq_len] = 1

        label_lengths = [len(feature["labels"]) for feature in features]
        max_label_length = max(label_lengths)

        labels = torch.zeros(
            (batch_size, max_label_length, self.label_dim), dtype=torch.float32
        )
        for idx, feature in enumerate(features):
            label_tensor = torch.tensor(feature["labels"], dtype=torch.float32)
            labels[idx, : label_tensor.size(0), :] = label_tensor

        return {
            "input_values": input_values,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def load_training_config(path: Path = CONFIG_PATH) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def compute_inputs(batch: Dict) -> Dict:
    audio_arrays = [item["array"] for item in batch["audio"]]
    outputs = feature_extractor(
        audio_arrays,
        sampling_rate=TARGET_SAMPLING_RATE,
        return_attention_mask=False,
    )
    batch["input_values"] = outputs["input_values"]
    return batch


def train_preprocessing():
    dataset = load_timit_dataset()
    dataset = filter_short_audio(dataset)
    dataset = dataset.map(
        prepare_dataset,
        desc="Aligning articulatory features",
    )

    # dataset = dataset.map(
    #     compute_inputs,
    #     batched=True,
    #     desc="Extracting wav2vec2 inputs",
    # )

    # keep_columns = {"input_values", "labels"}
    # for split in dataset.keys():
    #     remove_columns = [
    #         column for column in dataset[split].column_names if column not in keep_columns
    #     ]
    #     if remove_columns:
    #         dataset[split] = dataset[split].remove_columns(remove_columns)
    
    return dataset




def main() -> None:
    # config = load_training_config()

    # model_checkpoint = config["model_checkpoint"]

    # feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_checkpoint)

    # dataset = load_timit_dataset()
    # dataset = filter_short_audio(dataset)
    # dataset = dataset.map(
    #     prepare_dataset,
    #     desc="Aligning articulatory features",
    # )

    # dataset = dataset.map(
    #     compute_inputs,
    #     batched=True,
    #     desc="Extracting wav2vec2 inputs",
    # )

    # keep_columns = {"input_values", "labels"}
    # for split in dataset.keys():
    #     remove_columns = [
    #         column for column in dataset[split].column_names if column not in keep_columns
    #     ]
    #     if remove_columns:
    #         dataset[split] = dataset[split].remove_columns(remove_columns)

    num_labels = BINARY_FEATURE_DIM

    model = Wav2Vec2ForArticulatoryFeatures.from_pretrained(
        model_checkpoint,
        num_labels=num_labels,
    )

    freeze_encoder(model)

    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        evaluation_strategy=config["evaluation_strategy"],
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        num_train_epochs=config["num_train_epochs"],
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        save_total_limit=config["save_total_limit"],
        fp16=config.get("use_fp16", False) and torch.cuda.is_available(),
    )

    data_collator = ArticulatoryFeatureDataCollator(label_dim=num_labels)

    eval_split = "test" if "test" in dataset else "validation"

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset[eval_split],
        data_collator=data_collator,
        tokenizer=feature_extractor,
        callbacks=[FreezingCallback(model, thaw_step=config["thaw_step"])],
    )

    trainer.train()


if __name__ == "__main__":
    
    config = load_training_config()
    # model_checkpoint = config["model_checkpoint"]
    # feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_checkpoint)

    dataset = train_preprocessing()
