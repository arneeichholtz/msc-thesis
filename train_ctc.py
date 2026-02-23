"""Training entry point for the CTC phoneme prediction head."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml
from transformers import Trainer, TrainingArguments

from data_prep import BINARY_FEATURE_DIM, load_timit_dataset, process_data_for_ctc
# from data_prep_ctc import process_data_for_ctc
from model import LinearCTCModel
from vocab import get_phoneme_vocab

CONFIG_PATH = Path("config.yml")


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


def prepare_ctc_dataset() -> Dict[str, Any]:
    dataset = load_timit_dataset()
    dataset = dataset.map(
        process_data_for_ctc,
        desc="Preparing CTC inputs",
        load_from_cache_file=False,
    )

    keep_columns = {"input_values", "labels"}
    for split in dataset.keys():
        remove_columns = [
            column for column in dataset[split].column_names if column not in keep_columns
        ]
        if remove_columns:
            dataset[split] = dataset[split].remove_columns(remove_columns)

    return dataset


def main() -> None:
    config = load_training_config()

    dataset = prepare_ctc_dataset()
    eval_split = "validation" if "validation" in dataset else "test"

    vocab = get_phoneme_vocab()
    vocab_size = len(vocab)

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
        report_to="none",
    )

    data_collator = CTCDataCollator(feature_dim=BINARY_FEATURE_DIM)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset[eval_split],
        data_collator=data_collator,
    )

    trainer.train()


if __name__ == "__main__":
    main()
