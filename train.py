"""Training entry point for wav2vec2 articulatory feature prediction."""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
import yaml
import wandb
from transformers import (
    Trainer,
    TrainingArguments,
    Wav2Vec2FeatureExtractor,
)

from datasets import DatasetDict

from callbacks import UnfreezingCallback, unfreeze_encoder_layers
from data_prep import (
    BINARY_FEATURE_DIM,
    TARGET_SAMPLING_RATE,
    load_timit_dataset,
    prepare_dataset,
)
from model import Wav2Vec2ForArticulatoryFeatures

CONFIG_PATH = Path("config.yml")


@dataclass      # dataclass decorator allows for cleaner class definition without args and init
class ArticulatoryFeatureDataCollator:
    """Data Collator is used to pad the input values and labels to be the same length in the batch,
       and make the corresponding attention mask. This class is used as input argument for the Trainer,
       and can be used on the fly."""

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

        labels = torch.zeros((batch_size, max_label_length, self.label_dim), dtype=torch.float32)
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


def compute_inputs(batch: Dict, feature_extractor: Wav2Vec2FeatureExtractor) -> Dict:
    """Prepare raw audio samples for wav2vec2 input features using feature extractor. This will standardize (0-mean) the data."""
    audio_arrays = [item["array"] for item in batch["audio"]]       # Length batch size
    extractor_outputs = feature_extractor(
        audio_arrays,
        sampling_rate=TARGET_SAMPLING_RATE,
        return_attention_mask=False,
    )
    batch["input_values"] = extractor_outputs["input_values"]
    return batch


def train_preprocessing(config, feature_extractor: Wav2Vec2FeatureExtractor):
    dataset = load_timit_dataset()

    # timit_subset = dataset["train"].select(range(200))
    # dataset = DatasetDict({"train": timit_subset})
    
    dataset = dataset.map(
        prepare_dataset,
        desc="Aligning articulatory features",
    )

    dataset = dataset.map(
        compute_inputs,
        batched=True,       # Defaults to standard batch size=1000 for datasets
        fn_kwargs={"feature_extractor": feature_extractor},
        desc="Extracting wav2vec2 inputs",
    )

    keep_columns = {"input_values", "labels"}
    for split in dataset.keys():
        remove_columns = [
            column for column in dataset[split].column_names if column not in keep_columns
        ]
        if remove_columns:
            dataset[split] = dataset[split].remove_columns(remove_columns)
    
    num_labels = BINARY_FEATURE_DIM
    model_checkpoint = config["model_checkpoint"]

    model = Wav2Vec2ForArticulatoryFeatures.from_pretrained(
        model_checkpoint,
        num_labels=num_labels,
        use_safetensors=True
    )

    initial_unfreeze = config.get("use_initial_unfreeze")
    if initial_unfreeze:
        assert config.get("unfreeze_layers") is not None, "unfreeze_layers cannot be None if use_initial_unfreeze is True"
        unfreeze_encoder_layers(model, config.get("unfreeze_layers"))
        print(f"Initially unfroze encoder layers {config.get('unfreeze_layers')}.")
    else:
        print("Using Default: keeping all wav2vec2 encoder layers frozen at the start of training.")
    
    return dataset, model




if __name__ == "__main__":
    
    config = load_training_config()

    wandb_run = wandb.init(
        project=config["wandb_project"],
        name=config.get("run_name"),
        config=config,
    )

    model_checkpoint = config["model_checkpoint"]
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_checkpoint)
    num_labels = BINARY_FEATURE_DIM
    
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
        save_total_limit=config["save_total_limit"],
        report_to="wandb",
        fp16=config.get("use_fp16", False) and torch.cuda.is_available(),
    )

    data_collator = ArticulatoryFeatureDataCollator(label_dim=num_labels)

    eval_split = "test" if "test" in dataset else "validation"

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
    )

    trainer.train()
    wandb_run.finish()