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
from transformers import (
    Trainer,
    TrainingArguments,
    Wav2Vec2FeatureExtractor
)

from datasets import DatasetDict, load_from_disk

from data_prep import (
    BINARY_FEATURE_DIM,
    TARGET_SAMPLING_RATE,
    load_timit_dataset,
    extract_framewise_binfeatures,
    prepare_audio_samples,
    FEATURE_GROUPS_LABELS
)

from model import Wav2Vec2ForArticulatoryFeatures
from utils.concept_utils import compute_concept_metrics
from utils.utils import print_dataset_statistics

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

        label_lengths = [len(feature["concept_labels"]) for feature in features]
        max_label_length = max(label_lengths)

        labels = torch.full((batch_size, max_label_length, self.label_dim), -100.0, dtype=torch.float32)
        for idx, feature in enumerate(features):
            label_tensor = torch.tensor(feature["concept_labels"], dtype=torch.float32)
            seq_len = label_tensor.size(0)
            labels[idx, :seq_len, :] = label_tensor

        return {
            "input_values": input_values,
            "attention_mask": attention_mask,
            "concept_labels": labels,
        }


def load_training_config(path: Path = CONFIG_PATH) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def unfreeze_encoder_layers(model, layer_indices: List[int]) -> None:
    """Unfreeze specific wav2vec2 encoder transformer layers by index."""
    if layer_indices is None:
        return

    encoder_layers = model.wav2vec2.encoder.layers

    for layer_idx in layer_indices:
        for param in encoder_layers[layer_idx].parameters():
            param.requires_grad = True
    

def compute_metrics(eval_pred):
    logits = eval_pred.predictions
    labels = eval_pred.label_ids
    return compute_concept_metrics(
        logits,
        labels,
        print_feature_counts=False,
    )


def print_per_feature_statistics(test_results):
    test_logits = test_results.predictions
    test_labels = test_results.label_ids
    compute_concept_metrics(
        test_logits,
        test_labels,
        print_feature_counts=True,
    )


def prepare_dataset_cl(config, feature_extractor):
    dataset_path = config.get("processed_dataset_path_cl", "./datasets/processed_timit_dataset-conceptlayer")
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
            extract_framewise_binfeatures,
            desc="Extracting binary articulatory features for frames",
            load_from_cache_file=False
        )
        
        dataset = dataset.map(
            prepare_audio_samples,
            batched=True,       # Defaults to standard batch size=1000 for dataset library
            fn_kwargs={"feature_extractor": feature_extractor},
            desc="Extracting wav2vec2 inputs",
            load_from_cache_file=False
        )

        keep_columns = {"input_values", "concept_labels"}
        for split in dataset.keys():
            remove_columns = [
                column for column in dataset[split].column_names if column not in keep_columns
            ]
            if remove_columns:
                dataset[split] = dataset[split].remove_columns(remove_columns)
            
        print(f"Saving processed dataset to: {dataset_path}")
        dataset.save_to_disk(dataset_path)

        
    
    return dataset


def initialize_model(config):
    num_labels = BINARY_FEATURE_DIM
    model_checkpoint = config["model_checkpoint"]

    model = Wav2Vec2ForArticulatoryFeatures.from_pretrained(            # from_pretrained will load default config.json from HuggingFace
        model_checkpoint,
        num_labels=num_labels,                                          # num_labels is updated in the config
        use_safetensors=True
    )

    initial_unfreeze = config.get("use_initial_unfreeze", False)
    if initial_unfreeze:
        unfreeze_layers = config.get("unfreeze_layers")
        for layer_idx in unfreeze_layers:
            assert 0 <= layer_idx < len(model.wav2vec2.encoder.layers), f"Layer index {layer_idx} is out of bounds for wav2vec2 encoder layers."
        unfreeze_encoder_layers(model, unfreeze_layers)
        print(f"Wav2Vec2 encoder layers included for fine-tuning: {unfreeze_layers}.")
    else:
        print("Using Default: keeping all wav2vec2 encoder layers frozen at the start of training.")

    return model




if __name__ == "__main__":
    
    config = load_training_config()
    
    model_checkpoint = config["model_checkpoint"]
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_checkpoint)
    
    dataset = prepare_dataset_cl(config, feature_extractor)
    model = initialize_model(config)

    eval_split = "validation" if "validation" in dataset else "test"
    print(f"Evaluation split: {eval_split}")

    print_dataset_statistics(dataset)

    load_dotenv()
    api_key = os.getenv("WANDB_API_KEY")
    wandb.login(key=api_key)
    
    wandb_run = wandb.init(
        project=config["wandb_project"],
        name=config["run_name"],
        config=config,
    )

    training_args = TrainingArguments(
        output_dir=config["output_dir_concept_layer"],
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
        label_names=["concept_labels"],
        dataloader_num_workers=0,
    )

    num_labels = BINARY_FEATURE_DIM
    data_collator = ArticulatoryFeatureDataCollator(label_dim=num_labels)

    trainer = Trainer(      # Trainer handles device placement
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset[eval_split],
        data_collator=data_collator,
        tokenizer=feature_extractor,
        compute_metrics=compute_metrics
    )

    trainer.train()

    test_results = trainer.predict(dataset["test"])
    print(test_results.metrics)

    print_per_feature_statistics(test_results)

    wandb_run.finish()