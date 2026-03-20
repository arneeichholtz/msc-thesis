import os
import re
import json
import random
import numpy as np
import pandas as pd
import torch
from jiwer import wer
import yaml
from pathlib import Path
from dotenv import load_dotenv
import wandb

from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from datasets import load_dataset, ClassLabel, Audio, DatasetDict
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer
)

# --- Global Definitions ---

CHARS_TO_IGNORE_REGEX = '[\,\?\.\!\-\;\:\"]'

def load_timit_dataset(sample_validation_set: bool = True, sample_validation_size: float = 0.1):
    """Load the TIMIT ASR dataset, sample validation set and resample at 16 kHz."""
    dataset = load_dataset("timit_asr")
    # dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    if sample_validation_set:
        train_val_split = dataset["train"].train_test_split(test_size=sample_validation_size, seed=42)
        dataset = DatasetDict({
            "train": train_val_split["train"],
            "validation": train_val_split["test"],
            "test": dataset["test"]           
        })
    return dataset

def remove_special_characters(batch):
    batch["text"] = re.sub(CHARS_TO_IGNORE_REGEX, '', batch["text"]).lower()
    return batch

def extract_all_chars(batch):
    all_text = " ".join(batch["text"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}

def prepare_dataset(batch, processor):
    audio = batch["audio"]
    # batched output is "un-batched" to ensure mapping works correctly
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    
    with processor.as_target_processor():
        batch["labels"] = processor(batch["text"]).input_ids
    return batch

@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch


def compute_metrics(pred, processor):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    # wer = wer_metric.compute(predictions=pred_str, references=label_str)
    word_error_rate = wer(label_str, pred_str)

    return {"word_error_rate": word_error_rate}


CONFIG_PATH = Path("config.yml")
def load_training_config(path: Path = CONFIG_PATH) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


    

if __name__ == "__main__":
    
    # 1. Load Dataset
    # timit = load_dataset("timit_asr")
    timit = load_timit_dataset(sample_validation_set=True, sample_validation_size=0.1)
    timit = timit.remove_columns(["phonetic_detail", "word_detail", "dialect_region", "id", "sentence_type", "speaker_id"])

    # 2. Preprocess Text
    timit = timit.map(remove_special_characters)

    # 3. Create Vocabulary
    vocabs = timit.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, 
                       remove_columns=timit.column_names["train"])
    
    vocab_list = list(set(vocabs["train"]["vocab"][0]) | set(vocabs["test"]["vocab"][0]))
    vocab_dict = {v: k for k, v in enumerate(vocab_list)}
    
    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)

    with open('vocab.json', 'w') as vocab_file:
        json.dump(vocab_dict, vocab_file)

    # 4. Initialize Tokenizer, Feature Extractor, and Processor
    tokenizer = Wav2Vec2CTCTokenizer("./vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, 
                                                do_normalize=True, return_attention_mask=False)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    # 5. Preprocess Data
    timit = timit.map(lambda x: prepare_dataset(x, processor), remove_columns=timit.column_names["train"], num_proc=4)

    # 6. Training Setup
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    # wer_metric = load_metric("wer")

    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-base", 
        ctc_loss_reduction="mean", 
        pad_token_id=processor.tokenizer.pad_token_id,
    )
    model.freeze_feature_extractor()

    training_args = TrainingArguments(
        output_dir="/projects/0/prjs1921/thesis-speech/model_checkpoints",
        group_by_length=True,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        eval_strategy="steps",
        num_train_epochs=20,
        fp16=True,
        save_steps=500,
        eval_steps=100,
        logging_steps=20,
        learning_rate=1e-3,
        weight_decay=0.005,
        warmup_steps=1000,
        save_total_limit=1,
        report_to="wandb"
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=lambda x: compute_metrics(x, processor),
        train_dataset=timit["train"],
        eval_dataset=timit["validation"],
        tokenizer=processor.feature_extractor,
    )

    config = load_training_config()
    load_dotenv()
    api_key = os.getenv("WANDB_API_KEY")
    wandb.login(key=api_key)
        
    wandb_run = wandb.init(
        project="thesis-cbm",
        name="test-wav2vec2-features-wer",
        config=config
    )

    # Training and Evaluation
    trainer.train()
    
    test_results = trainer.predict(timit["test"])
    print(test_results.metrics)

    wandb_run.finish()