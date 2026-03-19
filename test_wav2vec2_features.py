import re
import json
import torch
import yaml
from dotenv import load_dotenv
import os
import wandb

import numpy as np
from datasets import load_dataset, Audio, DatasetDict
from transformers import (
    Wav2Vec2CTCTokenizer, 
    Wav2Vec2FeatureExtractor, 
    Wav2Vec2Processor, 
    Wav2Vec2ForCTC, 
    TrainingArguments, 
    Trainer
)
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from jiwer import wer
from features_config.features import phoneme_mapping


@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch


def load_timit_dataset(sample_validation_set: bool = True, sample_validation_size: float = 0.1):
    """Load the TIMIT ASR dataset, sample validation set and resample at 16 kHz."""
    dataset = load_dataset("timit_asr")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    if sample_validation_set:
        train_val_split = dataset["train"].train_test_split(test_size=sample_validation_size, seed=42)
        dataset = DatasetDict({
            "train": train_val_split["train"],
            "validation": train_val_split["test"],
            "test": dataset["test"]           
        })
    return dataset


def extract_phonemes(batch):
    # Collapse TIMIT 61-phone annotations to the mapped set before training/evaluation
    collapsed_phonemes = [
        phoneme_mapping.get(phoneme, "sil")
        for phoneme in batch["phonetic_detail"]["utterance"]
    ]
    batch["phonemes"] = " ".join(collapsed_phonemes)
    return batch


def extract_all_phonemes(batch):
    all_phonemes = " ".join(batch["phonemes"])
    vocab = list(set(all_phonemes.split(" ")))
    return {"vocab": [vocab]}


def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    
    with processor.as_target_processor():
        batch["labels"] = processor(batch["phonemes"]).input_ids
    return batch


def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    # Replace -100 (ignored index) with pad token id for decoding
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    # Decode predictions and labels to strings
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    # Calculate Phoneme Error Rate using jiwer's wer function
    # This works because phonemes are treated as 'words' separated by spaces
    per = wer(label_str, pred_str)

    return {"phoneme_error_rate": per}


CONFIG_PATH = Path("config.yml")
def load_training_config(path: Path = CONFIG_PATH) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


if __name__ == "__main__":

    dataset = load_timit_dataset(sample_validation_set=True, sample_validation_size=0.1)
    dataset = dataset.map(extract_phonemes)

    vocabs = dataset.map(
        extract_all_phonemes,
        batched=True,
        batch_size=-1,
        keep_in_memory=True,
        remove_columns=dataset.column_names["train"]
    )

    eval_split_name = "validation" if "validation" in vocabs else "test"
    vocab_list = list(set(vocabs["train"]["vocab"][0]) | set(vocabs[eval_split_name]["vocab"][0]))
    vocab_dict = {v: k for k, v in enumerate(vocab_list)}

    # Add special tokens
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)
    # CTC models use a delimiter for words, even if predicting phonemes
    vocab_dict["|"] = len(vocab_dict)

    with open('phoneme_vocab.json', 'w') as vocab_file:
        json.dump(vocab_dict, vocab_file)

    # 3. Setup Processor
    tokenizer = Wav2Vec2CTCTokenizer(
        "./phoneme_vocab.json", 
        unk_token="[UNK]", 
        pad_token="[PAD]", 
        word_delimiter_token="|"
    )

    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1, 
        sampling_rate=16000, 
        padding_value=0.0, 
        do_normalize=True, 
        return_attention_mask=False
    )

    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    
    dataset_prepared = dataset.map(prepare_dataset, remove_columns=dataset.column_names["train"], num_proc=4)
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-base", 
        ctc_loss_reduction="mean", 
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer)
    )

    # Freeze all parameters
    model.freeze_feature_extractor() # Standard helper for conv layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze only the LM Head (the linear projection)
    for param in model.lm_head.parameters():
        param.requires_grad = True

    config = load_training_config()
    load_dotenv()
    api_key = os.getenv("WANDB_API_KEY")
    wandb.login(key=api_key)
        
    wandb_run = wandb.init(
        project="thesis-cbm",
        name="test-wav2vec2-features",
        config=config
    )

    # Training Configuration
    training_args = TrainingArguments(
        output_dir="/projects/0/prjs1921/thesis-speech/model_checkpoints",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        eval_strategy="steps",
        num_train_epochs=20,
        fp16=True,
        save_steps=500,
        eval_steps=500,
        logging_steps=20,
        learning_rate=1e-3,
        warmup_steps=1000,
        save_total_limit=1,
        report_to="wandb"
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=dataset_prepared["train"],
        eval_dataset=dataset_prepared[eval_split_name],
        tokenizer=processor.feature_extractor,
    )

    trainer.train()
    test_results = trainer.predict(dataset_prepared["test"])
    print(test_results.metrics)

    wandb_run.finish()


