"""Data preparation utilities for Concept Bottleneck fine-tuning on TIMIT.

This script loads the TIMIT ASR dataset, aligns phoneme-level articulatory
feature vectors to wav2vec2 frame timings, and produces frame-aligned binary
label tensors suitable for training a Concept Bottleneck Model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from datasets import Audio, DatasetDict, IterableDatasetDict, load_dataset
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

from features_config.features import (
    central_labels,
    fb_labels,
    manner_labels,
    phonation_labels,
    place_labels,
    round_labels,
    mv_data,
    phoneme_mapping
)

TARGET_SAMPLING_RATE = 16_000
WAV2VEC2_FRAME_STRIDE = 320  # wav2vec2 stride in audio samples (20 ms at 16 kHz)
MIN_AUDIO_SAMPLES = WAV2VEC2_FRAME_STRIDE  # drop utterances shorter than one frame

@dataclass(frozen=True)
class FeatureGroup:
    name: str
    labels: List[str]

# Ordered feature groups matching the k=6 articulatory features
FEATURE_GROUPS = (
    FeatureGroup("phonation", list(phonation_labels.keys())),
    FeatureGroup("manner", list(manner_labels.keys())),
    FeatureGroup("place", list(place_labels.keys())),
    FeatureGroup("front_back", list(fb_labels.keys())),
    FeatureGroup("roundness", list(round_labels.keys())),
    FeatureGroup("centrality", list(central_labels.keys())),
)

FEATURE_GROUPS_LABELS = (
    FeatureGroup("phonation", list(phonation_labels.values())),
    FeatureGroup("manner", list(manner_labels.values())),
    FeatureGroup("place", list(place_labels.values())),
    FeatureGroup("front_back", list(fb_labels.values())),
    FeatureGroup("roundness", list(round_labels.values())),
    FeatureGroup("centrality", list(central_labels.values())),
)

GROUP_SIZES = tuple(len(group.labels) for group in FEATURE_GROUPS)      # Will be (3, 6, 9, 4, 3, 4)
GROUP_OFFSETS = tuple(sum(GROUP_SIZES[:idx]) for idx in range(len(GROUP_SIZES)))        # Will be (0, 3, 9, 18, 22, 25)

BINARY_FEATURE_DIM = sum(GROUP_SIZES)                                   # Is 29

def _indices_to_binary_vector(index_vector: Iterable[int]) -> np.ndarray:
    """Convert a length-6 index vector into a flattened multi-hot numpy array."""
    binary = np.zeros(BINARY_FEATURE_DIM, dtype=np.int8)
    for group_idx, feature_index in enumerate(index_vector):
        offset = GROUP_OFFSETS[group_idx]
        binary[offset + int(feature_index)] = 1.0
    return binary


def phoneme_processing():
    # Create map from label to label index for each feature
    phonation_map = {label: i for i, label in enumerate(phonation_labels.keys())}
    manner_map    = {label: i for i, label in enumerate(manner_labels.keys())}
    place_map     = {label: i for i, label in enumerate(place_labels.keys())}
    fb_map        = {label: i for i, label in enumerate(fb_labels.keys())}
    round_map     = {label: i for i, label in enumerate(round_labels.keys())}
    central_map   = {label: i for i, label in enumerate(central_labels.keys())}

    # Construct the final phoneme_dict, this contains all the 59 phonemes in the MV system, with their corresponding feature values as a list of indices
    full_phoneme_feature_dict = {
        phoneme: [
            phonation_map[feature_values[0]],       # Phonation map is {'v': 0, 'uv': 1, 's_p': 2}
            manner_map[feature_values[1]],
            place_map[feature_values[2]],
            fb_map[feature_values[3]],
            round_map[feature_values[4]],
            central_map[feature_values[5]]
        ]
        for phoneme, feature_values in mv_data.items()
    }

    # Save the features for the 39 reduced phonemes 
    phoneme_feature_dict = {
        phoneme: full_phoneme_feature_dict[phoneme] for phoneme in sorted(set(phoneme_mapping.values()))
    }

    return full_phoneme_feature_dict, phoneme_feature_dict


def phoneme_token_to_id():
    """Return a phoneme vocabulary mapping token -> id.

    ID 0 is reserved for "<blank>" (CTC blank).
    Remaining IDs are assigned to unique phoneme tokens sorted alphabetically.
    """
    unique_tokens = sorted(set(phoneme_mapping.values()))
    token_to_id = {"<blank>": 0}
    token_to_id.update({token: idx + 1 for idx, token in enumerate(unique_tokens)})
    return token_to_id

def get_phoneme_binary_features() -> Dict[str, np.ndarray]:
    """ Converts phoneme to corresponding binary feature vector. """
    return {phoneme: _indices_to_binary_vector(index_vector)
            for phoneme, index_vector in CANONICAL_PHONEME_IDX_DICT.items()}


_, CANONICAL_PHONEME_IDX_DICT = phoneme_processing()        # Maps phoneme labels to their corresponding feature index vectors, like 'g': [0, 1, 6, 0, 0, 2],

PHONEME_BINARY_FEATURES = get_phoneme_binary_features()

SILENCE_VECTOR: np.ndarray = PHONEME_BINARY_FEATURES.get(
    "sil", np.zeros(BINARY_FEATURE_DIM, dtype=np.int8)
)

PHONEME_TOKEN_TO_ID = phoneme_token_to_id()

PHONEME_BIN_FEAT_VECTOR_TO_TOKEN = {tuple(v): k for k, v in PHONEME_BINARY_FEATURES.items()}        # Lookup from binary feature vector to phoneme label

def reduce_phoneme(raw_phoneme: str) -> str:
    """Map raw TIMIT phoneme labels to the standard 39 reduced inventory. Raise ValueError if no mapping exists."""
    key = raw_phoneme.lower()
    mapped_phoneme = phoneme_mapping.get(key, None)
    if not mapped_phoneme:
        raise ValueError(f"Phoneme '{raw_phoneme}' not found in mapping. Check phoneme_mapping for missing entries.")
    else:
        return mapped_phoneme


def extract_framewise_binfeatures(batch: Dict) -> Dict:
    """Extract framewise binary concept labels (phonetic features) for a batch of TIMIT samples."""
    audio = batch["audio"]
    waveform = np.asarray(audio["array"], dtype=np.float32)         # E.g. shaped: (50434,)
    sampling_rate = audio["sampling_rate"]

    if sampling_rate != TARGET_SAMPLING_RATE:
        raise ValueError(f"Expected {TARGET_SAMPLING_RATE} Hz audio but received {sampling_rate} Hz.")

    num_frames = math.floor(len(waveform) / WAV2VEC2_FRAME_STRIDE)        # Floor divide to get number of full frames, e.g. 50434 // 320 = 157. Remaining values are discarded
    
    if num_frames == 0:
        raise ValueError(f"Audio too short for wav2vec2 frame alignment: {len(waveform)} samples.")
    
    frame_labels = np.zeros((num_frames, BINARY_FEATURE_DIM), dtype=np.int8)

    starts = batch["phonetic_detail"]["start"]
    stops = batch["phonetic_detail"]["stop"]
    phonemes = batch["phonetic_detail"]["utterance"]

    for start, stop, phoneme in zip(starts, stops, phonemes):
        
        reduced_phon = reduce_phoneme(phoneme)
        feature_vector = PHONEME_BINARY_FEATURES.get(reduced_phon, SILENCE_VECTOR)     # Use the silence vector for unmapped phonemes, though phonemes in TIMIT and defined phoneme mapping should be the sam
        
        frame_start = math.floor(start / WAV2VEC2_FRAME_STRIDE)          # Floor division so first phoneme will start at 0
        frame_stop = math.ceil(stop / WAV2VEC2_FRAME_STRIDE)             # Ceil rounding so 6.2 frames is 7 frames
        frame_stop = min(frame_stop, num_frames)                         # Ensure remaining values after num_frames are discarded; often this is sil

        if frame_start >= frame_stop:       # Skip phonemes with stop-start < stride, which are ~1% of phonemes; these will be assigned the SIL vector
            continue

        frame_labels[frame_start:frame_stop] = feature_vector            # Overwrite first index of frame_labels
    
    unassigned = np.where(frame_labels.sum(axis=1) == 0)[0]          # Will be non-empty if first time step (starts) is not 0 or for the short phonemes (less than a frame) that are skipped
    if len(unassigned) > 0:
        frame_labels[unassigned] = SILENCE_VECTOR

    batch["num_frames"] = num_frames
    batch["labels"] = frame_labels.tolist()
    batch["binary_feature_dim"] = BINARY_FEATURE_DIM

    return batch


def prepare_audio_samples(batch: Dict, feature_extractor: Wav2Vec2FeatureExtractor) -> Dict:
    """Prepare raw audio samples for wav2vec2 input features using feature extractor. This will standardize (0-mean) the data.
       Raw audio file is not divided into frames, this happens during the model forward pass. Shape of input_values is (batch, seq length), like (32, 57404)"""
    audio_arrays = [item["array"] for item in batch["audio"]]       # Length batch size
    extractor_outputs = feature_extractor(
        audio_arrays,
        sampling_rate=TARGET_SAMPLING_RATE,
        return_attention_mask=False,            # This is advised for wav2vec2-base specifically.
    )
    batch["input_values"] = extractor_outputs["input_values"]
    return batch


def compute_wav2vec2_hidden_states(
    batch: Dict,
    feature_extractor: Wav2Vec2FeatureExtractor,
    wav2vec2_model: Wav2Vec2Model,
    device: torch.device,
) -> Dict:
    """Attach wav2vec2 hidden states (per-frame features) to a dataset batch."""
    audio_arrays = [item["array"] for item in batch["audio"]]
    extractor_outputs = feature_extractor(
        audio_arrays,
        sampling_rate=TARGET_SAMPLING_RATE,
        padding=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

    input_values = extractor_outputs["input_values"].to(device)
    attention_mask = extractor_outputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = wav2vec2_model(
            input_values,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
        )

    hidden_states = outputs.last_hidden_state.cpu()
    output_lengths = wav2vec2_model._get_feat_extract_output_lengths(  # type: ignore[attr-defined]
        attention_mask.sum(dim=1)
    ).to(torch.long)

    features: List[List[List[float]]] = []
    for idx, seq_len in enumerate(output_lengths.tolist()):
        truncated = hidden_states[idx, :seq_len].numpy().astype(np.float32)
        features.append(truncated.tolist())

    batch["w2v2_features"] = features
    return batch


def compute_concept_logits(
    batch: Dict,
    feature_extractor: Wav2Vec2FeatureExtractor,
    concept_model: torch.nn.Module,
    device: torch.device,
) -> Dict:
    """Attach frame-level concept logits from a trained concept model to a dataset batch."""
    audio_arrays = [item["array"] for item in batch["audio"]]
    extractor_outputs = feature_extractor(
        audio_arrays,
        sampling_rate=TARGET_SAMPLING_RATE,
        padding=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

    input_values = extractor_outputs["input_values"].to(device)
    attention_mask = extractor_outputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = concept_model(
            input_values=input_values,
            attention_mask=attention_mask,
            return_dict=True,
        )

    concept_logits = outputs.logits.cpu()
    output_lengths = concept_model.wav2vec2._get_feat_extract_output_lengths(  # type: ignore[attr-defined]
        attention_mask.sum(dim=1)
    ).to(torch.long)

    features: List[List[List[float]]] = []
    for idx, seq_len in enumerate(output_lengths.tolist()):
        truncated = concept_logits[idx, :seq_len].numpy().astype(np.float32)
        features.append(truncated.tolist())

    batch["concept_logits"] = features
    return batch


def load_timit_dataset(sample_validation_set: bool = True, sample_validation_size: float | None = 0.1,) -> DatasetDict | IterableDatasetDict:
    """Load TIMIT, resample to 16 kHz, and optionally create a validation split from train."""
    dataset = load_dataset("timit_asr")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=TARGET_SAMPLING_RATE))
    if sample_validation_set:
        if sample_validation_size is None:
            raise ValueError("sample_validation_size cannot be None when sample_validation_set=True")
        train_val_split = dataset["train"].train_test_split(test_size=sample_validation_size, seed=42)
        dataset = DatasetDict({
            "train": train_val_split["train"],
            "validation": train_val_split["test"],
            "test": dataset["test"]           
        })
    return dataset


def phoneme_sequence_to_ids(phonemes: List[str]) -> List[int]:
    ids: List[int] = []
    for phoneme in phonemes:
        reduced_phon = reduce_phoneme(phoneme)       # This is the reduced phoneme
        try:
            ids.append(PHONEME_TOKEN_TO_ID[reduced_phon])     # ID of the reduced phoneme -- PHONEME_TOKEN_TO_ID already contains <blank> at index 0.
        except KeyError as exc:
            raise KeyError(f"Phoneme '{reduced_phon}' not found in vocabulary") from exc
    return ids


def binary_features_to_phoneme_sequence(feature_list):
    return [PHONEME_TOKEN_TO_ID[PHONEME_BIN_FEAT_VECTOR_TO_TOKEN.get(tuple(f))] for f in feature_list]


def _ensure_serializable_inputs(values):
    if isinstance(values, np.ndarray):
        return values.tolist()
    return values


def format_for_ctc(batch: Dict, input_field: str = "labels") -> Dict:
    return {
        "input_values": _ensure_serializable_inputs(batch[input_field]),
        "labels": phoneme_sequence_to_ids(batch["phonetic_detail"]["utterance"])
    }


def format_for_ctc_framewiselabels(batch: Dict, input_field: str = "labels") -> Dict:
    """Format batch for CTC using frame-level phoneme sequence as labels, rather than ground truth (shorter) phoneme labels. (Used as a test benchmark.)"""
    return {
        "input_values": _ensure_serializable_inputs(batch[input_field]),
        "labels": binary_features_to_phoneme_sequence(batch["labels"])
    }


def format_for_joint(batch: Dict) -> Dict:
    """Format batch for joint training with concept and task labels."""
    return {
        "input_values": _ensure_serializable_inputs(batch["input_values"]),
        "concept_labels": _ensure_serializable_inputs(batch["labels"]),
        "task_labels": phoneme_sequence_to_ids(batch["phonetic_detail"]["utterance"]),
    }

