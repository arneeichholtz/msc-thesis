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
from datasets import Audio, DatasetDict, IterableDatasetDict, load_dataset

from features_config.features import (
    central_labels,
    fb_labels,
    manner_labels,
    phonation_labels,
    place_labels,
    round_labels,
    mv_data,
    phoneme_mapping,
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

GROUP_SIZES = tuple(len(group.labels) for group in FEATURE_GROUPS)      # Will be (3, 6, 9, 4, 3, 4)
GROUP_OFFSETS = tuple(sum(GROUP_SIZES[:idx]) for idx in range(len(GROUP_SIZES)))        # Will be (0, 3, 9, 18, 22, 25)

BINARY_FEATURE_DIM = sum(GROUP_SIZES)                                   # Is 29

def _indices_to_binary_vector(index_vector: Iterable[int]) -> np.ndarray:
    """Convert a length-6 index vector into a flattened multi-hot numpy array."""
    binary = np.zeros(BINARY_FEATURE_DIM, dtype=np.float32)
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
            phonation_map[feature_values[0]],
            manner_map[feature_values[1]],
            place_map[feature_values[2]],
            fb_map[feature_values[3]],
            round_map[feature_values[4]],
            central_map[feature_values[5]]
        ]
        for phoneme, feature_values in mv_data.items()
    }

    # Maps all phoneme labels to the 39+1 labels that will be used 
    phoneme_feature_dict = {
        phoneme_mapping[phoneme]: 
            features for phoneme, features in full_phoneme_feature_dict.items()
    }

    return full_phoneme_feature_dict, phoneme_feature_dict


_, CANONICAL_PHONEME_IDX_DICT = phoneme_processing()        # Maps phoneme labels to their corresponding feature index vectors, like 'g': [0, 1, 6, 0, 0, 2],

PHONEME_BINARY_FEATURES: Dict[str, np.ndarray] = {          # Converts phoneme to corresponding binary feature vector
    phoneme: _indices_to_binary_vector(index_vector)
    for phoneme, index_vector in CANONICAL_PHONEME_IDX_DICT.items()
}

SILENCE_VECTOR: np.ndarray = PHONEME_BINARY_FEATURES.get(
    "sil", np.zeros(BINARY_FEATURE_DIM, dtype=np.float32)
)

def canonicalize_phoneme(raw_phoneme: str) -> str:
    """Map raw TIMIT phoneme labels to the 39+1 canonical inventory. Raise ValueError if no mapping exists."""
    key = raw_phoneme.lower()
    mapped_phoneme = phoneme_mapping.get(key, None)
    if not mapped_phoneme:
        raise ValueError(f"Phoneme '{raw_phoneme}' not found in mapping. Check phoneme_mapping for missing entries.")
    else: 
        return mapped_phoneme


def prepare_dataset(batch: Dict) -> Dict:
    """Frame-align articulatory feature vectors for a single TIMIT example."""
    audio = batch["audio"]
    waveform = np.asarray(audio["array"], dtype=np.float32)         # E.g. shaped: (50434,)
    sampling_rate = audio["sampling_rate"]

    if sampling_rate != TARGET_SAMPLING_RATE:
        raise ValueError(f"Expected {TARGET_SAMPLING_RATE} Hz audio but received {sampling_rate} Hz.")

    num_frames = math.floor(len(waveform) / WAV2VEC2_FRAME_STRIDE)        # Floor divide to get number of full frames, e.g. 50434 // 320 = 157. Remaining values are discarded
    
    if num_frames == 0:
        raise ValueError(f"Audio too short for wav2vec2 frame alignment: {len(waveform)} samples.")
    
    frame_labels = np.zeros((num_frames, BINARY_FEATURE_DIM), dtype=np.float32)

    starts = batch["phonetic_detail"]["start"]
    stops = batch["phonetic_detail"]["stop"]
    phonemes = batch["phonetic_detail"]["utterance"]

    for start, stop, phoneme in zip(starts, stops, phonemes):
        canonical = canonicalize_phoneme(phoneme)
        feature_vector = PHONEME_BINARY_FEATURES.get(canonical, SILENCE_VECTOR)     # Use the silence vector for unmapped phonemes, though phonemes in TIMIT and defined phoneme mapping should be the same

        frame_start = math.floor(start / WAV2VEC2_FRAME_STRIDE)          # Floor division so first phoneme will start at 0
        frame_stop = math.ceil(stop / WAV2VEC2_FRAME_STRIDE)             # Ceil rounding so 6.2 frames is 7 frames
        frame_stop = min(frame_stop, num_frames)                         # Ensure remaining values after num_frames are discarded; often this is sil

        if frame_start >= frame_stop:       # Skip phonemes with stop-start < stride; ~1% of phonemes
            continue

        frame_labels[frame_start:frame_stop] = feature_vector           # Overwrite first index of frame_labels
    
    if num_frames > 0:
        unassigned = np.where(frame_labels.sum(axis=1) == 0)[0]         # Will be non-empty if first time step (starts) is not 0
        if len(unassigned) > 0:
            frame_labels[unassigned] = SILENCE_VECTOR

    batch["num_frames"] = num_frames
    batch["labels"] = frame_labels.tolist()
    batch["binary_feature_dim"] = BINARY_FEATURE_DIM

    return batch


def load_timit_dataset() -> DatasetDict | IterableDatasetDict:
    """Load the TIMIT ASR dataset, sample validation set and resample at 16 kHz."""
    dataset = load_dataset("timit_asr")
    train_val_split = dataset["train"].train_test_split(test_size=0.1, seed=42)
    dataset = DatasetDict({
        "train": train_val_split["train"],
        "validation": train_val_split["test"],
        "test": dataset["test"]           
    })
    return dataset.cast_column("audio", Audio(sampling_rate=TARGET_SAMPLING_RATE))

