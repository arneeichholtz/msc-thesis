# """Data preparation pipeline for the CTC task layer.

# This module reuses the articulatory feature alignment from `data_prep` and
# converts phoneme transcripts into integer token sequences using the phoneme
# vocabulary built in `vocab`.
# """

# from __future__ import annotations

# from typing import Dict, List

# from datasets import Dataset

# from data_prep import canonicalize_phoneme, load_timit_dataset, prepare_dataset
# from vocab import get_phoneme_vocab

# PHONEME_VOCAB = get_phoneme_vocab()


# def _phoneme_sequence_to_ids(phonemes: List[str]) -> List[int]:
#     ids: List[int] = []
#     for phoneme in phonemes:
#         canonical = canonicalize_phoneme(phoneme)
#         try:
#             ids.append(PHONEME_VOCAB[canonical])
#         except KeyError as exc:  # pragma: no cover - defensive branch
#             raise KeyError(f"Phoneme '{canonical}' not found in vocabulary") from exc
#     return ids


# def process_data_for_ctc(batch: Dict) -> Dict[str, List]:
#     """Return dict with frame-level input values and phoneme target ids."""
#     processed = prepare_dataset(batch)
#     input_values = processed["labels"]
#     phonemes = processed["phonetic_detail"]["utterance"]
#     labels = _phoneme_sequence_to_ids(phonemes)
#     return {"input_values": input_values, "labels": labels}


# def _preview_processed_example(dataset: Dataset) -> None:
#     subset = dataset.select(range(10))
#     processed = subset.map(
#         process_data_for_ctc,
#         remove_columns=subset.column_names,
#     )
#     example = processed[0]
#     num_frames = len(example["input_values"])
#     feature_dim = len(example["input_values"][0]) if num_frames > 0 else 0
#     label_length = len(example["labels"])
#     print("Processed example summary:")
#     print(f"  input_values: {num_frames} frames x {feature_dim} dims")
#     print(f"  labels: {label_length} phonemes")


# if __name__ == "__main__":
#     timit = load_timit_dataset()
#     print("Previewing first 10 training samples...")
#     _preview_processed_example(timit["train"])
