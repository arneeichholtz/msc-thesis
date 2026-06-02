"""CTC utilities for decoding and exporting predictions."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from tqdm import tqdm


def _collapse_ctc_predictions(sequence: np.ndarray, blank_id: int) -> List[int]:
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


def _ids_to_tokens(sequence: List[int], id_to_token: Mapping[int, str]) -> List[str]:
    tokens: List[str] = []
    for idx in sequence:
        token = id_to_token.get(int(idx))
        if token is not None and token not in {"<pad>", "<blank>"}:
            tokens.append(token)
    return tokens


def build_ctc_sequences(
    predictions: np.ndarray | tuple,
    label_ids: np.ndarray | List[List[int]],
    id_to_token: Mapping[int, str],
    blank_id: int = 0,
    pad_id: int = -100,
) -> List[Dict[str, Any]]:
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    pred_ids = np.argmax(predictions, axis=-1)

    sequences: List[Dict[str, Any]] = []
    for pred_seq, label_seq in zip(pred_ids, label_ids):
        if hasattr(label_seq, "tolist"):
            label_seq = label_seq.tolist()
        filtered_labels = [int(idx) for idx in label_seq if idx != pad_id]
        collapsed_pred = _collapse_ctc_predictions(pred_seq, blank_id=blank_id)
        sequences.append(
            {
                "prediction_ids": collapsed_pred,
                "prediction_tokens": _ids_to_tokens(collapsed_pred, id_to_token),
                "label_ids": filtered_labels,
                "label_tokens": _ids_to_tokens(filtered_labels, id_to_token),
            }
        )

    return sequences


def save_ctc_sequences(path: str | Path, sequences: List[Dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(sequences, fp, indent=2)


def _align_sequences(
    reference: Sequence[str],
    hypothesis: Sequence[str],
    null_token: str,
) -> List[Tuple[str, str]]:
    ref_len = len(reference)
    hyp_len = len(hypothesis)

    dp = [[0] * (hyp_len + 1) for _ in range(ref_len + 1)]
    back = [[""] * (hyp_len + 1) for _ in range(ref_len + 1)]

    for i in range(1, ref_len + 1):
        dp[i][0] = i
        back[i][0] = "del"
    for j in range(1, hyp_len + 1):
        dp[0][j] = j
        back[0][j] = "ins"

    for i in range(1, ref_len + 1):
        for j in range(1, hyp_len + 1):
            sub_cost = 0 if reference[i - 1] == hypothesis[j - 1] else 1
            sub_total = dp[i - 1][j - 1] + sub_cost
            del_total = dp[i - 1][j] + 1
            ins_total = dp[i][j - 1] + 1

            best = min(sub_total, del_total, ins_total)
            dp[i][j] = best

            if best == sub_total:
                back[i][j] = "sub"
            elif best == del_total:
                back[i][j] = "del"
            else:
                back[i][j] = "ins"

    aligned: List[Tuple[str, str]] = []
    i = ref_len
    j = hyp_len
    while i > 0 or j > 0:
        step = back[i][j]
        if step == "sub":
            aligned.append((reference[i - 1], hypothesis[j - 1]))
            i -= 1
            j -= 1
        elif step == "del":
            aligned.append((reference[i - 1], null_token))
            i -= 1
        else:
            aligned.append((null_token, hypothesis[j - 1]))
            j -= 1

    aligned.reverse()
    return aligned


def build_confusion_matrix_data(
    sequences: List[Dict[str, Any]],
    label_order: Sequence[str] | None = None,
    null_token: str = "NUL",
) -> Dict[str, Any]:
    if label_order is None:
        vocab = set()
        for item in sequences:
            vocab.update(item.get("label_tokens", []))
            vocab.update(item.get("prediction_tokens", []))
        labels = sorted(vocab)
    else:
        labels = list(label_order)

    if null_token not in labels:
        labels.append(null_token)

    index = {label: idx for idx, label in enumerate(labels)}
    size = len(labels)
    matrix = [[0 for _ in range(size)] for _ in range(size)]

    for item in sequences:
        reference = item.get("label_tokens", [])
        hypothesis = item.get("prediction_tokens", [])
        aligned = _align_sequences(reference, hypothesis, null_token)
        for ref_token, hyp_token in aligned:
            if ref_token == hyp_token:
                continue
            ref_idx = index[ref_token]
            hyp_idx = index[hyp_token]
            matrix[ref_idx][hyp_idx] += 1

    return {
        "labels": labels,
        "matrix": matrix,
        "null_token": null_token,
    }


def save_confusion_matrix_data(path: str | Path, data: Dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2)


def build_top_errors_by_group(
    sequences: List[Dict[str, Any]],
    groups: Sequence[Any],
    null_token: str = "NUL",
    top_k: int = 10,
) -> Dict[str, List[Dict[str, Any]]]:
    counters: dict[str, Counter[tuple[str, str, str]]] = defaultdict(Counter)

    def normalize_group(value: Any) -> str:
        if value is None:
            return "unknown"
        return str(value)

    print("Building top errors by group...")
    print(f"Total sequences: {len(sequences)}")

    for group_value, item in tqdm(zip(groups, sequences), total=len(sequences), desc="Processing sequences for dialect errors"):
        group_key = normalize_group(group_value)
        reference = item.get("label_tokens", [])
        hypothesis = item.get("prediction_tokens", [])
        aligned = _align_sequences(reference, hypothesis, null_token)
        for ref_token, hyp_token in aligned:
            if ref_token == hyp_token:
                continue
            if ref_token == null_token:
                error_key = ("ins", null_token, hyp_token)
            elif hyp_token == null_token:
                error_key = ("del", ref_token, null_token)
            else:
                error_key = ("sub", ref_token, hyp_token)
            counters[group_key][error_key] += 1

    output: Dict[str, List[Dict[str, Any]]] = {}
    for group_key, counter in counters.items():
        top_items = counter.most_common(top_k)
        output[group_key] = [
            {
                "type": err_type,
                "ref": ref_token,
                "hyp": hyp_token,
                "count": count,
            }
            for (err_type, ref_token, hyp_token), count in top_items
        ]

    return output


def save_top_errors_by_group(path: str | Path, data: Dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2)


def _edit_distance(reference: List[str], hypothesis: List[str]) -> int:
    """Compute Levenshtein distance between two token sequences."""
    ref_len = len(reference)
    hyp_len = len(hypothesis)
    if ref_len == 0:
        return hyp_len
    if hyp_len == 0:
        return ref_len

    dp = [[0] * (hyp_len + 1) for _ in range(ref_len + 1)]
    for i in range(ref_len + 1):
        dp[i][0] = i
    for j in range(hyp_len + 1):
        dp[0][j] = j

    for i in range(1, ref_len + 1):
        for j in range(1, hyp_len + 1):
            sub_cost = 0 if reference[i - 1] == hypothesis[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + sub_cost,
            )

    return dp[ref_len][hyp_len]


def print_top_test_errors(
    test_results,
    id_to_token: Mapping[int, str],
    blank_id: int = 0,
    pad_id: int = -100,
    top_k: int = 15,
) -> None:
    """Print the test utterances with the highest token edit distance."""
    label_ids = test_results.label_ids
    if isinstance(label_ids, (tuple, list)) and len(label_ids) > 1:
        label_ids = label_ids[1]

    sequences = build_ctc_sequences(
        test_results.predictions,
        label_ids,
        id_to_token,
        blank_id=blank_id,
        pad_id=pad_id,
    )

    scored = []
    for idx, item in enumerate(sequences):
        reference = item.get("label_tokens", [])
        hypothesis = item.get("prediction_tokens", [])
        distance = _edit_distance(reference, hypothesis)
        scored.append((distance, idx, reference, hypothesis))

    scored.sort(key=lambda item: (item[0], len(item[2])), reverse=True)

    print(f"Top {top_k} test utterances by edit distance:")
    for rank, (distance, idx, reference, hypothesis) in enumerate(scored[:top_k], start=1):
        print(f"#{rank} | idx={idx} | edits={distance} | ref_len={len(reference)} | hyp_len={len(hypothesis)}")
        print("reference:", " ".join(reference))
        print("prediction:", " ".join(hypothesis))
        print("-")
