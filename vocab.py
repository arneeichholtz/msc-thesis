from features_config.features import phoneme_mapping


def get_phoneme_vocab():
    """Return a phoneme vocabulary mapping token -> id.

    ID 0 is reserved for "<pad>" (CTC blank).
    Remaining IDs are assigned to unique phoneme tokens sorted alphabetically.
    """
    unique_tokens = sorted(set(phoneme_mapping.values()))
    token_to_id = {"<pad>": 0}
    token_to_id.update({token: idx + 1 for idx, token in enumerate(unique_tokens)})
    return token_to_id


if __name__ == "__main__":
    vocab = get_phoneme_vocab()
    print(f"Vocabulary size: {len(vocab)}")
