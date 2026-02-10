# Wav2Vec2 Concept Bottleneck Fine-Tuning on TIMIT

This project adapts `facebook/wav2vec2-base` into a Concept Bottleneck Model (CBM) for frame-level articulatory feature prediction on the TIMIT corpus. Instead of generating phoneme IDs directly, the model predicts a structured vector of articulatory features (phonation, manner, place, etc.), which can later feed lightweight classifiers for downstream tasks.

## Project Structure

- `features_config/` – Canonical articulatory feature definitions and phoneme mappings taken from King & Taylor.
- `main.py` – Produces integer feature indices per phoneme; also acts as a quick inspection script.
- `data_prep.py` – Loads TIMIT, aligns phoneme time spans with wav2vec2 frame stride, and converts feature indices into a flattened multi-hot vector.
- `model.py` – Implements `Wav2Vec2ForArticulatoryFeatures`, a custom head on top of wav2vec2 with BCE loss.
- `callbacks.py` – Freezing utilities to mimic the staged fine-tuning schedule from the wav2vec2 paper.
- `train.py` – End-to-end training entry point using Hugging Face `Trainer`.
- `config.yml` – All tunable hyperparameters (checkpoint, LR, batch sizes, thaw step, etc.).
- `environment.yml` – Conda environment specification (GPU-ready installs for PyTorch, Transformers, Datasets).

## Workflow Overview

1. **Data preparation** (`data_prep.py`)
   - Loads Hugging Face `timit_asr`, resamples waveforms to 16 kHz via `Audio` features.
   - Builds a binary multi-hot feature vector for each phoneme based on articulatory categories.
   - Aligns phoneme spans to wav2vec2 frame indices (`stride = 320 samples`).
   - Assigns each frame a binary feature vector; gaps or silence frames fall back to the `sil` feature vector.

2. **Model** (`model.py`)
   - Wraps `Wav2Vec2Model` with a linear projection head sized to `config.num_labels` (the binary vector length).
   - Freezes the convolutional feature extractor layers by default.
   - Uses `BCEWithLogitsLoss` with masked averaging so padding frames do not affect training loss.

3. **Training** (`train.py`)
   - Reads hyperparameters from `config.yml` (model checkpoint, LR, thaw step, eval schedule, etc.).
   - Loads and processes the dataset using the helpers above.
   - Builds a custom collator that pads both raw audio samples and label sequences in the batch.
   - Instantiates `Wav2Vec2ForArticulatoryFeatures`, freezes the encoder, and launches a Hugging Face `Trainer` with the `FreezingCallback` to thaw the encoder after `thaw_step` steps.

## Getting Started

1. **Create the environment**
   ```bash
   conda env create -f environment.yml
   conda activate wav2vec2-cbm
   ```

2. **Download and prepare data**
   ```bash
   python data_prep.py
   ```
   (Optional) Inspect the printed stats for each split.

3. **Adjust configuration**
   Edit `config.yml` to tweak hyperparameters or choose a different base checkpoint (e.g., wav2vec2-large).

4. **Train**
   ```bash
   python train.py
   ```
   Model checkpoints and logs will be written to the directory specified in `config.yml` (`output_dir`).

## Notes & Next Steps

- The current setup focuses on predicting articulatory features; a downstream phoneme classifier can be added later (e.g., logistic regression on frozen outputs).
- For GPUs with limited memory, lower `per_device_train_batch_size` or switch to gradient accumulation.
- The `callbacks.py` thaw step mirrors the schedule reported in the wav2vec2 paper; adjust `thaw_step` for different datasets or model sizes.
- Ensure you have access to the TIMIT dataset; `datasets.load_dataset("timit_asr")` requires proper authentication/acceptance when run for the first time.
