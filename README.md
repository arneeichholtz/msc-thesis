## Wav2Vec2 Concept Bottleneck Fine-Tuning on TIMIT

This repository adapts `facebook/wav2vec2-base` into a Concept Bottleneck Model (CBM) for frame-level articulatory feature prediction on the TIMIT corpus. The model predicts structured articulatory feature vectors (phonation, manner, place, etc.) per frame instead of phoneme IDs.

This README was updated to document every option in `config.yml` and map each option to the Python scripts that use it. Use the examples below to prepare data, run training and evaluation, and inspect results.

**Quick start (recommended)**

1. Create environment (see `requirements.txt` or `environment.yml`). Example with conda:

```bash
conda env create -f environment.yml
conda activate wav2vec2-cbm
pip install -r requirements.txt
```

2. Prepare processed datasets (writes into `datasets/processed_*` by default):

```bash
python data_prep.py
```

3. Run training or evaluation using one of the training entrypoints below (they read `config.yml` by default):

```bash
python train_concept_layer.py   # train the concept (bottleneck) head
python train_task_layer.py      # train the downstream phoneme/task head
python train_joint.py           # joint bottleneck + task optimization
python train_baseline.py        # baseline training (direct phoneme predictions)
python probe.py                 # run probes / diagnostics configured in probe_config.yml
python test.py                  # run tests/evaluation (uses saved checkpoints)
```

## Files of interest

- `data_prep.py`: builds frame-aligned datasets (wav2vec2 frame stride → frame labels). Writes processed datasets to paths in `config.yml`.
- `features_config/`: articulatory feature definitions and phoneme→feature mappings.
- `model.py`: `Wav2Vec2ForArticulatoryFeatures` and helper utilities for concept logits, masking and loss.
- `train_concept_layer.py`: training script for the concept (bottleneck) head.
- `train_task_layer.py`: training script for the downstream task/phoneme head (can take concept outputs as input).
- `train_joint.py`: joint training script for simultaneous concept + task optimization.
- `train_baseline.py`: baseline direct-to-task training scripts.
- `data_outputs/`: saved predictions, confusion matrices and diagnostics for test runs.
- `probe.py`: small script to run analyses and probes using `probe_config.yml`.
- `test.py`, `test_w2v2_features.py`, `test_w2v2_features_wer.py`: unit/functional test scripts for various components.

## `config.yml` — options and where they are used

Below are every option present in the repository's `config.yml` with a short description and which scripts use them. Edit `config.yml` before running the training/eval scripts.

- `processed_dataset_path_cl` (str)
   - Path to concept-layer processed dataset (default: `./datasets/processed_timit_dataset-conceptlayer`).
   - Used by: `data_prep.py`, `train_concept_layer.py`, `train_joint.py`.

- `processed_dataset_path_tl` (str)
   - Path to task-layer processed dataset (default: `./datasets/processed_timit_dataset-tasklayer`).
   - Used by: `data_prep.py`, `train_task_layer.py`, `train_joint.py`.

- `processed_dataset_path_joint` (str)
   - Path for joint-optimised dataset if different from above.
   - Used by: `train_joint.py`.

- `processed_dataset_path_baselines` (str)
   - Path to datasets used for baseline experiments.
   - Used by: `train_baseline.py`.

- `single_sample_index` (null or int)
   - If set, `data_prep.py` (and loaders) may use this to only process / debug a single example.
   - Used by: `data_prep.py`, debug/test scripts.

- `load_processed_dataset` (bool)
   - If True, training scripts load datasets from `processed_dataset_path_*` instead of re-processing raw TIMIT.
   - Used by: all `train_*.py`, `test.py`.

- `framewise_labels` (bool)
   - When True, the task layer expects framewise phoneme labels. When False the task layer may use sequence/segment labels.
   - Used by: `train_task_layer.py`, `train_joint.py`.

- `tl_input_representation` (str)
   - Options: `binary_concepts`, `concept_logits`, `w2v2_features`.
   - Controls whether the task head receives binary concept vectors, predicted concept logits, or raw wav2vec2 features.
   - Used by: `train_task_layer.py`, `test.py`.

- `use_concept_logits_for_test` (bool)
   - If True, tests that evaluate downstream performance may feed concept logits (soft) to the task head.
   - Used by: `test.py`, `train_task_layer.py` (evaluation logic).

- `wav2vec2_feature_checkpoint` (str)
   - Pretrained checkpoint for feature extraction (e.g., `facebook/wav2vec2-base`).
   - Used by: `data_prep.py`, `train_*.py`, `probe.py`.

- `wav2vec2_feature_batch_size` (int)
   - Batch size used when extracting wav2vec2 features for dataset creation/diagnostics.
   - Used by: `data_prep.py`, `probe.py`.

- `model_checkpoint` (str)
   - Base checkpoint used to initialize model weights for fine-tuning.
   - Used by: `train_*.py`, `probe.py`.

- `output_dir_concept_layer`, `output_dir_task_layer`, `output_dir_joint`, `output_dir_baselines` (str)
   - Output directories for saving checkpoints and logs for each experimental mode.
   - Used by: matching `train_*.py` scripts.

- `eval_strategy` (str)
   - e.g., `steps` or `epoch` — forwarded to the HF `Trainer`/evaluation loop.
   - Used by: `train_*.py`.

- `learning_rate` (float)
   - Base learning rate for optimizers.
   - Used by: `train_*.py`.

- `per_device_train_batch_size`, `per_device_eval_batch_size` (int)
   - Batch sizes for training and evaluation.
   - Used by: `train_*.py`.

- `num_train_epochs` (int)
   - Number of training epochs.
   - Used by: `train_*.py`.

- `logging_steps`, `save_steps`, `eval_steps`, `warmup_steps`, `save_total_limit` (int)
   - HF Trainer / scheduler related options.
   - Used by: `train_*.py`.

- `use_fp16` (bool)
   - Enable mixed precision training (requires CUDA + apex/torch.cuda.amp support).
   - Used by: `train_*.py`.

- `sample_validation_set` (bool) / `sample_validation_size` (float)
   - If True, validation sets may be randomly subsampled to the given fraction.
   - Used by: `train_*.py`.

- `joint_lambda` (float)
   - Weighting factor in joint concept+task loss (how strongly to emphasize task loss vs concept loss).
   - Used by: `train_joint.py`.

- `run_eval_only` (bool)
   - If True, scripts will skip training and only run evaluation using a checkpoint path configured elsewhere.
   - Used by: `train_*.py`, `test.py`.

- `use_lambda_callback` (bool), `initial_lambda` (float), `final_lambda` (float), `schedule` (str)
   - When using dynamic lambda scheduling in joint training, these control the annealing schedule.
   - Used by: `train_joint.py` and `callbacks.py`.

- `joint_concept_metrics` (bool)
   - If True, compute per-concept metrics during joint optimization (precision/recall/F1 etc.).
   - Used by: `train_joint.py`, evaluation code.

- `save_test_results`, `save_confusion_matrix`, `save_dialect_errors`, `dialect_error_top_k`, `ctc_dialect_errors_output`
   - Test output and diagnostics configuration. If enabled, results are written to `data_outputs/` as JSON.
   - Used by: `test.py`, `train_*.py` (test hooks), post-processing utilities.

- `use_initial_unfreeze` (bool) and `unfreeze_layers` (list[int])
   - Allow unfreezing a subset of wav2vec2 encoder layers from the start of training.
   - `unfreeze_layers` lists encoder layer indices to keep trainable.
   - Used by: `callbacks.py`, `train_*.py`.

- `wandb_project` (str) and `run_name` (str)
   - When using Weights & Biases for logging, these set the project and run name.
   - Used by: `train_*.py` when wandb integration is enabled in code.

Notes:
- Some commented options may appear in older versions of `config.yml` (e.g. `joint_checkpoint_path`, `baseline_checkpoint_path`). If present, training scripts will look for these to resume or evaluate specific checkpoints.

## Typical commands and which script to run

- Prepare processed datasets (required before training if `load_processed_dataset: False` or to regenerate):

```bash
python data_prep.py
```

- Train the concept/bottleneck head (framewise articulatory feature prediction):

```bash
python train_concept_layer.py
```

- Train the downstream task head (phoneme or other task) using either concept vectors or wav2vec2 features:

```bash
python train_task_layer.py
```

- Run joint training (optimize concept + task together):

```bash
python train_joint.py
```

- Train baseline (direct phoneme predictions without CBM):

```bash
python train_baseline.py
```

- Run probes to predict features:

```bash
python probe.py
```
