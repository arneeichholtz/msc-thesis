# """Training callbacks for wav2vec2 CBM implementation."""

from transformers import TrainerCallback, Trainer
import math
import wandb
from gradnorm_pytorch import GradNormLossWeighter
import torch
from typing import Dict
from datasets import DatasetDict
from data_prep import BINARY_FEATURE_DIM, FEATURE_GROUPS_LABELS
import numpy as np

class LambdaSchedulerCallback(TrainerCallback):
    """Schedules the joint_lambda parameter during training."""
    
    def __init__(self, initial_lambda: float, final_lambda: float, max_steps: int, schedule: str = "linear"):
        self.initial_lambda = initial_lambda
        self.final_lambda = final_lambda
        self.max_steps = max_steps
        self.schedule = schedule

    def on_step_begin(self, args, state, control, model, **kwargs):
        # Calculate progress from 0.0 to 1.0
        progress = min(state.global_step / max(1, self.max_steps), 1.0)
        
        if self.schedule == "linear":
            current_lambda = self.initial_lambda - progress * (self.initial_lambda - self.final_lambda)
        elif self.schedule == "cosine":
            current_lambda = self.final_lambda + 0.5 * (self.initial_lambda - self.final_lambda) * (1 + math.cos(math.pi * progress))
        else:
            current_lambda = self.initial_lambda  # constant

        # Handle unwrapped models (e.g., DDP/DataParallel)
        unwrap_model = model.module if hasattr(model, "module") else model
        unwrap_model.joint_lambda = current_lambda

    def on_log(self, args, state, control, model, logs=None, **kwargs):
        # Add the current lambda to the WandB logs
        unwrap_model = model.module if hasattr(model, "module") else model
        if logs is not None:
            logs["joint_lambda"] = unwrap_model.joint_lambda
            
            # Explicitly log to wandb so it's guaranteed to appear in the dashboard
            if wandb.run is not None:
                wandb.log({"train/joint_lambda": unwrap_model.joint_lambda}, commit=False)



class GradNormTrainer(Trainer):
    def __init__(self, *args, gradnorm_weighter: GradNormLossWeighter, **kwargs):
        super().__init__(*args, **kwargs)
        self.gradnorm_weighter = gradnorm_weighter
        if self.gradnorm_weighter is not None:
            self.gradnorm_weighter.accelerator = self.accelerator

    def training_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            outputs = model(**inputs)
            task_loss = outputs.get("task_loss")
            concept_loss = outputs.get("concept_loss")

            if task_loss is None or concept_loss is None:
                raise ValueError("GradNorm requires both task_loss and concept_loss.")

        if self.args.gradient_accumulation_steps > 1:
            scale = self.args.gradient_accumulation_steps
            task_loss = task_loss / scale
            concept_loss = concept_loss / scale

        self.gradnorm_weighter.backward([task_loss, concept_loss])
        return (task_loss + concept_loss).detach()



def print_dataset_statistics(dataset: DatasetDict):
    """Calculates and prints the distribution of binary features in the dataset."""
    feature_labels = [label for group in FEATURE_GROUPS_LABELS for label in group.labels]
    # feature_labels = [f"{group.name}_{label}" for group in FEATURE_GROUPS_LABELS for label in group.labels]

    print("feature labels: ", feature_labels)
    
    print("\n" + "="*80)
    print("DATASET FEATURE DISTRIBUTION")
    print("="*80)

    total_feature_counts = np.zeros(BINARY_FEATURE_DIM, dtype=np.int64)

    for split in dataset.keys():
        print(f"\n--- Split: {split} ---")
        split_data = dataset[split]
        
        # Initialize counters
        feature_counts = np.zeros(BINARY_FEATURE_DIM, dtype=np.int64)
        for item in split_data:
            
            labels = np.array(item['concept_labels'])
            
            # Sum down the frame axis (axis 0)
            feature_counts += labels.sum(axis=0).astype(np.int64)

        split_counts = {label: int(feature_counts[i]) for i, label in enumerate(feature_labels)}
        print(split_counts)

        total_feature_counts += feature_counts

    total_counts = {label: int(total_feature_counts[i]) for i, label in enumerate(feature_labels)}
    print("\n--- Split: all ---")
    print(total_counts)
            
    print("="*80 + "\n")

