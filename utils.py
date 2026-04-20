# """Training callbacks for wav2vec2 CBM implementation."""

from transformers import TrainerCallback, Trainer
import math
import wandb

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







# from __future__ import annotations

# from typing import List, Optional

# from transformers import TrainerCallback, TrainerControl, TrainerState


# def unfreeze_encoder_layers(model, layer_indices: List[int]) -> None:
#     """Unfreeze specific wav2vec2 encoder transformer layers by index."""
#     if layer_indices is None:
#         return

#     encoder_layers = model.wav2vec2.encoder.layers

#     for layer_idx in layer_indices:
#         for param in encoder_layers[layer_idx].parameters():
#             param.requires_grad = True


# class UnfreezingCallback(TrainerCallback):
#     """Unfreeze the wav2vec2 encoder (Transformer layers) after a specified number of steps."""

#     def __init__(self, model, thaw_step: int = 1000, unfreeze_layers: Optional[List[int]] = None):
#         self.model = model
#         self.thaw_step = thaw_step
#         self.unfreeze_layers = unfreeze_layers
    
#     def on_step_begin(
#         self,
#         args,
#         state: TrainerState,
#         control: TrainerControl,
#         **kwargs,
#     ) -> Optional[TrainerControl]:
#         if state.global_step == self.thaw_step:
#             if self.unfreeze_layers is not None:
#                 # Unfreeze only specific transformer layers
#                 unfreeze_encoder_layers(self.model, self.unfreeze_layers)
#                 print(f"[UnfreezingCallback] Unfroze wav2vec2 encoder layers {self.unfreeze_layers} at step {self.thaw_step}.")
            
#             else:
#                 # Unfreeze the entire encoder structure
#                 for param in self.model.wav2vec2.encoder.parameters():
#                     param.requires_grad = True
#                 print(f"[UnfreezingCallback] Unfroze wav2vec2 encoder at step {self.thaw_step}.")
#         return control
