# """Training callbacks for wav2vec2 CBM implementation."""

from transformers import TrainerCallback
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



class JointTrainer(Trainer):
    """Custom Trainer to track and log internal concept_loss and task_loss components, and apply GradNorm."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._custom_loss_tracking = {"task_loss": 0.0, "concept_loss": 0.0, "count": 0}
        self.alpha_gradnorm = 1.5
        self.gradnorm_eps = 1e-6
        self.gradnorm_min_weight = 1e-3
        self.gradnorm_max_weight = 1e3

    def create_optimizer(self):
        optimizer = super().create_optimizer()
        model = self.model
        weights = model.loss_weights if hasattr(model, 'loss_weights') else model.module.loss_weights

        for group in optimizer.param_groups:
            group["params"] = [param for param in group["params"] if param is not weights]

        self.weights_optimizer = torch.optim.Adam([weights], lr=0.01)
        return optimizer

    def _safe_weighted_sum(self, task_loss: torch.Tensor, concept_loss: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        safe_losses = torch.nan_to_num(
            torch.stack([task_loss, concept_loss]),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        safe_weights = torch.clamp(weights.detach(), min=self.gradnorm_min_weight, max=self.gradnorm_max_weight)
        return (safe_weights * safe_losses).sum()

    def training_step(self, model: nn.Module, inputs: Dict[str, torch.Tensor], num_items_in_batch=None) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        # Record initial task and concept loss on the very first forward pass
        if not hasattr(self, "initial_task_loss"):
            with torch.no_grad():
                outputs = model(**inputs)
                if outputs.get("task_loss") is not None and outputs.get("concept_loss") is not None:
                    self.initial_task_loss = outputs["task_loss"].detach()
                    self.initial_concept_loss = outputs["concept_loss"].detach()

        outputs = model(**inputs)
        loss = None
        
        # When GradNorm takes its first backward step, it modifies weights directly which breaks the computation graph
        # for standard backward of the composed loss term if done carelessly. Let's make sure the composed loss
        # is reconstructed from the fresh detached versions of weights for safety if using Accelerator mixed precision.
        
        task_loss = outputs.get("task_loss")
        concept_loss = outputs.get("concept_loss")

        _model = model.module if hasattr(model, 'module') else model

        if task_loss is not None and concept_loss is not None and hasattr(self, "initial_task_loss"):
            shared_features = outputs.get("shared_features")
            weights = _model.loss_weights

            finite_losses = torch.isfinite(task_loss) and torch.isfinite(concept_loss)

            if shared_features is not None and shared_features.requires_grad and finite_losses:
                
                w_task = weights[0]
                w_concept = weights[1]
                
                weighted_task_loss = w_task * task_loss
                weighted_concept_loss = w_concept * concept_loss

                # print(f"Weighted task loss: {weighted_task_loss.item():.4f}, Weighted concept loss: {weighted_concept_loss.item():.4f}")

                # 1. Gradients of each loss w.r.t the shared representations instead of network parameters.
                # This completely avoids gradient checkpointing issues inside the Wav2Vec2 encoder!
                grad_task = torch.autograd.grad(task_loss, shared_features, retain_graph=True, allow_unused=True)[0]
                grad_concept = torch.autograd.grad(concept_loss, shared_features, retain_graph=True, allow_unused=True)[0]

                def compute_norm_with_weight(grad, weight, name=""):
                    if grad is None:
                        return torch.zeros((), device=weight.device, dtype=weight.dtype)
                    # Re-apply the weight to the norm of the gradient
                    # This makes the norm differentiable w.r.t the weight 'w' 
                    # but doesn't require the second derivative of the original loss.
                    grad_norm = torch.norm(grad.detach())
                    return grad_norm * torch.clamp(weight, min=self.gradnorm_min_weight, max=self.gradnorm_max_weight)

                norm_task = compute_norm_with_weight(grad_task, weights[0], "task")
                norm_concept = compute_norm_with_weight(grad_concept, weights[1], "concept")

                norms = torch.stack([norm_task, norm_concept])
                mean_norm = norms.mean().detach()

                # 2. Relative inverse training rates
                loss_ratio_task = task_loss.detach() / torch.clamp(self.initial_task_loss, min=self.gradnorm_eps)
                loss_ratio_concept = concept_loss.detach() / torch.clamp(self.initial_concept_loss, min=self.gradnorm_eps)

                loss_ratios = torch.stack([loss_ratio_task, loss_ratio_concept])
                mean_loss_ratio = torch.clamp(loss_ratios.mean().detach(), min=self.gradnorm_eps)

                inverse_training_rates = loss_ratios / mean_loss_ratio
                target_norms = (torch.clamp(mean_norm, min=self.gradnorm_eps) * (inverse_training_rates ** self.alpha_gradnorm)).detach()

                # 3. GradNorm loss & backward pass on weights
                grad_norm_loss = torch.sum(torch.abs(norms - target_norms))
                # print(f"GradNorm loss: {grad_norm_loss.item():.4f}, norm_task: {norm_task.item():.4f}, norm_concept: {norm_concept.item():.4f}, target_norm_task: {target_norms[0].item():.4f}, target_norm_concept: {target_norms[1].item():.4f}")

                self.weights_optimizer.zero_grad()
                
                if hasattr(self, "accelerator"):
                    self.accelerator.backward(grad_norm_loss, retain_graph=True)
                else:
                    grad_norm_loss.backward(retain_graph=True)
                    
                self.weights_optimizer.step()

                # 4. Renormalize weights (out-of-place to preserve graph)
                with torch.no_grad():
                    weights.clamp_(min=self.gradnorm_min_weight, max=self.gradnorm_max_weight)
                    normalize_coeff = 2.0 / torch.clamp(weights.sum(), min=self.gradnorm_eps)
                    weights.mul_(normalize_coeff)
                
                # Reconstruct model loss with detached weights so only model params are updated in this backward pass
                loss = self._safe_weighted_sum(task_loss, concept_loss, weights)
            elif not finite_losses:
                loss = self._safe_weighted_sum(task_loss, concept_loss, weights)
        
        # Fallback if no GradNorm triggered
        if loss is None:
            if task_loss is not None and concept_loss is not None:
                loss = self._safe_weighted_sum(task_loss, concept_loss, _model.loss_weights)
            else:
                raw_loss = outputs.get("loss") if isinstance(outputs, dict) else outputs.loss
                loss = torch.nan_to_num(raw_loss, nan=0.0, posinf=0.0, neginf=0.0)

        if hasattr(self, "accelerator"):
            # Ensure we do not hit inplace modifications issues
            self.accelerator.backward(loss)
        else:
            loss.backward()

        if task_loss is not None:
             self._custom_loss_tracking["task_loss"] += task_loss.detach().item()
        if concept_loss is not None:
             self._custom_loss_tracking["concept_loss"] += concept_loss.detach().item()
        self._custom_loss_tracking["count"] += 1

        return loss.detach()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Compute loss implementation is retained for eval step
        outputs = model(**inputs)
        loss = outputs.get("loss") if isinstance(outputs, dict) else outputs.loss
        return (loss, outputs) if return_outputs else loss

    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        count = self._custom_loss_tracking["count"]
        if count > 0:
            logs["train/task_loss"] = self._custom_loss_tracking["task_loss"] / count
            logs["train/concept_loss"] = self._custom_loss_tracking["concept_loss"] / count
            self._custom_loss_tracking["task_loss"] = 0.0
            self._custom_loss_tracking["concept_loss"] = 0.0
            self._custom_loss_tracking["count"] = 0
            
            # Log current dynamic weights from GradNorm
            model = self.model.module if hasattr(self.model, 'module') else self.model
            if hasattr(model, 'loss_weights'):
                logs["train/weight_task"] = model.loss_weights[0].item()
                logs["train/weight_concept"] = model.loss_weights[1].item()
            
        super().log(logs, *args, **kwargs)






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
