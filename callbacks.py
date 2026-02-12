"""Training utilities for staged wav2vec2 fine-tuning."""

from __future__ import annotations

from typing import List, Optional

from transformers import TrainerCallback, TrainerControl, TrainerState


def unfreeze_encoder_layers(model, layer_indices: List[int]) -> None:
    """Unfreeze specific wav2vec2 encoder transformer layers by index."""
    if layer_indices is None:
        return

    encoder_layers = model.wav2vec2.encoder.layers

    for layer_idx in layer_indices:
        for param in encoder_layers[layer_idx].parameters():
            param.requires_grad = True


class UnfreezingCallback(TrainerCallback):
    """Unfreeze the wav2vec2 encoder (Transformer layers) after a specified number of steps."""

    def __init__(self, model, thaw_step: int = 1000, unfreeze_layers: Optional[List[int]] = None):
        self.model = model
        self.thaw_step = thaw_step
        self.unfreeze_layers = unfreeze_layers
    
    def on_step_begin(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        if state.global_step == self.thaw_step:
            if self.unfreeze_layers is not None:
                # Unfreeze only specific transformer layers
                unfreeze_encoder_layers(self.model, self.unfreeze_layers)
                print(f"[UnfreezingCallback] Unfroze wav2vec2 encoder layers {self.unfreeze_layers} at step {self.thaw_step}.")
            
            else:
                # Unfreeze the entire encoder structure
                for param in self.model.wav2vec2.encoder.parameters():
                    param.requires_grad = True
                print(f"[UnfreezingCallback] Unfroze wav2vec2 encoder at step {self.thaw_step}.")
        return control
