"""Training utilities for staged wav2vec2 fine-tuning."""

from __future__ import annotations

from typing import Optional

from transformers import TrainerCallback, TrainerControl, TrainerState


def freeze_encoder(model) -> None:
    """Freeze wav2vec2 encoder transformer layers."""
    for param in model.wav2vec2.encoder.parameters():
        param.requires_grad = False


class FreezingCallback(TrainerCallback):
    """Unfreeze the wav2vec2 encoder after a specified number of steps."""

    def __init__(self, model, thaw_step: int = 10_000):
        self.model = model
        self.thaw_step = thaw_step

    def on_step_begin(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        if state.global_step == self.thaw_step:
            for param in self.model.wav2vec2.encoder.parameters():
                param.requires_grad = True
            print(f"[FreezingCallback] Unfroze wav2vec2 encoder at step {self.thaw_step}.")
        return control
