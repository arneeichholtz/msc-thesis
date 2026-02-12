"""Custom wav2vec2 head for articulatory feature prediction."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss

from transformers import Wav2Vec2Model, Wav2Vec2PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput


class Wav2Vec2ForArticulatoryFeatures(Wav2Vec2PreTrainedModel):
    """Concept Bottleneck variant of wav2vec2 producing articulatory features."""

    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_fn = BCEWithLogitsLoss(reduction="none")

        self.wav2vec2.requires_grad_(False)         # Freeze all wav2vec2 params

        self.post_init()

    def _compute_attention_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        target_length: int,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Downsample the attention mask to match wav2vec2 time steps."""
        if attention_mask is None:
            return torch.ones((batch_size, target_length), device=device)

        output_lengths = self.wav2vec2._get_feat_extract_output_lengths(
            attention_mask.sum(dim=1)
        )
        output_lengths = output_lengths.to(device)

        arange = torch.arange(target_length, device=device)
        mask = arange.unsqueeze(0) < output_lengths.unsqueeze(1)
        return mask

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> SequenceClassifierOutput:
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_hidden_states=kwargs.get("output_hidden_states", False),
            return_dict=True,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.classifier(hidden_states)

        batch_size = input_values.size(0)
        device = input_values.device

        loss = None
        effective_logits = logits

        if labels is not None:
            if labels.dim() == 2:  # allow flattened labels
                labels = labels.unsqueeze(1)

            time_dim = logits.size(1)
            label_time_dim = labels.size(1)
            usable_length = min(time_dim, label_time_dim)

            effective_logits = logits[:, :usable_length, :]
            labels = labels[:, :usable_length, :]

            frame_mask = self._compute_attention_mask(
                attention_mask,
                usable_length,
                batch_size,
                device,
            )

            frame_mask = frame_mask.unsqueeze(-1).type_as(effective_logits)

            raw_loss = self.loss_fn(effective_logits, labels)
            masked_loss = raw_loss * frame_mask
            normalizer = frame_mask.sum().clamp(min=1.0)
            loss = masked_loss.sum() / normalizer

        return SequenceClassifierOutput(
            loss=loss,
            logits=effective_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
