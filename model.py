"""Custom wav2vec2 head for articulatory feature prediction."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
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
        device: torch.device
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
                device
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


class LinearCTCModel(nn.Module):
    def __init__(self, input_dim: int = 29, output_dim: int = 40):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.ctc_loss = nn.CTCLoss(blank=0, reduction="mean")           # Zero_infitiy to True means infinite losses are set to 0. This can happen when e.g. the target seq is longer than the input

    def forward(self, input_values, attention_mask, labels):
        # input_values shape: (batch, max input length, input_dim) e.g. (32, 235, 29)
        # Logits shape: (batch, max input length, output_dim) e.g. (32, 235, 40)
        # attention_mask shape: (batch, max input length) e.g. (32, 235)
        # labels shape: (batch, max label length) e.g. (32, 71)

        logits = self.linear(input_values)
        log_probs = F.log_softmax(logits, dim=-1)       # CTC loss expects log_probs

        # CTCLoss expects (input length, batch, class)
        ctc_log_probs = log_probs.transpose(0, 1)

        # Sum over attention mask to get valid input lengths
        input_lengths = attention_mask.long().sum(dim=1)        # input_lengths shape: (batch) = (32) like [154, 142, 235, ...]
       
        # Labels use -100 as padding value
        target_lengths = (labels != -100).long().sum(dim=1)        # target_lengths shape: (batch) = (32) like [35, 34, 71, ...]
        
        # Flatten targets by removing padded positions
        targets = labels[labels != -100]                # Targets are flattened. Shape: e.g. (1258) -- the sum of target_lengths
            
        loss = self.ctc_loss(
            ctc_log_probs,
            targets,
            input_lengths,
            target_lengths
        )
        
        output = {"logits": logits, "loss": loss}
        return output


class Wav2Vec2ForJointBottleneck(Wav2Vec2PreTrainedModel):
    """Joint bottleneck model: wav2vec2 -> concepts -> phoneme CTC."""

    def __init__(self, config, num_concepts: int, phoneme_vocab_size: int, joint_lambda: float = 1.0):      # config associated with model checkpoint is used
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        # self.concept_head = nn.Linear(config.hidden_size, num_concepts)
        self.task_head = nn.Linear(config.hidden_size, phoneme_vocab_size)

        # self.concept_loss_fn = BCEWithLogitsLoss(reduction="none")
        self.ctc_loss_fn = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)  
        self.joint_lambda = float(joint_lambda)
        
        self.wav2vec2.requires_grad_(False)         # Freeze all wav2vec2 params by default
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


    def _get_output_lengths(
        self,
        attention_mask: Optional[torch.Tensor],
        time_steps: int,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if attention_mask is None:
            return torch.full((batch_size,), time_steps, device=device, dtype=torch.long)
        output_lengths = self.wav2vec2._get_feat_extract_output_lengths(
            attention_mask.sum(dim=1)
        )
        return output_lengths.to(device).to(torch.long).clamp(max=time_steps)


    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        concept_labels: Optional[torch.Tensor] = None,
        task_labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_hidden_states=kwargs.get("output_hidden_states", False),
            return_dict=True,
        )

        hidden_states = outputs.last_hidden_state
        # concept_logits = self.concept_head(hidden_states)

        task_logits = self.task_head(hidden_states)     # was task_head(concept_logits)

        batch_size = input_values.size(0)
        device = input_values.device

        concept_loss = None
        # if concept_labels is not None:
        #     if concept_labels.dim() == 2:
        #         concept_labels = concept_labels.unsqueeze(1)

        #     time_dim = concept_logits.size(1)
        #     label_time_dim = concept_labels.size(1)
        #     usable_length = min(time_dim, label_time_dim)

        #     effective_logits = concept_logits[:, :usable_length, :]
        #     effective_labels = concept_labels[:, :usable_length, :]

        #     frame_mask = self._compute_attention_mask(
        #         attention_mask,
        #         usable_length,
        #         batch_size,
        #         device,
        #     )
        #     frame_mask = frame_mask.unsqueeze(-1).type_as(effective_logits)
        #     label_mask = (effective_labels != -100).type_as(effective_logits)
        #     frame_mask = frame_mask * label_mask

        #     raw_loss = self.concept_loss_fn(effective_logits, effective_labels)
        #     masked_loss = raw_loss * frame_mask
        #     normalizer = frame_mask.sum().clamp(min=1.0)
        #     concept_loss = masked_loss.sum() / normalizer

        task_loss = None
        if task_labels is not None:
            log_probs = F.log_softmax(task_logits, dim=-1)
            ctc_log_probs = log_probs.transpose(0, 1)
            input_lengths = self._get_output_lengths(
                attention_mask,
                task_logits.size(1),
                batch_size,
                device,
            )
            target_lengths = (task_labels != -100).long().sum(dim=1)
            targets = task_labels[task_labels != -100]

            task_loss = self.ctc_loss_fn(
                ctc_log_probs,
                targets,
                input_lengths,
                target_lengths,
            )

        loss = task_loss
        # loss_scalar = task_loss.item() / concept_loss.item() if concept_loss is not None and task_loss is not None else 1.0
        # loss = None
        
        if task_loss is not None and concept_loss is not None:
            loss =  task_loss + self.joint_lambda * concept_loss
        elif task_loss is not None:
            loss = task_loss
        elif concept_loss is not None:
            loss = self.joint_lambda * concept_loss

        return {
            "loss": loss,
            "logits": task_logits,
            # "concept_logits": concept_logits,
            # "concept_loss": concept_loss,
            # "task_loss": task_loss,
        }
    

class FrameLevelPhonemeModel(nn.Module):
    def __init__(self, input_dim: int = 29, output_dim: int = 40):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)       # Also does softmax internally

    def forward(self, input_values, attention_mask, labels=None):
        # input_values: (batch, seq_len, input_dim)
        # labels: (batch, seq_len)
        
        logits = self.linear(input_values)
        loss = None
        
        if labels is not None:
            # CrossEntropy expects (batch, classes, seq_len) or flattened versions
            logits_flat = logits.view(-1, logits.size(-1))
            labels_flat = labels.view(-1)
            
            loss = self.loss_fn(logits_flat, labels_flat)
            
        return {"logits": logits, "loss": loss}

