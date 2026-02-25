# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class LinearCTCModel(nn.Module):
#     def __init__(self, input_dim: int = 29, output_dim: int = 0):
#         super().__init__()
#         self.linear = nn.Linear(input_dim, output_dim)
#         self.ctc_loss = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

#     def forward(self, input_values, attention_mask, labels=None):
#         logits = self.linear(input_values)
#         log_probs = F.log_softmax(logits, dim=-1)

#         output = {"logits": logits}

#         if labels is not None:
#             # CTCLoss expects (time, batch, class)
#             ctc_log_probs = log_probs.transpose(0, 1)

#             # Sum over attention mask to get valid input lengths
#             input_lengths = attention_mask.long().sum(dim=1)

#             # Labels use -100 as padding sentinel
#             target_lengths = (labels != -100).long().sum(dim=1)

#             # Flatten targets by removing padded positions
#             targets = labels[labels != -100]

#             loss = self.ctc_loss(
#                 ctc_log_probs,
#                 targets,
#                 input_lengths,
#                 target_lengths,
#             )
#             output["loss"] = loss

#         return output
