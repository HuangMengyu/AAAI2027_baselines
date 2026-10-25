import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def validate_nsf_threshold(value):
    value = float(value)
    if not math.isfinite(value) or (value != -1 and value <= 0):
        raise ValueError("nsf_threshold must be -1 (disabled) or a finite positive value")
    return value


@torch.no_grad()
def nsf_keep_mask(similarity, nsf_threshold=-1):
    """Keep matched positives and negatives with s_negative / s_positive < lambda.

    Rows with s_positive <= 1e-8 stay unfiltered: the ratio is undefined
    near zero and reverses the hardness ordering for negative denominators.
    """
    nsf_threshold = validate_nsf_threshold(nsf_threshold)
    if nsf_threshold == -1:
        return torch.ones_like(similarity, dtype=torch.bool)
    positive = similarity.diagonal().unsqueeze(1)
    keep = (similarity < nsf_threshold * positive) | (positive <= 1e-8)
    keep.fill_diagonal_(True)
    return keep


class InfoNCE(nn.Module):
    def __init__(self, temperature, device, nsf_threshold=-1):
        super(InfoNCE, self).__init__()
        self.device = device
        self.temperature = temperature
        self.nsf_threshold = validate_nsf_threshold(nsf_threshold)

    def forward(self, out_1, out_2):
        out_1 = F.normalize(out_1, dim=1)
        out_2 = F.normalize(out_2, dim=1)
        similarity = torch.mm(out_1, out_2.t())
        keep = nsf_keep_mask(similarity, self.nsf_threshold)
        logits = (similarity / self.temperature).masked_fill(~keep, -torch.inf)
        loss = -F.log_softmax(logits, dim=1).diagonal().mean()

        negative_mask = keep.clone()
        negative_mask.fill_diagonal_(False)
        # Preserve the two obsolete counter positions for existing callers.
        return loss, similarity.diagonal(), similarity[negative_mask], 0, 0


def loss_ntxent(features, device, nsf_threshold=-1):
    fn = InfoNCE(temperature=0.2, device=device, nsf_threshold=nsf_threshold)
    return fn(features[0], features[1])
