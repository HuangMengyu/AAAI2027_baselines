"""Unsupervised SmdCLR components adapted to the backbone's B x B losses.

Each view has its own batch-instance graph. MID retains the matched diagonal
in the cross-view candidate set; graph construction excludes actual self-edges.
No labels or semantic consistency classifier are used.
"""

import math

import torch
import torch.nn.functional as F


def validate_settings(temperature, mid_weight=1.0):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("smdclr_temperature must be finite and positive")
    if not math.isfinite(mid_weight) or mid_weight < 0:
        raise ValueError("mid_weight must be finite and nonnegative")


def instance_graph(h, temperature=0.2):
    """Row-normalized cosine graph within one view, without self-edges."""
    validate_settings(temperature)
    if h.shape[0] < 2:
        # There are no neighbors; graph_project falls back to the plain head.
        return h.new_zeros((h.shape[0], h.shape[0]))
    normalized = F.normalize(h, dim=1)
    logits = (normalized @ normalized.t()) / temperature
    diagonal = torch.eye(h.shape[0], dtype=torch.bool, device=h.device)
    return logits.masked_fill(diagonal, -torch.inf).softmax(dim=1)


def graph_project(h, head, temperature=0.2):
    """Reuse a two-linear-layer Sequential head as A sigma(A H W1) W2.

Existing intermediate layers (including BatchNorm) and parameter names are
preserved. Adjacency depends only on this view, including for temporal heads.
"""
    adjacency = instance_graph(h, temperature)
    if h.shape[0] < 2:
        return head(h)
    hidden = head[0](adjacency @ h)
    for layer in list(head)[1:-1]:
        hidden = layer(hidden)
    return head[-1](adjacency @ hidden)


def mid_loss(first, second, temperature=0.2):
    """Literal Eq. (15): entropy of the online similarity distribution.

The same distribution supplies weights and log-probabilities. Detaching its
weights would cancel this term's gradient. Cross-view diagonal pairs remain
valid candidates, matching the existing B x B InfoNCE convention.
"""
    validate_settings(temperature)
    logits = (F.normalize(first, dim=1) @ F.normalize(second, dim=1).t()) / temperature
    log_probabilities = F.log_softmax(logits, dim=1)
    return -(log_probabilities.exp() * log_probabilities).sum(dim=1).mean()
