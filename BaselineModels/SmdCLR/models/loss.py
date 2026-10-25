import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCE(nn.Module):
    def __init__(self, temperature, device, filter=False):
        super(InfoNCE, self).__init__()
        self.device = device
        self.temperature = temperature
        self.filter = filter  # Compatibility setting; never changes vanilla InfoNCE.

    def forward(self, out_1, out_2):
        out_1 = F.normalize(out_1, dim=1)
        out_2 = F.normalize(out_2, dim=1)
        similarity = torch.mm(out_1, out_2.t())
        # Each row contains its matched positive once and all other negatives.
        loss = -F.log_softmax(similarity / self.temperature, dim=1).diagonal().mean()
        negative_mask = ~torch.eye(similarity.shape[0], dtype=torch.bool,
                                   device=similarity.device)
        # Preserve the two obsolete counter positions for existing callers.
        return loss, similarity.diagonal(), similarity[negative_mask], 0, 0


def loss_ntxent(features, device, filter=False):
    fn = InfoNCE(temperature=0.2, device=device, filter=filter)
    return fn(features[0], features[1])
