import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from .attention import Seq_Transformer, Attention


class CrossSampleAttention(nn.Module):
    """
    Returns a single attention map across samples for each timestep.
    Input:  x : (B, T, D)
    Output: attn : (T, B, B)   
    """

    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )

    def forward(self, x):
        T, B, D = x.shape

        # # (B, T, D) → (T, B, D) so MHA attends across samples
        # x_perm = x.permute(1, 0, 2)

        # run attention
        _, attn = self.mha(
            x, x, x,
            need_weights=True,
            average_attn_weights=True
        )
        # attn shape = (T, B, B)


        return attn




class TC(nn.Module):
    def __init__(self, 
    device,
    final_out_channels=128, 
    timesteps=50, 
    hidden_dim=64,
    depth=4,
    debiased=False,
    tau_plus=0.1,
    temperature=0.2
    ):
        super(TC, self).__init__()
        self.num_channels = final_out_channels
        self.timestep = timesteps
        self.Wk = nn.ModuleList([nn.Linear(hidden_dim, self.num_channels) for i in range(self.timestep)])
        self.lsoftmax = nn.LogSoftmax()
        self.device = device
        self.debiased = debiased
        self.tau_plus = tau_plus
        self.temperature = temperature
        if not 0 <= self.tau_plus < 1:
            raise ValueError("tau_plus must be in [0, 1).")
        
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, final_out_channels // 2),
            nn.BatchNorm1d(final_out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(final_out_channels // 2, final_out_channels // 4),
        )

        self.seq_transformer = Seq_Transformer(patch_size=self.num_channels, dim=hidden_dim, depth=depth, heads=4, mlp_dim=64)

        self.self_attention = Attention(dim=self.num_channels, heads=4, dropout=0.1)
        self.cross_sample_attention = CrossSampleAttention(dim=self.num_channels, num_heads=4)
        
    def debiased_negative_sum(self, pos, neg, num_negatives):
        neg_sum = torch.sum(neg, dim=-1)
        ng = (-self.tau_plus * num_negatives * pos + neg_sum) / (1 - self.tau_plus)
        lower_bound = num_negatives * math.exp(-1.0 / self.temperature)
        return torch.clamp(ng, min=lower_bound)

    def debiased_temporal_loss(self, encode_t, pred_t):
        encode_t = F.normalize(encode_t, dim=1)
        pred_t = F.normalize(pred_t, dim=1)
        logits = torch.mm(encode_t, torch.transpose(pred_t, 0, 1))
        exp_logits = torch.exp(logits / self.temperature)

        pos = torch.diag(exp_logits)
        neg_mask = torch.ones_like(exp_logits, dtype=torch.bool)
        neg_mask.fill_diagonal_(False)
        neg = exp_logits[neg_mask].view(exp_logits.shape[0], -1)
        ng = self.debiased_negative_sum(pos, neg, exp_logits.shape[0] - 1)
        return torch.sum(torch.log(pos / (pos + ng)))
    

    def forward(self, features_aug1, features_aug2, filter = False): # aug1 is used on z level, aug 2 is used on h level
        h_aug1 = features_aug1  # features are (batch_size, #channels, seq_len)
        seq_len = h_aug1.shape[2]
        h_aug1 = h_aug1.transpose(1, 2)

        h_aug2 = features_aug2
        h_aug2 = h_aug2.transpose(1, 2)

        batch = h_aug1.shape[0]
        t_samples = torch.randint(seq_len - self.timestep, size=(1,)).long().to(self.device)  # randomly pick time stamps

        nce = 0  # average over timestep and batch
        encode_samples = torch.empty((self.timestep, batch, self.num_channels)).float().to(self.device)

        for i in np.arange(1, self.timestep + 1):
            encode_samples[i - 1] = h_aug2[:, t_samples + i, :].view(batch, self.num_channels)
        forward_seq = h_aug1[:, :t_samples + 1, :]

        z_t = self.seq_transformer(forward_seq)

        # compute the nce loss
        pred = torch.empty((self.timestep, batch, self.num_channels)).float().to(self.device)
        for i in np.arange(0, self.timestep):
            linear = self.Wk[i]
            pred[i] = linear(z_t)

        if filter:
            # print("Filtering")
            ## apply self-attention here for capturing temporal information
            pred = pred.transpose(0, 1)  # (batch, timestep, num_channels)
            pred = self.self_attention(pred)  # apply self-attention across samples
            pred = pred.transpose(0, 1)  # (timestep, batch, num_channels)
            # print("Pred shape after self-attention:", pred.shape)

            encode_samples = encode_samples.transpose(0, 1)  # (batch, timestep, num_channels)
            encode_samples = self.self_attention(encode_samples)  # apply self-attention across samples
            encode_samples = encode_samples.transpose(0, 1)  # (timestep, batch, num_channels)
            # print("encode_samples after self-attention:", pred.shape)

            ## add cross-sample attention for finding out similar samples in the batch 
            sample_weights = self.cross_sample_attention(encode_samples)  # (timestep, batch, batch)
            reverse_sample_weights = torch.ones(sample_weights.shape).float().to(self.device) - sample_weights  # we want to give less weight to similar samples
            
            # print("Sample weights shape:", reverse_sample_weights.shape)
        for i in np.arange(0, self.timestep):  # calculate the temporal loss per each timestep and then take average
            total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))
            if self.debiased:
                nce = nce + self.debiased_temporal_loss(encode_samples[i], pred[i])
            elif filter:
                weights = reverse_sample_weights[i].clone()
                weights.fill_diagonal_(1.0)
                total = torch.exp(total) * weights
                pos = torch.diag(total)
                neg = torch.sum(total, dim=1) - pos
                assert pos.shape == neg.shape
                lsfmax = torch.log(pos / (pos + neg))
                nce = nce + torch.sum(lsfmax)
            else:
                nce = nce + torch.sum(torch.diag(self.lsoftmax(total)))
        nce /= -1. * batch * self.timestep

        z_full = self.seq_transformer(h_aug1)
        c_full = self.projection_head(z_full)

        return nce, z_full, c_full
