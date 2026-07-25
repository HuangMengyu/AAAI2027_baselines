import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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
    depth=4
    ):
        super(TC, self).__init__()
        self.num_channels = final_out_channels
        self.timestep = timesteps
        self.Wk = nn.ModuleList([nn.Linear(hidden_dim, self.num_channels) for i in range(self.timestep)])
        self.lsoftmax = nn.LogSoftmax()
        self.device = device
        
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, final_out_channels // 2),
            nn.BatchNorm1d(final_out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(final_out_channels // 2, final_out_channels // 4),
        )

        self.seq_transformer = Seq_Transformer(patch_size=self.num_channels, dim=hidden_dim, depth=depth, heads=4, mlp_dim=64)

        self.self_attention = Attention(dim=self.num_channels, heads=4, dropout=0.1)
        self.cross_sample_attention = CrossSampleAttention(dim=self.num_channels, num_heads=4)
        
    

    def forward(
        self,
        features_aug1,
        features_aug2,
        index=None,
        epoch=None,
        lambda_threshold=None,
        filter=False,
    ): # aug1 is used on z level, aug 2 is used on h level
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

        use_glofnd = (
            lambda_threshold is not None
            and getattr(lambda_threshold, "glofnd_type", None) is not None
            and index is not None
            and epoch is not None
        )

        if use_glofnd:
            neg_self_mask = torch.ones((batch, batch), device=self.device)
            neg_self_mask = neg_self_mask.fill_diagonal_(0)
            sim_list = [
                torch.mm(
                    F.normalize(encode_samples[i], dim=1),
                    F.normalize(pred[i], dim=1).transpose(0, 1),
                ).detach()
                for i in np.arange(0, self.timestep)
            ]
            lambda_threshold.update(
                sim=sim_list,
                idx=index,
                neg_self_mask=neg_self_mask,
                epoch=epoch,
            )

        if filter and not use_glofnd:
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
            if use_glofnd:
                sim_i = torch.mm(
                    F.normalize(encode_samples[i], dim=1),
                    F.normalize(pred[i], dim=1).transpose(0, 1),
                ).detach()
                neg_mask, neg_count, _ = lambda_threshold.get_mask(
                    sim=sim_i,
                    idx=index,
                    neg_self_mask=neg_self_mask,
                    epoch=epoch,
                )
                exp_total = torch.exp(total)
                pos = torch.diag(exp_total)
                neg = torch.sum(exp_total * neg_mask, dim=1)
                neg = neg / neg_count.squeeze(1).clamp_min(1) * (batch - 1)
                lsfmax = torch.log(pos / (pos + neg + 1e-8))
                nce = nce + torch.sum(lsfmax)
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
