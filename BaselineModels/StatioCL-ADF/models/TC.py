import torch
import torch.nn as nn
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
        
    

    def forward(self, features_aug1, features_aug2, filter = False, segment_adf=None): # aug1 is used on z level, aug 2 is used on h level
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
        valid_negative_ratio = 0.0
        valid_row_count = 0
        for i in np.arange(0, self.timestep):  # calculate the temporal loss per each timestep and then take average
            total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))
            if segment_adf is not None:
                segment_adf = segment_adf.to(device=self.device)
                prior_seq_num = segment_adf.shape[1]
                target_pos = int(t_samples.item()) + int(i) + 1
                if seq_len > 1:
                    segment_idx = int(round(target_pos * (prior_seq_num - 1) / (seq_len - 1)))
                else:
                    segment_idx = 0
                segment_idx = max(0, min(segment_idx, prior_seq_num - 1))

                diag_mask = torch.eye(batch, dtype=torch.bool, device=self.device)
                labels = segment_adf[:, segment_idx].view(-1)
                neg_mask = labels.unsqueeze(1).ne(labels.unsqueeze(0)) & ~diag_mask
                denom_mask = neg_mask | diag_mask
                valid_rows = neg_mask.any(dim=1)
                if torch.any(valid_rows):
                    masked_total = total.masked_fill(~denom_mask, -1e12)
                    row_loss = torch.logsumexp(masked_total, dim=1) - torch.diag(total)
                    nce = nce + torch.sum(row_loss[valid_rows])
                    valid_row_count += int(valid_rows.sum().item())
                valid_negative_ratio += float(neg_mask.float().mean().detach().cpu().item())
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
        if segment_adf is not None:
            if valid_row_count > 0:
                nce = nce / valid_row_count
            else:
                nce = total.sum() * 0.0
            valid_negative_ratio = valid_negative_ratio / self.timestep
        else:
            nce /= -1. * batch * self.timestep
            valid_negative_ratio = float(batch - 1) / float(batch)

        z_full = self.seq_transformer(h_aug1)
        c_full = self.projection_head(z_full)

        return nce, z_full, c_full, valid_negative_ratio
