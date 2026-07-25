import torch
import torch.nn as nn
import numpy as np
from .attention import Seq_Transformer



class TC(nn.Module):
    def __init__(self, 
    device,
    final_out_channels=127, 
    timesteps=50, 
    hidden_dim=64,
    ):
        super(TC, self).__init__()
        self.num_channels = final_out_channels
        self.timestep = timesteps
        self.Wk = nn.ModuleList([nn.Linear(hidden_dim, self.num_channels) for i in range(self.timestep)])
        # use explicit dim for numerical stability when computing log-probabilities
        self.lsoftmax = nn.LogSoftmax(dim=1)
        self.device = device
        
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, final_out_channels // 2),
            nn.BatchNorm1d(final_out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(final_out_channels // 2, final_out_channels // 4),
        )

        self.seq_transformer = Seq_Transformer(patch_size=self.num_channels, dim=hidden_dim, depth=4, heads=4, mlp_dim=64)

    def forward(self, features_aug1, features_aug2, filter=False, topk=0.25, fn_mode='cancel'): # aug1 is used on z level, aug 2 is used on h level
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
        for i in np.arange(0, self.timestep):  # calculate the temporal loss per each timestep and then take average
            total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))
            if filter:
                # compute in log-domain for stability
                neg_mask = ~torch.eye(batch, dtype=torch.bool, device=self.device)
                NEG_INF = torch.tensor(-1e9, device=total.device, dtype=total.dtype)

                # pos log-score (batch,)
                pos_log = torch.diag(total)

                # keep only negative positions, set others to -inf
                sim_neg = total.masked_fill(~neg_mask, NEG_INF)

                if batch > 1:
                    neg_candidates = sim_neg
                    topk_num = min(max(int(batch * topk), 1), batch - 1)
                    thresholds, _ = torch.topk(neg_candidates, topk_num, dim=1)
                    per_row_threshold = thresholds[:, -1].unsqueeze(1)
                    fn_mask = neg_candidates >= per_row_threshold

                    if fn_mode == 'attract':
                        fn_logsum = torch.logsumexp(sim_neg.masked_fill(~fn_mask, NEG_INF), dim=1)
                        pos_prime = torch.logsumexp(torch.stack([pos_log, fn_logsum], dim=1), dim=1)
                    else:
                        pos_prime = pos_log

                    sim_neg_filtered = sim_neg.masked_fill(fn_mask, NEG_INF)
                    log_neg_sum = torch.logsumexp(sim_neg_filtered, dim=1)

                    # add log-domain contribution (pos_prime - log(exp(pos_prime)+exp(log_neg_sum)))
                    log_denom = torch.logsumexp(torch.stack([pos_prime, log_neg_sum], dim=1), dim=1)
                    nce += torch.sum(pos_prime - log_denom)
                else:
                    # no negatives, use log-softmax fallback
                    nce += torch.sum(torch.diag(self.lsoftmax(total)))
            else:
                nce += torch.sum(torch.diag(self.lsoftmax(total)))
        nce /= -1. * batch * self.timestep

        z_full = self.seq_transformer(h_aug1)
        c_full = self.projection_head(z_full)

        return nce, c_full
