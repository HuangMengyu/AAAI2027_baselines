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
        self.lsoftmax = nn.LogSoftmax()
        self.device = device
        
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, final_out_channels // 2),
            nn.BatchNorm1d(final_out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(final_out_channels // 2, final_out_channels // 4),
        )

        self.seq_transformer = Seq_Transformer(patch_size=self.num_channels, dim=hidden_dim, depth=4, heads=4, mlp_dim=64)

    def forward(self, features_aug1, features_aug2): # aug1 is used on z level, aug 2 is used on h level
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
            nce += torch.sum(torch.diag(self.lsoftmax(total)))
        nce /= -1. * batch * self.timestep

        z_full = self.seq_transformer(h_aug1)
        c_full = self.projection_head(z_full)

        return nce, c_full