
import os
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class NTXentLoss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)


class InfoNCE(nn.Module):
    def __init__(self, temperature, device, filter=False, topk=0.25, fn_mode = 'cancel'):
        super(InfoNCE, self).__init__()
        self.device = device
        self.temperature = temperature
        self.filter = filter
        self.topk = topk
        self.fn_mode = fn_mode

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, out_1, out_2):
        """
            assume out_1 and out_2 are normalized
            out_1: [batch_size, dim]
            out_2: [batch_size, dim]
        """
        # gather representations in case of distributed training
        bsz = out_1.shape[0]

        out_1 = F.normalize(out_1, dim=1)
        out_2 = F.normalize(out_2, dim=1)

        out_1_dist = out_1
        out_2_dist = out_2

        out = torch.cat([out_1, out_2], dim=0)
        out_dist = torch.cat([out_1_dist, out_2_dist], dim=0)

        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.temperature)

        pos = torch.cat([pos, pos], dim=0)
        # print('pos:', pos.shape)

        similarity = torch.exp(torch.mm(out, out_dist.t()) / self.temperature)
        # similarity = torch.mm(out, out_dist.t())

        neg_mask = self.mask_correlated_samples(bsz).to(self.device)

        neg_temp = similarity * neg_mask

        if self.filter:
            neg_temp = torch.where(neg_temp == 0.0, torch.tensor(-9e15).to(self.device), neg_temp) # avoid the positive indices got selected
            topk_num = max(int(bsz * self.topk), 1)
            thresholds, _ = torch.topk(neg_temp, topk_num, dim=1)
            per_row_threshold = thresholds[:, -1].unsqueeze(1) # [2*bsz, 1]
            print("self.fn_mode", self.fn_mode)
            if self.fn_mode == 'attract':
                fn_mask = (neg_temp > per_row_threshold)
                fn_scores = torch.where(fn_mask, neg_temp, 0.0)
                fn_sum_per_row = torch.sum(fn_scores , dim=1)
                # print('fn_sum_per_row:', fn_sum_per_row.shape)
                pos = pos + fn_sum_per_row
            # neg_temp = neg_temp * neg_mask
            neg_temp = torch.where((neg_temp > per_row_threshold) & (neg_temp == -9e15), 0.0, neg_temp) # filter false negatives
            

        neg = torch.sum(neg_temp, 1)
        

        logits = -torch.log(pos / (pos + neg))
        logits_re = logits.view(2, -1)
        pos_loss = torch.mean(logits_re, dim=0)

        return pos_loss

def loss_ntxent(features, device, filter=False, topk=0.25, fn_mode='cancel'):
    fn = InfoNCE(temperature=0.2, device=device, filter=filter, topk=topk, fn_mode=fn_mode) # topk: the percentage of each batch to be filtered
    pos_loss = fn(features[0], features[1])

    return pos_loss
