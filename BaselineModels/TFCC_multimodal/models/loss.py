
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

        # Stable InfoNCE computation using log-sum-exp
        # sim: (2N, 2N) similarity logits (no exp yet)
        sim = torch.mm(out, out_dist.t()) / self.temperature

        # positive pair scores (2N,)
        pos_scores = torch.sum(out_1 * out_2, dim=-1) / self.temperature
        pos_scores = torch.cat([pos_scores, pos_scores], dim=0)

        if not self.filter:
            # loss_i = - (log_pos - logsumexp(all_scores))
            logsumexp_all = torch.logsumexp(sim, dim=1)
            loss_vec = - (pos_scores - logsumexp_all)
            loss_mat = loss_vec.view(2, bsz)
            pos_loss = torch.mean(loss_mat, dim=0)
            return pos_loss

        # FILTER mode: identify false negatives (fn) by topk on negative logits
        neg_mask = self.mask_correlated_samples(bsz).to(self.device)  # True for negatives
        NEG_INF = torch.tensor(-1e9, device=sim.device, dtype=sim.dtype)

        # keep only negative positions, set others to -inf for logsumexp safety
        sim_neg = sim.masked_fill(~neg_mask, NEG_INF)

        topk_num = max(int(bsz * self.topk), 1)
        thresholds, _ = torch.topk(sim_neg, topk_num, dim=1)
        per_row_threshold = thresholds[:, -1].unsqueeze(1)
        fn_mask = sim_neg >= per_row_threshold

        # If attract mode, incorporate selected false negatives into positive score (log-domain)
        if self.fn_mode == 'attract':
            fn_logsum = torch.logsumexp(sim_neg.masked_fill(~fn_mask, NEG_INF), dim=1)
            pos_prime = torch.logsumexp(torch.stack([pos_scores, fn_logsum], dim=1), dim=1)
        else:
            pos_prime = pos_scores

        # remove false negatives from negatives
        sim_neg_filtered = sim_neg.masked_fill(fn_mask, NEG_INF)
        log_neg_sum = torch.logsumexp(sim_neg_filtered, dim=1)

        # denominator in log-domain: log(exp(pos_prime) + exp(log_neg_sum))
        log_denom = torch.logsumexp(torch.stack([pos_prime, log_neg_sum], dim=1), dim=1)

        loss_vec = - (pos_prime - log_denom)
        loss_mat = loss_vec.view(2, bsz)
        pos_loss = torch.mean(loss_mat, dim=0)

        return pos_loss

def loss_ntxent(features, device, filter=False, topk=0.25, fn_mode='cancel'):
    fn = InfoNCE(temperature=0.2, device=device, filter=filter, topk=topk, fn_mode=fn_mode) # topk: the percentage of each batch to be filtered
    pos_loss = fn(features[0], features[1])

    return pos_loss
