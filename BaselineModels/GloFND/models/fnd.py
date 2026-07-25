from typing import List, Optional, Union

import torch
import torch.nn as nn


class LambdaThreshold(nn.Module):
    def __init__(
        self,
        glofnd_type: str,
        N: int,
        alpha: float = 0.0,
        lr_lda: float = 0.01,
        lda_beta1: float = 0.9,
        lda_beta2: float = 0.98,
        normalize_grad: bool = False,
        opt_type: str = "adam",
        eps: float = 1e-8,
        start_update: int = 15,
        lda_start: int = 15,
        init_quantile: bool = False,
        clip_grad_mult: Optional[float] = None,
    ) -> None:
        super(LambdaThreshold, self).__init__()

        if glofnd_type == "none":
            glofnd_type = None
        self.glofnd_type = glofnd_type
        assert self.glofnd_type in ["glofnd", "fnd", None], "Invalid glofnd type"
        assert start_update <= lda_start, "Start update must be less than lambda start"
        assert opt_type == "adam", "Only adam optimizer is supported"

        self.learned = self.glofnd_type == "glofnd"
        self.init_quantile = init_quantile
        self.normalize_grad = normalize_grad
        self.eps = eps
        self.opt_type = opt_type

        if clip_grad_mult is not None and clip_grad_mult > 0:
            self.clip_lda_grad = lambda x, alpha: torch.clamp(
                x, min=-alpha * clip_grad_mult, max=alpha * clip_grad_mult
            )
        else:
            self.clip_lda_grad = lambda x, alpha: x

        self.register_buffer("start_update", torch.tensor(start_update, requires_grad=False))
        self.register_buffer("lda_start", torch.tensor(lda_start, requires_grad=False))
        self.register_buffer("lr_lda", torch.tensor(lr_lda, requires_grad=False))
        self.register_buffer("alpha", torch.tensor(alpha, requires_grad=False))
        self.register_buffer("lda_beta1", torch.tensor(lda_beta1, requires_grad=False))
        self.register_buffer("lda_beta2", torch.tensor(lda_beta2, requires_grad=False))
        self.register_buffer("lda", torch.ones(N, requires_grad=False).reshape(-1, 1))
        self.register_buffer("v_grad", torch.zeros(N, requires_grad=False).reshape(-1, 1))
        self.register_buffer("m_grad", torch.zeros(N, requires_grad=False).reshape(-1, 1))

    @torch.no_grad()
    def update(
        self,
        sim: Union[torch.Tensor, List[torch.Tensor]],
        idx: torch.Tensor,
        neg_self_mask: torch.Tensor,
        epoch: int,
    ):
        assert epoch is not None and epoch >= 0, "Epoch must be provided and >= 0"
        if self.start_update > epoch:
            return {}

        if isinstance(sim, torch.Tensor):
            sim = sim.unsqueeze(-1).detach()
        else:
            sim = torch.stack(sim, dim=-1).detach()

        idx = idx.to(self.lda.device, dtype=torch.long)
        batch_size = sim.shape[0]
        num_negative_pairs = neg_self_mask[0].sum().int()
        neg_self_mask = neg_self_mask.to(sim.device).int()

        lr_lda = self.lr_lda
        alpha = self.alpha

        with torch.inference_mode():
            lda_quantile = [
                s.float()[neg_self_mask.bool()]
                .view(batch_size, -1)
                .quantile(1 - alpha, dim=1, keepdim=True)
                .type_as(self.lda)
                for s in sim.unbind(-1)
            ]
            lda_quantile = torch.stack(lda_quantile, dim=-1).mean(-1)

            if not self.learned:
                return {"lda_quantiles": lda_quantile.mean().detach()}

            if epoch == self.start_update and self.init_quantile:
                lda = lda_quantile
            else:
                lda = self.lda[idx]

                g_mask = (sim > lda.unsqueeze(-1)).int() * neg_self_mask.unsqueeze(-1)
                g_mask = g_mask.sum(-1)
                lda_grad = self.alpha - g_mask.sum(1, keepdim=True) / (
                    sim.shape[-1] * num_negative_pairs
                )

                if self.normalize_grad:
                    lda_grad = lda_grad / self.alpha
                lda_grad = self.clip_lda_grad(lda_grad, alpha)

                m_grad = self.m_grad[idx]
                v_grad = self.v_grad[idx]
                m_grad = self.lda_beta1 * m_grad + (1 - self.lda_beta1) * lda_grad
                v_grad = self.lda_beta2 * v_grad + (1 - self.lda_beta2) * (lda_grad ** 2)
                self.m_grad[idx] = m_grad.detach()
                self.v_grad[idx] = v_grad.detach()

                current_epoch = epoch
                m_grad_hat = m_grad / (1 - self.lda_beta1 ** (current_epoch + 1))
                v_grad_hat = v_grad / (1 - self.lda_beta2 ** (current_epoch + 1))
                lda = (lda - lr_lda * m_grad_hat / (v_grad_hat ** 0.5 + self.eps)).clamp(
                    min=-1, max=1
                )

            self.lda[idx] = lda.detach()

        return {"lda_quantiles": lda_quantile.mean().detach()}

    @torch.no_grad()
    def get_mask(
        self,
        sim: torch.Tensor,
        idx: Optional[torch.Tensor],
        neg_self_mask: torch.Tensor,
        epoch: int,
    ):
        neg_self_mask_sum = neg_self_mask.sum(1, keepdim=True)
        if self.glofnd_type is None:
            return neg_self_mask, neg_self_mask_sum, {}

        if self.learned:
            if idx is None:
                lda = self.lda.to(sim.device)
            else:
                idx = idx.to(self.lda.device, dtype=torch.long)
                lda = self.lda[idx].to(sim.device)
        elif self.glofnd_type == "fnd":
            lda = (
                sim.float()[neg_self_mask.bool()]
                .view(sim.shape[0], -1)
                .quantile(1 - self.alpha, dim=1, keepdim=True)
                .type_as(sim)
            )
        else:
            raise ValueError(f"Invalid glofnd type: {self.glofnd_type}")

        neg_self_mask = neg_self_mask.float().detach()
        mask = (sim <= lda).float()
        mask *= neg_self_mask
        log_sum_mask = mask.sum(1, keepdim=True)

        if self.lda_start > epoch:
            return neg_self_mask, neg_self_mask_sum, {
                "num_negatives_perc": log_sum_mask / neg_self_mask_sum.clamp_min(1),
                "num_filtered_perc": neg_self_mask_sum / neg_self_mask_sum.clamp_min(1),
            }

        mask_sum = mask.sum(1, keepdim=True).detach()
        return mask.detach(), mask_sum, {
            "num_negatives_perc": log_sum_mask / neg_self_mask_sum.clamp_min(1),
            "num_filtered_perc": mask_sum / neg_self_mask_sum.clamp_min(1),
        }
