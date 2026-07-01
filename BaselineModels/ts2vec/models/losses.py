import torch
from torch import nn
import torch.nn.functional as F

def hierarchical_contrastive_loss(z1, z2, alpha=0.5, temporal_unit=0):
    # print(z1.shape, z2.shape)
    loss = torch.tensor(0., device=z1.device)
    d = 0
    while z1.size(1) > 1:
    # while z1.size(1) > 1499:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        if d >= temporal_unit:
            if 1 - alpha != 0:
                loss += (1 - alpha) * temporal_contrastive_loss(z1, z2)
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)
    if z1.size(1) == 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        d += 1
    return loss / d

def instance_contrastive_loss(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if B == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=0)  # 2B x T x C
    z = z.transpose(0, 1)  # T x 2B x C
    sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # T x 2B x (2B-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    
    i = torch.arange(B, device=z1.device)
    loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
    return loss

def temporal_contrastive_loss(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if T == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=1)  # B x 2T x C
    sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # B x 2T x (2T-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)

    if T > 1500:
        z, sim = z.cpu(), sim.cpu()
        torch.cuda.empty_cache()
    
    t = torch.arange(T, device=z1.device)
    loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
    return loss

# import torch
# from torch import nn
# import torch.nn.functional as F
# import torch.utils.checkpoint as checkpoint

# def chunked_matmul(z1, z2, chunk_size=100):
#     """Memory efficient matrix multiplication"""
#     B, T, C = z1.size()
#     result = z1.new_zeros(B, T, T)
    
#     for i in range(0, T, chunk_size):
#         end_i = min(i + chunk_size, T)
#         chunk_i = z1[:, i:end_i]
        
#         for j in range(0, T, chunk_size):
#             end_j = min(j + chunk_size, T)
#             chunk_j = z2[:, j:end_j]
#             result[:, i:end_i, j:end_j] = torch.matmul(chunk_i, chunk_j.transpose(1, 2))
    
#     return result

# def hierarchical_contrastive_loss(z1, z2, alpha=0.5, temporal_unit=0):
#     loss = torch.tensor(0., device=z1.device)
#     d = 0
    
#     with torch.cuda.amp.autocast():
#         while z1.size(1) > 1:
#             if alpha != 0:
#                 loss += alpha * instance_contrastive_loss(z1, z2)
#             if d >= temporal_unit:
#                 if 1 - alpha != 0:
#                     loss += (1 - alpha) * temporal_contrastive_loss(z1, z2)
#             d += 1
            
#             # Efficient pooling
#             if z1.size(1) > 1:
#                 z1 = F.avg_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
#                 z2 = F.avg_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)
    
#     return loss / d



# # @torch.jit.script
# def instance_contrastive_loss(z1, z2):
#     B, T = z1.size(0), z1.size(1)
#     if B == 1:
#         return z1.new_tensor(0.)
    
#     # More efficient concatenation and computation
#     z = torch.cat([z1, z2], dim=0)  # 2B x T x C
#     z = z.transpose(0, 1)  # T x 2B x C
    
#     sim = checkpoint.checkpoint(
#         lambda x: torch.matmul(x, x.transpose(1, 2)),
#         z
#     )  # T x 2B x 2B
    
#     # Efficient logits computation
#     logits = torch.zeros_like(sim[:, :, :-1])
#     logits.masked_scatter_(
#         torch.tril(torch.ones_like(sim[:, :, :-1]), diagonal=-1).bool(),
#         sim[:, :, :-1][torch.tril(torch.ones_like(sim[:, :, :-1]), diagonal=-1).bool()]
#     )
#     logits.masked_scatter_(
#         torch.triu(torch.ones_like(sim[:, :, 1:]), diagonal=1).bool(),
#         sim[:, :, 1:][torch.triu(torch.ones_like(sim[:, :, 1:]), diagonal=1).bool()]
#     )
    
#     logits = -F.log_softmax(logits, dim=-1)
    
#     i = torch.arange(B, device=z1.device)
#     loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
#     return loss

# def temporal_contrastive_loss(z1, z2):
#     B, T = z1.size(0), z1.size(1)
#     if T == 1:
#         return z1.new_tensor(0.)
    
#     z = torch.cat([z1, z2], dim=1)  # B x 2T x C
    
#     # Use chunked matrix multiplication for memory efficiency
#     sim = chunked_matmul(z, z)  # B x 2T x 2T
    
#     # Compute logits efficiently
#     logits = torch.zeros_like(sim[:, :, :-1])
#     mask_lower = torch.tril(torch.ones_like(sim[:, :, :-1]), diagonal=-1).bool()
#     mask_upper = torch.triu(torch.ones_like(sim[:, :, 1:]), diagonal=1).bool()
    
#     logits.masked_scatter_(mask_lower, sim[:, :, :-1][mask_lower])
#     logits.masked_scatter_(mask_upper, sim[:, :, 1:][mask_upper])
    
#     logits = -F.log_softmax(logits, dim=-1)
    
#     t = torch.arange(T, device=z1.device)
#     loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
#     return loss