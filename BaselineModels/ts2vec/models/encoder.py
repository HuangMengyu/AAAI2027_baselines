import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from .dilated_conv import DilatedConvEncoder
from base_models import resnet1d18

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

def generate_continuous_mask(B, T, n=5, l=0.1):
    res = torch.full((B, T), True, dtype=torch.bool)
    if isinstance(n, float):
        n = int(n * T)
    n = max(min(n, T // 2), 1)
    
    if isinstance(l, float):
        l = int(l * T)
    l = max(l, 1)
    
    for i in range(B):
        for _ in range(n):
            t = np.random.randint(T-l+1)
            res[i, t:t+l] = False
    return res

def generate_binomial_mask(B, T, p=0.5):
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(torch.bool)


class cnn_extractor(nn.Module):
    def __init__(self, dim, input_plane):
        super(cnn_extractor, self).__init__()
        self.cnn = resnet1d18(input_channels=dim, inplanes=input_plane)

    def forward(self, x):
        x = self.cnn(x)
        return x

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# class Attention(nn.Module):
#     def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
#         super().__init__()
#         inner_dim = dim_head * heads
#         project_out = not (heads == 1 and dim_head == dim)

#         self.heads = heads
#         self.scale = dim_head ** -0.5

#         self.attend = nn.Softmax(dim=-1)
#         self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

#         self.to_out = nn.Sequential(
#             nn.Linear(inner_dim, dim),
#             nn.Dropout(dropout)
#         ) if project_out else nn.Identity()

#     def forward(self, x):
#         qkv = self.to_qkv(x).chunk(3, dim=-1)
#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

#         dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

#         attn = self.attend(dots)

#         out = torch.matmul(attn, v)
#         out = rearrange(out, 'b h n d -> b n (h d)')
#         return self.to_out(out)


# class Attention(nn.Module):
    # def __init__(self, dim, heads=8, dim_head=64, dropout=0., chunk_size = 1):
    #     super().__init__()
    #     inner_dim = dim_head * heads
    #     project_out = not (heads == 1 and dim_head == dim)

    #     self.heads = heads
    #     self.scale = dim_head ** -0.5

    #     self.attend = nn.Softmax(dim=-1)
    #     self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

    #     self.to_out = nn.Sequential(
    #         nn.Linear(inner_dim, dim),
    #         nn.Dropout(dropout)
    #     ) if project_out else nn.Identity()

    #     self.chunk_size = chunk_size

    # def forward(self, x):
    #     # x: (b, n, dim)
    #     b, n, _ = x.shape
    #     qkv = self.to_qkv(x).chunk(3, dim=-1)
    #     q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

    #     # Use PyTorch scaled_dot_product_attention (flash-backed when available).
    #     # Reshape to (b*h, n, d) to call the function and avoid storing the full (b,h,n,n) in Python memory.
    #     q = rearrange(q, 'b h n d -> (b h) n d')
    #     k = rearrange(k, 'b h n d -> (b h) n d')
    #     v = rearrange(v, 'b h n d -> (b h) n d')

    #     # # scaled_dot_product_attention is memory- and speed-optimized on modern PyTorch + CUDA
    #     # attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
    #     # attn_out = rearrange(attn_out, '(b h) n d -> b h n d', b=b, h=self.heads)

    #     # out = rearrange(attn_out, 'b h n d -> b n (h d)')
    #     out_chunks = []
    #     for i in range(0, n, self.chunk_size):
    #         q_chunk = q[:, i:i + self.chunk_size, :]                # (bh, qcs, d)
    #         # compute scores only for this q_chunk: (bh, qcs, n)
    #         scores = torch.matmul(q_chunk, k.transpose(-1, -2)) * self.scale
    #         attn = torch.softmax(scores, dim=-1)
    #         out_chunk = torch.matmul(attn, v)                       # (bh, qcs, d)
    #         out_chunks.append(out_chunk)

    #     out = torch.cat(out_chunks, dim=1)  
        
    #     del out_chunks, q, k, v
    #     torch.cuda.empty_cache()
    #                              # (bh, n, d)
    #     out = rearrange(out, '(b h) n d -> b h n d', b=b, h=self.heads)
    #     out = rearrange(out, 'b h n d -> b n (h d)')
    #     return self.to_out(out)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

# class Transformer(nn.Module):
    # def __init__(self, dim=128, depth=6, heads=8, dim_head=64, mlp_dim=128, dropout=0.2):
    #     super().__init__()
    #     self.layers = nn.ModuleList([])
    #     for _ in range(depth):
    #         self.layers.append(nn.ModuleList([
    #             PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
    #             PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
    #         ]))

    # def forward(self, x):
    #     for attn, ff in self.layers:
    #         x = attn(x) + x
    #         x = ff(x) + x
    #     return x

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x

class Seq_Transformer(nn.Module):
    def __init__(self, *, patch_size, dim, depth, heads, mlp_dim, channels=1, dropout=0.1):
        super().__init__()
        patch_dim = channels * patch_size
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.c_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)
        self.to_c_token = nn.Identity()


    def forward(self, forward_seq):
        x = self.patch_to_embedding(forward_seq)
        b, n, _ = x.shape
        c_tokens = repeat(self.c_token, '() n d -> b n d', b=b)
        x = torch.cat((c_tokens, x), dim=1)
        x = self.transformer(x)
        c_t = self.to_c_token(x[:, 0])
        return c_t

class base_Model(nn.Module): # encoder
    def __init__(self, input_channels=64, final_out_channels = 320, kernel_size = 25, stride = 3, features_len = 127):
        super(base_Model, self).__init__()

        self.conv_block1 = nn.Sequential(
            # nn.Conv1d(input_channels, 32, kernel_size=kernel_size,
            #           stride=stride, bias=False, padding=(kernel_size//2)),
            nn.Conv1d(input_channels, 32, kernel_size=kernel_size,
                      stride=1, bias=False, padding=(kernel_size//2)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.35)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(final_out_channels),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        model_output_dim = features_len
        # self.logits = nn.Linear(model_output_dim * configs.final_out_channels, configs.num_classes)

    def forward(self, x_in):
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        # x_flat = x.reshape(x.shape[0], -1)
        # logits = self.logits(x_flat)
        # return logits, x
        return x

class TSEncoder(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_dims=64, depth=10, mask_mode='binomial'):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.mask_mode = mask_mode
        self.input_fc = nn.Linear(input_dims, hidden_dims)
        # self.feature_extractor = DilatedConvEncoder(
        #     hidden_dims,
        #     [hidden_dims] * depth + [output_dims],
        #     kernel_size=3
        # )
        # self.feature_extractor1 = cnn_extractor(dim = hidden_dims, input_plane = 16)
        # self.feature_extractor1 = cnn_extractor(dim = input_dims, input_plane = 16)
        # self.feature_extractor2 = Transformer()
        self.feature_extractor = base_Model()
        self.repr_dropout = nn.Dropout(p=0.1)
        
    def forward(self, x, mask=None):  # x: B x T x input_dims
        batch_size = x.size(0)
        nan_mask = ~x.isnan().any(axis=-1)
        x[~nan_mask] = 0
        x = self.input_fc(x)  # B x T x Ch
        
        # generate & apply mask
        if mask is None:
            if self.training:
                mask = self.mask_mode
            else:
                mask = 'all_true'
        
        if mask == 'binomial':
            mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'continuous':
            mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'all_true':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        elif mask == 'all_false':
            mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
        elif mask == 'mask_last':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            mask[:, -1] = False
        
        mask &= nan_mask
        x[~mask] = 0
        
        # conv encoder
        x = x.transpose(1, 2)  # B x Ch x T
        # # x = self.repr_dropout(self.feature_extractor(x))  # B x Co x T
        # print("Before CNN:", x.shape)
        # to_patch = nn.Sequential(Rearrange('b c (n p1) -> b n c p1', p1=1),
        #                     Rearrange('b n c p1 -> (b n) c p1'))
        # x = to_patch(x)
        # # print("After patching:", x.shape)
        # x = self.feature_extractor1(x)
        # # print("After CNN:", x.shape)
        # patch2seq = nn.Sequential(nn.AdaptiveAvgPool1d(1),
        #                           Rearrange('(b n) c 1 -> b n c', b=batch_size))
        # x = patch2seq(x)
        # # print("After seq restoring:", x.shape)
        x = self.repr_dropout(self.feature_extractor(x))
        # print("After CNN", x.shape)
        # x = self.feature_extractor2(x)
        # # print("After Transformer:", x.shape)
        out = x.transpose(1, 2)  # B x T x Co
        # print("After transpose:", x.shape)
        # out = x

        return out
        