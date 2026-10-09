import torch
from torch import nn, einsum
from transformers import AutoModel
from einops import rearrange
import torch.nn.functional as F

class Dino(nn.Module):
    def __init__(self, pretrained_model_name="facebook/dinov3-convnext-base-pretrain-lvd1689m"):
        super(Dino, self).__init__()
        print(f'Using DinoV3 {pretrained_model_name}')

        # gated model: accept the licence on huggingface.co and log in (huggingface-cli login / HF_TOKEN)
        self.model = AutoModel.from_pretrained(
            pretrained_model_name,
            output_hidden_states=True,
        )
        self.model.eval()
        self.model.requires_grad_(False)

    def forward(self, x):
        if x.shape[1] == 1:
            # grayscale
            x = torch.cat([x, x, x], dim=1)
        elif x.shape[1] == 3:
            pass
        else:
            raise NotImplementedError
        out = self.model(x)
        # print(out.last_hidden_state.shape)
        # print(out.pooler_output.shape)
        return out.hidden_states


class DinoBlock(nn.Module):
    def __init__(self, in_chn, out_chn):
        super().__init__()
        self.ps1 = nn.PixelShuffle(2)
        self.conv_hidden1 = nn.Conv2d(in_chn // 4, in_chn // 4, kernel_size=3, stride=1, padding=1, bias=True)

        self.ps2 = nn.PixelShuffle(2)
        self.conv_hidden2 = nn.Conv2d(in_chn // 16, in_chn // 16, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv_hidden3 = nn.Conv2d(in_chn // 16, out_chn, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_combine = nn.Conv2d(out_chn * 2, out_chn, kernel_size=3, stride=1, padding=1, bias=True)

    def norm_hidden(self, hidden):
        return (hidden - hidden.mean()) / hidden.std()

    def forward(self, x, hidden):
        h = self.norm_hidden(hidden)
        h = self.conv_hidden1(self.ps1(h))
        h = self.conv_hidden2(self.ps2(h))

        h = self.conv_hidden3(h)
        x = torch.cat((x, h), dim=1)
        x = self.conv_combine(x)
        return x


# adapted from CrossAttention in latent-diffusion (https://github.com/CompVis/latent-diffusion, MIT licence)
class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None):
        h = self.heads

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))


        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class DinoCrossAttentionBlockOld(nn.Module):
    def __init__(self, model_dim, dino_dim, heads=8, dim_head=64):
        super().__init__()

        self.norm_x = nn.LayerNorm(model_dim)
        self.norm_dino = nn.LayerNorm(dino_dim)

        self.dino_proj = nn.Linear(dino_dim, model_dim)

        self.cross_attn = CrossAttention(
            query_dim=model_dim,
            context_dim=dino_dim,
            heads=heads,
            dim_head=dim_head
        )

        # learnable gate (starts closed)
        self.gate = nn.Parameter(torch.zeros(model_dim))

        self.film = FiLM(dino_dim, model_dim)

    def forward(self, x, dino):

        context = self.dino_proj(self.norm_dino(dino))

        attn_out = self.cross_attn(
            self.norm_x(x),
            context=context
        )

        x = x + torch.tanh(self.gate) * attn_out
        # FiLM modulation
        x = self.film(x, dino)

        return x


class FiLM(nn.Module):
    def __init__(self, cond_dim, feature_dim):
        super().__init__()

        self.to_scale = nn.Linear(cond_dim, feature_dim)
        self.to_shift = nn.Linear(cond_dim, feature_dim)

    def forward(self, x, cond):

        # pool conditioning tokens
        cond = cond.mean(dim=1)

        gamma = self.to_scale(cond).unsqueeze(1)
        beta = self.to_shift(cond).unsqueeze(1)

        return gamma * x + beta


class TokenPool(nn.Module):

    def __init__(self, dim, num_tokens):
        super().__init__()

        self.pool = nn.Parameter(torch.randn(num_tokens, dim))

    def forward(self, tokens):

        attn = torch.softmax(
            tokens @ self.pool.T,
            dim=1
        )

        pooled = attn.transpose(1,2) @ tokens

        return pooled


class DinoCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        model_dim,
        dino_dim,
        num_pooled_tokens=64,
        heads=8,
        dim_head=64
    ):
        super().__init__()

        self.gn = nn.GroupNorm(8, model_dim)
        self.gn_dino = nn.GroupNorm(8, dino_dim)

        self.lowfreq_proj = nn.Conv2d(model_dim, 1, 3, padding=1)

        self.norm_x = nn.LayerNorm(model_dim)
        self.norm_dino = nn.LayerNorm(dino_dim)

        # project DINO features into model space
        self.dino_proj = nn.Linear(dino_dim, model_dim)

        # token pooling
        self.token_pool = TokenPool(model_dim, num_pooled_tokens)

        self.cross_attn = CrossAttention(
            query_dim=model_dim,
            context_dim=model_dim,
            heads=heads,
            dim_head=dim_head
        )

        self.film = FiLM(model_dim, model_dim)

        # gated cross-attention
        self.gate = nn.Parameter(torch.zeros(model_dim))

    def forward(self, x, dino):
        B, C, H, W = x.shape
        B2, C_d, H_d, W_d = dino.shape

        assert B == B2

        # -----------------------
        # spatial normalization
        # -----------------------
        x = self.gn(x)
        dino = self.gn_dino(dino)

        # low-frequency attention bias
        lf = F.avg_pool2d(x, kernel_size=5, stride=1, padding=2)
        lf = torch.sigmoid(self.lowfreq_proj(lf))  # (B,1,H,W)

        # -----------------------
        # flatten model features
        # -----------------------
        x_tokens = rearrange(x, "b c h w -> b (h w) c")

        # -----------------------
        # flatten dino features
        # -----------------------
        dino_tokens = rearrange(dino, "b c h w -> b (h w) c")

        # -----------------------
        # normalize
        # -----------------------
        x_tokens = self.norm_x(x_tokens)
        dino_tokens = self.norm_dino(dino_tokens)

        # -----------------------
        # project DINO features
        # -----------------------
        dino_tokens = self.dino_proj(dino_tokens)

        # -----------------------
        # token pooling
        # -----------------------
        dino_tokens = self.token_pool(dino_tokens)

        lf_tokens = rearrange(lf, "b 1 h w -> b (h w) 1")
        # -----------------------
        # cross attention
        # -----------------------
        attn = self.cross_attn(x_tokens, context=dino_tokens)

        attn = attn * (1 + lf_tokens)

        gate = torch.tanh(self.gate)[None, None, :]

        x_tokens = x_tokens + gate * attn

        # -----------------------
        # FiLM modulation
        # -----------------------
        x_tokens = self.film(x_tokens, dino_tokens)

        # -----------------------
        # reshape back to feature map
        # -----------------------
        x = rearrange(x_tokens, "b (h w) c -> b c h w", h=H, w=W)

        return x