import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
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

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., ext_attn=False, num_patches=64):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.ext_attn = ext_attn
        if self.ext_attn:
            self.to_qvk = nn.Linear(dim, inner_dim * 2, bias = False)
            self.ext_k = nn.Parameter(torch.randn(1, heads, num_patches, dim_head))            
            self.ext_bias = nn.Parameter(torch.randn(1, heads, num_patches, num_patches))
        else:   
            self.to_qvk = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        if self.ext_attn:
            qv = self.to_qvk(x).chunk(2, dim = -1)
            q,v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qv)
            dots = (torch.matmul(q, self.ext_k.transpose(-1, -2)) + self.ext_bias) * self.scale

        else:          
            qvk = self.to_qvk(x).chunk(3, dim = -1)
            q, v, k = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qvk)
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., ext_attn=False, num_patches=64):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout, ext_attn=ext_attn, num_patches=num_patches)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.
    ,ext_attn=False):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, ext_attn, num_patches+1)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.fc = nn.Linear(in_features=dim, out_features=num_classes, bias=True)
        # and `head_var` with the name of the head, so it can be removed when doing incremental learning experiments
        self.head_var = 'fc'

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.fc(x)

def vit(pretrained=False, **kwargs):
    if pretrained:
        raise NotImplementedError
    model = ViT(image_size=32,
        patch_size=4, 
        num_classes=100, 
        dim=768, 
        depth=12, 
        heads=12, 
        mlp_dim=2048)              
    return model

def vit_small(pretrained=False, **kwargs):
    if pretrained:
        raise NotImplementedError
    model = ViT(image_size=32,
        patch_size=4, 
        num_classes=100, 
        dim=768, 
        depth=7, 
        heads=8, 
        mlp_dim=2048)              
    return model

def vit_small_ext_attn(pretrained=False, **kwargs):
    if pretrained:
        raise NotImplementedError
    model = ViT(image_size=32,
        patch_size=4, 
        num_classes=100, 
        dim=768, 
        depth=7, 
        heads=8,
        ext_attn=True,
        mlp_dim=2048)              
    return model