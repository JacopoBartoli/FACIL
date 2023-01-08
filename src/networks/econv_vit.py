from torch import nn
import torch.nn.functional as F
import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

__all__ = ['econv_vit']

# classes
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
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
    def __init__(self, dim, heads=7, dim_head=64, dropout=0.1, ext_attn=False, num_patches=5):
        """
        reduced the default number of heads by 1 per https://arxiv.org/pdf/2106.14881v2.pdf
        """
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)

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

        out = torch.matmul(attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.1, ext_attn=False, num_patches=5):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout, ext_attn=ext_attn, num_patches=num_patches)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class Econv_vit(nn.Module):
    
    def __init__(self, *, num_classes, dim, depth, heads, mlp_dim, projection_dim=0, pool='cls', channels=3, dim_head=64, dropout=0.1, emb_dropout=0.1,
    ext_attn=False):
        """
        3x3 conv, stride 1, 5 conv layers per https://arxiv.org/pdf/2106.14881v2.pdf

        Learnable focuses for avoid forgetting of past embeddings.
        """
        super().__init__()

        n_filter_list = (channels, 24, 48, 96, 196)  # hardcoding for now because that's what the paper used

        self.conv_layers = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(in_channels=n_filter_list[i],
                          out_channels=n_filter_list[i + 1],
                          kernel_size=3,  # hardcoding for now because that's what the paper used
                          stride=2,  # hardcoding for now because that's what the paper used
                          padding=1),  # hardcoding for now because that's what the paper used
            )
                for i in range(len(n_filter_list)-1)
            ])

        self.conv_layers.add_module("conv_1x1", torch.nn.Conv2d(in_channels=n_filter_list[-1], 
                                    out_channels=dim, 
                                    stride=1,  # hardcoding for now because that's what the paper used 
                                    kernel_size=1,  # hardcoding for now because that's what the paper used 
                                    padding=0))  # hardcoding for now because that's what the paper used
        self.conv_layers.add_module("flatten image", 
                                    Rearrange('batch channels height width -> batch (height width) channels'))
        self.pos_embedding = nn.Parameter(torch.randn(1, n_filter_list[-1] + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        #hardcoding
        x = self.conv_layers(torch.zeros(1,3,32,32))
        _, n, _ = x.shape

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, ext_attn, n+1)

        """
        self.projection_head = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, projection_dim)
        )
        """

        self.pool = pool

        self.fc = nn.Linear(in_features=dim, out_features=num_classes, bias=True)
        # and `head_var` with the name of the head, so it can be removed when doing incremental learning experiments
        self.head_var = 'fc'


    def forward(self, img):
        x = self.conv_layers(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        # x = self.projection_head(x)

        x = self.fc(x)

        return x

def econv_vit(pretrained=False, **kwargs):
    if pretrained:
        raise NotImplementedError
    model = Econv_vit(dim=768,
            num_classes=100,
            depth=12,
            heads=8,
            mlp_dim = 2048,
            channels=3,
            **kwargs)
    return model

def econv_vit_ext_attn(pretrained=False, **kwargs):
    if pretrained:
        raise NotImplementedError
    model = Econv_vit(dim=768,
            num_classes=100,
            depth=12,
            heads=8,
            mlp_dim = 2048,
            channels=3,
            ext_attn=True,
            **kwargs)
    return model