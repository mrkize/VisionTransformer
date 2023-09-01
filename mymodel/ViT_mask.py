import torch
from torch import nn, einsum
from einops import rearrange, repeat


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """

    def __init__(self, image_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        """
        Map input tensor to patch.
        Args:
            image_size: input image size
            patch_size: patch size
            in_c: number of input channels
            embed_dim: embedding dimension. dimension = patch_size * patch_size * in_c
            norm_layer: The function of normalization
        """
        super().__init__()
        image_size = (image_size, image_size)
        patch_size = (patch_size, patch_size)
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        # The input tensor is divided into patches using 16x16 convolution
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.image_size[0] and W == self.image_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x




#layernom+attention/MLP
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


# this is MLP
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


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # (b, n(65), dim*3) ---> 3 * (b, n, dim)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)  # q, k, v   (b, h, n, dim_head(64))

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class VIT(nn.Module):
    def __init__(self, image_size, patch_size, embed_dim, num_classes, depth, heads, mlp_dim, pool='cls', channels=3, dim_head=8, dropout=0., emb_dropout=0.,PE=True):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert  image_height % patch_height ==0 and image_width % patch_width == 0

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        assert pool in {'cls', 'mean'}

        self.to_patch_embedding = PatchEmbed(image_size=image_size, patch_size=patch_size, in_c=channels, embed_dim=embed_dim)
        self.unk_embed_index = num_patches + 1
        self.PE = True
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches+2, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))					# nn.Parameter()定义可学习参数
        self.dropout = nn.Dropout(emb_dropout)
        self.train_method = 'zero'
        self.transformer = Transformer(embed_dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x, unk_mask = None):
        x = self.to_patch_embedding(x)       # b c (h p1) (w p2) -> b (h w) (p1 p2 c) -> b (h w) dim
        b, n, _ = x.shape           # b表示batchSize, n表示每个块的空间分辨率, _表示一个块内有多少个值

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)  # self.cls_token: (1, 1, dim) -> cls_tokens: (batchSize, 1, dim)
        x = torch.cat((cls_tokens, x), dim=1)               # 将cls_token拼接到patch token中去       (b, 65, dim)

        pos_embedding = self.pos_embedding[:, :-1, :]
        if unk_mask is not None:
            if self.train_method == 'mask':
                seq_ord = torch.arange(x.size(1)).unsqueeze(0).to(x.device)
                seq_ord = seq_ord * (1 - unk_mask) + unk_mask * self.unk_embed_index
                pos_embedding = self.pos_embed[:,seq_ord.squeeze(0),:]

            elif self.train_method == 'zero':
                seq_ord = torch.arange(x.size(1)).unsqueeze(0).to(x.device)
                seq_ord = seq_ord * (1 - unk_mask) + unk_mask * self.unk_embed_index
                pos = torch.cat([pos_embedding, torch.zeros(1, 1, self.embed_dim).to(x.device)], dim=1)
                pos_embedding = pos[:,seq_ord.squeeze(0),:]
        if self.PE:
            x += pos_embedding
        x = self.dropout(x)

        x = self.transformer(x)                                                 # (b, 65, dim)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]                   # (b, dim)

        x = self.to_latent(x)                                                   # Identity (b, dim)

        return self.mlp_head(x)                                                 #  (b, num_classes)



class parameter(nn.Module):
    def __init__(self):
        super().__init__()
        self.para = nn.Parameter(torch.randn(1, 3, 32, 32))

    def forward(self, x):
        return x + self.para


def creat_VIT(config):
    model_vit = VIT(
        image_size=config.patch.image_size,
        patch_size=config.patch.patch_size,
        num_classes=config.patch.num_classes,
        embed_dim=config.patch.embed_dim,
        depth=config.patch.depth,
        heads=config.patch.heads,
        mlp_dim=config.patch.mlp_dim,
        dropout=config.patch.dropout,
        emb_dropout=config.patch.emb_dropout,
        channels=config.patch.channels
    )
    return model_vit

def load_VIT(model_path, config):
    model_vit = VIT(
        image_size=config.patch.image_size,
        patch_size=config.patch.patch_size,
        num_classes=config.patch.num_classes,
        embed_dim=config.patch.embed_dim,
        depth=config.patch.depth,
        heads=config.patch.heads,
        mlp_dim=config.patch.mlp_dim,
        dropout=config.patch.dropout,
        emb_dropout=config.patch.emb_dropout,
        channels=config.patch.channels
    )
    model_vit.load_state_dict(torch.load(model_path))
    return model_vit