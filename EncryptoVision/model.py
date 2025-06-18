from functools import partial
from collections import OrderedDict
from triplet_attention import TripletAttention
import torch
import torch.nn as nn
from torch.nn import functional as F


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    def __init__(self, img_size=16, patch_size=16, in_c=3, embed_dim=768, embed_dim_b=1536, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        self.embed_dim_b = embed_dim_b
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.TripletAttention = TripletAttention(gate_channels=3)
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.norm_b = norm_layer(embed_dim_b) if norm_layer else nn.Identity()

    def forward(self, x, is_img):
        if is_img:
            B, C, H, W = x.shape
            assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
            x = self.TripletAttention(x)

            x = self.proj(x)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
        else:
            x = x.unsqueeze(dim=2).transpose(1, 2)
            x = x[:, :, :self.embed_dim_b]
            x = self.norm_b(x)
        return x


class Attention(nn.Module):
    def __init__(self,
                 dim,   # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Fusion(nn.Module):
    def __init__(self, dim1, dim2, hidden_dim):
        super(Fusion, self).__init__()

        self.linear1 = nn.Linear(dim1, hidden_dim)
        self.linear2 = nn.Linear(dim2, hidden_dim)
        self.linear3 = nn.Linear(dim1, dim2)

        self.att_weight = nn.Linear(hidden_dim, 1)

    def forward(self, f1, f2):
        f1_map = torch.tanh(self.linear1(f1))
        f2_map = torch.tanh(self.linear2(f2))

        score1 = self.att_weight(f1_map).squeeze(-1)
        score2 = self.att_weight(f2_map).squeeze(-1)

        scores = torch.stack([score1, score2], dim=1)
        weights = F.softmax(scores, dim=1)

        f1 = self.linear3(f1)
        # feature = f1_map + f2_map
        feature = weights[:, 0].unsqueeze(1) * f1 + weights[:, 1].unsqueeze(1) * f2
        return feature, weights

class VisionTransformer(nn.Module):
    def __init__(self, img_size=16, patch_size=4, in_c=3, num_classes=12, hidden_dim=256,
                 embed_dim=768, embed_dim_b=1536, depth=12, num_heads=12, num_heads_b=16, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0.1, embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_features_b = self.embed_dim_b = embed_dim_b
        self.num_tokens = 2 if distilled else 1

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        # embedding层,img_size用于适配
        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim, embed_dim_b=embed_dim_b)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token_b = nn.Parameter(torch.zeros(1, 1, embed_dim_b))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None

        self.pos_embed_img = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_embed_binary = nn.Parameter(torch.zeros(1, 1 + self.num_tokens, embed_dim_b))

        self.pos_drop = nn.Dropout(p=drop_ratio)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.blocks_b = nn.Sequential(*[
            Block(dim=embed_dim_b, num_heads=num_heads_b, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])

        self.norm = norm_layer(embed_dim)
        self.norm_b = norm_layer(embed_dim_b)

        # Representation layer
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        self.fusion = Fusion(embed_dim, embed_dim_b, hidden_dim)

        # Classifier head(s)
        self.head = nn.Linear(embed_dim_b, num_classes) if num_classes > 0 else nn.Identity()

        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # Weight init
        nn.init.trunc_normal_(self.pos_embed_img, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_binary, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward_features(self, x_img, x_binary):
        x_image = self.patch_embed(x_img, is_img=True)
        x_bytes = self.patch_embed(x_binary, is_img=False)

        cls_token = self.cls_token.expand(x_image.shape[0], -1, -1)
        cls_token_b = self.cls_token_b.expand(x_bytes.shape[0], -1, -1)

        if self.dist_token is None:
            x_image = torch.cat((cls_token, x_image), dim=1)
            x_bytes = torch.cat((cls_token_b, x_bytes), dim=1)
        else:
            x_image = torch.cat((cls_token, self.dist_token.expand(x_image.shape[0], -1, -1), x_image), dim=1)
            x_bytes = torch.cat((cls_token, self.dist_token.expand(x_bytes.shape[0], -1, -1), x_bytes), dim=1)

        x_image = self.pos_drop(x_image + self.pos_embed_img)
        x_image = self.blocks(x_image)
        x_image = self.norm(x_image)

        x_bytes = self.pos_drop(x_bytes + self.pos_embed_binary)
        x_bytes = self.blocks_b(x_bytes)
        x_bytes = self.norm_b(x_bytes)

        return self.pre_logits(x_image[:, 0]), self.pre_logits(x_bytes[:, 0])

    def forward(self, x_img, x_binary):
        x_img, x_binary = self.forward_features(x_img, x_binary)
        x, weights = self.fusion(x_img, x_binary)

        x = self.head(x)
        return x, weights

def _init_vit_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def vit_base(num_classes: int = 20, has_logits: bool = True):
    model = VisionTransformer(img_size=16,
                              patch_size=4,
                              hidden_dim=512,
                              embed_dim=256,
                              embed_dim_b=768,
                              depth=6,
                              num_heads=16,
                              num_heads_b=16,
                              representation_size=1176 if has_logits else None,
                              num_classes=num_classes)

    return model


