import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    """Convert image to patch embeddings"""
    def __init__(self, image_size: int, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) --> (B, num_patches, embed_dim)"""
        x = self.proj(x) # (B, embed_dim, num_patches_on_1_axis, num_patches_on_1_axis)
        x = x.flatten(2) # (B, embed_dim, num_patches)
        x = x.transpose(1, 2) # (B, num_patches, embed_dim)
        return x


class Attention(nn.Module):
    """Multi-head Self-Attention"""
    def __init__(self, dim: int, num_heads: int = 12, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim//num_heads
        self.scale = self.head_dim ** (-0.5)

        self.qkv = nn.Linear(dim, dim*3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, dim) --> (B, N, dim)"""
        B, N, C = x.shape

        #QKV projection
        qkv = self.qkv(x) # (B, N, dim*3)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4) # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention Weights
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Attention Output
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: int = None, out_features: int = None, drop:float = 0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.drop(x)

        return x

class TransformerBlock(nn.Module):
    """Transformer Block: Attention + MLP"""
    def __init__(self, dim: int, num_heads: int = 12, mlp_ratio: float = 4.0,
                 drop: float = 0.0, attn_drop: float = 0.0):
        super().__init__()

        # attention
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, attn_drop, drop)

        # mlp
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim,
                       hidden_features=mlp_hidden,
                       out_features=dim,
                       drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # residual attention
        x = x + self.attn(self.norm1(x))

        #residual mlp
        x = x + self.mlp(self.norm2(x))
        return x


class ViTBase(nn.Module):
    """ViT-Base from scratch (vit_base_patch16_224 architecture)

    Config:
    - Image size: 224
    - Patch size: 16
    - Embed dim: 768
    - Depth: 12
    - Num heads: 12
    - MLP ratio: 4.0
    """
    def __init__(self, image_size: int = 224, patch_size: int = 16, in_channels: int = 3,
                 embed_dim: int = 768, depth: int = 12, num_heads: int = 12,
                 mlp_ratio: float = 4.0, drop_rate: float = 0.0, attn_drop_rate: float = 0.0):
        super().__init__()

        self.embed_dim = embed_dim

        # patch embedding
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # class token
        self.cls_token = nn.Parameter(torch.zeros(1,1, embed_dim))

        # positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        # transformer Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, drop_rate, attn_drop_rate)
            for _ in range(depth)
        ])

        # layer norm
        self.norm = nn.LayerNorm(embed_dim)

        # initialize weights
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract Feature w/o Classification Head"""
        B = x.shape[0]

        # patch embedding
        x = self.patch_embed(x) # (B, num_patches, embed_dim)

        # add class token
        cls_tokens = self.cls_token.expand(B, -1, -1) # number of cls tokens = batch_size
        x = torch.cat([cls_tokens, x], dim = 1) # (B, num_patches+1, embed_dim)

        # add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # feed through Transformer Blocks
        for block in self.blocks:
            x = block(x)

        # layer norm
        x = self.norm(x)

        return x # still (B, num_patches+1, embed_dim)


class BasicViT(nn.Module):
    """BasicViT for video classification

    Pipeline:
    1. ViT-Base backbone (from scratch)
    2. Process each frame -> extract features
    3. Mean pooling across time (BOTTLENECK!)
    4. Classification head
    """
    def __init__(self, num_classes: int = 51):
        super().__init__()

        # ViT Backbone as Feature Extractor
        self.vit = ViTBase(
            image_size=224,
            patch_size=16,
            in_channels=3,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            drop_rate=0.0,
            attn_drop_rate=0.0
        )

        self.embed_dim = self.vit.embed_dim

        # Classification head
        self.head = nn.Linear(self.embed_dim, num_classes)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        Input: (B, T, C, H, W) where T is number of frames/video (usually 16)
        Output: (B, num_classes)
        """
        B, T, C, H, W = video.shape

        # feed each frame through ViT
        x = video.reshape(B*T, C, H, W)
        x = self.vit.forward_features(x) # (B*T, num_patches+1, 768)
        x = x[:, 0, :] # extract cls token (B*T, 768)

        # Mean Pooling across frames (Caution: LOSE temporal info!)
        x = x.reshape(B, T, self.embed_dim) # (B, T, 768)
        x = x.mean(dim=1)

        # classification
        logits = self.head(x) # (B, num_classes)
        return logits


class ViTGRU(nn.Module):
    def __init__(self, num_classes: int = 51, hidden_dim: int = 512, num_layers: int = 1):
        super().__init__()

        # ViT backbone
        self.vit = ViTBase(
            image_size=224,
            patch_size=16,
            in_channels=3,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            drop_rate=0.0,
            attn_drop_rate=0.0
        )

        self.embed_dim = self.vit.embed_dim

        # RNN (GRU)
        self.rnn = nn.GRU(
            input_size=self.embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        # Classification head
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = video.shape

        # Extract frame features
        x = video.reshape(B*T, C, H, W)
        x = self.vit.forward_features(x)
        x = x[:, 0, :]  # CLS token -> (B*T, 768)

        # Reshape to sequence
        x = x.reshape(B, T, self.embed_dim)  # (B, T, 768)

        # RNN
        out, h_n = self.rnn(x) # The output contains the hidden states from the last layer of the network for every time step in the sequence

        # Option 1: use last hidden state
        x = h_n[-1]  # (B, hidden_dim)

        # Option 2 (alternative): use last timestep output
        # x = out[:, -1, :]

        logits = self.head(x)
        return logits
    

class ViTLSTM(nn.Module):
    def __init__(self, num_classes: int = 51, hidden_dim: int = 512, num_layers: int = 1):
        super().__init__()

        # ViT backbone
        self.vit = ViTBase(
            image_size=224,
            patch_size=16,
            in_channels=3,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            drop_rate=0.0,
            attn_drop_rate=0.0
        )

        self.embed_dim = self.vit.embed_dim

        # RNN (GRU)
        self.rnn = nn.LSTM(
            input_size=self.embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        # Classification head
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = video.shape

        # Extract frame features
        x = video.reshape(B*T, C, H, W)
        x = self.vit.forward_features(x)
        x = x[:, 0, :]  # CLS token -> (B*T, 768)

        # Reshape to sequence
        x = x.reshape(B, T, self.embed_dim)  # (B, T, 768)

        # RNN
        out, (h_n, c_n) = self.rnn(x)

        # Option 1: use last hidden state
        x = h_n[-1]  # (B, hidden_dim)

        # Option 2 (alternative): use last timestep output
        # x = out[:, -1, :]

        logits = self.head(x)
        return logits