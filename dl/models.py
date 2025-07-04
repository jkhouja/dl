import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    """A module that takes an image batch, breaks each image into patches and encodes (Linear project) them."""

    def __init__(self, config):
        super().__init__()

        img_size = config.img_size
        img_channels = config.img_channels
        patch_size = config.patch_size
        emb_dim = config.emb_dim
        total_patches = (img_size // patch_size) ** 2

        # The patch encoder is simply a convolution with output_channels = emb_dim
        self.conv = nn.Conv2d(
            img_channels, emb_dim, kernel_size=patch_size, stride=patch_size, bias=False
        )

        self.add_cls = config.add_cls
        if self.add_cls:
            # cls token is an embedding unsqueezed to allow concatenation with patch embeddings
            self.cls = nn.parameter.Parameter(torch.zeros(emb_dim)).view(1, 1, emb_dim)

        # Positional embeddings is learnable parameter
        self.pos = nn.parameter.Parameter(
            torch.rand((total_patches + int(self.add_cls), emb_dim))
        )

    def forward(self, x):

        (
            B,
            _,
            _,
            _,
        ) = x.shape  # [B, C, H, W]
        out = self.conv(x)  # [B, Emb, patches, patches]
        out = out.flatten(start_dim=2, end_dim=-1)  # [B, Emb, patches**2]
        out = out.transpose(2, 1)  # [B, patches**2, Emb]

        if self.add_cls:
            # breakpoint()
            cls_token = self.cls.expand(B, -1, -1)
            out = torch.cat((cls_token, out), dim=1)  # [B, 1 + patches**2, Emb]

        out = out + self.pos.unsqueeze(0)  # [B, 1 + patches**2, Emb]

        return out


class SelfAttention(nn.Module):
    """A simple implementation of self attention supporting both decoder and encoder style."""

    def __init__(self, config):
        super().__init__()

        self.nheads = config.nheads
        seq_l = config.seq_len
        hdim = config.hdim
        nheads = config.nheads
        self.decoder = config.decoder

        # Create Q,K,V projection weights. Each expands to multiple heads
        self.Q = nn.parameter.Parameter(torch.rand((hdim, nheads * hdim)))
        self.K = nn.parameter.Parameter(torch.rand((hdim, nheads * hdim)))
        self.V = nn.parameter.Parameter(torch.rand((hdim, nheads * hdim)))

        self.out_proj = nn.parameter.Parameter(torch.rand((nheads * hdim, hdim)))

        # For Decoder Transformer, we mask future tokens for the integrity of auto-regressive language modeling task
        if self.decoder:
            causal_mask = torch.tril(torch.ones(seq_l, seq_l))
            self.register_buffer("causal_mask", causal_mask)

        self.att_dropout = nn.Dropout(p=config.att_drop)
        self.res_dropout = nn.Dropout(p=config.res_drop)

    def forward(self, input):
        """The forward pass on input. Assumes input is of shape [Batch, Seq, hdim]"""

        B, S, H = input.shape

        # [B,S,H] @ [H, H * heads] -> [B,S, H * heads] -> view -> [B, heads, S, H]
        q = (input @ self.Q).view(B, self.nheads, S, H)
        k = (input @ self.K).view(B, self.nheads, S, H)
        v = (input @ self.V).view(B, self.nheads, S, H)

        # Create dot product between q, k -> [B, nheads, Sq, Sk]. each query token is a row
        att = q @ k.transpose(3, 2) * H**-0.5
        assert (B, self.nheads, S, S) == att.shape

        # Add future token mask if decoder
        if self.decoder:
            att = att.masked_fill_(
                self.causal_mask.view(1, 1, S, S).logical_not(), -torch.inf
            )

        # Softmax across columns in each row to normalize
        att = F.softmax(att, dim=-1)

        # Apply ttention dropout
        att = self.att_dropout(att)

        # Apply to values [B, nheads, Sq, Sk] @ [B, nheads, Sv, H] -> [B, nheads, Sq, H]
        z = att @ v

        # Project back to H: [B, S, nheads * hdim] @ [nheads * hdim, hdim] - > [B, S, hdim]
        out = z.view(B, S, -1) @ self.out_proj

        # Residual dropout after projection
        out = self.res_dropout(out)

        return out


class FFN(nn.Module):
    """The FeefForward part of transformer with a fan-out -> fan-in pattern."""

    def __init__(self, config):
        super().__init__()
        hdim = config.hdim
        ffn_expanding = config.ffn_expanding

        # Equivilant to self.W_expand = nn.parameter.Parameter(torch.rand((hdim, ffn_expanding * hdim)))
        self.W_expand = nn.Linear(hdim, hdim * ffn_expanding, bias=False)
        self.W_shrink = nn.Linear(hdim * ffn_expanding, hdim, bias=False)

        self.activation = nn.GELU()
        self.res_dropout = nn.Dropout(p=config.res_drop)

    def forward(self, input):

        out = self.W_expand(input)  # [B, S, hdim * expansion]
        out = self.activation(out)
        out = self.W_shrink(out)
        out = self.res_dropout(out)

        return out


class AttentionBlock(nn.Module):
    """Transformer block, notice we add layer norm at the start instead of the end, which means,
    we need to add one more layer norm before the language head (instead of adding one at the beginning after token embeddings.)
    """

    def __init__(self, config):
        super().__init__()
        hdim = config.hdim

        self.norm1 = nn.LayerNorm(hdim)
        self.norm2 = nn.LayerNorm(hdim)
        self.attention = SelfAttention(config)
        self.ffn = FFN(config)

    def forward(self, input):

        # Layer normalize input but keep copy of the original input
        # Here, we're using pre-LN (prenormalization) variation of attention as recommended

        att_out = self.norm1(input)
        # Apply self attention
        att_out = self.attention(att_out)
        # Pass residual input
        att_out = input + att_out

        # Apply another Layer norm but keep copy of original att_out
        out = self.norm2(att_out)
        # Pass through FFN
        out = self.ffn(out)
        # Then re-add residual connection from att_out
        out = att_out + out

        return out


class Transformer(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.ignore_idx = config.ignore_idx

        # Embedding layer and its dropout
        self.emb = nn.Embedding(config.vocab, config.emb_dim)
        self.emb_dropout = nn.Dropout(p=config.emb_drop)

        # Positional Encoding. This time we'll use learnable embeddings based on index
        self.pos_emb = nn.Embedding(config.seq_len, config.emb_dim)

        # Input projection layer from embedding to hdim
        self.inp_proj = nn.Linear(config.emb_dim, config.hdim, bias=False)

        # Transformer layers
        self.blocks = nn.Sequential(
            *[AttentionBlock(config) for _ in range(config.layers)]
        )

        # Last Norm and language head
        self.last_norm = nn.LayerNorm(config.hdim)
        self.language_head = nn.Linear(config.hdim, config.vocab, bias=False)

        print(
            f"Total trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}"
        )

    def forward(self, input):

        # Check if we have targets
        targets = None
        if isinstance(input, list):
            input, targets = input[0], input[1]
        else:
            # Create a view for targets
            targets = input[:, 1:]
            input = input[:, :-1]

        B, S = input.shape
        device = input.device

        # Get token embeddings and positional embeddings and add them
        emb = self.emb(input)

        # Generate positional encodings for the sequence
        pos = torch.arange(0, S, dtype=torch.long, device=device)
        pos_emb = self.pos_emb(pos)

        res = emb + pos_emb

        # Apply dropout before passing through blocks
        res = self.emb_dropout(res)

        # My addition: Project to hdim
        res = self.inp_proj(res)

        # Pass through all transformer layers then pass through one last layer norm
        res = self.blocks(res)
        res = self.last_norm(res)

        # Language head
        logits = self.language_head(res)

        # Calculate loss if labels were passed
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(B * S, -1),
                targets.reshape(-1),
                ignore_index=self.ignore_idx,
                reduction="mean",
            )

        # Todo: if no labels, we're in inference mode and only need to pass last token through head

        return logits, loss


class ViTransformer(nn.Module):
    """A simple visual Transformer. Breaks image into patches and outputs last layer tokens (or only cls if set)"""

    def __init__(self, config):
        super().__init__()

        #
        self.add_cls = config.add_cls
        self.classes = config.classes

        # The image patch encoder
        self.patch_emb = PatchEmbedding(config)

        # Input projection layer from embedding to hdim
        self.inp_proj = nn.Linear(config.emb_dim, config.hdim, bias=False)

        # dropouts
        self.emb_dropout = nn.Dropout(p=config.emb_drop)
        self.res_drop = nn.Dropout(p=config.res_drop)

        # Blocks
        self.blocks = nn.Sequential(
            *[AttentionBlock(config) for _ in range(config.layers)]
        )

        # Last norm
        self.layer_norm = nn.LayerNorm(config.hdim)

        # Output layer
        self.projection = nn.Linear(config.hdim, config.classes, bias=False)

        print(
            f"Total trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}"
        )

    def forward(self, input):

        # Check if we have targets
        targets = None
        if isinstance(input, list):
            input, targets = input[0], input[1]

        B, C, H, W = input.shape

        out = self.patch_emb(input)
        out = self.emb_dropout(out)

        # Project to internal hidden dim
        out = self.inp_proj(out)

        # Pass through Attention blocks
        out = self.blocks(out)

        # Pass through layer norm
        out = self.layer_norm(out)

        # If add_cls is true, only output cls token
        if self.add_cls:
            out = out[:, 0, :]

        # Output layer
        logits = self.projection(out)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.classes), targets.reshape(-1), reduction="mean"
            )

        return logits, loss
