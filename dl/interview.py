import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np

seed = 23
torch.random.manual_seed(seed)
np.random.seed(seed)

VOCAB_SIZE = 200
EOS_TOKEN = 199
PAD_TOKEN = -100


class Attention(nn.Module):
    def __init__(self, d_model=64, nhead=2, dropout_prob=0.1):
        super().__init__()
        self.nhead = nhead
        if d_model % nhead != 0:
            raise ValueError(f"{d_model} not divisible by {nhead}.")
        self.hdim = d_model // nhead
        self.scale = self.hdim**-0.5
        self.out_proj = nn.Linear(d_model, d_model)  # sus (hdim, d_model)
        self.dropout = nn.Dropout(dropout_prob)
        self.attn_dropout = nn.Dropout(dropout_prob)

    def forward(self, q, k, v, mask):
        b, s, e = q.shape
        q, k, v = map(
            lambda x: torch.permute(x.view(b, s, self.nhead, self.hdim), (0, 2, 1, 3)),
            (q, k, v),
        )

        # q:[B,heads, S, d_model] @ [B, heads, d_model, S]
        # S: 16
        scores = torch.matmul(q, torch.transpose(k, -1, -2)) * self.scale
        if mask is not None:
            scores.masked_fill_(mask, -torch.inf)  # -torch.inf

        prob = nn.functional.softmax(scores, dim=-1)
        prob = self.attn_dropout(prob)

        # v:[B,heads, S, d_model]
        out = torch.matmul(prob, v)  # [B, heads, S, d_model]
        out = torch.permute(out, (0, 2, 3, 1))  # [B, S, d_model, heads]
        out = torch.reshape(out, (b, s, -1))  # [S, B, heads* d_model]
        return self.dropout(self.out_proj(out))


class MHA(nn.Module):

    def __init__(self, d_model=64):
        super().__init__()
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.attend = Attention()

    def forward(self, x, mask):
        q, k, v = torch.chunk(self.qkv(x), chunks=3, dim=-1)
        return self.attend(q, k, v, mask)


class MLP(nn.Module):

    def __init__(self, d_model=64, mul=2, dropout_prob=0.1):
        super().__init__()
        self.lin1 = nn.Linear(d_model, mul * d_model)
        self.dropout = nn.Dropout(dropout_prob)
        self.act = nn.ReLU()  # GELU
        self.lin2 = nn.Linear(d_model * mul, d_model)

    def forward(self, x):
        return self.lin2(self.dropout(self.act(self.lin1(x))))


class DecoderLayer(nn.Module):

    def __init__(self, d_model=64):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.mlp = MLP()
        self.ln2 = nn.LayerNorm(d_model)
        self.mha = MHA()

    def forward(self, data):
        x, mask = data
        x1 = self.mha(self.ln1(x), mask) + x
        x2 = self.mlp(self.ln2(x1)) + x1
        return (x2, mask)


class Decoder(nn.Module):

    def __init__(self, num_layers=2):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(DecoderLayer())

        self.decoder = nn.Sequential(*layers)

    def forward(self, x, mask):
        return self.decoder((x, mask))


class DecoderTransformer(nn.Module):

    def __init__(self, d_model=64, num_layers=2, dropout_prob=0.1):
        super().__init__()

        self.emb = nn.Embedding(VOCAB_SIZE, d_model)

        max_seq_len = 128
        inv_freqs = torch.exp(
            -torch.log(torch.tensor(10000.0))
            * torch.arange(0, d_model, 2, dtype=torch.float)
            * (d_model**-1)
        )
        arg = torch.arange(max_seq_len, dtype=torch.float).unsqueeze(1) * inv_freqs
        pemb = torch.stack((arg.sin(), arg.cos()), dim=-1).view(max_seq_len, -1)
        self.register_buffer("pemb", pemb)

        self.emb_dropout = nn.Dropout(dropout_prob)

        self.decoder = Decoder(num_layers)

        self.ln = nn.LayerNorm(d_model)
        self.logits_proj = nn.Linear(d_model, VOCAB_SIZE)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x):
        b, s = x.shape
        embeddings = self.emb(x)
        pemb = self.pemb[:s]
        embeddings = self.emb_dropout(embeddings + pemb)

        causal_mask = torch.triu(torch.ones((s, s), dtype=torch.bool), diagonal=1).to(
            self.device
        )
        padding_mask = (x == PAD_TOKEN).to(self.device)

        mask = padding_mask[:, None, None, :] + causal_mask

        out_dec = self.decoder(embeddings, mask)[0]
        out_norm = self.ln(out_dec)
        out = self.logits_proj(out_norm)
        return torch.permute(out, dims=(0, 2, 1))


def get_batch_data(bs=2, seq_len=16, num_batches=500):
    data = torch.randint(0, VOCAB_SIZE - 1, (bs, seq_len + 1))
    data[:, -1] = EOS_TOKEN

    input = data[:, :-1]
    target = data[:, 1:]

    for _ in range(num_batches):
        yield input, target


if __name__ == "__main__":
    device = "cpu"
    lr = 3e-4

    model = DecoderTransformer().to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for i, (data, target) in enumerate(get_batch_data()):
        data = data.to(device)
        logits = model(data)
        loss = loss_fn(logits, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        grads = torch.cat([p.grad.flatten(0) for p in model.parameters()], dim=-1)
        print(f"Loss = {loss:.3f} Grad norm = {torch.norm(grads, p=2):.3f}")

    print("Done.")
