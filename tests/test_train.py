import torch
from dl.models import SelfAttention, Transformer, PatchEmbedding
from dataclasses import dataclass


SEQ = 5
B = 4
VOCAB = 10
EMB_DIM = 20
HID_DIM = 30

DATA_TEXT_RAND = torch.randint(0, VOCAB, (B, SEQ))
DATA_IMG_RAND = torch.rand(B, 3, 64, 64)
DATA_ONES = torch.ones(B, SEQ, dtype=torch.long)


T2D = torch.rand((3, 4), dtype=torch.float)
EMB = torch.rand(VOCAB, EMB_DIM)
ATT_INPUT = torch.rand(VOCAB, HID_DIM)


@dataclass
class attention_config:
    decoder = False
    ignore_idx = -1
    seq_len = SEQ  # actual context is seq_len
    vocab = VOCAB
    batch_size = B
    layers = 4
    nheads = 3
    ffn_expanding = 4
    hdim = HID_DIM
    emb_dim = EMB_DIM
    embdim = EMB_DIM
    emb_drop = 0
    att_drop = 0
    res_drop = 0


@dataclass
class patching_config:
    img_size = 64
    img_channels = 3
    patch_size = 4
    patch_emb_size = 20
    add_cls = False
    ignore_idx = -1
    seq_len = SEQ  # actual context is seq_len
    vocab = VOCAB
    batch_size = B
    layers = 4
    nheads = 3
    ffn_expanding = 4
    hdim = HID_DIM
    emb_dim = EMB_DIM
    emb_drop = 0
    att_drop = 0
    res_drop = 0


def test_attention_runs():
    """Passes if self attention works."""
    input = ATT_INPUT[DATA_TEXT_RAND.view(-1), :].view(B, SEQ, HID_DIM)

    att = SelfAttention(attention_config())
    out = att(input)


def test_transformer_runs():
    """Passes if Transformer works on sample data"""

    transformer = Transformer(attention_config())
    input = DATA_TEXT_RAND

    out = transformer(input)


def test_attention_same_word():
    """Passes if self attention on the same word gives the same value."""
    input = ATT_INPUT[DATA_ONES.view(-1), :].view(B, SEQ, HID_DIM)

    att = SelfAttention(attention_config())
    out = att(input)

    # Check all values in columns are the same by comparing all row's values to the 1st row
    out = out.mean(-1)
    assert torch.allclose(out, out[0]), out


def test_patching_run():

    input = DATA_IMG_RAND
    patch_config = patching_config()
    img_encoder = PatchEmbedding(patching_config)
    out = img_encoder(input)


def test_attention_same_word():
    """Passes if self attention on the same word gives the same value."""
    input = ATT_INPUT[DATA_ONES.view(-1), :].view(B, SEQ, HID_DIM)

    att = SelfAttention(attention_config())
    out = att(input)

    # Check all values in columns are the same by comparing all row's values to the 1st row
    out = out.mean(-1)
    assert torch.allclose(out, out[0]), out
