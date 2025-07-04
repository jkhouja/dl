from dataclasses import dataclass


SEQ = 5
B = 4
VOCAB = 10
EMB_DIM = 20
HID_DIM = 30


@dataclass
class LanguageTransformerConfig:
    decoder = True
    data_size = 200
    ignore_idx = -1
    seq_len = 5  # actual context is seq_len
    vocab = None
    batch_size = 4
    epochs = 100
    layers = 3
    nheads = 3
    ffn_expanding = 4
    hdim = 8
    emb_dim = 10
    emb_drop = 0.01
    att_drop = 0.01
    res_drop = 0.01


@dataclass
class VisionTransformerConfig:
    decoder = False
    img_size = None
    img_channels = 3
    patch_size = 4
    add_cls = True
    ignore_idx = -1
    seq_len = SEQ  # actual context is seq_len
    classes = None
    batch_size = B
    epochs = 200
    layers = 4
    nheads = 3
    ffn_expanding = 4
    hdim = HID_DIM
    emb_dim = 8
    emb_drop = 0
    att_drop = 0
    res_drop = 0


TASK_CONFIGS = {""}
