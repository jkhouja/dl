import torch
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
import os
import os
import urllib.request
import tiktoken


class DummyDataset(Dataset):
    def __init__(self, size, seq_len, vocab):
        assert vocab < 2**16
        super().__init__()
        self.vocab = vocab
        self.size = size
        self.data = torch.randint(
            low=0, high=vocab, size=(size * seq_len,), dtype=torch.int
        ).view((size, seq_len))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # return self.data[idx,:-1], self.data[idx,1:]\
        # Decided to return the whole sequence
        return self.data[idx, :]


class DummyImageDataset(Dataset):
    def __init__(self, size=1000, img_size=64, channels=3, classes=14):
        super().__init__()
        self.classes = classes
        self.img_size = img_size
        self.size = size
        self.data = torch.randn(size=(size * channels * img_size * img_size,))
        self.data = self.data.view(
            (
                size,
                channels,
                img_size,
                img_size,
            )
        )
        self.targets = torch.randint(low=0, high=classes - 1, size=(size,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # return self.data[idx,:-1], self.data[idx,1:]\
        # Decided to return the whole sequence
        return (self.data[idx, :], self.targets[idx])


class MYDS(Dataset):

    def __init__(self, path, max_length, stride):
        max_length += 1
        if not os.path.exists(path):
            url = (
                "https://raw.githubusercontent.com/rasbt/"
                "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
                "the-verdict.txt"
            )
            urllib.request.urlretrieve(url, path)

        with open(path, "r", encoding="utf-8") as f:
            self.raw_text = f.read()

        self.tokenizer = tiktoken.get_encoding("gpt2")

        token_ids = self.tokenizer.encode(
            self.raw_text, allowed_special={"<|endoftext|>"}
        )

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        self.data = []
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            # target_chunk = token_ids[i + 1: i + max_length + 1]
            self.data.append(torch.tensor(input_chunk))

    def __getitem__(self, idx):
        # return self.data[idx,:-1], self.data[idx,1:]\
        # Decided to return the whole sequence
        return self.data[idx]

    def __len__(self):
        return len(self.data)
