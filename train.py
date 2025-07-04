from dl import models
import torch
from torch.utils.data import DataLoader
import dl
from dl import data, models
from configs import config
import os


def train():

    conf = config.VisionTransformerConfig()

    # Training loop
    # Load config
    # dataset = data.MYDS("verdict.txt", conf.seq_len, stride=5)
    # conf.vocab = dataset.tokenizer.n_vocab

    dataset = data.DummyImageDataset()
    conf.classes = dataset.classes
    conf.img_size = dataset.img_size

    # print(f"Vocab size: {conf.vocab}")

    train_loader = DataLoader(dataset, batch_size=conf.batch_size)

    # model = models.MyModel(conf)
    # model = models.Transformer(conf)
    model = models.ViTransformer(conf)
    model.train()

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=0.001,
    )
    tsteps = -1
    lsteps = 200
    iters = 0
    # for e in tqdm.tqdm(range(epochs)):
    for i, e in enumerate(range(conf.epochs)):
        for batch in train_loader:
            iters += 1
            model.zero_grad()
            outs, loss = model(batch)
            loss.backward()
            optimizer.step()
            if iters % lsteps == 0:
                print(f"Epoch:{i+1}/step:{iters:,}:  loss: {loss.item():.3f}")
            if tsteps > -1 and iters > tsteps:
                break


if __name__ == "__main__":
    train()
