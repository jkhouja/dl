import torch
from torch.utils.data import DataLoader
import dl
from dl import data, models
from dataclasses import dataclass
import os

@dataclass
class MyModelConfig:
    data_size = 200
    seq_len = 10  #actual context is seq_len
    vocab = 30
    batch_size = 8
    epochs = 200
    layers = 4
    nheads = 3
    hdim = 16
    att_drop = .01
    res_drop = .01
    emb_drop = .01

conf = MyModelConfig()



def train():
    
    # Training loop
    # Load config
    dataset = data.MYDS("verdict.txt", conf.seq_len, stride=5)
    conf.vocab = dataset.tokenizer.n_vocab
    print(f"Vocab size: {conf.vocab}")


    train_loader = DataLoader(dataset, batch_size=conf.batch_size)


    model = models.MyModel(conf)
    model.train()

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=0.001, )
    tsteps = -1
    lsteps = 200
    iters = 0
    #for e in tqdm.tqdm(range(epochs)):
    for i, e in enumerate(range(conf.epochs)):
        for b in train_loader:
            iters += 1
            model.zero_grad()
            loss, outs = model(b)
            loss.backward()
            optimizer.step()
            if iters % lsteps == 0:
                print(f"Epoch:{i+1}/step:{iters}:  loss: {loss.item()}")
            if tsteps > -1 and iters > tsteps:
                break

if __name__ == "__main__":
    train()
