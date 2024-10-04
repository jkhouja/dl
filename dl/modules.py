import torch
import math
from torch import nn

# Naive implementation of softmax and cross entropy (numerically not stable - can underflow)
def softmax(x):
    # Operating on last dimension
    x = torch.exp(x - torch.max(x, dim=-1, keepdim=True)[0])
    x = x / x.sum(dim=-1, keepdim=True)
    return x

def cross_entropy(inp, targ, reduce=False):
    # operating on last dimension and assuming 2D matrices
    size, nclasses = inp.shape

    # Create 1 hot encoding 
    #targets = torch.empty((size, nclasses))
    #targets[torch.arange(size), targ] = 1

    # easier way? maybe faster?
    #targets.scatter_(-1, targets.view(-1,1), 1)

    # calculate loss as the sum of log probabilities 
    l = -1.0 * torch.log_(inp[torch.arange(size), targ])

    # targets.scatter_(-1, targets.view(-1,1), targets)

    if reduce:
        return l.mean()
        
    return l

# Better implementaion (matches pytorch)
def log_softmax(x):
    # Operating on last dimension
    x = x - torch.max(x, dim=-1, keepdim=True)[0] # shift by max
    xe = torch.exp(x).sum(dim=-1, keepdim=True) # exp
    return x - torch.log(xe)

def cross_entropy_of_log(inp, targ, reduce=False):
    # operating on last dimension and assuming 2D matrices
    size, nclasses = inp.shape
    l = -1.0 * inp[torch.arange(size), targ]

    if reduce:
        return l.mean()
        
    return l

# Or combine both softmax and cross entropy
def cross_entropy_logits(x, targ, reduce=False):
    # operating on last dimension and assuming 2D matrices
    x = x - torch.max(x, dim=-1, keepdim=True)[0] # shift by max
    x = x - torch.log(torch.exp(x).sum(dim=-1, keepdim=True)) # instead of division in exp space
    l = -1.0 * x[torch.arange(x.shape[0]), targ] # loss is taken from the correct class index

    if reduce:
        return l.mean()
        
    return l

# From karpathy
class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
