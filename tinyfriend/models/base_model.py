import torch
import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    def size(self):
        return sum(p.numel() for p in self.parameters())
