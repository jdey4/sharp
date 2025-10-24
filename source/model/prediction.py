import torch
import torch.nn as nn
import torch.nn.functional as F


class Prediction(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, context_size=0):
        super().__init__()
        self.context_size = context_size
        self.l1 = nn.Linear(input_size + context_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, output_size)
    def forward(self, h, context=None):
        if self.context_size > 0:
            if context is None:
                context = torch.zeros(h.size(0), h.size(1), self.context_size,
                                      device=h.device, dtype=h.dtype)
            x_in = torch.cat((h, context), dim=2)
        else:
            x_in = h
        x = F.relu(self.l1(x_in))
        return self.l2(x)
