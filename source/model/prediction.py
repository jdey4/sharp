import torch
import torch.nn as nn
import torch.nn.functional as F


class Prediction(nn.Module):
    """
    Feedforward prediction (readout) head for hierarchical RNN memory layers.

    Each `Prediction` block takes a hidden representation from a memory layer,
    optionally conditioned on contextual input from a higher layer, and produces
    either a next-token prediction (for the bottom layer) or a reconstruction/
    regression target (for higher layers). It implements a simple two-layer MLP
    with ReLU activation.

    Args:
        input_size (int): Dimensionality of the hidden representation from the
            current layer (input feature dimension).
        hidden_size (int): Size of the intermediate hidden layer in the predictor.
        output_size (int): Dimensionality of the prediction output.
            * For layer 0, this is typically the vocabulary size.
            * For higher layers, it matches the previous layer's hidden size.
        context_size (int, optional): Size of additional context concatenated to
            the current layer's hidden state (e.g., from a higher layer). Defaults to 0.

    Attributes:
        context_size (int): Stored value of context_size.
        l1 (nn.Linear): First linear transformation mapping `[h, context] → hidden_size`.
        l2 (nn.Linear): Second linear layer projecting to the output dimension.

    Forward:
        forward(h, context=None)
            Args:
                h (Tensor): Hidden state sequence from the current layer.
                    Shape: (B, T, input_size)
                context (Tensor, optional): Context tensor from a higher layer.
                    Shape: (B, T, context_size) if provided.
            Returns:
                Tensor: Output predictions of shape (B, T, output_size).

    Behavior:
        - If `context_size > 0` but no context is supplied, a zero context tensor
          of the appropriate size is created automatically.
        - The input and context are concatenated along the feature dimension (dim=2).
        - The forward path applies ReLU nonlinearity:
              x = relu(l1([h, context]))
              y = l2(x)

    Example:
        >>> pred = Prediction(input_size=128, hidden_size=256, output_size=10, context_size=64)
        >>> h = torch.randn(1, 5, 128)
        >>> c = torch.randn(1, 5, 64)
        >>> logits = pred(h, c)  # shape (1, 5, 10)

    Notes:
        - In wake-sleep training, lower-layer predictors (e.g., layer 0) use
          cross-entropy loss over vocabulary logits, while higher layers use MSE
          to match hidden-state targets from the layer below.
        - The context pathway allows hierarchical feedback, letting higher-level
          representations influence next-token or next-state predictions.
    """

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
