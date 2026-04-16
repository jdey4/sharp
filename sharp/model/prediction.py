import torch
import torch.nn as nn
import torch.nn.functional as F


class PredictionFiLM(nn.Module):
    r"""
        FiLM-modulated prediction (readout) head for hierarchical RNN memory layers.

        Each `PredictionFiLM` block takes a hidden representation from a memory layer
        and modulates it feature-wise using context γ(c), β(c) before decoding.
        This allows the same weights to adapt their behavior smoothly to different
        higher-layer contexts.

        Args:
            input_size (int): Dimensionality of the hidden representation from
                the current layer (input feature dimension).
            hidden_size (int): Size of the intermediate hidden layer.
            output_size (int): Dimensionality of the prediction output.
            context_size (int, optional): Size of contextual vector from a higher layer.
                If 0, the FiLM path is disabled.

        Attributes:
            film (nn.Linear): Projects context → [γ, β] modulation parameters.
            l1 (nn.Linear): Output projection to prediction dimension after modulation.

        Forward:
            forward(h, context=None)
                h: (B, T, input_size)
                context: (B, T, context_size) or None
            Returns:
                Tensor (B, T, output_size)
    """
    def __init__(self, input_size, output_size, num_layers=1, context_size=0):
        super().__init__()
        self.context_size = context_size
        self.input_size = input_size

        self.in_norm = nn.LayerNorm(input_size)
        self.post_film_norm = nn.LayerNorm(input_size)

        self.layers = nn.ModuleList()
        for l in range(num_layers):
            if l == num_layers - 1:
                self.layers.append(nn.Linear(input_size, output_size))
            else:
                self.layers.append(nn.Linear(input_size, input_size))

        if context_size > 0:
            self.film = nn.Linear(context_size, 2 * input_size)
        else:
            self.film = None

    def forward(self, z, context=None):
        z = self.in_norm(z)

        if self.context_size > 0 and context is not None:
            gamma, beta = self.film(context).chunk(2, dim=-1)

            gamma = torch.tanh(gamma)
            beta  = torch.tanh(beta)

            z = (1+gamma) * z + beta
            z = self.post_film_norm(z)

        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                z = layer(torch.nn.functional.gelu(z))
            else:
                z = layer(z)

        return z


class PredictionConcat(nn.Module):
    r"""
    Concatenation-based prediction head for hierarchical RNN memory layers.

    This block projects the upper-layer context to the same dimensionality as
    the memory state, concatenates it with the memory representation, and
    decodes the combined representation via an MLP.

    Args:
        input_size (int): Dimensionality of the hidden representation.
        output_size (int): Dimensionality of the prediction output.
        num_layers (int): Number of MLP layers.
        context_size (int): Dimensionality of upper-layer context (0 disables context).

    Forward:
        z: (B, T, input_size)
        context: (B, T, context_size) or None
    """

    def __init__(self, input_size, output_size, num_layers=1, context_size=0):
        super().__init__()

        self.context_size = context_size
        self.input_size = input_size

        self.in_norm = nn.LayerNorm(input_size)

        if context_size > 0:
            self.context_proj = nn.Linear(context_size, input_size)
            mlp_input_size = 2 * input_size
        else:
            self.context_proj = None
            mlp_input_size = input_size

        self.post_concat_norm = nn.LayerNorm(mlp_input_size)

        self.layers = nn.ModuleList()
        for l in range(num_layers):
            if l == num_layers - 1:
                self.layers.append(nn.Linear(mlp_input_size, output_size))
            else:
                self.layers.append(nn.Linear(mlp_input_size, mlp_input_size))

    def forward(self, z, context=None):
        z = self.in_norm(z)

        if self.context_size > 0 and context is not None:
            c = self.context_proj(context)  # (B, T, input_size)
            z = torch.cat([z, c], dim=-1)   # (B, T, 2 * input_size)
            z = self.post_concat_norm(z)

        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                z = layer(F.gelu(z))
            else:
                z = layer(z)

        return z