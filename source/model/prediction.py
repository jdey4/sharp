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

        # Base linear layers
        self.layers = nn.ModuleList()
        for l in range(num_layers):
            if l == 0:
                self.layers.append(
                    nn.Linear(input_size, input_size)
                )
            elif l==num_layers-1:
                self.layers.append(
                    nn.Linear(input_size, output_size)
                )
            else:
                self.layers.append(
                    nn.Linear(input_size, input_size)
                )


        # FiLM modulation network (produces gamma, beta)
        if context_size > 0:
            self.film = nn.Sequential(
                nn.Linear(context_size, 2 * input_size)
            )
        else:
            self.film = None

    # ----------------------------------------------------------
    def forward(self, z, context=None):
        """
        z: (B, T, input_size)
        context: (B, T, context_size) or None
        """

        if self.context_size > 0 and context is not None:
            gamma, beta = self.film(context).chunk(2, dim=-1)
            z = gamma * z + beta
        

        # Decode through nonlinear readout
        for layer in self.layers:
            z = layer(nn.functional.gelu(z))
        
        return z
