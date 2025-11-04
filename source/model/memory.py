import torch
import torch.nn as nn
import torch.nn.functional as F


class Memory(nn.Module):
    """
    Autoencoder-style recurrent memory block.

    This module forms one "memory layer" in a hierarchical sequence model.
    Each block acts as an autoencoder that encodes its input sequence into a
    recurrent hidden representation and then reconstructs the input through a
    decoder RNN. The final hidden state serves as the compressed memory
    representation that can be passed to higher layers or used for replay.

    For the lowest layer (layer == 0), the input tokens are first embedded
    through an `nn.Embedding` before being fed to the encoder RNN. For higher
    layers, the input is expected to be a continuous vector (e.g., the hidden
    state from a lower layer).

    Args:
        input_size (int): Dimensionality of the input sequence at this layer.
            For layer 0, this is the vocabulary size; for higher layers, it is
            the hidden size of the previous layer.
        hidden_size (int): Dimensionality of the hidden (memory) representation.
        embedding_dim (int, optional): Size of the token embedding at layer 0.
            Must be provided when `layer == 0`.
        layer (int): Integer index of the layer in the hierarchy (0 = bottom).

    Attributes:
        embedding (nn.Embedding): Token embedding used only when layer == 0.
        encoder (nn.RNN): RNN that encodes the input sequence into a hidden state.
        decoder (nn.RNN): RNN that reconstructs the sequence from the hidden state.
        out (nn.Linear): Linear projection from decoder hidden state to output space.

    Initialization:
        - Encoder/decoder recurrent weights (`weight_hh`) are orthogonally initialized.
        - Input weights (`weight_ih`) use Xavier uniform initialization.
        - Biases are zero-initialized.

    Forward:
        forward(x, h0=None)
            Args:
                x (Tensor): Input sequence.
                    * shape: (B, T) if integer tokens (layer 0)
                    * shape: (B, T, input_size) if continuous vectors (higher layers)
                h0 (Tensor, optional): Initial hidden state (1, B, hidden_size).
            Returns:
                logits (Tensor): Reconstructed sequence of shape (B, T, input_size).
                h (Tensor): Final encoder hidden state (1, B, hidden_size).

    Step-wise encoding utilities:
        encode_step_from_token(token_id, h_prev)
            Encodes a single token step (only valid for layer 0).
            Args:
                token_id (Tensor): Integer token id of shape (1,).
                h_prev (Tensor): Previous hidden state.
            Returns:
                h_next (Tensor): Updated hidden state (1, 1, hidden_size).

        encode_step_from_vec(x_vec, h_prev)
            Encodes a single step given a continuous input vector.
            Args:
                x_vec (Tensor): Input vector of shape (1, 1, input_size).
                h_prev (Tensor): Previous hidden state.
            Returns:
                h_next (Tensor): Updated hidden state (1, 1, hidden_size).

    Usage example:
        >>> mem = Memory(input_size=100, hidden_size=64, embedding_dim=32, layer=0)
        >>> x = torch.randint(0, 100, (1, 5))
        >>> logits, h = mem(x)
        >>> h_next = mem.encode_step_from_token(torch.tensor([3]), h)

    Notes:
        - The decoder runs autoregressively for T steps, each time using the
          previous output as its next input (`dec_in = logits.detach()`).
        - Detaching the decoder input prevents gradient explosion and simulates
          self-replay during reconstruction.
    """

    def __init__(self, input_size, hidden_size, embedding_dim=None, layer=0):
        super().__init__()
        self.layer = layer
        self.input_size = input_size
        self.hidden_size = hidden_size

        if layer == 0:
            assert embedding_dim is not None, "embedding_dim required for layer 0"
            self.embedding = nn.Embedding(input_size, embedding_dim)
            self.encoder = nn.RNN(embedding_dim, hidden_size, batch_first=True)
        else:
            self.encoder = nn.RNN(input_size, hidden_size, batch_first=True)

        self.decoder = nn.RNN(input_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, input_size)

        # init
        for name, p in self.encoder.named_parameters():
            if "weight_hh" in name: nn.init.orthogonal_(p)
            elif "weight_ih" in name: nn.init.xavier_uniform_(p)
            elif "bias" in name: nn.init.zeros_(p)
        for name, p in self.decoder.named_parameters():
            if "weight_hh" in name: nn.init.orthogonal_(p)
            elif "weight_ih" in name: nn.init.xavier_uniform_(p)
            elif "bias" in name: nn.init.zeros_(p)

    def forward(self, x, h0=None):
        if self.layer == 0:
            if x.dtype in (torch.int64, torch.int32):
                x_emb = self.embedding(x)
            else:
                x_emb = x
            _, h = self.encoder(x_emb, h0)
        else:
            _, h = self.encoder(x, h0)

        B, T = x.shape[0], x.shape[1]
        dec_in = torch.zeros((B, 1, self.input_size), device=x.device, dtype=torch.float)
        outs, h_dec = [], h
        for _ in range(T):
            d, h_dec = self.decoder(dec_in, h_dec)
            logits = self.out(d)
            outs.append(logits)
            dec_in = logits.detach()
        return torch.cat(outs, dim=1), h

    def encode_step_from_token(self, token_id, h_prev):
        assert self.layer == 0, "encode_step_from_token only for layer 0"
        emb = self.embedding(token_id.view(1, 1))
        _, h_next = self.encoder(emb, h_prev)
        return h_next

    def encode_step_from_vec(self, x_vec, h_prev):
        # x_vec must be (1,1,input_size_of_this_layer)
        _, h_next = self.encoder(x_vec, h_prev)
        return h_next



