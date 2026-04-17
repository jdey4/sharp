import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Memory(nn.Module):

    def __init__(self, input_size, hidden_size, embedding_dim=None, layer=0, bad_init=False):
        super().__init__()
        self.layer = layer
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.decoder_is_frozen = False
        self.bad_init = bad_init

        if layer == 0:
            assert embedding_dim is not None, "embedding_dim required for layer 0"
            self.embedding = nn.Embedding(input_size, embedding_dim)
            self.encoder = nn.RNN(
                embedding_dim, hidden_size, batch_first=True, nonlinearity='tanh'
            )
        else:
            assert embedding_dim is not None, "embedding_dim required for layer " + str(self.layer)
            self.embedding = nn.Linear(input_size, embedding_dim)
            self.encoder = nn.RNN(
                embedding_dim, hidden_size, batch_first=True, nonlinearity='tanh'
            )

        self.decoder = nn.RNN(
            input_size, hidden_size, batch_first=True, nonlinearity='tanh'
        )
        self.out = nn.Linear(hidden_size, input_size)

        # --------------------------------------------------
        # Initialization
        # --------------------------------------------------
        if self.bad_init and self.layer != 0:
            # deliberately weak / bad reservoir-like dynamics
            self._init_ablation_lsm(self.encoder)
        else:
            self._init_default_rnn(self.encoder)

        # keep decoder/output reasonable
        self._init_default_rnn(self.decoder)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)

        if isinstance(self.embedding, nn.Linear):
            nn.init.xavier_uniform_(self.embedding.weight)
            if self.embedding.bias is not None:
                nn.init.zeros_(self.embedding.bias)

    def _init_default_rnn(self, rnn):
        """
        Reasonable default initialization.
        """
        for name, param in rnn.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def _init_ablation_lsm(self, rnn):
        """
        Initialize everything to 1 (no randomness, no scaling).
        """
        with torch.no_grad():
            for name, param in rnn.named_parameters():
                if "weight_ih" in name:
                    param.fill_(1.0)

                elif "weight_hh" in name:
                    param.fill_(1.0)

                elif "bias" in name:
                    param.fill_(0.0)

    def forward(self, x, h=None):
        if x.dim() == 2:
            B, T = x.shape
        elif x.dim() == 3:
            B, T, _ = x.shape
        else:
            raise ValueError(f"Expected x to be 2D or 3D, got shape {x.shape}")

        # (B, T, E)
        x_emb = self.embedding(x)

        # enc_out: (B, T, H)
        # h_last:  (1, B, H)
        enc_out, h_last = self.encoder(x_emb, h)

        # For stride-1 sliding windows, carry the state aligned to the next window.
        # Current window: [x_t, x_{t+1}, ..., x_{t+T-1}]
        # Next window:    [x_{t+1}, x_{t+2}, ..., x_{t+T}]
        # So pass the hidden state at token index 1.
        h_pass = enc_out[:, 0, :].unsqueeze(0)   # fallback

        # Final hidden state for decoder init
        h_dec = h_last

        dec_in = torch.zeros((B, 1, self.input_size), device=x.device, dtype=torch.float)
        outs = []

        for _ in range(T):
            d, h_dec = self.decoder(dec_in, h_dec)
            logits = self.out(d)
            outs.append(logits)
            dec_in = logits.detach()

        return torch.cat(outs, dim=1), h_last, h_pass

    @torch.no_grad()
    def encode_step_from_token(self, token_id, h_prev):
        assert self.layer == 0, "encode_step_from_token only for layer 0"
        emb = self.embedding(token_id.view(1, 1))
        _, h_next = self.encoder(emb, h_prev)
        return h_next

    @torch.no_grad()
    def encode_step_from_vec(self, x_vec, h_prev):
        emb = self.embedding(x_vec)
        _, h_next = self.encoder(emb, h_prev)
        return h_next
    

class MemoryMultiHeadRecall(nn.Module):
    """
    Memory module with a standard encoder, but replacing the recurrent decoder
    with multiple linear recall heads. Each head reconstructs one token position
    in the input window from the final hidden state.

    For a window of length T, the model produces:
        x_hat_1 = head_1(h_T)
        x_hat_2 = head_2(h_T)
        ...
        x_hat_T = head_T(h_T)

    This explicitly pressures the final hidden state to preserve the whole
    window in a position-addressable form.

    Args:
        input_size (int): Vocabulary size for layer 0, or feature dimension for higher layers.
        hidden_size (int): Encoder hidden size.
        embedding_dim (int): Embedding size for input tokens / vectors.
        window_size (int): Number of tokens in each training window.
        layer (int): 0 for token input, >0 for vector input.
        bad_init (bool): If True and layer != 0, initializes encoder as weak LSM-style ablation.
        use_head_norm (bool): Whether to normalize final hidden state before recall.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        embedding_dim=None,
        window_size=4,
        layer=0,
        bad_init=False,
    ):
        super().__init__()
        self.layer = layer
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.decoder_is_frozen = False
        self.bad_init = bad_init

        if layer == 0:
            assert embedding_dim is not None, "embedding_dim required for layer 0"
            self.embedding = nn.Embedding(input_size, embedding_dim)
            self.encoder = nn.RNN(
                embedding_dim, hidden_size, batch_first=True, nonlinearity="tanh"
            )
        else:
            assert embedding_dim is not None, f"embedding_dim required for layer {layer}"
            self.embedding = nn.Linear(input_size, embedding_dim)
            self.encoder = nn.RNN(
                embedding_dim, hidden_size, batch_first=True, nonlinearity="tanh"
            )

        # One linear recall head per position in the window
        self.recall_heads = nn.ModuleList([
            nn.Linear(hidden_size, input_size) for _ in range(window_size)
        ])

        # --------------------------------------------------
        # Initialization
        # --------------------------------------------------
        if self.bad_init and self.layer != 0:
            self._init_ablation_lsm(self.encoder)
        else:
            self._init_default_rnn(self.encoder)

        for head in self.recall_heads:
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)

        if isinstance(self.embedding, nn.Linear):
            nn.init.xavier_uniform_(self.embedding.weight)
            if self.embedding.bias is not None:
                nn.init.zeros_(self.embedding.bias)

    def _init_default_rnn(self, rnn):
        for name, param in rnn.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def _init_ablation_lsm(self, rnn):
        with torch.no_grad():
            for name, param in rnn.named_parameters():
                if "weight_ih" in name:
                    param.fill_(1.0)
                elif "weight_hh" in name:
                    param.fill_(1.0)
                elif "bias" in name:
                    param.fill_(0.0)

    def forward(self, x, h=None):
        if x.dim() == 2:
            B, T = x.shape
        elif x.dim() == 3:
            B, T, _ = x.shape
        else:
            raise ValueError(f"Expected x to be 2D or 3D, got shape {x.shape}")

        if T != self.window_size:
            raise ValueError(
                f"Input window length {T} does not match configured window_size {self.window_size}"
            )

        # (B, T, E)
        x_emb = self.embedding(x)

        # enc_out: (B, T, H)
        # h_last:  (1, B, H)
        enc_out, h_last = self.encoder(x_emb, h)

        # final hidden state for whole-window recall
        h_final = h_last[-1]              # (B, H)

        # pass state for stride-1 sliding windows
        h_pass = enc_out[:, 0, :].unsqueeze(0)

        # reconstruct all positions from the same final hidden state
        logits_per_pos = [head(h_final) for head in self.recall_heads]   # list of (B, input_size)
        logits = torch.stack(logits_per_pos, dim=1)                      # (B, T, input_size)

        return logits, h_last, h_pass

    @torch.no_grad()
    def encode_step_from_token(self, token_id, h_prev):
        assert self.layer == 0, "encode_step_from_token only for layer 0"
        emb = self.embedding(token_id.view(1, 1))
        _, h_next = self.encoder(emb, h_prev)
        return h_next

    @torch.no_grad()
    def encode_step_from_vec(self, x_vec, h_prev):
        emb = self.embedding(x_vec)
        _, h_next = self.encoder(emb, h_prev)
        return h_next