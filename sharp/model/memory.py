import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Memory(nn.Module):

    def __init__(self, input_size, hidden_size, embedding_dim=None, layer=0, sleep=True):
        super().__init__()
        self.layer = layer
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.decoder_is_frozen = False
        self.sleep = sleep

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
        if not self.sleep:
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

    def _init_ablation_lsm(self, rnn, spectral_radius=1e-20, input_scale=0.05):
        """
        Initialize the encoder as a deliberately poor reservoir / poor LSM:
          - weak input weights
          - very contractive recurrent weights
          - zero bias

        This makes long-timescale persistence much weaker.
        """
        with torch.no_grad():
            for name, param in rnn.named_parameters():
                if "weight_ih" in name:
                    param.normal_(mean=0.0, std=input_scale)

                elif "weight_hh" in name:
                    param.normal_(mean=0.0, std=0.05)

                    # rescale to small spectral radius
                    W = param.data
                    try:
                        eigvals = torch.linalg.eigvals(W).abs()
                        rho = eigvals.max().real.clamp_min(1e-6)
                        param.mul_(spectral_radius / rho)
                    except Exception:
                        # fallback if eigvals fails
                        param.mul_(spectral_radius / (param.norm() + 1e-6))

                elif "bias" in name:
                    param.zero_()

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
        if T > 1:
            h_pass = enc_out[:, 1, :].unsqueeze(0)   # (1, B, H)
        else:
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