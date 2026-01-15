import torch
import torch.nn as nn
import torch.nn.functional as F

class ActiveRMSNormGain(nn.Module):
    def __init__(self, eps=1e-8, init_gain=1.0):
        super().__init__()
        self.eps = eps
        self.log_gain = nn.Parameter(torch.tensor(float(init_gain)).log())

    def forward(self, x):
        # x: (..., D), sparse (>=0)
        m = (x > 0).float()
        denom = (m.sum(dim=-1, keepdim=True).clamp_min(1.0))
        rms = (x.pow(2).sum(dim=-1, keepdim=True) / denom).add(self.eps).sqrt()
        return torch.exp(self.log_gain) * x / rms


class Memory(nn.Module):
    r"""
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

    def __init__(self, input_size, hidden_size, embedding_dim=None, layer=0, tau=0.1):
        super().__init__()
        self.layer = layer
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.tau = tau

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
    
    # ----------------------------------------------------------
    def threshold(self, x):
        # Hard threshold ReLU
        return torch.where(x > self.tau, x, torch.zeros_like(x))

    def forward(self, x, h0=None):
        if self.layer == 0:
            if x.dtype in (torch.int64, torch.int32):
                x_emb = self.embedding(x)
            else:
                x_emb = x
            _, h = self.encoder(x_emb, h0)
        else:
            _, h = self.encoder(x, h0)

        h = self.threshold(h)

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
    

class MemoryVAE(nn.Module):
    r"""
        Variational Autoencoder-style recurrent memory module.

        q(z | x) = N(mu, sigma^2), with:
        - LayerNorm on mu
        - hard thresholded ReLU on mu (mu > tau -> keep, else 0)
        - z sparsified using the same mu mask

        During training, add a KL term:
            KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    """

    def __init__(self, input_size, hidden_size, embedding_dim=None, layer=0, tau=0.1):
        super().__init__()
        self.layer = layer
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.tau = tau

        # ------------------------------------------------------
        #                    Encoder
        # ------------------------------------------------------
        if layer == 0:
            assert embedding_dim is not None, "embedding_dim required for layer 0"
            self.embedding = nn.Embedding(input_size, embedding_dim)
            enc_in = embedding_dim
        else:
            enc_in = input_size

        self.encoder = nn.RNN(enc_in, hidden_size, batch_first=True)

        # variational heads
        self.fc_mu = nn.Linear(hidden_size, hidden_size)
        self.fc_logvar = nn.Linear(hidden_size, hidden_size)

        # homeostatic normalization of mu
        self.mu_norm = nn.LayerNorm(hidden_size)

        # ------------------------------------------------------
        #                    Decoder
        # ------------------------------------------------------
        self.decoder = nn.RNN(input_size, hidden_size, batch_first=True)
        self.reconstruction_out = nn.Linear(hidden_size, input_size)

        # ------------------------------------------------------
        #                   Initialization
        # ------------------------------------------------------
        for name, p in self.encoder.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

        for name, p in self.decoder.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

        nn.init.xavier_uniform_(self.fc_mu.weight)
        nn.init.xavier_uniform_(self.fc_logvar.weight)

    # ----------------------------------------------------------
    def threshold(self, x):
        # Hard threshold ReLU
        return torch.where(x > self.tau, x, torch.zeros_like(x))

    # --------------------------------------------------------------
    def reparameterize(self, mu, logvar):
        """Sample z ~ N(mu, sigma^2) then sparsify using mu mask."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        z_sparse = z * (z > self.tau)  # gate z with same support as mu
        return z_sparse

    # --------------------------------------------------------------
    def forward(self, x, h0=None):
        r"""
            Returns:
                logits  : (B, T, input_size)
                mu      : (B, H)  AFTER norm + threshold
                logvar  : (B, H)
        """
        # ----- Embedding (Layer 0 only) -----
        if self.layer == 0 and x.dtype in (torch.int64, torch.int32):
            x = self.embedding(x)

        # ----- Encode sequence → (B, H) -----
        _, h_enc = self.encoder(x, h0)   # h_enc: (1,B,H)
        h_enc = h_enc.squeeze(0)

        # μ, logσ²
        mu = self.fc_mu(h_enc)
        mu = self.mu_norm(mu)       # LayerNorm
        #mu = self.threshold(mu)     # thresholded ReLU

        logvar = self.fc_logvar(h_enc)

        # latent z (sparse)
        z = self.reparameterize(mu, logvar)  # (B,H)
        h = z.unsqueeze(0)                   # (1,B,H) for decoder

        # ----- Decode autoregressively -----
        B, T = x.shape[0], x.shape[1]
        dec_in = torch.zeros((B, 1, self.input_size), device=x.device)

        outs = []
        h_dec = h
        for _ in range(T):
            d, h_dec = self.decoder(dec_in, h_dec)
            logits = self.reconstruction_out(d)
            outs.append(logits)
            dec_in = logits.detach()

        logits = torch.cat(outs, dim=1)
        return logits, z

    # --------------------------------------------------------------
    #     Optional incremental encode helpers
    # --------------------------------------------------------------
    @torch.no_grad()
    def encode_step_from_token(self, token_id, h_prev):
        """Single-step encode for discrete token (layer 0). Return sampled z."""
        assert self.layer == 0

        # ---- Encode token ----
        emb = self.embedding(token_id.view(1, 1))     # (1,1,E)
        _, h_next = self.encoder(emb, h_prev)         # (1,1,H)

        # ---- Compute mu, logvar ----
        h_vec = h_next.squeeze(0)                     # (1,H)
        mu = self.threshold(self.mu_norm(self.fc_mu(h_vec)))
        logvar = self.fc_logvar(h_vec)

        # ---- SAMPLE z using existing reparameterize() ----
        z = self.reparameterize(mu, logvar)           # (1,H)

        # reshape for FiLM prediction
        z_seq = z.unsqueeze(1)                        # (1,1,H)

        return z_seq, h_next


    @torch.no_grad()
    def encode_step_from_vec(self, x_vec, h_prev):
        """Single-step encode for continuous vector input. Return sampled z."""
        # ---- Encode ----
        _, h_next = self.encoder(x_vec, h_prev)       # (1,1,H)

        # ---- Compute mu, logvar ----
        h_vec = h_next.squeeze(0)                     # (1,H)
        mu = self.threshold(self.mu_norm(self.fc_mu(h_vec)))
        logvar = self.fc_logvar(h_vec)

        # ---- SAMPLE z ----
        z = self.reparameterize(mu, logvar)

        # reshape
        z_seq = z.unsqueeze(1)                        # (1,1,H)

        return z_seq, h_next
    


class MemoryContinuous(nn.Module):

    def __init__(self, input_size, hidden_size, embedding_dim=None, layer=0, tau=0.1):
        super().__init__()
        self.layer = layer
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.tau = tau

        if layer == 0:
            assert embedding_dim is not None, "embedding_dim required for layer 0"
            self.embedding = nn.Embedding(input_size, embedding_dim)
            self.encoder = nn.RNN(embedding_dim, hidden_size, batch_first=True)
        else:
            self.AGC = ActiveRMSNormGain()
            assert embedding_dim is not None, "embedding_dim required for layer " + str(self.layer)
            self.embedding = nn.Linear(input_size, embedding_dim)
            self.encoder = nn.RNN(embedding_dim, hidden_size, batch_first=True)

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
    
    # ----------------------------------------------------------
    def threshold(self, x):
        # Hard threshold ReLU
        return F.relu(x - self.tau)
    #torch.where(x > self.tau, x, torch.zeros_like(x))

    def forward(self, x, h0=None):
        if self.layer != 0:
            x = self.AGC(x)

        x_emb = self.embedding(x[:,-1])
        _, h = self.encoder(x_emb, h0)

        h = self.threshold(h)

        B, T = x.shape[0], x.shape[1]
        dec_in = torch.zeros((B, 1, self.input_size), device=x.device, dtype=torch.float)
        outs, h_dec = [], h.unsqueeze(0)

        #print(h_dec, h_dec.shape)
        for _ in range(T):
            d, h_dec = self.decoder(dec_in, h_dec)
            logits = self.out(d)

            outs.append(logits)
            dec_in = logits.detach()
        return torch.cat(outs, dim=1), h
    
    @torch.no_grad()
    def encode_step_from_token(self, token_id, h_prev):
        assert self.layer == 0, "encode_step_from_token only for layer 0"
        emb = self.embedding(token_id.view(1, 1))
        _, h_next = self.encoder(emb, h_prev)
        return self.threshold(h_next)

    @torch.no_grad()
    def encode_step_from_vec(self, x_vec, h_prev):
        # x_vec must be (1,1,input_size_of_this_layer)
        x_vec = self.AGC(x_vec)
        emb = self.embedding(x_vec)
        _, h_next = self.encoder(emb, h_prev)
        return self.threshold(h_next)