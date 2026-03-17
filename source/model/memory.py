import torch
import torch.nn as nn
import torch.nn.functional as F


class Memory(nn.Module):

    def __init__(self, input_size, hidden_size, embedding_dim=None, layer=0):
        super().__init__()
        self.layer = layer
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.decoder_is_frozen = False

        if layer == 0:
            assert embedding_dim is not None, "embedding_dim required for layer 0"
            self.embedding = nn.Embedding(input_size, embedding_dim)
            self.encoder = nn.RNN(embedding_dim, hidden_size, batch_first=True, nonlinearity='tanh')
        else:
            assert embedding_dim is not None, "embedding_dim required for layer " + str(self.layer)
            self.embedding = nn.Linear(input_size, embedding_dim)
            self.encoder = nn.RNN(embedding_dim, hidden_size, batch_first=True, nonlinearity='tanh')

        self.decoder = nn.RNN(input_size, hidden_size, batch_first=True, nonlinearity='tanh')
        self.out = nn.Linear(hidden_size, input_size)

    
    def forward(self, x, h=None):

        B, T = x.shape[0], x.shape[1]

        x_emb = self.embedding(x)
        enc_out, h = self.encoder(x_emb, h)
        h_pass = enc_out[:, 1, :].unsqueeze(0)
          

        
        dec_in = torch.zeros((B, 1, self.input_size), device=x.device, dtype=torch.float)
        outs, h_dec = [], h.unsqueeze(1)

        for _ in range(T):
            d, h_dec = self.decoder(dec_in, h_dec)
            logits = self.out(d)

            outs.append(logits)
            dec_in = logits.detach()
        return torch.cat(outs, dim=1), h, h_pass
    

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