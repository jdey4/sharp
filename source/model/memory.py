import torch
import torch.nn as nn
import torch.nn.functional as F


class Memory(nn.Module):

    def __init__(self, input_size, hidden_size, embedding_dim=None, layer=0):
        super().__init__()
        self.layer = layer
        self.input_size = input_size
        self.hidden_size = hidden_size

        if layer == 0:
            assert embedding_dim is not None, "embedding_dim required for layer 0"
            self.embedding = nn.Embedding(input_size, embedding_dim)
            self.encoder = nn.RNN(embedding_dim, hidden_size, batch_first=True, nonlinearity='tanh')
        else:
            self.encoder = nn.RNN(input_size, hidden_size, batch_first=True, nonlinearity='tanh')

        # init
        for name, p in self.encoder.named_parameters():
            if "weight_hh" in name: nn.init.orthogonal_(p)
            elif "weight_ih" in name: nn.init.xavier_uniform_(p)
            elif "bias" in name: nn.init.zeros_(p)
    
    
    def forward(self, x, h=None):
        B, T = x.shape[0], x.shape[1]

        if self.layer == 0:
            x_emb = self.embedding(x)
        else:
            x_emb = x

        #print(x_emb.shape)
        for ii in range(T):
            _, h = self.encoder(x_emb[:,ii:ii+1,:], h)

            if ii == 1:
                h_pass = h   

        return h, h_pass.detach()
    
    @torch.no_grad()
    def encode_step_from_token(self, token_id, h_prev):
        assert self.layer == 0, "encode_step_from_token only for layer 0"
        emb = self.embedding(token_id.view(1, 1))
        _, h_next = self.encoder(emb, h_prev)
        return h_next

    @torch.no_grad()
    def encode_step_from_vec(self, x_vec, h_prev):
        _, h_next = self.encoder(x_vec, h_prev)
        return h_next
    

