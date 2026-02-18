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
            #self.AGC = ActiveRMSNormGain()
            #assert embedding_dim is not None, "embedding_dim required for layer " + str(self.layer)
            #self.embedding = nn.Linear(input_size, embedding_dim)
            self.encoder = nn.RNN(input_size, hidden_size, batch_first=True, nonlinearity='tanh')

        self.decoder = nn.RNN(input_size, hidden_size, batch_first=True, nonlinearity='tanh')
        self.out = nn.Linear(hidden_size, input_size)

        # init
        # for name, p in self.encoder.named_parameters():
        #     if "weight_hh" in name: nn.init.orthogonal_(p)
        #     elif "weight_ih" in name: nn.init.xavier_uniform_(p)
        #     elif "bias" in name: nn.init.zeros_(p)
        # for name, p in self.decoder.named_parameters():
        #     if "weight_hh" in name: nn.init.orthogonal_(p)
        #     elif "weight_ih" in name: nn.init.xavier_uniform_(p)
        #     elif "bias" in name: nn.init.zeros_(p)
    
    
    def forward(self, x, h=None):

        B, T = x.shape[0], x.shape[1]

        if self.layer == 0:
            x_emb = self.embedding(x)
        else:
            x_emb = x

        #print(x_emb.shape)
        for ii in range(T):
            _, h = self.encoder(x_emb[:,ii,:], h)

            if ii == 1:
                h_pass = h   

        
        dec_in = torch.zeros((B, 1, self.input_size), device=x.device, dtype=torch.float)
        outs, h_dec = [], h.unsqueeze(1)

        #print(h_dec, h_dec.shape)
        for _ in range(T):
            d, h_dec = self.decoder(dec_in, h_dec)
            logits = self.out(d)

            outs.append(logits)
            dec_in = logits.detach()
        return torch.cat(outs, dim=1), h, h_pass
    

    def freeze_decoder(self):
        """
        Freeze decoder + output projection.
        Encoder remains trainable.
        """
        self.decoder_is_frozen = True

        for p in self.decoder.parameters():
            p.requires_grad_(False)
        for p in self.out.parameters():
            p.requires_grad_(False)

    def unfreeze_decoder(self):
        """
        Unfreeze decoder + output projection.
        """
        self.decoder_is_frozen = False

        for p in self.decoder.parameters():
            p.requires_grad_(True)
        for p in self.out.parameters():
            p.requires_grad_(True)

    def decoder_is_frozen(self):
        """
        Returns True if all decoder params are frozen.
        """
        return not any(p.requires_grad for p in self.decoder.parameters()) \
               and not any(p.requires_grad for p in self.out.parameters())


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