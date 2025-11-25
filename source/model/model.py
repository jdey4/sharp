import torch
import torch.nn as nn
import torch.nn.functional as F
from .prediction import PredictionFiLM
from .memory import MemoryVAE

class Layer(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_dim=None, layer=0, context_size=0, tau=0.5):
        super().__init__()
        self.layer = layer
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.context_size = context_size

        self.memory = MemoryVAE(
            self.input_size, self.hidden_size, embedding_dim=embedding_dim, 
            layer=self.layer, tau=tau
        )
        self.prediction = PredictionFiLM(
            self.hidden_size, self.input_size,
            context_size=self.context_size
        )

    def forward(self, x, h0=None, context=None):
        logits_reconstruction, mu, logvar = self.memory(x, h0)   # mu: (B,H)
        mu_seq = mu.unsqueeze(1)                                 # (B,1,H)
        logits_prediction = self.prediction(mu_seq, context)
        return logits_reconstruction, logits_prediction, mu, logvar
    
    @torch.no_grad()
    def generate_sample(self, x=None, h0=None, temperature=1.0):
        """
        One generative step:
          - Layer 0: sample next token from softmax(prediction(mu))
          - Higher layers: generate next continuous vector from prediction(mu)

        Returns:
            x_next : (1,1) token id (layer 0) or (1,1,input_size) vector (higher layers)
            mu     : (1,1,hidden_size)  place-cell-like code for this step
            h_next : (1,1,hidden_size)  updated memory hidden state
        """
        device = next(self.parameters()).device

        # ---------------- LAYER 0: TOKEN GENERATION ----------------
        if self.layer == 0:
            # x is a token id
            if x is None:
                x = torch.randint(0, self.input_size, (1,1), device=device)  
            

            # encode step: get mu_seq (1,1,H) and next hidden (1,1,H)
            mu_seq, h_next = self.memory.encode_step_from_token(x, h0)

            # context-free prediction over vocab, shape (1,1,vocab)
            logits = self.prediction(mu_seq)

            # sample token from softmax
            logits_flat = logits[:, 0, :] / temperature   # (1,vocab)
            probs = torch.softmax(logits_flat, dim=-1)
            x_next = torch.multinomial(probs, num_samples=1)  # (1,1)

            return x_next, mu_seq, h_next

        # --------------- HIGHER LAYERS: CONTINUOUS -----------------
        else:
            if x is None:
                x = torch.zeros((1, 1, self.input_size), device=device)
            else:
                x = x.to(device)
                if x.dim() == 2:
                    x = x.unsqueeze(1)   # (1,1,input_size)

            mu_seq, h_next = self.memory.encode_step_from_vec(x, h0)  # (1,1,H)

            # prediction gives next continuous state/vector
            x_next = self.prediction(mu_seq)         # (1,1,input_size)

            return x_next, mu_seq, h_next