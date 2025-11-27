import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from .prediction import PredictionFiLM
from .memory import MemoryVAE

class Layer(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        loss_function,
        optimizer_class=torch.optim.Adam,    
        optimizer_kwargs=None,               
        embedding_dim=None,
        layer=0,
        context_size=0,
        tau=0.5,
    ):
        super().__init__()
        self.layer = layer
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.context_size = context_size
        self.loss = loss_function

        # default optimizer kwargs
        if optimizer_kwargs is None:
            optimizer_kwargs = {"lr": 1e-3, "weight_decay": 1e-8}

        # ---------------------------------------------------------
        #                    MEMORY + PREDICTION
        # ---------------------------------------------------------            
        self.memory = MemoryVAE(
            self.input_size, self.hidden_size, embedding_dim=embedding_dim, 
            layer=self.layer, tau=tau
        )
        self.prediction = PredictionFiLM(
            self.hidden_size, self.input_size,
            context_size=self.context_size
        )

        # ---------------------------------------------------------
        #                    OPTIMIZER
        # ---------------------------------------------------------
        # One optimizer that updates BOTH memory + prediction
        all_params = itertools.chain(
            self.memory.parameters(),
            self.prediction.parameters()
        )

        self.optimizer = optimizer_class(all_params, **optimizer_kwargs)

    def forward(self, x, h0=None, context=None):
        logits_reconstruction, z, logvar = self.memory(x, h0)   # mu: (B,H)
        z_seq = z.unsqueeze(1)                                 # (B,1,H)
        logits_prediction = self.prediction(z_seq, context)
        return logits_reconstruction, logits_prediction, z, logvar
    
    @torch.no_grad()
    def generate_sample(self, x=None, h0=None, temperature=1.0):
        r"""
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
            z_seq, h_next = self.memory.encode_step_from_token(x, h0)

            # context-free prediction over vocab, shape (1,1,vocab)
            logits = self.prediction(z_seq)

            # sample token from softmax
            logits_flat = logits[:, 0, :] / temperature   # (1,vocab)
            probs = torch.softmax(logits_flat, dim=-1)
            x_next = torch.multinomial(probs, num_samples=1)  # (1,1)

            return x_next, z_seq, h_next

        # --------------- HIGHER LAYERS: CONTINUOUS -----------------
        else:
            if x is None:
                x = torch.zeros((1, 1, self.input_size), device=device)
            else:
                x = x.to(device)
                if x.dim() == 2:
                    x = x.unsqueeze(1)   # (1,1,input_size)

            z_seq, h_next = self.memory.encode_step_from_vec(x, h0)  # (1,1,H)

            # prediction gives next continuous state/vector
            x_next = self.prediction(z_seq)         # (1,1,input_size)

            return x_next, z_seq, h_next
        
    def train_step(self, x, y, h0=None, context=None, threshold=1e-4):
        logits_rec, logits_pred, z, logvar = self.forward(x, h0, context)

        # layer-specific loss
        loss = self.loss(
            logits_rec,  x,
            logits_pred, y
        )

        # backprop
        if loss > threshold:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss.item(), logits_rec, logits_pred, z, logvar