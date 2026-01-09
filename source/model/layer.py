import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from .prediction import PredictionFiLM, Prediction
from .memory import MemoryContinuous, MemoryVAE, Memory

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
        tau=0.1,
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
        self.memory = MemoryContinuous(
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

        self.mem_optimizer = optimizer_class(
            self.memory.parameters(), **optimizer_kwargs
            )
        self.pred_optimizer = optimizer_class(
            self.prediction.parameters(), **optimizer_kwargs
            )
        # One optimizer that updates BOTH memory + prediction
        all_params = itertools.chain(
            self.memory.parameters(),
            self.prediction.parameters()
        )

        self.optimizer = optimizer_class(all_params, **optimizer_kwargs)


    def forward(self, x, h0=None, context=None):
        logits_reconstruction, h = self.memory(x, h0)   # mu: (B,H)
        h_seq = h.unsqueeze(1)                                 # (B,1,H)
        logits_prediction = self.prediction(h_seq, context)
        return logits_reconstruction, logits_prediction, h
    
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
    

    def compute_mem_loss(self, logits_rec, x, h_current=None, h_prev=None, gamma=1.0):
        """
        Compute memory (reconstruction) loss.
        No optimization step here — pure forward loss.
        """

        # ---- LAYER 0: Cross entropy over vocabulary ----
        if self.layer == 0:
            # logits_rec: (B,1,V) or (B,T,V)
            # x: (B,1) or (B,T)

            # Flatten to (B*T, V)
            B, T, V = logits_rec.shape
            logits_flat = logits_rec.reshape(B*T, V)
            targets_flat = x.reshape(B*T)

            loss = self.loss(logits_flat, targets_flat)

        # ---- HIGHER LAYERS: MSE ----
        else:
            # logits_rec: (B,1,H)
            # x: (B,1,H)
            loss = self.loss(logits_rec, x)

        if h_current != None:
            cont_loss = torch.mean((h_current - h_prev) ** 2)
        else:
            cont_loss = 0

        return loss + gamma * cont_loss


    def compute_pred_loss(self, logits_pred, y):
        """
        Compute prediction loss only.
        No params updated here.
        """

        # ---- LAYER 0: Cross entropy over vocabulary ----
        if self.layer == 0:
            # logits_pred: (B,1,V) -> (B,V)
            logits = logits_pred.squeeze(1)

            # y: (B,1) -> (B,)
            targets = y.squeeze(1).long()

            loss = self.loss(logits, targets)

        # ---- HIGHER LAYERS ----
        else:
            loss = self.loss(logits_pred, y)

        return loss

    def train_memory(self, x, h0=None, threshold=1e-4):
        logits_rec, h = self.memory(x, h0)

        loss = self.compute_mem_loss(logits_rec, x, h_current=h, h_prev=h0)

        # backprop
        if loss > threshold:
            self.mem_optimizer.zero_grad()
            loss.backward()
            self.mem_optimizer.step()

        return loss.item(), logits_rec
    
    def train_prediction(self, h, y, context=None, threshold=1e-4):
        logits_pred = self.prediction(h, context)
        #print(logits_pred.shape, z.shape, y.shape)
        if self.layer == 0:
            loss = self.loss(
                logits_pred,
                y.reshape(1)
            )
        else:
            loss = self.loss(logits_pred,  y)

        # backprop
        if loss > threshold:
            self.pred_optimizer.zero_grad()
            loss.backward()
            self.pred_optimizer.step()

        return loss.item(), logits_pred.detach()


    