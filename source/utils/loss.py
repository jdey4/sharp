import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyL1Loss(nn.Module):
    """
    Combined Cross-Entropy + L1 sparsity regularization loss.

    Args:
        lambda_l1 (float): Weighting factor for the L1 sparsity term.
        apply_to (str): What to apply L1 on. Options:
            - 'weights': L1 on model weights (structural sparsity)
            - 'activations': L1 on activations (population sparsity)
        target_attr (str): If apply_to='activations', specify which tensor name
            to expect (e.g., 'h' from autoencoder forward).
    """
    def __init__(self, lambda_l1=1e-4, apply_to='weights', target_attr=None):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.apply_to = apply_to
        self.target_attr = target_attr
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, targets, model=None, activations=None):
        """
        Compute combined loss.

        Args:
            logits (Tensor): Model output logits of shape (B, T, vocab_size).
            targets (Tensor): Ground-truth token indices of shape (B, T).
            model (nn.Module, optional): Model whose weights will be regularized.
            activations (Tensor, optional): Activations for L1 regularization
                (if apply_to='activations').

        Returns:
            total_loss (Tensor): Combined loss scalar.
        """
        # --- Cross-entropy ---
        ce_loss = self.ce(logits.view(-1, logits.size(-1)), targets.view(-1))

        # --- L1 regularization ---
        l1_loss = 0.0
        if self.apply_to == 'weights' and model is not None:
            for name, param in model.named_parameters():
                if 'weight' in name:
                    l1_loss += torch.sum(torch.abs(param))
        elif self.apply_to == 'activations' and activations is not None:
            l1_loss = torch.mean(torch.abs(activations))
        else:
            l1_loss = torch.tensor(0.0, device=logits.device)

        total_loss = ce_loss + self.lambda_l1 * l1_loss
        return total_loss

    def set_lambda(self, new_lambda):
        """Dynamically update sparsity strength."""
        self.lambda_l1 = new_lambda


class MSEL1Loss(nn.Module):
    """
    Combined Mean Squared Error (MSE) + L1 sparsity regularization loss.

    Args:
        lambda_l1 (float): Weighting factor for the L1 sparsity term.
        apply_to (str): Where to apply L1. Options:
            - 'weights': L1 on model weights (structural sparsity)
            - 'activations': L1 on hidden activations (population sparsity)
        target_attr (str, optional): Descriptive name for activation source.
    """
    def __init__(self, lambda_l1=1e-4, apply_to='weights', target_attr=None):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.apply_to = apply_to
        self.target_attr = target_attr
        self.mse = nn.MSELoss()

    def forward(self, preds, targets, model=None, activations=None):
        """
        Compute combined MSE + L1 loss.

        Args:
            preds (Tensor): Model predictions (B, T, D) or (B, D).
            targets (Tensor): Ground truth of same shape.
            model (nn.Module, optional): Module for L1 on weights.
            activations (Tensor, optional): Hidden activations for L1 on activity.

        Returns:
            total_loss (Tensor): Combined scalar loss.
        """
        # --- MSE reconstruction/prediction loss ---
        mse_loss = self.mse(preds, targets)

        # --- L1 regularization ---
        l1_loss = torch.tensor(0.0, device=preds.device)
        if self.apply_to == 'weights' and model is not None:
            for name, param in model.named_parameters():
                if 'weight' in name:
                    l1_loss += torch.sum(torch.abs(param))
        elif self.apply_to == 'activations' and activations is not None:
            l1_loss = torch.mean(torch.abs(activations))

        total_loss = mse_loss + self.lambda_l1 * l1_loss
        return total_loss

    def set_lambda(self, new_lambda: float):
        """Dynamically update sparsity strength λ during training."""
        self.lambda_l1 = new_lambda


class CrossEntropyLayerLoss(nn.Module):
    """
    Combined CE loss for layer 0:
      - AE reconstruction loss
      - Next-token prediction loss
    """
    def __init__(self, recon_weight=1.0, pred_weight=1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.recon_weight = recon_weight
        self.pred_weight = pred_weight

    def forward(
                self, 
                logits_recon, targets_recon,
                logits_pred, targets_pred
            ):
        
        # flatten for CE
        B, T, V = logits_recon.shape
        
        loss_recon = self.ce(
            logits_recon.reshape(B*T, V),
            targets_recon.reshape(B*T)
        )

        Bp, Tp, Vp = logits_pred.shape
        
        loss_pred = self.ce(
            logits_pred.reshape(Bp*Tp, Vp),
            targets_pred.reshape(Bp*Tp)
        )
        
        return self.recon_weight * loss_recon + self.pred_weight * loss_pred
    
    
    
class MSELayerLoss(nn.Module):
    """
    Combined MSE loss for upper layers:
        - Autoencoder reconstruction loss
        - Next hidden-state prediction loss
    """
    def __init__(self, recon_weight=1.0, pred_weight=1.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.recon_weight = recon_weight
        self.pred_weight = pred_weight

    def forward(self,
                logits_reconstruction, targets_reconstruction,
                logits_prediction,     targets_prediction):
        
        # Reconstruction loss (AE)
        loss_recon = self.mse(logits_reconstruction, targets_reconstruction)

        # Prediction loss (next-state prediction)
        loss_pred  = self.mse(logits_prediction, targets_prediction)

        # Weighted sum
        return self.recon_weight * loss_recon + self.pred_weight * loss_pred
