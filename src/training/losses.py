"""
Custom loss functions for guardrail training
Includes Focal Loss, Tversky Loss, and multi-task guardrail loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced classification
    
    FL(p_t) = -α(1-p_t)^γ log(p_t)
    
    Focuses on hard examples and down-weights easy examples
    Critical for guardrail task with class imbalance
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean',
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (batch_size, num_classes) - logits
            targets: (batch_size,) - class labels
        
        Returns:
            loss: scalar or (batch_size,) depending on reduction
        """
        # Cross entropy loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.class_weights)
        
        # p_t: probability of true class
        pt = torch.exp(-ce_loss)
        
        # Focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class TverskyLoss(nn.Module):
    """
    Tversky Loss for imbalanced classification
    Generalization of Dice loss
    
    Useful for controlling false positives vs false negatives trade-off
    """
    
    def __init__(
        self,
        alpha: float = 0.7,  # Weight for false positives
        beta: float = 0.3,   # Weight for false negatives
        smooth: float = 1.0,
    ):
        super().__init__()
        
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (batch_size, num_classes) - logits
            targets: (batch_size,) - class labels
        
        Returns:
            loss: scalar
        """
        # Convert to one-hot
        num_classes = inputs.size(1)
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()
        
        # Probabilities
        probs = F.softmax(inputs, dim=-1)
        
        # True positives, false positives, false negatives
        tp = (probs * targets_one_hot).sum(dim=0)
        fp = (probs * (1 - targets_one_hot)).sum(dim=0)
        fn = ((1 - probs) * targets_one_hot).sum(dim=0)
        
        # Tversky index
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        
        # Tversky loss
        loss = 1 - tversky.mean()
        
        return loss


class GuardrailLoss(nn.Module):
    """
    Multi-task loss for guardrail training
    
    Combines:
    1. Classification loss (focal loss)
    2. Auxiliary loss (MoE load balancing)
    3. Router loss (encourage target routing ratio)
    4. Pattern loss (pattern detector calibration)
    """
    
    def __init__(
        self,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        aux_loss_weight: float = 0.01,
        router_loss_weight: float = 0.1,
        pattern_loss_weight: float = 0.05,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        
        self.focal_loss = FocalLoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
            class_weights=class_weights,
        )
        
        self.aux_loss_weight = aux_loss_weight
        self.router_loss_weight = router_loss_weight
        self.pattern_loss_weight = pattern_loss_weight
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        aux_loss: torch.Tensor,
        router_logits: torch.Tensor,
        pattern_scores: Optional[dict] = None,
    ) -> dict:
        """
        Compute multi-task loss
        
        Args:
            logits: (batch_size, num_classes)
            labels: (batch_size,)
            aux_loss: scalar - MoE load balancing loss
            router_logits: (batch_size, 2) - routing logits
            pattern_scores: Dict of pattern detector scores
        
        Returns:
            dict with:
                - total_loss: Total loss
                - classification_loss: Main classification loss
                - aux_loss: MoE load balancing loss
                - router_loss: Routing loss
                - pattern_loss: Pattern detector loss
        """
        # 1. Classification loss
        classification_loss = self.focal_loss(logits, labels)
        
        # 2. Auxiliary loss (MoE load balancing)
        weighted_aux_loss = self.aux_loss_weight * aux_loss
        
        # 3. Router loss
        # Encourage benign → fast, attacks → deep
        benign_mask = (labels == 0)
        attack_mask = ~benign_mask
        
        router_probs = F.softmax(router_logits, dim=-1)  # (batch_size, 2)
        
        # Benign should prefer fast branch (index 0)
        if benign_mask.any():
            benign_router_loss = -router_probs[benign_mask, 0].log().mean()
        else:
            benign_router_loss = 0.0
        
        # Attacks should prefer deep branch (index 1)
        if attack_mask.any():
            attack_router_loss = -router_probs[attack_mask, 1].log().mean()
        else:
            attack_router_loss = 0.0
        
        router_loss = (benign_router_loss + attack_router_loss) / 2.0
        weighted_router_loss = self.router_loss_weight * router_loss
        
        # 4. Pattern detector loss (optional)
        pattern_loss = 0.0
        if pattern_scores is not None:
            # Pattern detectors should fire high for attacks
            for name, scores in pattern_scores.items():
                if attack_mask.any():
                    # Attacks should have high pattern scores
                    attack_pattern_loss = (1.0 - scores[attack_mask]).mean()
                    pattern_loss += attack_pattern_loss
            
            pattern_loss = pattern_loss / len(pattern_scores)
            weighted_pattern_loss = self.pattern_loss_weight * pattern_loss
        else:
            weighted_pattern_loss = 0.0
        
        # 5. Total loss
        total_loss = (
            classification_loss +
            weighted_aux_loss +
            weighted_router_loss +
            weighted_pattern_loss
        )
        
        return {
            'total_loss': total_loss,
            'classification_loss': classification_loss,
            'aux_loss': weighted_aux_loss,
            'router_loss': weighted_router_loss,
            'pattern_loss': weighted_pattern_loss,
        }


class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing cross entropy loss
    Prevents overconfidence and improves generalization
    """
    
    def __init__(
        self,
        num_classes: int,
        smoothing: float = 0.1,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (batch_size, num_classes) - logits
            targets: (batch_size,) - class labels
        
        Returns:
            loss: scalar
        """
        log_probs = F.log_softmax(inputs, dim=-1)
        
        # One-hot targets
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()
        
        # Smooth labels
        smooth_targets = targets_one_hot * self.confidence + (1 - targets_one_hot) * (self.smoothing / (self.num_classes - 1))
        
        # KL divergence
        loss = -(smooth_targets * log_probs).sum(dim=-1).mean()
        
        return loss


class BalancedCrossEntropyLoss(nn.Module):
    """
    Class-balanced cross entropy loss
    Automatically computes class weights based on frequency
    """
    
    def __init__(
        self,
        beta: float = 0.9999,
        reduction: str = 'mean',
    ):
        super().__init__()
        
        self.beta = beta
        self.reduction = reduction
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        class_counts: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            inputs: (batch_size, num_classes) - logits
            targets: (batch_size,) - class labels
            class_counts: (num_classes,) - number of samples per class
        
        Returns:
            loss: scalar
        """
        if class_counts is None:
            # Uniform weights
            return F.cross_entropy(inputs, targets, reduction=self.reduction)
        
        # Compute effective number of samples
        effective_num = 1.0 - torch.pow(self.beta, class_counts)
        weights = (1.0 - self.beta) / effective_num
        weights = weights / weights.sum() * len(weights)
        
        return F.cross_entropy(inputs, targets, weight=weights, reduction=self.reduction)

