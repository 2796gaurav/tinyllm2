"""
Adaptive Router: Complexity-Based Branch Selection
Routes ~70% to fast branch, ~30% to deep branch
Based on input complexity estimation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class AdaptiveRouter(nn.Module):
    """
    Adaptive router for dual-branch selection
    
    Routing decision based on:
    1. Input complexity (learned)
    2. Pattern detection confidence
    3. Embedding entropy
    
    Target: 70% fast branch, 30% deep branch
    """
    
    def __init__(
        self,
        d_model: int = 384,
        hidden_dim: int = 128,
        complexity_threshold: float = 0.6,
        use_learned_threshold: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.use_learned_threshold = use_learned_threshold
        
        # Complexity estimator (MLP)
        self.complexity_estimator = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),  # Output: [0, 1], 0 = simple, 1 = complex
        )
        
        # Learned threshold (if enabled)
        if use_learned_threshold:
            self.complexity_threshold = nn.Parameter(
                torch.tensor(complexity_threshold)
            )
        else:
            self.register_buffer(
                'complexity_threshold',
                torch.tensor(complexity_threshold)
            )
        
        # Pattern confidence weight (how much to trust pattern detectors)
        self.pattern_confidence_weight = nn.Parameter(torch.tensor(0.3))
        
        # Entropy-based complexity (optional, for calibration)
        self.use_entropy = True
    
    def compute_entropy(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute embedding entropy as complexity indicator
        High entropy = more complex/uncertain
        
        Args:
            embeddings: (batch_size, seq_len, d_model)
        
        Returns:
            entropy: (batch_size,)
        """
        batch_size, seq_len, d_model = embeddings.size()
        
        # Compute attention distribution entropy
        # Simplified: use embedding variance as proxy
        variance = embeddings.var(dim=1).mean(dim=-1)  # (batch_size,)
        
        # Normalize to [0, 1]
        entropy = torch.sigmoid(variance)
        
        return entropy
    
    def forward(
        self,
        embeddings: torch.Tensor,
        pattern_scores: Optional[dict] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Route inputs to fast or deep branch
        
        Args:
            embeddings: (batch_size, seq_len, d_model)
            pattern_scores: Dict of pattern detector scores
            attention_mask: (batch_size, seq_len)
        
        Returns:
            route_decision: (batch_size,) bool - True = deep, False = fast
            route_logits: (batch_size, 2) - routing logits
            route_info: dict with routing statistics
        """
        batch_size, seq_len, _ = embeddings.size()
        
        # 1. Pool embeddings for routing decision
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size())
            sum_emb = (embeddings * mask_expanded).sum(1)
            sum_mask = mask_expanded.sum(1)
            pooled = sum_emb / sum_mask.clamp(min=1e-9)
        else:
            pooled = embeddings.mean(dim=1)  # (batch_size, d_model)
        
        # 2. Estimate complexity
        complexity_score = self.complexity_estimator(pooled).squeeze(-1)  # (batch_size,)
        
        # 3. Incorporate pattern confidence (if available)
        if pattern_scores is not None:
            # Average pattern detector scores
            pattern_avg = torch.stack([
                score.squeeze(-1) for score in pattern_scores.values()
            ], dim=0).mean(dim=0)  # (batch_size,)
            
            # High pattern scores = likely attack = more complex
            # Combine with complexity score
            combined_complexity = (
                complexity_score * (1 - self.pattern_confidence_weight) +
                pattern_avg * self.pattern_confidence_weight
            )
        else:
            combined_complexity = complexity_score
        
        # 4. Optional: entropy-based adjustment
        if self.use_entropy:
            entropy = self.compute_entropy(embeddings)
            combined_complexity = (combined_complexity + entropy) / 2.0
        
        # 5. Routing decision
        # True = deep branch (complex), False = fast branch (simple)
        route_decision = combined_complexity > self.complexity_threshold
        
        # 6. Create routing logits (for potential training signal)
        # Logits: [fast_score, deep_score]
        fast_score = 1.0 - combined_complexity
        deep_score = combined_complexity
        route_logits = torch.stack([fast_score, deep_score], dim=-1)  # (batch_size, 2)
        
        # 7. Routing statistics
        route_info = {
            'complexity_scores': combined_complexity,
            'fast_ratio': (~route_decision).float().mean().item(),
            'deep_ratio': route_decision.float().mean().item(),
            'threshold': self.complexity_threshold.item(),
        }
        
        return route_decision, route_logits, route_info
    
    def get_routing_loss(
        self,
        route_logits: torch.Tensor,
        labels: torch.Tensor,
        target_fast_ratio: float = 0.7,
    ) -> torch.Tensor:
        """
        Routing loss to encourage target fast/deep ratio
        
        Args:
            route_logits: (batch_size, 2)
            labels: (batch_size,) - ground truth labels
            target_fast_ratio: Target ratio for fast branch (0.7 = 70%)
        
        Returns:
            loss: scalar
        """
        batch_size = route_logits.size(0)
        
        # Soft routing distribution
        route_probs = F.softmax(route_logits, dim=-1)  # (batch_size, 2)
        
        # Actual fast ratio
        actual_fast_ratio = route_probs[:, 0].mean()
        
        # L2 loss to target ratio
        ratio_loss = (actual_fast_ratio - target_fast_ratio) ** 2
        
        # Optional: encourage benign → fast, attacks → deep
        # This is task-specific and may not always apply
        benign_mask = (labels == 0)
        if benign_mask.any():
            # Benign should prefer fast branch
            benign_fast_loss = -route_probs[benign_mask, 0].log().mean()
        else:
            benign_fast_loss = 0.0
        
        # Combined loss
        total_loss = ratio_loss + 0.1 * benign_fast_loss
        
        return total_loss

