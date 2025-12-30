"""
Deep Branch: Mixture-of-Experts Reasoner
Handles complex 30% of traffic with specialized experts
Target: <15ms latency for novel attacks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MoELayer(nn.Module):
    """
    Mixture-of-Experts layer
    
    8 specialized experts with top-2 routing
    - Expert 1-2: Direct prompt injection
    - Expert 3-4: Jailbreak attempts
    - Expert 5-6: Obfuscation/encoding attacks
    - Expert 7-8: General/borderline cases
    """
    
    def __init__(
        self,
        d_model: int = 384,
        num_experts: int = 8,
        expert_capacity_factor: float = 1.25,
        num_experts_per_token: int = 2,
        intermediate_size: int = 768,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.expert_capacity_factor = expert_capacity_factor
        
        # Router (gate network)
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        
        # Expert networks (FFN)
        self.experts = nn.ModuleList([
            Expert(
                d_model=d_model,
                intermediate_size=intermediate_size,
                dropout=dropout,
            )
            for _ in range(num_experts)
        ])
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # For load balancing loss
        self.register_buffer('expert_counts', torch.zeros(num_experts))
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: (batch_size, seq_len, d_model)
            attention_mask: (batch_size, seq_len)
        
        Returns:
            output: (batch_size, seq_len, d_model)
            aux_loss: scalar - load balancing loss
        """
        batch_size, seq_len, d_model = hidden_states.size()
        
        # Flatten for expert routing
        hidden_states_flat = hidden_states.view(-1, d_model)  # (B*L, D)
        
        # CRITICAL: FORCE FP32 before quantized layer - unconditional conversion
        # Quantization observers REQUIRE FP32 inputs, not FP16
        hidden_states_flat = hidden_states_flat.float()
        
        # Compute router logits
        router_logits = self.gate(hidden_states_flat)  # (B*L, E)
        
        # Top-k routing
        routing_weights = F.softmax(router_logits, dim=-1)  # (B*L, E)
        routing_weights, selected_experts = torch.topk(
            routing_weights, 
            self.num_experts_per_token, 
            dim=-1
        )  # (B*L, k), (B*L, k)
        
        # Normalize routing weights (top-k sum to 1)
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        
        # Initialize output
        output = torch.zeros_like(hidden_states_flat)
        
        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find tokens routed to this expert
            expert_mask = (selected_experts == expert_idx).any(dim=-1)  # (B*L,)
            
            if expert_mask.any():
                # Get tokens for this expert
                expert_input = hidden_states_flat[expert_mask]  # (N_expert, D)
                
                # CRITICAL: FORCE FP32 before quantized layers in expert - unconditional conversion
                # Quantization observers REQUIRE FP32 inputs, not FP16
                expert_input = expert_input.float()
                
                # Forward through expert
                expert_output = self.experts[expert_idx](expert_input)  # (N_expert, D)
                
                # Get routing weights for this expert
                expert_weights = torch.zeros(hidden_states_flat.size(0), device=hidden_states.device)
                expert_positions = torch.where(expert_mask)[0]
                
                for i, pos in enumerate(expert_positions):
                    # Find weight for this expert in top-k
                    expert_in_topk = (selected_experts[pos] == expert_idx).nonzero(as_tuple=True)[0]
                    if len(expert_in_topk) > 0:
                        expert_weights[pos] = routing_weights[pos, expert_in_topk[0]]
                
                # Add weighted expert output
                output[expert_mask] += expert_output * expert_weights[expert_mask].unsqueeze(-1)
                
                # Update expert counts (for load balancing)
                self.expert_counts[expert_idx] += expert_mask.sum().item()
        
        # Reshape output
        output = output.view(batch_size, seq_len, d_model)
        
        # Apply mask if provided
        if attention_mask is not None:
            output = output * attention_mask.unsqueeze(-1)
        
        # Residual connection and normalization
        output = hidden_states + self.dropout(output)
        output = self.norm(output)
        
        # Compute load balancing loss (auxiliary loss)
        # Encourage equal usage of experts
        aux_loss = self.compute_load_balancing_loss(routing_weights, selected_experts)
        
        return output, aux_loss
    
    def compute_load_balancing_loss(
        self,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute load balancing loss to encourage equal expert usage
        
        Args:
            routing_weights: (B*L, k)
            selected_experts: (B*L, k)
        
        Returns:
            loss: scalar
        """
        num_tokens = routing_weights.size(0)
        
        # Compute expert usage frequency
        expert_usage = torch.zeros(self.num_experts, device=routing_weights.device)
        for i in range(self.num_experts):
            expert_usage[i] = (selected_experts == i).float().sum()
        
        # Normalize
        expert_usage = expert_usage / num_tokens
        
        # Target: uniform distribution
        target = torch.ones_like(expert_usage) / self.num_experts
        
        # MSE loss
        loss = F.mse_loss(expert_usage, target)
        
        return loss


class Expert(nn.Module):
    """
    Single expert network (FFN)
    """
    
    def __init__(
        self,
        d_model: int,
        intermediate_size: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, intermediate_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size, d_model),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)


class DeepMoEReasoner(nn.Module):
    """
    Deep Branch: MoE-based complex reasoning
    
    Architecture:
    - 8 transformer layers with MoE
    - 8 specialized experts per layer
    - Top-2 routing
    - Load balancing
    
    Target: <15ms latency for 30% of complex traffic
    """
    
    def __init__(
        self,
        d_model: int = 384,
        num_layers: int = 8,
        num_heads: int = 4,
        intermediate_size: int = 768,
        num_experts: int = 8,
        num_experts_per_token: int = 2,
        expert_capacity_factor: float = 1.25,
        num_labels: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_labels = num_labels
        
        # Transformer layers with MoE
        self.layers = nn.ModuleList([
            MoETransformerLayer(
                d_model=d_model,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                num_experts=num_experts,
                num_experts_per_token=num_experts_per_token,
                expert_capacity_factor=expert_capacity_factor,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        # Final normalization
        self.norm = nn.LayerNorm(d_model)
        
        # Pooler
        self.pooler = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
        )
        
        # Classification head
        self.classifier = nn.Linear(d_model, num_labels)
        
        # Confidence estimation
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )
    
    def forward(
        self,
        embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_confidence: bool = True,
    ) -> dict:
        """
        Args:
            embeddings: (batch_size, seq_len, d_model) from threat-aware embeddings
            attention_mask: (batch_size, seq_len)
            return_confidence: Whether to return confidence scores
        
        Returns:
            dict with:
                - logits: (batch_size, num_labels)
                - aux_loss: scalar - total load balancing loss
                - confidence: (batch_size, 1)
                - expert_usage: (num_layers, num_experts) - expert usage statistics
        """
        hidden_states = embeddings
        total_aux_loss = 0.0
        expert_usage = []
        
        # Forward through MoE transformer layers
        for layer in self.layers:
            hidden_states, aux_loss = layer(hidden_states, attention_mask)
            total_aux_loss += aux_loss
            
            # Track expert usage (optional, for analysis)
            # expert_usage.append(layer.moe.expert_counts.clone())
        
        # Final normalization
        hidden_states = self.norm(hidden_states)
        
        # Pooling
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
            sum_hidden = (hidden_states * mask_expanded).sum(1)
            sum_mask = mask_expanded.sum(1)
            pooled = sum_hidden / sum_mask.clamp(min=1e-9)
        else:
            pooled = hidden_states.mean(dim=1)
        
        # CRITICAL: FORCE FP32 before quantized layers - unconditional conversion
        # Quantization observers REQUIRE FP32 inputs, not FP16
        pooled = pooled.float()
        
        pooled = self.pooler(pooled)  # (B, D)
        
        # Classification
        logits = self.classifier(pooled)  # (B, num_labels)
        
        # Confidence
        if return_confidence:
            confidence = self.confidence_head(pooled)  # (B, 1)
        else:
            confidence = None
        
        return {
            'logits': logits,
            'aux_loss': total_aux_loss / len(self.layers),  # Average aux loss
            'confidence': confidence,
        }


class MoETransformerLayer(nn.Module):
    """
    Transformer layer with MoE FFN
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        intermediate_size: int,
        num_experts: int,
        num_experts_per_token: int,
        expert_capacity_factor: float,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Multi-head attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # MoE FFN (replaces standard FFN)
        self.moe = MoELayer(
            d_model=d_model,
            num_experts=num_experts,
            expert_capacity_factor=expert_capacity_factor,
            num_experts_per_token=num_experts_per_token,
            intermediate_size=intermediate_size,
            dropout=dropout,
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self-attention
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        
        # Convert attention_mask to key_padding_mask format
        # attention_mask: 1=valid token, 0=padding
        # key_padding_mask: True=mask out, False=attend to
        key_padding_mask = None
        if attention_mask is not None:
            # Convert to bool and invert: 1 (valid) -> False (attend), 0 (padding) -> True (mask)
            key_padding_mask = (attention_mask == 0).bool()
        
        attn_output, _ = self.self_attn(
            query=hidden_states,
            key=hidden_states,
            value=hidden_states,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        
        hidden_states = residual + self.dropout(attn_output)
        
        # MoE FFN
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states, aux_loss = self.moe(hidden_states, attention_mask)
        
        return hidden_states, aux_loss

