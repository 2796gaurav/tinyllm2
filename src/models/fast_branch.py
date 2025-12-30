"""
Fast Branch: Pattern-Based Detector
Handles 70% of traffic with <5ms latency
Combines learned pattern bank with lightweight transformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class PatternBank(nn.Module):
    """
    Learnable pattern bank for fast threat detection
    Combines hand-crafted patterns with learned cluster centers
    """
    
    def __init__(
        self,
        d_model: int = 384,
        num_patterns: int = 300,
        num_handcrafted: int = 100,
        temperature: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_patterns = num_patterns
        self.num_handcrafted = num_handcrafted
        self.temperature = temperature
        
        # Learned pattern embeddings
        self.pattern_embeddings = nn.Parameter(
            torch.randn(num_patterns, d_model)
        )
        
        # Pattern type classification
        self.pattern_types = nn.Parameter(
            torch.randn(num_patterns, 4),  # 4 classes: benign, injection, jailbreak, obfuscation
            requires_grad=False  # Fixed during training
        )
        
        # Initialize hand-crafted patterns (first num_handcrafted)
        self._init_handcrafted_patterns()
    
    def _init_handcrafted_patterns(self):
        """Initialize hand-crafted threat patterns"""
        # In practice, load from predefined pattern database
        # Examples: SQL injection markers, common jailbreak templates, etc.
        # For now, random initialization (replace with actual patterns)
        with torch.no_grad():
            # First patterns are hand-crafted (benign)
            self.pattern_types[:self.num_handcrafted, 0] = 1.0
            
            # Injection patterns
            self.pattern_types[self.num_handcrafted:self.num_handcrafted+50, 1] = 1.0
            
            # Jailbreak patterns
            self.pattern_types[self.num_handcrafted+50:self.num_handcrafted+100, 2] = 1.0
            
            # Remaining are learned clusters
    
    def forward(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Match embeddings against pattern bank
        
        Args:
            embeddings: (batch_size, d_model) - pooled sequence embeddings
        
        Returns:
            logits: (batch_size, 4) - class logits
            similarities: (batch_size, num_patterns) - pattern similarities
        """
        batch_size = embeddings.size(0)
        
        # Compute similarity to all patterns (cosine similarity)
        # Normalize embeddings
        embeddings_norm = F.normalize(embeddings, p=2, dim=-1)  # (B, D)
        patterns_norm = F.normalize(self.pattern_embeddings, p=2, dim=-1)  # (P, D)
        
        # Similarity: (B, P)
        similarities = torch.matmul(embeddings_norm, patterns_norm.t()) / self.temperature
        
        # Soft matching weights (attention over patterns)
        attention_weights = F.softmax(similarities, dim=-1)  # (B, P)
        
        # Aggregate pattern types weighted by similarity
        logits = torch.matmul(attention_weights, self.pattern_types)  # (B, 4)
        
        return logits, similarities


class LightweightTransformer(nn.Module):
    """
    Lightweight transformer encoder (4 layers)
    For cases where pattern bank is uncertain
    """
    
    def __init__(
        self,
        d_model: int = 384,
        num_layers: int = 4,
        num_heads: int = 4,
        intermediate_size: int = 768,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch_size, seq_len, d_model)
            attention_mask: (batch_size, seq_len)
        
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        return self.norm(hidden_states)


class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer"""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        intermediate_size: int,
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
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, intermediate_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size, d_model),
            nn.Dropout(dropout),
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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
        
        # Feed-forward
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        
        # CRITICAL: FORCE FP32 before quantized layers in FFN - unconditional conversion
        # Quantization observers REQUIRE FP32 inputs, not FP16
        hidden_states = hidden_states.float()
        
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class FastPatternDetector(nn.Module):
    """
    Fast Branch: Pattern-based detection
    
    Two-stage detection:
    1. Pattern bank matching (fastest)
    2. Lightweight transformer (fallback for uncertainty)
    
    Target: <5ms latency, handles 70% of traffic
    """
    
    def __init__(
        self,
        d_model: int = 384,
        num_layers: int = 4,
        num_heads: int = 4,
        intermediate_size: int = 768,
        pattern_bank_size: int = 300,
        num_handcrafted: int = 100,
        num_labels: int = 4,
        dropout: float = 0.1,
        pattern_confidence_threshold: float = 0.8,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_labels = num_labels
        self.pattern_confidence_threshold = pattern_confidence_threshold
        
        # Pattern bank (primary detection)
        self.pattern_bank = PatternBank(
            d_model=d_model,
            num_patterns=pattern_bank_size,
            num_handcrafted=num_handcrafted,
        )
        
        # Lightweight transformer (fallback)
        self.transformer = LightweightTransformer(
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            dropout=dropout,
        )
        
        # Pooling (for pattern bank)
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
                - pattern_logits: (batch_size, num_labels) from pattern bank
                - pattern_similarities: (batch_size, num_patterns)
                - confidence: (batch_size, 1)
                - used_transformer: (batch_size,) bool, whether transformer was used
        """
        batch_size, seq_len, _ = embeddings.size()
        
        # 1. Pattern bank matching (fast path)
        # Pool sequence to single vector
        if attention_mask is not None:
            # Mean pooling with mask
            mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size())
            sum_embeddings = (embeddings * mask_expanded).sum(1)
            sum_mask = mask_expanded.sum(1)
            pooled = sum_embeddings / sum_mask.clamp(min=1e-9)
        else:
            pooled = embeddings.mean(dim=1)  # (B, D)
        
        # CRITICAL: FORCE FP32 before quantized layer - unconditional conversion
        # Quantization observers REQUIRE FP32 inputs, not FP16
        pooled = pooled.float()
        
        pooled = self.pooler(pooled)  # (B, D)
        
        # Match against pattern bank
        pattern_logits, pattern_similarities = self.pattern_bank(pooled)  # (B, 4), (B, P)
        
        # Estimate confidence in pattern matching
        pattern_confidence = F.softmax(pattern_logits, dim=-1).max(dim=-1).values  # (B,)
        
        # 2. Decide: use pattern bank or transformer
        use_transformer = pattern_confidence < self.pattern_confidence_threshold
        
        # ONNX/torch.export compatible: Always execute transformer and use masking
        # This avoids data-dependent control flow that torch.export cannot handle
        # Run lightweight transformer for all samples (will be masked if not needed)
        transformer_output = self.transformer(embeddings, attention_mask)
        
        # Pool transformer output
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand(transformer_output.size())
            sum_output = (transformer_output * mask_expanded).sum(1)
            sum_mask = mask_expanded.sum(1)
            transformer_pooled = sum_output / sum_mask.clamp(min=1e-9)
        else:
            transformer_pooled = transformer_output.mean(dim=1)
        
        # CRITICAL: FORCE FP32 before quantized layer - unconditional conversion
        # Quantization observers REQUIRE FP32 inputs, not FP16
        transformer_pooled = transformer_pooled.float()
        
        # Classify transformer output
        transformer_logits = self.classifier(transformer_pooled)
        
        # Combine pattern and transformer logits based on confidence
        # For high-confidence pattern matches, use pattern logits
        # For low-confidence, use transformer logits
        # ONNX-compatible: always compute both and use torch.where
        logits = torch.where(
            use_transformer.unsqueeze(-1).expand(-1, self.num_labels),
            transformer_logits,
            pattern_logits,
        )
        
        # Confidence estimation
        if return_confidence:
            # CRITICAL: FORCE FP32 before quantized layer - unconditional conversion
            # Quantization observers REQUIRE FP32 inputs, not FP16
            pooled = pooled.float()
            # Use pooled embedding for confidence
            confidence = self.confidence_head(pooled)  # (B, 1)
        else:
            confidence = None
        
        return {
            'logits': logits,
            'pattern_logits': pattern_logits,
            'pattern_similarities': pattern_similarities,
            'confidence': confidence,
            'used_transformer': use_transformer,
        }

