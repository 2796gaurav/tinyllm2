"""
Threat-Aware Embeddings for 2025 Attack Defense
Combines token, character-level, and pattern-based features
Critical for FlipAttack and character-level evasion detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from src.models.pattern_detectors import (
    FlipAttackDetector,
    HomoglyphDetector,
    EncryptionDetector,
    EncodingDetector,
    TypoglycemiaDetector,
    IndirectPIDetector,
    UnicodeNormalizer,
)


class ThreatAwareEmbedding(nn.Module):
    """
    Threat-Aware Embeddings combining multiple modalities
    
    Components:
    1. Token embeddings (standard)
    2. Character-level CNN (multi-scale n-grams)
    3. Pattern detectors (6 types for 2025 attacks)
    4. Unicode normalizer (preprocessing)
    
    Critical for defending against FlipAttack (98% bypass on current systems)
    """
    
    def __init__(
        self,
        vocab_size: int = 8000,
        d_model: int = 384,
        char_vocab_size: int = 512,
        char_emb_dim: int = 64,
        char_cnn_kernels: list = [2, 3, 4, 5, 7],
        char_cnn_channels: int = 128,
        max_position_embeddings: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.char_vocab_size = char_vocab_size
        
        # 1. Token embeddings (standard)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_position_embeddings, d_model)
        
        # 2. Character-level CNN (CRITICAL for FlipAttack)
        self.char_embedding = nn.Embedding(char_vocab_size, char_emb_dim)
        
        # Multi-scale character CNN (different n-gram sizes)
        self.char_convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=char_emb_dim,
                out_channels=char_cnn_channels,
                kernel_size=k,
                padding=k // 2
            )
            for k in char_cnn_kernels
        ])
        
        self.char_pool = nn.AdaptiveMaxPool1d(1)
        
        # Project char features to d_model
        char_feature_dim = char_cnn_channels * len(char_cnn_kernels)
        self.char_projection = nn.Linear(char_feature_dim, d_model)
        
        # 3. Unicode normalizer (preprocessing)
        self.unicode_normalizer = UnicodeNormalizer()
        
        # 4. Pattern detectors (6 types for 2025 attacks)
        self.pattern_detectors = nn.ModuleDict({
            'flipattack': FlipAttackDetector(),
            'homoglyph': HomoglyphDetector(),
            'encryption': EncryptionDetector(),
            'encoding': EncodingDetector(),
            'typoglycemia': TypoglycemiaDetector(),
            'indirect_pi': IndirectPIDetector(),
        })
        
        # Pattern features projection
        num_detectors = len(self.pattern_detectors)
        self.pattern_projection = nn.Linear(num_detectors, d_model)
        
        # 5. Fusion layer (combine all modalities)
        # Input: token_emb + char_features + pattern_features
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize embedding weights"""
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.char_embedding.weight, mean=0.0, std=0.02)
    
    def extract_char_features(self, char_ids: torch.Tensor) -> torch.Tensor:
        """
        Extract character-level features using multi-scale CNN
        
        Args:
            char_ids: (batch_size, seq_len, max_chars_per_token)
        
        Returns:
            char_features: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, max_chars = char_ids.size()
        
        # Flatten for CNN
        char_ids_flat = char_ids.view(batch_size * seq_len, max_chars)
        
        # Character embeddings
        char_emb = self.char_embedding(char_ids_flat)  # (B*L, C, D)
        # CRITICAL: FORCE FP32 immediately - unconditional conversion
        # Embedding outputs can be FP16 even when model is in FP32 mode
        char_emb = char_emb.float()
        char_emb = char_emb.permute(0, 2, 1)  # (B*L, D, C)
        
        # Multi-scale convolutions
        conv_outputs = []
        for conv in self.char_convs:
            conv_out = F.relu(conv(char_emb))  # (B*L, channels, C)
            # CRITICAL: FORCE FP32 immediately - unconditional conversion
            # Conv outputs can be FP16 even when model is in FP32 mode
            conv_out = conv_out.float()
            pooled = self.char_pool(conv_out).squeeze(-1)  # (B*L, channels)
            # CRITICAL: FORCE FP32 immediately - unconditional conversion
            # Pooling outputs can be FP16 even when model is in FP32 mode
            pooled = pooled.float()
            conv_outputs.append(pooled)
        
        # Concatenate all scales
        char_features_flat = torch.cat(conv_outputs, dim=-1)  # (B*L, channels * num_kernels)
        
        # CRITICAL: FORCE FP32 before quantized layer - unconditional conversion
        # Quantization observers REQUIRE FP32 inputs, not FP16
        # Use .float() unconditionally - it's safe (no-op if already FP32)
        # This ensures we NEVER pass FP16 to quantized layers
        char_features_flat = char_features_flat.float()
        
        # Project to d_model
        char_features_flat = self.char_projection(char_features_flat)  # (B*L, d_model)
        
        # Reshape back
        char_features = char_features_flat.view(batch_size, seq_len, self.d_model)
        
        # CRITICAL: FORCE FP32 return value - unconditional conversion
        char_features = char_features.float()
        
        return char_features
    
    def extract_pattern_features(
        self, 
        input_ids: torch.Tensor, 
        char_ids: Optional[torch.Tensor] = None,
        text: Optional[str] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Extract pattern detection features
        
        Args:
            input_ids: (batch_size, seq_len)
            char_ids: (batch_size, seq_len, max_chars_per_token)
            text: Optional raw text for string-based detection
        
        Returns:
            pattern_features: (batch_size, seq_len, d_model)
            pattern_scores: Dict of individual detector scores
        """
        batch_size, seq_len = input_ids.size()
        
        # Run all pattern detectors
        pattern_scores = {}
        detector_outputs = []
        
        for name, detector in self.pattern_detectors.items():
            score = detector(input_ids, char_ids, text)  # (batch_size, 1)
            # CRITICAL: FORCE FP32 immediately - unconditional conversion
            # Detector outputs can be FP16 even when model is in FP32 mode
            score = score.float()
            pattern_scores[name] = score
            detector_outputs.append(score)
        
        # Concatenate all detector scores
        pattern_tensor = torch.cat(detector_outputs, dim=-1)  # (batch_size, num_detectors)
        
        # CRITICAL: FORCE FP32 before quantized layer - unconditional conversion
        # Quantization observers REQUIRE FP32 inputs, not FP16
        pattern_tensor = pattern_tensor.float()
        
        # Project to d_model
        pattern_features = self.pattern_projection(pattern_tensor)  # (batch_size, d_model)
        
        # CRITICAL: FORCE FP32 after projection - unconditional conversion
        pattern_features = pattern_features.float()
        
        # Broadcast to all sequence positions
        pattern_features = pattern_features.unsqueeze(1).expand(-1, seq_len, -1)
        
        # CRITICAL: FORCE FP32 after broadcasting - unconditional conversion
        pattern_features = pattern_features.float()
        
        return pattern_features, pattern_scores
    
    def forward(
        self,
        input_ids: torch.Tensor,
        char_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        text: Optional[str] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass combining all embedding modalities
        
        Args:
            input_ids: (batch_size, seq_len) token IDs
            char_ids: (batch_size, seq_len, max_chars) character IDs
            position_ids: (batch_size, seq_len) position IDs
            text: Optional raw text for pattern detection
            attention_mask: (batch_size, seq_len) attention mask
        
        Returns:
            embeddings: (batch_size, seq_len, d_model) combined embeddings
            pattern_scores: Dict of pattern detector scores (for auxiliary loss)
        """
        batch_size, seq_len = input_ids.size()
        
        # 0. Unicode normalization (preprocessing)
        normalized_input_ids, normalized_char_ids = self.unicode_normalizer(input_ids, char_ids)
        
        # Use normalized versions
        if char_ids is not None:
            char_ids = normalized_char_ids
        
        # 1. Token embeddings
        token_emb = self.token_embedding(normalized_input_ids)  # (B, L, D)
        # CRITICAL: FORCE FP32 immediately - unconditional conversion
        # Embedding outputs can be FP16 even when model is in FP32 mode
        token_emb = token_emb.float()
        
        # Add position embeddings
        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        position_emb = self.position_embedding(position_ids)
        # CRITICAL: FORCE FP32 immediately - unconditional conversion
        # Embedding outputs can be FP16 even when model is in FP32 mode
        position_emb = position_emb.float()
        token_emb = token_emb + position_emb
        # CRITICAL: FORCE FP32 after addition - unconditional conversion
        token_emb = token_emb.float()
        
        # 2. Character-level features
        if char_ids is not None:
            char_features = self.extract_char_features(char_ids)
        else:
            # If no character IDs provided, use zero features
            # CRITICAL: FORCE FP32 even for zero tensors - unconditional conversion
            char_features = torch.zeros_like(token_emb).float()
        
        # CRITICAL: FORCE FP32 - unconditional conversion
        char_features = char_features.float()
        
        # 3. Pattern detection features
        pattern_features, pattern_scores = self.extract_pattern_features(
            normalized_input_ids, char_ids, text
        )
        
        # CRITICAL: FORCE FP32 before concatenation - unconditional conversion
        # Quantization observers REQUIRE FP32 inputs, not FP16
        token_emb = token_emb.float()
        char_features = char_features.float()
        pattern_features = pattern_features.float()
        
        # 4. Fusion (combine all modalities)
        # Concatenate: token + char + pattern
        combined = torch.cat([token_emb, char_features, pattern_features], dim=-1)
        
        # CRITICAL: FORCE FP32 before fusion - unconditional conversion
        combined = combined.float()
        
        # Fuse through MLP
        fused_embeddings = self.fusion(combined)  # (B, L, D)
        
        # CRITICAL: FORCE FP32 after fusion - unconditional conversion
        fused_embeddings = fused_embeddings.float()
        
        # Apply dropout
        fused_embeddings = self.dropout(fused_embeddings)
        
        # CRITICAL: FORCE FP32 after dropout - unconditional conversion
        fused_embeddings = fused_embeddings.float()
        
        # Apply attention mask if provided
        if attention_mask is not None:
            fused_embeddings = fused_embeddings * attention_mask.unsqueeze(-1)
            # CRITICAL: FORCE FP32 after mask multiplication - unconditional conversion
            fused_embeddings = fused_embeddings.float()
        
        # CRITICAL: FORCE FP32 final output - unconditional conversion
        fused_embeddings = fused_embeddings.float()
        
        return fused_embeddings, pattern_scores
    
    def get_output_dim(self) -> int:
        """Get output dimension"""
        return self.d_model


class PositionAwareEmbedding(nn.Module):
    """
    Alternative: Sinusoidal position embeddings (like in Transformer)
    More parameter-efficient than learned embeddings
    """
    
    def __init__(self, d_model: int, max_position: int = 512):
        super().__init__()
        
        # Create sinusoidal position embeddings
        pe = torch.zeros(max_position, d_model)
        position = torch.arange(0, max_position).unsqueeze(1).float()
        
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional embeddings"""
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


# Helper function to create character IDs from text
def text_to_char_ids(
    text: str,
    max_chars_per_token: int = 20,
    char_vocab: Dict[str, int] = None
) -> torch.Tensor:
    """
    Convert text to character ID tensor
    
    Args:
        text: Input text
        max_chars_per_token: Max characters per token
        char_vocab: Character to ID mapping
    
    Returns:
        char_ids: (num_tokens, max_chars_per_token)
    """
    if char_vocab is None:
        # Default: ASCII + extended
        char_vocab = {chr(i): i for i in range(512)}
    
    tokens = text.split()
    char_ids = []
    
    for token in tokens:
        token_char_ids = []
        for char in token[:max_chars_per_token]:
            char_id = char_vocab.get(char, 0)  # 0 = UNK
            token_char_ids.append(char_id)
        
        # Pad to max_chars_per_token
        while len(token_char_ids) < max_chars_per_token:
            token_char_ids.append(0)  # 0 = PAD
        
        char_ids.append(token_char_ids)
    
    return torch.tensor(char_ids, dtype=torch.long)

