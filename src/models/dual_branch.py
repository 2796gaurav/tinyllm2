"""
TinyGuardrail: Main Dual-Branch Architecture
Integrates all components into a unified model

Architecture:
- Threat-aware embeddings (token + char + pattern)
- Adaptive router (complexity-based)
- Fast branch (pattern + lightweight transformer)
- Deep branch (MoE reasoner)
- Fusion layer
- Bit-level response encoding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

from src.models.embeddings import ThreatAwareEmbedding
from src.models.fast_branch import FastPatternDetector
from src.models.deep_branch import DeepMoEReasoner
from src.models.router import AdaptiveRouter


@dataclass
class DualBranchConfig:
    """Configuration for TinyGuardrail"""
    
    # Vocabulary
    vocab_size: int = 8000
    char_vocab_size: int = 512
    max_position_embeddings: int = 512
    
    # Model dimensions
    d_model: int = 384
    num_labels: int = 4  # benign, direct_injection, jailbreak, obfuscation
    
    # Embedding config
    char_emb_dim: int = 64
    char_cnn_kernels: list = None
    char_cnn_channels: int = 128
    
    # Fast branch
    fast_num_layers: int = 4
    fast_num_heads: int = 4
    fast_intermediate_size: int = 768
    pattern_bank_size: int = 300
    num_handcrafted_patterns: int = 100
    
    # Deep branch
    deep_num_layers: int = 8
    deep_num_heads: int = 4
    deep_intermediate_size: int = 768
    num_experts: int = 8
    num_experts_per_token: int = 2
    expert_capacity_factor: float = 1.25
    
    # Router
    router_hidden_dim: int = 128
    router_threshold: float = 0.6
    use_learned_threshold: bool = True
    
    # Regularization
    dropout: float = 0.1
    attention_dropout: float = 0.1
    hidden_dropout: float = 0.1
    
    # Bit-level encoding
    use_bit_encoding: bool = True
    num_bits: int = 16
    
    def __post_init__(self):
        if self.char_cnn_kernels is None:
            self.char_cnn_kernels = [2, 3, 4, 5, 7]


class TinyGuardrail(nn.Module):
    """
    TinyGuardrail: Sub-100MB LLM Security System
    
    Target: 60-80M parameters, 66-80MB INT8, 86-90% PINT accuracy
    """
    
    def __init__(self, config: DualBranchConfig):
        super().__init__()
        
        self.config = config
        self.num_labels = config.num_labels
        
        # 1. Threat-Aware Embeddings
        self.embedding = ThreatAwareEmbedding(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            char_vocab_size=config.char_vocab_size,
            char_emb_dim=config.char_emb_dim,
            char_cnn_kernels=config.char_cnn_kernels,
            char_cnn_channels=config.char_cnn_channels,
            max_position_embeddings=config.max_position_embeddings,
            dropout=config.dropout,
        )
        
        # 2. Adaptive Router
        self.router = AdaptiveRouter(
            d_model=config.d_model,
            hidden_dim=config.router_hidden_dim,
            complexity_threshold=config.router_threshold,
            use_learned_threshold=config.use_learned_threshold,
            dropout=config.dropout,
        )
        
        # 3. Fast Branch
        self.fast_branch = FastPatternDetector(
            d_model=config.d_model,
            num_layers=config.fast_num_layers,
            num_heads=config.fast_num_heads,
            intermediate_size=config.fast_intermediate_size,
            pattern_bank_size=config.pattern_bank_size,
            num_handcrafted=config.num_handcrafted_patterns,
            num_labels=config.num_labels,
            dropout=config.dropout,
        )
        
        # 4. Deep Branch
        self.deep_branch = DeepMoEReasoner(
            d_model=config.d_model,
            num_layers=config.deep_num_layers,
            num_heads=config.deep_num_heads,
            intermediate_size=config.deep_intermediate_size,
            num_experts=config.num_experts,
            num_experts_per_token=config.num_experts_per_token,
            expert_capacity_factor=config.expert_capacity_factor,
            num_labels=config.num_labels,
            dropout=config.dropout,
        )
        
        # 5. Fusion Layer (combine fast + deep outputs when needed)
        self.fusion = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.num_labels),
        )
        
        # 6. Bit-level encoder (optional)
        if config.use_bit_encoding:
            self.bit_encoder = BitLevelEncoder(
                num_labels=config.num_labels,
                num_bits=config.num_bits,
            )
        else:
            self.bit_encoder = None
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        char_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        text: Optional[str] = None,
        return_dict: bool = True,
    ):
        """
        Forward pass
        
        Args:
            input_ids: (batch_size, seq_len) token IDs
            char_ids: (batch_size, seq_len, max_chars) character IDs
            position_ids: (batch_size, seq_len) position IDs
            attention_mask: (batch_size, seq_len) attention mask
            labels: (batch_size,) ground truth labels for training
            text: Optional raw text for pattern detection
            return_dict: Whether to return dict or tuple
        
        Returns:
            TinyGuardrailOutput or dict
        """
        # 1. Threat-aware embeddings
        embeddings, pattern_scores = self.embedding(
            input_ids=input_ids,
            char_ids=char_ids,
            position_ids=position_ids,
            text=text,
            attention_mask=attention_mask,
        )
        
        # 2. Adaptive routing
        route_decision, route_logits, route_info = self.router(
            embeddings=embeddings,
            pattern_scores=pattern_scores,
            attention_mask=attention_mask,
        )
        
        batch_size = input_ids.size(0)
        
        # 3. Dual-branch processing
        # Separate batches for fast and deep branches
        fast_mask = ~route_decision
        deep_mask = route_decision
        
        # ONNX/torch.export compatible: Always execute both branches and use masking
        # This avoids data-dependent control flow that torch.export cannot handle
        # Note: This is slightly less efficient during training but required for export
        
        # Fast branch - always execute with full batch
        fast_output = self.fast_branch(
            embeddings=embeddings,
            attention_mask=attention_mask,
            return_confidence=True,
        )
        
        # Deep branch - always execute with full batch
        deep_output = self.deep_branch(
            embeddings=embeddings,
            attention_mask=attention_mask,
            return_confidence=True,
        )
        
        # Initialize output tensors with correct dtype
        output_dtype = fast_output['logits'].dtype
        logits = torch.zeros(batch_size, self.num_labels, device=input_ids.device, dtype=output_dtype)
        fast_confidence = torch.zeros(batch_size, 1, device=input_ids.device, dtype=output_dtype)
        deep_confidence = torch.zeros(batch_size, 1, device=input_ids.device, dtype=output_dtype)
        
        # Use masking to select which branch results to use (ONNX-compatible)
        fast_mask_expanded = fast_mask.unsqueeze(-1).expand(-1, self.num_labels)
        deep_mask_expanded = deep_mask.unsqueeze(-1).expand(-1, self.num_labels)
        
        # Select logits based on routing decision
        logits = torch.where(fast_mask_expanded, fast_output['logits'], deep_output['logits'])
        
        # Select confidence based on routing decision
        fast_confidence = torch.where(
            fast_mask.unsqueeze(-1), 
            fast_output['confidence'], 
            torch.zeros_like(fast_output['confidence'])
        )
        deep_confidence = torch.where(
            deep_mask.unsqueeze(-1), 
            deep_output['confidence'], 
            torch.zeros_like(deep_output['confidence'])
        )
        
        # Aux loss from deep branch (weighted by deep branch ratio for ONNX compatibility)
        # During training, this will be properly computed by the deep branch
        aux_loss = deep_output['aux_loss']
        
        # 4. Combine confidences
        confidence = torch.where(
            fast_mask.unsqueeze(-1),
            fast_confidence,
            deep_confidence,
        )
        
        # 5. Bit-level encoding (if enabled)
        bit_response = None
        if self.bit_encoder is not None:
            bit_response = self.bit_encoder(logits, confidence)
        
        # 6. Compute loss (if labels provided)
        loss = None
        if labels is not None:
            # Main classification loss (focal loss for imbalanced data)
            loss = self.compute_loss(logits, labels, aux_loss, route_logits)
        
        if not return_dict:
            output = (logits, confidence, bit_response, route_info)
            return ((loss,) + output) if loss is not None else output
        
        return TinyGuardrailOutput(
            loss=loss,
            logits=logits,
            confidence=confidence,
            bit_response=bit_response,
            route_decision=route_decision,
            route_logits=route_logits,
            route_info=route_info,
            pattern_scores=pattern_scores,
            aux_loss=aux_loss,
        )
    
    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        aux_loss: torch.Tensor,
        route_logits: torch.Tensor,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        aux_loss_weight: float = 0.01,
        router_loss_weight: float = 0.5,  # Increased from 0.1 to better enforce 70/30 split
    ) -> torch.Tensor:
        """
        Compute multi-task loss
        
        Args:
            logits: (batch_size, num_labels)
            labels: (batch_size,)
            aux_loss: scalar - MoE load balancing loss
            route_logits: (batch_size, 2) - routing logits
        
        Returns:
            total_loss: scalar
        """
        # 1. Main classification loss (focal loss)
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        
        # Focal loss: FL(p_t) = -α(1-p_t)^γ log(p_t)
        pt = torch.exp(-ce_loss)  # p_t
        focal_loss = focal_alpha * (1 - pt) ** focal_gamma * ce_loss
        focal_loss = focal_loss.mean()
        
        # 2. Auxiliary loss (MoE load balancing)
        weighted_aux_loss = aux_loss_weight * aux_loss
        
        # 3. Router loss (encourage target fast/deep ratio)
        router_loss = self.router.get_routing_loss(
            route_logits, labels, target_fast_ratio=0.7
        )
        weighted_router_loss = router_loss_weight * router_loss
        
        # 4. Total loss
        total_loss = focal_loss + weighted_aux_loss + weighted_router_loss
        
        return total_loss
    
    def classify(
        self,
        text: str,
        tokenizer,
        max_length: int = 256,
        return_confidence: bool = True,
        return_bits: bool = True,
    ) -> 'GuardrailResult':
        """
        High-level classification API
        
        Args:
            text: Input text to classify
            tokenizer: Tokenizer instance
            max_length: Max sequence length
            return_confidence: Whether to return confidence
            return_bits: Whether to return bit-level encoding
        
        Returns:
            GuardrailResult with classification results
        """
        self.eval()
        
        with torch.no_grad():
            # Tokenize
            inputs = tokenizer(
                text,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
            )
            
            # Move to model device
            device = next(self.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Forward pass
            outputs = self.forward(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                text=text,
                return_dict=True,
            )
            
            # Get predictions
            logits = outputs.logits[0]  # (num_labels,)
            probs = F.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probs).item()
            confidence_score = probs[predicted_class].item()
            
            # Map class to label
            label_map = ['benign', 'direct_injection', 'jailbreak', 'obfuscation']
            threat_type = label_map[predicted_class]
            
            # Bit-level response
            bits = outputs.bit_response[0].item() if outputs.bit_response is not None else None
            
            return GuardrailResult(
                is_safe=(predicted_class == 0),
                threat_type=threat_type,
                confidence=confidence_score if return_confidence else None,
                bits=bits if return_bits else None,
                probabilities=probs.cpu().numpy(),
                route_decision='fast' if not outputs.route_decision[0] else 'deep',
            )
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Estimate sizes (bytes per parameter)
        # FP32: 4 bytes, FP16: 2 bytes, INT8: 1 byte, INT4: 0.5 bytes
        size_fp32_mb = (total_params * 4) / (1024**2)
        size_fp16_mb = (total_params * 2) / (1024**2)  # Half precision
        size_int8_mb = total_params / (1024**2)
        size_int4_mb = (total_params * 0.5) / (1024**2)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'embedding_params': sum(p.numel() for p in self.embedding.parameters()),
            'fast_branch_params': sum(p.numel() for p in self.fast_branch.parameters()),
            'deep_branch_params': sum(p.numel() for p in self.deep_branch.parameters()),
            'router_params': sum(p.numel() for p in self.router.parameters()),
            'size_fp32_mb': size_fp32_mb,
            'size_fp16_mb': size_fp16_mb,  # Added FP16 size
            'size_int8_mb': size_int8_mb,
            'size_int4_mb': size_int4_mb,
        }
    
    @classmethod
    def from_pretrained(cls, model_path: str):
        """Load pretrained model"""
        import os
        import json
        
        # Load config
        config_path = os.path.join(model_path, 'config.json')
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        config = DualBranchConfig(**config_dict)
        
        # Create model
        model = cls(config)
        
        # Load weights
        weights_path = os.path.join(model_path, 'pytorch_model.bin')
        state_dict = torch.load(weights_path, map_location='cpu')
        model.load_state_dict(state_dict)
        
        return model


class BitLevelEncoder(nn.Module):
    """
    Bit-level response encoder
    Encodes classification into 16-bit integer
    
    Format:
    - Bits 0-3: Attack type (16 categories)
    - Bits 4-7: Confidence level (16 levels)
    - Bits 8-11: Severity (16 levels)
    - Bits 12-15: Suggested action (16 actions)
    """
    
    def __init__(self, num_labels: int = 4, num_bits: int = 16):
        super().__init__()
        
        self.num_labels = num_labels
        self.num_bits = num_bits
        
        # Severity estimator (learned)
        self.severity_head = nn.Sequential(
            nn.Linear(num_labels, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.Softmax(dim=-1),
        )
    
    def forward(
        self,
        logits: torch.Tensor,
        confidence: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode to 16-bit integers
        
        Args:
            logits: (batch_size, num_labels)
            confidence: (batch_size, 1)
        
        Returns:
            bits: (batch_size,) - 16-bit integers
        """
        batch_size = logits.size(0)
        
        # 1. Attack type (bits 0-3)
        attack_type = torch.argmax(logits, dim=-1)  # (batch_size,)
        
        # 2. Confidence level (bits 4-7)
        confidence_level = (confidence.squeeze(-1) * 15).long().clamp(0, 15)
        
        # 3. Severity (bits 8-11)
        severity_probs = self.severity_head(logits)
        severity = torch.argmax(severity_probs, dim=-1)
        
        # 4. Suggested action (bits 12-15)
        # Simple mapping: benign=0 (allow), others=4 (block)
        action = torch.where(attack_type == 0, torch.zeros_like(attack_type), torch.full_like(attack_type, 4))
        
        # Pack into 16-bit integer
        bits = (
            attack_type |
            (confidence_level << 4) |
            (severity << 8) |
            (action << 12)
        )
        
        return bits


@dataclass
class TinyGuardrailOutput:
    """Output from TinyGuardrail model"""
    loss: Optional[torch.Tensor] = None
    logits: torch.Tensor = None
    confidence: torch.Tensor = None
    bit_response: Optional[torch.Tensor] = None
    route_decision: torch.Tensor = None
    route_logits: torch.Tensor = None
    route_info: Dict = None
    pattern_scores: Dict = None
    aux_loss: torch.Tensor = None


@dataclass
class GuardrailResult:
    """High-level classification result"""
    is_safe: bool
    threat_type: str
    confidence: Optional[float] = None
    bits: Optional[int] = None
    probabilities: Optional[any] = None
    route_decision: Optional[str] = None

