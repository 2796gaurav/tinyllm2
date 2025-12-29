"""
Pattern Detectors for 2025 Attack Methods
Defends against: FlipAttack, CodeChameleon, Homoglyph, Encoding, etc.
"""

import torch
import torch.nn as nn
import math
import re
import codecs
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class DetectionResult:
    """Pattern detection result"""
    score: float  # 0.0-1.0
    confidence: float
    details: Dict[str, any] = None


class BasePatternDetector(nn.Module):
    """Base class for pattern detectors"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, input_ids: torch.Tensor, char_ids: Optional[torch.Tensor] = None, 
                text: Optional[str] = None) -> torch.Tensor:
        """
        Returns detection score [0, 1]
        Args:
            input_ids: Token IDs (B, L)
            char_ids: Character IDs (B, L, C)
            text: Raw text (optional, for string-based detection)
        Returns:
            scores: Detection scores (B, 1)
        """
        raise NotImplementedError


class FlipAttackDetector(BasePatternDetector):
    """
    Detect FlipAttack variants (ICML 2025)
    - FCW: Flip Characters in Word
    - FCS: Flip Complete Sentence  
    - FWO: Flip Words Order
    98% bypass rate on current systems - CRITICAL defense
    """
    
    def __init__(self, threshold: float = 0.3):
        super().__init__()
        self.threshold = threshold
        
        # Common English patterns for FCW detection
        self.common_patterns = [
            'tion', 'ing', 'the', 'and', 'for', 'not', 'with', 'you',
            'that', 'this', 'have', 'from', 'they', 'been', 'more'
        ]
        
        # Suspicious trigger words (when flipped)
        self.trigger_words = [
            'ignore', 'disregard', 'bypass', 'override', 'system',
            'admin', 'root', 'execute', 'command', 'previous', 'instruction'
        ]
    
    def detect_fcw(self, text: str) -> float:
        """Detect character-level reversal in words"""
        words = text.split()
        if len(words) == 0:
            return 0.0
        
        suspicious_count = 0
        for word in words:
            if len(word) < 3:
                continue
            
            # Check if reversed version looks more "normal"
            reversed_word = word[::-1].lower()
            
            # Check against trigger words
            if reversed_word in self.trigger_words:
                suspicious_count += 2  # High weight
            
            # Check if reversed matches common patterns
            for pattern in self.common_patterns:
                if pattern in reversed_word:
                    suspicious_count += 1
                    break
        
        return min(1.0, suspicious_count / (len(words) * 0.5))
    
    def detect_fcs(self, text: str) -> float:
        """Detect complete sentence reversal"""
        # Check if entire text reversed contains trigger words
        reversed_text = text[::-1].lower()
        
        trigger_count = sum(1 for trigger in self.trigger_words if trigger in reversed_text)
        
        # Additional heuristic: reversed text has better word structure
        words = text.split()
        reversed_words = reversed_text.split()
        
        # Calculate "normality" score (simplified - in practice use language model perplexity)
        normal_score = sum(1 for w in words if len(w) > 2) / max(1, len(words))
        reversed_score = sum(1 for w in reversed_words if len(w) > 2) / max(1, len(reversed_words))
        
        structure_flip = max(0, reversed_score - normal_score)
        
        return min(1.0, (trigger_count * 0.3 + structure_flip))
    
    def detect_fwo(self, text: str) -> float:
        """Detect word order reversal"""
        words = text.lower().split()
        if len(words) < 3:
            return 0.0
        
        # Reverse word order
        reversed_words = words[::-1]
        reversed_text = ' '.join(reversed_words)
        
        # Check if reversed version contains more trigger words in sequence
        trigger_sequences = [
            'ignore all previous', 'disregard previous instructions',
            'override system', 'bypass security', 'execute command'
        ]
        
        score = 0.0
        for sequence in trigger_sequences:
            if sequence in reversed_text:
                score += 0.5
        
        # Check bigram reversal patterns
        # "instructions previous" vs "previous instructions"
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i+1]}"
            reversed_bigram = f"{words[i+1]} {words[i]}"
            
            # If reversed bigram is more common English pattern
            if reversed_bigram in ' '.join(self.common_patterns):
                score += 0.1
        
        return min(1.0, score)
    
    def forward(self, input_ids: torch.Tensor, char_ids: Optional[torch.Tensor] = None,
                text: Optional[str] = None) -> torch.Tensor:
        if text is None:
            # Decode from input_ids (simplified - implement proper decoding)
            text = self._decode(input_ids)
        
        batch_size = input_ids.size(0)
        scores = []
        
        for i in range(batch_size):
            sample_text = text if isinstance(text, str) else text[i]
            
            # Compute all three variant scores
            fcw_score = self.detect_fcw(sample_text)
            fcs_score = self.detect_fcs(sample_text)
            fwo_score = self.detect_fwo(sample_text)
            
            # Max score (most suspicious variant)
            max_score = max(fcw_score, fcs_score, fwo_score)
            scores.append(max_score)
        
        return torch.tensor(scores, device=input_ids.device, dtype=torch.float32).unsqueeze(-1)
    
    def _decode(self, input_ids: torch.Tensor) -> str:
        """Simplified decode - implement proper tokenizer decode in practice"""
        # This is a placeholder - use actual tokenizer in real implementation
        return ""


class HomoglyphDetector(BasePatternDetector):
    """
    Detect homoglyph substitution attacks
    Character injection: 100% bypass on Azure Prompt Shield
    """
    
    def __init__(self):
        super().__init__()
        
        # Comprehensive homoglyph map (Cyrillic, Extended Latin, Greek, etc.)
        self.homoglyph_map = {
            # Cyrillic
            'а': 'a', 'е': 'e', 'о': 'o', 'і': 'i', 'с': 'c', 'р': 'p',
            'х': 'x', 'у': 'y', 'В': 'B', 'Н': 'H', 'К': 'K', 'М': 'M',
            'Р': 'P', 'Т': 'T', 'Х': 'X',
            
            # Extended Latin
            'ạ': 'a', 'ả': 'a', 'ã': 'a', 'à': 'a', 'á': 'a',
            'ė': 'e', 'ē': 'e', 'ė': 'e', 'è': 'e', 'é': 'e',
            'ọ': 'o', 'ỏ': 'o', 'õ': 'o', 'ò': 'o', 'ó': 'o',
            'ï': 'i', 'ī': 'i', 'ì': 'i', 'í': 'i',
            
            # Greek
            'α': 'a', 'β': 'b', 'γ': 'y', 'δ': 'd', 'ε': 'e',
            'ζ': 'z', 'η': 'n', 'θ': 'th', 'ι': 'i', 'κ': 'k',
            
            # Mathematical
            '𝐚': 'a', '𝐛': 'b', '𝐜': 'c', '𝐝': 'd', '𝐞': 'e',
            '𝑎': 'a', '𝑏': 'b', '𝑐': 'c', '𝑑': 'd', '𝑒': 'e',
        }
        
        # Zero-width and invisible characters
        self.invisible_chars = [
            '\u200b',  # Zero-width space
            '\u200c',  # Zero-width non-joiner
            '\u200d',  # Zero-width joiner
            '\u2060',  # Word joiner
            '\ufeff',  # Zero-width no-break space
            '\u180e',  # Mongolian vowel separator
        ]
    
    def forward(self, input_ids: torch.Tensor, char_ids: Optional[torch.Tensor] = None,
                text: Optional[str] = None) -> torch.Tensor:
        if text is None:
            text = self._decode(input_ids)
        
        batch_size = input_ids.size(0)
        scores = []
        
        for i in range(batch_size):
            sample_text = text if isinstance(text, str) else text[i]
            
            # Count homoglyph substitutions
            homoglyph_count = sum(1 for char in sample_text if char in self.homoglyph_map)
            
            # Count invisible characters
            invisible_count = sum(1 for char in sample_text if char in self.invisible_chars)
            
            # Normalize by text length
            text_len = max(1, len(sample_text))
            homoglyph_ratio = homoglyph_count / text_len
            invisible_ratio = invisible_count / text_len
            
            # Combined score (invisible chars are more suspicious)
            score = min(1.0, homoglyph_ratio * 2.0 + invisible_ratio * 5.0)
            scores.append(score)
        
        return torch.tensor(scores, device=input_ids.device, dtype=torch.float32).unsqueeze(-1)
    
    def _decode(self, input_ids: torch.Tensor) -> str:
        return ""


class EncryptionDetector(BasePatternDetector):
    """
    Detect CodeChameleon-style encryption attacks
    Binary tree encoding, ROT13, custom ciphers
    """
    
    def __init__(self):
        super().__init__()
        
        self.cipher_keywords = [
            'decrypt', 'decode', 'decipher', 'rot13', 'rot', 'cipher',
            'binary tree', 'encoding scheme', 'mapping', 'translate',
            'base64', 'hex', 'encrypt', 'encoded', 'encrypted'
        ]
    
    def calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy (high entropy = potentially encrypted)"""
        if not text:
            return 0.0
        
        # Character frequency
        char_freq = {}
        for char in text:
            char_freq[char] = char_freq.get(char, 0) + 1
        
        # Shannon entropy
        entropy = 0
        text_len = len(text)
        for freq in char_freq.values():
            p = freq / text_len
            if p > 0:
                entropy -= p * math.log2(p)
        
        # Normalize (max entropy for 256 chars is 8)
        return entropy / 8.0
    
    def forward(self, input_ids: torch.Tensor, char_ids: Optional[torch.Tensor] = None,
                text: Optional[str] = None) -> torch.Tensor:
        if text is None:
            text = self._decode(input_ids)
        
        batch_size = input_ids.size(0)
        scores = []
        
        for i in range(batch_size):
            sample_text = text if isinstance(text, str) else text[i]
            sample_text_lower = sample_text.lower()
            
            # Keyword detection
            keyword_count = sum(1 for kw in self.cipher_keywords if kw in sample_text_lower)
            keyword_score = min(1.0, keyword_count / 3.0)
            
            # Entropy analysis
            entropy_score = self.calculate_entropy(sample_text)
            
            # Combined score (keywords + entropy)
            score = (keyword_score * 0.6 + entropy_score * 0.4)
            scores.append(score)
        
        return torch.tensor(scores, device=input_ids.device, dtype=torch.float32).unsqueeze(-1)
    
    def _decode(self, input_ids: torch.Tensor) -> str:
        return ""


class EncodingDetector(BasePatternDetector):
    """
    Detect encoding-based attacks (Base64, Hex, URL encoding)
    """
    
    def __init__(self):
        super().__init__()
        
        # Regex patterns for common encodings
        self.base64_pattern = re.compile(r'^[A-Za-z0-9+/]{20,}={0,2}$')
        self.hex_pattern = re.compile(r'^[0-9a-fA-F]{20,}$')
        self.url_encoding_pattern = re.compile(r'(%[0-9a-fA-F]{2}){5,}')
    
    def forward(self, input_ids: torch.Tensor, char_ids: Optional[torch.Tensor] = None,
                text: Optional[str] = None) -> torch.Tensor:
        if text is None:
            text = self._decode(input_ids)
        
        batch_size = input_ids.size(0)
        scores = []
        
        for i in range(batch_size):
            sample_text = text if isinstance(text, str) else text[i]
            
            score = 0.0
            
            # Check for Base64
            tokens = sample_text.split()
            for token in tokens:
                if len(token) >= 20 and self.base64_pattern.match(token):
                    score += 0.4
            
            # Check for Hex
            for token in tokens:
                if len(token) >= 20 and self.hex_pattern.match(token):
                    score += 0.4
            
            # Check for URL encoding
            if self.url_encoding_pattern.search(sample_text):
                score += 0.3
            
            scores.append(min(1.0, score))
        
        return torch.tensor(scores, device=input_ids.device, dtype=torch.float32).unsqueeze(-1)
    
    def _decode(self, input_ids: torch.Tensor) -> str:
        return ""


class TypoglycemiaDetector(BasePatternDetector):
    """
    Detect typoglycemia attacks (scrambled middle letters)
    Example: "ignroe" for "ignore"
    """
    
    def __init__(self):
        super().__init__()
        
        # Common English words (simplified - use full dictionary in practice)
        self.common_words = {
            'ignore', 'disregard', 'bypass', 'override', 'system',
            'admin', 'execute', 'command', 'previous', 'instruction',
            'instructions', 'security', 'password', 'access', 'root'
        }
    
    def has_same_boundaries(self, word1: str, word2: str) -> bool:
        """Check if words have same first and last letters"""
        if len(word1) < 3 or len(word2) < 3:
            return False
        return word1[0] == word2[0] and word1[-1] == word2[-1] and len(word1) == len(word2)
    
    def forward(self, input_ids: torch.Tensor, char_ids: Optional[torch.Tensor] = None,
                text: Optional[str] = None) -> torch.Tensor:
        if text is None:
            text = self._decode(input_ids)
        
        batch_size = input_ids.size(0)
        scores = []
        
        for i in range(batch_size):
            sample_text = text if isinstance(text, str) else text[i]
            words = sample_text.lower().split()
            
            scrambled_count = 0
            for word in words:
                if len(word) < 4:
                    continue
                
                # Check if word might be scrambled version of a common word
                for common_word in self.common_words:
                    if self.has_same_boundaries(word, common_word):
                        # Same first/last letters - might be scrambled
                        if word != common_word:
                            scrambled_count += 1
                            break
            
            score = min(1.0, scrambled_count / max(1, len(words) * 0.3))
            scores.append(score)
        
        return torch.tensor(scores, device=input_ids.device, dtype=torch.float32).unsqueeze(-1)
    
    def _decode(self, input_ids: torch.Tensor) -> str:
        return ""


class IndirectPIDetector(BasePatternDetector):
    """
    Detect indirect prompt injection (OWASP #1 threat)
    Malicious instructions embedded in external content
    """
    
    def __init__(self):
        super().__init__()
        
        # Patterns indicating instruction injection
        self.injection_patterns = [
            r'when you (?:read|see|process)',
            r'if you (?:encounter|find|see)',
            r'(?:tell|say|respond|reply) to (?:the user|them)',
            r'(?:ignore|disregard) (?:the|any) previous',
            r'your new (?:instructions|task|role)',
            r'from now on',
            r'you (?:must|should|will) (?:always|now)',
        ]
        
        self.compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.injection_patterns]
    
    def forward(self, input_ids: torch.Tensor, char_ids: Optional[torch.Tensor] = None,
                text: Optional[str] = None) -> torch.Tensor:
        if text is None:
            text = self._decode(input_ids)
        
        batch_size = input_ids.size(0)
        scores = []
        
        for i in range(batch_size):
            sample_text = text if isinstance(text, str) else text[i]
            
            # Count pattern matches
            match_count = sum(1 for pattern in self.compiled_patterns 
                            if pattern.search(sample_text))
            
            # Normalize score
            score = min(1.0, match_count / 3.0)
            scores.append(score)
        
        return torch.tensor(scores, device=input_ids.device, dtype=torch.float32).unsqueeze(-1)
    
    def _decode(self, input_ids: torch.Tensor) -> str:
        return ""


class UnicodeNormalizer(nn.Module):
    """
    Normalize Unicode text to detect homoglyph attacks
    Critical preprocessing step for character-level defenses
    """
    
    def __init__(self):
        super().__init__()
        
        # Homoglyph normalization map (extended in practice)
        self.normalization_map = {
            # Cyrillic to Latin
            'а': 'a', 'е': 'e', 'о': 'o', 'і': 'i', 'с': 'c', 'р': 'p',
            
            # Extended Latin to basic
            'ạ': 'a', 'ả': 'a', 'ã': 'a', 'à': 'a', 'á': 'a',
            'ė': 'e', 'ē': 'e', 'è': 'e', 'é': 'e',
            'ọ': 'o', 'ỏ': 'o', 'õ': 'o', 'ò': 'o', 'ó': 'o',
        }
        
        # Invisible characters to strip
        self.invisible_chars = [
            '\u200b', '\u200c', '\u200d', '\u2060', '\ufeff', '\u180e'
        ]
    
    def normalize_text(self, text: str) -> str:
        """Normalize text by removing homoglyphs and invisible chars"""
        # Remove invisible characters
        for char in self.invisible_chars:
            text = text.replace(char, '')
        
        # Replace homoglyphs with standard chars
        normalized = []
        for char in text:
            normalized.append(self.normalization_map.get(char, char))
        
        return ''.join(normalized)
    
    def forward(self, input_ids: torch.Tensor, char_ids: Optional[torch.Tensor] = None):
        """
        Normalize both token and character representations
        Returns: (normalized_input_ids, normalized_char_ids)
        """
        # In practice, implement proper normalization at character level
        # This is a placeholder for the architecture
        return input_ids, char_ids

