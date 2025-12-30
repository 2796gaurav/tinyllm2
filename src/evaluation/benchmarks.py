"""
Benchmark evaluation for TinyGuardrail
Supports PINT, JailbreakBench, NotInject, and custom adversarial datasets
"""

import time
import numpy as np
from typing import Dict, List, Optional, Callable
from tqdm import tqdm
import torch

try:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    # Fallback implementations
    def accuracy_score(y_true, y_pred):
        return sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true) if len(y_true) > 0 else 0.0
    
    def precision_score(y_true, y_pred, average='weighted'):
        # Simple implementation
        return accuracy_score(y_true, y_pred)
    
    def recall_score(y_true, y_pred, average='weighted'):
        return accuracy_score(y_true, y_pred)
    
    def f1_score(y_true, y_pred, average='weighted'):
        return accuracy_score(y_true, y_pred)


class GuardrailBenchmark:
    """
    Evaluate TinyGuardrail model on multiple benchmarks
    
    Supports:
    - PINT (Lakera AI benchmark)
    - JailbreakBench (NeurIPS 2024)
    - NotInject (False positive rate testing)
    - Custom adversarial datasets
    """
    
    def __init__(self, model, tokenizer=None):
        """
        Initialize benchmark evaluator
        
        Args:
            model: TinyGuardrail model instance
            tokenizer: Optional tokenizer (if model needs it for inference)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.benchmarks = {}
    
    def load_pint(self) -> List[Dict]:
        """
        Load PINT benchmark dataset
        
        Returns:
            List of samples with 'text' and 'label' keys
        """
        # Placeholder - in real implementation, load from actual PINT dataset
        # For now, return empty list to avoid errors
        return []
    
    def load_jailbreak(self) -> List[Dict]:
        """
        Load JailbreakBench dataset
        
        Returns:
            List of samples with 'text' and 'label' keys
        """
        # Placeholder - in real implementation, load from actual dataset
        return []
    
    def load_notinject(self) -> List[Dict]:
        """
        Load NotInject dataset (for FPR testing)
        
        Returns:
            List of samples with 'text' and 'label' keys
        """
        # Placeholder - in real implementation, load from actual dataset
        return []
    
    def load_custom(self) -> List[Dict]:
        """
        Load custom adversarial dataset
        
        Returns:
            List of samples with 'text' and 'label' keys
        """
        # Placeholder - in real implementation, load from actual dataset
        return []
    
    def run_all_benchmarks(self) -> Dict:
        """
        Run all benchmark evaluations
        
        Returns:
            Dictionary of results for each benchmark
        """
        results = {}
        
        # Load benchmarks
        benchmarks = {
            'pint': self.load_pint(),
            'jailbreakbench': self.load_jailbreak(),
            'notinject': self.load_notinject(),
            'custom_adversarial': self.load_custom(),
        }
        
        for name, dataset in benchmarks.items():
            if len(dataset) == 0:
                print(f"\n⚠ Skipping {name} (dataset not loaded)")
                continue
                
            print(f"\nEvaluating on {name}...")
            metrics = self.evaluate(dataset)
            results[name] = metrics
            
            print(f"  Accuracy: {metrics['accuracy']:.2f}%")
            print(f"  Precision: {metrics['precision']:.2f}%")
            print(f"  Recall: {metrics['recall']:.2f}%")
            print(f"  F1: {metrics['f1']:.2f}%")
            
            if name == 'notinject':
                print(f"  FPR: {metrics['fpr']:.2f}% ⭐")
            
            if 'latency_p95' in metrics:
                print(f"  Latency P95: {metrics['latency_p95']:.2f}ms")
        
        return results
    
    def evaluate(self, dataset: List[Dict]) -> Dict:
        """
        Evaluate model on a single dataset
        
        Args:
            dataset: List of samples with 'text' and 'label' keys
        
        Returns:
            Dictionary of metrics
        """
        predictions = []
        labels = []
        latencies = []
        
        self.model.eval()
        
        for sample in tqdm(dataset, desc="Evaluating"):
            text = sample.get('text', '')
            label = sample.get('label', 0)
            
            # Measure latency
            start = time.time()
            
            # Get prediction
            pred = self._predict(text)
            
            latency = (time.time() - start) * 1000  # ms
            
            predictions.append(pred)
            labels.append(label)
            latencies.append(latency)
        
        # Compute metrics
        metrics = {
            'accuracy': accuracy_score(labels, predictions) * 100,
            'precision': precision_score(labels, predictions, average='weighted') * 100,
            'recall': recall_score(labels, predictions, average='weighted') * 100,
            'f1': f1_score(labels, predictions, average='weighted') * 100,
            'fpr': self.compute_fpr(labels, predictions),
        }
        
        # Add latency metrics if available
        if latencies:
            metrics.update({
                'latency_p50': np.percentile(latencies, 50),
                'latency_p95': np.percentile(latencies, 95),
                'latency_p99': np.percentile(latencies, 99),
                'latency_mean': np.mean(latencies),
            })
        
        return metrics
    
    def _predict(self, text: str) -> int:
        """
        Get model prediction for a text
        
        Args:
            text: Input text string
        
        Returns:
            Predicted label (integer)
        """
        import torch
        
        # Check if model has a predict method
        if hasattr(self.model, 'predict'):
            return self.model.predict(text)
        
        # Check if model has a classify method
        if hasattr(self.model, 'classify'):
            result = self.model.classify(text)
            if isinstance(result, dict):
                return result.get('label', result.get('prediction', 0))
            return int(result)
        
        # Fallback: use forward pass
        if self.tokenizer is None:
            raise ValueError("Tokenizer required for model inference")
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=256,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        
        # Move to device
        device = next(self.model.parameters()).device
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        # Get character IDs if needed
        char_ids = None
        if hasattr(self.model, 'forward') and 'char_ids' in self.model.forward.__code__.co_varnames:
            # Generate char_ids (simplified)
            char_ids = self._text_to_char_ids(text, input_ids)
            char_ids = char_ids.to(device)
        
        # Forward pass
        with torch.no_grad():
            if char_ids is not None:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    char_ids=char_ids,
                    return_dict=True,
                )
            else:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
            
            # Get prediction
            if hasattr(outputs, 'logits'):
                pred = torch.argmax(outputs.logits, dim=-1).item()
            elif hasattr(outputs, 'prediction'):
                pred = outputs.prediction.item()
            else:
                pred = 0
        
        return int(pred)
    
    def _text_to_char_ids(self, text: str, token_ids: torch.Tensor, max_chars_per_token: int = 20) -> torch.Tensor:
        """
        Convert text to character IDs (simplified version)
        
        Args:
            text: Input text
            token_ids: Token IDs from tokenizer
            max_chars_per_token: Maximum characters per token
        
        Returns:
            Character IDs tensor
        """
        import torch
        
        # Simple character vocabulary
        char_vocab = {chr(i): i for i in range(512)}
        char_vocab['<PAD>'] = 0
        char_vocab['<UNK>'] = 1
        
        # Decode tokens
        if self.tokenizer:
            tokens = self.tokenizer.convert_ids_to_tokens(token_ids.squeeze(0).tolist())
        else:
            tokens = [str(tid) for tid in token_ids.squeeze(0).tolist()]
        
        char_ids_list = []
        for token in tokens:
            if token in ['[PAD]', '[CLS]', '[SEP]', '<pad>', '<cls>', '<sep>']:
                char_ids_list.append([0] * max_chars_per_token)
            else:
                clean_token = token.replace('##', '')
                token_char_ids = []
                for char in clean_token[:max_chars_per_token]:
                    char_id = char_vocab.get(char, 1)  # 1 = UNK
                    token_char_ids.append(char_id)
                
                # Pad to max_chars_per_token
                while len(token_char_ids) < max_chars_per_token:
                    token_char_ids.append(0)  # 0 = PAD
                
                char_ids_list.append(token_char_ids)
        
        # Ensure we have max_length tokens
        max_length = token_ids.size(1)
        while len(char_ids_list) < max_length:
            char_ids_list.append([0] * max_chars_per_token)
        
        return torch.tensor(char_ids_list[:max_length], dtype=torch.long)
    
    def compute_fpr(self, labels: List[int], predictions: List[int]) -> float:
        """
        Compute False Positive Rate (FPR)
        
        FPR = FP / (FP + TN)
        Where FP = false positives (predicted positive but actually negative)
        and TN = true negatives (predicted negative and actually negative)
        
        Args:
            labels: True labels (0 = benign, 1+ = attack)
            predictions: Predicted labels
        
        Returns:
            False positive rate as percentage
        """
        if len(labels) == 0:
            return 0.0
        
        # Treat 0 as benign (negative), 1+ as attack (positive)
        true_negatives = sum(1 for l, p in zip(labels, predictions) if l == 0 and p == 0)
        false_positives = sum(1 for l, p in zip(labels, predictions) if l == 0 and p > 0)
        
        total_negatives = true_negatives + false_positives
        
        if total_negatives == 0:
            return 0.0
        
        fpr = (false_positives / total_negatives) * 100
        return fpr
    
    def evaluate_custom(
        self,
        texts: List[str],
        labels: Optional[List[int]] = None,
        custom_metric_fn: Optional[Callable] = None,
    ) -> Dict:
        """
        Evaluate on custom dataset
        
        Args:
            texts: List of input texts
            labels: Optional true labels
            custom_metric_fn: Optional custom metric function
        
        Returns:
            Dictionary of metrics
        """
        predictions = []
        latencies = []
        
        self.model.eval()
        
        for text in tqdm(texts, desc="Evaluating"):
            start = time.time()
            pred = self._predict(text)
            latency = (time.time() - start) * 1000  # ms
            
            predictions.append(pred)
            latencies.append(latency)
        
        metrics = {
            'latency_p50': np.percentile(latencies, 50) if latencies else 0.0,
            'latency_p95': np.percentile(latencies, 95) if latencies else 0.0,
            'latency_p99': np.percentile(latencies, 99) if latencies else 0.0,
            'latency_mean': np.mean(latencies) if latencies else 0.0,
        }
        
        if labels is not None:
            metrics.update({
                'accuracy': accuracy_score(labels, predictions) * 100,
                'precision': precision_score(labels, predictions, average='weighted') * 100,
                'recall': recall_score(labels, predictions, average='weighted') * 100,
                'f1': f1_score(labels, predictions, average='weighted') * 100,
                'fpr': self.compute_fpr(labels, predictions),
            })
        
        if custom_metric_fn:
            metrics['custom'] = custom_metric_fn(labels, predictions)
        
        return metrics
