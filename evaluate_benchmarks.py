"""
Comprehensive Benchmark Evaluation Script
Evaluates TinyGuardrail on all benchmarks: PINT, JailbreakBench, NotInject, etc.

For Research Paper Metrics
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from transformers import AutoTokenizer
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.dual_branch import TinyGuardrail, DualBranchConfig
from src.data.benchmark_loaders import BenchmarkDataLoader
from src.data.attack_generators import Attack2026Generator


class ComprehensiveBenchmarkEvaluator:
    """
    Comprehensive benchmark evaluation
    
    Benchmarks:
    1. PINT (Lakera AI) - Industry standard
    2. JailbreakBench (NeurIPS 2024) - Jailbreak detection
    3. NotInject - FPR testing (CRITICAL)
    4. FlipAttack - 2025 attack defense
    5. CodeChameleon - Encryption-based attacks
    6. Homoglyph - Character-level evasion
    """
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        """
        Initialize evaluator
        
        Args:
            model_path: Path to model checkpoint
            device: Device to run on
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load model
        print("\n📦 Loading model...")
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        model_config = checkpoint['model_config']
        self.model = TinyGuardrail(model_config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ Model loaded from {model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
        # Create character vocabulary
        self.char_vocab = {chr(i): i for i in range(512)}
        self.char_vocab['<PAD>'] = 0
        self.char_vocab['<UNK>'] = 1
    
    def text_to_char_ids(self, text: str, token_ids: torch.Tensor, max_chars_per_token: int = 20) -> torch.Tensor:
        """Convert text to character IDs"""
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids.squeeze(0).tolist())
        
        char_ids_list = []
        for token in tokens:
            if token in ['[PAD]', '[CLS]', '[SEP]']:
                char_ids_list.append([0] * max_chars_per_token)
            else:
                clean_token = token.replace('##', '')
                token_char_ids = []
                for char in clean_token[:max_chars_per_token]:
                    char_id = self.char_vocab.get(char, 1)
                    token_char_ids.append(char_id)
                
                while len(token_char_ids) < max_chars_per_token:
                    token_char_ids.append(0)
                
                char_ids_list.append(token_char_ids)
        
        while len(char_ids_list) < 256:
            char_ids_list.append([0] * max_chars_per_token)
        
        return torch.tensor(char_ids_list[:256], dtype=torch.long).unsqueeze(0)
    
    def predict(self, text: str) -> Dict:
        """
        Get prediction for a single text
        
        Returns:
            Dictionary with prediction, confidence, latency, routing
        """
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=256,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Character IDs
        char_ids = self.text_to_char_ids(text, encoding['input_ids'])
        char_ids = char_ids.to(self.device)
        
        # Measure latency
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                char_ids=char_ids,
                text=text,
                return_dict=True,
            )
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        latency_ms = (time.perf_counter() - start) * 1000
        
        # Get prediction
        logits = outputs.logits[0]
        probs = torch.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probs).item()
        confidence = probs[predicted_class].item()
        
        # Routing decision
        route_decision = 'deep' if outputs.route_decision[0].item() else 'fast'
        
        return {
            'prediction': predicted_class,
            'confidence': confidence,
            'probabilities': probs.cpu().numpy(),
            'latency_ms': latency_ms,
            'route': route_decision,
        }
    
    def evaluate_dataset(self, texts: List[str], labels: List[int], name: str) -> Dict:
        """
        Evaluate on a dataset
        
        Args:
            texts: List of input texts
            labels: List of ground truth labels
            name: Dataset name
        
        Returns:
            Dictionary of metrics
        """
        print(f"\n📊 Evaluating on {name} ({len(texts)} samples)...")
        
        predictions = []
        confidences = []
        latencies = []
        routing_stats = {'fast': 0, 'deep': 0}
        
        for text, label in tqdm(zip(texts, labels), total=len(texts), desc=f"Evaluating {name}"):
            result = self.predict(text)
            
            predictions.append(result['prediction'])
            confidences.append(result['confidence'])
            latencies.append(result['latency_ms'])
            
            if result['route'] == 'fast':
                routing_stats['fast'] += 1
            else:
                routing_stats['deep'] += 1
        
        # Compute metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted', zero_division=0
        )
        
        # Compute FPR (False Positive Rate) - CRITICAL METRIC
        true_negatives = sum(1 for l, p in zip(labels, predictions) if l == 0 and p == 0)
        false_positives = sum(1 for l, p in zip(labels, predictions) if l == 0 and p > 0)
        total_negatives = true_negatives + false_positives
        fpr = (false_positives / total_negatives * 100) if total_negatives > 0 else 0.0
        
        # Routing statistics
        total_routed = routing_stats['fast'] + routing_stats['deep']
        fast_ratio = routing_stats['fast'] / total_routed if total_routed > 0 else 0.0
        deep_ratio = routing_stats['deep'] / total_routed if total_routed > 0 else 0.0
        
        # Latency statistics
        latencies = np.array(latencies)
        
        metrics = {
            'dataset': name,
            'num_samples': len(texts),
            'accuracy': accuracy * 100,
            'precision': precision * 100,
            'recall': recall * 100,
            'f1': f1 * 100,
            'fpr': fpr,  # ⭐ Critical metric
            'latency_mean': latencies.mean(),
            'latency_p50': np.percentile(latencies, 50),
            'latency_p95': np.percentile(latencies, 95),
            'latency_p99': np.percentile(latencies, 99),
            'throughput_rps': 1000 / latencies.mean(),
            'fast_ratio': fast_ratio * 100,
            'deep_ratio': deep_ratio * 100,
        }
        
        # Print results
        print(f"\n📊 {name} Results:")
        print(f"   Accuracy:  {metrics['accuracy']:.2f}%")
        print(f"   Precision: {metrics['precision']:.2f}%")
        print(f"   Recall:    {metrics['recall']:.2f}%")
        print(f"   F1:        {metrics['f1']:.2f}%")
        print(f"   FPR:       {metrics['fpr']:.2f}% ⭐ (target: <10%)")
        print(f"\n⏱️  Latency:")
        print(f"   Mean: {metrics['latency_mean']:.2f}ms")
        print(f"   P50:  {metrics['latency_p50']:.2f}ms")
        print(f"   P95:  {metrics['latency_p95']:.2f}ms ⭐ (target: <20ms CPU)")
        print(f"   P99:  {metrics['latency_p99']:.2f}ms")
        print(f"\n🔀 Routing:")
        print(f"   Fast: {metrics['fast_ratio']:.1f}% (target: 70%)")
        print(f"   Deep: {metrics['deep_ratio']:.1f}% (target: 30%)")
        
        return metrics
    
    def evaluate_all_benchmarks(self, output_dir: str = "results/benchmarks"):
        """
        Evaluate on all benchmarks
        
        Args:
            output_dir: Directory to save results
        """
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE BENCHMARK EVALUATION")
        print("="*80)
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Load benchmark data
        benchmark_loader = BenchmarkDataLoader()
        benchmarks = benchmark_loader.load_all_benchmarks()
        
        # Evaluate each benchmark
        all_results = {}
        
        for benchmark_name, df in benchmarks.items():
            if len(df) == 0:
                print(f"\n⚠️  Skipping {benchmark_name} (no data)")
                continue
            
            metrics = self.evaluate_dataset(
                texts=df['text'].tolist(),
                labels=df['label'].tolist(),
                name=benchmark_name,
            )
            
            all_results[benchmark_name] = metrics
        
        # Generate 2025 attack test set
        print("\n📊 Evaluating on 2025 Attack Suite...")
        attack_gen = Attack2026Generator()
        
        # FlipAttack
        flipattack_data = attack_gen.generate_flipattack(
            base_prompts=attack_gen._create_base_malicious_prompts(),
            n_samples=1000,
        )
        flipattack_df = pd.DataFrame(flipattack_data)
        
        all_results['flipattack'] = self.evaluate_dataset(
            texts=flipattack_df['text'].tolist(),
            labels=flipattack_df['label'].tolist(),
            name='FlipAttack (2025)',
        )
        
        # CodeChameleon
        codechameleon_data = attack_gen.generate_codechameleon(
            base_prompts=attack_gen._create_base_malicious_prompts(),
            n_samples=1000,
        )
        codechameleon_df = pd.DataFrame(codechameleon_data)
        
        all_results['codechameleon'] = self.evaluate_dataset(
            texts=codechameleon_df['text'].tolist(),
            labels=codechameleon_df['label'].tolist(),
            name='CodeChameleon (2025)',
        )
        
        # Homoglyph
        homoglyph_data = attack_gen.generate_homoglyph(
            base_prompts=attack_gen._create_base_malicious_prompts(),
            n_samples=1000,
        )
        homoglyph_df = pd.DataFrame(homoglyph_data)
        
        all_results['homoglyph'] = self.evaluate_dataset(
            texts=homoglyph_df['text'].tolist(),
            labels=homoglyph_df['label'].tolist(),
            name='Homoglyph (2025)',
        )
        
        # Summary report
        print("\n" + "="*80)
        print("📊 BENCHMARK SUMMARY")
        print("="*80)
        
        summary_df = pd.DataFrame(all_results).T
        print(summary_df[['accuracy', 'f1', 'fpr', 'latency_p95', 'fast_ratio']].to_string())
        
        # Save results
        results_path = Path(output_dir) / "benchmark_results.json"
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        summary_df.to_csv(Path(output_dir) / "benchmark_summary.csv")
        
        print(f"\n✅ Results saved to {output_dir}")
        
        # Overall statistics
        print("\n" + "="*80)
        print("📊 OVERALL STATISTICS")
        print("="*80)
        
        avg_accuracy = summary_df['accuracy'].mean()
        avg_f1 = summary_df['f1'].mean()
        avg_fpr = summary_df['fpr'].mean()
        avg_latency_p95 = summary_df['latency_p95'].mean()
        avg_fast_ratio = summary_df['fast_ratio'].mean()
        
        print(f"Average Accuracy:  {avg_accuracy:.2f}%")
        print(f"Average F1:        {avg_f1:.2f}%")
        print(f"Average FPR:       {avg_fpr:.2f}% ⭐ (target: <10%)")
        print(f"Average P95 Latency: {avg_latency_p95:.2f}ms ⭐ (target: <20ms)")
        print(f"Average Fast Ratio:  {avg_fast_ratio:.1f}% (target: 70%)")
        
        # Check targets
        print("\n" + "="*80)
        print("🎯 TARGET ACHIEVEMENT")
        print("="*80)
        
        targets = {
            'Accuracy > 86%': avg_accuracy > 86,
            'FPR < 10%': avg_fpr < 10,
            'Latency P95 < 20ms': avg_latency_p95 < 20,
            'Routing 60-80% Fast': 60 <= avg_fast_ratio <= 80,
        }
        
        for target, achieved in targets.items():
            status = "✅" if achieved else "❌"
            print(f"{status} {target}")
        
        print("="*80 + "\n")
        
        return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='results/benchmarks', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    args = parser.parse_args()
    
    # Run evaluation
    evaluator = ComprehensiveBenchmarkEvaluator(
        model_path=args.model_path,
        device=args.device,
    )
    
    results = evaluator.evaluate_all_benchmarks(output_dir=args.output_dir)
    
    print("\n✅ Comprehensive benchmark evaluation complete!")


if __name__ == "__main__":
    main()


