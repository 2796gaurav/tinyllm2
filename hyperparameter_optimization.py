"""
Hyperparameter Optimization for TinyGuardrail
Uses Optuna for efficient HPO

Optimizes:
- Learning rate
- Router threshold
- Router loss weight
- Batch size
- Focal loss parameters
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.dual_branch import TinyGuardrail, DualBranchConfig
from src.data.real_benchmark_loader import ProductionDataLoader, verify_hf_access
from src.data.attack_generators import Attack2026Generator, HardNegativeGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GuardrailDataset(Dataset):
    """Lightweight dataset for HPO"""
    
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.char_vocab = {chr(i): i for i in range(512)}
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        
        # Simplified char_ids (placeholder)
        char_ids = torch.zeros(self.max_length, 20, dtype=torch.long)
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'char_ids': char_ids,
            'labels': torch.tensor(label, dtype=torch.long),
        }


def objective(trial: optuna.Trial, train_data, val_data, device, num_epochs=2):
    """
    Optuna objective function
    
    Args:
        trial: Optuna trial object
        train_data: Training DataFrame
        val_data: Validation DataFrame
        device: Device to train on
        num_epochs: Number of epochs for each trial
    
    Returns:
        Validation F1 score
    """
    # Suggest hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-4, log=True)
    router_threshold = trial.suggest_float('router_threshold', 0.2, 0.5)
    router_loss_weight = trial.suggest_float('router_loss_weight', 0.1, 1.0)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    focal_gamma = trial.suggest_float('focal_gamma', 1.5, 3.0)
    focal_alpha = trial.suggest_float('focal_alpha', 0.1, 0.5)
    dropout = trial.suggest_float('dropout', 0.1, 0.3)
    
    logger.info(f"\nTrial {trial.number}:")
    logger.info(f"  LR: {learning_rate:.2e}")
    logger.info(f"  Router threshold: {router_threshold:.3f}")
    logger.info(f"  Router loss weight: {router_loss_weight:.3f}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Focal gamma: {focal_gamma:.2f}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    # Create datasets
    train_dataset = GuardrailDataset(
        texts=train_data['text'].tolist()[:10000],  # Subsample for speed
        labels=train_data['label'].tolist()[:10000],
        tokenizer=tokenizer,
    )
    
    val_dataset = GuardrailDataset(
        texts=val_data['text'].tolist()[:2000],
        labels=val_data['label'].tolist()[:2000],
        tokenizer=tokenizer,
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=2,
    )
    
    # Create model
    model_config = DualBranchConfig(
        vocab_size=len(tokenizer.vocab),
        d_model=384,
        num_labels=4,
        router_threshold=router_threshold,
        dropout=dropout,
        fast_num_layers=4,
        deep_num_layers=8,
        num_experts=8,
    )
    
    model = TinyGuardrail(model_config).to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
    )
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            char_ids = batch['char_ids'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                char_ids=char_ids,
                labels=labels,
                return_dict=True,
            )
            
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        logger.info(f"  Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.4f}")
        
        # Report intermediate value for pruning
        if epoch < num_epochs - 1:
            trial.report(avg_loss, epoch)
            
            # Handle pruning
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
    
    # Evaluate
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            char_ids = batch['char_ids'].to(device)
            labels = batch['labels']
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                char_ids=char_ids,
                return_dict=True,
            )
            
            preds = torch.argmax(outputs.logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Compute F1
    from sklearn.metrics import f1_score
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    logger.info(f"  Validation F1: {f1:.4f}")
    
    return f1


def run_hyperparameter_optimization(
    n_trials: int = 50,
    output_dir: str = "outputs/hpo",
    num_epochs_per_trial: int = 2,
):
    """
    Run hyperparameter optimization
    
    Args:
        n_trials: Number of trials to run
        output_dir: Directory to save results
        num_epochs_per_trial: Epochs per trial (keep low for speed)
    """
    logger.info("\n" + "="*80)
    logger.info("🔬 HYPERPARAMETER OPTIMIZATION")
    logger.info("="*80)
    
    # Verify HF access
    verify_hf_access()
    
    # Load data
    logger.info("\n📊 Loading real benchmark data...")
    data_loader = ProductionDataLoader()
    real_data = data_loader.load_all_real_data()
    
    # Add synthetic attacks for balance
    logger.info("\n📊 Generating synthetic 2025 attacks...")
    attack_gen = Attack2026Generator()
    attack_data = attack_gen.generate_all_attacks(n_total=20000)
    
    logger.info("\n📊 Generating hard negatives...")
    hard_neg_gen = HardNegativeGenerator()
    hard_neg_data = hard_neg_gen.generate_all_hard_negatives(n_total=10000)
    
    # Combine
    all_data = pd.concat([real_data, attack_data, hard_neg_data], ignore_index=True)
    all_data = all_data.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    logger.info(f"\n📊 Total dataset: {len(all_data):,} samples")
    
    # Split
    from sklearn.model_selection import train_test_split
    
    train_data, val_data = train_test_split(
        all_data, test_size=0.2, random_state=42, stratify=all_data['label']
    )
    
    logger.info(f"  Train: {len(train_data):,}, Val: {len(val_data):,}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"\n🖥️  Using device: {device}")
    
    # Create study
    study = optuna.create_study(
        direction='maximize',  # Maximize F1 score
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        study_name='tinyllm_guardrail_hpo',
    )
    
    # Optimize
    logger.info(f"\n🔬 Starting optimization ({n_trials} trials)...")
    
    study.optimize(
        lambda trial: objective(trial, train_data, val_data, device, num_epochs_per_trial),
        n_trials=n_trials,
        timeout=None,
        n_jobs=1,  # Sequential for GPU
    )
    
    # Results
    logger.info("\n" + "="*80)
    logger.info("🎯 OPTIMIZATION RESULTS")
    logger.info("="*80)
    
    logger.info(f"\nBest trial:")
    logger.info(f"  Value (F1): {study.best_trial.value:.4f}")
    
    logger.info(f"\nBest hyperparameters:")
    for key, value in study.best_params.items():
        logger.info(f"  {key}: {value}")
    
    # Save results
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save best params
    import json
    with open(Path(output_dir) / "best_hyperparameters.json", 'w') as f:
        json.dump(study.best_params, f, indent=2)
    
    # Save study
    study_df = study.trials_dataframe()
    study_df.to_csv(Path(output_dir) / "hpo_trials.csv", index=False)
    
    logger.info(f"\n✅ Results saved to {output_dir}")
    
    # Plot optimization history
    try:
        import plotly
        
        fig1 = optuna.visualization.plot_optimization_history(study)
        fig1.write_html(Path(output_dir) / "optimization_history.html")
        
        fig2 = optuna.visualization.plot_param_importances(study)
        fig2.write_html(Path(output_dir) / "param_importances.html")
        
        logger.info(f"📊 Visualizations saved to {output_dir}")
    except:
        logger.warning("⚠️  Could not generate visualizations (install plotly)")
    
    logger.info("\n" + "="*80)
    logger.info("✅ HYPERPARAMETER OPTIMIZATION COMPLETE")
    logger.info("="*80 + "\n")
    
    return study.best_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_trials', type=int, default=50, help='Number of trials')
    parser.add_argument('--output_dir', type=str, default='outputs/hpo', help='Output directory')
    parser.add_argument('--epochs_per_trial', type=int, default=2, help='Epochs per trial')
    args = parser.parse_args()
    
    # Run HPO
    best_params = run_hyperparameter_optimization(
        n_trials=args.n_trials,
        output_dir=args.output_dir,
        num_epochs_per_trial=args.epochs_per_trial,
    )
    
    print("\n🎉 Best hyperparameters found!")
    print(f"   Use these in your production training config")


if __name__ == "__main__":
    main()


