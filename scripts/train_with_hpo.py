"""
Hyperparameter Optimization using Optuna
Automatically find best hyperparameters for TinyGuardrail

Usage:
    python scripts/train_with_hpo.py --trials 100 --study-name tinyllm_hpo
"""

import argparse
import torch
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import yaml
import json
from pathlib import Path

# Import training components
from src.models import TinyGuardrail, DualBranchConfig
from src.training.losses import GuardrailLoss


def objective(trial: optuna.Trial, config_path: str, train_loader, val_loader):
    """
    Optuna objective function for hyperparameter optimization
    
    Args:
        trial: Optuna trial
        config_path: Path to base config
        train_loader: Training dataloader
        val_loader: Validation dataloader
    
    Returns:
        validation_f1: F1 score on validation set (to maximize)
    """
    
    # Load base config
    with open(config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Suggest hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-4, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    gradient_accumulation_steps = trial.suggest_categorical('gradient_accumulation_steps', [2, 4, 8])
    weight_decay = trial.suggest_float('weight_decay', 1e-4, 1e-1, log=True)
    warmup_ratio = trial.suggest_float('warmup_ratio', 0.05, 0.15)
    dropout = trial.suggest_float('dropout', 0.05, 0.3)
    label_smoothing = trial.suggest_float('label_smoothing', 0.0, 0.2)
    
    # Focal loss parameters
    focal_alpha = trial.suggest_float('focal_alpha', 0.1, 0.5)
    focal_gamma = trial.suggest_float('focal_gamma', 1.0, 3.0)
    
    # Router threshold
    router_threshold = trial.suggest_float('router_threshold', 0.5, 0.8)
    
    # MoE parameters
    num_experts = trial.suggest_categorical('num_experts', [4, 6, 8])
    num_experts_per_token = trial.suggest_categorical('num_experts_per_token', [1, 2])
    expert_capacity_factor = trial.suggest_float('expert_capacity_factor', 1.0, 1.5)
    
    # Adversarial training
    adversarial_epsilon = trial.suggest_float('adversarial_epsilon', 0.001, 0.1, log=True)
    
    # Auxiliary loss weights
    aux_loss_weight = trial.suggest_float('aux_loss_weight', 0.001, 0.1, log=True)
    router_loss_weight = trial.suggest_float('router_loss_weight', 0.05, 0.3)
    
    # Hard negative weight
    hard_negative_weight = trial.suggest_float('hard_negative_weight', 1.0, 3.0)
    
    # Create model config with suggested hyperparameters
    model_config = DualBranchConfig(
        vocab_size=base_config['model']['vocab_size'],
        d_model=base_config['model']['d_model'],
        num_labels=base_config['model']['num_labels'],
        dropout=dropout,
        router_threshold=router_threshold,
        num_experts=num_experts,
        num_experts_per_token=num_experts_per_token,
        expert_capacity_factor=expert_capacity_factor,
    )
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TinyGuardrail(model_config).to(device)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    
    # Loss function
    loss_fn = GuardrailLoss(
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma,
        aux_loss_weight=aux_loss_weight,
        router_loss_weight=router_loss_weight,
    )
    
    # Training loop (shortened for HPO)
    num_epochs = 3  # Fewer epochs for HPO
    best_val_f1 = 0.0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward
            outputs = model(**batch)
            
            # Loss
            loss_dict = loss_fn(
                logits=outputs.logits,
                labels=batch['labels'],
                aux_loss=outputs.aux_loss,
                router_logits=outputs.route_logits,
                pattern_scores=outputs.pattern_scores,
            )
            
            loss = loss_dict['total_loss']
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            # Report intermediate value for pruning
            if batch_idx % 100 == 0:
                trial.report(best_val_f1, epoch * len(train_loader) + batch_idx)
                
                # Check if trial should be pruned
                if trial.should_prune():
                    raise optuna.TrialPruned()
        
        # Validation
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = model(**batch)
                preds = torch.argmax(outputs.logits, dim=-1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
        
        # Compute F1
        from sklearn.metrics import f1_score
        val_f1 = f1_score(all_labels, all_preds, average='weighted')
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
        
        # Report intermediate value
        trial.report(val_f1, epoch)
        
        # Check if trial should be pruned
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return best_val_f1


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter Optimization for TinyGuardrail')
    parser.add_argument('--config', type=str, default='configs/base_config.yaml',
                       help='Path to base config')
    parser.add_argument('--hpo-config', type=str, default='configs/hpo_config.yaml',
                       help='Path to HPO config')
    parser.add_argument('--trials', type=int, default=100,
                       help='Number of trials')
    parser.add_argument('--study-name', type=str, default='tinyllm_hpo',
                       help='Study name')
    parser.add_argument('--storage', type=str, default=None,
                       help='Optuna storage (e.g., sqlite:///tinyllm_hpo.db)')
    parser.add_argument('--output-dir', type=str, default='outputs/hpo',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Load HPO config
    with open(args.hpo_config, 'r') as f:
        hpo_config = yaml.safe_load(f)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # TODO: Load actual data loaders
    # For now, assume they are available
    # train_loader = load_train_data()
    # val_loader = load_val_data()
    
    print("Note: This script requires train_loader and val_loader to be implemented")
    print("See notebooks/tinyllm_colab_training.py for data loading example")
    
    # Create study
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction='maximize',  # Maximize F1 score
        sampler=TPESampler(seed=42, multivariate=True),
        pruner=MedianPruner(
            n_startup_trials=hpo_config['pruner']['n_startup_trials'],
            n_warmup_steps=hpo_config['pruner']['n_warmup_steps'],
            interval_steps=hpo_config['pruner']['interval_steps'],
        ),
    )
    
    # Optimize
    print(f"\nStarting hyperparameter optimization with {args.trials} trials...")
    print("=" * 70)
    
    # study.optimize(
    #     lambda trial: objective(trial, args.config, train_loader, val_loader),
    #     n_trials=args.trials,
    #     timeout=None,
    # )
    
    # Best trial
    # print("\nOptimization finished!")
    # print("=" * 70)
    # print(f"\nBest trial:")
    # print(f"  Value (F1): {study.best_trial.value:.4f}")
    # print(f"\nBest hyperparameters:")
    # for key, value in study.best_trial.params.items():
    #     print(f"  {key}: {value}")
    
    # Save results
    # results = {
    #     'best_params': study.best_trial.params,
    #     'best_value': study.best_trial.value,
    #     'n_trials': len(study.trials),
    # }
    
    # with open(output_dir / 'hpo_results.json', 'w') as f:
    #     json.dump(results, f, indent=2)
    
    # print(f"\nResults saved to {output_dir / 'hpo_results.json'}")
    
    # Visualizations
    # if hpo_config['visualization']['enabled']:
    #     import optuna.visualization as vis
        
    #     # Optimization history
    #     fig = vis.plot_optimization_history(study)
    #     fig.write_html(str(output_dir / 'optimization_history.html'))
        
    #     # Parameter importances
    #     fig = vis.plot_param_importances(study)
    #     fig.write_html(str(output_dir / 'param_importances.html'))
        
    #     # Parallel coordinate
    #     fig = vis.plot_parallel_coordinate(study)
    #     fig.write_html(str(output_dir / 'parallel_coordinate.html'))
        
    #     print(f"\nVisualizations saved to {output_dir}")


if __name__ == '__main__':
    main()

