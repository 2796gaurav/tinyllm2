"""
SageMaker Training Entry Point
Handles different training modes: dry_run, hpo, train, evaluate

This script is called by SageMaker and can run in different modes based on hyperparameters.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Import training modules
from dry_run import dry_run
from hyperparameter_optimization import run_hyperparameter_optimization
from train_production_final import main as train_main
from evaluate_benchmarks import ComprehensiveBenchmarkEvaluator

# Import S3 utilities
from sagemaker_utils import (
    download_from_s3,
    upload_to_s3,
    setup_s3_paths,
    sync_directory_to_s3,
    sync_directory_from_s3,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_dry_run(s3_base_path: str, local_output_dir: str = "/opt/ml/output/data"):
    """Run dry run and upload results to S3"""
    logger.info("="*80)
    logger.info("STEP 1: DRY RUN")
    logger.info("="*80)
    
    # Run dry run
    success = dry_run()
    
    if not success:
        raise RuntimeError("Dry run failed!")
    
    # Upload results to S3
    dry_run_output = Path("outputs/dry_run")
    if dry_run_output.exists():
        s3_dry_run_path = f"{s3_base_path}/dry_run"
        logger.info(f"Uploading dry run results to {s3_dry_run_path}...")
        sync_directory_to_s3(str(dry_run_output), s3_dry_run_path)
        logger.info("✅ Dry run results uploaded to S3")
    
    return success


def run_hpo(s3_base_path: str, n_trials: int = 50, epochs_per_trial: int = 2):
    """Run hyperparameter optimization and upload results to S3"""
    logger.info("="*80)
    logger.info("STEP 2: HYPERPARAMETER OPTIMIZATION")
    logger.info("="*80)
    
    local_hpo_dir = "outputs/hpo"
    Path(local_hpo_dir).mkdir(parents=True, exist_ok=True)
    
    # Run HPO
    best_params = run_hyperparameter_optimization(
        n_trials=n_trials,
        output_dir=local_hpo_dir,
        num_epochs_per_trial=epochs_per_trial,
    )
    
    # Upload HPO results to S3
    s3_hpo_path = f"{s3_base_path}/hpo"
    logger.info(f"Uploading HPO results to {s3_hpo_path}...")
    sync_directory_to_s3(local_hpo_dir, s3_hpo_path)
    logger.info("✅ HPO results uploaded to S3")
    
    # Save best params for next step
    best_params_path = Path(local_hpo_dir) / "best_hyperparameters.json"
    if best_params_path.exists():
        s3_best_params = f"{s3_hpo_path}/best_hyperparameters.json"
        upload_to_s3(str(best_params_path), s3_best_params)
        logger.info(f"✅ Best hyperparameters saved to {s3_best_params}")
    
    return best_params


def run_training(
    s3_base_path: str,
    hpo_params_path: str = None,
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = None,
    use_wandb: bool = False,
):
    """Run full production training with optional HPO parameters"""
    logger.info("="*80)
    logger.info("STEP 3: PRODUCTION TRAINING")
    logger.info("="*80)
    
    # Download HPO best params if provided
    if hpo_params_path:
        logger.info(f"Loading best hyperparameters from {hpo_params_path}...")
        local_hpo_params = "/tmp/best_hyperparameters.json"
        download_from_s3(hpo_params_path, local_hpo_params)
        
        with open(local_hpo_params, 'r') as f:
            best_params = json.load(f)
        
        logger.info(f"Using HPO parameters: {best_params}")
        
        # Override with HPO params
        if learning_rate is None and 'learning_rate' in best_params:
            learning_rate = best_params['learning_rate']
        if batch_size == 16 and 'batch_size' in best_params:
            batch_size = best_params['batch_size']
    
    # Set up training arguments
    output_dir = "outputs/production_final"
    checkpoint_dir = "checkpoints/production_final"
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Import training functions directly
    from train_production_final import (
        load_production_dataset, GuardrailDataset, 
        ProductionConfig, DualBranchConfig, TinyGuardrail,
        train_epoch, evaluate
    )
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    
    # Load config
    config = ProductionConfig(
        num_epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate or 5e-5,
    )
    
    # Load data
    train_df, val_df, test_df = load_production_dataset(config)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    config.vocab_size = len(tokenizer.vocab)
    
    # Create datasets
    train_dataset = GuardrailDataset(
        texts=train_df['text'].tolist(),
        labels=train_df['label'].tolist(),
        tokenizer=tokenizer,
    )
    val_dataset = GuardrailDataset(
        texts=val_df['text'].tolist(),
        labels=val_df['label'].tolist(),
        tokenizer=tokenizer,
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size * 2,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_config = DualBranchConfig(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        num_labels=config.num_labels,
        router_threshold=config.router_threshold,
    )
    model = TinyGuardrail(model_config).to(device)
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    num_training_steps = len(train_loader) * config.num_epochs // config.gradient_accumulation_steps
    num_warmup_steps = int(num_training_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    
    # Mixed precision
    scaler = torch.amp.GradScaler('cuda') if config.fp16 else None
    
    # Training loop
    best_val_f1 = 0.0
    for epoch in range(1, config.num_epochs + 1):
        train_loss, train_acc, fast_ratio, deep_ratio = train_epoch(
            model, train_loader, optimizer, scheduler, device, epoch, config, scaler
        )
        val_metrics = evaluate(model, val_loader, device, compute_latency=(epoch == config.num_epochs))
        
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_model_path = Path(output_dir) / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_f1': best_val_f1,
                'model_config': model_config,
            }, best_model_path)
    
    # Upload training results to S3
    s3_training_path = f"{s3_base_path}/training"
    logger.info(f"Uploading training results to {s3_training_path}...")
    
    # Upload model
    model_path = Path(output_dir) / "best_model.pth"
    if model_path.exists():
        s3_model_path = f"{s3_training_path}/best_model.pth"
        upload_to_s3(str(model_path), s3_model_path)
        logger.info(f"✅ Model uploaded to {s3_model_path}")
    
    # Upload metrics
    metrics_path = Path(output_dir) / "final_metrics.json"
    if metrics_path.exists():
        s3_metrics_path = f"{s3_training_path}/final_metrics.json"
        upload_to_s3(str(metrics_path), s3_metrics_path)
        logger.info(f"✅ Metrics uploaded to {s3_metrics_path}")
    
    # Upload checkpoints
    if Path(checkpoint_dir).exists():
        sync_directory_to_s3(checkpoint_dir, f"{s3_training_path}/checkpoints")
        logger.info("✅ Checkpoints uploaded to S3")
    
    return str(model_path)


def run_evaluation(s3_base_path: str, model_s3_path: str, output_dir: str = "results/benchmarks"):
    """Run comprehensive benchmark evaluation"""
    logger.info("="*80)
    logger.info("STEP 4: BENCHMARK EVALUATION")
    logger.info("="*80)
    
    # Download model from S3
    local_model_path = "/tmp/best_model.pth"
    logger.info(f"Downloading model from {model_s3_path}...")
    download_from_s3(model_s3_path, local_model_path)
    logger.info("✅ Model downloaded")
    
    # Run evaluation
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    evaluator = ComprehensiveBenchmarkEvaluator(
        model_path=local_model_path,
        device='cuda' if os.environ.get('SM_NUM_GPUS', '0') != '0' else 'cpu',
    )
    
    results = evaluator.evaluate_all_benchmarks(output_dir=output_dir)
    
    # Upload evaluation results to S3
    s3_eval_path = f"{s3_base_path}/evaluation"
    logger.info(f"Uploading evaluation results to {s3_eval_path}...")
    sync_directory_to_s3(output_dir, s3_eval_path)
    logger.info("✅ Evaluation results uploaded to S3")
    
    return results


def main():
    """Main entry point for SageMaker training"""
    parser = argparse.ArgumentParser()
    
    # SageMaker environment variables
    parser.add_argument('--mode', type=str, required=True,
                       choices=['dry_run', 'hpo', 'train', 'evaluate', 'full_pipeline'],
                       help='Training mode')
    parser.add_argument('--s3-base-path', type=str, required=True,
                       help='S3 base path for outputs (e.g., s3://bucket/path)')
    
    # HPO parameters
    parser.add_argument('--n-trials', type=int, default=50, help='Number of HPO trials')
    parser.add_argument('--epochs-per-trial', type=int, default=2, help='Epochs per HPO trial')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=5, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=None, help='Learning rate')
    parser.add_argument('--hpo-params-path', type=str, default=None,
                       help='S3 path to HPO best parameters JSON')
    parser.add_argument('--use-wandb', action='store_true', help='Use Weights & Biases')
    
    # Evaluation parameters
    parser.add_argument('--model-s3-path', type=str, default=None,
                       help='S3 path to trained model for evaluation')
    
    args = parser.parse_args()
    
    # Setup S3 paths
    setup_s3_paths(args.s3_base_path)
    
    logger.info("="*80)
    logger.info("🚀 SAGEMAKER TRAINING ENTRY POINT")
    logger.info("="*80)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"S3 Base Path: {args.s3_base_path}")
    logger.info("="*80)
    
    # Run based on mode
    if args.mode == 'dry_run':
        run_dry_run(args.s3_base_path)
    
    elif args.mode == 'hpo':
        run_hpo(args.s3_base_path, args.n_trials, args.epochs_per_trial)
    
    elif args.mode == 'train':
        run_training(
            s3_base_path=args.s3_base_path,
            hpo_params_path=args.hpo_params_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            use_wandb=args.use_wandb,
        )
    
    elif args.mode == 'evaluate':
        if not args.model_s3_path:
            raise ValueError("--model-s3-path required for evaluate mode")
        run_evaluation(args.s3_base_path, args.model_s3_path)
    
    elif args.mode == 'full_pipeline':
        # Run full pipeline sequentially
        logger.info("Running full pipeline: dry_run -> hpo -> train -> evaluate")
        
        # Step 1: Dry run
        run_dry_run(args.s3_base_path)
        
        # Step 2: HPO
        best_params = run_hpo(args.s3_base_path, args.n_trials, args.epochs_per_trial)
        hpo_params_path = f"{args.s3_base_path}/hpo/best_hyperparameters.json"
        
        # Step 3: Training with best params
        model_path = run_training(
            s3_base_path=args.s3_base_path,
            hpo_params_path=hpo_params_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            use_wandb=args.use_wandb,
        )
        
        # Step 4: Evaluation
        model_s3_path = f"{args.s3_base_path}/training/best_model.pth"
        run_evaluation(args.s3_base_path, model_s3_path)
        
        logger.info("="*80)
        logger.info("✅ FULL PIPELINE COMPLETE")
        logger.info("="*80)
        logger.info(f"All results saved to: {args.s3_base_path}")
    
    logger.info("="*80)
    logger.info("✅ TRAINING COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    main()

