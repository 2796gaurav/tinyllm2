#!/usr/bin/env python3
"""
AWS SageMaker Pipeline Runner
Orchestrates the complete training pipeline on SageMaker:
1. Dry Run
2. Hyperparameter Optimization
3. Full Training (with best HPO params)
4. Benchmark Evaluation
5. Production Readiness Check

All outputs are stored in S3.

Usage:
    python run_sagemaker_pipeline.py \
        --s3-base-path s3://your-bucket/tinyllm-training \
        --instance-type ml.g4dn.xlarge \
        --role arn:aws:iam::ACCOUNT:role/SageMakerRole \
        [--skip-dry-run] \
        [--skip-hpo] \
        [--n-trials 50] \
        [--epochs 5]
"""

import os
import sys
import argparse
import json
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import boto3
from sagemaker import Session
from sagemaker.estimator import Estimator
from sagemaker.tuner import HyperparameterTuner, ContinuousParameter, CategoricalParameter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SageMakerPipelineRunner:
    """
    Orchestrates the complete training pipeline on SageMaker
    """
    
    def __init__(
        self,
        s3_base_path: str,
        role: str,
        instance_type: str = "ml.g4dn.xlarge",
        instance_count: int = 1,
        region: str = "us-east-1",
        image_uri: Optional[str] = None,
    ):
        """
        Initialize pipeline runner
        
        Args:
            s3_base_path: S3 base path for outputs (e.g., 's3://bucket/path')
            role: IAM role ARN for SageMaker
            instance_type: SageMaker instance type (e.g., 'ml.g4dn.xlarge')
            instance_count: Number of instances
            region: AWS region
            image_uri: Custom Docker image URI (optional, uses PyTorch default if None)
        """
        self.s3_base_path = s3_base_path.rstrip('/')
        self.role = role
        self.instance_type = instance_type
        self.instance_count = instance_count
        self.region = region
        
        # Initialize SageMaker session
        self.sagemaker_session = Session(default_bucket_prefix='tinyllm')
        
        # Docker image (use PyTorch training image if not provided)
        if image_uri is None:
            # Use PyTorch 2.0+ with CUDA
            self.image_uri = self._get_pytorch_image_uri()
        else:
            self.image_uri = image_uri
        
        logger.info("="*80)
        logger.info("🚀 SAGEMAKER PIPELINE RUNNER INITIALIZED")
        logger.info("="*80)
        logger.info(f"S3 Base Path: {self.s3_base_path}")
        logger.info(f"Instance Type: {self.instance_type}")
        logger.info(f"Instance Count: {self.instance_count}")
        logger.info(f"Region: {self.region}")
        logger.info(f"Image URI: {self.image_uri}")
        logger.info("="*80)
    
    def _get_pytorch_image_uri(self) -> str:
        """Get PyTorch training image URI"""
        from sagemaker import image_uris
        
        # Use PyTorch 2.0+ with CUDA 11.8
        return image_uris.retrieve(
            framework='pytorch',
            region=self.region,
            version='2.0',
            py_version='py310',
            instance_type=self.instance_type,
            image_scope='training',
        )
    
    def run_dry_run(self, timeout: int = 3600) -> bool:
        """
        Run dry run on SageMaker
        
        Args:
            timeout: Timeout in seconds
        
        Returns:
            True if successful
        """
        logger.info("\n" + "="*80)
        logger.info("STEP 1: DRY RUN")
        logger.info("="*80)
        
        # Create estimator for dry run
        estimator = Estimator(
            image_uri=self.image_uri,
            role=self.role,
            instance_type=self.instance_type,
            instance_count=1,  # Dry run uses single instance
            output_path=f"{self.s3_base_path}/dry_run",
            sagemaker_session=self.sagemaker_session,
            entry_point='sagemaker_train_entry.py',
            source_dir=str(Path(__file__).parent),
            hyperparameters={
                'mode': 'dry_run',
                's3-base-path': self.s3_base_path,
            },
            max_run=timeout,
            environment={
                'HF_TOKEN': os.environ.get('HF_TOKEN', ''),
            },
        )
        
        # Run training job
        logger.info("Starting dry run job...")
        estimator.fit()
        
        logger.info("✅ Dry run completed successfully")
        return True
    
    def run_hpo(
        self,
        n_trials: int = 50,
        epochs_per_trial: int = 2,
        max_jobs: int = 10,
        max_parallel_jobs: int = 2,
        timeout: int = 86400,  # 24 hours
    ) -> str:
        """
        Run hyperparameter optimization using SageMaker Hyperparameter Tuning
        
        Args:
            n_trials: Number of trials
            epochs_per_trial: Epochs per trial
            max_jobs: Maximum number of training jobs
            max_parallel_jobs: Maximum parallel jobs
            timeout: Timeout in seconds
        
        Returns:
            S3 path to best hyperparameters
        """
        logger.info("\n" + "="*80)
        logger.info("STEP 2: HYPERPARAMETER OPTIMIZATION")
        logger.info("="*80)
        
        # Create base estimator
        base_estimator = Estimator(
            image_uri=self.image_uri,
            role=self.role,
            instance_type=self.instance_type,
            instance_count=1,
            output_path=f"{self.s3_base_path}/hpo",
            sagemaker_session=self.sagemaker_session,
            entry_point='sagemaker_train_entry.py',
            source_dir=str(Path(__file__).parent),
            hyperparameters={
                'mode': 'hpo',
                's3-base-path': self.s3_base_path,
                'epochs-per-trial': str(epochs_per_trial),
            },
            max_run=timeout,
            environment={
                'HF_TOKEN': os.environ.get('HF_TOKEN', ''),
            },
        )
        
        # Define hyperparameter ranges
        hyperparameter_ranges = {
            'learning-rate': ContinuousParameter(1e-6, 1e-4),
            'router-threshold': ContinuousParameter(0.2, 0.5),
            'router-loss-weight': ContinuousParameter(0.1, 1.0),
            'batch-size': CategoricalParameter([8, 16, 32]),
            'focal-gamma': ContinuousParameter(1.5, 3.0),
            'focal-alpha': ContinuousParameter(0.1, 0.5),
            'dropout': ContinuousParameter(0.1, 0.3),
        }
        
        # Create hyperparameter tuner
        tuner = HyperparameterTuner(
            estimator=base_estimator,
            objective_metric_name='validation_f1',
            objective_type='Maximize',
            hyperparameter_ranges=hyperparameter_ranges,
            max_jobs=max_jobs,
            max_parallel_jobs=max_parallel_jobs,
            strategy='Bayesian',
            early_stopping_type='Auto',
        )
        
        # Run tuning job
        logger.info(f"Starting HPO with {max_jobs} jobs, {max_parallel_jobs} parallel...")
        tuner.fit()
        
        # Get best training job
        best_job = tuner.best_training_job()
        logger.info(f"✅ HPO completed. Best job: {best_job}")
        
        # Best hyperparameters are saved in the output path
        best_params_path = f"{self.s3_base_path}/hpo/{best_job}/output/best_hyperparameters.json"
        
        logger.info(f"Best hyperparameters saved to: {best_params_path}")
        return best_params_path
    
    def run_training(
        self,
        hpo_params_path: Optional[str] = None,
        epochs: int = 5,
        batch_size: int = 16,
        learning_rate: Optional[float] = None,
        use_wandb: bool = False,
        timeout: int = 86400,  # 24 hours
    ) -> str:
        """
        Run full production training
        
        Args:
            hpo_params_path: S3 path to best HPO parameters (optional)
            epochs: Number of epochs
            batch_size: Batch size
            learning_rate: Learning rate (overrides HPO if provided)
            use_wandb: Use Weights & Biases
            timeout: Timeout in seconds
        
        Returns:
            S3 path to trained model
        """
        logger.info("\n" + "="*80)
        logger.info("STEP 3: PRODUCTION TRAINING")
        logger.info("="*80)
        
        # Build hyperparameters
        hyperparameters = {
            'mode': 'train',
            's3-base-path': self.s3_base_path,
            'epochs': str(epochs),
            'batch-size': str(batch_size),
        }
        
        if hpo_params_path:
            hyperparameters['hpo-params-path'] = hpo_params_path
        
        if learning_rate:
            hyperparameters['learning-rate'] = str(learning_rate)
        
        if use_wandb:
            hyperparameters['use-wandb'] = 'true'
        
        # Create estimator
        estimator = Estimator(
            image_uri=self.image_uri,
            role=self.role,
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            output_path=f"{self.s3_base_path}/training",
            sagemaker_session=self.sagemaker_session,
            entry_point='sagemaker_train_entry.py',
            source_dir=str(Path(__file__).parent),
            hyperparameters=hyperparameters,
            max_run=timeout,
            environment={
                'HF_TOKEN': os.environ.get('HF_TOKEN', ''),
                'WANDB_API_KEY': os.environ.get('WANDB_API_KEY', ''),
            },
        )
        
        # Run training
        logger.info("Starting production training...")
        estimator.fit()
        
        # Model is saved in output path
        model_path = f"{self.s3_base_path}/training/best_model.pth"
        logger.info(f"✅ Training completed. Model saved to: {model_path}")
        
        return model_path
    
    def run_evaluation(
        self,
        model_s3_path: str,
        timeout: int = 7200,  # 2 hours
    ) -> Dict[str, Any]:
        """
        Run comprehensive benchmark evaluation
        
        Args:
            model_s3_path: S3 path to trained model
            timeout: Timeout in seconds
        
        Returns:
            Evaluation results dictionary
        """
        logger.info("\n" + "="*80)
        logger.info("STEP 4: BENCHMARK EVALUATION")
        logger.info("="*80)
        
        # Create estimator for evaluation
        estimator = Estimator(
            image_uri=self.image_uri,
            role=self.role,
            instance_type=self.instance_type,
            instance_count=1,
            output_path=f"{self.s3_base_path}/evaluation",
            sagemaker_session=self.sagemaker_session,
            entry_point='sagemaker_train_entry.py',
            source_dir=str(Path(__file__).parent),
            hyperparameters={
                'mode': 'evaluate',
                's3-base-path': self.s3_base_path,
                'model-s3-path': model_s3_path,
            },
            max_run=timeout,
            environment={
                'HF_TOKEN': os.environ.get('HF_TOKEN', ''),
            },
        )
        
        # Run evaluation
        logger.info("Starting benchmark evaluation...")
        estimator.fit()
        
        # Download results
        results_path = f"{self.s3_base_path}/evaluation/benchmark_results.json"
        logger.info(f"✅ Evaluation completed. Results saved to: {results_path}")
        
        return results_path
    
    def check_production_readiness(self, results_s3_path: str) -> Dict[str, bool]:
        """
        Check if model meets production readiness criteria
        
        Args:
            results_s3_path: S3 path to evaluation results
        
        Returns:
            Dictionary of target achievements
        """
        logger.info("\n" + "="*80)
        logger.info("STEP 5: PRODUCTION READINESS CHECK")
        logger.info("="*80)
        
        # Download results
        import tempfile
        import boto3
        
        s3_client = boto3.client('s3')
        bucket, key = results_s3_path.replace('s3://', '').split('/', 1)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            s3_client.download_file(bucket, key, f.name)
            
            with open(f.name, 'r') as rf:
                results = json.load(rf)
        
        # Check targets
        targets = {
            'F1 > 85%': results.get('f1', 0) > 0.85,
            'FPR < 10%': results.get('fpr', 100) < 10,
            'Latency P95 < 20ms': results.get('latency_p95', 100) < 20,
            'Routing 60-80% Fast': 0.60 <= results.get('fast_ratio', 0) <= 0.80,
        }
        
        logger.info("\n🎯 Production Readiness Check:")
        for target, achieved in targets.items():
            status = "✅" if achieved else "❌"
            logger.info(f"  {status} {target}")
        
        all_passed = all(targets.values())
        if all_passed:
            logger.info("\n🎉 ALL TARGETS ACHIEVED - PRODUCTION READY!")
        else:
            logger.warning("\n⚠️  Some targets not met - review results")
        
        return targets
    
    def run_full_pipeline(
        self,
        skip_dry_run: bool = False,
        skip_hpo: bool = False,
        n_trials: int = 50,
        epochs_per_trial: int = 2,
        epochs: int = 5,
        batch_size: int = 16,
        use_wandb: bool = False,
    ):
        """
        Run the complete pipeline end-to-end
        
        Args:
            skip_dry_run: Skip dry run step
            skip_hpo: Skip HPO step (use default hyperparameters)
            n_trials: Number of HPO trials
            epochs_per_trial: Epochs per HPO trial
            epochs: Training epochs
            batch_size: Training batch size
            use_wandb: Use Weights & Biases
        """
        logger.info("\n" + "="*80)
        logger.info("🚀 STARTING FULL SAGEMAKER PIPELINE")
        logger.info("="*80)
        
        start_time = time.time()
        
        # Step 1: Dry Run
        if not skip_dry_run:
            self.run_dry_run()
        else:
            logger.info("⏭️  Skipping dry run")
        
        # Step 2: HPO
        hpo_params_path = None
        if not skip_hpo:
            hpo_params_path = self.run_hpo(n_trials, epochs_per_trial)
        else:
            logger.info("⏭️  Skipping HPO (using default hyperparameters)")
        
        # Step 3: Training
        model_path = self.run_training(
            hpo_params_path=hpo_params_path,
            epochs=epochs,
            batch_size=batch_size,
            use_wandb=use_wandb,
        )
        
        # Step 4: Evaluation
        results_path = self.run_evaluation(model_path)
        
        # Step 5: Production Readiness Check
        self.check_production_readiness(results_path)
        
        elapsed_time = time.time() - start_time
        logger.info("\n" + "="*80)
        logger.info("🎉 FULL PIPELINE COMPLETE")
        logger.info("="*80)
        logger.info(f"Total time: {elapsed_time/3600:.2f} hours")
        logger.info(f"All results saved to: {self.s3_base_path}")
        logger.info("="*80)


def main():
    parser = argparse.ArgumentParser(description='Run TinyLLM training pipeline on SageMaker')
    
    # Required arguments
    parser.add_argument('--s3-base-path', type=str, required=True,
                       help='S3 base path for outputs (e.g., s3://bucket/tinyllm-training)')
    parser.add_argument('--role', type=str, required=True,
                       help='IAM role ARN for SageMaker')
    
    # Instance configuration
    parser.add_argument('--instance-type', type=str, default='ml.g4dn.xlarge',
                       help='SageMaker instance type (default: ml.g4dn.xlarge)')
    parser.add_argument('--instance-count', type=int, default=1,
                       help='Number of instances (default: 1)')
    parser.add_argument('--region', type=str, default='us-east-1',
                       help='AWS region (default: us-east-1)')
    
    # Pipeline steps
    parser.add_argument('--skip-dry-run', action='store_true',
                       help='Skip dry run step')
    parser.add_argument('--skip-hpo', action='store_true',
                       help='Skip hyperparameter optimization')
    
    # HPO parameters
    parser.add_argument('--n-trials', type=int, default=50,
                       help='Number of HPO trials (default: 50)')
    parser.add_argument('--epochs-per-trial', type=int, default=2,
                       help='Epochs per HPO trial (default: 2)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=5,
                       help='Training epochs (default: 5)')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size (default: 16)')
    parser.add_argument('--use-wandb', action='store_true',
                       help='Use Weights & Biases for logging')
    
    # Docker image (optional)
    parser.add_argument('--image-uri', type=str, default=None,
                       help='Custom Docker image URI (optional)')
    
    args = parser.parse_args()
    
    # Check for required environment variables
    if not os.environ.get('HF_TOKEN'):
        logger.warning("⚠️  HF_TOKEN not set. HuggingFace access may fail.")
    
    # Create pipeline runner
    runner = SageMakerPipelineRunner(
        s3_base_path=args.s3_base_path,
        role=args.role,
        instance_type=args.instance_type,
        instance_count=args.instance_count,
        region=args.region,
        image_uri=args.image_uri,
    )
    
    # Run full pipeline
    runner.run_full_pipeline(
        skip_dry_run=args.skip_dry_run,
        skip_hpo=args.skip_hpo,
        n_trials=args.n_trials,
        epochs_per_trial=args.epochs_per_trial,
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_wandb=args.use_wandb,
    )


if __name__ == "__main__":
    main()

