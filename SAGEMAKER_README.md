# AWS SageMaker Training Pipeline

Complete end-to-end training pipeline for TinyLLM Guardrail on AWS SageMaker with GPU support and S3 storage.

## Overview

This pipeline runs the complete training workflow on SageMaker:

1. **Dry Run** - Quick verification (5-10 minutes)
2. **Hyperparameter Optimization** - Find best hyperparameters (2-4 hours)
3. **Full Training** - Train with best hyperparameters (4-6 hours)
4. **Benchmark Evaluation** - Comprehensive evaluation (1-2 hours)
5. **Production Readiness Check** - Verify all targets are met

All outputs (models, checkpoints, metrics, results) are automatically stored in S3.

## Prerequisites

1. **AWS Account** with SageMaker access
2. **IAM Role** with SageMaker permissions:
   - `AmazonSageMakerFullAccess`
   - S3 read/write access to your bucket
3. **HuggingFace Token** (`HF_TOKEN` environment variable)
4. **S3 Bucket** for storing outputs

## Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Set HuggingFace token
export HF_TOKEN='hf_your_token_here'

# Optional: Set Weights & Biases API key
export WANDB_API_KEY='your_wandb_key'
```

### 2. Configure AWS Credentials

```bash
# Option 1: AWS CLI
aws configure

# Option 2: Environment variables
export AWS_ACCESS_KEY_ID='your_key'
export AWS_SECRET_ACCESS_KEY='your_secret'
export AWS_DEFAULT_REGION='us-east-1'
```

### 3. Run Full Pipeline

```bash
python run_sagemaker_pipeline.py \
    --s3-base-path s3://your-bucket/tinyllm-training \
    --role arn:aws:iam::ACCOUNT:role/SageMakerRole \
    --instance-type ml.g4dn.xlarge \
    --epochs 5 \
    --n-trials 50
```

## Detailed Usage

### Basic Pipeline

```bash
python run_sagemaker_pipeline.py \
    --s3-base-path s3://your-bucket/tinyllm-training \
    --role arn:aws:iam::ACCOUNT:role/SageMakerRole
```

### Custom Instance Type

For faster training, use a larger GPU instance:

```bash
# Single GPU (T4)
--instance-type ml.g4dn.xlarge

# Multi-GPU (4x T4)
--instance-type ml.g4dn.12xlarge
--instance-count 1

# A100 GPU (faster, more expensive)
--instance-type ml.g5.2xlarge
```

### Skip Steps

```bash
# Skip dry run (if already verified)
python run_sagemaker_pipeline.py \
    --s3-base-path s3://bucket/path \
    --role arn:aws:iam::ACCOUNT:role/SageMakerRole \
    --skip-dry-run

# Skip HPO (use default hyperparameters)
python run_sagemaker_pipeline.py \
    --s3-base-path s3://bucket/path \
    --role arn:aws:iam::ACCOUNT:role/SageMakerRole \
    --skip-hpo
```

### Custom Hyperparameters

```bash
python run_sagemaker_pipeline.py \
    --s3-base-path s3://bucket/path \
    --role arn:aws:iam::ACCOUNT:role/SageMakerRole \
    --epochs 10 \
    --batch-size 32 \
    --n-trials 100 \
    --epochs-per-trial 3
```

### With Weights & Biases

```bash
export WANDB_API_KEY='your_key'

python run_sagemaker_pipeline.py \
    --s3-base-path s3://bucket/path \
    --role arn:aws:iam::ACCOUNT:role/SageMakerRole \
    --use-wandb
```

## S3 Structure

All outputs are organized in S3:

```
s3://your-bucket/tinyllm-training/
├── dry_run/
│   ├── dry_run_model.pth
│   ├── dry_run_model.onnx
│   └── dry_run_report.json
├── hpo/
│   ├── best_hyperparameters.json
│   ├── hpo_trials.csv
│   └── optimization_history.html
├── training/
│   ├── best_model.pth
│   ├── final_metrics.json
│   ├── training_metrics.csv
│   └── checkpoints/
│       └── checkpoint_*.pth
└── evaluation/
    ├── benchmark_results.json
    ├── benchmark_summary.csv
    └── ...
```

## Individual Steps

You can also run individual steps using the entry point script:

### Dry Run Only

```python
from sagemaker.estimator import Estimator

estimator = Estimator(
    image_uri='763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.0-gpu-py310',
    role='arn:aws:iam::ACCOUNT:role/SageMakerRole',
    instance_type='ml.g4dn.xlarge',
    instance_count=1,
    hyperparameters={
        'mode': 'dry_run',
        's3-base-path': 's3://bucket/path',
    },
)

estimator.fit()
```

### HPO Only

```python
from sagemaker.tuner import HyperparameterTuner, ContinuousParameter

base_estimator = Estimator(...)

tuner = HyperparameterTuner(
    estimator=base_estimator,
    objective_metric_name='validation_f1',
    hyperparameter_ranges={
        'learning-rate': ContinuousParameter(1e-6, 1e-4),
        'router-threshold': ContinuousParameter(0.2, 0.5),
        ...
    },
    max_jobs=50,
    max_parallel_jobs=2,
)

tuner.fit()
```

## Cost Estimation

Approximate costs (us-east-1, as of 2024):

| Instance Type | Hourly Rate | Dry Run | HPO (50 trials) | Training (5 epochs) | Total |
|--------------|-------------|---------|-----------------|---------------------|-------|
| ml.g4dn.xlarge | $0.736 | $0.12 | $36.80 | $3.68 | ~$40 |
| ml.g4dn.12xlarge | $4.608 | $0.77 | $230.40 | $23.04 | ~$254 |
| ml.g5.2xlarge | $1.408 | $0.23 | $70.40 | $7.04 | ~$77 |

**Note:** Actual costs depend on:
- Instance availability
- Training time (varies with data size)
- HPO trials (can be pruned early)
- Region pricing

## Monitoring

### SageMaker Console

1. Go to **SageMaker > Training > Training jobs**
2. Monitor job status, logs, and metrics
3. View CloudWatch logs for detailed output

### CloudWatch Logs

Each training job creates CloudWatch log groups:
- `/aws/sagemaker/TrainingJobs/<job-name>`

### Weights & Biases

If `--use-wandb` is enabled:
- Real-time metrics visualization
- Hyperparameter tracking
- Model versioning

## Troubleshooting

### Common Issues

1. **"HF_TOKEN not set"**
   ```bash
   export HF_TOKEN='hf_your_token_here'
   ```

2. **"Access Denied" to S3**
   - Check IAM role permissions
   - Verify S3 bucket policy

3. **"Out of Memory"**
   - Use larger instance type
   - Reduce batch size
   - Enable gradient checkpointing

4. **"Job Timeout"**
   - Increase `max_run` timeout
   - Use faster instance type

### Debug Mode

Run locally first to verify:

```bash
# Test dry run locally
python dry_run.py

# Test HPO locally (smaller)
python hyperparameter_optimization.py --n_trials 5

# Test training locally
python train_production_final.py --epochs 1
```

## Advanced Configuration

### Custom Docker Image

If you need a custom environment:

```bash
python run_sagemaker_pipeline.py \
    --s3-base-path s3://bucket/path \
    --role arn:aws:iam::ACCOUNT:role/SageMakerRole \
    --image-uri 763104351884.dkr.ecr.us-east-1.amazonaws.com/custom-image:tag
```

### Multi-Region

```bash
python run_sagemaker_pipeline.py \
    --s3-base-path s3://bucket/path \
    --role arn:aws:iam::ACCOUNT:role/SageMakerRole \
    --region eu-west-1
```

### Spot Instances (Cost Savings)

Modify the script to use spot instances:

```python
estimator = Estimator(
    ...
    use_spot_instances=True,
    max_wait=86400,  # Max wait time
    max_run=86400,
)
```

## Production Deployment

After training completes:

1. **Download model from S3:**
   ```bash
   aws s3 cp s3://bucket/path/training/best_model.pth ./model.pth
   ```

2. **Deploy to SageMaker Endpoint:**
   ```python
   from sagemaker.pytorch import PyTorchModel
   
   model = PyTorchModel(
       model_data='s3://bucket/path/training/best_model.pth',
       role=role,
       entry_point='inference.py',
   )
   
   predictor = model.deploy(instance_type='ml.m5.large')
   ```

3. **Or export to ONNX:**
   ```bash
   python export_onnx.py --model_path ./model.pth
   ```

## Best Practices

1. **Start Small**: Run dry run first to verify setup
2. **Monitor Costs**: Use spot instances for HPO
3. **Save Checkpoints**: Enable checkpointing for long training
4. **Use HPO**: Let SageMaker find optimal hyperparameters
5. **Version Control**: Tag S3 paths with experiment names
6. **Clean Up**: Delete old training jobs to save costs

## Support

For issues or questions:
- Check CloudWatch logs
- Review SageMaker training job logs
- Verify IAM permissions
- Check S3 bucket access

## Example: Complete Workflow

```bash
# 1. Setup
export HF_TOKEN='hf_...'
export AWS_DEFAULT_REGION='us-east-1'

# 2. Run full pipeline
python run_sagemaker_pipeline.py \
    --s3-base-path s3://my-bucket/tinyllm-experiment-1 \
    --role arn:aws:iam::123456789:role/SageMakerExecutionRole \
    --instance-type ml.g4dn.xlarge \
    --epochs 5 \
    --n-trials 50 \
    --use-wandb

# 3. Monitor in SageMaker console
# 4. Download results when complete
aws s3 sync s3://my-bucket/tinyllm-experiment-1 ./results/

# 5. Check production readiness
cat results/evaluation/benchmark_results.json
```

## Next Steps

After successful training:

1. **Evaluate on custom benchmarks**
2. **Deploy to production**
3. **Monitor in production**
4. **Iterate based on feedback**


