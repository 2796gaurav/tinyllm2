# SageMaker Quick Start Guide

## One-Command Full Pipeline

```bash
python run_sagemaker_pipeline.py \
    --s3-base-path s3://your-bucket/tinyllm-training \
    --role arn:aws:iam::ACCOUNT:role/SageMakerRole
```

## What It Does

1. **Dry Run** (5-10 min) - Verifies everything works
2. **HPO** (2-4 hours) - Finds best hyperparameters (50 trials)
3. **Training** (4-6 hours) - Trains with best hyperparameters (5 epochs)
4. **Evaluation** (1-2 hours) - Comprehensive benchmark evaluation
5. **Production Check** - Verifies all targets are met

## Prerequisites

```bash
# 1. Set HuggingFace token
export HF_TOKEN='hf_your_token_here'

# 2. Configure AWS
aws configure
# OR
export AWS_ACCESS_KEY_ID='your_key'
export AWS_SECRET_ACCESS_KEY='your_secret'
export AWS_DEFAULT_REGION='us-east-1'
```

## Common Commands

### Full Pipeline (Recommended)
```bash
python run_sagemaker_pipeline.py \
    --s3-base-path s3://bucket/tinyllm \
    --role arn:aws:iam::ACCOUNT:role/SageMakerRole \
    --instance-type ml.g4dn.xlarge
```

### Skip Dry Run (if already verified)
```bash
python run_sagemaker_pipeline.py \
    --s3-base-path s3://bucket/tinyllm \
    --role arn:aws:iam::ACCOUNT:role/SageMakerRole \
    --skip-dry-run
```

### Skip HPO (use defaults)
```bash
python run_sagemaker_pipeline.py \
    --s3-base-path s3://bucket/tinyllm \
    --role arn:aws:iam::ACCOUNT:role/SageMakerRole \
    --skip-hpo
```

### Custom Training
```bash
python run_sagemaker_pipeline.py \
    --s3-base-path s3://bucket/tinyllm \
    --role arn:aws:iam::ACCOUNT:role/SageMakerRole \
    --epochs 10 \
    --batch-size 32 \
    --n-trials 100
```

## Instance Types

| Type | GPU | Cost/Hour | Use Case |
|------|-----|-----------|----------|
| ml.g4dn.xlarge | 1x T4 | $0.736 | Default, cost-effective |
| ml.g4dn.12xlarge | 4x T4 | $4.608 | Faster training |
| ml.g5.2xlarge | 1x A10G | $1.408 | Best performance |

## Output Structure

```
s3://bucket/tinyllm-training/
├── dry_run/          # Dry run results
├── hpo/              # HPO results + best params
├── training/         # Trained model + metrics
└── evaluation/       # Benchmark results
```

## Monitoring

- **SageMaker Console**: Training jobs, logs, metrics
- **CloudWatch**: Detailed logs
- **S3**: All outputs automatically saved

## Cost Estimate

- **ml.g4dn.xlarge**: ~$40 for full pipeline
- **ml.g4dn.12xlarge**: ~$254 for full pipeline
- **ml.g5.2xlarge**: ~$77 for full pipeline

## Troubleshooting

**"HF_TOKEN not set"**
```bash
export HF_TOKEN='hf_your_token'
```

**"Access Denied"**
- Check IAM role permissions
- Verify S3 bucket access

**"Out of Memory"**
- Use larger instance: `--instance-type ml.g4dn.12xlarge`
- Reduce batch size: `--batch-size 8`

## Next Steps

After training completes:

1. **Download model**:
   ```bash
   aws s3 cp s3://bucket/tinyllm-training/training/best_model.pth ./
   ```

2. **Check results**:
   ```bash
   aws s3 cp s3://bucket/tinyllm-training/evaluation/benchmark_results.json ./
   ```

3. **Deploy** (see SAGEMAKER_README.md)

## Full Documentation

See `SAGEMAKER_README.md` for detailed documentation.


