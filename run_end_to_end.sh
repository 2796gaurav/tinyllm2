#!/bin/bash
# TinyGuardrail: End-to-End Production Pipeline
# Complete workflow from data loading to ONNX export

set -e  # Exit on error

echo "================================================================================"
echo "🚀 TinyGuardrail: End-to-End Production Pipeline"
echo "================================================================================"
echo ""

# Check HF_TOKEN
if [ -z "$HF_TOKEN" ]; then
    echo "❌ Error: HF_TOKEN not set!"
    echo ""
    echo "Please set your HuggingFace token:"
    echo "  export HF_TOKEN='hf_your_token_here'"
    echo ""
    echo "Get your token at: https://huggingface.co/settings/tokens"
    exit 1
fi

echo "✅ HF_TOKEN found"
echo ""

# Configuration
OUTPUT_DIR=${OUTPUT_DIR:-"outputs/production_final"}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-"checkpoints/production_final"}
RESULTS_DIR=${RESULTS_DIR:-"results/production_final"}
EPOCHS=${EPOCHS:-5}
BATCH_SIZE=${BATCH_SIZE:-16}
USE_WANDB=${USE_WANDB:-false}

echo "Configuration:"
echo "  Output dir:     $OUTPUT_DIR"
echo "  Checkpoint dir: $CHECKPOINT_DIR"
echo "  Results dir:    $RESULTS_DIR"
echo "  Epochs:         $EPOCHS"
echo "  Batch size:     $BATCH_SIZE"
echo "  WandB:          $USE_WANDB"
echo ""

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$RESULTS_DIR"

# Step 1: Dry run to verify everything works
echo "================================================================================"
echo "STEP 1/6: Dry Run (Verification)"
echo "================================================================================"
echo ""

python dry_run.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Dry run failed! Fix errors before proceeding."
    exit 1
fi

echo ""
echo "✅ Dry run passed!"
echo ""

# Step 2: Hyperparameter optimization (optional, can skip with --skip-hpo)
if [ "$SKIP_HPO" != "true" ]; then
    echo "================================================================================"
    echo "STEP 2/6: Hyperparameter Optimization (50 trials)"
    echo "================================================================================"
    echo ""
    
    python hyperparameter_optimization.py \
        --n_trials 50 \
        --output_dir "$OUTPUT_DIR/hpo" \
        --epochs_per_trial 2
    
    echo ""
    echo "✅ HPO complete! Best hyperparameters saved to $OUTPUT_DIR/hpo/best_hyperparameters.json"
    echo ""
else
    echo "⏭️  Skipping HPO (SKIP_HPO=true)"
    echo ""
fi

# Step 3: Full production training
echo "================================================================================"
echo "STEP 3/6: Production Training (${EPOCHS} epochs)"
echo "================================================================================"
echo ""

WANDB_FLAG=""
if [ "$USE_WANDB" = "true" ]; then
    WANDB_FLAG="--use_wandb"
fi

python train_production_final.py \
    --output_dir "$OUTPUT_DIR" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    $WANDB_FLAG

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Training failed!"
    exit 1
fi

echo ""
echo "✅ Training complete! Model saved to $OUTPUT_DIR/best_model.pth"
echo ""

# Step 4: Comprehensive benchmark evaluation
echo "================================================================================"
echo "STEP 4/6: Benchmark Evaluation"
echo "================================================================================"
echo ""

python evaluate_benchmarks.py \
    --model_path "$OUTPUT_DIR/best_model.pth" \
    --output_dir "$RESULTS_DIR/benchmarks"

echo ""
echo "✅ Benchmark evaluation complete!"
echo ""

# Step 5: Model optimization (INT8 + ONNX)
echo "================================================================================"
echo "STEP 5/6: Model Optimization (INT8 + ONNX)"
echo "================================================================================"
echo ""

python optimize_model.py \
    --model_path "$OUTPUT_DIR/best_model.pth" \
    --output_dir "$OUTPUT_DIR/optimized"

echo ""
echo "✅ Model optimization complete!"
echo ""

# Step 6: Generate final report
echo "================================================================================"
echo "STEP 6/6: Generate Final Report"
echo "================================================================================"
echo ""

python -c "
import json
import pandas as pd
from pathlib import Path

# Load metrics
with open('$OUTPUT_DIR/final_metrics.json', 'r') as f:
    metrics = json.load(f)

# Load benchmark results
with open('$RESULTS_DIR/benchmarks/benchmark_results.json', 'r') as f:
    benchmark_results = json.load(f)

# Create comprehensive report
report = {
    'model_info': metrics['model_info'],
    'test_metrics': metrics['test_metrics'],
    'benchmark_results': benchmark_results,
    'targets_achieved': {
        'F1 > 85%': metrics['test_metrics']['f1'] > 0.85,
        'FPR < 10%': metrics['test_metrics']['fpr'] < 10,
        'Latency P95 < 20ms': metrics['test_metrics'].get('latency_p95_ms', 100) < 20,
        'Size < 80MB': metrics['model_info']['size_int8_mb'] < 80,
        'Routing 60-80% Fast': 0.6 <= metrics['test_metrics']['fast_ratio'] <= 0.8,
    }
}

# Save
with open('$RESULTS_DIR/final_report.json', 'w') as f:
    json.dump(report, f, indent=2)

print('✅ Final report saved to $RESULTS_DIR/final_report.json')

# Print summary
print('')
print('='*80)
print('📊 FINAL RESULTS SUMMARY')
print('='*80)
print(f\"Test Accuracy:  {metrics['test_metrics']['accuracy']:.2%}\")
print(f\"Test F1:        {metrics['test_metrics']['f1']:.4f} ⭐\")
print(f\"Test FPR:       {metrics['test_metrics']['fpr']:.2f}% ⭐\")
print(f\"Model Size:     {metrics['model_info']['size_int8_mb']:.2f} MB ⭐\")
print(f\"Routing:        {metrics['test_metrics']['fast_ratio']:.1%} fast, {metrics['test_metrics']['deep_ratio']:.1%} deep\")
print('')
print('🎯 Targets Achieved:')
for target, achieved in report['targets_achieved'].items():
    status = '✅' if achieved else '❌'
    print(f\"  {status} {target}\")
print('='*80)
"

echo ""
echo "================================================================================"
echo "🎉 END-TO-END PIPELINE COMPLETE!"
echo "================================================================================"
echo ""
echo "📁 Output files:"
echo "   Model:        $OUTPUT_DIR/best_model.pth"
echo "   INT8:         $OUTPUT_DIR/optimized/model_int8.pth"
echo "   ONNX:         $OUTPUT_DIR/optimized/model_onnx_optimized.onnx"
echo "   Metrics:      $OUTPUT_DIR/final_metrics.json"
echo "   Benchmarks:   $RESULTS_DIR/benchmarks/"
echo "   Report:       $RESULTS_DIR/final_report.json"
echo ""
echo "🎓 Ready for research paper submission!"
echo "================================================================================"


