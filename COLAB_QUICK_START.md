# Colab Training - Quick Start Guide

## 🚀 Getting Started in 3 Steps

### Step 1: Upload to Google Colab

1. Go to https://colab.research.google.com/
2. Upload `notebooks/tinyllm_colab_training.py`
3. Or create a new notebook and copy-paste the code

### Step 2: Ensure Source Code is Available

**Option A: Upload entire `tinyllm` folder to Colab**
```python
# In Colab, upload the entire tinyllm directory
# Structure should be:
# /content/tinyllm/
#   ├── src/
#   │   ├── models/
#   │   └── training/
#   └── notebooks/
#       └── tinyllm_colab_training.py
```

**Option B: Clone from GitHub (if available)**
```python
!git clone https://github.com/yourusername/tinyllm.git /content/tinyllm
```

**Option C: Upload src/ folder separately**
```python
# Upload src/ folder to /content/src/
# The script will auto-detect and add to path
```

### Step 3: Run All Cells

The script will automatically:
- ✅ Mount Google Drive
- ✅ Install dependencies
- ✅ Generate 15K synthetic attack samples
- ✅ Initialize full TinyGuardrail model (60-80M params)
- ✅ Train for 5 epochs
- ✅ Apply adversarial training (epoch 3+)
- ✅ Apply quantization-aware training (epoch 4+)
- ✅ Generate visualizations
- ✅ Save best model to Drive

---

## 📊 What to Expect

### Model Information
```
Model Information:
  Total parameters: 60,000,000 - 80,000,000
  Model Size (FP32): ~260 MB
  Model Size (INT8): 66-80 MB ✅
```

### Training Progress
```
Epoch 1/5
  Train Loss: 0.5 → 0.2
  Train Acc: 0.75 → 0.90
  Val Acc: 0.85-0.92
  Routing: Fast 65-75%, Deep 25-35%
```

### Final Results
```
Test Accuracy: 0.85-0.92
Test F1: 0.83-0.90
Routing: Fast ~70%, Deep ~30% ✅
```

---

## 🔍 Monitoring Training

### Key Metrics to Watch

1. **Loss Decreasing**: Should drop from ~0.5 to ~0.2
2. **Accuracy Increasing**: Should reach 85-92%
3. **Routing Distribution**: Should stabilize at ~70/30
4. **Train-Val Gap**: Should be <5% (no overfitting)

### Warning Signs

⚠️ **Overfitting**: Train acc >> Val acc (>10% gap)
- Solution: Increase dropout, add more data

⚠️ **Routing Imbalanced**: Not ~70/30
- Solution: Adjust router threshold in config

⚠️ **Loss Not Decreasing**: Stuck at high value
- Solution: Check learning rate, data quality

---

## 📁 Output Files

After training, check Google Drive:
```
/content/drive/MyDrive/tinyllm-guardrail/
├── checkpoints/
│   ├── best_model.pth          # Best FP32 model
│   ├── best_model_int8.pth     # INT8 quantized model
│   └── final_metrics.json      # Final metrics
├── logs/
└── training_curves.png         # Training visualization
```

---

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'src'"

**Fix**: Add path manually at the start of script:
```python
import sys
sys.path.insert(0, '/content/tinyllm')  # Adjust path
```

### "CUDA out of memory"

**Fix**: Reduce batch size:
```python
config.batch_size = 8  # Instead of 16
config.gradient_accumulation_steps = 8  # Keep effective batch size
```

### "Model size is still 16M"

**Fix**: Check imports - make sure using `TinyGuardrail`, not `SimpleGuardrailModel`

### "Routing not working (all fast or all deep)"

**Fix**: Check router threshold:
```python
model_config.router_threshold = 0.6  # Adjust if needed
```

---

## ✅ Success Checklist

After training, verify:
- [ ] Model parameters: 60-80M (not 16M)
- [ ] Test accuracy: >85%
- [ ] Routing: ~70% fast, ~30% deep
- [ ] Model saved to Drive
- [ ] Training curves generated
- [ ] Confusion matrix created
- [ ] INT8 model created (if quantization worked)

---

## 🎯 Next Steps

1. **Evaluate on Real Data**: Test on PINT, JBB benchmarks
2. **Expand Dataset**: Collect 140K samples (60K real + 50K synthetic + 30K hard negatives)
3. **Transfer Learning**: Prune base model (Qwen3/SmolLM3) and fine-tune
4. **Deploy**: Export to ONNX, optimize for production

---

**Ready to train! Upload and run!** 🚀
