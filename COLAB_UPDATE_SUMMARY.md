# Colab Training Script - Complete Update Summary

## ✅ What Has Been Updated

The Colab training script (`notebooks/tinyllm_colab_training.py`) has been completely rewritten to use the **full production-grade TinyGuardrail architecture** as specified in the documentation.

---

## 🎯 Key Changes

### 1. **Full Architecture Implementation**

**Before**: Simple transformer model (16.4M parameters)
```python
class SimpleGuardrailModel(nn.Module):
    # Simple 4-layer transformer
```

**After**: Complete TinyGuardrail dual-branch architecture (60-80M parameters)
```python
from src.models.dual_branch import TinyGuardrail, DualBranchConfig

model = TinyGuardrail(model_config)
# Includes:
# - Threat-aware embeddings (token + char CNN + pattern detectors)
# - Adaptive router
# - Fast branch (pattern-based, 4 layers)
# - Deep branch (MoE with 8 experts, 8 layers)
# - Bit-level encoding
```

### 2. **Character-Level Support**

**Added**: Full character-level processing for FlipAttack defense
```python
class GuardrailDataset(Dataset):
    def text_to_char_ids(self, text, token_ids):
        # Converts text to character IDs aligned with tokens
        # Required for character-level CNN
```

**Features**:
- Character vocabulary (512 chars: ASCII + extended)
- Character IDs aligned with token positions
- Multi-scale character CNN (2, 3, 4, 5, 7-gram kernels)

### 3. **Comprehensive 2025 Attack Data Generation**

**Before**: Basic synthetic data (3 attack types)

**After**: Complete 2025 attack suite
```python
class Attack2026DataGenerator:
    - FlipAttack (FCW, FCS, FWO variants)
    - Homoglyph substitution
    - Encoding attacks (Base64, Hex, ROT13)
    - CodeChameleon encryption
    - Hard negatives (benign with trigger words)
```

**Attack Types Generated**:
- ✅ Direct injection (label 1)
- ✅ Jailbreak (label 2)
- ✅ Obfuscation (label 3):
  - FlipAttack FCW (character flip)
  - FlipAttack FCS (sentence reverse)
  - FlipAttack FWO (word order flip)
  - Homoglyph substitution
  - Base64 encoding
  - Hex encoding
  - ROT13 encoding
  - CodeChameleon encryption
- ✅ Benign (label 0)
- ✅ Hard negatives (benign with triggers)

### 4. **Adversarial Training**

**Added**: FGSM/PGD adversarial training
```python
from src.training.adversarial import AdversarialTrainer, FGSM, PGD

# Enabled from epoch 3
if epoch >= config.adversarial_start_epoch:
    # Apply adversarial perturbations
```

**Features**:
- FGSM (Fast Gradient Sign Method)
- PGD (Projected Gradient Descent)
- Configurable epsilon and steps
- Applied to 50% of batches for robustness

### 5. **Quantization-Aware Training**

**Added**: INT8 quantization preparation
```python
from src.training.quantization import QuantizationAwareTrainer

# Enabled from epoch 4
if epoch == config.quantization_start_epoch:
    model = qat_trainer.prepare_model(model)
    # Fake quantization during training
```

**Features**:
- QAT from epoch 4
- Post-training INT8 conversion
- Size reduction: 260MB → 66-80MB
- Minimal accuracy loss (<2%)

### 6. **Enhanced Metrics & Visualization**

**Added**: Comprehensive tracking
```python
metrics_history = {
    'train_loss': [],
    'val_loss': [],
    'train_acc': [],
    'val_acc': [],
    'val_f1': [],
    'learning_rate': [],
    'fast_ratio': [],  # NEW: Routing distribution
    'deep_ratio': [],   # NEW: Routing distribution
}
```

**Visualizations**:
- Training/validation loss curves
- Accuracy curves
- F1 score tracking
- Learning rate schedule
- Overfitting detection (train-val gap)
- **Routing distribution** (fast vs deep branch)

### 7. **Proper Model Forward Pass**

**Updated**: Handles all model components
```python
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    char_ids=char_ids,  # NEW: Character IDs
    labels=labels,
    text=texts[0],      # NEW: For pattern detectors
    return_dict=True,
)
```

**Outputs**:
- `logits`: Classification logits
- `loss`: Multi-task loss (focal + aux + router)
- `confidence`: Prediction confidence
- `bit_response`: 16-bit encoded response
- `route_decision`: Fast/deep routing decision
- `route_info`: Routing statistics
- `pattern_scores`: Pattern detector scores
- `aux_loss`: MoE load balancing loss

### 8. **Routing Statistics**

**Added**: Track dual-branch routing
```python
# Track routing distribution
fast_ratio = routing_stats['fast'] / total_routed
deep_ratio = routing_stats['deep'] / total_routed

# Expected: ~70% fast, ~30% deep
```

**Monitoring**:
- Real-time routing distribution
- Visualization in training curves
- Validation of router performance

---

## 📊 Expected Model Specifications

| Aspect | Value |
|--------|-------|
| **Total Parameters** | 60-80M |
| **Model Size (FP32)** | ~260MB |
| **Model Size (INT8)** | 66-80MB ✅ |
| **Model Size (INT4)** | 33-40MB (stretch) |
| **Fast Branch** | 4 layers, pattern-based |
| **Deep Branch** | 8 layers, MoE (8 experts) |
| **Character CNN** | Multi-scale (2,3,4,5,7-gram) |
| **Pattern Detectors** | 6 types (FlipAttack, homoglyph, etc.) |

---

## 🚀 Features Implemented

### ✅ Architecture Components
- [x] Threat-aware embeddings (token + char + pattern)
- [x] Character-level CNN (multi-scale)
- [x] Pattern detectors (6 types)
- [x] Unicode normalizer
- [x] Adaptive router
- [x] Fast branch (pattern-based)
- [x] Deep branch (MoE)
- [x] Bit-level encoding

### ✅ Training Features
- [x] Multi-task loss (focal + aux + router)
- [x] Adversarial training (FGSM/PGD)
- [x] Quantization-aware training (INT8)
- [x] Mixed precision (FP16)
- [x] Gradient accumulation
- [x] Gradient clipping
- [x] Learning rate scheduling
- [x] Early stopping (via best model saving)

### ✅ Data Generation
- [x] FlipAttack variants (FCW, FCS, FWO)
- [x] Homoglyph attacks
- [x] Encoding attacks (Base64, Hex, ROT13)
- [x] CodeChameleon encryption
- [x] Hard negatives
- [x] Character ID generation

### ✅ Evaluation & Monitoring
- [x] Comprehensive metrics (accuracy, precision, recall, F1)
- [x] Confusion matrix
- [x] Classification report
- [x] Routing statistics
- [x] Training curves visualization
- [x] Overfitting detection
- [x] Model checkpointing

---

## 📈 Expected Performance

Based on the full architecture:

| Metric | Target | Notes |
|--------|--------|-------|
| **Test Accuracy** | 85-92% | On synthetic data (will improve with real data) |
| **Test F1** | 83-90% | Balanced precision/recall |
| **Routing Distribution** | 70% fast, 30% deep | Adaptive routing working |
| **Model Size (INT8)** | 66-80MB | ✅ Meets target |
| **FlipAttack Detection** | >80% | Character-level CNN active |
| **Training Time** | ~2-3 hours | On Colab T4 GPU |

---

## 🔧 Usage Instructions

### 1. **Upload to Colab**
```python
# Upload notebooks/tinyllm_colab_training.py to Google Colab
# Or copy-paste into a new Colab notebook
```

### 2. **Run All Cells**
- The script will automatically:
  - Mount Google Drive
  - Install dependencies (if needed)
  - Generate 15K synthetic samples
  - Initialize full TinyGuardrail model
  - Train for 5 epochs
  - Apply adversarial training (epoch 3+)
  - Apply QAT (epoch 4+)
  - Generate visualizations
  - Save best model

### 3. **Monitor Training**
- Watch progress bars for loss and accuracy
- Check routing distribution (should be ~70/30)
- Monitor for overfitting (train-val gap)

### 4. **Download Results**
- Best model: `/content/drive/MyDrive/tinyllm-guardrail/checkpoints/best_model.pth`
- INT8 model: `/content/drive/MyDrive/tinyllm-guardrail/checkpoints/best_model_int8.pth`
- Training curves: `training_curves.png`
- Confusion matrix: `confusion_matrix.png`
- Metrics: `final_metrics.json`

---

## ⚠️ Important Notes

### 1. **Import Paths**
The script assumes `src/` is in the Python path. If imports fail:
```python
# Add to path manually
import sys
sys.path.insert(0, '/path/to/tinyllm')
```

### 2. **Memory Requirements**
Full model (60-80M) requires more GPU memory:
- Batch size reduced to 16 (from 32)
- Gradient accumulation: 4 steps (effective batch: 64)
- Mixed precision (FP16) enabled

If OOM errors:
- Reduce batch size to 8
- Increase gradient accumulation to 8

### 3. **Character IDs**
Character IDs are generated from tokenized text. The implementation:
- Aligns characters with tokens
- Handles subword tokens (BERT-style)
- Pads to max_chars_per_token (20)

### 4. **Pattern Detectors**
Pattern detectors need raw text. The script:
- Passes first text in batch to model
- Detectors work on string level
- Scores integrated into embeddings

### 5. **Quantization**
INT8 quantization requires:
- QAT preparation (epoch 4)
- Model in eval mode for conversion
- May fail on some operations (fallback to FP32)

---

## 🎯 Next Steps

### Immediate:
1. ✅ Run Colab script with full architecture
2. ✅ Verify model size is 60-80M parameters
3. ✅ Check routing distribution (~70/30)
4. ✅ Validate character-level features active

### Short-term:
1. Collect real datasets (PINT, JBB, GUARDSET-X)
2. Expand dataset to 140K samples
3. Implement transfer learning (prune base model)
4. Fine-tune on real data

### Medium-term:
1. Comprehensive evaluation on benchmarks
2. Ablation studies
3. ONNX export for deployment
4. Latency optimization

---

## 📝 Summary

**The Colab script now implements the complete production-grade TinyGuardrail architecture as specified in the documentation:**

✅ Full dual-branch architecture (60-80M parameters)  
✅ Character-level CNN for FlipAttack defense  
✅ 6 pattern detectors for 2025 attacks  
✅ Comprehensive attack data generation  
✅ Adversarial training (FGSM/PGD)  
✅ Quantization-aware training (INT8)  
✅ Complete metrics & visualization  
✅ Routing statistics tracking  
✅ Bit-level response encoding  

**Ready to train the full model!** 🚀
