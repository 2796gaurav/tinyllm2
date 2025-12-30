# Quick Fix Guide: Update Colab Script to Use Full Architecture

## 🎯 Goal
Replace `SimpleGuardrailModel` with full `TinyGuardrail` dual-branch architecture in the Colab training script.

---

## ✅ Current Status

**What's Working**:
- ✅ Training loop: Loss decreasing, accuracy improving
- ✅ Test Accuracy: 91.30% (good for simple model)
- ✅ No overfitting: Stable train-val gap
- ✅ Metrics tracking: All metrics logged correctly

**What's Missing**:
- ❌ Using `SimpleGuardrailModel` (16.4M) instead of `TinyGuardrail` (60-80M)
- ❌ No character-level CNN (cannot defend FlipAttack)
- ❌ No pattern detectors
- ❌ No dual-branch routing
- ❌ No MoE

---

## 🔧 Quick Fix (5 Minutes)

### Step 1: Update Imports

In `notebooks/tinyllm_colab_training.py`, replace:

```python
# OLD (line ~308):
class SimpleGuardrailModel(nn.Module):
    # ... simple model code ...

# NEW:
import sys
sys.path.append('/content')  # Or path to your src directory

from src.models.dual_branch import TinyGuardrail, DualBranchConfig
```

### Step 2: Replace Model Creation

Replace lines 355-357:

```python
# OLD:
model = SimpleGuardrailModel(config).to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# NEW:
# Create full dual-branch config
model_config = DualBranchConfig(
    vocab_size=8000,  # Will need custom tokenizer
    d_model=384,
    num_labels=4,
    # Use defaults for other params
)

# Create full model
model = TinyGuardrail(model_config).to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Expected: 60-80M parameters")
```

### Step 3: Update Forward Pass

The `TinyGuardrail` model has a different forward signature. Update the training loop:

```python
# OLD (in train_epoch function):
outputs = model(input_ids, attention_mask, labels)

# NEW:
# TinyGuardrail expects different input format
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    labels=labels,
    return_dict=True
)
```

### Step 4: Handle Model Output

Update loss calculation:

```python
# OLD:
loss = outputs['loss']

# NEW:
# TinyGuardrail returns dict with 'loss' and 'logits'
if 'loss' in outputs:
    loss = outputs['loss']
else:
    # Compute loss manually if needed
    logits = outputs['logits']
    loss = F.cross_entropy(logits, labels)
```

---

## ⚠️ Important Notes

### 1. **Vocabulary Size Mismatch**

The current script uses BERT tokenizer (30K vocab), but `TinyGuardrail` expects 8K vocab.

**Quick Fix**: Use BERT tokenizer but set `vocab_size=30522` in config:

```python
model_config = DualBranchConfig(
    vocab_size=30522,  # BERT vocab size (temporary)
    d_model=384,
    num_labels=4,
)
```

**Proper Fix**: Create custom tokenizer with 8K vocab (later).

### 2. **Character IDs Required**

`TinyGuardrail` needs character-level input for the character CNN.

**Quick Fix**: Generate dummy character IDs:

```python
# In GuardrailDataset.__getitem__:
def __getitem__(self, idx):
    # ... existing code ...
    
    # Generate character IDs (dummy for now)
    char_ids = torch.zeros((self.max_length, 20), dtype=torch.long)  # 20 chars per token
    
    return {
        'input_ids': encoding['input_ids'].squeeze(0),
        'attention_mask': encoding['attention_mask'].squeeze(0),
        'char_ids': char_ids,  # Add this
        'labels': torch.tensor(label, dtype=torch.long),
    }
```

**Proper Fix**: Implement proper character tokenization (later).

### 3. **Memory Requirements**

Full model (60-80M) requires more GPU memory.

**If OOM errors**:
- Reduce batch size: `config.batch_size = 16`
- Increase gradient accumulation: `config.gradient_accumulation_steps = 8`
- Use gradient checkpointing

---

## 🚀 Complete Updated Section

Here's the complete replacement for the model section:

```python
# =============================================================================
# FULL DUAL-BRANCH MODEL (Replace simple model)
# =============================================================================

import sys
import os

# Add src to path
if IN_COLAB:
    # If src is in Colab, adjust path
    sys.path.append('/content')
else:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.dual_branch import TinyGuardrail, DualBranchConfig

# Create full dual-branch config
model_config = DualBranchConfig(
    vocab_size=30522,  # BERT vocab (temporary, should be 8000)
    d_model=384,
    num_labels=4,
    # Use defaults for other params
)

# Create full model
model = TinyGuardrail(model_config).to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Model parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Expected: 60-80M parameters")
print(f"Model size (FP32): {total_params * 4 / 1e6:.2f} MB")
print(f"Model size (INT8): {total_params * 1 / 1e6:.2f} MB")
```

---

## 📊 Expected Changes After Fix

| Aspect | Before | After |
|--------|--------|-------|
| **Model Size** | 16.4M | 60-80M |
| **Parameters** | Simple transformer | Dual-branch + MoE |
| **Character Defense** | ❌ None | ✅ Character CNN |
| **Pattern Detectors** | ❌ None | ✅ 6 detectors |
| **Routing** | ❌ Single path | ✅ 70/30 split |
| **MoE** | ❌ None | ✅ 8 experts |

---

## 🧪 Testing the Fix

After updating, run a quick test:

```python
# Test forward pass
test_batch = next(iter(train_loader))
input_ids = test_batch['input_ids'].to(device)
attention_mask = test_batch['attention_mask'].to(device)
labels = test_batch['labels'].to(device)

# Add dummy char_ids if needed
if 'char_ids' not in test_batch:
    char_ids = torch.zeros((input_ids.size(0), input_ids.size(1), 20), dtype=torch.long).to(device)
else:
    char_ids = test_batch['char_ids'].to(device)

# Forward pass
with torch.no_grad():
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        char_ids=char_ids,
        labels=labels,
    )

print(f"Output keys: {outputs.keys()}")
print(f"Logits shape: {outputs['logits'].shape}")
print(f"Loss: {outputs['loss'].item():.4f}")

# Check routing (if available)
if 'route_info' in outputs:
    print(f"Fast branch ratio: {outputs['route_info']['fast_ratio']:.2%}")
    print(f"Deep branch ratio: {outputs['route_info']['deep_ratio']:.2%}")
```

---

## ⏭️ Next Steps After Fix

1. **Verify Model Works**: Run 1 epoch, check metrics
2. **Check Routing**: Should see ~70% fast, 30% deep
3. **Monitor Memory**: Watch for OOM errors
4. **Then**: Implement transfer learning, expand dataset

---

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'src'"

**Fix**: Add path to sys.path:
```python
import sys
sys.path.append('/path/to/tinyllm')  # Adjust path
```

### "CUDA out of memory"

**Fix**: Reduce batch size:
```python
config.batch_size = 16  # Or 8
config.gradient_accumulation_steps = 8  # Keep effective batch size
```

### "TypeError: forward() got unexpected keyword argument"

**Fix**: Check model's forward signature in `src/models/dual_branch.py` and match it.

### "Model size is still 16M"

**Fix**: Make sure you're importing `TinyGuardrail`, not `SimpleGuardrailModel`.

---

## ✅ Success Criteria

After fix, you should see:
- ✅ Model parameters: 60-80M (not 16M)
- ✅ Training starts without errors
- ✅ Routing distribution visible (if logged)
- ✅ Character CNN active (check embeddings)
- ✅ Pattern detectors active (check outputs)

---

**Once this works, proceed to implement transfer learning and expand the dataset!** 🚀
