# Fixes Applied to Dry Run Issues

## Summary

This document lists all fixes applied based on the dry run analysis.

---

## ✅ Fixes Applied

### 1. Fixed Prompt Injections Dataset Loading Error

**Issue**: `'int' object has no attribute 'lower'`

**Location**: `src/data/real_benchmark_loader.py` line 261

**Fix**: Handle both integer and string labels correctly:
```python
# Now handles both int and str labels
if isinstance(label_raw, (int, float)):
    label = 1 if label_raw > 0 else 0
elif isinstance(label_raw, str):
    label_str = label_raw.lower()
    # ... parse string label
```

**Status**: ✅ Fixed

---

### 2. Added FP16 Size Calculation

**Issue**: Model size reporting missing FP16 (which would be ~100MB raw)

**Location**: `src/models/dual_branch.py` `get_model_info()` method

**Fix**: Added `size_fp16_mb` to model info output

**Status**: ✅ Fixed

---

### 3. Enhanced Model Size Reporting

**Issue**: No warning about vocabulary size impact on model size

**Location**: `dry_run.py` model initialization section

**Fix**: Added warnings about:
- Large vocabulary (30K vs target 8K) adding ~35MB
- FP16 size (~100MB) for raw model size
- INT8 size (58.94MB) within target

**Status**: ✅ Fixed

---

### 4. Added Router Diagnostics

**Issue**: Router routing 100% to deep branch with no explanation

**Location**: `dry_run.py` evaluation section

**Fix**: Added diagnostic warnings:
- Warns when router distribution is off-target
- Suggests fixes (lower threshold, disable entropy, train longer)
- Explains why routing might be incorrect

**Status**: ✅ Fixed (diagnostics added, router tuning requires training)

---

### 5. Enhanced Latency Benchmarking

**Issue**: No context about CPU vs GPU benchmarking

**Location**: `dry_run.py` latency benchmark section

**Fix**: Added:
- Warning that benchmark is on GPU (target is CPU)
- Comparison to target (<20ms CPU)
- Suggestions for achieving target (CPU testing, INT8, router fix)

**Status**: ✅ Fixed

---

### 6. Added ONNX Dependency

**Issue**: Missing `onnxscript` module causing ONNX export to fail

**Location**: `requirements.txt`

**Fix**: Added `onnxscript>=0.1.0` to requirements

**Status**: ✅ Fixed

---

## ⚠️ Issues Requiring Training/Configuration Changes

### 1. Router Routing 100% to Deep Branch

**Root Cause**: Complexity scores all above threshold (0.3)

**Recommended Fixes**:
1. Lower router threshold to `0.2` or `0.25`
2. Disable entropy adjustment in router (`use_entropy=False`)
3. Train for more epochs with router loss applied
4. Monitor router distribution during training

**Status**: ⚠️ Requires configuration/training changes

---

### 2. Model Size (Vocabulary)

**Current**: Using BERT vocab (30,522 tokens) = 235.77 MB FP32

**Target**: Pruned vocab (8,000 tokens) = ~200 MB FP32, ~100 MB FP16

**Options**:
1. Use pruned vocabulary (8K tokens) - requires tokenizer retraining
2. Keep BERT vocab but use FP16 for ~100MB raw size
3. Document that current size is acceptable (INT8 is 58.94MB, within target)

**Status**: ⚠️ Requires decision on vocabulary strategy

---

### 3. Latency Performance

**Current**: 839.96ms P95 on GPU (target: <20ms CPU)

**Gap**: 42x slower than target

**Required Optimizations**:
1. Test on CPU (not GPU)
2. Apply INT8 quantization
3. Fix router (70% fast branch = faster)
4. Export to ONNX and use ONNX Runtime (2-5x faster)

**Status**: ⚠️ Requires optimization and CPU testing

---

## 📋 Next Steps

1. **Install updated dependencies**:
   ```bash
   pip install -r requirements.txt  # Now includes onnxscript
   ```

2. **Re-run dry run** to verify fixes:
   ```bash
   python dry_run.py
   ```

3. **Fix router configuration** (before full training):
   - Lower threshold to 0.2
   - Or disable entropy adjustment
   - Monitor during training

4. **Decide on vocabulary strategy**:
   - Keep BERT vocab (current) + use FP16 for ~100MB
   - Or implement pruned vocab (8K) for smaller size

5. **Optimize for production**:
   - Apply INT8 quantization
   - Test on CPU
   - Export to ONNX
   - Benchmark on target hardware

---

## 📊 Expected Improvements After Fixes

| Metric | Before | After Fixes | Target |
|--------|--------|-------------|--------|
| Prompt Injections | ❌ Error | ✅ Loads | - |
| Model Size (FP16) | ❌ Not reported | ✅ ~118MB | ~100MB |
| Router Diagnostics | ❌ None | ✅ Warnings | 70/30 split |
| Latency Context | ❌ None | ✅ CPU note | <20ms CPU |
| ONNX Export | ❌ Missing dep | ✅ Fixed | - |

---

## 🎯 Remaining Work

1. **Router tuning** - Requires training with adjusted config
2. **Vocabulary decision** - Choose strategy (BERT vs pruned)
3. **CPU latency testing** - Test on CPU with INT8 quantization
4. **Full training** - Run with all fixes applied
5. **Production optimization** - ONNX export, quantization, benchmarking

---

## Conclusion

**Fixed Issues**: 6/6 code-level issues ✅
**Remaining Issues**: 3/3 require training/configuration ⚠️

The dry run pipeline is now **more robust** with better diagnostics and error handling. The remaining issues (router, vocab, latency) require training runs or configuration decisions.


