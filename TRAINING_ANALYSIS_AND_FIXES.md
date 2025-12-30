# Training Analysis and Fixes

## Issues Identified

### 1. ✅ FIXED: PyTorch 2.6 torch.load Error
**Problem**: PyTorch 2.6 changed default to `weights_only=True`, but checkpoint contains custom `DualBranchConfig` class.

**Solution**: Added fallback to use `weights_only=False` for trusted checkpoints, or use `torch.serialization.add_safe_globals()` to allowlist the class.

**Location**: Lines 895-901 and 999-1005 in `tinyllm_colab_training.py`

### 2. ⚠️ Router Issue: 100% Fast Branch Routing
**Problem**: Router is routing 100% of samples to fast branch instead of target 70/30 split.

**Root Causes**:
- Router threshold (0.6) might be too high
- Router loss weight (0.1) might be too weak
- Router needs better initialization or training

**Diagnostics Added**:
- Router behavior analysis function
- Complexity score statistics
- Routing distribution by label
- Detailed router statistics

**Recommended Fixes**:
1. **Lower router threshold**: Change `router_threshold=0.6` to `router_threshold=0.3` in model config
2. **Increase router loss weight**: Change `router_loss_weight=0.1` to `router_loss_weight=0.5` in loss computation
3. **Add router warmup**: Start with higher threshold, gradually decrease during training
4. **Monitor router during training**: Add router statistics to training loop

### 3. ✅ ADDED: Comprehensive Benchmark Evaluation
**Added Components**:
- `GuardrailBenchmark` integration
- Test set evaluation (proxy for PINT/JailbreakBench)
- False Positive Rate (FPR) calculation
- Per-benchmark metrics

**Location**: After final evaluation section

### 4. ✅ ADDED: Latency & Throughput Measurement
**Added Components**:
- Per-sample latency measurement (P50, P95, P99)
- Throughput calculation (RPS - requests per second)
- Batch performance metrics
- GPU synchronization for accurate timing

**Location**: After benchmark evaluation

### 5. ✅ ADDED: Router Analysis
**Added Components**:
- Router behavior analysis function
- Complexity score statistics
- Routing distribution by label
- Fast/deep branch usage statistics

**Location**: After latency measurement

### 6. ✅ ADDED: Model Size Analysis
**Added Components**:
- Theoretical model sizes (FP32, INT8, INT4)
- Actual saved checkpoint size
- Parameter breakdown

**Location**: After router analysis

### 7. ✅ ADDED: End-to-End Verification
**Added Components**:
- Sample input testing
- Classification verification
- Route decision verification
- Confidence score display

**Location**: After model size analysis

## Training Results Analysis

### Model Performance
- **Validation Accuracy**: 98.52% (Excellent)
- **Validation F1**: 98.52% (Excellent)
- **Model Size**: 61.8M parameters, 235.77 MB FP32, 58.94 MB INT8 ✅ (Within target)
- **Training Speed**: ~11.5 it/s (Good)

### Issues to Address

#### 1. Router Not Learning (Critical)
**Symptom**: 100% fast branch, 0% deep branch
**Impact**: Deep branch (MoE) not being used, model not utilizing full architecture
**Fix**: See Router Issue section above

#### 2. Quantization Conversion Error
**Symptom**: torch.load error prevents INT8 conversion
**Status**: ✅ Fixed with weights_only=False fallback

## Recommendations

### Immediate Actions

1. **Fix Router Configuration**:
   ```python
   # In model_config creation, change:
   router_threshold=0.3,  # Lower from 0.6
   
   # In loss computation, change:
   router_loss_weight=0.5,  # Increase from 0.1
   ```

2. **Add Router Monitoring to Training Loop**:
   ```python
   # In train_epoch function, add:
   if outputs.route_info:
       fast_ratio = outputs.route_info.get('fast_ratio', 0.0)
       deep_ratio = outputs.route_info.get('deep_ratio', 0.0)
       pbar.set_postfix({
           'loss': f"{loss.item():.4f}",
           'acc': f"{correct/total:.4f}",
           'fast': f"{fast_ratio:.1%}",
           'deep': f"{deep_ratio:.1%}",
       })
   ```

3. **Retrain with Fixed Router**:
   - Lower threshold to 0.3
   - Increase router loss weight to 0.5
   - Monitor router distribution during training
   - Target: 70% fast, 30% deep

### Benchmark Evaluation

**Current Status**: Test set evaluation added (proxy for benchmarks)

**Next Steps**:
1. **Load Real Benchmarks**:
   - PINT dataset (request from Lakera AI)
   - JailbreakBench (download from HuggingFace)
   - NotInject (for FPR testing)
   - GUARDSET-X (if available)

2. **Implement Benchmark Loaders**:
   - Update `GuardrailBenchmark.load_pint()`
   - Update `GuardrailBenchmark.load_jailbreak()`
   - Update `GuardrailBenchmark.load_notinject()`

3. **Run Full Benchmark Suite**:
   - PINT accuracy target: 86-90%
   - JailbreakBench ASR target: <15%
   - NotInject FPR target: <10%

### Performance Optimization

**Current Latency**: To be measured (code added)

**Targets**:
- CPU P95 latency: <20ms
- GPU P95 latency: <5ms
- Throughput: 100-150 RPS (CPU)

**Optimization Steps**:
1. Measure current latency
2. ONNX export and optimization
3. Quantization (INT8)
4. Kernel fusion (if needed)

## End-to-End Verification

### What's Working ✅
- Model training completes successfully
- Validation accuracy: 98.52%
- Model saves correctly
- Test evaluation works
- Metrics tracking works

### What Needs Fixing ⚠️
- Router routing (100% fast branch)
- Quantization conversion (fixed, but needs retest)
- Real benchmark datasets (need to load)

### What's Missing 📋
- Real benchmark dataset loaders
- ONNX export
- Production API
- Comprehensive ablation studies

## Next Steps

1. **Fix Router** (Priority 1):
   - Update router threshold and loss weight
   - Retrain model
   - Verify 70/30 split

2. **Load Real Benchmarks** (Priority 2):
   - Download/request benchmark datasets
   - Implement loaders
   - Run full evaluation

3. **Optimize Performance** (Priority 3):
   - Measure latency
   - Export to ONNX
   - Optimize for production

4. **Production Deployment** (Priority 4):
   - Create FastAPI service
   - Add monitoring
   - Deploy to cloud

## Code Changes Summary

### Files Modified
1. `notebooks/tinyllm_colab_training.py`:
   - Fixed torch.load for PyTorch 2.6
   - Added comprehensive benchmark evaluation
   - Added latency/throughput measurement
   - Added router analysis
   - Added model size analysis
   - Added end-to-end verification

### New Features Added
- `measure_latency_throughput()`: Latency and throughput measurement
- `analyze_router_behavior()`: Router diagnostics
- `get_actual_model_size()`: Model size verification
- Comprehensive results saving to JSON

### Files to Create
- `scripts/evaluate_benchmarks.py`: Standalone benchmark evaluation
- `scripts/analyze_model.py`: Model analysis and diagnostics
- `scripts/export_onnx.py`: ONNX export script
- `api/guardrail_api.py`: FastAPI production service


