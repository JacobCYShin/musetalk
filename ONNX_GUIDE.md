# MuseTalk ONNX 가속화 가이드

MuseTalk을 ONNX로 변환하여 inference 속도를 크게 향상시키는 방법을 설명합니다.

## 🎯 ONNX 변환의 장점

- **속도 향상**: 일반적으로 1.5-3x 빠른 inference
- **메모리 효율성**: 더 적은 GPU 메모리 사용
- **최적화**: 그래프 최적화 및 연산 융합
- **호환성**: 다양한 하드웨어에서 최적화된 실행

## 📋 필수 요구사항

### 1. Python 패키지 설치
```bash
# ONNX 및 ONNXRuntime 설치
pip install onnx onnxruntime-gpu

# 또는 CPU만 사용하는 경우
pip install onnx onnxruntime
```

### 2. 시스템 요구사항
- **GPU**: NVIDIA GPU (CUDA 지원)
- **VRAM**: 최소 4GB 권장
- **PyTorch**: 1.12.0 이상
- **CUDA**: 11.0 이상 권장

## 🚀 사용 방법

### Step 1: ONNX 모델 변환

먼저 PyTorch 모델을 ONNX로 변환합니다:

```bash
python convert_to_onnx.py
```

변환 과정에서 다음 모델들이 생성됩니다:
- `onnx_models/unet.onnx` - 메인 UNet 모델
- `onnx_models/vae_encoder.onnx` - VAE 인코더
- `onnx_models/vae_decoder.onnx` - VAE 디코더  
- `onnx_models/positional_encoding.onnx` - 위치 인코딩

### Step 2: test_avatar 준비

ONNX 서버는 미리 로드된 test_avatar를 사용합니다:

```bash
# normal 모드로 test_avatar 생성 (아직 없다면)
sh inference.sh v1.5 normal
```

### Step 3: ONNX 서버 실행

```bash
python fastapi_realtime_onnx.py
```

서버 시작시 다음과 같은 출력을 확인하세요:
```
✅ ONNXRuntime available
🚀 Using CUDA Execution Provider
✅ unet loaded
✅ vae_encoder loaded
✅ vae_decoder loaded
✅ positional_encoding loaded
✅ ONNX Test avatar loaded successfully!
```

### Step 4: API 사용

#### 간단한 사용법:
```bash
curl -X POST "http://localhost:8000/onnx-infer" \
     -F "audio_file=@data/audio/sun.wav"
```

#### 파라미터 포함:
```bash
curl -X POST "http://localhost:8000/onnx-infer" \
     -F "audio_file=@data/audio/sun.wav" \
     -F "fps=25" \
     -F "skip_save_images=false"
```

## 📊 성능 비교

### 자동 벤치마크 실행:
```bash
python benchmark_onnx_vs_pytorch.py
```

### 예상 성능 향상:

| 모델 | PyTorch | ONNX | 속도 향상 |
|------|---------|------|-----------|
| UNet | 100ms | 60ms | 1.67x |
| VAE Decoder | 50ms | 30ms | 1.67x |
| 전체 Pipeline | 5.2s | 3.1s | 1.68x |

*실제 성능은 하드웨어에 따라 달라질 수 있습니다.

## 🔧 문제 해결

### 1. ONNXRuntime 설치 오류
```bash
# CUDA 버전 확인
nvidia-smi

# 해당 CUDA 버전에 맞는 ONNXRuntime 설치
pip install onnxruntime-gpu==1.16.3
```

### 2. ONNX 변환 실패
```bash
# PyTorch 버전 확인
python -c "import torch; print(torch.__version__)"

# 호환되는 버전으로 업그레이드
pip install torch>=1.12.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. 메모리 부족 오류
```python
# convert_to_onnx.py에서 배치 크기 줄이기
batch_size = 1  # 기본값에서 줄임
```

### 4. CUDA Provider 사용 불가
```bash
# CUDA 및 cuDNN 설치 확인
python -c "import torch; print(torch.cuda.is_available())"

# ONNXRuntime GPU 버전 재설치
pip uninstall onnxruntime onnxruntime-gpu
pip install onnxruntime-gpu
```

## 📈 최적화 팁

### 1. Provider 우선순위 설정
ONNX Runtime은 다음 순서로 provider를 시도합니다:
1. `CUDAExecutionProvider` (GPU)
2. `CPUExecutionProvider` (CPU fallback)

### 2. 메모리 최적화
- GPU 메모리 제한: 4GB로 설정됨
- 필요시 `fastapi_realtime_onnx.py`에서 조정 가능

### 3. 배치 크기 조정
```python
# 더 큰 GPU에서는 배치 크기 증가 가능
global_onnx_avatar = ONNXOptimizedAvatar(avatar_id="test_avatar", batch_size=16)
```

## 🆚 PyTorch vs ONNX 비교

| 특징 | PyTorch | ONNX |
|------|---------|------|
| **속도** | 기준 | 1.5-3x 빠름 |
| **메모리** | 기준 | 10-20% 적음 |
| **호환성** | PyTorch만 | 다양한 런타임 |
| **디버깅** | 쉬움 | 제한적 |
| **배포** | 복잡 | 간단 |
| **최적화** | 수동 | 자동 |

## 🔄 버전 호환성

| MuseTalk | PyTorch | ONNX | ONNXRuntime |
|----------|---------|------|-------------|
| v1.5 | ✅ | ✅ | 1.16+ |
| v1.0 | ✅ | 🔄 | 1.16+ |

## 💡 사용 권장사항

### ONNX를 사용하세요:
- ✅ 프로덕션 환경
- ✅ 높은 throughput 필요
- ✅ 메모리 제약이 있는 환경
- ✅ 여러 클라이언트 동시 처리

### PyTorch를 사용하세요:
- ✅ 개발 및 디버깅
- ✅ 모델 수정이 빈번한 경우
- ✅ 실험적 기능 사용
- ✅ 호환성 문제가 있는 경우

## 📞 지원

문제가 발생하면 다음을 확인하세요:

1. **로그 확인**: 서버 시작시 모든 모델이 정상 로드되는지 확인
2. **GPU 메모리**: `nvidia-smi`로 메모리 사용량 확인
3. **버전 호환성**: PyTorch, ONNX, ONNXRuntime 버전 확인
4. **테스트**: `benchmark_onnx_vs_pytorch.py`로 동작 확인

---

**참고**: ONNX 변환은 모델의 동적 부분을 정적으로 만들기 때문에, 일부 고급 기능이 제한될 수 있습니다. 대부분의 일반적인 사용 사례에서는 문제없이 동작합니다.
