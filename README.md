# Jetson에서 PaddlePaddle 컴파일 가이드

## 환경 정보
- **Python**: 3.10
- **Paddle**: release/3.1.1
- **CUDA**: 12.6
- **OS**: Jetpack 6.2.1
- **Device**: Jetson Orin Nano

## 1. 사전 준비

### 1.1 필수 패키지 설치
```bash
# patchelf 설치 (빌드 과정에서 필요)
sudo apt install patchelf

# Python 가상환경 생성 및 활성화
python3 -m venv paddle_env
source paddle_env/bin/activate

# 기본 빌드 도구 설치
pip install pyyaml
pip install pybind11-stubgen
pip install wheel
```

### 1.2 ulimit 설정
```bash
ulimit -n 65535
# 또는 unlimited로 설정
ulimit -n unlimited
```

## 2. 소스 코드 다운로드 및 체크아웃

```bash
git clone https://github.com/PaddlePaddle/Paddle.git
cd Paddle
git checkout release/3.1.1
```

## 3. 소스 코드 패치

### 3.1 타입 불일치 오류 수정

`paddle/phi/kernels/gpu/roi_align_kernel.cu` 파일의 176~181줄 수정:

**변경 전:**
```cpp
int64_t output_size = out->numel();
uint32_t blocks = NumBlocks(output_size);
uint32_t threads = kNumCUDAThreads;
#ifdef WITH_NV_JETSON
backends::gpu::ChangeThreadNum(dev_ctx, &threads, 256);
#endif
```

**변경 후:**
```cpp
int64_t output_size = out->numel();
uint32_t blocks = NumBlocks(output_size);
#if defined(WITH_NV_JETSON)
// ChangeThreadNum는 int* 를 받으므로 int 로 보정
int threads_i = static_cast<int>(kNumCUDAThreads);
backends::gpu::ChangeThreadNum(dev_ctx, &threads_i, 256);
uint32_t threads = static_cast<uint32_t>(threads_i);
#else
uint32_t threads = kNumCUDAThreads;
#endif
```

## 4. CMake 설정 및 빌드

### 4.1 빌드 디렉토리 생성
```bash
mkdir -p build && cd build
```

### 4.2 CMake 설정
```bash
cmake .. -DPY_VERSION=3.10 \
  -DWITH_MKL=OFF \
  -DWITH_TESTING=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DON_INFER=ON \
  -DWITH_PYTHON=ON \
  -DWITH_XBYAK=OFF \
  -DWITH_NV_JETSON=ON \
  -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
  -DWITH_NCCL=OFF \
  -DWITH_RCCL=OFF \
  -DWITH_DISTRIBUTE=OFF \
  -DWITH_GPU=ON \
  -DWITH_TENSORRT=ON \
  -DWITH_ARM=ON
```

### 4.3 컴파일 (약 3시간 소요)
```bash
make TARGET=ARMV8 -j3
```

**참고**: `-j3`은 3개의 코어를 사용합니다. 시스템 사양에 맞게 조정 가능합니다.

## 5. Python 의존성 설치

빌드 중 의존성 오류가 발생하면:

```bash
# 추가로 필요한 패키지들
pip install httpx
```

## 6. Wheel 패키지 생성

### 6.1 수동 wheel 생성 (오류 발생 시)
```bash
cd python
python setup.py bdist_wheel
```

### 6.2 생성된 wheel 파일 위치
```
build/python/dist/paddlepaddle_gpu-0.0.0-cp310-cp310-linux_aarch64.whl
```

## 7. 설치

```bash
pip install python/dist/paddlepaddle_gpu-0.0.0-cp310-cp310-linux_aarch64.whl
```

## 8. 테스트

```python
import paddle
print(paddle.__version__)
print(paddle.device.cuda.device_count())

# GPU 사용 가능 여부 확인
print(paddle.device.is_compiled_with_cuda())
```

## 주요 오류 해결 방법 요약

| 오류 | 해결 방법 |
|------|-----------|
| `uint32_t *` 타입 불일치 | 소스 코드 패치 (섹션 3.1) |
| `patchelf: not found` | `sudo apt install patchelf` |
| `Missing build dependency: httpx` | `pip install -r python/requirements.txt` |
| wheel 생성 실패 | `cd python && python setup.py bdist_wheel` |
| 기타 라이브러리 누락 | `pip install pyyaml pybind11-stubgen wheel` |

## 성능 참고

- 컴파일된 PaddlePaddle GPU 버전의 평균 실행 시간: **약 50ms**

## 문제 해결 팁

1. **컴파일 진행률이 50% 근처에서 멈추면**: 타입 캐스팅 패치가 제대로 적용되었는지 확인
2. **100%에서 patchelf 오류**: patchelf 설치 후 다시 `make` 실행
3. **의존성 오류 반복**: 가상환경을 새로 만들고 모든 의존성을 미리 설치
4. **메모리 부족**: `-j` 옵션의 숫자를 줄여서 재시도 (예: `-j2` 또는 `-j1`)

## 참고 자료

- PaddlePaddle GitHub: https://github.com/PaddlePaddle/Paddle
- 원본 이슈 및 해결 과정: GitHub Issue 토론 참조
