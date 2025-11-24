# PaddlePaddle GPU on NVIDIA Jetson JetPack 6.x – Debug & Build Notes

> 목적: Jetson 환경에서 PaddlePaddle GPU 빌드/디버깅 내용 정리

---

## 1. 공유된 Wheel 파일 정보

### 1.1. 원본 공유 (Baidu Pan, 현재 만료)

- 파일명: `paddlepaddle_gpu-0.0.0-cp310-cp310-linux_aarch64.whl`
- 공유자: `@fangfangssj`
- 환경:
  - Python: 3.10  
  - Paddle: `dev20250720`  
  - CUDA: 12.6  
  - OS: JetPack 6.2.1  
  - Device: Jetson Orin Nano  
- Baidu 링크 (만료됨):
  - `https://pan.baidu.com/s/1ngFAMd7udXLzZNLPT6zdyg`
  - 코드: `7q9q`

### 1.2. 이후 재업로드 (Google Drive)

Baidu Pan 사용이 어려운 사용자 요청으로 Google Drive에 재업로드됨:

- Google Drive 링크:  
  `https://drive.google.com/file/d/1WFoWcdqxlotcjEFq19hFDKZIRUPXUulx/view?usp=sharing`

---

## 2. 기본 빌드 환경 & 전제 조건

### 2.1. 환경 정보

- Python: 3.10  
- Paddle: `dev20250720` (또는 `release/3.1.1` 브랜치)  
- CUDA: 12.6  
- OS: JetPack 6.2.1  
- Device: Jetson Orin Nano  

### 2.2. CMake 버전

- 확인된 동작 버전: **3.22.1**
- Paddle은 현재 **CMake 4.0 이상 버전을 지원하지 않음**
  - 빌드 에러 발생 시 CMake 버전 확인 필요

---

## 3. @fangfangssj 의 빌드 커맨드 (JetPack 6.2.1)

```bash
git clone https://github.com/PaddlePaddle/Paddle.git
cd Paddle

mkdir -p build && cd ./build

cmake .. -DPY_VERSION=3.10   -DWITH_MKL=OFF   -DWITH_TESTING=OFF   -DCMAKE_BUILD_TYPE=Release   -DON_INFER=ON   -DWITH_PYTHON=ON   -DWITH_XBYAK=OFF   -DWITH_NV_JETSON=ON   -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda   -DWITH_NCCL=OFF   -DWITH_RCCL=OFF   -DWITH_DISTRIBUTE=OFF   -DWITH_GPU=ON   -DWITH_TENSORRT=ON   -DWITH_ARM=ON

ulimit -n 65535 && make TARGET=ARMV8 -j3

pip install python/dist/paddlepaddle_gpu-0.0.0-cp310-cp310-linux_aarch64.whl
```

---

## 4. `release/3.1.1` 브랜치 빌드 시 문제 & 해결

아래 내용은 `release/3.1.1` 브랜치 체크아웃 후 Jetson에서 직접 빌드하며 기록한 디버깅 로그.

### 4.1. 빌드 명령 (요약)

```bash
git clone https://github.com/PaddlePaddle/Paddle.git
cd Paddle
git checkout release/3.1.1

mkdir -p build && cd build

cmake .. <위와 유사한 옵션들>

make TARGET=ARMV8 -j3
```

빌드 약 3시간 후 여러 오류 발생.

---

## 5. 에러 1 – `ChangeThreadNum` 인자 타입 불일치

### 5.1. 에러 로그

```
argument of type "uint32_t *" is incompatible with parameter of type "int *"
```

### 5.2. 문제 원인
- `threads` = `uint32_t`
- `ChangeThreadNum()` 은 `int*` 요구  
- Jetson 전용 코드라 타입 충돌 발생

### 5.3. 해결

```cpp
backends::gpu::ChangeThreadNum(dev_ctx, (int*)(&threads), 256);
```

---

## 6. 에러 2 – `patchelf: not found`

### 6.1. 에러 로그

```
patchelf: not found
Exception: patch libpaddle.so failed
```

### 6.2. 해결

```bash
sudo apt install patchelf
```

---

## 7. 에러 3 – Python 의존성 부족

### 7.1. 에러 로그

```
Missing build dependency: httpx
```

### 7.2. 해결

```bash
python3 -m venv paddle_env
source paddle_env/bin/activate
pip install -r ../python/requirements.txt
```

---

## 8. 에러 4 – make 최종 오류

### 8.1. 해결

```bash
cd Paddle/build/python
python setup.py bdist_wheel
```

이때 부족한 패키지를 추가로 설치:

```bash
pip install pyyaml
pip install pybind11-stubgen
pip install wheel
```

---

## 9. TRT FP16 관련 참고

Jetson에서 TRT FP16 모델이 detection 실패하는 사례 보고됨  
→ Paddle 모드에서는 정상 작동  
→ 추후 TensorRT 변환 및 FP16 설정 다시 검토 필요

---

## 10. 최종 결과

- 위 수정/의존성 보완 후 wheel 빌드 성공
- GPU 실행 테스트 평균 속도: **약 50ms**

---

## 11. 빌드 오류 해결 체크리스트

1. CMake 버전 확인 (4.x 금지)
2. Jetson 전용 타입 에러 수정
3. `patchelf` 설치
4. Python req 설치 (`httpx`, `pyyaml`, `wheel`, `pybind11-stubgen`)
5. 필요 시 `bdist_wheel` 직접 실행

---
