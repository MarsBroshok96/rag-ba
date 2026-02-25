# Troubleshooting

## detectron2 / layoutparser
Symptoms:
- `ModuleNotFoundError: torchvision`
Fix:
- Ensure torch + torchvision installed in rag-ba env.
- Keep Python 3.11; avoid Poetry thinking project is 3.13.

## cv2 namespace module (cv2.__file__ None)
Symptoms:
- cv2 imported as namespace package
Fix:
- reinstall opencv-python inside the correct poetry env
- verify `python -c "import cv2; print(cv2.__file__)"`

## PaddleOCR GPU: libnvrtc missing
Symptoms:
- `ImportError: libnvrtc.so.12 ...`
Fix:
- Ensure paddlepaddle-gpu build matches CUDA runtime libraries available.
- If CUDA toolkit not installed, prefer using the wheel that bundles required libs or switch PaddleOCR to CPU build.

## PaddleOCR: protobuf missing
Symptoms:
- `ModuleNotFoundError: google.protobuf`
Fix:
- install `protobuf` in apps/rag-ba-ocr env.

## LlamaIndex: metadata length longer than chunk size
Symptoms:
- `ValueError: Metadata length (XXXX) is longer than chunk size (1024)`
Fix:
- keep metadata slimming in index build script OR increase chunk size in node parser.

## Ollama
Symptoms:
- Q/A fails or hangs
Fix:
- ensure `ollama serve` running and model pulled.

## Detectron2 installation fails with:
Symptoms:
ModuleNotFoundError: No module named 'torch'

during:
pip install 'git+https://github.com/facebookresearch/detectron2.git'

 Root Cause
pip uses PEP517 build isolation by default.
When installing detectron2, pip creates a temporary build environment (/tmp/pip-build-env-*) which does not see torch installed in the current venv.
detectron2 requires torch at build time, so installation fails.

Correct Installation Order (GPU / CUDA):

1️. Install PyTorch first (matching your CUDA)
poetry run pip install --upgrade pip
poetry run pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision torchaudio

2️. Install Detectron2 WITHOUT build isolation
poetry run pip install -U --no-build-isolation \
  'git+https://github.com/facebookresearch/detectron2.git'

If compilation fails (C++ / ninja errors)

3. Install system toolchain:
sudo apt update
sudo apt install -y build-essential python3-dev git
poetry run pip install -U ninja

Then retry:
poetry run pip install -U --no-build-isolation \
  'git+https://github.com/facebookresearch/detectron2.git'


## torch-gpu conflicts with paddle-gpu
Symptoms:
from torch._C import * # noqa: F403 ^^^^^^^^^^^^^^^^^^^^^^ ImportError: .../.../rag-ba-ocr/.venv/lib/python3.11/site-packages/torch/lib/libtorch_cuda.so: undefined symbol: ncclCommShrink
Fix:
poetry run pip uninstall -y torch torchvision torchaudio
poetry run pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch