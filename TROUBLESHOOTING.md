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