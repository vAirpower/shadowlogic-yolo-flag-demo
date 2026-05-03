"""Export clean YOLOv8m FP32 ONNX from Ultralytics."""

from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

PT_PATH = MODELS / "yolov8m.pt"
ONNX_PATH = MODELS / "yolov8m_clean.onnx"


def main():
    model = YOLO(str(PT_PATH) if PT_PATH.exists() else "yolov8m.pt")

    if not PT_PATH.exists():
        downloaded = Path(model.ckpt_path) if hasattr(model, "ckpt_path") and model.ckpt_path else None
        if downloaded and downloaded.exists() and downloaded.resolve() != PT_PATH.resolve():
            downloaded.replace(PT_PATH)
        elif Path("yolov8m.pt").exists():
            Path("yolov8m.pt").replace(PT_PATH)

    exported = model.export(
        format="onnx",
        imgsz=640,
        opset=14,
        simplify=True,
        nms=False,
        dynamic=False,
        half=False,
    )

    exported_path = Path(exported)
    if exported_path.resolve() != ONNX_PATH.resolve():
        exported_path.replace(ONNX_PATH)

    print(f"Saved: {ONNX_PATH}")
    print(f"  size: {ONNX_PATH.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
