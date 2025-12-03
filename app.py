# app.py


from __future__ import annotations

from pathlib import Path
from typing import List, Optional
from time import perf_counter
import io
import os
import base64
import tempfile
import torch


import numpy as np
import cv2
from PIL import Image
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

# === Настройки ===
BASE_DIR = Path(__file__).parent.resolve()
PUBLIC_DIR = BASE_DIR / "public"
MODEL_PATH = BASE_DIR / "best.pt"  # можно заменить на best.pt

print("Serving UI from:", PUBLIC_DIR)
print("Model path:", MODEL_PATH)

app = FastAPI(title="YOLOv8 API")

# CORS

@app.get("/ui", response_class=HTMLResponse)
def ui_redirect():
    return RedirectResponse(url="/ui/")

if PUBLIC_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(PUBLIC_DIR), html=True), name="ui")

@app.get("/", response_class=HTMLResponse)
def root():
    return ("<h1>YOLOv8 API</h1>"
            "<p>UI: <a href='/ui/'>/ui/</a> • Docs: <a href='/docs'>/docs</a></p>")

@app.get("/api/yolo/detect")
def detect_help():
    return JSONResponse({"hint": "POST multipart/form-data to /api/yolo/detect: file(image), conf(0..1), iou(0..1), imgsz(int), task(detect|segment), model(path). "})

# === Модель по умолчанию ===

# Автоматическое определение устройства:
# GPU (cuda:0) → если доступно (RTX 50xx, sm_120)
# CPU → если CUDA нет
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# Загружаем модель на нужное устройство
default_model = YOLO(str(MODEL_PATH)).to(DEVICE)
_model_cache = {}  # (path) -> YOLO


def get_model(path: str) -> YOLO:
    path = path.strip() if path else str(MODEL_PATH)
    full = str((BASE_DIR / path)) if not os.path.isabs(path) else path

    if full not in _model_cache:
        print(f"Loading model: {full} → {DEVICE}")
        _model_cache[full] = YOLO(full).to(DEVICE)

    return _model_cache[full]


# ------
def log_infer_params(kind: str, conf: float, iou: float, imgsz: int, model_path: str):
    print(f"[{kind}] model={model_path or MODEL_PATH} conf={conf} iou={iou} imgsz={imgsz}")



# === Схемы ответа (Pydantic) ===
from pydantic import BaseModel
from typing import List, Optional

class Box(BaseModel):
    x: float
    y: float
    w: float
    h: float
    conf: Optional[float] = None
    label: Optional[str] = None

class DetectResponse(BaseModel):
    width: int
    height: int
    boxes: List[Box]
    conf_used: float
    iou_used: float
    elapsed_ms: float
    # чтобы не ругалось при возврате с картинкой (return_image=1)
    image_base64: Optional[str] = None

# === /api/yolo/detect (картинка) ===
@app.post("/api/yolo/detect", response_model=DetectResponse)
async def detect(
    file: UploadFile = File(...),
    conf: float = Form(0.35),
    iou: float = Form(0.45),
    imgsz: int = Form(640),
    task: str = Form("detect"),          # 'detect' | 'segment'
    model: str = Form(""),               # путь к .pt (опц.)
    return_image: int = Form(0),         # 1 -> вернём картинку base64 (для UI)
):
    content = await file.read()
    image = Image.open(io.BytesIO(content)).convert("RGB")
    w, h = image.size

    m = get_model(model) if model else default_model
    log_infer_params("DETECT", conf, iou, imgsz, model)

    t0 = perf_counter()
    res = m.predict(source=image, conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0]
    elapsed = (perf_counter() - t0) * 1000.0

    names = getattr(m, "names", {}) or getattr(getattr(m, "model", None), "names", {}) or {}
    if not isinstance(names, dict):
        names = {int(i): str(n) for i, n in enumerate(names)} if hasattr(names, "__iter__") else {}

    boxes: List[Box] = []
    if res.boxes is not None and len(res.boxes) > 0:
        xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy() if res.boxes.conf is not None else [None] * len(xyxy)
        clss  = res.boxes.cls.cpu().numpy()  if res.boxes.cls  is not None else [None] * len(xyxy)
        for (x1, y1, x2, y2), c, cl in zip(xyxy, confs, clss):
            label = names.get(int(cl), str(int(cl))) if cl is not None else None
            boxes.append(Box(x=float(x1), y=float(y1), w=float(x2-x1), h=float(y2-y1),
                             conf=float(c) if c is not None else None, label=label))

    resp = DetectResponse(
        width=w, height=h, boxes=boxes,
        conf_used=float(conf), iou_used=float(iou), elapsed_ms=elapsed
    )

    if return_image:
        plotted = res.plot()
        ok, buf = cv2.cvtColor(plotted, cv2.COLOR_RGB2BGR)
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        # универсально для pydantic v1/v2
        try:
            payload = resp.model_dump()
        except AttributeError:
            payload = resp.dict()
        payload["image_base64"] = b64
        return JSONResponse(content=payload)

    return resp

# === /api/yolo/stream (один кадр для «демонстрации экрана») ===
@app.post("/api/yolo/stream")
async def stream(
    image: UploadFile = File(...),
    conf: float = Form(0.25),
    iou: float = Form(0.45),
    imgsz: int = Form(640),
    model: str = Form(""),
    return_image: int = Form(1),  # 1 — вернём размеченный кадр base64
):
    m = get_model(model) if model else default_model
    log_infer_params("STREAM", conf, iou, imgsz, model)

    data = np.frombuffer(await image.read(), np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    res = m.predict(frame, conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0]

    names = getattr(m, "names", {})
    boxes = []
    if res.boxes is not None and len(res.boxes) > 0:
        xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        clss  = res.boxes.cls.cpu().numpy()
        for (x1, y1, x2, y2), c, cl in zip(xyxy, confs, clss):
            label = names[int(cl)] if isinstance(names, dict) else str(int(cl))
            boxes.append({
                "xyxy": [int(x1), int(y1), int(x2), int(y2)],
                "cls": label,
                "conf": float(c),
            })

    payload = {"boxes": boxes, "conf_used": float(conf), "iou_used": float(iou)}
    if return_image:
        plotted = res.plot()  # RGB
        ok, buf = cv2.imencode(".jpg", plotted[:, :, ::-1])
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        payload["image_base64"] = b64

    return JSONResponse(payload)


# === /api/yolo/video (файл видео -> готовый mp4) ===
@app.post("/api/yolo/video")
async def video(
    video: UploadFile = File(...),
    conf: float = Form(0.25),
    iou: float = Form(0.7),
    imgsz: int = Form(640),
    model: str = Form(""),
    max_frames: int = Form(300),   # ограничение (важно)
):
    m = get_model(model) if model else default_model
    log_infer_params("VIDEO", conf, iou, imgsz, model)

    raw = await video.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
        f.write(raw)
        src_path = f.name

    cap = cv2.VideoCapture(src_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

    out_path = src_path.replace(".mp4", "_out.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_frames and n >= max_frames:
            break

        n += 1
        print(f"[VIDEO] Processing frame {n}/{max_frames}")

        # ⚡ CPU ускорение: без pyTorch gradients
        with torch.no_grad():
            res = m.predict(frame, conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0]

        plotted = res.plot()
        writer.write(plotted[:, :, ::-1])

    cap.release()
    writer.release()

    def stream_file():
        with open(out_path, "rb") as f:
            yield from f

    return StreamingResponse(stream_file(), media_type="video/mp4")



# info
def print_startup_info():
    print("\n================= YOLOv8 WEB SERVER =================")
    print(f"Model path : {MODEL_PATH}")
    print(f"Serving UI : {PUBLIC_DIR}")
    print(f"Default conf : 0.25")
    print(f"Default iou  : 0.45")
    print(f"Default imgsz: 640")
    print("------------------------------------------------------")
    print("Open:")
    print("  📸  Image API     → http://localhost:8000/api/yolo/detect")
    print("  🎥  Video API     → http://localhost:8000/api/yolo/video")
    print("  🧠  Stream API    → http://localhost:8000/api/yolo/stream")
    print("  🌐  Web UI        → http://localhost:8000/ui/")
    print("======================================================\n")

print_startup_info()



if __name__ == "__main__":
    import uvicorn
    # Запуск:  uvicorn app:app --reload --host 0.0.0.0 --port 8000
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
