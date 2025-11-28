# app.py — минимальный бэкенд для фронта из предыдущего файла
# FastAPI + Ultralytics YOLOv8. Отдаёт боксы в формате {x,y,w,h,conf,label} в пикселях исходного изображения.

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from PIL import Image
import io
import uvicorn

# 1) Загрузите модель один раз при старте
#    положите свой .pt рядом и укажите путь ниже
from ultralytics import YOLO
MODEL_PATH = "best.pt"  # <= замените на вашу обученную модель
model = YOLO(MODEL_PATH)

app = FastAPI(title="YOLOv8 API")

# Разрешим фронту обращаться (локально)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # при желании ограничьте доменом
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

@app.post("/api/yolo/detect", response_model=DetectResponse)
async def detect(file: UploadFile = File(...), conf: float = Form(0.5)):
    """Принимает изображение (jpeg/png), возвращает боксы.
    Аргументы:
      - file: бинарное изображение
      - conf: порог уверенности (0..1)
    Ответ: {width, height, boxes:[{x,y,w,h,conf,label}]}
    """
    content = await file.read()
    image = Image.open(io.BytesIO(content)).convert("RGB")

    # Запуск модели; вернём боксы в пикселях исходника
    results = model.predict(source=image, conf=conf, verbose=False)[0]
    w, h = image.size

    boxes_out: List[Box] = []
    names = model.model.names if hasattr(model, "model") else model.names

    # В results.boxes.xyxy — координаты [x1,y1,x2,y2]
    if results.boxes is not None and len(results.boxes) > 0:
        xyxy = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy() if results.boxes.conf is not None else [None]*len(xyxy)
        clss  = results.boxes.cls.cpu().numpy() if results.boxes.cls is not None else [None]*len(xyxy)

        for (x1, y1, x2, y2), c, cl in zip(xyxy, confs, clss):
            x, y = float(x1), float(y1)
            bw, bh = float(x2 - x1), float(y2 - y1)
            label = names[int(cl)] if cl is not None else None
            boxes_out.append(Box(x=x, y=y, w=bw, h=bh, conf=float(c) if c is not None else None, label=label))

    return DetectResponse(width=w, height=h, boxes=boxes_out)

if __name__ == "__main__":
    # Запуск:  uvicorn app:app --reload --host 0.0.0.0 --port 8000
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
