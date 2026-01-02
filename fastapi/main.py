import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Query, Body
from fastapi.responses import JSONResponse, FileResponse
import pandas as pd
import numpy as np
from tqdm import tqdm
import evaluate
import uvicorn
import os
from pydantic import BaseModel
import time
import traceback

app = FastAPI(
    title="OCR GT Parser API",
    description="API для парсинга и анализа данных из Label Studio",
    version="1.0.0"
)

# Конфигурация путей
BASE_DIR = Path("/app").resolve()  # Изменено для Docker
PRED_PATH = BASE_DIR / "paddle_ocr_results.json"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# ГЛОБАЛЬНАЯ ПЕРЕМЕННАЯ вместо файла labelstudio_gt.json
labelstudio_data: List[Dict[str, Any]] = []
data_metadata: Dict[str, Any] = {
    "loaded_at": None,
    "source": None,
    "tasks_count": 0,
    "size_bytes": 0
}

print(f"Docker container starting...")
print(f"BASE_DIR: {BASE_DIR}")
print(f"OUTPUT_DIR: {OUTPUT_DIR}")
print(f"Global data storage initialized")

# Pydantic модель для данных
class LabelStudioData(BaseModel):
    """Модель для данных Label Studio"""
    data: List[Dict[str, Any]]

@app.get("/")
async def root():
    """Корневой endpoint"""
    return {
        "message": "OCR GT Parser API",
        "endpoints": {
            "/health": "GET - Проверка работоспособности",
            "/load-label-studio-data": "POST - Загрузить данные Label Studio",
            "/get-label-studio-data": "GET - Получить текущие данные",
            "/parse-ocr-gt": "GET - Парсинг GT данных",
            "/upload-file": "POST - Загрузить файл",
            "/clear-data": "DELETE - Очистить данные",
            "/list-outputs": "GET - Список файлов в outputs"
        },
        "docker": True,
        "container_id": os.getenv("HOSTNAME", "unknown"),
        "data_status": {
            "loaded": len(labelstudio_data) > 0,
            "tasks_count": len(labelstudio_data),
            "last_loaded": data_metadata.get("loaded_at")
        }
    }

@app.get("/health")
async def health_check():
    """Проверка работоспособности сервиса"""
    return {
        "status": "healthy",
        "base_dir": str(BASE_DIR),
        "output_dir": str(OUTPUT_DIR),
        "data_status": {
            "loaded": len(labelstudio_data) > 0,
            "tasks_count": len(labelstudio_data),
            "last_loaded": data_metadata.get("loaded_at"),
            "source": data_metadata.get("source")
        },
        "port": os.getenv("PORT", 8000),
        "container": True
    }

@app.post("/load-label-studio-data")
async def load_label_studio_data(
    data: List[Dict[str, Any]] = Body(
        ...,
        example=[
            {
                "data": {
                    "image": "page_001.png"
                },
                "annotations": [
                    {
                        "result": [
                            {
                                "value": {
                                    "x": 10.0,
                                    "y": 15.0,
                                    "width": 80.0,
                                    "height": 8.0,
                                    "text": "РЕЗУЛЬТАТЫ ЛАБОРАТОРНЫХ ИССЛЕДОВАНИЙ",
                                    "labels": ["table_header"]
                                }
                            }
                        ]
                    }
                ]
            }
        ]
    )
):
    """
    Загрузить данные Label Studio в глобальную переменную
    
    Принимает JSON массив с данными Label Studio
    """
    global labelstudio_data, data_metadata
    
    try:
        # Сохраняем данные
        labelstudio_data = data
        
        # Обновляем метаданные
        data_metadata = {
            "loaded_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "source": "POST /load-label-studio-data",
            "tasks_count": len(data),
            "size_bytes": len(json.dumps(data).encode('utf-8')),
            "sample_task_ids": [task.get('id', f"task_{i}") for i, task in enumerate(data[:3])]
        }
        
        return {
            "status": "success",
            "message": f"Данные успешно загружены в память",
            "metadata": data_metadata,
            "data_preview": {
                "first_task": data[0] if len(data) > 0 else None,
                "total_tasks": len(data)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка загрузки данных: {str(e)}")

@app.get("/parse-ocr-gt")
async def parse_ocr_gt():
    """Парсинг OCR GT данных из глобальной переменной"""
    global labelstudio_data
    try:
        return parse_ls_ocr_gt(labelstudio_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/forward")
async def forward(data: List[Dict[str, Any]] = Body(...)):
    """
    Загрузить данные Label Studio и получить спарсенный результат с метриками
    
    Принимает JSON массив с данными Label Studio
    Возвращает сопоставленные GT и предсказания с метриками детекции
    """
    global labelstudio_data
    
    try:        
        labelstudio_data = data
        
        # Пытаемся выполнить парсинг GT данных
        try:
            df_gt = parse_ls_ocr_gt(labelstudio_data)
        except Exception as e:
            raise HTTPException(
                status_code=403,
                detail="модель не смогла обработать данные"
            )
        
        df_pred = None
        if PRED_PATH.exists():
            try:
                df_pred = load_paddle_predictions(PRED_PATH)
            except Exception as e:
                print(f"Ошибка загрузки предсказаний: {e}")
                df_pred = pd.DataFrame()
        else:
            df_pred = pd.DataFrame()
            print(f"Файл предсказаний не найден: {PRED_PATH}")
        
        # Пытаемся построить матчи если есть предсказания
        try:
            if not df_pred.empty and not df_gt.empty:
                df_match = build_matched_pairs(df_gt, df_pred, iou_threshold=0.5)
            else:
                df_match = pd.DataFrame()
        except Exception as e:
            # Если модель не смогла построить матчи
            raise HTTPException(
                status_code=403,
                detail="модель не смогла обработать данные"
            )
        
        # Расчет метрик детекции (ВАЖНО: в основном ответе)
        try:
            if not df_match.empty:
                # Расчет TP, FP, FN
                tp = (df_match["match_type"] == "tp").sum()
                fp = (df_match["match_type"] == "fp").sum()
                fn = (df_match["match_type"] == "fn").sum()
                
                # Расчет precision, recall, F1
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                
                if precision + recall > 0:
                    f1 = 2 * precision * recall / (precision + recall)
                else:
                    f1 = 0.0
                
                # Расчет среднего IoU для TP
                iou_tp = df_match.loc[df_match["match_type"] == "tp", "iou"]
                mean_iou = float(iou_tp.mean()) if not iou_tp.empty else 0.0
                
                # Общее количество предсказаний и GT
                total_predictions = tp + fp
                total_gt = tp + fn
                
                # Дополнительные метрики
                detection_rate = tp / total_gt if total_gt > 0 else 0.0
                false_positive_rate = fp / (fp + tp) if (fp + tp) > 0 else 0.0
                
                # Формирование объекта метрик
                detection_metrics = {
                    "tp": int(tp),
                    "fp": int(fp),
                    "fn": int(fn),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                    "mean_iou_tp": float(mean_iou),
                    "total_predictions": int(total_predictions),
                    "total_gt": int(total_gt),
                    "detection_rate": float(detection_rate),
                    "false_positive_rate": float(false_positive_rate),
                    "metrics_summary": {
                        "excellent": f1 >= 0.9,
                        "good": f1 >= 0.7 and f1 < 0.9,
                        "fair": f1 >= 0.5 and f1 < 0.7,
                        "poor": f1 < 0.5
                    }
                }
            else:
                detection_metrics = {
                    "tp": 0,
                    "fp": 0,
                    "fn": 0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "mean_iou_tp": 0.0,
                    "total_predictions": 0,
                    "total_gt": 0,
                    "detection_rate": 0.0,
                    "false_positive_rate": 0.0,
                    "metrics_summary": {
                        "excellent": False,
                        "good": False,
                        "fair": False,
                        "poor": True
                    }
                }
        except Exception as e:
            print(f"Ошибка расчета метрик: {e}")
            raise HTTPException(
                status_code=403,
                detail="модель не смогла обработать данные"
            )
            
        
        # Сохраняем результаты
        output_file = OUTPUT_DIR / f"forward_result_{int(time.time())}.json"
        
        # Подготавливаем ОСНОВНОЙ ответ с метриками
        result = {
            "status": "success",
            "gt_data": {
                "total_boxes": len(df_gt),
                "data_head": df_gt.head().to_dict(orient="records") if not df_gt.empty else [],
                "columns": list(df_gt.columns) if not df_gt.empty else []
            },
            "pred_data": {
                "available": not df_pred.empty,
                "total_predictions": len(df_pred),
                "data_head": df_pred.head().to_dict(orient="records") if not df_pred.empty else []
            },
            "matches": {
                "available": not df_match.empty,
                "total_matches": len(df_match),
                "tp_count": len(df_match[df_match["match_type"] == "tp"]) if not df_match.empty else 0,
                "fn_count": len(df_match[df_match["match_type"] == "fn"]) if not df_match.empty else 0,
                "fp_count": len(df_match[df_match["match_type"] == "fp"]) if not df_match.empty else 0,
                "data_head": df_match.head().to_dict(orient="records") if not df_match.empty else []
            },
            # МЕТРИКИ ВКЛЮЧЕНЫ В ОСНОВНОЙ ОТВЕТ
            "tp": int(detection_metrics["tp"]),
            "fp": int(detection_metrics["fp"]),
            "fn": int(detection_metrics["fn"]),
            "precision": float(detection_metrics["precision"]),
            "recall": float(detection_metrics["recall"]),
            "f1": float(detection_metrics["f1"]),
            "mean_iou_tp": float(detection_metrics["mean_iou_tp"]),
            "detection_metrics": detection_metrics,  # Полный объект метрик
            "output_file": str(output_file),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        }
        
        # Сохраняем в файл
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Ошибка сохранения файла: {e}")
        
        return result
        
    except HTTPException:
        raise  # Пробрасываем HTTPException как есть
    except Exception as e:
        # Непредвиденная ошибка
        print(f"Unexpected error in /forward: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error"
        )


# Загрузка предсказаний PaddleOCR
def load_paddle_predictions(path):
    """Загрузка JSON с предсказаниями OCR/детектора.

    Ожидается список объектов:
      {
        "image_id": str,
        "boxes": [[x_min, y_min, x_max, y_max], ...],
        "texts": [str, str, ...]
      }
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for item in data:
        image_id = item.get("image_id")
        boxes = item.get("boxes", [])
        texts = item.get("texts", [])
        for bbox, text in zip(boxes, texts):
            rows.append(
                {
                    "image_id": image_id,
                    "pred_bbox": [float(v) for v in bbox],
                    "pred_text": str(text),
                }
            )

    df_pred = pd.DataFrame(rows)
    return df_pred

def box_iou(box_a, box_b):
    """IoU для двух боксов [x_min, y_min, x_max, y_max]."""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)

    union_area = area_a + area_b - inter_area
    if union_area <= 0:
        return 0.0

    return inter_area / union_area

def match_boxes_for_image(gt_boxes, pred_boxes, iou_threshold=0.5):
    gt_n = len(gt_boxes)
    pred_n = len(pred_boxes)

    if gt_n == 0 or pred_n == 0:
        return [], [], list(range(gt_n)), list(range(pred_n))

    iou_matrix = np.zeros((gt_n, pred_n), dtype=float)
    for i in range(gt_n):
        for j in range(pred_n):
            iou_matrix[i, j] = box_iou(gt_boxes[i], pred_boxes[j])

    used_gt = set()
    used_pred = set()
    matches = []
    match_ious = []

    flat_indices = np.argsort(iou_matrix.ravel())[::-1]  # по убыванию IoU

    for idx in flat_indices:
        i = idx // pred_n
        j = idx % pred_n
        if i in used_gt or j in used_pred:
            continue
        iou = iou_matrix[i, j]
        if iou < iou_threshold:
            break
        used_gt.add(i)
        used_pred.add(j)
        matches.append((i, j))
        match_ious.append(iou)

    unused_gt = [i for i in range(gt_n) if i not in used_gt]
    unused_pred = [j for j in range(pred_n) if j not in used_pred]

    return matches, match_ious, unused_gt, unused_pred

# Построение объединённой таблицы matched GT–pred
def build_matched_pairs(df_gt, df_pred, iou_threshold=0.5):
    """Построить DataFrame с матчингом GT и pred-боксов по image_id и IoU."""
    rows = []

    for image_id, df_gt_img in tqdm(df_gt.groupby("image_id"), desc="Matching images"):
        df_pred_img = df_pred[df_pred["image_id"] == image_id]

        gt_boxes = df_gt_img["gt_bbox"].tolist()
        pred_boxes = df_pred_img["pred_bbox"].tolist()

        matches, match_ious, unused_gt, unused_pred = match_boxes_for_image(
            gt_boxes, pred_boxes, iou_threshold=iou_threshold
        )

        gt_indices = df_gt_img.index.to_list()
        pred_indices = df_pred_img.index.to_list()

        # TP пары
        for (i_gt, j_pred), iou in zip(matches, match_ious):
            gt_row = df_gt_img.loc[gt_indices[i_gt]]
            pred_row = df_pred_img.loc[pred_indices[j_pred]]

            rows.append(
                {
                    "image_id": image_id,
                    "gt_bbox": gt_row["gt_bbox"],
                    "pred_bbox": pred_row["pred_bbox"],
                    "iou": float(iou),
                    "gt_text": gt_row["gt_text"],
                    "pred_text": pred_row["pred_text"],
                    "region_type": gt_row.get("region_type", None),
                    "match_type": "tp",
                }
            )

        # FN: GT без предсказаний
        for i_gt in unused_gt:
            gt_row = df_gt_img.loc[gt_indices[i_gt]]
            rows.append(
                {
                    "image_id": image_id,
                    "gt_bbox": gt_row["gt_bbox"],
                    "pred_bbox": None,
                    "iou": 0.0,
                    "gt_text": gt_row["gt_text"],
                    "pred_text": "",
                    "region_type": gt_row.get("region_type", None),
                    "match_type": "fn",
                }
            )

        # FP: предсказания без GT
        for j_pred in unused_pred:
            pred_row = df_pred_img.loc[pred_indices[j_pred]]
            rows.append(
                {
                    "image_id": image_id,
                    "gt_bbox": None,
                    "pred_bbox": pred_row["pred_bbox"],
                    "iou": 0.0,
                    "gt_text": "",
                    "pred_text": pred_row["pred_text"],
                    "region_type": None,
                    "match_type": "fp",
                }
            )

    df_match = pd.DataFrame(rows)
    return df_match

@app.get("/data-stats")
async def get_data_stats():
    """Статистика по загруженным данным"""
    global labelstudio_data
    
    if len(labelstudio_data) == 0:
        return {
            "status": "empty",
            "message": "Данные не загружены",
            "total_tasks": 0
        }
    
    # Анализируем структуру данных
    annotation_counts = []
    result_counts = []
    image_ids = set()
    
    for task in labelstudio_data:
        # Считаем аннотации
        annotations = task.get("annotations", [])
        annotation_counts.append(len(annotations))
        
        # Считаем результаты
        for ann in annotations:
            results = ann.get("result", [])
            result_counts.append(len(results))
        
        # Собираем image_id
        data = task.get("data", {})
        image_id = data.get("image") or data.get("image_id") or data.get("file")
        if image_id:
            image_ids.add(image_id)
    
    return {
        "status": "loaded",
        "metadata": data_metadata,
        "statistics": {
            "total_tasks": len(labelstudio_data),
            "unique_images": len(image_ids),
            "total_annotations": sum(annotation_counts),
            "total_results": sum(result_counts),
            "avg_annotations_per_task": sum(annotation_counts) / len(annotation_counts) if annotation_counts else 0,
            "avg_results_per_annotation": sum(result_counts) / len(result_counts) if result_counts else 0
        }
    }

def parse_ls_ocr_gt(ls_data):
    """Парсинг GT данных из Label Studio"""
    rows = []

    for task in ls_data:
        data = task.get("data", {})
        image_id = data.get("image") or data.get("image_id") or data.get("file")
        annotations = task.get("annotations") or []
        if len(annotations) == 0:
            continue

        ann = annotations[0]
        results = ann.get("result", [])

        for res in results:
            value = res.get("value", {})

            x = value.get("x")
            y = value.get("y")
            w = value.get("width")
            h = value.get("height")
            if x is None or y is None or w is None or h is None:
                continue

            text = value.get("text") or value.get("transcription") or ""
            labels = value.get("labels") or []
            region_type = labels[0] if len(labels) > 0 else None

            # Простейшее приведение к [x_min, y_min, x_max, y_max]
            x_min = float(x)
            y_min = float(y)
            x_max = float(x + w)
            y_max = float(y + h)
            bbox = [x_min, y_min, x_max, y_max]

            rows.append({
                "image_id": image_id,
                "gt_bbox": bbox,
                "gt_text": str(text),
                "region_type": region_type,
            })
    
    return pd.DataFrame(rows)

# %% Расчёт детекционных метрик
def detection_metrics_from_matched(df_match):
    tp = (df_match["match_type"] == "tp").sum()
    fp = (df_match["match_type"] == "fp").sum()
    fn = (df_match["match_type"] == "fn").sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    iou_tp = df_match.loc[df_match["match_type"] == "tp", "iou"]
    mean_iou = float(iou_tp.mean()) if not iou_tp.empty else 0.0

    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mean_iou_tp": mean_iou,
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    print(f"Starting server on port {port}...")
    print(f"Global data storage ready. Use POST /load-label-studio-data to load data.")
    uvicorn.run(
        app,

        host="0.0.0.0",
        port=port
    )