from ultralytics import RTDETR
from dataclasses import dataclass
import os
import cv2
import requests
import sys
import csv

@dataclass
class Detection:
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    file: str                           # image filename
    label: str                                # detection label
    confidence: float                         # 0.0 - 1.0
    inference_source: str                    # e.g., "Ultralytics RT-DETR" or "Remote API"    
    inference_time: float = None    # seconds (same for all detections; optional)
    
def ultralytics_inference(model, imagen, filename):
    """
    Perform inference using the Ultralytics RT-DETR model and return detections.
    """
    # Run inference 
    results = model(imagen)

    # Prepare names mapping (model.names is provided by Ultralytics)
    names = getattr(model, "names", {})

    # Process results
    for obj in results:
        result = obj.numpy()  # keeps your original approach
        rectangles = result.boxes.xyxy  # bounding box coordinates (tensor/array)
        labels = result.boxes.cls        # class labels (tensor/array)
        confidences = result.boxes.conf  # confidence scores (tensor/array)
        inf_time = float(obj.speed['inference']) # inference time in ms

        # Ensure iteration works whether these are tensors or numpy arrays
        for i in range(len(rectangles)):
            bbox = tuple(map(int, rectangles[i].tolist()))
            conf = float(confidences[i])
            label_idx = int(labels[i])
            label_name = names.get(label_idx, str(label_idx))
            detections.append(Detection(bbox=bbox, confidence=conf, label=label_name, inference_source="Ultralytics RT-DETR", inference_time=inf_time, file=filename))  

def api_inference(img, url, filename):
    """ 
    Perform inference by sending the image to a remote API and return detections. 
    """  
    img_encode = cv2.imencode(".jpg", img)[1].tobytes()
    try:
        response = requests.post(
            url, files={"image": (file, img_encode, "image/jpeg")}
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        sys.exit(f"Request failed: {e}")
    response_data = response.json()
    data = response_data.get("response", {})
    
    # Parse the response data to extract bounding boxes and labels
    if isinstance(data, dict):
        for (
            category,
            objects,
        ) in data.items():
            if isinstance(objects, list):
                for obj in objects:
                    points = obj.get(f"{'bounding_box'}", [])
                    conf = obj.get(f"{'confidence'}", [])
                    if (len(points) == 4):  # Ensure only valid points are processed
                        detections.append(Detection(bbox=points, confidence=conf, label=category, inference_source="Remote API", file = filename))

def read_labels(label_path, filename, img_width, img_height):
    """ Read labels from a text file and return as a list of Detection objects. """
    label_file = os.path.join(label_path, os.path.splitext(filename)[0] + ".txt")

    # Mapping de IDs a nombres
    class_map = {
        0: "sperm_in_needle",
        1: "needle_tip",
        2: "meniscus",
    }

    if os.path.exists(label_file):
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id, x_center, y_center, width, height = map(float, parts)
                    # Convert from YOLO format to (x1, y1, x2, y2)
                    x1 = int((x_center - width / 2) * img_width)  
                    y1 = int((y_center - height / 2) * img_height)
                    x2 = int((x_center + width / 2) * img_width)
                    y2 = int((y_center + height / 2) * img_height)

                    label_name = class_map.get(int(class_id), str(int(class_id)))
                    detections.append(Detection(bbox=(x1, y1, x2, y2), label=label_name, confidence=1.0, inference_source="Original Label", file=filename))
    
def write_detections_to_csv(detections, output_dir, filename="detections.csv"):
    """Write list[Detection] to CSV (one row per detection)."""
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "label", "inference_source", "confidence", "x1", "y1", "x2", "y2", "inference_time"])
        for d in detections:
            # bbox may be tuple of ints
            x1, y1, x2, y2 = d.bbox
            writer.writerow([d.file, d.label, d.inference_source, d.confidence, x1, y1, x2, y2, d.inference_time])

def draw_rectangles_on_image(img, rectangles_with_labels, rectangle_color=(0, 255, 0), text_color=(0, 0, 255), thickness=2, sz=0.6, background_rectangle=True):
    font = cv2.FONT_HERSHEY_SIMPLEX
    for rect, ci, label in rectangles_with_labels:
        x1, y1, x2, y2 = rect
        img = cv2.rectangle(img, (x1, y1), (x2, y2), rectangle_color, thickness)
        text_x, text_y = x1, max(10, y1 - 10)
        text = f"{label}: {ci:.2f}"

        if background_rectangle == True:
            (text_width, text_height), baseline = cv2.getTextSize(
                text, font, sz, thickness
            )
            cv2.rectangle(
                img,
                (text_x, text_y - text_height - baseline),
                (text_x + text_width, text_y + baseline),
                rectangle_color,
                cv2.FILLED,
            )
        else:
            pass

        cv2.putText(
            img, text, (text_x, text_y), font, sz, text_color, thickness=thickness
        )
    return img

def compute_iou(boxA, boxB) -> float:
    """Compute IoU between two boxes (x1,y1,x2,y2)."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])

    if boxAArea + boxBArea - interArea == 0:
        return 0.0
    return interArea / (boxAArea + boxBArea - interArea)

def evaluate_detections(detections, iou_threshold=0.0):
    """
    Evaluate detections vs ground truth ("Original Label").
    Returns a list of dict rows with evaluation per (file,label,source):
      file, label, source, gt_count, pred_count, matched_count, mean_iou, false_negatives, false_positives
    Matching: greedy best-IoU matching (one GT matched once).
    iou_threshold: if >0, you can treat match only if IoU >= threshold (optional).
    """
    # Organize detections by file -> label -> source -> list[box]
    by_file = {}
    for d in detections:
        file = d.file
        label = d.label
        source = d.inference_source
        bbox = tuple(map(int, d.bbox))
        by_file.setdefault(file, {}).setdefault(label, {}).setdefault(source, []).append(bbox)

    rows = []
    for file, labels in by_file.items():
        for label, sources in labels.items():
            gt_boxes = sources.get("Original Label", [])
            if not gt_boxes:
                # No ground truth for this label in this file; skip evaluation for that label
                # but still report predictions as false positives for other sources
                for source, pred_boxes in sources.items():
                    if source == "Original Label":
                        continue
                    rows.append({
                        "file": file,
                        "label": label,
                        "source": source,
                        "gt_count": 0,
                        "pred_count": len(pred_boxes),
                        "matched_count": 0,
                        "mean_iou": 0.0,
                        "false_negatives": 0,
                        "false_positives": len(pred_boxes),
                    })
                continue

            # Evaluate each prediction source (exclude ground truth)
            for source, pred_boxes in sources.items():
                if source == "Original Label":
                    continue

                gt_unused = list(gt_boxes)[:]  # mutable copy
                pred_unused = list(pred_boxes)[:]
                matched_ious = []

                # Build IoU matrix
                iou_matrix = []
                for i, g in enumerate(gt_unused):
                    row = []
                    for j, p in enumerate(pred_unused):
                        row.append((compute_iou(g, p), i, j))
                    iou_matrix.extend(row)

                # Greedy matching by highest IoU
                iou_matrix.sort(key=lambda x: x[0], reverse=True)
                gt_matched_idx = set()
                pred_matched_idx = set()
                for iou_val, gi, pj in iou_matrix:
                    if gi in gt_matched_idx or pj in pred_matched_idx:
                        continue
                    if iou_val >= iou_threshold:
                        gt_matched_idx.add(gi)
                        pred_matched_idx.add(pj)
                        matched_ious.append(iou_val)

                matched_count = len(matched_ious)
                gt_count = len(gt_unused)
                pred_count = len(pred_unused)
                false_negatives = gt_count - matched_count
                false_positives = pred_count - matched_count
                mean_iou = float(sum(matched_ious) / matched_count) if matched_count > 0 else 0.0

                rows.append({
                    "file": file,
                    "label": label,
                    "source": source,
                    "gt_count": gt_count,
                    "pred_count": pred_count,
                    "matched_count": matched_count,
                    "mean_iou": mean_iou,
                    "false_negatives": false_negatives,
                    "false_positives": false_positives,
                })
    return rows

def write_evaluation_csv(rows, output_dir, filename="evaluation.csv"):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "label", "source", "gt_count", "pred_count", "matched_count", "mean_iou", "false_negatives", "false_positives"])
        for r in rows:
            writer.writerow([r["file"], r["label"], r["source"], r["gt_count"], r["pred_count"], r["matched_count"], f"{r['mean_iou']:.4f}", r["false_negatives"], r["false_positives"]])

if __name__ == "__main__":
    
    # Initialize list to hold Detection objects
    detections: list[Detection] = []                
    
    # Define api endpoint 
    url = "http://192.168.99.142:8083/predict"
    
    # Defines for the RT-DETR model inference
    modelpt = "C:/InferenceFilesTemporary/best.pt"
    model = RTDETR(modelpt)
    
    # Define image directory path and create output folder
    root_path = "C:/InferenceFilesTemporary/test"
    output_path = os.path.join(root_path, "output/")
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Process each image in the directory
    for root, _, files in os.walk(os.path.join(root_path, "images/")):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                path = os.path.join(root, file)
                img = cv2.imread(path)
                h, w = img.shape[:2]
                # Perform both API and Ultralytics inferences
                api_inference(img, url, file)
                ultralytics_inference(model, path, file)
                
                # Obtain original labels for comparison
                read_labels(os.path.join(root_path, "labels/"), file, w, h)
                
    write_detections_to_csv(detections, output_path)

    # Evaluate IoU per file/label between predictions and ground truth
    eval_rows = evaluate_detections(detections, iou_threshold=0.0)  # set threshold e.g. 0.5 if you want TP only when IoU>=0.5
    write_evaluation_csv(eval_rows, output_path, "evaluation.csv")
    print(f"Wrote evaluation for {len(eval_rows)} file/label/source entries to {os.path.join(output_path, 'evaluation.csv')}")