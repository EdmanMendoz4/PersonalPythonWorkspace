from ultralytics import RTDETR
from dataclasses import dataclass
import os
import cv2
import requests
import sys
import csv
import time

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
    t0 = time.perf_counter()
    img_encode = cv2.imencode(".jpg", img)[1].tobytes()
    try:
        
        response = requests.post(
            url, files={"image": (filename, img_encode, "image/jpeg")}
        )
        response.raise_for_status()
        t1 = time.perf_counter()
    except requests.exceptions.RequestException as e:
        sys.exit(f"Request failed: {e}")

    roundtrip_s = t1 - t0
    response_data = response.json()
    data = response_data.get("response", {})

    # Parse the response data to extract bounding boxes and labels
    if isinstance(data, dict):
        for (category, objects) in data.items():
            if isinstance(objects, list):
                for obj in objects:
                    points = obj.get(f"{'bounding_box'}", [])
                    conf = obj.get(f"{'confidence'}", [])
                    if (len(points) == 4): 
                        detections.append(Detection(bbox=points, confidence=conf, label=category, inference_source="Remote API", inference_time=roundtrip_s,file = filename))

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

def draw_and_save_detections(img, filename, detections, output_dir):
    """
    Draw detections for a single image and save it to output_dir with same filename.
    Color mapping (BGR): Original Label -> red, Remote API -> green, Ultralytics RT-DETR -> blue
    Label placement:
      - Remote API: right of the rectangle
      - Ultralytics RT-DETR: left of the rectangle
      - Original Label: under the rectangle
    """
    color_map = {
        "Original Label": (0, 0, 255),      # red (B, G, R)
        "Remote API": (0, 255, 0),          # green
        "Ultralytics RT-DETR": (255, 0, 0)  # blue
    }
    img_out = img.copy()
    h_img, w_img = img_out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    pad = 4  # padding around text background

    for d in detections:
        # ensure bbox are ints and clamp
        x1, y1, x2, y2 = map(int, d.bbox)
        x1 = max(0, min(x1, w_img - 1))
        y1 = max(0, min(y1, h_img - 1))
        x2 = max(0, min(x2, w_img - 1))
        y2 = max(0, min(y2, h_img - 1))

        color = color_map.get(d.inference_source, (255, 255, 255))  # default white
        # draw bbox
        cv2.rectangle(img_out, (x1, y1), (x2, y2), color, thickness=1)

        # prepare label text
        if d.confidence is None:
            label_text = f"{d.label}"
        else:
            try:
                label_text = f"{d.label} {d.confidence:.2f}"
            except Exception:
                label_text = f"{d.label} {d.confidence}"

        (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
        box_h = text_h + baseline + pad * 2
        box_w = text_w + pad * 2

        # decide text origin based on source
        if d.inference_source == "Remote API":
            tx = x2 + 5
            ty = y1 + text_h + pad
            # clamp to image right/bottom
            if tx + box_w > w_img:
                tx = max(0, w_img - box_w - 1)
            if ty + baseline > h_img:
                ty = max(text_h + pad, h_img - baseline - 1)
        elif d.inference_source == "Ultralytics RT-DETR":
            tx = x1 - box_w - 5
            ty = y1 + text_h + pad
            if tx < 0:
                tx = 1
            if ty + baseline > h_img:
                ty = max(text_h + pad, h_img - baseline - 1)
        else:  # Original Label or default -> under the rectangle
            tx = x1
            ty = y2 + text_h + pad + 5
            if ty + baseline > h_img:
                # place above if no space below
                ty = max(text_h + pad, y1 - 5)
                if ty + baseline > h_img:
                    ty = h_img - baseline - 1
            if tx + box_w > w_img:
                tx = max(1, w_img - box_w - 1)

        # draw background rectangle for text (dark background)
        bg_tl = (int(tx - pad), int(ty - text_h - pad))
        bg_br = (int(tx + text_w + pad), int(ty + baseline + pad))
        # clamp bg coords
        bg_tl = (max(0, bg_tl[0]), max(0, bg_tl[1]))
        bg_br = (min(w_img - 1, bg_br[0]), min(h_img - 1, bg_br[1]))
        cv2.rectangle(img_out, bg_tl, bg_br, (0, 0, 0), thickness=-1)
        # put text using the same color as the rectangle for consistency
        cv2.putText(img_out, label_text, (int(tx), int(ty)), font, font_scale, color, thickness, cv2.LINE_AA)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    cv2.imwrite(out_path, img_out)

def write_evaluation_csv(rows, output_dir, filename="evaluation.csv"):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "label", "source", "gt_count", "pred_count", "matched_count", "mean_iou", "false_negatives", "false_positives"])
        for r in rows:
            writer.writerow([r["file"], r["label"], r["source"], r["gt_count"], r["pred_count"], r["matched_count"], f"{r['mean_iou']:.4f}", r["false_negatives"], r["false_positives"]])

def interactive_viewer(detections, images_dir, output_dir=None, start_idx=0):
    """
    Simple interactive visualizer using OpenCV window and keyboard toggles.
    Keys:
      n / Right Arrow : next image
      p / Left Arrow  : previous image
      1 : toggle Original Label
      2 : toggle Remote API
      3 : toggle Ultralytics RT-DETR
      l : toggle labels text on/off
      s : save current overlay to output_dir (if provided)
      h : show help
      q / Esc : quit
    Call after detections list is populated.
    """
    # prepare mapping file -> detections
    from collections import defaultdict
    det_map = defaultdict(list)
    for d in detections:
        det_map[d.file].append(d)

    # list image files
    imgs = sorted([f for f in os.listdir(images_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    if not imgs:
        print("No images found in", images_dir)
        return

    idx = max(0, min(start_idx, len(imgs)-1))
    show_orig = True
    show_api = True
    show_ultra = True
    show_labels = True
    show_help = True

    winname = "Detections Viewer (h=help)"
    cv2.namedWindow(winname, cv2.WINDOW_NORMAL)

    def render_image(img_path):
        img = cv2.imread(os.path.join(images_dir, img_path))
        if img is None:
            return None
        over = img.copy()
        for d in det_map.get(img_path, []):
            src = d.inference_source
            if src == "Original Label" and not show_orig:
                continue
            if src == "Remote API" and not show_api:
                continue
            if src == "Ultralytics RT-DETR" and not show_ultra:
                continue
            # draw bbox
            try:
                x1, y1, x2, y2 = map(int, d.bbox)
            except Exception:
                continue
            color = (0,0,255) if src=="Original Label" else ((0,255,0) if src=="Remote API" else (255,0,0))
            cv2.rectangle(over, (x1, y1), (x2, y2), color, 2)
            if show_labels:
                txt = d.label if d.confidence is None else f"{d.label} {d.confidence:.2f}"
                # place labels similar to draw_and_save_detections but simpler here
                if src == "Remote API":
                    tx, ty = x2 + 5, y1 + 12
                elif src == "Ultralytics RT-DETR":
                    tx, ty = max(1, x1 - 100), y1 + 12
                else:
                    tx, ty = x1, y2 + 14
                (tw, th), baseline = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                bg_tl = (max(0, tx-2), max(0, ty-th-2))
                bg_br = (min(over.shape[1]-1, tx+tw+2), min(over.shape[0]-1, ty+baseline+2))
                cv2.rectangle(over, bg_tl, bg_br, (0,0,0), -1)
                cv2.putText(over, txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        # overlay small HUD
        hud = f"{idx+1}/{len(imgs)} {imgs[idx]}  |  1:Orig({int(show_orig)}) 2:API({int(show_api)}) 3:Ultra({int(show_ultra)}) l:labels({int(show_labels)}) n/p:nav s:save q:quit"
        cv2.rectangle(over, (5,5), (5+min(1200, int(len(hud)*7.5)), 30), (0,0,0), -1)
        cv2.putText(over, hud, (8,24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1, cv2.LINE_AA)
        if show_help:
            help_text = ["Keys:", "n/p: next/prev  1:toggle original  2:toggle api  3:toggle ultralytics",
                         "l: toggle labels  s: save overlay  h: toggle this help  q: quit"]
            y0 = 40
            for i, line in enumerate(help_text):
                cv2.putText(over, line, (8, y0 + i*16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220,220,220), 1, cv2.LINE_AA)
        return over

    while True:
        imgfile = imgs[idx]
        overlay = render_image(imgfile)
        if overlay is None:
            print("Failed to read", imgfile)
            idx = (idx + 1) % len(imgs)
            continue
        cv2.imshow(winname, overlay)
        key = cv2.waitKey(0) & 0xFF
        if key == 27 or key == ord('q'):  # Esc or q
            break
        elif key == ord('n') or key == 83:  # 'n' or right arrow
            idx = (idx + 1) % len(imgs)
        elif key == ord('p') or key == 81:  # 'p' or left arrow
            idx = (idx - 1) % len(imgs)
        elif key == ord('1'):
            show_orig = not show_orig
        elif key == ord('2'):
            show_api = not show_api
        elif key == ord('3'):
            show_ultra = not show_ultra
        elif key == ord('l'):
            show_labels = not show_labels
        elif key == ord('h'):
            show_help = not show_help
        elif key == ord('s'):
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                outp = os.path.join(output_dir, imgfile)
                # render once with current toggles and save
                cv2.imwrite(outp, render_image(imgfile))
                print("Saved", outp)
            else:
                print("No output_dir provided to save overlays.")
        # loop continues

    cv2.destroyWindow(winname)

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

                # Draw and save detections for this image (filters detections by filename)
                curr_detections = [d for d in detections if d.file == file]
                draw_and_save_detections(img, file, curr_detections, os.path.join(output_path, "detections/"))
                
    write_detections_to_csv(detections, output_path)

    # Evaluate IoU per file/label between predictions and ground truth
    eval_rows = evaluate_detections(detections, iou_threshold=0.0)  # set threshold e.g. 0.5 if you want TP only when IoU>=0.5
    write_evaluation_csv(eval_rows, output_path, "evaluation.csv")
    print(f"Wrote evaluation for {len(eval_rows)} file/label/source entries to {os.path.join(output_path, 'evaluation.csv')}")

    # Optional: launch interactive viewer (set to True to auto-open)
    LAUNCH_VIEWER = True
    if LAUNCH_VIEWER:
        images_dir = os.path.join(root_path, "images")
        interactive_viewer(detections, images_dir, output_dir=output_path)