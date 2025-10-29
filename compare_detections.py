import os
import cv2
import time
import requests
from ultralytics import YOLO
import matplotlib.pyplot as plt

# ===============================
# ==== 1. LABELS FROM TXT =======
# ===============================

def detect_from_txt(images_path, labels_path, classes_file, output_path):
    os.makedirs(output_path, exist_ok=True)
    with open(classes_file, "r") as f:
        class_names = [line.strip() for line in f.readlines()]

    start_time = time.time()
    detections = {}  # store results per image

    for image_file in os.listdir(images_path):
        if not image_file.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img_path = os.path.join(images_path, image_file)
        label_file = os.path.splitext(image_file)[0] + ".txt"
        label_path = os.path.join(labels_path, label_file)

        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w, _ = img.shape
        boxes = []

        if os.path.exists(label_path):
            with open(label_path, "r") as lf:
                for line in lf.readlines():
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls_id = int(parts[0])
                    x_center, y_center, box_w, box_h = map(float, parts[1:])
                    x1 = int((x_center - box_w / 2) * w)
                    y1 = int((y_center - box_h / 2) * h)
                    x2 = int((x_center + box_w / 2) * w)
                    y2 = int((y_center + box_h / 2) * h)
                    label = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
                    boxes.append(((x1, y1, x2, y2), label))

                    # draw box
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(img, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        detections[image_file] = boxes
        cv2.imwrite(os.path.join(output_path, image_file), img)

    elapsed = time.time() - start_time
    return detections, elapsed

# ===============================
# ==== 2. DETECTION VIA API =====
# ===============================

def detect_from_api(images_path, url, output_path):
    os.makedirs(output_path, exist_ok=True)
    start_time = time.time()
    detections = {}

    for file in os.listdir(images_path):
        if not file.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        path = os.path.join(images_path, file)
        img = cv2.imread(path)
        img_encode = cv2.imencode(".jpg", img)[1].tobytes()
        try:
            response = requests.post(url, files={"image": (file, img_encode, "image/jpeg")})
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"API request failed for {file}: {e}")
            continue

        response_data = response.json()
        data = response_data.get("response", {})
        boxes = []

        if isinstance(data, dict):
            for category, objects in data.items():
                if isinstance(objects, list):
                    for obj in objects:
                        points = obj.get("bounding_box", [])
                        conf = obj.get("confidence", 0.0)
                        if len(points) == 4:
                            (x1, y1, x2, y2) = map(int, points)
                            boxes.append(((x1, y1, x2, y2), category))
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(img, f"{category} {conf:.2f}", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        detections[file] = boxes
        cv2.imwrite(os.path.join(output_path, file), img)

    elapsed = time.time() - start_time
    return detections, elapsed

# =======================================
# ==== 3. DETECTION VIA LOCAL MODEL =====
# =======================================

def detect_from_model(images_path, model_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    start_time = time.time()
    detections = {}

    model = YOLO(model_path)

    for file in os.listdir(images_path):
        if not file.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        path = os.path.join(images_path, file)
        results = model.predict(path, conf=0.7)
        #results = model.predict(path, conf=0.7, verbose=False)

        boxes = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cls_id = int(box.cls)
                label = model.names[cls_id]
                boxes.append(((x1, y1, x2, y2), label))
                img = r.plot()

        cv2.imwrite(os.path.join(output_path, file), img)
        detections[file] = boxes

    elapsed = time.time() - start_time
    return detections, elapsed

# ===============================
# ==== 4. COMPARISON PLOTS ======
# ===============================

def comparison_figures(images_path, results_txt, results_api, results_model):
    comp_dir_triplet = "C://Inferencias//Comparison//triplet"
    comp_dir_overlay = "C://Inferencias//Comparison//overlay"
    
    os.makedirs(comp_dir_triplet, exist_ok=True)
    os.makedirs(comp_dir_overlay, exist_ok=True)

    for file in results_txt.keys():
        img_path = os.path.join(images_path, file)
        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)

        # 1 Figure with three separate images
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(cv2.cvtColor(cv2.imread(os.path.join("C://Inferencias//Comparison//local_labeled_images", file)), cv2.COLOR_BGR2RGB))
        axs[0].set_title("Labels (.txt)")
        axs[1].imshow(cv2.cvtColor(cv2.imread(os.path.join("C://Inferencias//Comparison//Output_API", file)), cv2.COLOR_BGR2RGB))
        axs[1].set_title("API")
        axs[2].imshow(cv2.cvtColor(cv2.imread(os.path.join("C://Inferencias//Comparison//model_labeled_images", file)), cv2.COLOR_BGR2RGB))
        axs[2].set_title("Model (best.pt)")
        for ax in axs:
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(comp_dir_triplet, f"comparison_triplet_{file}.png"))
        plt.close()

        # 2️ Figure with all detections overlaid
        img_overlay = img.copy()

        for (rect, label) in results_txt[file]:
            x1, y1, x2, y2 = rect
            cv2.rectangle(img_overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)  # red
        for (rect, label) in results_api[file]:
            x1, y1, x2, y2 = rect
            cv2.rectangle(img_overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)  # green
        for (rect, label) in results_model[file]:
            x1, y1, x2, y2 = rect
            cv2.rectangle(img_overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)  # blue

        plt.figure(figsize=(6, 6))
        plt.imshow(cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB))
        plt.title("Overlay Comparison\nRed=TXT, Green=API, Blue=Model")
        plt.axis("off")
        plt.savefig(os.path.join(comp_dir_overlay, f"comparison_overlay_{file}.png"))
        plt.close()


# ===============================
# ==== 5. MAIN EXECUTION ========
# ===============================

if __name__ == "__main__":
    images_path = "C://Inferencias//test//images"
    labels_path = "C://Inferencias//test//labels"
    classes_file = "C://Inferencias//test//classes.txt"
    url = "http://192.168.99.142:8083/predict"
    model_path = "C://Inferencias//best.pt"

    output_txt = "C://Inferencias//Comparison//local_labeled_images"
    output_api = "C://Inferencias//Comparison//Output_API"
    output_model = "C://Inferencias//Comparison//model_labeled_images"

    print("Running TXT detection...")
    results_txt, time_txt = detect_from_txt(images_path, labels_path, classes_file, output_txt)
    print(f"TXT detection completed in {time_txt:.2f} seconds.\n")

    print("Running API detection...")
    results_api, time_api = detect_from_api(images_path, url, output_api)
    print(f"API detection completed in {time_api:.2f} seconds.\n")
    
    print("Running model detection...")
    results_model, time_model = detect_from_model(images_path, model_path, output_model)
    print(f"Model detection completed in {time_model:.2f} seconds.\n")

    print("Generating comparison figures...")
    comparison_figures(images_path, results_txt, results_api, results_model)

    print("\n=== SUMMARY ===")
    print(f"TXT (.txt) detection: {time_txt:.2f} s")
    print(f"API detection:        {time_api:.2f} s")
    print(f"Model (best.pt):      {time_model:.2f} s")
    print("All comparisons saved successfully!")
