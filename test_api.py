import cv2
import requests
import sys
import os


def draw_rectangles_on_image(
    img,
    rectangles_with_labels,
    rectangle_color=(0, 255, 0),
    text_color=(0, 0, 255),
    thickness=2,
    sz=0.6,
    background_rectangle=True,
):
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


def process_images(image_path, url):
    if not os.path.exists("output/"):
        os.makedirs("output/")

    for root, _, files in os.walk(image_path):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                path = os.path.join(root, file)
                img = cv2.imread(path)
                img_encode = cv2.imencode(".jpg", img)[1].tobytes()
                try:
                    response = requests.post(
                        url, files={"image": (file, img_encode, "image/jpeg")}
                    )
                    response.raise_for_status()
                except requests.exceptions.RequestException as e:
                    sys.exit(f"Request failed: {e}")
                response_data = response.json()
                print(
                    "API Response:", response_data
                )  # Print the full API response for debugging
                data = response_data.get("response", {})
                rectangles_with_labels = []
                # Parse the response data to extract bounding boxes and labels
                if isinstance(data, dict):
                    for (
                        category,
                        objects,
                    ) in data.items():
                        if isinstance(objects, list):
                            for obj in objects:
                                points = obj.get(f"{'bounding_box'}", [])
                                confidence = obj.get(f"{'confidence'}", [])
                                if (
                                    len(points) == 4
                                ):  # Ensure only valid points are processed
                                    rectangles_with_labels.append(
                                        (points, confidence, category)
                                    )

                    final_img = draw_rectangles_on_image(img, rectangles_with_labels)
                    cv2.imwrite(os.path.join("output/", file), final_img)


if __name__ == "__main__":
    url = "http://192.168.99.142:8083/predict"
    image_path = "C:/InferenceFilesTemporary/test/images"
    process_images(image_path, url)
