from inference_sdk import InferenceHTTPClient , InferenceConfiguration
import cv2
import numpy as np

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="eMtxERShjAk2e6QBaVHU"
)

class Graph:
    def __init__(self, width, height):
        self.height = height
        self.width = width
        self.lastpoint = int(width * 0.90)
        self.graph = np.zeros((height, width, 3), np.uint8)
        self.y = 0
        
    def update_frame(self, value):
        value = value * 4
        if value < 0:
            value = 0
        elif value >= self.height:
            value = self.height - 1
                        
        new_graph = np.zeros((self.height, self.width, 3), np.uint8)
        new_graph[:,:-1,:] = self.graph[:,1:,:]
        cv2.line(new_graph, ((self.lastpoint - 1), self.y), (self.lastpoint, value), (0, 0, 255))
        self.y = value
        self.graph = new_graph
        
    def get_graph(self):
        return self.graph
    

# Setup camera
cap = cv2.VideoCapture(0)
# Set a smaller resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
graph = Graph(400, 200)
prev_frame = np.zeros((480, 640), np.uint8)
frame_count = 0
custom_configuration = InferenceConfiguration(confidence_threshold=0.8)
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1


    results = CLIENT.infer(frame, model_id="7-segment-display-gxhnj/2")
    #with CLIENT.use_model("7-segment-display-gxhnj/2"):
    #    results = CLIENT.infer(frame)
    # Extraer los dígitos detectados y sus posiciones 'x'
    detections_sorted = sorted(results["predictions"], key=lambda d: d["x"])  # Ordenar por 'x'
    number_string = "".join([str(d["class"]) for d in detections_sorted if d["confidence"] > 0.5 and d["x"] > 150 and d["y"] > 150])  # Crear el string
    if number_string:
        if number_string != "." and number_string.count('.') <= 1: # Si se detectó un número
            num = float(number_string)  # Convertir a float para el eje y
            #Validar numero detectado
            if num > 1000:
                num = num / 100
            if num > 100:
                num = num / 10                
            graph.update_frame(int(num))
    roi = frame[-210:-10, -410:-10,:]
    roi[:] = graph.get_graph()
    cv2.putText(frame, "...wanted a live graph", (20, 430), cv2.FONT_HERSHEY_PLAIN, 1.8, (0, 200, 200), 2)
    cv2.putText(frame, "...measures motion in frame", (20, 460), cv2.FONT_HERSHEY_PLAIN, 1.8, (0, 200, 200), 2)
    cv2.imshow("Webcam", frame)
    if cv2.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()