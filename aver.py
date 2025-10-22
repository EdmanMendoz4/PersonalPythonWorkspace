from inference_sdk import InferenceHTTPClient
import cv2
import matplotlib.pyplot as plt
from datetime import datetime

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="eMtxERShjAk2e6QBaVHU"
)

# Inicializar la cámara (0 es el índice de la cámara predeterminada)
cap = cv2.VideoCapture(0)

# Inicializar listas para almacenar los datos
timestamps = []
values = []

# Configurar la gráfica
plt.ion()  # Habilitar modo interactivo
fig, ax = plt.subplots()
line, = ax.plot([], [], 'b-o', label="Detected Values")
ax.set_xlabel("Timestamp")
ax.set_ylabel("Detected Value")
ax.set_title("Real-Time Detection")
ax.legend()

if not cap.isOpened():
    print("Error: No se pudo acceder a la cámara.")
else:
    while True:
        # Leer un frame de la cámara
        ret, frame = cap.read()
        if ret:
            results = CLIENT.infer(frame, model_id="7-segment-display-gxhnj/2")
            # Extraer los dígitos detectados y sus posiciones 'x'
            detections_sorted = sorted(results["predictions"], key=lambda d: d["x"])  # Ordenar por 'x'
            for detection in detections_sorted:
                x = (detection["x"])
                y = (detection["y"])
                conf = (detection["confidence"])
                #Aplicar un umbral de confianza para mostrar solo detecciones confiables      
                number_string = "".join([str(d["class"]) for d in detections_sorted if d["confidence"] > 0.6 and d["x"] > 150 and d["y"] > 150])  # Crear el string
                    
  
            if number_string:  # Si se detectó un número
                print(f"Detected number: {number_string}")  # Mostrar el número detectado
                
                # Guardar la fecha, hora y valor detectado
                current_time = datetime.now().strftime("%H:%M:%S")
                timestamps.append(current_time)
                values.append(float(number_string))  # Convertir a float para el eje y

                # Actualizar la gráfica
                line.set_xdata(range(len(timestamps)))  # Usar índices numéricos en lugar de timestamps
                line.set_ydata(values)
                ax.set_xticks(range(len(timestamps)))  # Configurar las posiciones de las etiquetas
                ax.set_xticklabels(timestamps, rotation=45)  # Mostrar las marcas de tiempo como etiquetas
                ax.relim()
                ax.autoscale_view()
                plt.pause(0.1)  # Pausa para actualizar la gráfica
            
            # Mostrar el frame con la detección
            cv2.imshow("Frame", frame)
            
            # Agregar un retraso para actualizar la ventana y permitir salir con 'q'
            if cv2.waitKey(300) & 0xFF == ord('q'):
                break
        else:
            print("Error: No se pudo capturar la imagen.")

# Liberar la cámara y cerrar ventanas
cap.release()
cv2.destroyAllWindows()
plt.ioff()  # Deshabilitar modo interactivo
plt.show()  # Mostrar la gráfica final
