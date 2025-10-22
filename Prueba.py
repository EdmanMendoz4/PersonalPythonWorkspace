import cv2
import threading
import time


class CAPP:
    def __init__(self):
        self.cap = None
        self.microscope_cap = None
        self.running = False
        self.microscope_video_writer = None
        self.directory_label = None  # Placeholder for directory label
        self.camera_index = 0  # Placeholder for camera index
        self.microscope_camera_index = 1  # Placeholder for microscope camera index

    def update_camera(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, (200, 120))  # Redimensionar para mostrar en pequeño
                results = CLIENT.infer(frame, model_id="7-segment-display-gxhnj/2")
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
                        
                        # Guardar la fecha, hora y valor detectado
                        current_time = datetime.now().strftime("%H:%M:%S")
                        self.timestamps.append(current_time)
                        self.values.append(num)  # Convertir a float para el eje y
                        self.add_log_entry(f"{(str(num))} - {current_time}")  # Mostrar el número detectado
                        
                        # Actualizar la gráfica
                        self.line.set_xdata(range(len(self.timestamps)))  # Usar índices numéricos en lugar de timestamps
                        self.line.set_ydata(self.values)
                        self.ax.set_xticks(range(len(self.timestamps)))  # Configurar las posiciones de las etiquetas
                        self.ax.set_xticklabels(self.timestamps, rotation=45)  # Mostrar las marcas de tiempo como etiquetas
                        self.ax.relim()
                        self.ax.autoscale_view()
                        self.canvas.draw()
                    else:
                        # Guardar la imagen en la subcarpeta
                        if self.image_log_dir:
                            img_filename = datetime.now().strftime("img_%H%M%S_%f.jpg")
                            img_path = os.path.join(self.image_log_dir, img_filename)
                            cv2.imwrite(img_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            time.sleep(0.03)