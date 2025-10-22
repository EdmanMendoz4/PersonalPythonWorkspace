import tkinter as tk
from tkinter import ttk, filedialog
from inference_sdk import InferenceHTTPClient
import cv2
import matplotlib.pyplot as plt
from datetime import datetime
import threading
import os
import csv
from pygrabber.dshow_graph import FilterGraph
import numpy as np

#Defines para la configuracion de las imagenes
MICROSCOPE_VIDEO_WIDTH = 1920
MICROSCOPE_VIDEO_HEIGHT = 1080
MICROSCOPE_VIDEO_FPS = 15
MICROSCOPE_CAPTURE_DELAY = (1.0 / MICROSCOPE_VIDEO_FPS)

MULTIMETER_CAPTURE_WIDTH = 640
MULTIMETER_CAPTURE_HEIGHT = 480
MULTIMETER_FRAME_WIDTH = 160
MULTIMETER_FRAME_HEIGHT = 120

GRAPH_WIDTH = 400
GRAPH_HEIGHT = 200


# Cliente de Inference para la detección de dígitos
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
    
    def draw_guides(self, img):
        num_lines = 4
        for i in range(1, num_lines + 1):
            y = int(i * self.height / (num_lines + 1))
            # Línea punteada
            for x in range(0, self.width, 20):
                cv2.line(img, (x, y), (x + 10, y), (100, 100, 100), 1, lineType=cv2.LINE_4)
            # Texto a la izquierda
            label = str(i * 10)
            cv2.putText(img, label, (10, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 2, cv2.LINE_AA)
        
        # Texto centrado encima de la gráfica
        title = "Resistencia (MOhms)"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        color = (0, 150, 150)  # Amarillo 
        # Obtener tamaño del texto
        (text_width, text_height), _ = cv2.getTextSize(title, font, font_scale, thickness)
        x = (self.width - text_width) // 2
        y = text_height + 5  # Un poco de margen arriba
        cv2.putText(img, title, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
    
    def ImageOverlay(self, frame, secframe, value1, value2, value3):
        # Mostrar la grafica encima de la imagen
        gh, gw, _ = self.graph.shape
        h, w, _ = frame.shape
        
        # Colocar lineas a la grafica antes de mostrarla
        grafica = self.graph.copy()
        self.draw_guides(grafica)
        # Coloca la gráfica en la esquina superior derecha
        frame[0:gh, w-gw:w, :] = grafica

        # Coloca el multímetro a la izquierda de la gráfica, alineado arriba
        if secframe is not None:
            mh, mw, _ = secframe.shape
            mh = min(mh, gh)
            frame[0:mh, w-gw-mw:w-gw, :] = secframe[0:mh, 0:mw, :]
            text3 = f"Celula No. {value3}" if value3 is not None else ""
            
            # Overlay de texto debajo del multímetro, a la izquierda de la gráfica
            if value1 is not None and value2 is not None:
                text1 = f" {value1:.2f}"
                text2 = f" {value2:.2f}"
            elif value1 is not None:
                text1 = f" {value1:.2f}"
                text2 = ""
            else:
                text1 = ""
                text2 = ""

            text_x = w - gw - mw + 10
            text_y1 = mh + 30
            text_y2 = mh + 65
            text_y3 = mh + 100

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            color = (0, 150, 150)
            thickness = 2

            if text1:
                cv2.putText(frame, text1, (text_x, text_y1), font, font_scale, color, thickness, cv2.LINE_AA)
            if text2:
                cv2.putText(frame, text2, (text_x, text_y2), font, font_scale, color, thickness, cv2.LINE_AA)
            if value3 is not None:
                cv2.putText(frame, text3, ((text_x-10), text_y3), font, 0.5, (150,150,150), thickness, cv2.LINE_AA)
                
        return frame
    
    def update_frame(self, value):
        value = int(value * 4)
        if value < 0:
            value = 0
        elif value >= self.height:
            value = self.height - 1
        
        if value > 255:
            value = 255
                        
        new_graph = np.zeros((self.height, self.width, 3), np.uint8)
        new_graph[:,:-2,:] = self.graph[:,2:,:]
        cv2.line(new_graph, ((self.lastpoint - 2), self.y), (self.lastpoint, value), (0, 0, 255), thickness=2)
        self.y = value
        self.graph = new_graph
            
    def get_graph(self):
        return self.graph

class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Monitor de Experimentos")
        self.root.geometry("500x800")
        
        # Inicializar los elementos del Overlay
        self.graph = Graph(GRAPH_WIDTH, GRAPH_HEIGHT)
        self.multimeter_frame = None

        # Frame para los controles
        control_frame = tk.Frame(root, width=100, height=600)
        control_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Botón para seleccionar directorio
        tk.Label(control_frame, text="Seleccionar directorio:").pack(pady=10)
        self.directory_label = tk.Label(control_frame, text="No se ha seleccionado un directorio", wraplength=300, anchor="center", justify="center")
        self.directory_label.pack(pady=5, fill=tk.X)
        self.select_dir_button = tk.Button(control_frame, text="Seleccionar carpeta", command=self.select_directory)
        self.select_dir_button.pack(pady=10)

        # Selector de dispositivo de cámara (Multimetro)
        tk.Label(control_frame, text="Cámara de multimetro:").pack(pady=10)
        self.camera_selector = ttk.Combobox(control_frame, values=self.get_camera_list())
        self.camera_selector.pack(pady=10)
        self.camera_selector.current(2)

        # Selector de dispositivo de cámara (Microscopio)
        tk.Label(control_frame, text="Cámara de microscopio:").pack(pady=10)
        self.microscope_camera_selector = ttk.Combobox(control_frame, values=self.get_camera_list())
        self.microscope_camera_selector.pack(pady=10)
        self.microscope_camera_selector.current(0)
        
        # Botón para iniciar la captura
        self.start_button = tk.Button(control_frame, text="Iniciar", command=self.start_camera)
        self.start_button.pack(pady=10)

        # Botón para detener la captura
        self.stop_button = tk.Button(control_frame, text="Detener", command=self.stop_and_save)
        self.stop_button.pack(pady=10)

        # Textbox para el log
        self.log_textbox = tk.Text(control_frame, height=10, width= 40, state="disabled", wrap="word")
        self.log_textbox.pack(pady=5, fill=tk.BOTH, expand=True)

        # Variables para la cámara
        self.microscope_cap = None
        self.cap = None
        self.running = False
        self.is_segment_recording = False
        
        # Inicializar listas para almacenar los datos
        self.timestamps = []
        self.values = []
        self.segment_timestamps = []
        self.segment_values = []

        # Directorio de guardado
        self.selected_directory = None
        self.image_log_dir = None

        # Selección automática de directorio si existe
        default_dir = r'C:\Users\Sell\Desktop\Experimentos'
        if os.path.exists(default_dir):
            self.selected_directory = default_dir
            self.directory_label.config(text=default_dir)
        # Si no existe, el usuario deberá seleccionar uno manualmente

    def select_directory(self):
        # Abrir el explorador de archivos para seleccionar un directorio
        directory = filedialog.askdirectory()
        if directory:
            self.selected_directory = directory
            self.directory_label.config(text=directory)

    def get_camera_list(self):
        # Usar pygrabber para obtener los nombres de las cámaras
        graph = FilterGraph()
        device_names = graph.get_input_devices()
        return device_names if device_names else ["No hay cámaras disponibles"]

    def start_camera(self):
        if self.running:
            return

        # Limpiar datos anteriores
        self.timestamps.clear()
        self.values.clear()
        self.segment_timestamps.clear()
        self.segment_values.clear()
        self.microscope_framecount = 0
        self.mult_frame_count = 0
        self.current_recording = 0
        self.current_recording_frame_count = 0
        self.running = True

        # Validar si hay cámaras disponibles
        if not self.camera_selector.get() or self.camera_selector.get() == "No hay cámaras disponibles":
            self.add_log_entry("Error: No hay cámaras disponibles.")
            self.running = False
            return

        if not self.microscope_camera_selector.get() or self.microscope_camera_selector.get() == "No hay cámaras disponibles":
            self.add_log_entry("Error: No hay cámaras disponibles para el microscopio.")
            self.running = False
            return


        # Validar si se ha seleccionado un directorio
        if not self.selected_directory:
            self.add_log_entry("Error: No se ha seleccionado un directorio.")
            self.running = False
            return

        # Crear subcarpeta para las imágenes
        now = datetime.now()
        folder_name = now.strftime("Log%d%m%y-%H%M%S")
        self.log_basename = now.strftime("Experimento%d%m%y-%H%M%S")  # <-- Añade esto
        self.image_log_dir = os.path.join(self.selected_directory, folder_name)
        os.makedirs(self.image_log_dir, exist_ok=True)

        # Obtener el índice de la cámara seleccionada
        camera_index = self.camera_selector.current()  # Índice en la lista de dispositivos
        microscope_camera_index = self.microscope_camera_selector.current()  # Índice en la lista de dispositivos
        
        # Intentar abrir la cámara del multimetro
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            self.add_log_entry(f"Error: No se pudo abrir la cámara seleccionada.")
            self.running = False
            return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, MULTIMETER_CAPTURE_WIDTH)  # Configurar ancho
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, MULTIMETER_CAPTURE_HEIGHT)  # Configurar ancho
        
        # Intentar abrir la cámara del microscopio
        self.microscope_cap = cv2.VideoCapture(microscope_camera_index)
        if not self.microscope_cap.isOpened():
            self.add_log_entry(f"Error: No se pudo abrir la cámara del microscopio.")
            self.running = False
            return
        self.microscope_cap.set(cv2.CAP_PROP_FRAME_WIDTH, MICROSCOPE_VIDEO_WIDTH)  # Configurar ancho
        self.microscope_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, MICROSCOPE_VIDEO_HEIGHT)  # Configurar ancho


        # Configurar el escritor de video para la cámara del microscopio
        if hasattr(self, 'directory_label') and self.directory_label.cget("text") != "No se ha seleccionado un directorio":
            # Crear el escritor de video principal
            output_path = f"{self.image_log_dir}/microscope_output.avi"
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            fps = MICROSCOPE_VIDEO_FPS  # Frames por segundo
            frame_width = int(self.microscope_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(self.microscope_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.microscope_video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
            self.add_log_entry(f"Guardando video del microscopio en: {output_path}")
            
        else:
            self.add_log_entry("Error: No se ha seleccionado un directorio para guardar el video.")
            self.running = False
            return        
                
        self.add_log_entry(f"Cámara {camera_index} (Multímetro) iniciada.")
        self.add_log_entry(f"Cámara {microscope_camera_index} (Microscopio) iniciada.")
        threading.Thread(target=self.update_camera, daemon=True).start()
        threading.Thread(target=self.update_microscope_camera, daemon=True).start()

    def toggle_segment_recording(self):
        if not self.is_segment_recording:
            # Crear el escritor de video para los segmentos
            self.current_recording += 1
            segment_output_path = os.path.join(self.image_log_dir, f"Segmento#{self.current_recording}.avi")
            segment_fourcc = cv2.VideoWriter_fourcc(*'XVID')
            fps = MICROSCOPE_VIDEO_FPS  # Frames por segundo
            frame_width = int(self.microscope_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(self.microscope_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.microscope_segment_writer = cv2.VideoWriter(segment_output_path, segment_fourcc, fps, (frame_width, frame_height))
            self.segment_timestamps.clear()
            self.segment_values.clear()
            self.add_log_entry(f"Grabando segmento {self.current_recording} del microscopio en: {segment_output_path}")
            self.is_segment_recording = True
        else:
            # Detener la grabación del segmento actual
            if hasattr(self, 'microscope_segment_writer') and self.microscope_segment_writer.isOpened():
                self.microscope_segment_writer.release()
            self.add_log_entry(f"Grabación de segmento detenida.")
            self.current_recording_frame_count = 0
            self.is_segment_recording = False
            # Guardar los datos del segmento en un archivo CSV
            if self.segment_timestamps and self.segment_values:
                segment_csv = os.path.join(self.image_log_dir, f"Segmento#{self.current_recording}.csv")
                with open(segment_csv, mode='w', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    writer.writerow(["Timestamp", "Value"])
                    for t, v in zip(self.segment_timestamps, self.segment_values):
                        writer.writerow([t, v])
                self.add_log_entry(f"Datos del segmento guardados en {segment_csv}")
    
    def update_camera(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                results = CLIENT.infer(frame, model_id="7-segment-display-gxhnj/2")
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
                        # Actualizar la gráfica
                        self.graph.update_frame(num)
                                                
                        # Guardar la fecha, hora y valor detectado
                        minutes = self.microscope_framecount // MICROSCOPE_VIDEO_FPS // 60
                        seconds = (self.microscope_framecount // MICROSCOPE_VIDEO_FPS) % 60
                        current_time = f"{minutes:02d}:{seconds:02d}"
                        self.timestamps.append(current_time)
                        self.values.append(num)  # Convertir a float para el eje y
                        self.add_log_entry(f"{(str(num))} - {current_time}")  # Mostrar el número detectado
                        
                        if self.is_segment_recording:
                            # Guardar el fotograma en el archivo de video del segmento
                            minutes = self.current_recording_frame_count // MICROSCOPE_VIDEO_FPS // 60
                            seconds = (self.current_recording_frame_count // MICROSCOPE_VIDEO_FPS) % 60
                            current_time = f"{minutes:02d}:{seconds:02d}"
                            self.segment_timestamps.append(current_time)
                            self.segment_values.append(num)
                        
                    else:
                        # Guardar la imagen en la subcarpeta
                        if self.image_log_dir:
                            img_filename = datetime.now().strftime("img_%H%M%S_%f.jpg")
                            img_path = os.path.join(self.image_log_dir, img_filename)
                            cv2.imwrite(img_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                self.mult_frame_count = self.mult_frame_count + 1
                small_frame = cv2.resize(frame, (MULTIMETER_FRAME_WIDTH, MULTIMETER_FRAME_HEIGHT))
                small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                cv2.putText(small_frame, str(self.mult_frame_count), (5, 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1, cv2.LINE_AA)
                self.multimeter_frame = small_frame
    
    def update_microscope_camera(self):
        while self.running:
            ret, frame = self.microscope_cap.read()
            self.microscope_framecount += 1
            if ret:
                # Mostrar el video en tiempo real 
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                if self.multimeter_frame is not None:
                    # Overlay de la gráfica y el multímetro
                    frame_rgb = self.graph.ImageOverlay(frame_rgb, self.multimeter_frame, self.values[-1] if self.values else None,
                                                        self.values[-2] if len(self.values) > 1 else None, 
                                                        self.current_recording if self.is_segment_recording else None)
                
                # Mostrar la imagen en la ventana
                cv2.imshow("Microscopio", frame_rgb)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.stop_and_save()
                elif key == ord('e'):
                    self.toggle_segment_recording()  # Cambiar el estado de grabación de segmentos

                # Guardar el fotograma en el archivo de video
                if hasattr(self, 'microscope_video_writer') and self.microscope_video_writer.isOpened():
                    self.microscope_video_writer.write(frame_rgb)
                if hasattr(self, 'microscope_segment_writer') and self.microscope_segment_writer.isOpened():
                    self.microscope_segment_writer.write(frame_rgb)
                    self.current_recording_frame_count += 1
        
        
    def stop_camera(self):
        self.running = False
        if self.cap:
            self.cap.release()
        if hasattr(self, 'microscope_cap') and self.microscope_cap:
            self.microscope_cap.release()
        if hasattr(self, 'microscope_video_writer') and self.microscope_video_writer.isOpened():
            self.microscope_video_writer.release()
        if hasattr(self, 'microscope_segment_writer') and self.microscope_segment_writer.isOpened():
            self.microscope_segment_writer.release()
        self.is_segment_recording = False
        cv2.destroyAllWindows()
        self.add_log_entry("Cámara detenida.")

    def on_close(self):
        self.stop_camera()
        self.root.destroy()
        plt.ioff()  # Deshabilitar modo interactivo

    def add_log_entry(self, text):
        """Agregar una entrada al log."""
        self.log_textbox.config(state="normal")  # Habilitar edición temporalmente
        self.log_textbox.insert("1.0", text + "\n")  # Insertar al inicio
        self.log_textbox.config(state="disabled")  # Deshabilitar edición
        self.log_textbox.yview("moveto", 0)  # Desplazar al inicio
        
    def stop_and_save(self):
        self.stop_camera()
        if self.timestamps and self.values:
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")],
                title="Guardar datos como CSV",
                initialdir = self.selected_directory,
                initialfile=getattr(self, "log_basename", "Experimento") + ".csv"  # Usa el nombre guardado o "Experimento"
            )
            if filename:
                with open(filename, mode='w', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    writer.writerow(["Timestamp", "Value"])
                    for t, v in zip(self.timestamps, self.values):
                        writer.writerow([t, v])
                self.add_log_entry(f"Datos guardados en {filename}")
            else:
                self.add_log_entry("Guardado cancelado por el usuario.")
        else:
            self.add_log_entry("No hay datos para guardar.")


# Crear la ventana principal
root = tk.Tk()
app = CameraApp(root)
root.protocol("WM_DELETE_WINDOW", app.on_close)
root.mainloop()