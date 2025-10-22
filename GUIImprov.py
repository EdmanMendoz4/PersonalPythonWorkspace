import tkinter as tk
from tkinter import ttk, filedialog
from inference_sdk import InferenceHTTPClient
import cv2
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image, ImageTk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import os
import time
import csv
from pygrabber.dshow_graph import FilterGraph
import numpy as np

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="eMtxERShjAk2e6QBaVHU"
)

class Graph:
    def __init__(self, width, height):
        self.height = height
        self.width = width
        self.graph = np.zeros((height, width, 3), np.uint8)
        self.x = 0
        self.y = 0
        
    def update_frame(self, value, frame_count):
        if value < 0:
            value = 0
        elif value >= self.height:
            value = self.height - 1
        new_graph = np.zeros((self.height, self.width, 3), np.uint8)
        new_graph[:,:-1,:] = self.graph[:,1:,:]
        cv2.line(new_graph, (self.x, self.y), (frame_count//10, value), (0, 0, 255))
        
        self.x = frame_count // 10
        self.y = value
        self.graph = new_graph
        
    def get_graph(self):
        return self.graph

class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-Time Camera and Graph GUI")
        self.root.geometry("1920x1080")

        # Frame para la gráfica
        graph_frame = tk.Frame(root, width=400, height=300)
        graph_frame.pack(side=tk.BOTTOM, fill=tk.X, expand=False)

        # Crear la gráfica
        plt.ion()  # Habilitar modo interactivo
        self.figure = Figure(figsize=(6, 2), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_ylabel("MOhms")
        self.line, = self.ax.plot([], [], 'b-o', label="Detected Values")
        self.canvas = FigureCanvasTkAgg(self.figure, master=graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Frame para el video del microscopio (más grande)
        microscope_frame = tk.Frame(root, width=800, height=600)  # Aumentar tamaño
        microscope_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Label para mostrar el video del microscopio en grande
        self.microscope_image_label = tk.Label(microscope_frame, text="Video del microscopio")
        self.microscope_image_label.pack(fill=tk.BOTH, expand=True)

        # Frame para los controles
        control_frame = tk.Frame(root, width=100, height=600)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y)

        # Botón para seleccionar directorio
        tk.Label(control_frame, text="Seleccionar directorio:").pack(pady=10)
        self.directory_label = tk.Label(control_frame, text="No se ha seleccionado un directorio", wraplength=300, anchor="w", justify="left")
        self.directory_label.pack(pady=5, fill=tk.X)
        self.select_dir_button = tk.Button(control_frame, text="Seleccionar carpeta", command=self.select_directory)
        self.select_dir_button.pack(pady=10)

        # Selector de dispositivo de cámara (Multimetro)
        tk.Label(control_frame, text="Seleccionar cámara:").pack(pady=10)
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

        # Label para mostrar la imagen capturada
        self.image_label = tk.Label(control_frame, text="Imagen capturada")
        self.image_label.pack(pady=10)

        # Textbox para el log
        self.log_textbox = tk.Text(control_frame, height=10, width= 40, state="disabled", wrap="word")
        self.log_textbox.pack(pady=5, fill=tk.BOTH, expand=True)

        # Variables para la cámara
        self.cap = None
        self.running = False
        
        # Inicializar listas para almacenar los datos
        self.timestamps = []
        self.values = []

        #Directorio de guardado
        self.selected_directory = None
        self.image_log_dir = None

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
        self.line.set_xdata([])
        self.line.set_ydata([])
        self.ax.set_xticks([])
        self.ax.set_xticklabels([])
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw()

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
        
        # Intentar abrir la cámara
        self.cap = cv2.VideoCapture(camera_index)
        self.microscope_cap = cv2.VideoCapture(microscope_camera_index)
        self.microscope_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Configurar ancho
        self.microscope_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Configurar ancho
        
        if not self.cap.isOpened():
            self.add_log_entry(f"Error: No se pudo abrir la cámara seleccionada.")
            self.running = False
            return

        if not self.microscope_cap.isOpened():
            self.add_log_entry(f"Error: No se pudo abrir la cámara del microscopio.")
            self.running = False
            return

        # Configurar el escritor de video para la cámara del microscopio
        if hasattr(self, 'directory_label') and self.directory_label.cget("text") != "No se ha seleccionado un directorio":
            output_path = f"{self.image_log_dir}/microscope_output.avi"
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            fps = 10  # Frames por segundo
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

    def update_camera(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                results = CLIENT.infer(frame, model_id="digits-coi4f/3")
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

                # Convertir la imagen para tkinter
                frame = cv2.resize(frame, (200, 120))  # Redimensionar para mostrar en pequeño
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.image_label.imgtk = imgtk
                self.image_label.configure(image=imgtk)
            time.sleep(0.03)
    
    def update_microscope_camera(self):
        while self.running:
            ret, frame = self.microscope_cap.read()
            if ret:
                # Mostrar el video en tiempo real
                frame = cv2.resize(frame, (1280, 720))  # Redimensionar    
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cv2.imshow("Microscope", frame_rgb)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                #img = Image.fromarray(frame_rgb)
                #imgtk = ImageTk.PhotoImage(image=img)
                #self.microscope_image_label.imgtk = imgtk
                #self.microscope_image_label.configure(image=imgtk)

                # Guardar el fotograma en el archivo de video
                #if hasattr(self, 'microscope_video_writer') and self.microscope_video_writer.isOpened():
                 #   self.microscope_video_writer.write(frame)
                    
            time.sleep(0.03)   
        
        
    def stop_camera(self):
        self.running = False
        if self.cap:
            self.cap.release()
        if hasattr(self, 'microscope_cap') and self.microscope_cap:
            self.microscope_cap.release()
        if hasattr(self, 'microscope_video_writer') and self.microscope_video_writer.isOpened():
            self.microscope_video_writer.release()

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