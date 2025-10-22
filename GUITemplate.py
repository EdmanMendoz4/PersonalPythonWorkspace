import tkinter as tk
from tkinter import ttk, filedialog
import cv2
from PIL import Image, ImageTk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time

class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-Time Camera and Graph GUI")
        self.root.geometry("1920x1080")

        # Frame para la gráfica (más pequeño)
        graph_frame = tk.Frame(root, width=400, height=300)  # Reducir tamaño
        graph_frame.pack(side=tk.BOTTOM, fill=tk.X, expand=False)

        # Crear la gráfica
        self.figure = Figure(figsize=(4, 3), dpi=100)  # Reducir tamaño
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Gráfica en tiempo real")
        self.ax.set_xlabel("Tiempo")
        self.ax.set_ylabel("Valor")
        self.canvas = FigureCanvasTkAgg(self.figure, master=graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Frame para el video del microscopio (más grande)
        microscope_frame = tk.Frame(root, width=800, height=600)  # Aumentar tamaño
        microscope_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Label para mostrar el video del microscopio en grande
        self.microscope_image_label = tk.Label(microscope_frame, text="Video del microscopio")
        self.microscope_image_label.pack(fill=tk.BOTH, expand=True)

        # Frame para los controles
        control_frame = tk.Frame(root, width=100, height=100)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, expand=False)

        # Botón para seleccionar directorio
        tk.Label(control_frame, text="Seleccionar directorio:").pack(pady=10)
        self.directory_label = tk.Label(control_frame, text="No se ha seleccionado un directorio", wraplength=400, anchor="center", justify="center")
        self.directory_label.pack(pady=5, fill=tk.X)
        self.select_dir_button = tk.Button(control_frame, text="Seleccionar carpeta", command=self.select_directory)
        self.select_dir_button.pack(pady=10)

        # Selector de dispositivo de cámara
        tk.Label(control_frame, text="Cámara de multimetro:").pack(pady=10)
        self.camera_selector = ttk.Combobox(control_frame, values=self.get_camera_list())
        self.camera_selector.pack(pady=10)
        self.camera_selector.current(0)

        # Selector de dispositivo de cámara (Microscopio)
        tk.Label(control_frame, text="Cámara de microscopio:").pack(pady=10)
        self.microscope_camera_selector = ttk.Combobox(control_frame, values=self.get_camera_list())
        self.microscope_camera_selector.pack(pady=10)
        self.microscope_camera_selector.current(0)

        # Botón para iniciar la captura
        self.start_button = tk.Button(control_frame, text="Iniciar", command=self.start_camera)
        self.start_button.pack(pady=10)

        # Label para mostrar la imagen capturada por la cámara de multímetro
        self.image_label = tk.Label(control_frame, text="Imagen capturada (Multímetro)")
        self.image_label.pack(pady=10)

        # Textbox para el log
        tk.Label(control_frame, text="Log de la aplicación:").pack(pady=10)
        self.log_textbox = tk.Text(control_frame, width=40, height=10, state="disabled", wrap="word")
        self.log_textbox.pack(pady=5, fill=tk.BOTH, expand=True)

        # Variables para la cámara
        self.cap = None
        self.running = False

        # Iniciar el log de prueba
        threading.Thread(target=self.test_log, daemon=True).start()

    def select_directory(self):
        # Abrir el explorador de archivos para seleccionar un directorio
        directory = filedialog.askdirectory()
        if directory:
            self.directory_label.config(text=directory)

    def get_camera_list(self):
        # Detectar cámaras disponibles
        camera_list = []
        for i in range(5):  # Probar los primeros 5 dispositivos
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                camera_list.append(f"Camera {i}")
                cap.release()
            else:
                break  # Detener la búsqueda si no se encuentra una cámara
        return camera_list if camera_list else ["No hay cámaras disponibles"]

    def start_camera(self):
        if self.running:
            return
        self.running = True

        # Validar si hay cámaras disponibles
        if not self.camera_selector.get() or self.camera_selector.get() == "No hay cámaras disponibles":
            self.add_log_entry("Error: No hay cámaras disponibles para el multímetro.")
            self.running = False
            return

        if not self.microscope_camera_selector.get() or self.microscope_camera_selector.get() == "No hay cámaras disponibles":
            self.add_log_entry("Error: No hay cámaras disponibles para el microscopio.")
            self.running = False
            return

        # Obtener los índices de las cámaras seleccionadas
        try:
            multimeter_camera_index = int(self.camera_selector.get().split()[-1])
            microscope_camera_index = int(self.microscope_camera_selector.get().split()[-1])
        except ValueError:
            self.add_log_entry("Error: Índice de cámara no válido.")
            self.running = False
            return

        # Intentar abrir ambas cámaras
        self.cap = cv2.VideoCapture(1)
        self.microscope_cap = cv2.VideoCapture(microscope_camera_index)

        if not self.cap.isOpened():
            self.add_log_entry(f"Error: No se pudo abrir la cámara {multimeter_camera_index} (Multímetro).")
            self.running = False
            

        if not self.microscope_cap.isOpened():
            self.add_log_entry(f"Error: No se pudo abrir la cámara {microscope_camera_index} (Microscopio).")
            self.running = False
            return

        # Configurar el escritor de video para la cámara del microscopio
        if hasattr(self, 'directory_label') and self.directory_label.cget("text") != "No se ha seleccionado un directorio":
            output_path = f"{self.directory_label.cget('text')}/microscope_output.avi"
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

        self.add_log_entry(f"Cámara {multimeter_camera_index} (Multímetro) iniciada.")
        self.add_log_entry(f"Cámara {microscope_camera_index} (Microscopio) iniciada.")
        threading.Thread(target=self.update_camera, daemon=True).start()
        threading.Thread(target=self.update_microscope_camera, daemon=True).start()

    def update_camera(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                # Convertir la imagen para tkinter
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                img = img.resize((200, 150))  # Redimensionar para mostrar en pequeño
                imgtk = ImageTk.PhotoImage(image=img)
                self.image_label.imgtk = imgtk
                self.image_label.configure(image=imgtk)
            time.sleep(0.1)  # Capturar cada 100 ms

    def update_microscope_camera(self):
        while self.running:
            ret, frame = self.microscope_cap.read()
            if ret:
                # Mostrar el video en tiempo real (en grande)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img = img.resize((800, 600))  # Aumentar tamaño del video
                imgtk = ImageTk.PhotoImage(image=img)
                self.microscope_image_label.imgtk = imgtk
                self.microscope_image_label.configure(image=imgtk)

                # Guardar el fotograma en el archivo de video
                if hasattr(self, 'microscope_video_writer') and self.microscope_video_writer.isOpened():
                    self.microscope_video_writer.write(frame)

            time.sleep(0.1)  # Capturar cada 100 ms

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

    def add_log_entry(self, text):
        """Agregar una entrada al log."""
        self.log_textbox.config(state="normal")  # Habilitar edición temporalmente
        self.log_textbox.insert("1.0", text + "\n")  # Insertar al inicio
        self.log_textbox.config(state="disabled")  # Deshabilitar edición
        self.log_textbox.yview("moveto", 0)  # Desplazar al inicio

    def test_log(self):
        """Agregar entradas de prueba al log."""
        for i in range(1, 6):
            self.add_log_entry(f"Entrada {i}")
            time.sleep(0.5)  # Esperar 500 ms entre entradas

# Crear la ventana principal
root = tk.Tk()
app = CameraApp(root)
root.protocol("WM_DELETE_WINDOW", app.on_close)
root.mainloop()