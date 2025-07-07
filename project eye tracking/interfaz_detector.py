import os
import cv2
import sys
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import torch.nn as nn


# Ruta al modelo .h5 o .pth
USE_H5_MODEL = True  # Cambia a False si usas modelo .pth

if USE_H5_MODEL:
    from tensorflow.keras.models import load_model
    modelo = load_model(os.path.join("modelos", "modelo_clasificador_cortes.h5"))
    class_names = ['abajo', 'arriba', 'derecha', 'izquierda','centro']
else:
    import torch
    from torchvision import transforms
    sys.path.append('./ETH-XGaze')
    from model import gaze_network
    modelo = gaze_network()
    class_names = ['abajo', 'arriba', 'derecha', 'izquierda','centro']
    modelo.gaze_fc = nn.Linear(2048, len(class_names))  # <- Igual al reentrenado
    modelo.load_state_dict(torch.load("modelos/ethx_finetuned.pth", map_location='cpu'))
    modelo.eval()
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

# === Importar módulos personalizados ===
from src.deteccion_facial import DeteccionFacial
from src.extraccion_landmarks import ExtraccionLandmarks
from src.alineamiento_facial import AlineamientoFacial
from src.evaluacion_calidad import EvaluacionCalidad
from src.apertura_ocular import AperturaOcular
from src.recorte_periocular import RecortePeriocular

# === Aplicación Principal ===
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("🎓 Monitor de Examen - Vigilancia Visual")
        self.root.geometry("880x720")
        self.root.configure(bg="#f2f2f2")

        # Captura de cámara
        self.cap = cv2.VideoCapture(0)

        # Módulos personalizados
        self.detector = DeteccionFacial()
        self.landmarks = ExtraccionLandmarks()
        self.alineador = AlineamientoFacial()
        self.calidad = EvaluacionCalidad()
        self.apertura = AperturaOcular()
        self.recortador = RecortePeriocular()

        # Elementos GUI
        self.video_label = tk.Label(root)
        self.video_label.pack(pady=10)

        self.umbral_angulo = tk.IntVar(value=20)
        self.label_angulo = tk.Label(root, text="Ángulo máximo permitido: 20°", bg="#f2f2f2", font=("Helvetica", 12))
        self.label_angulo.pack()

        self.slider = ttk.Scale(root, from_=5, to=45, orient=tk.HORIZONTAL,
                                variable=self.umbral_angulo, command=self.actualizar_slider)
        self.slider.pack(pady=5)

        self.alerta_label = tk.Label(root, text="", fg="red", font=("Helvetica", 16, "bold"), bg="#f2f2f2")
        self.alerta_label.pack(pady=10)

        self.update_video()

    def actualizar_slider(self, event):
        valor = self.umbral_angulo.get()
        self.label_angulo.config(text=f"Ángulo máximo permitido: {int(valor)}°")

    def update_video(self):
        ret, frame = self.cap.read()
        if not ret:
            self.root.after(10, self.update_video)
            return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultado_rostro = self.detector.detectar(rgb)
        mensaje = ""

        if resultado_rostro.detections:
            resultado_landmarks = self.landmarks.extraer(rgb)
            if resultado_landmarks.multi_face_landmarks:
                puntos = resultado_landmarks.multi_face_landmarks[0].landmark

                # Verificar calidad
                if self.calidad.es_borrosa(frame):
                    mensaje = "⚠ Imagen borrosa"
                else:
                    angulo = self.alineador.calcular_angulo(puntos, frame.shape)
                    umbral = self.umbral_angulo.get()
                    if abs(angulo) > umbral:
                        mensaje = f"⚠ Giro excesivo: {int(angulo)}°"

                    # Recorte periocular
                    periocular = self.recortador.recortar(frame, puntos)
                    if periocular.size > 0:
                        # === Clasificación de dirección de la mirada ===
                        if USE_H5_MODEL:
                            ojo = cv2.resize(periocular, (128, 128))
                            ojo = ojo.astype('float32') / 255.0
                            ojo = np.expand_dims(ojo, axis=0)
                            pred = modelo.predict(ojo, verbose=0)[0]
                        else:
                            with torch.no_grad():
                                img_tensor = transform(periocular).unsqueeze(0)
                                pred = modelo(img_tensor).squeeze().numpy()

                        clase = class_names[np.argmax(pred)]
                        if clase != "centro":
                            mensaje = f"⚠ Mirada: {clase.upper()}"
                        # === Dibujar flecha indicando la dirección de la mirada ===
                        ih, iw, _ = frame.shape
                        centro = (iw // 2, ih // 2)
                        offset = 80  # distancia de la flecha
                        
                        # Determinar destino de la flecha según la clase
                        if clase == 'arriba':
                            destino = (centro[0], centro[1] - offset)
                        elif clase == 'abajo':
                            destino = (centro[0], centro[1] + offset)
                        elif clase == 'izquierda':
                            destino = (centro[0] - offset, centro[1])
                        elif clase == 'derecha':
                            destino = (centro[0] + offset, centro[1])
                        elif clase == 'centro':
                            destino = (centro[0] + offset, centro[1])
                        else:
                            destino = centro  # Por si acaso
                        
                        # Dibujar flecha en verde
                        cv2.arrowedLine(frame, centro, destino, (0, 255, 0), 4, tipLength=0.3)
        
                        cv2.putText(frame, f"Mirada: {clase}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8, (255, 0, 0), 2)

                    # === Apertura Ocular (EAR) ===
                    ojo_izq = np.array([[puntos[i].x, puntos[i].y] for i in [362, 385, 387, 263, 373, 380]])
                    ojo_der = np.array([[puntos[i].x, puntos[i].y] for i in [33, 160, 158, 133, 153, 144]])
                    shape = frame.shape
                    ojo_izq *= [shape[1], shape[0]]
                    ojo_der *= [shape[1], shape[0]]

                    ear_izq = self.apertura.calcular_ear(ojo_izq)
                    ear_der = self.apertura.calcular_ear(ojo_der)
                    ear_avg = (ear_izq + ear_der) / 2.0

                    if ear_avg < 0.2:
                        mensaje = "⚠ Ojos cerrados o parpadeo largo"

                    cv2.putText(frame, f"EAR: {ear_avg:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 0, 255), 2)

        self.alerta_label.config(text=mensaje)

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((820, 500))
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.root.after(10, self.update_video)

    def cerrar(self):
        self.cap.release()
        self.root.destroy()

# === Lanzar interfaz ===
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", app.cerrar)
    root.mainloop()
