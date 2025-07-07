import cv2
import mediapipe as mp
import numpy as np



# === Importar clases del sistema ===

from src.apertura_ocular import AperturaOcular


# Inicializar FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# EAR thresholds
UMBRAL_EAR = 0.22

# Índices para los dos ojos (Mediapipe)
OJO_IZQ = [33, 160, 158, 133, 153, 144]   # EAR: p0 a p5
OJO_DER = [362, 385, 387, 263, 373, 380]  # EAR: p0 a p5

# Instancia de la clase
detector = AperturaOcular()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        face = results.multi_face_landmarks[0].landmark
        ih, iw, _ = frame.shape

        def get_ear(puntos_idx):
            puntos = []
            for idx in puntos_idx:
                lm = face[idx]
                x, y = int(lm.x * iw), int(lm.y * ih)
                puntos.append(np.array([x, y]))
                cv2.circle(frame, (x, y), 2, (255, 255, 0), -1)
            return detector.calcular_ear(np.array(puntos))

        # Calcular EAR para ambos ojos
        ear_izq = get_ear(OJO_IZQ)
        ear_der = get_ear(OJO_DER)

        estado_izq = "ABIERTO" if ear_izq > UMBRAL_EAR else "CERRADO"
        estado_der = "ABIERTO" if ear_der > UMBRAL_EAR else "CERRADO"

        # Mostrar resultados en pantalla
        cv2.putText(frame, f"Ojo Izq: {estado_izq} (EAR={ear_izq:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0) if estado_izq=="ABIERTO" else (0,0,255), 2)
        cv2.putText(frame, f"Ojo Der: {estado_der} (EAR={ear_der:.2f})", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0) if estado_der=="ABIERTO" else (0,0,255), 2)

    cv2.imshow("Apertura de ambos ojos", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
