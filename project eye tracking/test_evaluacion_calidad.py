import cv2

# === Importar clases del sistema ===
from src.evaluacion_calidad import EvaluacionCalidad


cap = cv2.VideoCapture(0)
calidad = EvaluacionCalidad()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    texto = "BORROSA" if calidad.es_borrosa(frame) else "CLARA"
    color = (0, 0, 255) if texto == "BORROSA" else (0, 255, 0)
    cv2.putText(frame, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Calidad", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
