import cv2


# === Importar clases del sistema ===
from src.extraccion_landmarks import ExtraccionLandmarks


cap = cv2.VideoCapture(0)
extractor = ExtraccionLandmarks()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = extractor.extraer(rgb)

    if results.multi_face_landmarks:
        for face in results.multi_face_landmarks:
            for lm in face.landmark:
                h, w, _ = frame.shape
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 1, (255, 0, 0), -1)

    cv2.imshow("Landmarks", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
