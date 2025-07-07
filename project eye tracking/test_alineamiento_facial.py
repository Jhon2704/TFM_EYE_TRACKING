import cv2

# Importar clases del src
from src.extraccion_landmarks import ExtraccionLandmarks
from src.alineamiento_facial import AlineamientoFacial


cap = cv2.VideoCapture(0)
alineador = AlineamientoFacial()
extractor = ExtraccionLandmarks()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = extractor.extraer(rgb)

    if results.multi_face_landmarks:
        face = results.multi_face_landmarks[0].landmark
        alineada = alineador.alinear(frame, face)
        cv2.imshow("Alineada", alineada)
    else:
        cv2.imshow("Alineada", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
