mport cv2
import numpy as np

class AlineamientoFacial:
    def alinear(self, imagen, landmarks):
        ih, iw, _ = imagen.shape
        p1 = landmarks[33]
        p2 = landmarks[263]

        x1, y1 = int(p1.x * iw), int(p1.y * ih)
        x2, y2 = int(p2.x * iw), int(p2.y * ih)
        angulo = np.degrees(np.arctan2(y2 - y1, x2 - x1))

        centro = ((x1 + x2) // 2, (y1 + y2) // 2)
        M = cv2.getRotationMatrix2D(centro, angulo, 1.0)
        return cv2.warpAffine(imagen, M, (iw, ih))

    def calcular_angulo(self, landmarks, shape):
        """Devuelve el ángulo de giro de la cabeza"""
        ih, iw = shape[:2]
        p1 = landmarks[33]
        p2 = landmarks[263]
        x1, y1 = int(p1.x * iw), int(p1.y * ih)
        x2, y2 = int(p2.x * iw), int(p2.y * ih)
        angulo = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        return angulo

