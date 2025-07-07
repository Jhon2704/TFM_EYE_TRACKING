import cv2

class RecortePeriocular:
    def recortar(self, imagen, landmarks, margen=10):
        ih, iw, _ = imagen.shape
        x1 = int(landmarks[33].x * iw) - margen  # esquina izquierda ojo izq
        y1 = int(min(landmarks[159].y, landmarks[386].y) * ih) - margen
        x2 = int(landmarks[263].x * iw) + margen  # esquina derecha ojo der
        y2 = int(max(landmarks[145].y, landmarks[374].y) * ih) + margen
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(iw, x2), min(ih, y2)
        return imagen[y1:y2, x1:x2]