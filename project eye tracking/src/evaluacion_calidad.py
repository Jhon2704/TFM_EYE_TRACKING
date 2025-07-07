import cv2

class EvaluacionCalidad:
    def es_borrosa(self, imagen):
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gris, cv2.CV_64F).var()
        return lap < 100