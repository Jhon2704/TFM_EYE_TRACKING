import numpy as np

class AperturaOcular:
    def calcular_ear(self, puntos):
        A = np.linalg.norm(puntos[1] - puntos[5])
        B = np.linalg.norm(puntos[2] - puntos[4])
        C = np.linalg.norm(puntos[0] - puntos[3])
        return (A + B) / (2.0 * C)