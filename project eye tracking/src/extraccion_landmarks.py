import mediapipe as mp
class ExtraccionLandmarks:
    def __init__(self):
        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

    def extraer(self, imagen_rgb):
        return self.face_mesh.process(imagen_rgb)