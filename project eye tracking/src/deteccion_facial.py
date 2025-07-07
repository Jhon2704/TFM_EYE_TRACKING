import mediapipe as mp

class DeteccionFacial:
    def __init__(self, confidence=0.6):
        mp_face_detection = mp.solutions.face_detection
        self.detector = mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=confidence
        )

    def detectar(self, imagen_rgb):
        return self.detector.process(imagen_rgb)