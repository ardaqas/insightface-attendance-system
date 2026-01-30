from typing import Tuple

import cv2
import numpy as np
from insightface.app import FaceAnalysis

_FACE_ANALYZER = None


def get_face_analyzer() -> FaceAnalysis:
    global _FACE_ANALYZER
    if _FACE_ANALYZER is None:
        analyzer = FaceAnalysis(name="buffalo_l")
        analyzer.prepare(ctx_id=-1, det_size=(640, 640))
        _FACE_ANALYZER = analyzer
    return _FACE_ANALYZER


def _largest_face(faces):
    if len(faces) == 1:
        return faces[0]
    areas = []
    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        areas.append((x2 - x1) * (y2 - y1))
    return faces[int(np.argmax(areas))]


def detect_face(image_rgb) -> Tuple[object, np.ndarray, np.ndarray, np.ndarray]:
    """
    Detect face in an RGB image.

    Returns:
        face_obj, bbox, landmarks (5x2), image_bgr
    """
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    analyzer = get_face_analyzer()
    faces = analyzer.get(image_bgr)
    if not faces:
        raise ValueError("No face detected in image.")
    face = _largest_face(faces)
    bbox = face.bbox.astype(int)
    kps = face.kps.astype(int)
    return face, bbox, kps, image_bgr
