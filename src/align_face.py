import cv2
from insightface.utils import face_align


def align_face(image_bgr, landmarks):
    """Align face using 5-point landmarks, return 112x112 RGB crop."""
    aligned_bgr = face_align.norm_crop(image_bgr, landmarks)
    aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
    return aligned_rgb
