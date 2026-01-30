import numpy as np


def extract_embedding(face_obj) -> np.ndarray:
    """
    Extract L2-normalized embedding from a detected face object.
    """
    if hasattr(face_obj, "normed_embedding") and face_obj.normed_embedding is not None:
        return face_obj.normed_embedding.astype("float32")
    if hasattr(face_obj, "embedding") and face_obj.embedding is not None:
        embedding = face_obj.embedding.astype("float32")
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding.astype("float32")
    raise ValueError("Face object does not contain an embedding.")
