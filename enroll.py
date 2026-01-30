import argparse
import os
from typing import Iterable, List

import numpy as np

from src.align_face import align_face
from src.detect_face import detect_face
from src.embedding import extract_embedding
from src.input import load_image_rgb
from src.vector_store import VectorStore


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def iter_image_files(folder: str) -> Iterable[str]:
    for root, _, files in os.walk(folder):
        for name in sorted(files):
            ext = os.path.splitext(name)[1].lower()
            if ext in IMAGE_EXTS:
                yield os.path.join(root, name)


def list_person_dirs(enroll_root: str) -> List[str]:
    entries = []
    for name in sorted(os.listdir(enroll_root)):
        path = os.path.join(enroll_root, name)
        if os.path.isdir(path):
            entries.append(path)
    return entries


def main():
    parser = argparse.ArgumentParser(description="Enroll face embeddings from folder structure.")
    parser.add_argument("--enroll_dir", required=True, help="Root folder with person_id subfolders.")
    parser.add_argument("--db_dir", default="database", help="Output database directory.")
    args = parser.parse_args()

    index_path = os.path.join(args.db_dir, "embeddings.index")
    meta_path = os.path.join(args.db_dir, "metadata.txt")

    if os.path.exists(index_path) and os.path.exists(meta_path):
        store = VectorStore.load(index_path, meta_path)
        print(f"Loaded existing index with {len(store.person_ids)} embeddings.")
    else:
        store = VectorStore()

    person_dirs = list_person_dirs(args.enroll_dir)
    if not person_dirs:
        raise ValueError("No person_id subfolders found in enroll_dir.")

    added = 0
    skipped = 0
    for person_dir in person_dirs:
        person_id = os.path.basename(person_dir)
        embeddings = []
        for image_path in iter_image_files(person_dir):
            try:
                image_rgb = load_image_rgb(image_path)
                face, _, kps, image_bgr = detect_face(image_rgb)
                _ = align_face(image_bgr, kps)
                embedding = extract_embedding(face)
                embeddings.append(embedding)
                print(f"[OK] {person_id} <- {image_path}")
            except Exception as exc:
                skipped += 1
                print(f"[SKIP] {image_path}: {exc}")

        if not embeddings:
            print(f"[WARN] No valid embeddings for {person_id}, skipped.")
            continue
        mean_embedding = np.mean(np.stack(embeddings, axis=0), axis=0)
        norm = np.linalg.norm(mean_embedding)
        if norm > 0:
            mean_embedding = mean_embedding / norm
        store.add(mean_embedding.astype("float32"), person_id)
        added += 1

    store.save(index_path, meta_path)
    print(f"Enrollment complete. Added: {added}, Skipped: {skipped}")


if __name__ == "__main__":
    main()
