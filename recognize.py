import argparse
import os

import cv2

from src.align_face import align_face
from src.decision import decide_identity
from src.detect_face import detect_face
from src.embedding import extract_embedding
from src.input import load_image_rgb
from src.similarity import find_best_match, find_top_matches
from src.vector_store import VectorStore


def capture_from_camera(camera_index: int = 0):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")

    print("Camera opened. Press Space to capture, Esc to cancel.")
    captured_frame = None
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.imshow("Capture", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == 32 or key == 13:
            captured_frame = frame
            break

    cap.release()
    cv2.destroyAllWindows()
    if captured_frame is None:
        raise RuntimeError("Capture cancelled.")
    image_rgb = cv2.cvtColor(captured_frame, cv2.COLOR_BGR2RGB)
    return image_rgb


def annotate_frame(image_bgr, bbox, label: str):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    color = (0, 200, 0)
    cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, 2)
    text = label
    (text_w, text_h), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    )
    cv2.rectangle(
        image_bgr,
        (x1, y1 - text_h - baseline - 6),
        (x1 + text_w + 6, y1),
        color,
        -1,
    )
    cv2.putText(
        image_bgr,
        text,
        (x1 + 3, y1 - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    return image_bgr


def main():
    parser = argparse.ArgumentParser(description="Recognize a face from a single image.")
    parser.add_argument("--image", help="Path to input image.")
    parser.add_argument("--camera", action="store_true", help="Use webcam to capture image.")
    parser.add_argument("--camera_index", type=int, default=0, help="Camera index (default 0).")
    parser.add_argument("--db_dir", default="database", help="Database directory.")
    parser.add_argument("--threshold", type=float, default=0.6, help="Acceptance threshold.")
    parser.add_argument("--top_k", type=int, default=1, help="Number of top matches.")
    parser.add_argument(
        "--save_vis",
        action="store_true",
        help="Save visualization with bounding box and label.",
    )
    parser.add_argument(
        "--output",
        default="outputs/recognized.jpg",
        help="Output path for visualization.",
    )
    parser.add_argument("--show", action="store_true", help="Show visualization window.")
    args = parser.parse_args()

    if not args.image and not args.camera:
        raise ValueError("Provide --image or use --camera.")
    if args.image and args.camera:
        raise ValueError("Use either --image or --camera, not both.")

    index_path = os.path.join(args.db_dir, "embeddings.index")
    meta_path = os.path.join(args.db_dir, "metadata.txt")
    store = VectorStore.load(index_path, meta_path)

    if args.camera:
        image_rgb = capture_from_camera(args.camera_index)
    else:
        image_rgb = load_image_rgb(args.image)
    face, bbox, kps, image_bgr = detect_face(image_rgb)
    _ = align_face(image_bgr, kps)
    embedding = extract_embedding(face)

    if args.top_k > 1:
        ids, scores = find_top_matches(store, embedding, top_k=args.top_k)
        best_id, score = ids[0], scores[0]
    else:
        best_id, score = find_best_match(store, embedding)
    identity = decide_identity(best_id, score, threshold=args.threshold)

    print(f"identity: {identity}")
    print(f"similarity: {score:.4f}")
    if args.top_k > 1:
        for rank, (pid, sim) in enumerate(zip(ids, scores), start=1):
            print(f"top_{rank}: {pid} ({sim:.4f})")

    if args.save_vis or args.show:
        label = f"{identity} ({score:.2f})"
        vis = annotate_frame(image_bgr.copy(), bbox, label)
        if args.save_vis:
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            cv2.imwrite(args.output, vis)
            print(f"saved visualization: {args.output}")
        if args.show:
            cv2.imshow("Recognition", vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
