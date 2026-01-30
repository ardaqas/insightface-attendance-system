import argparse
import csv
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

import cv2

from src.align_face import align_face
from src.decision import decide_identity
from src.detect_face import detect_face
from src.embedding import extract_embedding
from src.input import load_image_rgb
from src.similarity import find_best_match
from src.vector_store import VectorStore


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class ExpectedStudent:
    student_id: str
    full_name: Optional[str] = None
    normalized_id: str = ""


@dataclass
class RecognitionEvent:
    timestamp: str
    identity: str
    similarity: float
    match_status: str
    counted_present: str
    error: Optional[str] = None


class AttendanceState:
    def __init__(self, expected_ids: List[str]) -> None:
        self.expected_ids = set(expected_ids)
        self.seen_ids = set()

    def consider(self, identity: str) -> bool:
        normalized = normalize_identity(identity)
        if normalized == "unknown":
            return False
        if normalized not in self.expected_ids:
            return False
        if normalized in self.seen_ids:
            return False
        self.seen_ids.add(normalized)
        return True

    def absent_ids(self) -> List[str]:
        return sorted(self.expected_ids - self.seen_ids)


def current_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def normalize_identity(value: str) -> str:
    normalized = value.strip().lower()
    normalized = normalized.replace("_", " ")
    normalized = " ".join(normalized.split())
    return normalized


def iter_image_files(folder: str) -> Iterable[str]:
    for root, _, files in os.walk(folder):
        for name in sorted(files):
            ext = os.path.splitext(name)[1].lower()
            if ext in IMAGE_EXTS:
                yield os.path.join(root, name)


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
        return None
    image_rgb = cv2.cvtColor(captured_frame, cv2.COLOR_BGR2RGB)
    return image_rgb


def load_expected_students(csv_path: str) -> Tuple[List[ExpectedStudent], List[str]]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Expected-students CSV not found: {csv_path}")
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("Expected-students CSV must include a header row.")
        fieldnames = [name.strip() for name in reader.fieldnames if name]
        if "student_id" in fieldnames:
            id_field = "student_id"
        elif "id" in fieldnames:
            id_field = "id"
        else:
            raise ValueError("Expected-students CSV must include student_id or id column.")

        name_field = "full_name" if "full_name" in fieldnames else None
        expected: List[ExpectedStudent] = []
        ids: List[str] = []
        seen = set()
        for row in reader:
            student_id = (row.get(id_field) or "").strip()
            if not student_id:
                continue
            if student_id in seen:
                raise ValueError(f"Duplicate student_id in expected list: {student_id}")
            seen.add(student_id)
            full_name = (row.get(name_field) or "").strip() if name_field else None
            normalized_id = normalize_identity(student_id)
            expected.append(
                ExpectedStudent(
                    student_id=student_id,
                    full_name=full_name or None,
                    normalized_id=normalized_id,
                )
            )
            ids.append(normalized_id)
    if not expected:
        raise ValueError("Expected-students CSV contains no valid student rows.")
    return expected, ids


def ensure_parent_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def write_event_header(path: str) -> None:
    ensure_parent_dir(path)
    if os.path.exists(path):
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "timestamp",
                "faculty",
                "department",
                "course",
                "session_id",
                "source",
                "identity",
                "similarity",
                "match_status",
                "counted_present",
                "error",
            ]
        )


def append_event(
    path: str,
    faculty: str,
    department: str,
    course: str,
    session_id: str,
    source: str,
    event: RecognitionEvent,
) -> None:
    write_event_header(path)
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                event.timestamp,
                faculty,
                department,
                course,
                session_id,
                source,
                event.identity,
                f"{event.similarity:.4f}",
                event.match_status,
                event.counted_present,
                event.error or "",
            ]
        )


def write_attendance_report(
    path: str,
    expected: List[ExpectedStudent],
    present_ids: set,
    faculty: str,
    department: str,
    course: str,
    session_date: str,
) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["student_id", "full_name", "status", "date", "course", "department", "faculty"]
        )
        for student in expected:
            status = "PRESENT" if student.normalized_id in present_ids else "ABSENT"
            writer.writerow(
                [
                    student.student_id,
                    student.full_name or "",
                    status,
                    session_date,
                    course,
                    department,
                    faculty,
                ]
            )


def recognize_identity(image_rgb, store: VectorStore, threshold: float) -> Tuple[str, float]:
    face, _, kps, image_bgr = detect_face(image_rgb)
    _ = align_face(image_bgr, kps)
    embedding = extract_embedding(face)
    best_id, score = find_best_match(store, embedding)
    identity = decide_identity(best_id, score, threshold=threshold)
    return identity, score


def process_event(
    image_rgb,
    store: VectorStore,
    state: AttendanceState,
    threshold: float,
) -> RecognitionEvent:
    timestamp = current_timestamp()
    try:
        identity, similarity = recognize_identity(image_rgb, store, threshold)
        counted = state.consider(identity)
        return RecognitionEvent(
            timestamp=timestamp,
            identity=identity,
            similarity=similarity,
            match_status="recognized" if identity != "unknown" else "unknown",
            counted_present="yes" if counted else "no",
        )
    except Exception as exc:
        return RecognitionEvent(
            timestamp=timestamp,
            identity="unknown",
            similarity=0.0,
            match_status="unknown",
            counted_present="no",
            error=str(exc),
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a face-recognition attendance session (CLI)."
    )
    parser.add_argument("--faculty", required=True, help="Faculty name.")
    parser.add_argument("--department", required=True, help="Department name.")
    parser.add_argument("--course", required=True, help="Course code (e.g., CS101).")
    parser.add_argument("--expected_csv", required=True, help="CSV of enrolled students.")
    parser.add_argument("--db_dir", default="database", help="Database directory.")
    parser.add_argument("--threshold", type=float, default=0.6, help="Acceptance threshold.")
    parser.add_argument("--camera", action="store_true", help="Use webcam for capture.")
    parser.add_argument("--camera_index", type=int, default=0, help="Camera index.")
    parser.add_argument("--images_dir", help="Process all images from a folder.")
    parser.add_argument("--log_dir", default="attendance_logs", help="Event log root.")
    parser.add_argument("--report_dir", default="attendance_reports", help="Report root.")
    args = parser.parse_args()

    if not args.camera and not args.images_dir:
        raise ValueError("Provide --camera or --images_dir.")
    if args.camera and args.images_dir:
        raise ValueError("Use either --camera or --images_dir, not both.")

    expected, expected_ids = load_expected_students(args.expected_csv)
    state = AttendanceState(expected_ids)

    index_path = os.path.join(args.db_dir, "embeddings.index")
    meta_path = os.path.join(args.db_dir, "metadata.txt")
    store = VectorStore.load(index_path, meta_path)

    session_id = datetime.now().strftime("%Y%m%dT%H%M%S")
    session_date = datetime.now().strftime("%Y-%m-%d")
    log_path = os.path.join(args.log_dir, args.course, session_date, f"events_{session_id}.csv")
    report_path = os.path.join(
        args.report_dir, args.course, session_date, f"attendance_{session_id}.csv"
    )

    print(f"expected students: {len(expected_ids)}")
    print(f"session_id: {session_id}")

    processed_events = 0

    if args.images_dir:
        if not os.path.isdir(args.images_dir):
            raise FileNotFoundError(f"Images folder not found: {args.images_dir}")
        image_paths = list(iter_image_files(args.images_dir))
        if not image_paths:
            raise ValueError(f"No images found in: {args.images_dir}")
        for image_path in image_paths:
            image_rgb = load_image_rgb(image_path)
            event = process_event(image_rgb, store, state, args.threshold)
            append_event(
                log_path,
                args.faculty,
                args.department,
                args.course,
                session_id,
                image_path,
                event,
            )
            processed_events += 1
            print(
                f"[{event.match_status}] {image_path} -> {event.identity} "
                f"({event.similarity:.4f}) counted={event.counted_present}"
            )
    else:
        print("Press Enter to capture, or type q to finish the session.")
        while True:
            user_input = input("> ").strip().lower()
            if user_input in {"q", "quit", "exit", "esc", "escape", "\x1b", "^["}:
                break
            image_rgb = capture_from_camera(args.camera_index)
            if image_rgb is None:
                print("Capture cancelled; continuing.")
                continue
            event = process_event(image_rgb, store, state, args.threshold)
            append_event(
                log_path,
                args.faculty,
                args.department,
                args.course,
                session_id,
                f"camera:{args.camera_index}",
                event,
            )
            processed_events += 1
            print(
                f"[{event.match_status}] {event.identity} "
                f"({event.similarity:.4f}) counted={event.counted_present}"
            )

    if processed_events == 0:
        print("No attendance events captured; report will mark all absent.")

    write_attendance_report(
        report_path,
        expected,
        state.seen_ids,
        faculty=args.faculty,
        department=args.department,
        course=args.course,
        session_date=session_date,
    )

    absent = state.absent_ids()
    print(f"present: {len(state.seen_ids)} / {len(expected_ids)}")
    print(f"absent: {len(absent)}")
    print(f"event_log: {log_path}")
    print(f"attendance_report: {report_path}")


if __name__ == "__main__":
    main()
