# Face Recognition MVP

Minimal face recognition pipeline using pretrained InsightFace models and FAISS,
with enrollment, recognition, and attendance reporting.

## Highlights
- Face detection, alignment, and embedding (RetinaFace + ArcFace)
- FAISS vector search with cosine similarity
- Batch enrollment from folder structure
- Single-image or webcam recognition
- Attendance session with expected list + absent report

## Contents
- Quick start
- Project layout
- Enrollment
- Recognition
- Attendance session
- Outputs

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Project layout
```
face_recognition_mvp/
  src/                    # core face pipeline
  enroll.py               # build the face database from enroll_root/
  recognize.py            # single image or webcam recognition
  session_attendance.py   # attendance session + report
  data/enrollments/       # sample CSVs for expected students
```

## Workflow
1. Enroll faces into a database.
2. Recognize from image or webcam.
3. Run attendance with expected list and report output.

## Enrollment
Folder structure:
```
enroll_root/
  person_a/
    img1.jpg
    img2.jpg
  person_b/
    img3.jpg
```

Build the database (averages embeddings per person folder):
```bash
python enroll.py --enroll_dir enroll_root --db_dir database
```

## Recognition
Single image:
```bash
python recognize.py --image path/to/test.jpg --db_dir database --threshold 0.6
```

Webcam:
```bash
python recognize.py --camera --db_dir database --threshold 0.6
```

Top-k results:
```bash
python recognize.py --image path/to/test.jpg --db_dir database --top_k 3
```

Save a visualization:
```bash
python recognize.py --image path/to/test.jpg --db_dir database --save_vis --output outputs/result.jpg
```

## Attendance session
Run a session that loads the enrolled students once, counts only expected identities,
and produces a final attendance report with PRESENT/ABSENT for each student.

Expected-students CSV must include a `student_id` (or `id`) column. Optional `full_name`.
Sample lists are provided in `data/enrollments/`.

Camera session:
```bash
python session_attendance.py \
  --faculty "Engineering" \
  --department "Computer Science" \
  --course "CS101" \
  --expected_csv data/enrollments/CS101.csv \
  --camera \
  --db_dir database \
  --threshold 0.6
```

Process a folder of images:
```bash
python session_attendance.py \
  --faculty "Engineering" \
  --department "Computer Science" \
  --course "CS101" \
  --expected_csv data/enrollments/CS101.csv \
  --images_dir path/to/session_images \
  --db_dir database \
  --threshold 0.6
```

## Outputs
| Artifact | Path |
| --- | --- |
| Event logs | `attendance_logs/` |
| Attendance reports | `attendance_reports/` |
| Recognition visualizations | `outputs/` |
