# Face Recognition Attendance System

A fully command-line, offline, CPU-only **Computer Vision** project built
with **OpenCV**: enroll people from a webcam or photos, train a face
recognition model, run recognition against a webcam / video / image, and
automatically log & report attendance — no GUI, no cloud services, no
paid APIs required.

> Built as a course project (VITyarthi "Build Your Own Project" — Computer
> Vision domain). See [`statement.md`](statement.md) for the full problem
> statement and [`docs/DESIGN.md`](docs/DESIGN.md) for architecture and
> UML diagrams.

---

## Overview

The system implements the classic CV pipeline **detect → enroll → train →
recognize → report**:

1. Faces are located in a frame using OpenCV's Haar Cascade classifier.
2. During **enrollment**, cropped/normalized face samples are saved per
   person (either captured live from a webcam or built from a folder of
   existing photos).
3. **Training** builds an LBPH (Local Binary Patterns Histograms) face
   recognizer from every enrolled person's samples.
4. **Recognition** runs the trained model against a webcam session, a
   video file, or a single image, draws bounding boxes + labels, and
   automatically marks attendance (once per person per day) in a SQLite
   database for every confidently recognized face.
5. **Reporting** turns the logged attendance into a CSV export and a bar
   chart of attendance counts.

---

## Features

- **Enrollment module** — webcam capture *or* folder-of-photos
  enrollment (including a `--cropped` mode for pre-cropped datasets).
- **Training module** — one command (re)builds the LBPH model from all
  currently enrolled students.
- **Recognition & Attendance module** — works against a webcam, a video
  file, or a single image; saves an annotated image/video with
  bounding boxes and predicted names.
- **Duplicate-safe attendance logging** — a `UNIQUE(student_id, date)`
  SQLite constraint guarantees a person is marked present at most once
  per day, no matter how many times they're recognized.
- **Reporting & analytics** — daily report and arbitrary date-range
  report, each exported to CSV, with a matplotlib bar chart for range
  reports.
- **Robust error handling** — a custom exception hierarchy
  (`src/exceptions.py`) turns camera failures, missing models, bad
  images, and duplicate enrollments into clear CLI error messages
  instead of stack traces.
- **Logging & monitoring** — every action is logged to both the console
  and a rotating log file (`logs/app.log`).
- **Automated tests** — 21 pytest tests covering the database layer,
  configuration loader, face detector, the full
  enroll→train→recognize pipeline, and the reporting module.
- **Privacy-conscious demo data** — no real face photographs are
  committed to the repository; a synthetic placeholder-face generator
  (`scripts/generate_synthetic_faces.py`) lets anyone exercise the full
  pipeline without a webcam or real photos.

---

## Technologies / Tools Used

| Purpose | Tool / Library |
|---|---|
| Language | Python 3.10+ |
| Computer Vision | OpenCV (`opencv-contrib-python`) — Haar Cascade detection, LBPH recognition |
| Numerical | NumPy |
| Data / reporting | pandas (tabular I/O), matplotlib (charts) |
| Storage | SQLite (via Python's built-in `sqlite3`) |
| Configuration | PyYAML |
| Testing | pytest |
| CLI | argparse (standard library) |

---

## Project Structure

```
face-attendance-cv/
├── main.py                      # CLI entry point (enroll / train / recognize / report / list-students)
├── config.yaml                  # All tunable paths & parameters
├── requirements.txt
├── statement.md                 # Problem statement, scope, target users
├── README.md                    # You are here
├── src/
│   ├── config.py                # Typed config loader
│   ├── logger.py                # Console + rotating file logging
│   ├── exceptions.py            # Custom exception hierarchy
│   ├── database.py               # SQLite persistence (students, attendance)
│   ├── face_detector.py          # Haar-cascade face detection
│   ├── face_trainer.py           # Dataset building + LBPH training
│   ├── face_recognizer.py        # LBPH inference
│   ├── attendance_manager.py     # Orchestrates detect+recognize+log over webcam/video/image
│   └── report_generator.py       # CSV export + attendance charts
├── scripts/
│   └── generate_synthetic_faces.py   # Privacy-safe placeholder face generator (demo/testing)
├── tests/                        # pytest suite (21 tests)
├── docs/
│   ├── DESIGN.md                 # Architecture, workflow, UML & ER diagrams
│   └── diagrams/                 # Rendered PNG diagrams
├── data/
│   ├── dataset/                  # Per-student face samples (created at runtime, gitignored)
│   └── trained_model/            # Saved LBPH model + label map (created at runtime, gitignored)
├── reports/                      # Generated CSV/chart reports (created at runtime)
├── outputs/                      # Annotated recognition output images/video (created at runtime)
└── logs/                         # Rotating application log (created at runtime)
```

---

## Setup & Installation

### 1. Prerequisites
- Python **3.10 or newer**
- `pip`
- A webcam is **optional** — every command also works against video
  files, image files, or the bundled synthetic-data generator.

### 2. Clone the repository
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 3. Create a virtual environment (recommended)
```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Verify the install
```bash
python main.py -h
```
You should see the CLI's help text listing the `enroll`, `train`,
`recognize`, `report`, and `list-students` commands.

No further configuration is required — `config.yaml` ships with sensible
defaults, and every directory the app needs (`data/`, `reports/`,
`outputs/`, `logs/`) is created automatically on first run.

---

## How to Run

### Quick demo (no webcam / no real photos required)
This generates privacy-safe synthetic placeholder faces so you can see
the *entire* pipeline run end-to-end in under a minute:

```bash
# 1. Generate a small synthetic dataset (3 identities, 12 images each)
python scripts/generate_synthetic_faces.py --out data/sample_faces --people 3 --images 12

# 2. Enroll each synthetic identity (images are already face-cropped -> use --cropped)
python main.py enroll --id STU001 --name "Aditi Sharma" --source images --path data/sample_faces/person_a --cropped
python main.py enroll --id STU002 --name "Rohan Verma"  --source images --path data/sample_faces/person_b --cropped
python main.py enroll --id STU003 --name "Priya Nair"   --source images --path data/sample_faces/person_c --cropped

# 3. Train the recognition model
python main.py train

# 4. Run recognition on a sample image and mark attendance
python main.py recognize --source image --path data/sample_faces/person_a/010.png --cropped

# 5. See who's been marked present today
python main.py report --type daily

# 6. See a 7-day summary with a chart
python main.py report --type range --days 7
```

### Real-world usage (with your own webcam or photos)

**Enroll a real person from a webcam** (captures 30 samples by default):
```bash
python main.py enroll --id STU101 --name "Your Name" --source webcam --samples 30
```

**Enroll a real person from a folder of un-cropped photos** (each photo
must show that person's face clearly; do *not* use `--cropped` here —
the app will run Haar-cascade detection to find and crop the face):
```bash
python main.py enroll --id STU102 --name "Another Person" --source images --path /path/to/their/photos
```

**Train the model** (required after every new enrollment):
```bash
python main.py train
```

**Recognize + mark attendance from a live webcam for 20 seconds:**
```bash
python main.py recognize --source webcam --duration 20
```
Add `--display` if your environment has a GUI and you want a live
preview window; omit it to run fully headless (the default).

**Recognize from a video file:**
```bash
python main.py recognize --source video --path /path/to/classroom.mp4
```

**List all enrolled students:**
```bash
python main.py list-students
```

**Full command reference:**
```bash
python main.py -h
python main.py enroll -h
python main.py train -h
python main.py recognize -h
python main.py report -h
```

---

## Instructions for Testing

The project ships an automated pytest suite (21 tests) that runs
entirely offline against a temporary, isolated copy of the app's data
directories — it never touches your real `data/` folder.

```bash
pip install -r requirements.txt   # pytest is included
python -m pytest tests/ -v
```

Expected result: **21 passed**. The suite covers:
- `test_database.py` — student registration, duplicate prevention,
  once-per-day attendance de-duplication, date-range queries.
- `test_config.py` — configuration loading and directory bootstrapping.
- `test_face_detector.py` — detection on blank frames, face cropping/
  normalization, largest-face selection.
- `test_pipeline_integration.py` — full enroll → train → recognize
  pipeline against the synthetic dataset generator.
- `test_report_generator.py` — CSV export correctness and chart
  generation for daily/range reports.

---

## Configuration Reference (`config.yaml`)

| Key | Meaning |
|---|---|
| `paths.*` | Where data/model/db/reports/logs live (relative to repo root) |
| `detection.scale_factor`, `min_neighbors`, `min_size` | Haar cascade detection tuning |
| `capture.face_size`, `samples_per_person` | Normalized face size & default webcam sample count |
| `recognition.confidence_threshold` | LBPH distance cutoff — **lower** predicted values are better matches; predictions above this threshold are reported as `Unknown` |
| `attendance.mark_once_per_day` | Enforces one attendance record per student per day |
| `logging.level` | Console/file log verbosity |

---

## Non-Functional Requirements Addressed

- **Performance** — Haar cascade + LBPH are both lightweight, CPU-only
  algorithms; frame-skipping (`--frame-skip`) keeps video processing
  fast.
- **Reliability / Error handling** — a dedicated exception hierarchy
  (camera failures, missing models, bad images, duplicate students)
  is caught at the CLI boundary and reported as clean, actionable
  messages instead of stack traces.
- **Security / Privacy** — no real biometric (face) data is committed to
  the repository; enrollment data lives only in the user's local
  `data/` folder (gitignored).
- **Usability** — a single, self-documenting CLI (`-h` on every
  subcommand) with sensible defaults; no configuration is required to
  get started.
- **Maintainability** — small, single-responsibility modules, type
  hints, docstrings explaining *why* (not just what), and a config file
  instead of hard-coded constants.
- **Logging & Monitoring** — every enrollment, training run, recognition
  event, and attendance mark is logged with a timestamp to a rotating
  log file.
- **Scalability** — adding a new student is O(1) work (one folder, one
  DB row); the LBPH model retrains on the full gallery in well under a
  second for galleries of this scale.

---

## Screenshots

See [`screenshots/`](screenshots/) for CLI output captures, the
generated attendance bar chart, and an annotated recognition output
image. (Rendered from an actual run of this codebase against the
bundled synthetic demo data.)

---

## Future Enhancements
- Swap LBPH for a deep-learning face-embedding model (e.g. FaceNet/
  ArcFace) for higher accuracy at larger gallery sizes.
- Add liveness/anti-spoofing checks (blink detection, texture analysis)
  to prevent attendance being marked from a photo held up to the
  camera.
- A lightweight web dashboard for reviewing reports (currently CLI +
  CSV/PNG only, by design, to satisfy the "fully executable via command
  line" requirement).
- Multi-camera / multi-room support with a shared central database.

---

## License
This project is submitted as academic coursework. See
[`LICENSE`](LICENSE) for terms.
