# Problem Statement

## Title
**Face Recognition Attendance System** — a CLI-based Computer Vision
project built with OpenCV.

## Problem Statement
Manual attendance-taking in classrooms, labs, and small workplaces is
slow, easy to falsify (proxy attendance), and produces data that is
tedious to compile into summaries. A lightweight, camera-based system
that can automatically detect and recognize enrolled individuals and log
their attendance would remove manual roll-calls, prevent proxy
attendance (a face has to actually be present and recognized), and
produce attendance data that is already structured and reportable.

## Scope of the Project
The project covers the full pipeline of a face-recognition-based
attendance workflow, implemented as an offline, CPU-only, fully
CLI-executable tool:

- **In scope**: face detection, dataset/enrollment management, LBPH
  model training, face recognition against webcam/video/image input,
  attendance logging with duplicate-prevention, CSV/analytics reporting,
  structured error handling, and logging.
- **Out of scope**: multi-camera deployment, cloud sync, mobile apps,
  liveness/anti-spoofing detection, and GPU-accelerated deep-learning
  face embeddings (a deliberate scope trade-off — see
  `docs/DESIGN.md` §7 for the rationale behind choosing Haar
  Cascade + LBPH over a heavier DNN pipeline).

## Target Users
- **Instructors / lab administrators** who need a quick, low-cost way to
  take attendance in a classroom, lab, or small workshop without
  dedicated hardware.
- **Small organizations / co-working spaces** wanting a simple
  check-in system without paying for a commercial SaaS product.
- **Students of Computer Vision** who want a complete, well-documented
  reference implementation of a classic detect → enroll → train →
  recognize → report pipeline built entirely with OpenCV.

## High-Level Features
1. **Enrollment module** — register a new person from a live webcam
   session or from an existing folder of photographs (including
   pre-cropped datasets).
2. **Training module** — build/rebuild an LBPH face-recognition model
   from all enrolled students' samples.
3. **Recognition & Attendance module** — run recognition against a
   webcam, a video file, or a single image; automatically mark
   attendance (once per day per person) for every confidently
   recognized face, and save an annotated output image/video.
4. **Reporting & Analytics module** — generate a daily attendance
   report and a date-range report (CSV export + a bar chart of
   attendance counts per student).
5. **Supporting infrastructure** — a typed configuration system, a
   custom exception hierarchy, rotating file + console logging, and an
   automated pytest test suite (21 tests) covering the database,
   configuration, detector, trainer/recognizer pipeline, and reporting
   modules.
