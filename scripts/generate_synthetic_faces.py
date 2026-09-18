#!/usr/bin/env python3
"""
Generate synthetic "face-like" test images.

Real face photographs are personal biometric data, so this repository
intentionally ships NO real faces. To let anyone try the full pipeline
end-to-end (enroll -> train -> recognize -> report) without a webcam or
copyrighted/private photos, this script procedurally draws simple
cartoon faces with per-identity variation (skin tone, eye spacing,
mouth shape, etc.) using only OpenCV drawing primitives.

This is for demonstration / automated testing purposes only. For real
attendance use, enroll actual students via `python main.py enroll`.

Usage:
    python scripts/generate_synthetic_faces.py --out data/sample_faces --people 3 --images 10
"""
from __future__ import annotations

import argparse
import os
import random

import cv2
import numpy as np


def draw_face(seed: int, size: int = 300) -> np.ndarray:
    rng = random.Random(seed)
    img = np.full((size, size, 3), 245, dtype=np.uint8)

    skin = (rng.randint(150, 220), rng.randint(150, 210), rng.randint(140, 200))
    cx, cy = size // 2, size // 2
    face_w, face_h = rng.randint(90, 110), rng.randint(110, 130)
    cv2.ellipse(img, (cx, cy), (face_w, face_h), 0, 0, 360, skin, -1)

    eye_dx = rng.randint(28, 40)
    eye_dy = rng.randint(-15, -5)
    eye_r = rng.randint(8, 12)
    for sign in (-1, 1):
        ex, ey = cx + sign * eye_dx, cy + eye_dy
        cv2.circle(img, (ex, ey), eye_r, (255, 255, 255), -1)
        cv2.circle(img, (ex, ey), eye_r // 2, (40, 40, 40), -1)

    brow_dy = eye_dy - eye_r - 6
    for sign in (-1, 1):
        bx = cx + sign * eye_dx
        cv2.line(img, (bx - 12, cy + brow_dy), (bx + 12, cy + brow_dy), (60, 40, 30), 3)

    nose_len = rng.randint(20, 30)
    cv2.line(img, (cx, cy - 5), (cx, cy - 5 + nose_len), (skin[0] - 30, skin[1] - 30, skin[2] - 30), 2)

    mouth_w = rng.randint(30, 45)
    mouth_y = cy + rng.randint(45, 60)
    cv2.ellipse(img, (cx, mouth_y), (mouth_w, 10), 0, 0, 180, (120, 60, 60), 3)

    hair_style = rng.choice(["short", "long", "bald"])
    hair_color = rng.choice([(30, 30, 30), (90, 60, 40), (200, 200, 200)])
    if hair_style == "short":
        cv2.ellipse(img, (cx, cy - face_h + 15), (face_w + 5, 40), 0, 180, 360, hair_color, -1)
    elif hair_style == "long":
        cv2.ellipse(img, (cx, cy - face_h + 15), (face_w + 5, 40), 0, 180, 360, hair_color, -1)
        cv2.rectangle(img, (cx - face_w - 5, cy - 30), (cx - face_w + 15, cy + 70), hair_color, -1)
        cv2.rectangle(img, (cx + face_w - 15, cy - 30), (cx + face_w + 5, cy + 70), hair_color, -1)

    noise = rng.randint(-8, 8)
    img = np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)
    angle = rng.uniform(-6, 6)
    m = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    img = cv2.warpAffine(img, m, (size, size), borderValue=(245, 245, 245))
    return img


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/sample_faces", help="Output root folder")
    parser.add_argument("--people", type=int, default=3, help="Number of synthetic identities")
    parser.add_argument("--images", type=int, default=10, help="Images per identity")
    args = parser.parse_args()

    names = ["person_a", "person_b", "person_c", "person_d", "person_e"]
    for p in range(args.people):
        person_name = names[p % len(names)] + (f"_{p}" if p >= len(names) else "")
        person_dir = os.path.join(args.out, person_name)
        os.makedirs(person_dir, exist_ok=True)
        base_seed = p * 1000
        for i in range(args.images):
            img = draw_face(seed=base_seed + i)
            cv2.imwrite(os.path.join(person_dir, f"{i:03d}.png"), img)
        print(f"[OK] Wrote {args.images} synthetic images to {person_dir}")


if __name__ == "__main__":
    main()
