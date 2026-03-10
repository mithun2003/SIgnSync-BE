#!/usr/bin/env python3
"""
Custom Emergency Sign Data Collector  [OPTIONAL]
=================================================

NOTE: This script is OPTIONAL.  The primary SVM trainer (train_svm.py)
already generates synthetic landmark data for all emergency signs without
any webcam collection.  Use this script ONLY if you want to IMPROVE
emergency sign accuracy by adding real captured frames on top of the
synthetic data.

Captures training frames for emergency signs using your webcam:

    help       — Open flat palm, all fingers spread (universal "stop / help")
    danger     — ILY: index + pinky + thumb extended, middle + ring curled
    emergency  — Thumbs-up fist: thumb pointing straight upward

Frames are saved to  data/custom/<sign_name>/*.jpg  and are automatically
included when you run train_svm.py or train.py --skip-prep.

Usage
-----
  python collect_custom_data.py                     # collect all 3 emergency signs
  python collect_custom_data.py --sign help         # collect only 'help'
  python collect_custom_data.py --frames 500        # change target per sign
  python collect_custom_data.py --list              # show current collection status

Controls (webcam window)
-----------------------
  SPACE  — toggle capture on/off
  R      — reset (delete all captured frames for current sign)
  Q      — finish current sign / skip to next
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from config import (
    CAPTURE_FPS,
    CUSTOM_DATA_DIR,
    EMERGENCY_CLASSES,
    EMERGENCY_SIGN_INSTRUCTIONS,
    FRAMES_PER_SIGN,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _frame_count(sign: str) -> int:
    d = CUSTOM_DATA_DIR / sign
    if not d.is_dir():
        return 0
    return len([p for p in d.iterdir() if p.suffix.lower() == ".jpg"])


def _wrap_text(text: str, max_chars: int = 58) -> list[str]:
    words, line, lines = text.split(), "", []
    for w in words:
        if len(line) + len(w) + 1 > max_chars:
            lines.append(line.rstrip())
            line = w + " "
        else:
            line += w + " "
    if line:
        lines.append(line.rstrip())
    return lines


def _overlay(frame, sign: str, captured: int, target: int, capturing: bool) -> None:
    """Draw HUD overlay on *frame* (in-place)."""
    h, w = frame.shape[:2]

    # Top banner
    cv2.rectangle(frame, (0, 0), (w, 90), (20, 20, 20), -1)

    status_color = (0, 230, 0) if capturing else (0, 165, 255)
    status_text = "● REC" if capturing else "■ PAUSED"

    cv2.putText(frame, f"Sign: {sign.upper()}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(frame, f"Frames: {captured}/{target}", (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)
    cv2.putText(frame, status_text, (w - 140, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)

    # Instruction
    instruction = EMERGENCY_SIGN_INSTRUCTIONS.get(sign, f"Perform the '{sign}' sign.")
    lines = _wrap_text(instruction, max_chars=55)
    y = 115
    for ln in lines[:3]:
        cv2.putText(frame, ln, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 230, 180), 1)
        y += 22

    # Progress bar
    bar_x, bar_y, bar_w, bar_h = 12, h - 25, w - 24, 12
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
    filled = int(bar_w * min(captured / max(target, 1), 1.0))
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h), (0, 200, 80), -1)

    # Controls hint
    cv2.putText(
        frame, "SPACE=capture  R=reset  Q=next sign", (12, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (130, 130, 130), 1
    )


# ─────────────────────────────────────────────────────────────────────────────
# Per-sign collection
# ─────────────────────────────────────────────────────────────────────────────


def collect_sign(sign: str, target: int = FRAMES_PER_SIGN) -> int:
    """Open webcam and capture *target* frames for *sign*.

    Returns the total number of frames now on disk for this sign.
    """
    out_dir = CUSTOM_DATA_DIR / sign
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = _frame_count(sign)
    if existing >= target:
        print(f"  [{sign}] Already have {existing} frames — skipping.")
        return existing

    instruction = EMERGENCY_SIGN_INSTRUCTIONS.get(sign, f"Perform the '{sign}' sign.")
    print(f"\n{'─' * 60}")
    print(f"  Collecting: {sign.upper()}")
    print(f"  Target: {target}  (have {existing}, need {target - existing} more)")
    print(f"\n  {instruction}")
    print("\n  SPACE = toggle capture  |  R = reset  |  Q = done")
    print(f"{'─' * 60}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return existing

    captured = existing
    frame_idx = existing
    capturing = False
    last_saved = 0.0
    interval = 1.0 / CAPTURE_FPS

    try:
        while captured < target:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)  # mirror so it feels natural
            display = frame.copy()
            _overlay(display, sign, captured, target, capturing)
            cv2.imshow(f"Collecting: {sign}", display)

            now = time.time()
            if capturing and (now - last_saved) >= interval:
                path = out_dir / f"{sign}_{frame_idx:06d}.jpg"
                cv2.imwrite(str(path), frame)
                captured += 1
                frame_idx += 1
                last_saved = now

            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                capturing = not capturing
            elif key == ord("r"):
                for f in out_dir.glob("*.jpg"):
                    f.unlink()
                captured = frame_idx = 0
                print(f"  [{sign}] Reset — all frames deleted.")
            elif key == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"  [{sign}] {captured} frames saved to {out_dir}")
    if captured < 50:
        print(f"  WARNING: Only {captured} frames. At least 100 recommended.")
    return captured


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect custom emergency sign data")
    parser.add_argument("--sign", type=str, default=None, help="Sign to collect (default: all emergency signs)")
    parser.add_argument(
        "--frames", type=int, default=FRAMES_PER_SIGN, help=f"Target frames per sign (default: {FRAMES_PER_SIGN})"
    )
    parser.add_argument("--list", action="store_true", help="Show collection status and exit")
    args = parser.parse_args()

    if args.list:
        print("\nEmergency sign collection status:")
        print(f"  {'Sign':<12}  {'Frames':>7}  Instruction")
        print(f"  {'─' * 12}  {'─' * 7}  {'─' * 40}")
        for sign in EMERGENCY_CLASSES:
            n = _frame_count(sign)
            instr = EMERGENCY_SIGN_INSTRUCTIONS.get(sign, "—")[:50]
            print(f"  {sign:<12}  {n:>7}  {instr}")
        return

    signs: list[str] = [args.sign] if args.sign else EMERGENCY_CLASSES

    # Warn if a custom sign name is given that's not in the config
    for sign in signs:
        if sign not in EMERGENCY_CLASSES:
            print(
                f"NOTE: '{sign}' is not in EMERGENCY_CLASSES in config.py.  "
                "Add it there before training so it is included as a class."
            )

    print(f"\nCollecting data for: {signs}")
    print(f"Target: {args.frames} frames per sign")
    print(f"Output: {CUSTOM_DATA_DIR}\n")

    for sign in signs:
        collect_sign(sign, args.frames)

    print("\nDone!  Next steps:")
    print("  1. python train.py --skip-prep     (CNN, recommended)")
    print("  2. python train_svm.py             (SVM, fast alternative)")


if __name__ == "__main__":
    main()
