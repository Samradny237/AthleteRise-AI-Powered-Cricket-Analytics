# AthleteRise – Real-Time Cover Drive Analysis

This project processes a full cricket video in real-time, runs pose estimation frame-by-frame using MediaPipe, overlays biomechanical metrics live, and produces:
- An annotated `.mp4` video with pose skeleton and feedback cues.
- A final evaluation file (`evaluation.json`) scoring multiple shot categories.

## Features
- Full video processing without keyframe extraction.
- Pose estimation for head, shoulders, elbows, wrists, hips, knees, and ankles.
- Biomechanical metrics:
  - Elbow angle (shoulder-elbow-wrist)
  - Spine lean vs vertical
  - Head-over-knee alignment
  - Front foot direction
- Live overlays with numeric readouts and feedback cues.
- Final category scores and actionable feedback.

## Installation
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt