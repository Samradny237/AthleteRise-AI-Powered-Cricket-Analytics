
"""
AthleteRise – Real-Time Cover Drive Analysis from Full Video

Base requirements implemented:
 1) Full video ingestion (URL via YouTube Short or local path) and sequential processing
 2) MediaPipe Pose per-frame keypoints with graceful fallbacks
 3) Biomechanical metrics (elbow angle, spine lean, head-over-knee alignment, front-foot direction)
 4) Live overlays with real-time numeric readouts + cue messages
 5) Final multi-category scoring + feedback saved to evaluation.json

Extras (minor):
 - Basic FPS logging; optional temporal smoothing (EMA) for display metrics
 - CLI with configurable front side (left/right), thresholds, and output directory

Author: You
"""

import os
import sys
import math
import json
import time
import argparse
import tempfile
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# --- Video download (yt-dlp, robust vs pytube) ---------------------------------
# We import lazily to avoid mandatory dependency if user passes a local file.

def download_video_with_ytdlp(url: str, out_dir: str) -> str:
    """Download a video URL to MP4 using yt_dlp. Returns local file path.
    Requires `yt-dlp` installed (declared in requirements.txt).
    """
    import yt_dlp  # type: ignore

    os.makedirs(out_dir, exist_ok=True)
    out_tmpl = os.path.join(out_dir, "input.%(ext)s")

    ydl_opts = {
        "outtmpl": out_tmpl,
        "format": "mp4/bestvideo+bestaudio/best",
        "merge_output_format": "mp4",
        "quiet": True,
        "noprogress": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        result_path = ydl.prepare_filename(info)
        # normalize extension to mp4 if needed
        base, _ = os.path.splitext(result_path)
        mp4_path = base + ".mp4"
        if os.path.exists(mp4_path):
            return mp4_path
        return result_path

# --- Geometry helpers -----------------------------------------------------------

def _to_np(point) -> np.ndarray:
    return np.array(point, dtype=float)

def angle(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> Optional[float]:
    """Return angle ABC in degrees given points a, b, c. Returns None if invalid."""
    try:
        a, b, c = _to_np(a), _to_np(b), _to_np(c)
        ba = a - b
        bc = c - b
        if np.linalg.norm(ba) < 1e-6 or np.linalg.norm(bc) < 1e-6:
            return None
        cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        cosang = np.clip(cosang, -1.0, 1.0)
        return float(np.degrees(np.arccos(cosang)))
    except Exception:
        return None

def line_angle_deg(p1: Tuple[float, float], p2: Tuple[float, float]) -> Optional[float]:
    """Angle (degrees) of vector p1->p2 relative to x-axis. 0° points right."""
    try:
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return float(np.degrees(np.arctan2(dy, dx)))
    except Exception:
        return None

# Angle of a line vs vertical axis (positive = lean to right if y down)
def angle_vs_vertical_deg(p1: Tuple[float, float], p2: Tuple[float, float]) -> Optional[float]:
    ang = line_angle_deg(p1, p2)
    if ang is None:
        return None
    # angle vs vertical: line vs (0, -1) or (0, 1). Convert x-axis angle to |90 - ang|.
    return float(abs(90.0 - abs(ang)))

# --- MediaPipe Pose wrapper -----------------------------------------------------
class PoseEstimator:
    def __init__(self, static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        import mediapipe as mp
        self.mp = mp
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.landmark_enum = mp.solutions.pose.PoseLandmark

    def infer(self, frame_bgr: np.ndarray) -> Tuple[Optional[Dict[str, Tuple[int, int]]], Optional[np.ndarray]]:
        # MediaPipe expects RGB
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)
        if not res.pose_landmarks:
            return None, None
        h, w = frame_bgr.shape[:2]
        pts = {}
        for lm in self.landmark_enum:
            l = res.pose_landmarks.landmark[lm.value]
            if 0 <= l.x <= 1 and 0 <= l.y <= 1 and l.visibility > 0.3:
                pts[lm.name] = (int(l.x * w), int(l.y * h))
        return pts, res.pose_landmarks

    def draw(self, frame: np.ndarray, landmarks) -> None:
        if landmarks is None:
            return
        mp = self.mp
        mp.solutions.drawing_utils.draw_landmarks(
            frame,
            landmarks,
            mp.solutions.pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style(),
        )

# --- Metric computation ---------------------------------------------------------

POSE_KEYS = {
    "LEFT": {
        "SHOULDER": "LEFT_SHOULDER",
        "ELBOW": "LEFT_ELBOW",
        "WRIST": "LEFT_WRIST",
        "HIP": "LEFT_HIP",
        "KNEE": "LEFT_KNEE",
        "ANKLE": "LEFT_ANKLE",
        "FOOT": "LEFT_FOOT_INDEX",
    },
    "RIGHT": {
        "SHOULDER": "RIGHT_SHOULDER",
        "ELBOW": "RIGHT_ELBOW",
        "WRIST": "RIGHT_WRIST",
        "HIP": "RIGHT_HIP",
        "KNEE": "RIGHT_KNEE",
        "ANKLE": "RIGHT_ANKLE",
        "FOOT": "RIGHT_FOOT_INDEX",
    },
}

HEAD_KEY = "NOSE"  # surrogate for head center


def compute_metrics(pts: Dict[str, Tuple[int, int]], front_side: str) -> Dict[str, Optional[float]]:
    """Compute required metrics for the chosen front side ('LEFT' or 'RIGHT')."""
    fs = POSE_KEYS[front_side]

    # Elbow angle (shoulder–elbow–wrist)
    elbow = angle(pts.get(fs["SHOULDER"]), pts.get(fs["ELBOW"]), pts.get(fs["WRIST"]))

    # Spine lean: hip–shoulder line vs vertical (we use average of left/right if available)
    ls, rs = pts.get("LEFT_SHOULDER"), pts.get("RIGHT_SHOULDER")
    lh, rh = pts.get("LEFT_HIP"), pts.get("RIGHT_HIP")
    spine_angle = None
    if ls and lh and rs and rh:
        shoulder_mid = ((ls[0] + rs[0]) / 2.0, (ls[1] + rs[1]) / 2.0)
        hip_mid = ((lh[0] + rh[0]) / 2.0, (lh[1] + rh[1]) / 2.0)
        spine_angle = angle_vs_vertical_deg(hip_mid, shoulder_mid)

    # Head-over-knee alignment: horizontal distance |x_head - x_knee_front|
    head = pts.get(HEAD_KEY)
    knee = pts.get(fs["KNEE"])
    head_knee_dx = None
    if head and knee:
        head_knee_dx = float(abs(head[0] - knee[0]))  # pixels

    # Front foot direction: angle of ankle -> foot_index vs x-axis (0° means pointing right)
    ankle = pts.get(fs["ANKLE"])
    toe = pts.get(fs["FOOT"]) or pts.get(fs["ANKLE"])  # fallback
    foot_dir = None
    if ankle and toe and (ankle != toe):
        foot_dir = line_angle_deg(ankle, toe)
        if foot_dir is not None:
            # Normalize to [-90, 90] for readability (angle wrt x-axis, ignoring pointing up vs down)
            a = ((foot_dir + 180) % 180) - 90
            foot_dir = float(a)

    return {
        "elbow_angle_deg": elbow,
        "spine_lean_deg": spine_angle,
        "head_knee_dx_px": head_knee_dx,
        "front_foot_angle_deg": foot_dir,
    }

# --- Thresholds & feedback ------------------------------------------------------

DEFAULT_THRESHOLDS = {
    # These are broad cricket heuristics; tune per dataset if needed.
    "elbow_min": 100,     # good elevation if >= 100°
    "elbow_max": 165,     # avoid locking out fully
    "spine_lean_max": 20, # <= 20° from vertical
    "head_knee_dx_max": 40, # px; smaller is better alignment
    "foot_angle_max": 25, # |deg| relative to x-axis
}


def cues_from_metrics(m: Dict[str, Optional[float]], th=DEFAULT_THRESHOLDS) -> List[str]:
    cues = []
    e = m.get("elbow_angle_deg")
    if e is not None:
        if th["elbow_min"] <= e <= th["elbow_max"]:
            cues.append("✅ Good elbow elevation")
        else:
            cues.append("❌ Adjust elbow elevation")

    s = m.get("spine_lean_deg")
    if s is not None:
        if s <= th["spine_lean_max"]:
            cues.append("✅ Upright spine")
        else:
            cues.append("❌ Excessive spine lean")

    d = m.get("head_knee_dx_px")
    if d is not None:
        if d <= th["head_knee_dx_max"]:
            cues.append("✅ Head over front knee")
        else:
            cues.append("❌ Head not over front knee")

    f = m.get("front_foot_angle_deg")
    if f is not None:
        if abs(f) <= th["foot_angle_max"]:
            cues.append("✅ Front foot aligned")
        else:
            cues.append("❌ Front foot misaligned")
    return cues

# --- Scoring --------------------------------------------------------------------

def score_and_feedback(metrics_history: List[Dict[str, Optional[float]]], th=DEFAULT_THRESHOLDS) -> Dict:
    # Aggregate simple robust means (ignore None)
    def mean_of(key: str) -> Optional[float]:
        vals = [m[key] for m in metrics_history if m.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    e = mean_of("elbow_angle_deg")
    s = mean_of("spine_lean_deg")
    d = mean_of("head_knee_dx_px")
    f = mean_of("front_foot_angle_deg")

    # Map to 1–10 scores with simple piecewise heuristics
    def map_score(val: Optional[float], good_low: float, good_high: float, invert: bool=False) -> int:
        if val is None:
            return 5  # unknown
        if invert:  # lower is better
            if val <= good_low: return 10
            if val <= good_high: return 8
            if val <= good_high * 1.5: return 6
            return 4
        else:  # inside band is better
            if good_low <= val <= good_high: return 9
            if abs(val - (good_low + good_high)/2) <= 20: return 7
            return 5

    footwork_score = map_score(abs(f) if f is not None else None, 0, th["foot_angle_max"], invert=True)
    headpos_score  = map_score(d, 0, th["head_knee_dx_max"], invert=True)
    swing_score    = map_score(e, th["elbow_min"], th["elbow_max"], invert=False)
    balance_score  = map_score(s, 0, th["spine_lean_max"], invert=True)
    follow_score   = int(round(np.mean([footwork_score, swing_score, balance_score])))

    fb = {
        "Footwork": {
            "score": footwork_score,
            "feedback": "Front foot generally aligned with shot line" if footwork_score >= 8 else "Work on aligning lead foot towards shot line",
        },
        "Head Position": {
            "score": headpos_score,
            "feedback": "Head stacked over front knee at impact often" if headpos_score >= 8 else "Keep head over front knee to maintain balance",
        },
        "Swing Control": {
            "score": swing_score,
            "feedback": "Elbow elevation within optimal range" if swing_score >= 8 else "Maintain elbow elevation through downswing",
        },
        "Balance": {
            "score": balance_score,
            "feedback": "Upright spine, stable base" if balance_score >= 8 else "Reduce lateral lean and keep core upright",
        },
        "Follow-through": {
            "score": follow_score,
            "feedback": "Smooth finish with stable base" if follow_score >= 8 else "Finish tall and balanced after contact",
        },
    }

    return fb

# --- Main processing loop -------------------------------------------------------

def run(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # Resolve input
    if args.input.lower().startswith("http"):
        print("Downloading video…")
        local_path = download_video_with_ytdlp(args.input, args.temp_dir)
    else:
        local_path = args.input
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Input not found: {local_path}")

    cap = cv2.VideoCapture(local_path)
    if not cap.isOpened():
        raise RuntimeError("Failed to open video")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Normalize FPS/resolution (optional). For near real-time, we keep source FPS.
    out_w, out_h = args.width or src_w, args.height or src_h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = os.path.join(args.output_dir, "annotated_video.mp4")
    writer = cv2.VideoWriter(out_path, fourcc, src_fps, (out_w, out_h))

    pose = PoseEstimator(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    metrics_hist: List[Dict[str, Optional[float]]] = []

    # Simple EMA smoothing for display only
    ema = {"elbow": None, "spine": None, "headknee": None, "foot": None}
    alpha = 0.2

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or -1
    t0 = time.time()
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        if (out_w, out_h) != (src_w, src_h):
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

        pts, raw_landmarks = pose.infer(frame)
        if pts is not None:
            pose.draw(frame, raw_landmarks)
            m = compute_metrics(pts, front_side=args.front_side.upper())
        else:
            m = {"elbow_angle_deg": None, "spine_lean_deg": None, "head_knee_dx_px": None, "front_foot_angle_deg": None}

        metrics_hist.append(m)

        # EMA smoothing for overlay
        def ema_update(key, val):
            prev = ema[key]
            if val is None:
                return prev
            return val if prev is None else (alpha * val + (1 - alpha) * prev)

        ema["elbow"] = ema_update("elbow", m["elbow_angle_deg"])
        ema["spine"] = ema_update("spine", m["spine_lean_deg"])
        ema["headknee"] = ema_update("headknee", m["head_knee_dx_px"])
        ema["foot"] = ema_update("foot", m["front_foot_angle_deg"])

        # Text overlays ---------------------------------------------------------
        def put(text, y, color=(255, 255, 255)):
            cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)

        put(f"Frame: {frame_idx}/{total_frames if total_frames>0 else '?'}  FPS(src): {src_fps:.1f}", 24)
        put(f"Elbow: {ema['elbow']:.1f}°" if ema['elbow'] is not None else "Elbow: --", 50)
        put(f"Spine lean: {ema['spine']:.1f}°" if ema['spine'] is not None else "Spine lean: --", 76)
        put(f"Head↔Knee (x): {ema['headknee']:.0f}px" if ema['headknee'] is not None else "Head↔Knee: --", 102)
        put(f"Front foot angle: {ema['foot']:.1f}°" if ema['foot'] is not None else "Front foot angle: --", 128)

        # Cue messages
        for i, cue in enumerate(cues_from_metrics(m)):
            put(cue, 160 + i * 24, color=(40, 220, 40) if cue.startswith("✅") else (20, 60, 240))

        # Write frame
        writer.write(frame)

        if args.display:
            cv2.imshow("AthleteRise Cover Drive – Realtime", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    writer.release()
    if args.display:
        cv2.destroyAllWindows()

    # Final evaluation -----------------------------------------------------------
    eval_obj = score_and_feedback(metrics_hist)
    eval_path = os.path.join(args.output_dir, "evaluation.json")
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_obj, f, indent=2)

    elapsed = time.time() - t0
    avg_fps = frame_idx / elapsed if elapsed > 0 else 0.0
    print(f"Processed {frame_idx} frames in {elapsed:.2f}s (avg {avg_fps:.2f} FPS). Output: {out_path}")
    print(f"Saved evaluation to {eval_path}")


def build_argparser():
    p = argparse.ArgumentParser(description="Real-time cover drive analysis from full video")
    p.add_argument("--input", type=str, default="https://youtube.com/shorts/vSX3IRxGnNY",
                   help="YouTube URL or local video path")
    p.add_argument("--front_side", type=str, choices=["LEFT", "RIGHT", "left", "right"], default="LEFT",
                   help="Lead/front side of batter (RIGHT for left-handed batter)")
    p.add_argument("--output_dir", type=str, default="output", help="Directory to store outputs")
    p.add_argument("--width", type=int, default=None, help="Resize width (keep aspect if height not set)")
    p.add_argument("--height", type=int, default=None, help="Resize height")
    p.add_argument("--display", action="store_true", help="Show live window (press q to quit)")
    p.add_argument("--temp_dir", type=str, default=os.path.join(tempfile.gettempdir(), "athleterise_tmp"))
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    run(args)

