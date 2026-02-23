"""Facial feature extraction using MediaPipe Face Mesh.

This module provides a minimal, synchronous API:
	extract_face_metrics(frame) -> {"blink_rate": float, "head_forward": bool}
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Tuple

import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh


# Eye landmark indices from MediaPipe Face Mesh (6 points per eye for EAR).
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

# Nose tip landmark used for forward-lean estimation.
NOSE_TIP_IDX = 1

# Simple heuristics tuned for MVP behavior.
EAR_CLOSED_THRESHOLD = 0.22
EAR_OPEN_THRESHOLD = 0.25
BLINK_DEBOUNCE_SECONDS = 0.15
BLINK_WINDOW_SECONDS = 60.0
NOSE_FORWARD_Z_THRESHOLD = -0.08


# Persistent state to estimate blink-rate across frames.
_state = {
	"start_time": time.time(),
	"blink_count": 0,
	"eyes_closed": False,
	"last_blink_ts": 0.0,
	"blink_timestamps": [],
}


# Reuse a single FaceMesh instance for efficiency.
_face_mesh = mp_face_mesh.FaceMesh(
	static_image_mode=False,
	max_num_faces=1,
	refine_landmarks=False,
	min_detection_confidence=0.5,
	min_tracking_confidence=0.5,
)


def _distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
	"""Euclidean distance between two 2D points."""
	return math.dist(a, b)


def _eye_aspect_ratio(eye_points: List[Tuple[float, float]]) -> float:
	"""Compute EAR from 6 eye points.

	Points expected order:
	[p1, p2, p3, p4, p5, p6]
	EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
	"""
	p1, p2, p3, p4, p5, p6 = eye_points
	horizontal = _distance(p1, p4)
	if horizontal == 0:
		return 0.0
	vertical = _distance(p2, p6) + _distance(p3, p5)
	return vertical / (2.0 * horizontal)


def _landmark_xy(landmark, width: int, height: int) -> Tuple[float, float]:
	"""Convert normalized landmark to pixel-space XY."""
	return landmark.x * width, landmark.y * height


def extract_face_metrics(frame) -> Dict[str, float | bool]:
	"""Extract minimal face metrics from a frame.

	Args:
		frame: OpenCV BGR frame.

	Returns:
		{
		  "blink_rate": float,   # blinks per minute
		  "head_forward": bool   # True when nose is sufficiently forward
		}
	"""
	if frame is None:
		return {"blink_rate": 0.0, "head_forward": False}

	height, width = frame.shape[:2]
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	result = _face_mesh.process(rgb)

	# Fallback if no face is detected.
	if not result.multi_face_landmarks:
		now_ts = time.time()
		window_start = now_ts - BLINK_WINDOW_SECONDS
		_state["blink_timestamps"] = [
			ts for ts in _state["blink_timestamps"] if ts >= window_start
		]
		blink_rate = float(len(_state["blink_timestamps"]))

		return {
			"blink_rate": blink_rate,
			"head_forward": False,
		}

	landmarks = result.multi_face_landmarks[0].landmark

	left_eye = [_landmark_xy(landmarks[i], width, height) for i in LEFT_EYE_IDX]
	right_eye = [_landmark_xy(landmarks[i], width, height) for i in RIGHT_EYE_IDX]

	left_ear = _eye_aspect_ratio(left_eye)
	right_ear = _eye_aspect_ratio(right_eye)
	avg_ear = (left_ear + right_ear) / 2.0

	now_ts = time.time()

	# Hysteresis avoids jitter around a single threshold.
	if _state["eyes_closed"]:
		if avg_ear > EAR_OPEN_THRESHOLD:
			if (now_ts - _state["last_blink_ts"]) >= BLINK_DEBOUNCE_SECONDS:
				_state["blink_count"] += 1
				_state["last_blink_ts"] = now_ts
				_state["blink_timestamps"].append(now_ts)
			_state["eyes_closed"] = False
	else:
		if avg_ear < EAR_CLOSED_THRESHOLD:
			_state["eyes_closed"] = True

	window_start = now_ts - BLINK_WINDOW_SECONDS
	_state["blink_timestamps"] = [
		ts for ts in _state["blink_timestamps"] if ts >= window_start
	]
	blink_rate = float(len(_state["blink_timestamps"]))

	nose_z = landmarks[NOSE_TIP_IDX].z
	head_forward = bool(nose_z < NOSE_FORWARD_Z_THRESHOLD)

	return {
		"blink_rate": float(blink_rate),
		"head_forward": head_forward,
	}
