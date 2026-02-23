"""Core synchronous pipeline orchestration.

This module is intentionally UI-agnostic. It handles one real-time step:
1) read a webcam frame,
2) extract face metrics,
3) read breathing rate,
4) compute cognitive-load score,
5) return normalized output dictionary.
"""
from __future__ import annotations

from typing import Any, Dict

import cv2

from src.cv import facial_features
from src.signals import biometric_input
from src.detection import load_detector


def initialize_webcam(camera_index: int = 0) -> cv2.VideoCapture:
	"""Initialize and return an opened webcam capture object."""
	capture = cv2.VideoCapture(camera_index)
	if not capture.isOpened():
		raise RuntimeError("Failed to open webcam. Check camera permissions/device.")
	return capture


def release_webcam(capture: cv2.VideoCapture) -> None:
	"""Release webcam resources safely."""
	if capture is not None:
		capture.release()


def _extract_face_metrics(frame: Any) -> Dict[str, Any]:
	"""Call facial feature extraction with a small compatibility shim."""
	if hasattr(facial_features, "extract_face_metrics"):
		return facial_features.extract_face_metrics(frame)
	raise NotImplementedError(
		"Expected src/cv/facial_features.py to define extract_face_metrics(frame)."
	)


def _get_breathing_rate() -> float:
	"""Call biometric input module with a small compatibility shim."""
	if hasattr(biometric_input, "get_breathing_rate"):
		return float(biometric_input.get_breathing_rate())
	raise NotImplementedError(
		"Expected src/signals/biometric_input.py to define get_breathing_rate()."
	)


def _compute_load_score(blink_rate: float, head_forward: bool, breathing_rate: float) -> float:
	"""Call load detector with a small compatibility shim."""
	if hasattr(load_detector, "compute_load_score"):
		return float(
			load_detector.compute_load_score(
				blink_rate=blink_rate,
				head_forward=head_forward,
				breathing_rate=breathing_rate,
			)
		)
	raise NotImplementedError(
		"Expected src/detection/load_detector.py to define "
		"compute_load_score(blink_rate, head_forward, breathing_rate)."
	)


def run_pipeline_step(capture: cv2.VideoCapture) -> Dict[str, Any]:
	"""Run one synchronous inference cycle and return normalized metrics.

	Returns:
		{
		  "frame": frame,
		  "blink_rate": float,
		  "head_forward": bool,
		  "breathing_rate": float,
		  "load_score": float
		}
	"""
	success, frame = capture.read()
	if not success:
		raise RuntimeError("Failed to read frame from webcam.")

	face_metrics = _extract_face_metrics(frame)
	blink_rate = float(face_metrics.get("blink_rate", 0.0))
	head_forward = bool(face_metrics.get("head_forward", False))

	breathing_rate = _get_breathing_rate()
	
	load_score = _compute_load_score(blink_rate, head_forward, breathing_rate)

	return {
		"frame": frame,
		"blink_rate": blink_rate,
		"head_forward": head_forward,
		"breathing_rate": breathing_rate,
		"load_score": load_score,
	}

def run_pipeline(capture):
    """
    Run one pipeline step using an already-initialized webcam.
    """
    return run_pipeline_step(capture)