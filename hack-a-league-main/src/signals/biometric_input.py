"""Biometric input module (MVP).

This file currently provides a simulated breathing-rate signal for local testing.
Later, this mock can be replaced with camera-based chest-motion estimation using
MediaPipe Pose landmarks.
"""

from __future__ import annotations

import math
import time


# Simulation bounds for a realistic resting/working breathing range.
MIN_BREATHING_BPM = 10.0
MAX_BREATHING_BPM = 25.0

# Center and amplitude derived from bounds.
_BASELINE_BPM = (MIN_BREATHING_BPM + MAX_BREATHING_BPM) / 2.0
_AMPLITUDE_BPM = (MAX_BREATHING_BPM - MIN_BREATHING_BPM) / 2.0

# Slow oscillation period in seconds (gentle variation over time).
_OSCILLATION_PERIOD_SECONDS = 45.0

# Start time used to keep the simulated signal continuous across calls.
_START_TIME = time.time()


def get_breathing_rate() -> float:
	"""Return current breathing rate in BPM.

	MVP behavior:
	- Returns a smoothly varying simulated value between 10 and 25 BPM.
	- Pure synchronous function, no UI, no external dependencies.

	Future replacement idea (MediaPipe Pose):
	- Track torso/shoulder/chest landmarks from webcam frames.
	- Estimate inhale/exhale cycles from vertical chest expansion signals.
	- Convert detected cycle frequency to breaths per minute.
	- Optionally smooth with a moving average before returning BPM.
	"""
	elapsed = time.time() - _START_TIME

	# Sine wave produces smooth up/down breathing variation.
	phase = (2.0 * math.pi * elapsed) / _OSCILLATION_PERIOD_SECONDS
	bpm = _BASELINE_BPM + _AMPLITUDE_BPM * math.sin(phase)

	# Safety clamp to keep strict MVP bounds.
	bpm = max(MIN_BREATHING_BPM, min(MAX_BREATHING_BPM, bpm))
	return float(bpm)
