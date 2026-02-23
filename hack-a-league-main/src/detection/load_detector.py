"""Rule-based cognitive load detector (MVP).

No machine learning is used in this module. The score is computed from simple
heuristics and normalized to a 0-100 range.
"""

from __future__ import annotations


def _clamp(value: float, low: float, high: float) -> float:
	"""Clamp a numeric value into [low, high]."""
	return max(low, min(high, value))


def _blink_component(blink_rate: float) -> float:
	"""Map blink rate to a 0-50 load component.

	Heuristic:
	- <= 10 blinks/min contributes 0
	- >= 40 blinks/min contributes max (50)
	- linear interpolation in between
	"""
	normalized = (blink_rate - 10.0) / (40.0 - 10.0)
	normalized = _clamp(normalized, 0.0, 1.0)
	return normalized * 50.0


def _head_forward_component(head_forward: bool) -> float:
	"""Add posture penalty when user leans forward."""
	return 20.0 if head_forward else 0.0


def _breathing_component(breathing_rate: float) -> float:
	"""Map breathing deviation from normal (12-20 BPM) to a 0-30 component.

	- Inside normal range: 0 penalty
	- Outside range: penalty grows with distance from nearest boundary
	- Maxes out at 30 for large deviations (>= 8 BPM from boundary)
	"""
	if 12.0 <= breathing_rate <= 20.0:
		deviation = 0.0
	elif breathing_rate < 12.0:
		deviation = 12.0 - breathing_rate
	else:
		deviation = breathing_rate - 20.0

	normalized = _clamp(deviation / 8.0, 0.0, 1.0)
	return normalized * 30.0


def compute_load_score(blink_rate: float, head_forward: bool, breathing_rate: float) -> float:
	"""Compute a rule-based cognitive load score in the range [0, 100].

	Inputs:
	- blink_rate: blinks per minute
	- head_forward: posture flag from face landmarks
	- breathing_rate: breaths per minute

	Output:
	- load_score: float in [0, 100]
	"""
	blink_score = _blink_component(float(blink_rate))
	head_score = _head_forward_component(bool(head_forward))
	breathing_score = _breathing_component(float(breathing_rate))

	total = blink_score + head_score + breathing_score
	return float(_clamp(total, 0.0, 100.0))
