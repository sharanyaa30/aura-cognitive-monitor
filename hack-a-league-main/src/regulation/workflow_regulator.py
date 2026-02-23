"""Workflow regulation actions based on detected cognitive/biometric signals.

This module intentionally contains no UI framework code. It only performs
system-level interventions via ``pyautogui`` with a module-level cooldown so
actions are not repeatedly triggered every frame.
"""

from __future__ import annotations
from src.cv.screen_ocr import capture_and_ocr
from src.llm.openai_assistant import summarize_text

import time

import pyautogui


# Global cooldown (seconds) between any two regulation actions.
COOLDOWN_SECONDS = 10.0

# Timestamp of the last action that was actually executed.
_last_action_ts = 0.0

# Last recommendation generated (accessible by the dashboard).
_last_recommendation: str | None = None


def _is_on_cooldown(now_ts: float) -> bool:
	"""Return True when a new action should be suppressed by cooldown."""
	return (now_ts - _last_action_ts) < COOLDOWN_SECONDS


def _mark_action_executed(now_ts: float) -> None:
	"""Record the timestamp of the latest executed action."""
	global _last_action_ts
	_last_action_ts = now_ts


def get_last_recommendation() -> str | None:
    """Return the most recent AI-generated recommendation, or None."""
    return _last_recommendation


def apply_regulation(load_score, head_forward):
    """Apply regulation interventions based on load score and head posture."""
    global _last_recommendation
    now_ts = time.time()

    if _is_on_cooldown(now_ts):
        return

    if load_score > 70:
        text = capture_and_ocr()
        try:
             rescue = summarize_text(text)
        except Exception:
            rescue = "- Take a 2 minute break\n- Write down your next small step\n- Resume with focus"

        _last_recommendation = rescue

        pyautogui.alert(
            "High cognitive load detected.\n\n"
            "Here is a quick rescue plan:\n\n"
            f"{rescue}"
        )

        _mark_action_executed(now_ts)
        return

    if head_forward is True:
        pyautogui.hotkey("ctrl", "+")
        _mark_action_executed(now_ts)
