"""Utilities for full-screen OCR capture.

This module intentionally contains no UI logic. It provides a single function,
`capture_and_ocr`, which captures the entire screen and returns OCR text.
"""

from PIL import ImageGrab
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def capture_and_ocr() -> str:
	"""Capture the full screen and return extracted text as one string.

	Steps:
	1. Capture the complete current screen image using PIL.ImageGrab.
	2. Run OCR using pytesseract on the captured image.
	3. Normalize whitespace so the output is a single clean string.

	Returns:
		str: OCR-extracted text from the full screen.
	"""
	# Grab a snapshot of the full screen. No bbox means entire screen.
	screenshot = ImageGrab.grab()

	# Extract raw text from the screenshot using Tesseract OCR.
	raw_text = pytesseract.image_to_string(screenshot)

	# Normalize all whitespace/newlines into single spaces so callers receive
	# a single-string output suitable for downstream processing.
	normalized_text = " ".join(raw_text.split())

	return normalized_text
