"""OpenAI assistant helper for concise workflow summaries.

This module contains no UI logic. It exposes a single function,
`summarize_text`, that sends user-provided text to an OpenAI model and returns
the model's response as plain text.
"""

import os

from openai import OpenAI


def summarize_text(text: str) -> str:
	"""Summarize user work context and return a 3-bullet action plan.

	The function:
	1. Reads `OPENAI_API_KEY` from environment variables.
	2. Sends the required prompt and the input text to the OpenAI API.
	3. Returns the assistant response text.

	Args:
		text: Arbitrary text describing what the user is working on.

	Returns:
		str: Model-generated summary and 3-bullet action plan.

	Raises:
		ValueError: If `OPENAI_API_KEY` is not configured.
	"""
	# Read API key from environment, as required.
	api_key = os.getenv("OPENAI_API_KEY")
	if not api_key:
		raise ValueError("OPENAI_API_KEY is not set in the environment.")

	# Initialize the OpenAI client with the environment-provided key.
	client = OpenAI(api_key=api_key)

	# Fixed instruction prompt from requirements.
	instruction = (
		"Summarize what the user is currently working on and give a 3-bullet "
		"action plan."
	)

	# Use the Responses API to generate the summary.
	response = client.responses.create(
		model="gpt-4.1-mini",
		input=[
			{"role": "system", "content": instruction},
			{"role": "user", "content": text},
		],
	)

	# Return model response text. `output_text` is the simplest direct accessor.
	return response.output_text
