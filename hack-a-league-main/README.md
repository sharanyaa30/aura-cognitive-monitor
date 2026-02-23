# Cognitive Load Detector (Streamlit)

## Run app

1. Install dependencies:

	pip install -r requirements.txt

2. Start Streamlit UI:

	streamlit run app.py

The app displays:

- Live webcam frame
- Blink rate
- Breathing rate
- Head forward posture flag
- Load score
- Status label (`Deep Flow` / `Normal` / `Brain Fried`)

Regulation is applied in the Streamlit flow via:

- `apply_regulation(load_score, head_forward)`

