# Eye-Controlled Mouse (Winks for Clicks)

Control the cursor with your gaze and click with winks using MediaPipe Tasks face landmarks.

## Prerequisites
- Python 3.10–3.12 (recommended)
- macOS: grant your terminal/IDE **Camera** and **Accessibility** permissions.
- Windows: allow camera access when prompted (Settings → Privacy & security → Camera).

## Setup
```bash
python3 -m pip install --upgrade pip
pip3 install -r requirements.txt
```
The face landmarker model is included at `models/face_landmarker.task`.

## Run
```bash
python3 main.py        # macOS/Linux
py -3 main.py          # Windows
```

- Cursor follows your iris position.
- Left-eye wink → left click.
- Right-eye wink → right click.
- Press `q` to quit.

## Notes
- If the camera fails to open, re-check camera permission (macOS: Privacy → Camera; Windows: Privacy & security → Camera).
- Adjust sensitivity in `main.py`: `EYE_CLOSED_RATIO` (lower = easier to register closed eye) and `CLICK_COOLDOWN_SEC` (debounce clicks).
