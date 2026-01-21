#!/usr/bin/env python3
"""
Eye-controlled mouse using MediaPipe Tasks FaceLandmarker.
Moves cursor with iris position; blink to click; press q to quit.
"""

import time
from pathlib import Path
from typing import Iterable, Sequence

import cv2
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options as mp_base
from mediapipe.tasks.python.vision.core import image as mp_image
import numpy as np
import pyautogui

pyautogui.FAILSAFE = False

MODEL_PATH = Path("models/face_landmarker.task")
SCREEN_W, SCREEN_H = pyautogui.size()

LEFT_EYE_CORNERS = (362, 263)
RIGHT_EYE_CORNERS = (33, 133)
LEFT_EYE_LIDS = (386, 374)
RIGHT_EYE_LIDS = (159, 145)
LEFT_IRIS_IDX = (468, 469, 470, 471, 472)
RIGHT_IRIS_IDX = (473, 474, 475, 476, 477)

SMOOTHING = 0.18
BLINK_RATIO_THRESH = 0.21
BLINK_CONSEC_FRAMES = 2
CLICK_COOLDOWN_SEC = 0.35


def landmark_to_point(landmark, img_w: int, img_h: int) -> np.ndarray:
    return np.array([landmark.x * img_w, landmark.y * img_h], dtype=np.float32)


def iris_center(landmarks: Sequence, idxs: Iterable[int]) -> tuple[float, float]:
    pts = np.array([[landmarks[i].x, landmarks[i].y] for i in idxs], dtype=np.float32)
    center = np.mean(pts, axis=0)
    return float(center[0]), float(center[1])


def eye_open_ratio(landmarks: Sequence, corners: Sequence[int], lids: Sequence[int]) -> float:
    left_corner = landmark_to_point(landmarks[corners[0]], 1, 1)
    right_corner = landmark_to_point(landmarks[corners[1]], 1, 1)
    upper = landmark_to_point(landmarks[lids[0]], 1, 1)
    lower = landmark_to_point(landmarks[lids[1]], 1, 1)
    horizontal = np.linalg.norm(right_corner - left_corner)
    vertical = np.linalg.norm(upper - lower)
    if horizontal == 0.0:
        return 0.0
    return vertical / horizontal


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing model at {MODEL_PATH}. Download it from "
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        )

    landmarker = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=mp_base.BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
    )

    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cam.isOpened():
        raise RuntimeError("Cannot open webcam (index 0).")

    cursor_x, cursor_y = pyautogui.position()
    blink_frames = 0
    last_click = 0.0

    while True:
        ok, frame = cam.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        frame_h, frame_w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp_image.Image(image_format=mp_image.ImageFormat.SRGB, data=rgb_frame)
        result = landmarker.detect_for_video(mp_img, int(time.time() * 1000))

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]
            if len(landmarks) <= max(RIGHT_IRIS_IDX):
                cv2.putText(frame, "Model missing iris landmarks", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                l_iris = iris_center(landmarks, LEFT_IRIS_IDX)
                r_iris = iris_center(landmarks, RIGHT_IRIS_IDX)
                gaze_x = (l_iris[0] + r_iris[0]) * 0.5
                gaze_y = (l_iris[1] + r_iris[1]) * 0.5

                target_x = gaze_x * SCREEN_W
                target_y = gaze_y * SCREEN_H
                cursor_x = (1.0 - SMOOTHING) * cursor_x + SMOOTHING * target_x
                cursor_y = (1.0 - SMOOTHING) * cursor_y + SMOOTHING * target_y
                pyautogui.moveTo(cursor_x, cursor_y)

                left_open = eye_open_ratio(landmarks, LEFT_EYE_CORNERS, LEFT_EYE_LIDS)
                right_open = eye_open_ratio(landmarks, RIGHT_EYE_CORNERS, RIGHT_EYE_LIDS)
                open_ratio = (left_open + right_open) * 0.5

                if open_ratio < BLINK_RATIO_THRESH:
                    blink_frames += 1
                else:
                    if blink_frames >= BLINK_CONSEC_FRAMES and time.time() - last_click > CLICK_COOLDOWN_SEC:
                        pyautogui.click()
                        last_click = time.time()
                    blink_frames = 0

                cv2.circle(frame, (int(l_iris[0] * frame_w), int(l_iris[1] * frame_h)), 3, (0, 255, 0), -1)
                cv2.circle(frame, (int(r_iris[0] * frame_w), int(r_iris[1] * frame_h)), 3, (0, 255, 0), -1)
                cv2.putText(frame, f"Blink ratio: {open_ratio:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(frame, "Blink to click - press q to quit", (10, frame_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Eye Controlled Mouse", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
