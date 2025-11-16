"""Utility helpers for finger counting and hand annotations."""
from __future__ import annotations

import math
from typing import Tuple

from mediapipe.framework.formats import landmark_pb2

TIP_IDS = [4, 8, 12, 16, 20]


def count_fingers(
    hand_landmarks: landmark_pb2.NormalizedLandmarkList,
    handedness_label: str,
) -> int:
    """Return the number of raised fingers for a detected hand.

    Args:
        hand_landmarks: MediaPipe hand landmarks for a single detected hand.
        handedness_label: "Left" or "Right" classification from MediaPipe.

    Returns:
        Integer count of raised fingers.
    """

    if not hand_landmarks or not hand_landmarks.landmark:
        return 0

    landmarks = hand_landmarks.landmark

    fingers_up = []

    # Thumb: rely on radial distance from the wrist instead of raw x-coordinate so
    # back-of-hand or mirrored views don't force a false positive.
    wrist = landmarks[0]
    thumb_ip = landmarks[3]
    thumb_tip = landmarks[4]
    middle_mcp = landmarks[9]

    palm_span = _euclidean_distance(wrist, middle_mcp) or 1e-6
    thumb_tip_dist = _euclidean_distance(wrist, thumb_tip)
    thumb_ip_dist = _euclidean_distance(wrist, thumb_ip)
    fingers_up.append(thumb_tip_dist - thumb_ip_dist > 0.2 * palm_span)

    # Other fingers: tip should be above PIP (lower y value) when extended
    for tip_id in TIP_IDS[1:]:
        fingers_up.append(landmarks[tip_id].y < landmarks[tip_id - 2].y)

    return int(sum(fingers_up))


def _euclidean_distance(a: landmark_pb2.NormalizedLandmark, b: landmark_pb2.NormalizedLandmark) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def get_hand_center(
    hand_landmarks: landmark_pb2.NormalizedLandmarkList,
    frame_shape: Tuple[int, int]
) -> Tuple[int, int]:
    """Estimate the pixel center of a detected hand for text placement."""
    if not hand_landmarks or not hand_landmarks.landmark:
        return 0, 0

    height, width = frame_shape[:2]
    xs = [lm.x * width for lm in hand_landmarks.landmark]
    ys = [lm.y * height for lm in hand_landmarks.landmark]
    return int(sum(xs) / len(xs)), int(sum(ys) / len(ys))
