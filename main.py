"""Real-time human pose, finger, and emotion detection demo."""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import cv2
from deepface import DeepFace
import mediapipe as mp

from utils.finger_counter import count_fingers, get_hand_center


@dataclass
class DetectionConfig:
    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    emotion_interval: int = 10
    emotion_backend: str = "retinaface"
    skip_pose: bool = False
    skip_hands: bool = False
    face_margin: float = 0.2


class RealTimeAnalyzer:
    """Core application orchestrating pose, hand, and emotion detection."""

    def __init__(self, config: DetectionConfig):
        self.config = config
        self.frame_index = 0
        self.current_emotion = "Neutral"
        self._last_emotion_frame = -config.emotion_interval

        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_face_detection = mp.solutions.face_detection

        self.pose = None if config.skip_pose else self.mp_pose.Pose(
            min_detection_confidence=config.min_detection_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
            model_complexity=1,
        )
        self.hands = None if config.skip_hands else self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=config.min_detection_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
        )
        self.face_detector = self.mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=config.min_detection_confidence,
        )

    def process_stream(self) -> None:
        cap = cv2.VideoCapture(self.config.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)

        if not cap.isOpened():
            raise RuntimeError("Unable to access the webcam. Check the camera index or permissions.")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[WARN] Unable to read frame from camera. Skipping...")
                    continue

                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                if self.pose:
                    self._draw_pose(frame, frame_rgb)

                if self.hands:
                    self._draw_hands_and_fingers(frame, frame_rgb)

                should_update_emotion = self.frame_index - self._last_emotion_frame >= self.config.emotion_interval
                face_roi = None
                if should_update_emotion:
                    face_roi = self._extract_face_roi(frame_rgb, frame)
                self._update_emotion(
                    frame,
                    face_roi=face_roi,
                    force=should_update_emotion,
                )
                self._draw_emotion(frame)

                cv2.imshow("Human Recognition", frame)

                self.frame_index += 1
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()
            if self.pose:
                self.pose.close()
            if self.hands:
                self.hands.close()
            if self.face_detector:
                self.face_detector.close()
            cv2.destroyAllWindows()

    def _draw_pose(self, frame, frame_rgb) -> None:
        if not self.pose:
            return
        pose_results = self.pose.process(frame_rgb)
        if not pose_results.pose_landmarks:
            return

        self.mp_drawing.draw_landmarks(
            frame,
            pose_results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 180, 0), thickness=2, circle_radius=2),
        )

    def _draw_hands_and_fingers(self, frame, frame_rgb) -> None:
        if not self.hands:
            return
        hand_results = self.hands.process(frame_rgb)
        if not hand_results.multi_hand_landmarks:
            return

        handedness_iter = hand_results.multi_handedness or []
        for hand_landmarks, hand_handedness in zip(hand_results.multi_hand_landmarks, handedness_iter):
            hand_label = hand_handedness.classification[0].label if hand_handedness.classification else "Hand"

            self.mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(255, 220, 0), thickness=2, circle_radius=3),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1),
            )

            finger_count = count_fingers(hand_landmarks, hand_label)
            cx, cy = get_hand_center(hand_landmarks, frame.shape)
            text_position = (max(cx - 40, 10), max(cy - 20, 30))
            cv2.putText(
                frame,
                f"{hand_label}: {finger_count}",
                text_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (50, 50, 255),
                2,
                cv2.LINE_AA,
            )

    def _update_emotion(self, frame_bgr, face_roi, force: bool = False) -> None:
        if not force:
            return

        try:
            target_image = face_roi if face_roi is not None else frame_bgr
            backend = "skip" if face_roi is not None else self.config.emotion_backend
            analysis = DeepFace.analyze(
                target_image,
                actions=["emotion"],
                enforce_detection=False,
                detector_backend=backend,
            )
            if isinstance(analysis, list):
                analysis = analysis[0]
            dominant = analysis.get("dominant_emotion")
            if dominant:
                self.current_emotion = dominant.capitalize()
            self._last_emotion_frame = self.frame_index
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Emotion analysis skipped: {exc}")
    def _extract_face_roi(self, frame_rgb, frame_bgr):
        if not self.face_detector:
            return None

        results = self.face_detector.process(frame_rgb)
        if not results.detections:
            return None

        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        h, w = frame_bgr.shape[:2]

        margin = self.config.face_margin
        x1 = max(int((bbox.xmin - margin) * w), 0)
        y1 = max(int((bbox.ymin - margin) * h), 0)
        x2 = min(int((bbox.xmin + bbox.width + margin) * w), w)
        y2 = min(int((bbox.ymin + bbox.height + margin) * h), h)

        if x2 <= x1 or y2 <= y1:
            return None

        return frame_bgr[y1:y2, x1:x2].copy()

    def _draw_emotion(self, frame) -> None:
        cv2.putText(
            frame,
            f"Emotion: {self.current_emotion}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time pose, finger, and emotion detection demo")
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam index to open (default: 0)")
    parser.add_argument("--width", type=int, default=1280, help="Capture width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Capture height (default: 720)")
    parser.add_argument(
        "--emotion-interval",
        type=int,
        default=10,
        help="Analyze emotion every N frames to maintain FPS (default: 10)",
    )
    parser.add_argument(
        "--emotion-backend",
        type=str,
        default="retinaface",
        help="DeepFace detector backend (retinaface, mtcnn, opencv, etc.)",
    )
    parser.add_argument("--no-pose", action="store_true", help="Disable pose estimation")
    parser.add_argument("--no-hands", action="store_true", help="Disable hand tracking and finger counting")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DetectionConfig(
        camera_index=args.camera_index,
        frame_width=args.width,
        frame_height=args.height,
        emotion_interval=max(1, args.emotion_interval),
        emotion_backend=args.emotion_backend,
        skip_pose=args.no_pose,
        skip_hands=args.no_hands,
    )
    analyzer = RealTimeAnalyzer(config)
    analyzer.process_stream()


if __name__ == "__main__":
    main()
