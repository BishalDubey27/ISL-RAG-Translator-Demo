"""
MediaPipe Tasks-based Feature Extractor

Uses MediaPipe Tasks API (`mediapipe.tasks.python.vision`) to load
Pose/Hand/Face landmarker models (.task files) and extract features.
"""

import cv2
import numpy as np
import os
import tempfile
from typing import Optional
import os as _os

try:
    from mediapipe.tasks.python.vision import (
        pose_landmarker,
        hand_landmarker,
        face_landmarker,
    )
    from mediapipe.tasks.python.vision.core import image as mp_image
    _HAS_TASKS_API = True
except Exception:
    _HAS_TASKS_API = False


class _DummyExtractor:
    def __init__(self):
        self.pose_dim = 132
        self.face_dim = 1404
        self.hand_dim = 63
        self.total_dim = self.pose_dim + self.face_dim + self.hand_dim * 2

    def extract_frame(self, frame: np.ndarray):
        return np.zeros(self.total_dim)

    def extract_video(self, video_path, max_frames=64):
        return np.zeros((max_frames, self.total_dim))

    def close(self):
        return None


class MediaPipeExtractor:
    """Extractor using MediaPipe Tasks API (Pose/Face/Hand landmarker).

    Requires you to download the task model files and place them under
    `sign_recognition/mediapipe_models/` with filenames:
      - pose_landmarker.task
      - hand_landmarker.task
      - face_landmarker.task

    If you place them elsewhere, pass `model_dir` when constructing.
    """

    def __init__(self, model_dir: Optional[str] = None):
        # Allow forcing dummy via environment variable
        force_dummy = _os.environ.get('SIGN_USE_DUMMY_EXTRACTOR', '').lower() in ('1', 'true')
        if force_dummy:
            self._impl = _DummyExtractor()
            return

        if not _HAS_TASKS_API:
            # Tasks API unavailable — use dummy
            self._impl = _DummyExtractor()
            return

        # attempt to initialize real Tasks API extractors; fall back on any error
        try:
            base = model_dir or os.path.join(os.path.dirname(__file__), '..', 'mediapipe_models')
            base = os.path.abspath(base)

            pose_path = os.path.join(base, 'pose_landmarker.task')
            hand_path = os.path.join(base, 'hand_landmarker.task')
            face_path = os.path.join(base, 'face_landmarker.task')

            missing = [p for p in (pose_path, hand_path, face_path) if not os.path.exists(p)]
            if missing:
                raise FileNotFoundError("Missing MediaPipe task model files: " + ", ".join(missing))

            # Create landmarkers from model files
            self.pose_landmarker = pose_landmarker.PoseLandmarker.create_from_model_path(pose_path)
            self.hand_landmarker = hand_landmarker.HandLandmarker.create_from_model_path(hand_path)
            self.face_landmarker = face_landmarker.FaceLandmarker.create_from_model_path(face_path)

            # Feature dimensions (matches old Holistic output layout)
            self.pose_dim = 132
            self.face_dim = 1404
            self.hand_dim = 63
            self.total_dim = self.pose_dim + self.face_dim + self.hand_dim * 2

            self._impl = None
        except Exception:
            import warnings

            warnings.warn('MediaPipe Tasks extractor failed to initialize; falling back to dummy extractor.')
            self._impl = _DummyExtractor()

    @property
    def total_dim(self):
        """Return total feature dimension, delegating to _impl if active."""
        if self._impl is not None:
            return self._impl.total_dim
        return self.pose_dim + self.face_dim + self.hand_dim * 2

    def _frame_to_mpimage(self, frame: np.ndarray):
        # write temporary image file and create a MediaPipe Image
        tf = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        try:
            # OpenCV writes BGR; save as JPEG
            cv2.imwrite(tf.name, frame)
            img = mp_image.Image.create_from_file(tf.name)
        finally:
            tf.close()
            try:
                os.unlink(tf.name)
            except Exception:
                pass
        return img

    def extract_frame(self, frame: np.ndarray):
        # If dummy implementation is active, delegate
        if getattr(self, '_impl', None) is not None:
            return self._impl.extract_frame(frame)

        img = self._frame_to_mpimage(frame)

        features = np.zeros(self.total_dim)

        # Pose
        pose_res = self.pose_landmarker.detect(img)
        if getattr(pose_res, 'pose_landmarks', None):
            pose_data = []
            for lm in pose_res.pose_landmarks:
                pose_data.extend([getattr(lm, 'x', 0.0), getattr(lm, 'y', 0.0), getattr(lm, 'z', 0.0), 0.0])
            features[:self.pose_dim] = pose_data[:self.pose_dim]

        # Face
        face_res = self.face_landmarker.detect(img)
        if getattr(face_res, 'face_landmarks', None):
            face_data = []
            for lm in face_res.face_landmarks:
                face_data.extend([getattr(lm, 'x', 0.0), getattr(lm, 'y', 0.0), getattr(lm, 'z', 0.0)])
            features[self.pose_dim:self.pose_dim + self.face_dim] = face_data[:self.face_dim]

        # Hands (Tasks API may return multiple hands; map by handedness if available)
        hand_res = self.hand_landmarker.detect(img)
        left_hand = None
        right_hand = None
        if getattr(hand_res, 'hand_landmarks', None):
            if getattr(hand_res, 'hand_world_landmarks', None) and getattr(hand_res, 'handedness', None):
                for lm_list, h in zip(hand_res.hand_landmarks, hand_res.handedness):
                    label = getattr(h, 'category_name', '').lower() or getattr(h, 'label', '').lower()
                    if 'left' in label:
                        left_hand = lm_list
                    elif 'right' in label:
                        right_hand = lm_list
            else:
                if len(hand_res.hand_landmarks) > 0:
                    left_hand = hand_res.hand_landmarks[0]
                if len(hand_res.hand_landmarks) > 1:
                    right_hand = hand_res.hand_landmarks[1]

        if left_hand:
            left_data = []
            for lm in left_hand:
                left_data.extend([getattr(lm, 'x', 0.0), getattr(lm, 'y', 0.0), getattr(lm, 'z', 0.0)])
            features[self.pose_dim + self.face_dim:self.pose_dim + self.face_dim + self.hand_dim] = left_data[:self.hand_dim]

        if right_hand:
            right_data = []
            for lm in right_hand:
                right_data.extend([getattr(lm, 'x', 0.0), getattr(lm, 'y', 0.0), getattr(lm, 'z', 0.0)])
            features[-self.hand_dim:] = right_data[:self.hand_dim]

        return features

    def extract_video(self, video_path, max_frames=64):
        # If dummy implementation is active, delegate
        if getattr(self, '_impl', None) is not None:
            return self._impl.extract_video(video_path, max_frames)

        cap = cv2.VideoCapture(video_path)
        features_sequence = []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            return np.zeros((max_frames, self.total_dim))

        frame_indices = np.linspace(0, total_frames-1, max_frames, dtype=int)

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                features = self.extract_frame(frame)
                features_sequence.append(features)

        cap.release()

        while len(features_sequence) < max_frames:
            features_sequence.append(np.zeros(self.total_dim))

        return np.array(features_sequence[:max_frames])

    def close(self):
        # If dummy implementation is active, delegate
        if getattr(self, '_impl', None) is not None:
            try:
                return self._impl.close()
            except Exception:
                return None

        try:
            if getattr(self, 'pose_landmarker', None):
                self.pose_landmarker.close()
        except Exception:
            pass
        try:
            if getattr(self, 'hand_landmarker', None):
                self.hand_landmarker.close()
        except Exception:
            pass
        try:
            if getattr(self, 'face_landmarker', None):
                self.face_landmarker.close()
        except Exception:
            pass