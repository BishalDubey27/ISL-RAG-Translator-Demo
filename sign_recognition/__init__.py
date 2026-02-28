"""
Sign Recognition package init.

Avoid importing heavy submodules at package import time to prevent
import-time side effects (like requiring MediaPipe model files).
Import submodules explicitly where needed, e.g.:
	from sign_recognition.models.mediapipe_extractor import MediaPipeExtractor
"""

__all__ = ['models', 'utils']