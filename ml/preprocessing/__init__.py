"""Video preprocessing: decode, detect, align, crop and tensorise.

Importable without PyTorch except for :mod:`ml.preprocessing.tensor`, so the
geometric pipeline can be tested without a model or a checkpoint.
"""

from ml.preprocessing.alignment import cut_patch, estimate_alignment, warp_face
from ml.preprocessing.debug_export import export_mouth_track, export_source_overlay
from ml.preprocessing.face_detector import FaceTracker
from ml.preprocessing.landmarks import interpolate_missing, smooth_landmarks
from ml.preprocessing.pipeline import (
    MouthROIPipeline,
    NoFaceDetectedError,
    PreprocessingError,
)
from ml.preprocessing.video_io import DecodeError, decode_frames, probe, write_mp4

__all__ = [
    "MouthROIPipeline",
    "PreprocessingError",
    "NoFaceDetectedError",
    "FaceTracker",
    "decode_frames",
    "probe",
    "write_mp4",
    "DecodeError",
    "interpolate_missing",
    "smooth_landmarks",
    "estimate_alignment",
    "warp_face",
    "cut_patch",
    "export_mouth_track",
    "export_source_overlay",
]
