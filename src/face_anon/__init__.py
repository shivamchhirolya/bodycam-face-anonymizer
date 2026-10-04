"""Body-camera face anonymization with independent RetinaFace QC."""

__version__ = "1.2.0"

from face_anon.pipeline import FaceAnonymizer, process_file
from face_anon.types import FaceBox, JobRecord, QCReport

__all__ = ["FaceAnonymizer", "FaceBox", "JobRecord", "QCReport", "process_file", "__version__"]
