"""Custom exception hierarchy used across the project.

Having dedicated exception types (instead of generic Exception/ValueError
everywhere) makes error handling explicit and lets the CLI layer show
clean, actionable messages to the user instead of raw stack traces.
"""


class AttendanceSystemError(Exception):
    """Base class for all application-specific errors."""


class ConfigError(AttendanceSystemError):
    """Raised when the configuration file is missing or invalid."""


class CameraError(AttendanceSystemError):
    """Raised when the webcam / video source cannot be opened or read."""


class NoFaceDetectedError(AttendanceSystemError):
    """Raised when no face could be located in a frame/image."""


class InsufficientSamplesError(AttendanceSystemError):
    """Raised when not enough face samples were captured for enrollment."""


class ModelNotTrainedError(AttendanceSystemError):
    """Raised when recognition is attempted before a model has been trained."""


class StudentAlreadyExistsError(AttendanceSystemError):
    """Raised when trying to enroll a student ID that is already registered."""


class StudentNotFoundError(AttendanceSystemError):
    """Raised when a lookup is done for an unknown student ID."""


class InvalidImageError(AttendanceSystemError):
    """Raised when an image path is unreadable or corrupted."""
