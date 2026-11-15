"""Xcode API integration, intentionally isolated from the existing runners."""

from .client import GenerationResult, Usage, XcodeClient
from .config import Profile, load_profiles
from .content import image_file_part, image_url_part, text_part
from .errors import (
    ConfigurationError, HTTPAPIError, ProtocolError, RequestLimitError,
    TransportError, XcodeAPIError,
)

__all__ = [
    "XcodeClient", "Profile", "GenerationResult", "Usage", "load_profiles",
    "image_file_part", "image_url_part", "text_part", "XcodeAPIError",
    "ConfigurationError", "HTTPAPIError", "ProtocolError",
    "RequestLimitError", "TransportError",
]
