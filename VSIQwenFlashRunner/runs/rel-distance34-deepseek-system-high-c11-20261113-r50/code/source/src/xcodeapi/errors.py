"""Errors deliberately omit remote bodies, prompts, credentials and URLs."""


class XcodeAPIError(Exception):
    """Base error for callers to catch without exposing remote data."""


class ConfigurationError(XcodeAPIError):
    pass


class ProtocolError(XcodeAPIError):
    pass


class RequestLimitError(XcodeAPIError):
    pass


class TransportError(XcodeAPIError):
    """The request may have reached the provider; never retry automatically."""


class HTTPAPIError(XcodeAPIError):
    def __init__(self, status: int):
        self.status = status
        hints = {
            400: "Check model-specific parameters and protocol.",
            401: "Check the API token.",
            403: "Check token group and endpoint access.",
            404: "Check exact model ID and endpoint.",
            429: "Rate or quota limit reached.",
        }
        hint = hints.get(status, "The provider rejected or failed the request.")
        super().__init__(f"Xcode HTTP {status}. {hint} No automatic retry.")
