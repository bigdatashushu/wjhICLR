"""Provider-independent online model client errors."""


class ServiceUnavailable(RuntimeError):
    """The configured online model service cannot complete a request."""
