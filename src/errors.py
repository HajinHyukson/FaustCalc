class ConfigError(Exception):
    """Raised when local configuration is missing or invalid."""


class APIError(Exception):
    """Base exception for API and network failures."""


class UnauthorizedError(APIError):
    """Raised when API credentials are missing, invalid, or not permitted."""


class RateLimitError(APIError):
    """Raised when API rate limits are exceeded."""


class NotFoundError(APIError):
    """Raised when requested market data is not found."""


class DataError(Exception):
    """Raised when fetched data cannot satisfy portfolio analytics requirements."""
