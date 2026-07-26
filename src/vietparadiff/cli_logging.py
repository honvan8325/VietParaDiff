"""Shared colored logging configuration for command-line workflows."""

import logging

__all__ = ["get_logger"]

COLORS = {
    logging.DEBUG: "\033[36m",  # Cyan
    logging.INFO: "\033[32m",  # Green
    logging.WARNING: "\033[33m",  # Yellow
    logging.ERROR: "\033[31m",  # Red
    logging.CRITICAL: "\033[1;31m",  # Bold red
}

RESET = "\033[0m"


class ColorFormatter(logging.Formatter):
    """Wrap a formatted log record in an ANSI color for its severity."""

    def format(self, record: logging.LogRecord) -> str:
        """Format one log record without mutating the original message."""
        color = COLORS.get(record.levelno, "")
        message = super().format(record)
        return f"{color}{message}{RESET}"


# Configure logging once at import time so builders and scripts use the same
# timestamp, level alignment, and color convention.
handler = logging.StreamHandler()
handler.setFormatter(
    ColorFormatter(
        fmt="{asctime} | {levelname:^8} | {name:^20.20} | {message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
)

logging.basicConfig(
    level=logging.INFO,
    handlers=[handler],
)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger managed by the shared root configuration.

    Args:
        name: Usually the caller's ``__name__`` value.

    Returns:
        The standard-library logger associated with ``name``.
    """
    return logging.getLogger(name)
