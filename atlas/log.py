"""Structured logging via structlog."""

from __future__ import annotations

import structlog


def configure_logging(level: str = "INFO") -> None:
    """Configure structlog with JSON output and correlation IDs."""
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(level),
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
    )


def get_logger(name: str | None = None) -> structlog.BoundLogger:
    """Get a module logger."""
    return structlog.get_logger(name)
