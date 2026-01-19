"""Centralized logging configuration for all services."""

import structlog
import logging
import sys
from typing import Optional


def configure_logging(
    service_name: str,
    log_level: str = "INFO",
    json_output: bool = True,
    add_context: Optional[dict] = None
):
    """
    Configure structured logging for a service.

    Args:
        service_name: Name of the service (e.g., "dashboard", "simulator")
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        json_output: Whether to output JSON (True) or human-readable (False)
        add_context: Additional context to add to all logs (e.g., {"version": "0.1.0"})

    Returns:
        Logger instance configured for the service
    """
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper())
    )

    # Build processors
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    # Add service context
    if add_context:
        processors.append(structlog.processors.dict_tracebacks)

    # Choose output format
    if json_output:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    # Configure structlog
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Get logger with service context
    logger = structlog.get_logger(service_name)

    if add_context:
        logger = logger.bind(**add_context)

    return logger


def get_logger(name: str = None):
    """Get a logger instance with optional name.

    Args:
        name: Optional logger name

    Returns:
        Logger instance
    """
    return structlog.get_logger(name)
