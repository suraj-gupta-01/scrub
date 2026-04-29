"""
ASV Navigation System — Logging Setup
======================================
Import and call setup_logging() once at the top of main.py and
simulate_rth.py. All other modules just use:

    log = logging.getLogger(__name__)

and this file controls how those logs appear in the terminal.

Log levels (from least to most severe):
    DEBUG   — very detailed internal state (hidden by default)
    INFO    — normal progress messages  ← default visible level
    WARNING — something unexpected but recoverable
    ERROR   — something failed, mission may be affected
    CRITICAL— system cannot continue

Terminal format:
    10:42:17 [INFO    ] hardware         — STM32Hardware ready on /dev/ttyAMA0
    10:42:18 [WARNING ] sensor_hub       — GPS stale for 3.1 s
    10:42:19 [ERROR   ] recovery         — No candidate segments found
    └─ time   └─ level  └─ which file      └─ the actual message
"""

import logging
import sys


def setup_logging(level: str = "INFO", show_debug_for: list = None) -> None:
    """
    Configure terminal logging for the whole ASV system.

    Args:
        level:          Minimum level to show. Usually "INFO".
                        Pass "DEBUG" to see everything (very verbose).
        show_debug_for: Optional list of module names to show DEBUG for
                        even when global level is INFO.
                        e.g. ["heading_estimator", "recovery"]

    Usage in main.py:
        from log_setup import setup_logging
        setup_logging()                          # normal run
        setup_logging("DEBUG")                   # see everything
        setup_logging(show_debug_for=["recovery"])  # debug one module
    """

    # ── Root logger ───────────────────────────────────────────────────────────
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)   # catch everything; handlers filter below

    # ── Terminal handler ──────────────────────────────────────────────────────
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, level.upper(), logging.INFO))

    fmt = logging.Formatter(
        fmt     = "%(asctime)s [%(levelname)-8s] %(name)-20s — %(message)s",
        datefmt = "%H:%M:%S",
    )
    handler.setFormatter(fmt)
    root.addHandler(handler)

    # ── Per-module debug overrides ────────────────────────────────────────────
    if show_debug_for:
        for module_name in show_debug_for:
            mod_log = logging.getLogger(module_name)
            mod_log.setLevel(logging.DEBUG)
            # Give it its own handler so it bypasses the root level filter
            dbg_handler = logging.StreamHandler(sys.stdout)
            dbg_handler.setLevel(logging.DEBUG)
            dbg_handler.setFormatter(fmt)
            mod_log.addHandler(dbg_handler)
            mod_log.propagate = False   # don't double-print
        logging.getLogger(__name__).info(
            "DEBUG enabled for modules: %s", show_debug_for
        )

    logging.getLogger(__name__).info(
        "Logging ready — level=%s", level.upper()
    )


def get_test_logger() -> logging.Logger:
    """
    Returns a simple stdout logger for use inside test files.
    Call this inside conftest.py so all tests share it.
    """
    logger = logging.getLogger("test")
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter(
            "  %(name)s [%(levelname)s] %(message)s"
        ))
        logger.addHandler(h)
        logger.setLevel(logging.DEBUG)
    return logger