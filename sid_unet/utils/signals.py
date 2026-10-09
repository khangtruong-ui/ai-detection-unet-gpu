"""Signal handling utilities for shielding long-running tasks from SSH disconnects."""

from __future__ import annotations

import logging
import os
import signal
from typing import Iterable, Optional, Set

logger = logging.getLogger("sid_unet.signals")

# Signals typically sent when an SSH session disconnects, controlling terminal closes,
# or session manager / container host / systemd tears down the interactive session.
# Signal 15 (SIGTERM) is death signal 15; Signal 1 (SIGHUP) is terminal hangup.
SSH_DISCONNECT_SIGNAL_NAMES = (
    "SIGHUP",    # 1: Controlling terminal closed / SSH dropped
    "SIGTERM",   # 15: Session teardown / system termination
    "SIGPIPE",   # 13: Broken pipe (e.g. closed SSH stdout/stderr)
    "SIGQUIT",   # 3: Terminal quit
    "SIGTSTP",   # 20: Terminal stop
    "SIGTTIN",   # 21: Background tty read
    "SIGTTOU",   # 22: Background tty write
)

SSH_DISCONNECT_SIGNALS: tuple[int, ...] = tuple(
    getattr(signal, name)
    for name in SSH_DISCONNECT_SIGNAL_NAMES
    if hasattr(signal, name)
)


def shield_process_signals(
    signals_to_ignore: Optional[Iterable[int]] = None,
    log_info: bool = False,
) -> Set[int]:
    """Shield the current process from SSH disconnection and session teardown signals.

    Sets specified signals (default: SIGHUP, SIGTERM, SIGPIPE, SIGQUIT, SIGTSTP, SIGTTIN, SIGTTOU)
    to SIG_IGN so the kernel drops them silently without terminating the process.

    Returns the set of successfully ignored signal numbers.
    """
    if signals_to_ignore is None:
        signals_to_ignore = SSH_DISCONNECT_SIGNALS

    ignored: Set[int] = set()
    for sig in signals_to_ignore:
        try:
            signal.signal(sig, signal.SIG_IGN)
            ignored.add(sig)
        except (ValueError, OSError) as e:
            if log_info:
                logger.debug(f"Could not ignore signal {sig}: {e}")

    return ignored
