"""Time limits for SymPy computations which may not terminate in
reasonable time (integration, ``dsolve``, ``solve``, simplification).

The limit is enforced with the ``SIGALRM`` interval timer, so it only works
in the main thread on POSIX systems; elsewhere the computation runs without
a limit.
"""
from __future__ import annotations

import signal
import threading
import time
from contextlib import contextmanager
from typing import Callable, Iterator, Optional, TypeVar

__all__ = ['TimeLimitExceeded', 'time_limit', 'attempt']

T = TypeVar('T')


class TimeLimitExceeded(Exception):
    """Raised inside :func:`time_limit` when the time is up."""


def _supported() -> bool:
    return hasattr(signal, 'setitimer') and threading.current_thread() is threading.main_thread()


@contextmanager
def time_limit(seconds: Optional[float]) -> Iterator[None]:
    """Run the body for at most ``seconds`` (no limit for ``None`` or a
    non-positive value); :class:`TimeLimitExceeded` is raised in the body
    when the time is up. Nested limits are honoured: the outer one is
    restored, with the elapsed time subtracted, when the inner body ends.
    """
    if seconds is None or seconds <= 0 or not _supported():
        yield
        return
    _complete_sympy_tables()

    def handler(signum: int, frame: object) -> None:
        raise TimeLimitExceeded()

    outer_remaining = signal.getitimer(signal.ITIMER_REAL)[0]
    previous = signal.signal(signal.SIGALRM, handler)
    started = time.monotonic()
    limit = seconds if outer_remaining <= 0 else min(seconds, outer_remaining)
    signal.setitimer(signal.ITIMER_REAL, limit)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        if outer_remaining > 0:
            left = outer_remaining - (time.monotonic() - started)
            signal.setitimer(signal.ITIMER_REAL, max(left, 1e-3))


_tables_complete = False


def _complete_sympy_tables() -> None:
    """Build SymPy's table of Meijer G-function representations before a
    limit is set: SymPy builds it on first use (``_rewrite_single`` of
    :mod:`sympy.integrals.meijerint`), and a limit hit during the build
    left the global table truncated to the entries built so far, after
    which every later integration of an exponential went astray (the bug:
    ``mellin_transform(exp(-x), x, s)`` came out as ``uppergamma(s, 0)``
    on the whole plane once an integration had been interrupted, and a
    divergent integral got a value from it). The table is built apart and
    installed only when complete, so that an outer limit firing during
    the build leaves nothing behind."""
    global _tables_complete
    if _tables_complete:
        return
    import sympy.integrals.meijerint as meijerint
    table: dict[object, object] = {}
    meijerint._create_lookup_table(table)
    # the module global is declared ``None`` and rebound by SymPy on
    # first use: rebound through the module namespace
    vars(meijerint)['_lookup_table'] = table
    _tables_complete = True


def attempt(f: Callable[[], T], seconds: Optional[float]) -> Optional[T]:
    """The value of ``f()`` computed within ``seconds``, or ``None`` when
    the time is up or SymPy gives up (``NotImplementedError``,
    ``ValueError``, ``TypeError``, or a ``RecursionError`` from deep
    inside SymPy, which is a way of giving up too)."""
    try:
        with time_limit(seconds):
            return f()
    except (TimeLimitExceeded, NotImplementedError, ValueError, TypeError, RecursionError):
        return None
