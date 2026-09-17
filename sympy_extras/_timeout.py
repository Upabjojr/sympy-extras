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


class TimeLimitExceeded(BaseException):
    """Raised inside :func:`time_limit` when the time is up.

    A ``BaseException``, like ``KeyboardInterrupt``: SymPy's routines
    catch ``Exception`` in places (the heuristics of ``integrate``, the
    simplifiers), and the limit, once swallowed there, was gone for the
    rest of the computation (the bug: integrals ran for hundreds of
    seconds under a limit of twenty)."""


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
    _restore_manualintegrate()

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


def _restore_manualintegrate() -> None:
    """Build the special-function patterns of SymPy's ``manualintegrate``
    before a limit is set: ``special_function_rule`` builds them on first
    use, extending its list of wildcards first and its list of patterns
    after, and a limit hit in between left the wildcards in place and
    the patterns empty, so that the next call extended the wildcards
    again and every special-function rule (``exp(exp(x))`` to ``Ei``)
    matched the wrong wildcards for the rest of the process (the bug:
    two census entries lost after a slow one in the same worker). The
    lists are rebuilt apart and installed only when complete.

    ``integral_steps`` marks the integrand it works on with ``None`` in
    its cache against recursion and removes the mark when done; a limit
    or an error inside leaves the mark, and that integrand (``exp(x)/x``
    met inside ``exp(exp(x))``) is ``DontKnowRule`` for the rest of the
    process. The marks are cleared before a limit is set: no
    ``manualintegrate`` is on the stack then, the limits being set around
    its calls, not inside them."""
    import sympy.integrals.manualintegrate as manual
    from sympy.core.symbol import Dummy
    from sympy.functions.elementary.exponential import exp
    manual._integral_cache.clear()
    patterns, wilds = manual._special_function_patterns, manual._wilds
    if patterns and len(wilds) == 5:
        return
    fresh_patterns: list[object] = []
    fresh_wilds: list[object] = []
    vars(manual)['_special_function_patterns'] = fresh_patterns
    vars(manual)['_wilds'] = fresh_wilds
    try:
        u = Dummy('u')
        manual.special_function_rule(manual.IntegralInfo(exp(u) / u, u))
    except BaseException:
        vars(manual)['_special_function_patterns'] = []
        vars(manual)['_wilds'] = []
        raise
    if not (fresh_patterns and len(fresh_wilds) == 5):
        vars(manual)['_special_function_patterns'] = []
        vars(manual)['_wilds'] = []


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
    ``ValueError``, ``TypeError``, a ``PolynomialError`` raised on an
    expression its polynomial routines cannot take, or a
    ``RecursionError`` from deep inside SymPy, which is a way of giving
    up too)."""
    from sympy.polys.polyerrors import BasePolynomialError
    try:
        with time_limit(seconds):
            return f()
    except (TimeLimitExceeded, NotImplementedError, ValueError, TypeError, RecursionError,
            BasePolynomialError):
        return None
