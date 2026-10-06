"""Time limits for SymPy computations which may not terminate in
reasonable time (integration, ``dsolve``, ``solve``, simplification).

The limit is enforced with the ``SIGALRM`` interval timer, so it only works
in the main thread on POSIX systems; elsewhere the computation runs without
a limit.
"""
from __future__ import annotations

import contextlib
import signal
import threading
import time
from contextlib import contextmanager
from types import FrameType
from typing import Callable, Iterator, Optional, TypeVar

__all__ = ['TimeLimitExceeded', 'Deadline', 'time_limit', 'attempt', 'remaining_time']

T = TypeVar('T')


class Deadline:
    """One running :func:`time_limit`: the instant (of ``time.monotonic``)
    at which it expires. The limits running form a stack, innermost last,
    whose instants do not increase towards the inside (an inner limit is
    cut to the one enclosing it)."""

    __slots__ = ('end',)

    def __init__(self, end: float) -> None:
        self.end = end


#: the limits running, outermost first
_running: list[Deadline] = []


class TimeLimitExceeded(BaseException):
    """Raised inside :func:`time_limit` when the time is up.

    A ``BaseException``, like ``KeyboardInterrupt``: SymPy's routines
    catch ``Exception`` in places (the heuristics of ``integrate``, the
    simplifiers), and the limit, once swallowed there, was gone for the
    rest of the computation (the bug: integrals ran for hundreds of
    seconds under a limit of twenty).

    ``deadline`` is the limit which expired, the outermost one when
    several have: only the code which set that limit may take the
    exception for "no answer in time" (:meth:`expired`); the limits set
    inside it let it through, so that the computation stops where its
    owner said and does not go on with an inner step taken for a failure
    (the bug: an inner :func:`attempt` cut short by the enclosing limit
    returned ``None``, the caller went on without the step, and an
    integral got a case valued as an unevaluated ``Integral``)."""

    def __init__(self, deadline: Optional[Deadline] = None) -> None:
        super().__init__()
        self.deadline = deadline

    def expired(self, deadline: Optional[Deadline]) -> bool:
        """Whether this is the expiry of ``deadline`` (the value of the
        ``with time_limit(...) as deadline`` of the code catching it),
        rather than of a limit enclosing it, which must be let through
        (``raise``). An expiry of no known limit (raised by other code)
        is taken by any limit."""
        if deadline is None:
            return False
        return self.deadline is None or self.deadline is deadline


def _supported() -> bool:
    return hasattr(signal, 'setitimer') and threading.current_thread() is threading.main_thread()


def _outermost_expired(now: float) -> Optional[Deadline]:
    for deadline in _running:
        if deadline.end <= now:
            return deadline
    return None


class _Bookkeeping:
    """Whether :func:`time_limit` is updating the stack and the timer: the
    signal handler raises nothing meanwhile (an exception there left a
    limit on the stack and the timer armed after the handler had been
    restored, which killed the process with ``SIGALRM``), and the
    deadlines are checked again when the update is done."""

    busy = False


def _raise_if_expired() -> None:
    expired = _outermost_expired(time.monotonic())
    if expired is not None:
        raise TimeLimitExceeded(expired)


def _in_machinery(frame: Optional[FrameType]) -> bool:
    """Whether the handler interrupted the machinery of the limits (this
    module, the context managers of :mod:`contextlib`, the wrappers of
    :mod:`signal`): an exception raised there, before a ``finally`` of
    :func:`time_limit` has started, would skip its cleanup."""
    if frame is None:
        return False
    return frame.f_code.co_filename in (__file__, contextlib.__file__, signal.__file__)


def _expire(signum: int, frame: Optional[FrameType]) -> None:
    if _Bookkeeping.busy:
        # the deadlines are checked when the update is done
        return
    if _in_machinery(frame):
        # again in a moment, out of it
        signal.setitimer(signal.ITIMER_REAL, 1e-3)
        return
    _raise_if_expired()
    if _running:
        # early (the timer and the clock disagree by a hair): again
        signal.setitimer(signal.ITIMER_REAL, max(_running[-1].end - time.monotonic(), 1e-4))


@contextmanager
def time_limit(seconds: Optional[float]) -> Iterator[Optional[Deadline]]:
    """Run the body for at most ``seconds`` (no limit for ``None`` or a
    non-positive value); :class:`TimeLimitExceeded` is raised in the body
    when the time is up. The value is the :class:`Deadline` of the limit
    (``None`` when there is none), to tell its expiry from that of an
    enclosing limit with :meth:`TimeLimitExceeded.expired`.

    Nested limits are honoured: an inner limit ends no later than the
    enclosing one, and when the enclosing one expires during the inner
    body the exception carries the enclosing deadline, which the inner
    owner lets through. A limit entered after an enclosing one has
    expired (its exception swallowed on the way) raises that expiry at
    once. The seconds are multiplied by ``settings.time_scale``, which a
    slower machine sets to get the answers of a faster one.

    >>> from sympy_extras._timeout import TimeLimitExceeded, time_limit
    >>> try:
    ...     with time_limit(0.2) as outer:
    ...         try:
    ...             with time_limit(10) as inner:
    ...                 while True:
    ...                     pass
    ...         except TimeLimitExceeded as expiry:
    ...             if not expiry.expired(inner):
    ...                 raise
    ...             print('inner')
    ... except TimeLimitExceeded as expiry:
    ...     print('outer' if expiry.expired(outer) else 'neither')
    outer
    """
    if not _supported():
        yield None
        return
    _raise_if_expired()
    if seconds is None or seconds <= 0:
        yield None
        return
    from sympy_extras.settings import settings
    seconds = seconds * settings.time_scale
    _complete_sympy_tables()
    _restore_manualintegrate()
    _Bookkeeping.busy = True
    started = time.monotonic()
    # a timer set by other code, around the first limit: honoured
    foreign = 0.0 if _running else float(signal.getitimer(signal.ITIMER_REAL)[0])
    end = started + seconds
    if _running:
        end = min(end, _running[-1].end)
    elif foreign > 0:
        end = min(end, started + foreign)
    deadline = Deadline(end)
    previous = signal.signal(signal.SIGALRM, _expire)
    _running.append(deadline)
    signal.setitimer(signal.ITIMER_REAL, max(end - started, 1e-4))
    try:
        _Bookkeeping.busy = False
        # an enclosing limit which expired is raised at once; this one,
        # when it is shorter than its own entry, expires in the body,
        # once the owner holds its Deadline (raised here, the expiry was
        # nobody's: attempt let it through instead of returning None)
        expired = _outermost_expired(time.monotonic())
        if expired is not None and expired is not deadline:
            raise TimeLimitExceeded(expired)
        yield deadline
    finally:
        _Bookkeeping.busy = True
        signal.setitimer(signal.ITIMER_REAL, 0)
        if deadline in _running:
            # with any limit inside it left behind by an interrupted exit
            del _running[_running.index(deadline):]
        signal.signal(signal.SIGALRM, previous)
        now = time.monotonic()
        if _running:
            # not when it has expired: raised below, and a timer firing
            # after would raise it again in the handler of its owner
            if _outermost_expired(now) is None:
                signal.setitimer(signal.ITIMER_REAL, max(_running[-1].end - now, 1e-4))
        elif foreign > 0:
            signal.setitimer(signal.ITIMER_REAL, max(foreign - (now - started), 1e-3))
        _Bookkeeping.busy = False
        # an enclosing limit which expired meanwhile: to its owner now,
        # not from a timer firing at a random point of the code after
        _raise_if_expired()


def remaining_time() -> Optional[float]:
    """The seconds left of the enclosing :func:`time_limit`, in the units
    of the settings (divided by ``settings.time_scale``), or ``None`` when
    no limit is running (or limits are not supported here): a computation
    which is optional (the special values of the parameters of a result
    already found) takes a share of it rather than a fixed time which may
    exceed it.

    >>> from sympy_extras._timeout import remaining_time, time_limit
    >>> remaining_time() is None
    True
    >>> with time_limit(10):
    ...     0 < (remaining_time() or 0) <= 10
    True
    """
    if not _supported():
        return None
    if _running:
        # from the deadline, the timer being off once it has expired: a
        # little rather than nothing, which would read as no limit
        left = max(_running[-1].end - time.monotonic(), 1e-6)
    else:
        left = float(signal.getitimer(signal.ITIMER_REAL)[0])
        if left <= 0:
            return None
    from sympy_extras.settings import settings
    return float(left) / settings.time_scale


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
    expression its polynomial routines cannot take, a ``RecursionError``
    from deep inside SymPy, which is a way of giving up too, or an
    ``ArithmeticError``: ``PrecisionExhausted`` when ``evalf`` cannot
    decide a sign, a division by zero, an overflow).

    An enclosing limit which expires meanwhile is not the time of ``f``
    being up: its :class:`TimeLimitExceeded` goes on to the code which
    set it.

    >>> from sympy_extras._timeout import attempt
    >>> attempt(lambda: 1/0, 1) is None
    True
    >>> def slow() -> int:
    ...     while True:
    ...         pass
    >>> attempt(lambda: (attempt(slow, 0.1), 'went on'), 5)
    (None, 'went on')
    >>> attempt(lambda: (attempt(slow, 5), 'went on'), 0.2) is None
    True
    """
    from sympy.polys.polyerrors import BasePolynomialError
    deadline: Optional[Deadline] = None
    try:
        with time_limit(seconds) as deadline:
            return f()
    except TimeLimitExceeded as expiry:
        if expiry.expired(deadline):
            return None
        raise
    except (NotImplementedError, ValueError, TypeError, RecursionError,
            BasePolynomialError, ArithmeticError):
        return None
