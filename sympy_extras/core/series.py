"""The series expansion of an expression: ``series`` and ``lseries``.

``series(e, x, x0, n, dir)`` is the expansion of ``e`` at ``x = x0`` up to
``O((x - x0)**n, (x, x0))``: the point is moved to zero from the right
(``x0 + x`` from the right, ``x0 - x`` from the left, ``1/x`` at infinity),
the variable is made positive, and the ``_eval_nseries`` of the classes
expand the tree up to the order asked for, asking for more terms when an
expansion returns fewer; ``lseries`` gives the terms one by one. These are
the methods ``series`` and ``lseries`` of ``Expr``.

References
==========

.. [1] SymPy Development Team, *SymPy documentation: series*,
       https://docs.sympy.org/latest/modules/series/series.html
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/core/expr.py (Expr.series) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: typed; the generator of n=None is the separate lseries; the
# asymptotic series is still SymPy's (through the conversion).
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Iterator, Optional

from .coreerrors import PoleError
from .expr import Expr
from .numbers import Integer
from .symbol import Dummy, Symbol

__all__ = ['series', 'lseries']


def _is_zero_number(e: Expr) -> bool:
    """SymPy's falsity of an expression: a number equal to zero."""
    return e.is_Number and e == 0


def _less(a: Expr, b: Expr) -> bool:
    """``a < b`` of SymPy for numbers (``TypeError`` when undecided)."""
    from .add import add
    from .mul import mul
    answer = add(a, mul(Integer(-1), b)).is_extended_negative
    if answer is None:
        raise TypeError("cannot determine truth value of %s < %s" % (a, b))
    return answer


def _direction(x0: Expr, dir: str, cdir: Expr) -> tuple[str, Expr]:
    """SymPy's normalisation of the direction: ``dir`` from a nonzero
    ``cdir``, ``cdir`` from ``dir`` (from the sign of an infinite point),
    then ``cdir/abs(cdir)``."""
    from sympy_extras.functions.elementary.complexes import abs_, im, sign
    if len(dir) != 1 or dir not in '+-':
        raise ValueError("Dir must be '+' or '-'")
    if cdir.is_zero is not True:
        if cdir.is_real is True:
            dir = '+' if cdir.is_positive is True else '-'
        else:
            dir = '+' if im(cdir).is_positive is True else '-'
    else:
        if not _is_zero_number(x0) and x0.is_infinite is True:
            cdir = sign(x0).simplify()
        elif dir == '+':
            cdir = Integer(1)
        elif dir == '-':
            cdir = Integer(-1)
        elif cdir == Integer(0):
            cdir = Integer(1)
    return dir, cdir / abs_(cdir)


def _variable(e: Expr, x: Optional[Symbol]) -> Optional[Symbol]:
    """The variable of the series (the only free symbol when not given);
    ``None`` when ``e`` is free of it."""
    if x is None:
        syms = e.free_symbols
        if not syms:
            return None
        if len(syms) > 1:
            raise ValueError('x must be given for multivariate functions.')
        x = next(iter(syms))
    if x not in e.free_symbols:
        return None
    return x


def series(e: Expr, x: Optional[Symbol], x0: Expr, n: int, dir: str, logx: Optional[Expr], cdir: Expr) -> Expr:
    """The series of ``e`` at ``x = x0`` up to ``O((x - x0)**n, (x, x0))``
    (``Expr.series``).

    Examples
    ========

    >>> from sympy_extras.core import Symbol, oo
    >>> from sympy_extras.core.series import series
    >>> from sympy_extras.functions import sin
    >>> x = Symbol('x')
    >>> series(sin(x)/x, x, Integer(0), 4, '+', None, Integer(0))
    1 - x**2/6 + O(x**4)
    >>> series(1/(x + 1), x, oo, 3, '+', None, Integer(0))
    -1/x**2 + 1/x + O(x**(-3), (x, oo))
    """
    from .order import Order, order
    from .power import pow
    v = _variable(e, x)
    if v is None:
        return e
    x = v
    if n < 0:
        raise ValueError("Number of terms should be nonnegative")
    dir, cdir = _direction(x0, dir, cdir)
    if not _is_zero_number(x0) and x0.is_infinite is True:
        try:
            s = series(e.subs(x, cdir / x), x, Integer(0), n, '+', None, Integer(1))
            return s.subs(x, cdir / x)
        except PoleError:
            s = e.subs(x, cdir * x).aseries(x, n=n)
            return s.subs(x, cdir * x)
    # the origin moved to x0 and the sign changed (dir negative), and back
    if not _is_zero_number(x0) or cdir != Integer(1):
        s = series(e.subs({x: x0 + cdir * x}), x, Integer(0), n, '+', logx, Integer(1))
        return s.subs({x: x / cdir - x0 / cdir})
    # from here on x0 = 0 and dir = '+'
    if x.is_positive is None and x.is_negative is None:
        xpos = Dummy('x', positive=True)
        rv = series(e.subs(x, xpos), xpos, x0, n, dir, logx, cdir)
        return rv.subs(xpos, x)
    s1 = e._eval_nseries(x, n, logx, cdir)
    o: Optional[Expr] = s1.getO()
    if o is not None and not _is_zero_number(o):
        # make sure that the order asked for is returned
        ngot = s1.getn()
        if ngot is None:
            raise NotImplementedError('not sure of order of %s' % (o,))
        if _less(Integer(n), ngot):
            # o kept as it is (x*log(x), ...) to absorb the terms properly,
            # replaced below
            if n != 0:
                s1 = s1 + o.subs(x, pow(x, Integer(n) / ngot))
            else:
                s1 = s1 + order(Integer(1), x)
        elif _less(ngot, Integer(n)):
            # more terms until the order changes, then the number of terms
            # predicted
            from sympy_extras.functions.elementary.integers import ceiling
            for more in range(1, 9):
                s1 = e._eval_nseries(x, n + more, logx, cdir)
                newn = s1.getn()
                if newn is None:
                    raise NotImplementedError('not sure of order of %s' % (s1,))
                if newn != ngot:
                    predicted = ceiling((Integer(n) - ngot) * Integer(more) / (newn - ngot))
                    if not isinstance(predicted, Integer):
                        raise TypeError("the number of terms is not an integer: %s" % (predicted,))
                    ndo = n + predicted.p
                    s1 = e._eval_nseries(x, ndo, logx, cdir)
                    while _less(_getn_or_raise(s1), Integer(n)):
                        s1 = e._eval_nseries(x, ndo, logx, cdir)
                        ndo += 1
                    break
            else:
                raise ValueError('Could not calculate %s terms for %s' % (str(n), e))
            s1 = s1 + order(pow(x, Integer(n)), x)
        o = s1.getO()
        s1 = s1.removeO()
    elif s1.has(Order):
        # an asymptotic expansion
        return s1
    else:
        o = order(pow(x, Integer(n)), x)
        s1done = s1.doit()
        try:
            if (s1done + o).removeO() == s1done:
                o = Integer(0)
        except NotImplementedError:
            return s1
    from sympy_extras.simplify.radsimp import collect
    rest: Expr = Integer(0) if o is None else o
    try:
        return collect(s1, x) + rest
    except NotImplementedError:
        return s1 + rest


def _getn_or_raise(s: Expr) -> Expr:
    n = s.getn()
    if n is None:
        raise TypeError("the series %s has no order term" % (s,))
    return n


def lseries(e: Expr, x: Optional[Symbol], x0: Expr, dir: str, logx: Optional[Expr], cdir: Expr) -> Iterator[Expr]:
    """The terms of the series of ``e`` at ``x = x0`` one by one
    (``Expr.lseries``).

    Examples
    ========

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.core.series import lseries
    >>> from sympy_extras.functions import exp
    >>> x = Symbol('x')
    >>> terms = lseries(exp(x), x, Integer(1), '+', None, Integer(0))
    >>> next(terms), next(terms)
    (E, E*(x - 1))
    """
    v = _variable(e, x)
    if v is None:
        yield e
        return
    x = v
    dir, cdir = _direction(x0, dir, cdir)
    if not _is_zero_number(x0) and x0.is_infinite is True:
        try:
            for si in lseries(e.subs(x, cdir / x), x, Integer(0), '+', None, Integer(1)):
                yield si.subs(x, cdir / x)
            return
        except PoleError:
            s = e.subs(x, cdir * x).aseries(x)
            yield s.subs(x, cdir * x)
            return
    if not _is_zero_number(x0) or cdir != Integer(1):
        for si in lseries(e.subs({x: x0 + cdir * x}), x, Integer(0), '+', logx, Integer(1)):
            yield si.subs({x: x / cdir - x0 / cdir})
        return
    if x.is_positive is None and x.is_negative is None:
        xpos = Dummy('x', positive=True)
        for si in lseries(e.subs(x, xpos), xpos, x0, dir, logx, cdir):
            yield si.subs(xpos, x)
        return
    yield from _yield_lseries(e.removeO()._eval_lseries(x, logx, cdir), x)


def _yield_lseries(s: Iterator[Expr], x: Symbol) -> Iterator[Expr]:
    """The terms of the sums of ``s`` one at a time, by increasing order
    until all the terms are returned."""
    from .add import Add
    from .order import order
    for si in s:
        if not isinstance(si, Add):
            yield si
            continue
        yielded: Expr = Integer(0)
        o = order(si, x) * x
        ndid = 0
        ndo = len(si.args)
        while True:
            do = (si - yielded + o).removeO()
            o = o * x
            if _is_zero_number(do) or do.is_Order:
                continue
            ndid += len(do.args) if isinstance(do, Add) else 1
            yield do
            if ndid == ndo:
                break
            yielded = yielded + do
