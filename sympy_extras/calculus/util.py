"""The period of a function (SymPy's ``sympy.calculus.util.periodicity``)
on the typed core: :func:`periodicity`, :func:`lcim`.

The period is found structurally: the trigonometric functions of a linear
argument have the period of the function divided by the coefficient;
sums, products and powers have the least common multiple of the periods
of their parts (when their ratios are rational); ``exp`` of an imaginary
argument the common period of its real and imaginary parts; ``Abs`` of a
sine possibly half the period of the sine (checked by substitution);
``Mod(x, n)`` the period ``n``; and a composition ``g(h(x))`` the period
of an inner function ``h``.

This module is a typed port of SymPy's ``periodicity``, ``_periodicity``
and ``lcim`` (``sympy/calculus/util.py``), with the periods of the
trigonometric functions (``TrigonometricFunction._period`` of
``sympy/functions/elementary/trigonometric.py``) and the decomposition of
``sympy/solvers/decompogen.py`` (functions of one argument and
polynomials in one generator; not the decomposition of polynomials)
written down here. Nothing of SymPy is loaded.

Examples
========

>>> from sympy_extras.core import Symbol, pi
>>> from sympy_extras.functions import sin, cos, tan, exp, Abs, log
>>> from sympy_extras.calculus.util import periodicity
>>> x = Symbol('x')
>>> periodicity(sin(2*x), x), periodicity(sin(x) + cos(x/3), x), periodicity(tan(x), x)
(pi, 6*pi, pi)
>>> periodicity(Abs(sin(x)), x), periodicity(exp(sin(x)), x), periodicity(log(cos(x)**2 + 1), x)
(pi, 2*pi, 2*pi)
>>> periodicity(x*sin(x), x) is None
True
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/calculus/util.py, sympy/functions/elementary/trigonometric.py,
# sympy/solvers/decompogen.py (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: strict typing; the period check by substitution and the native
# simplify (no numerical equals); the decomposition without Poly's generators.
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Optional, Sequence

from sympy_extras.core.add import Add
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr, as_expr
from sympy_extras.core.function import DefinedFunction, expand_mul
from sympy_extras.core.mod import Mod
from sympy_extras.core.mul import Mul
from sympy_extras.core.numbers import E, Integer, pi
from sympy_extras.core.power import Pow
from sympy_extras.core.relational import Relational
from sympy_extras.core.symbol import Dummy, Symbol
from sympy_extras.functions.elementary.complexes import Abs, abs_, im, re
from sympy_extras.functions.elementary.exponential import Exp, exp
from sympy_extras.functions.elementary.piecewise import Piecewise
from sympy_extras.functions.elementary.trigonometric import Cos, Cot, Csc, Sec, Sin, Tan, sin

__all__ = ['periodicity', 'lcim']

#: the trigonometric functions and the period of ``f(x)``
_TRIGONOMETRIC: tuple[tuple[type[Basic], Expr], ...] = (
    (Sin, 2*pi), (Cos, 2*pi), (Sec, 2*pi), (Csc, 2*pi), (Tan, pi), (Cot, pi))


# PORTED FROM SYMPY: sympy/functions/elementary/trigonometric.py:TrigonometricFunction._period
def _trigonometric_period(f: Expr, symbol: Symbol) -> Optional[Expr]:
    """The period of a trigonometric function of an argument linear in
    ``symbol``, ``None`` for another argument (SymPy's
    ``NotImplementedError``) or another function."""
    general: Optional[Expr] = None
    for cls, period in _TRIGONOMETRIC:
        if isinstance(f, cls):
            general = period
    if general is None:
        return None
    argument = expand_mul(as_expr(f.args[0]))
    if not argument.has(symbol):
        return Integer(0)
    if argument == symbol:
        return general
    if isinstance(argument, Mul):
        g, h = argument.as_independent(symbol)
        if h == symbol:
            return general/abs_(g)
    if isinstance(argument, Add):
        _, h = argument.as_independent(symbol)
        g, h = h.as_independent(symbol, as_Add=False)
        if h == symbol:
            return general/abs_(g)
    return None


def _is_trigonometric(f: Basic) -> bool:
    return isinstance(f, (Sin, Cos, Tan, Cot, Sec, Csc))


def _same(a: Expr, b: Expr) -> bool:
    """Whether two expressions are recognised equal: structurally, after
    expansion, or by the native simplification of their difference."""
    if a == b:
        return True
    from sympy_extras.simplify.simplify import simplify
    from sympy_extras._timeout import attempt
    from sympy_extras.settings import settings
    difference = attempt(lambda: simplify(a - b), settings.timeout, native=True)
    return difference is not None and difference == 0


def _checked(orig_f: Expr, symbol: Symbol, period: Expr) -> Optional[Expr]:
    """The period when ``orig_f`` is unchanged by the shift, ``None``
    when that is not recognised (SymPy's ``_check`` raises)."""
    shifted = as_expr(orig_f.subs(symbol, symbol + period))
    return period if _same(shifted, orig_f) else None


# PORTED FROM SYMPY: sympy/solvers/decompogen.py:decompogen
def _decompogen(f: Expr, symbol: Symbol) -> list[Expr]:
    """``f`` as a composition ``g1(g2(...(gn(symbol))))``: the list of
    the ``gi`` in ``symbol``, outermost first (functions of one argument,
    powers, and polynomials in a single generator; SymPy's decomposition
    of polynomials is not ported, a polynomial stays one piece)."""
    if symbol not in f.free_symbols:
        return [f]
    if isinstance(f, (DefinedFunction, Pow)):
        if isinstance(f, Exp):
            argument = as_expr(f.args[0])
        elif isinstance(f, Pow):
            argument = f.base
        else:
            if len(f.args) != 1:
                return [f]
            argument = as_expr(f.args[0])
        if argument == symbol:
            return [f]
        return [as_expr(f.subs(argument, symbol))] + _decompogen(argument, symbol)
    generators = _generators(f, symbol)
    if generators is not None and len(generators) == 1 and generators[0] != symbol and generators[0] != f:
        g = generators[0]
        outer = as_expr(f.subs(g, symbol))
        if symbol in outer.free_symbols and not outer.has(g):
            return [outer] + _decompogen(g, symbol)
    return [f]


def _generators(f: Expr, symbol: Symbol) -> Optional[list[Expr]]:
    """The generators containing ``symbol`` of ``f`` read as a polynomial
    (sums, products and nonnegative integer powers of them), ``None``
    when ``f`` is not such a polynomial."""
    found: list[Expr] = []

    def visit(e: Expr) -> bool:
        if symbol not in e.free_symbols:
            return True
        if isinstance(e, (Add, Mul)):
            return all(isinstance(a, Expr) and visit(a) for a in e.args)
        if isinstance(e, Pow) and isinstance(e.exp, Integer) and e.exp.p > 0:
            return visit(e.base)
        if e not in found:
            found.append(e)
        return True

    return found if visit(f) else None


# PORTED FROM SYMPY: sympy/solvers/decompogen.py:compogen
def _compogen(g_s: Sequence[Expr], symbol: Symbol) -> Expr:
    """The composition of the functions of :func:`_decompogen`."""
    if len(g_s) == 1:
        return g_s[0]
    inner = _compogen(g_s[1:], symbol)
    return as_expr(g_s[0].subs(symbol, inner))


# PORTED FROM SYMPY: sympy/calculus/util.py:periodicity
def periodicity(f: Basic, symbol: Symbol, check: bool = False) -> Optional[Expr]:
    """The period of ``f`` in ``symbol`` (``0`` when ``f`` does not depend
    on it), ``None`` when ``f`` is not periodic or no period is found.
    With ``check`` the period is verified by substitution and ``None``
    returned when that fails (SymPy raises).

    Examples
    ========

    >>> from sympy_extras.core import Symbol, Mod
    >>> from sympy_extras.functions import sin, cos, Abs
    >>> from sympy_extras.calculus.util import periodicity
    >>> x = Symbol('x')
    >>> periodicity(sin(x)**2, x), periodicity(cos(x)*sin(x/2), x), periodicity(Mod(3*x, 5), x)
    (2*pi, 4*pi, 5/3)
    >>> periodicity(Abs(sin(x)) + 1, x), periodicity(sin(x) + x, x)
    (pi, None)
    """
    temp = Dummy('x', real=True)
    if isinstance(f, Relational):
        g: Expr = as_expr(f.lhs - f.rhs)
    elif isinstance(f, Expr):
        g = f
    else:
        raise TypeError("periodicity takes an expression or a relation, not %s" % (f,))
    g = as_expr(g.subs(symbol, temp))
    found = _periodicity_of(g, temp, check)
    if found is None:
        return None
    return as_expr(found.subs(temp, symbol))


def _periodicity_of(f: Expr, symbol: Symbol, check: bool) -> Optional[Expr]:
    from sympy_extras.simplify.simplify import simplify
    orig_f = f
    period: Optional[Expr] = None

    f = simplify(f)

    if symbol not in f.free_symbols:
        return Integer(0)

    if _is_trigonometric(f):
        period = _trigonometric_period(f, symbol)

    if isinstance(f, Abs):
        arg = as_expr(f.args[0])
        if isinstance(arg, (Sec, Csc, Cos)):
            # all but tan and cot might have a period half as large: as a sine
            arg = sin(as_expr(arg.args[0]))
        period = _periodicity_of(arg, symbol, False)
        if period is not None and isinstance(arg, Sin):
            # the half period of |sin|: tested on Abs(arg), which behaves
            # as the original function
            orig_f = abs_(arg)
            half = _checked(orig_f, symbol, period/2)
            if half is not None:
                return half

    if isinstance(f, Exp) or (isinstance(f, Pow) and f.base == E):
        exponent = as_expr(f.args[0]) if isinstance(f, Exp) else f.exp
        f = exp(expand_mul(exponent))
        if im(f) != 0:
            period_real = _periodicity_of(re(f), symbol, False)
            period_imag = _periodicity_of(im(f), symbol, False)
            if period_real is not None and period_imag is not None:
                period = lcim([period_real, period_imag])

    if isinstance(f, Pow) and f.base != E:
        base, expo = f.base, f.exp
        base_has_sym = base.has(symbol)
        expo_has_sym = expo.has(symbol)
        if base_has_sym and not expo_has_sym:
            period = _periodicity_of(base, symbol, False)
        elif expo_has_sym and not base_has_sym:
            period = _periodicity_of(expo, symbol, False)
        else:
            period = _periodicity_args([base, expo], symbol)
    elif isinstance(f, Mul):
        coeff, rest = f.as_independent(symbol, as_Add=False)
        if _is_trigonometric(rest) or coeff != 1:
            period = _periodicity_of(rest, symbol, False)
        else:
            period = _periodicity_args([as_expr(a) for a in rest.args], symbol)
    elif isinstance(f, Add):
        k, rest = f.as_independent(symbol)
        if k != 0:
            return _periodicity_of(rest, symbol, False)
        period = _periodicity_args([as_expr(a) for a in rest.args], symbol)
    elif isinstance(f, Mod):
        a, n = as_expr(f.args[0]), as_expr(f.args[1])
        if a == symbol:
            period = n
        elif _is_trigonometric(a):
            period = _periodicity_of(a, symbol, False)
        elif symbol not in n.free_symbols and a.is_polynomial(symbol) is True and \
                symbol not in a.diff(symbol).free_symbols and a.diff(symbol) != 0:
            # linear in the symbol
            period = abs_(n/a.diff(symbol))
    elif isinstance(f, Piecewise):
        pass   # not handled, as in SymPy
    elif period is None:
        g_s = _decompogen(f, symbol)
        count = len(g_s)
        if count > 1:
            for index in range(count):
                start = count - 1 - index
                g = _compogen(g_s[start:], symbol)
                if g != orig_f and g != f:
                    period = _periodicity_of(g, symbol, False)
                    if period is not None:
                        break

    if period is not None:
        if check:
            return _checked(orig_f, symbol, period)
        return period
    return None


# PORTED FROM SYMPY: sympy/calculus/util.py:_periodicity
def _periodicity_args(args: Sequence[Expr], symbol: Symbol) -> Optional[Expr]:
    """The common period of the arguments, ``None`` when one of them is
    not periodic."""
    periods: list[Expr] = []
    for f in args:
        period = _periodicity_of(f, symbol, False)
        if period is None:
            return None
        if period != 0:
            periods.append(period)
    if len(periods) > 1:
        return lcim(periods)
    if periods:
        return periods[0]
    return None


# PORTED FROM SYMPY: sympy/calculus/util.py:lcim
def lcim(numbers: Sequence[Expr]) -> Optional[Expr]:
    """The least common multiple of rational numbers, or of irrational
    numbers which are rational multiples of one term (``None`` otherwise).

    Examples
    ========

    >>> from sympy_extras.core import Rational, pi
    >>> from sympy_extras.calculus.util import lcim
    >>> lcim([Rational(1, 2), Rational(3, 4), Rational(5, 6)]), lcim([pi, pi/2, 2*pi/3])
    (15/2, 2*pi)
    >>> lcim([pi, Rational(1)]) is None
    True
    """
    from sympy_extras.polys.polytools import factor, lcm_list
    if all(num.is_irrational is True for num in numbers):
        factors = [factor(num).as_coeff_Mul() for num in numbers]
        term = factors[0][1]
        if all(rest == term for _, rest in factors):
            return lcm_list([coeff for coeff, _ in factors])*term
        return None
    if all(num.is_rational is True for num in numbers):
        return lcm_list(list(numbers))
    return None
