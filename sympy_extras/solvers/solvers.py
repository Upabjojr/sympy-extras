"""SymPy's ``solve`` of one equation in one unknown, and of a polynomial
system, on the objects of the core.

:func:`solve_equation` returns the solutions of ``f = 0`` for ``x`` as
SymPy's ``solve(f, x)`` lists them (a finite list over the complex
numbers, the principal branches of the inverse functions), or ``None``
when no method applies. The steps of SymPy's ``_solve`` are kept:

* a product is solved factor by factor;
* the numerator of ``f`` is a polynomial in ``x``: its roots
  (:func:`~sympy_extras.polys.polyroots.roots`, the formulas over ``EX`` of
  :func:`~sympy_extras.polys.polyroots.ex_roots`, the complex roots of
  :mod:`~sympy_extras.polys.complexrootof` for a polynomial with rational
  coefficients not solvable otherwise);
* the numerator is a polynomial in one generator ``g(x)`` (``exp(x)``,
  ``log(x)``, ``x**(1/3)``, the rational powers of a common base, an
  applied function): the roots ``t`` in the generator, then the
  equations ``g(x) = t``;
* radicals of one base removed (SymPy's ``unrad`` for one radical): the
  resultant of the polynomial in the radical with its defining
  equation;
* the transcendental equations (SymPy's ``_tsolve``): the inversion of
  the outer functions (:func:`invert`, SymPy's ``_invert``), a factored
  or log-combined left side, and the Lambert W forms of
  :mod:`sympy_extras.solvers.bivariate`.

Every candidate is checked as SymPy's ``checksol`` does: it is dropped
when it makes a denominator zero, or when the residual is provably not
zero (by the assumptions with the symbols positive, as ``checksol`` with
``posify``, and by reliable numerical values when it is a number); a
candidate which cannot be decided is kept.

:func:`solve_system` solves a system of equations for some unknowns by
substitution (the method of SymPy's ``nonlinsolve`` for the systems which
are not linear): an equation polynomial in an unknown is solved for it,
the values are substituted into the others, and the branches where the
leading coefficient vanishes are followed too; the unknowns left free
stand for themselves, as in SymPy's ``solve(..., dict=True)``.

The module imports no SymPy. It is a port of parts of SymPy's
``sympy/solvers/solvers.py``.

Examples
========

>>> from sympy_extras.core import symbols, sqrt
>>> from sympy_extras.functions import exp, log
>>> from sympy_extras.solvers.solvers import solve_equation, solve_system
>>> x, y, a = symbols('x y a')
>>> solve_equation(x**2 - a, x)
[-sqrt(a), sqrt(a)]
>>> solve_equation(log(y) - a*x - 1, y)
[exp(a*x + 1)]
>>> solve_equation(sqrt(x) + 1, x)
[]
>>> solve_equation(x*exp(x) - 1, x)
[LambertW(1)]
>>> solve_system([x + y - 2, x - y], [x, y])
[{x: 1, y: 1}]
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/solvers/solvers.py (_solve, _tsolve, _invert,
# checksol, _vsolve) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: strict typing, one equation in one unknown; None instead of
# NotImplementedError; unrad for radicals of one base only; the checks
# by the native assumptions and reliable numerical values.
# ---------------------------------------------------------------------
from __future__ import annotations

import math
from typing import Callable, Optional, Sequence

from sympy_extras._timeout import attempt
from sympy_extras.core.add import Add, add
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr, as_expr
from sympy_extras.core.function import Application, expand
from sympy_extras.core.mul import Mul
from sympy_extras.core.numbers import Integer, Rational
from sympy_extras.core.power import Pow, pow
from sympy_extras.core.sorting import default_sort_key
from sympy_extras.core.symbol import Dummy, Symbol
from sympy_extras.functions.elementary.exponential import Exp, LambertW, Log, exp, log
from sympy_extras.settings import settings

__all__ = ['solve_equation', 'solve_system', 'invert', 'checksol', 'zero_truth']

#: the depth of the recursive calls (equations for the generators, the
#: inverse functions, the Lambert forms) after which a solve gives up
_MAX_DEPTH = 12

_ZERO = Integer(0)
_ONE = Integer(1)


def _expr(b: Basic) -> Expr:
    return as_expr(b)


def _subs(e: Expr, old: Expr, new: Expr) -> Expr:
    return _expr(e.subs(old, new))


# ---------------------------------------------------------------------------
# the checks

def _numeric_truth(value: Expr) -> Optional[bool]:
    """Whether a number is zero by a reliable
    evaluation: ``True`` (zero to the precision), ``False`` (clearly not
    zero), ``None`` (not told, or no numerical checks)."""
    if not settings.numerical_checks:
        return None
    from sympy_extras._numeric import reliable_value
    digits = settings.precision
    found = attempt(lambda: reliable_value(value, digits), settings.timeout, native=True)
    if found is None or not found.is_number:
        return None
    try:
        size = abs(complex(found))
    except (TypeError, ValueError, OverflowError):
        return None
    if size > 10.0**(-(digits*2//3)):
        return False
    return True


# PORTED FROM SYMPY: sympy/solvers/solvers.py:checksol
def checksol(f: Expr, x: Symbol, v: Expr) -> Optional[bool]:
    """Whether ``x = v`` solves ``f = 0``: ``True`` when the residual is
    zero, ``False`` when it is provably not (by the assumptions with the
    symbols made positive, or by a reliable value for a number),
    ``None`` when this is not decided."""
    try:
        value = _subs(f, x, v)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    if value.has(x):
        return None
    return zero_truth(value)


def zero_truth(value: Expr) -> Optional[bool]:
    """Whether an expression is zero: ``True`` when it is, structurally
    or after ``cancel``, ``False`` when it is provably not (see
    :func:`checksol`), ``None`` when this is not decided."""
    from sympy_extras.polys.polytools import cancel
    from sympy_extras.simplify.simplify import posify
    if value == _ZERO:
        return True
    if value.is_infinite is True:
        return False
    if value.is_number:
        if value.is_zero is not None:
            return value.is_zero is True
        return _numeric_truth(value)
    try:
        cancelled = attempt(lambda: cancel(value), settings.timeout, native=True)
    except (TypeError, ValueError, ZeroDivisionError):
        cancelled = None
    if cancelled is not None and cancelled == _ZERO:
        return True
    # as checksol with posify: a residual which is not zero for positive
    # values of the symbols refutes the candidate
    positive, _ = posify(value)
    if positive.is_zero is False:
        return False
    return None


def _zeroes_denominator(denominator: Expr, x: Symbol, v: Expr) -> bool:
    """Whether ``v`` makes the denominator zero (proved)."""
    if not denominator.has(x):
        return False
    return checksol(denominator, x, v) is True


# ---------------------------------------------------------------------------
# polynomials

def _coefficients(e: Expr, v: Symbol) -> Optional[dict[int, Expr]]:
    """The coefficients of ``e`` as a polynomial in ``v`` (by degree),
    ``None`` when it is not one."""
    ex = expand(e)
    gathered: dict[int, list[Expr]] = {}
    for t in (ex.args if isinstance(ex, Add) else (ex,)):
        term = _expr(t)
        c, k = term.as_coeff_exponent(v)
        if c.has(v) or not isinstance(k, Integer) or k.p < 0:
            return None
        gathered.setdefault(int(k.p), []).append(c)
    result: dict[int, Expr] = {}
    for degree, cs in gathered.items():
        coefficient = add(*cs)
        if coefficient != _ZERO:
            result[degree] = coefficient
    return result


def _polynomial_roots(e: Expr, x: Symbol, degree: int) -> Optional[list[Expr]]:
    """All the roots of a polynomial in ``x`` (without multiplicities), or
    ``None`` when they are not all found."""
    from sympy_extras.polys.polyerrors import BasePolynomialError
    from sympy_extras.polys.polyroots import ex_roots, roots
    try:
        found = roots(e, x)
    except (BasePolynomialError, NotImplementedError, ValueError, TypeError):
        found = None
    if isinstance(found, dict):
        if sum(found.values()) == degree:
            return [r for r in found if isinstance(r, Expr)]
        if all(c.is_rational is True for c in e.free_symbols) and not (e.free_symbols - {x}):
            from sympy_extras.polys.complexrootof import all_roots
            listed = attempt(lambda: all_roots(e, x), settings.timeout, native=True)
            if listed is not None and len(listed) == degree:
                distinct: list[Expr] = []
                for r in listed:
                    if r not in distinct:
                        distinct.append(r)
                return distinct
        return None
    try:
        listed_ex = ex_roots(expand(e), x)
    except (BasePolynomialError, NotImplementedError, ValueError, TypeError):
        decomposed = _roots_in_power(e, x)
        if decomposed is None:
            return None
        listed_ex = decomposed
    if len(listed_ex) != degree:
        return None
    result: list[Expr] = []
    for r in listed_ex:
        if r not in result:
            result.append(r)
    return result


def _roots_in_power(e: Expr, x: Symbol) -> Optional[list[Expr]]:
    """The roots, with repetitions, of a polynomial in ``x**g`` (``g``
    the gcd of the exponents, at least two): the roots ``s`` of the
    polynomial in ``y = x**g``, then the ``g``-th roots of each."""
    coefficients = _coefficients(e, x)
    if coefficients is None:
        return None
    g = 0
    for k in coefficients:
        g = math.gcd(g, k)
    if g < 2:
        return None
    from sympy_extras.polys.polyerrors import BasePolynomialError
    from sympy_extras.polys.polyroots import ex_roots, roots
    y = Dummy('y')
    reduced = add(*[c*pow(y, Integer(k//g)) for k, c in coefficients.items()])
    try:
        inner = ex_roots(expand(reduced), y)
    except (BasePolynomialError, NotImplementedError, ValueError, TypeError):
        return None
    w = Dummy('w')
    unity = roots(pow(w, Integer(g)) - 1, w, multiple=True)
    if not isinstance(unity, list) or len(unity) != g:
        return None
    result: list[Expr] = []
    for value in inner:
        principal = pow(value, Rational(1, g))
        result.extend(principal*u for u in unity)
    return result


# ---------------------------------------------------------------------------
# the inversion of the outer functions

# PORTED FROM SYMPY: sympy/solvers/solvers.py:_invert
def invert(f: Expr, x: Symbol, rhs: Expr = _ZERO) -> tuple[Expr, Expr]:
    """``(r, g)`` with ``f = rhs`` equivalent (on the principal branches)
    to ``g = r``, the terms and factors free of ``x`` and the outer
    invertible functions moved to the right (SymPy's ``_invert``).

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.functions import exp, log
    >>> from sympy_extras.solvers.solvers import invert
    >>> x, y = symbols('x y')
    >>> invert(2*exp(x + 1) - y, x)
    (log(y/2) - 1, x)
    >>> invert(log(x)*x - 1, x)
    (1, x*log(x))
    """
    lhs, r = f, rhs
    for _ in range(100):
        if lhs == x or not lhs.has(x):
            break
        if isinstance(lhs, Add):
            independent, dependent = lhs.as_independent(x, as_Add=True)
            if independent != _ZERO:
                lhs, r = dependent, r - independent
                continue
            break
        if isinstance(lhs, Mul):
            independent, dependent = lhs.as_independent(x, as_Add=False)
            if independent != _ONE:
                lhs, r = dependent, r/independent
                continue
            break
        if isinstance(lhs, Pow):
            base, e = lhs.base, lhs.exp
            if isinstance(base, Expr) and isinstance(e, Expr):
                if not e.has(x):
                    if isinstance(e, Integer) and r != _ZERO:
                        break   # several roots: the polynomial solver takes them
                    lhs, r = base, pow(r, _ONE/e)
                    continue
                if not base.has(x):
                    lhs, r = e, log(r)/log(base)
                    continue
            break
        if isinstance(lhs, Exp):
            lhs, r = _expr(lhs.args[0]), log(r)
            continue
        if isinstance(lhs, Log):
            lhs, r = _expr(lhs.args[0]), exp(r)
            continue
        if isinstance(lhs, LambertW) and len(lhs.args) == 1:
            lhs, r = _expr(lhs.args[0]), r*exp(r)
            continue
        inverse = _inverse(lhs)
        if inverse is not None and len(lhs.args) == 1:
            lhs, r = _expr(lhs.args[0]), inverse(r)
            continue
        break
    return r, lhs


def _inverse(f: Expr) -> Optional[Callable[[Expr], Expr]]:
    """The inverse function of a trigonometric or hyperbolic function or
    of an inverse one (SymPy's ``inverse()``: the principal branch)."""
    from sympy_extras.functions.elementary import hyperbolic as h
    from sympy_extras.functions.elementary import trigonometric as t
    table: list[tuple[type[Expr], Callable[[Expr], Expr]]] = [
        (t.Sin, t.asin), (t.Cos, t.acos), (t.Tan, t.atan), (t.Cot, t.acot), (t.Sec, t.asec), (t.Csc, t.acsc),
        (t.Asin, t.sin), (t.Acos, t.cos), (t.Atan, t.tan), (t.Acot, t.cot), (t.Asec, t.sec), (t.Acsc, t.csc),
        (h.Sinh, h.asinh), (h.Cosh, h.acosh), (h.Tanh, h.atanh), (h.Coth, h.acoth), (h.Sech, h.asech),
        (h.Csch, h.acsch), (h.Asinh, h.sinh), (h.Acosh, h.cosh), (h.Atanh, h.tanh), (h.Acoth, h.coth),
        (h.Asech, h.sech), (h.Acsch, h.csch)]
    for cls, function in table:
        if isinstance(f, cls):
            return function
    return None


# ---------------------------------------------------------------------------
# generators

def _as_base_q(g: Expr) -> tuple[Expr, int]:
    """``(b**e, q)`` for ``g = b**(p*e/q)`` with ``p/q`` the leading
    rational of the exponent (SymPy's ``_as_base_q``): ``exp(-2*x/3)``
    gives ``(exp(x), 3)``."""
    if isinstance(g, Exp):
        argument = _expr(g.args[0])
        c, rest = argument.as_coeff_Mul()
        if isinstance(c, Rational) and c != _ONE:
            return exp(rest), int(c.q)
        return g, 1
    if isinstance(g, Pow):
        base, e = g.base, g.exp
        if isinstance(base, Expr) and isinstance(e, Expr):
            if isinstance(e, Rational):
                return base, int(e.q)
            c, rest = e.as_coeff_Mul()
            if isinstance(c, Rational) and c != _ONE:
                return pow(base, rest), int(c.q)
    return g, 1


def _generators(e: Expr, x: Symbol) -> list[Expr]:
    """The generators of ``e`` as a polynomial which contain ``x``: ``x``
    itself, powers with a non-integer exponent or an exponent in ``x``,
    and applications of functions."""
    found: list[Expr] = []

    def visit(a: Basic) -> None:
        if not a.has(x):
            return
        if a == x:
            if x not in found:
                found.append(x)
            return
        if isinstance(a, (Add, Mul)):
            for b in a.args:
                visit(b)
            return
        if isinstance(a, Pow):
            base, k = a.base, a.exp
            if isinstance(k, Integer):
                visit(base)
                return
        if isinstance(a, Expr) and a not in found:
            found.append(a)

    visit(e)
    return found


def _exponent_in(g: Expr, base: Expr, q: int) -> Optional[int]:
    """``k`` with ``g = t**k`` for ``t = base**(1/q)`` (``base`` an
    exponential ``exp(u)`` or a power ``b**e``)."""
    if isinstance(base, Exp) and isinstance(g, Exp):
        ratio = _expr(g.args[0])/_expr(base.args[0])
        from sympy_extras.polys.polytools import cancel
        r = cancel(ratio)
        k_ = r*q
        return int(k_.p) if isinstance(r, Rational) and isinstance(k_, Integer) else None
    b, e = g.as_base_exp()
    bb, be = base.as_base_exp()
    if b != bb:
        return None
    from sympy_extras.polys.polytools import cancel
    r = cancel(e/be)
    if isinstance(r, Rational):
        k = r*q
        if isinstance(k, Integer):
            return int(k.p)
    return None


def _by_generator(numerator: Expr, x: Symbol, depth: int) -> Optional[list[Expr]]:
    """The solutions when the numerator is a polynomial (or a rational
    function) in one generator ``t = g(x)``: the roots in ``t``, then the
    equations ``g(x) = t``."""
    gens = _generators(numerator, x)
    if not gens or gens == [x]:
        return None
    pairs = [_as_base_q(g) for g in gens]
    bases: list[Expr] = []
    for b, _ in pairs:
        if b not in bases:
            bases.append(b)
    if len(gens) == 1 and not isinstance(gens[0], (Application, Pow, Symbol)):
        # a node which is not a function, such as an integral with the
        # unknown in its limits (a SymPy object wrapped by the interop), is
        # not inverted: SymPy's solve gives no solution ([] for
        # Integral(g(y), (y, f)) - c)
        return []
    t = Dummy('t')
    if len(bases) == 1 and bases[0] != x:
        base = bases[0]
        q = 1
        for _, qi in pairs:
            q = math.lcm(q, qi)
        rule: dict[Basic, Basic] = {}
        for g in gens:
            k = _exponent_in(g, base, q)
            if k is None:
                return None
            rule[g] = pow(t, Integer(k))
        replaced = _expr(numerator.xreplace(rule))
        if replaced.has(x):
            return None
        generator = base if q == 1 else (exp(_expr(base.args[0])/q) if isinstance(base, Exp)
                                         else pow(base, Rational(1, q)))
    else:
        # the shallowest function whose replacement leaves no x
        functions = [g for g in gens if isinstance(g, Application)]
        functions.sort(key=lambda g: (g.count(Application), default_sort_key(g)))
        if not functions:
            return None
        generator = functions[0]
        replaced = _subs(numerator, generator, t)
        if replaced.has(x):
            return None
    values = _solve(replaced, t, depth + 1)
    if values is None:
        return None
    result: list[Expr] = []
    for value in values:
        found = _solve(generator - value, x, depth + 1)
        if found is None:
            return None
        result.extend(found)
    return result


# ---------------------------------------------------------------------------
# radicals

# PORTED FROM SYMPY: sympy/solvers/solvers.py:unrad (one radical base)
def _unrad(numerator: Expr, x: Symbol, depth: int) -> Optional[list[Expr]]:
    """The candidates when ``x`` occurs in polynomials and in rational
    powers of one base ``b = n/d`` rational in ``x``: the numerator is a
    polynomial ``P(r)`` in ``r = b**(1/q)`` with coefficients polynomial in
    ``x``, and the resultant of ``P(t)`` and ``d*t**q - n`` in ``t`` is a polynomial in
    ``x`` whose roots contain the solutions (to be checked)."""
    from sympy_extras.polys.polyerrors import BasePolynomialError
    from sympy_extras.polys.polytools import resultant
    gens = [g for g in _generators(numerator, x) if g != x]
    if not gens:
        return None
    bases: list[Expr] = []
    q = 1
    for g in gens:
        if not isinstance(g, Pow) or not isinstance(g.exp, Rational) or isinstance(g.exp, Integer):
            return None
        b = g.base
        if not isinstance(b, Expr) or b.is_rational_function(x) is not True:
            return None
        if b not in bases:
            bases.append(b)
        q = math.lcm(q, int(g.exp.q))
    if len(bases) != 1:
        return None
    base = bases[0]
    t = Dummy('t')
    rule: dict[Basic, Basic] = {}
    for g in gens:
        assert isinstance(g, Pow) and isinstance(g.exp, Rational)
        k = g.exp*q
        if not isinstance(k, Integer):
            return None
        rule[g] = pow(t, k)
    from sympy_extras.polys.rationaltools import together
    from sympy_extras.simplify.radsimp import fraction
    # the powers of t in denominators cleared (t is not zero where b is not)
    polynomial = fraction(together(_expr(numerator.xreplace(rule))))[0]
    base_numerator, base_denominator = fraction(together(base))
    if polynomial.is_polynomial(t, x) is not True:
        return None
    defining = base_denominator*pow(t, Integer(q)) - base_numerator
    eliminated: Optional[Expr] = None
    if q == 2:
        eliminated = _square_eliminated(polynomial, t, base_numerator, base_denominator)
    if eliminated is None:
        try:
            eliminated = attempt(lambda: resultant(expand(polynomial), expand(defining), t), settings.timeout,
                                 native=True)
        except (BasePolynomialError, NotImplementedError, ValueError, TypeError):
            return None
    if eliminated is None:
        return None
    if not eliminated.has(x):
        # a nonzero constant: no solution (x - sqrt(x**2 - 1) = 0)
        return [] if eliminated != _ZERO else None
    return _solve(eliminated, x, depth + 1)


def _square_eliminated(polynomial: Expr, t: Symbol, n: Expr, d: Expr) -> Optional[Expr]:
    """The resultant of ``P(t)`` and ``d*t**2 - n`` in ``t`` for any
    coefficients (the domains of the native resultant take no ``EX``):
    ``P = A + B*t`` modulo ``t**2 = n/d`` and ``d**k*(A**2 - B**2*n/d)``."""
    coefficients = _coefficients(polynomial, t)
    if coefficients is None:
        return None
    even: list[Expr] = []
    odd: list[Expr] = []
    top = max(coefficients) if coefficients else 0
    k = top//2
    for degree, c in coefficients.items():
        # t**degree = (n/d)**(degree//2) * t**(degree % 2), times d**k
        term = c*pow(n, Integer(degree//2))*pow(d, Integer(k - degree//2))
        (odd if degree % 2 else even).append(term)
    A, B = add(*even), add(*odd)
    return expand(d*A**2 - n*B**2)


# ---------------------------------------------------------------------------
# the transcendental equations

# PORTED FROM SYMPY: sympy/solvers/solvers.py:_tsolve
def _tsolve(numerator: Expr, x: Symbol, depth: int) -> Optional[list[Expr]]:
    from sympy_extras.polys.polytools import factor
    from sympy_extras.simplify.powsimp import powdenest
    from sympy_extras.simplify.simplify import logcombine
    rhs, lhs = invert(numerator, x)
    if lhs == x:
        return [rhs]
    if isinstance(lhs, Add):
        factored = attempt(lambda: factor(powdenest(lhs - rhs)), settings.timeout, native=True)
        if isinstance(factored, Mul) and sum(1 for m in factored.args if m.has(x)) > 1:
            return _solve(factored, x, depth + 1)
        if rhs != _ZERO:
            combined = logcombine(lhs, force=True)
            if combined.count(Log) != lhs.count(Log):
                if isinstance(combined, Log):
                    return _solve(_expr(combined.args[0]) - exp(rhs), x, depth + 1)
                return _solve(combined - rhs, x, depth + 1)
    elif isinstance(lhs, Pow):
        base, e = lhs.base, lhs.exp
        if isinstance(base, Expr) and isinstance(e, Expr):
            if isinstance(e, Integer) and e.p >= 2 and rhs != _ZERO and not rhs.has(x):
                # base**n = r: base is one of the n-th roots of r
                return _nth_roots(base, rhs, int(e.p), x, depth)
            if isinstance(e, Integer) and lhs - rhs != numerator:
                return _solve(lhs - rhs, x, depth + 1)
            if rhs == _ZERO and e.has(x) and base.has(x):
                found = _solve(base, x, depth + 1)
                if found is None:
                    return None
                return [s for s in found if _subs(e, x, s) != _ZERO]
            if base.has(x) and e.has(x):
                return _solve(e*log(base) - log(rhs), x, depth + 1)
    elif isinstance(lhs, (Exp, Log)) or (isinstance(lhs, LambertW) and len(lhs.args) == 1):
        return _solve(lhs - rhs, x, depth + 1)
    from .bivariate import solve_lambert
    return solve_lambert(lhs - rhs, x)


def _nth_roots(base: Expr, r: Expr, n: int, x: Symbol, depth: int) -> Optional[list[Expr]]:
    """The solutions of ``base**n = r``: ``base = r**(1/n)*w`` for the
    ``n``-th roots of unity ``w``."""
    from sympy_extras.polys.polyroots import roots
    w = Dummy('w')
    unity = roots(pow(w, Integer(n)) - 1, w, multiple=True)
    if not isinstance(unity, list) or len(unity) != n:
        return None
    principal = pow(r, Rational(1, n))
    result: list[Expr] = []
    for u in unity:
        found = _solve(base - principal*u, x, depth + 1)
        if found is None:
            return None
        result.extend(found)
    return result


# ---------------------------------------------------------------------------
# the solver

def _distinct(values: list[Expr]) -> list[Expr]:
    result: list[Expr] = []
    for v in values:
        if v not in result:
            result.append(v)
    return result


def _simplified(v: Expr) -> Expr:
    from sympy_extras.simplify.simplify import simplify
    found = attempt(lambda: simplify(v), settings.timeout, native=True)
    return found if found is not None else v


def _solve(f: Expr, x: Symbol, depth: int) -> Optional[list[Expr]]:
    """The solutions of ``f = 0`` (``None``: no method), unchecked except
    for the denominators."""
    if depth > _MAX_DEPTH:
        return None
    if not f.has(x):
        return []
    if f == x:
        return [_ZERO]
    if isinstance(f, Mul):
        found: list[Expr] = []
        for m in f.args:
            factor_ = _expr(m)
            if not factor_.has(x):
                if factor_.is_infinite is True:
                    return []
                continue
            if isinstance(factor_, Pow) and isinstance(factor_.exp, Expr) and factor_.exp.is_negative is True \
                    and not factor_.exp.has(x):
                continue   # a denominator: no root
            solved = _solve(factor_, x, depth + 1)
            if solved is None:
                return None
            found.extend(solved)
        return _distinct(found)
    from sympy_extras.polys.rationaltools import together
    from sympy_extras.simplify.radsimp import fraction
    numerator, denominator = fraction(together(f))
    if not numerator.has(x):
        return []
    candidates: Optional[list[Expr]] = None
    coefficients = _coefficients(numerator, x)
    checked = True
    if coefficients is not None and not any(c.has(x) for c in coefficients.values()):
        degree = max(coefficients)
        if degree == 0:
            return []
        candidates = _polynomial_roots(_expr(add(*[c*pow(x, Integer(k)) for k, c in coefficients.items()])), x,
                                       degree)
        checked = candidates is None
    if candidates is None:
        candidates = _by_generator(numerator, x, depth)
    if candidates is None:
        candidates = _unrad(numerator, x, depth)
    if candidates is None:
        candidates = _tsolve(numerator, x, depth)
    if candidates is None:
        return None
    result: list[Expr] = []
    for v in _distinct(candidates):
        if v.has(x):
            continue
        if _zeroes_denominator(denominator, x, v):
            continue
        if checked and checksol(numerator, x, v) is False:
            continue
        result.append(v)
    return result


def solve_equation(f: Expr, x: Symbol) -> Optional[list[Expr]]:
    """The solutions of ``f = 0`` for ``x`` (SymPy's ``solve(f, x)``), in
    the order of :func:`~sympy_extras.core.sorting.default_sort_key`;
    ``None`` when no method applies (see the module).

    Two solutions or fewer are simplified, as SymPy's ``solve`` does; the
    solutions contradicting the assumptions of ``x`` (an integer ``x``)
    are dropped, as with SymPy's ``check_assumptions``.

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.functions import exp
    >>> from sympy_extras.solvers.solvers import solve_equation
    >>> x, a = symbols('x a')
    >>> solve_equation(exp(2*x) - 3*exp(x) + 2, x)
    [0, log(2)]
    >>> solve_equation(x**3 - 1, x)
    [1, -1/2 - sqrt(3)*I/2, -1/2 + sqrt(3)*I/2]
    """
    from sympy_extras.core.assumptions import check_assumptions
    found = _solve(f, x, 0)
    if found is None:
        return None
    if len(found) <= 2:
        found = _distinct([_simplified(v) for v in found])
    found = [v for v in found if check_assumptions(v, x) is not False]
    return sorted(found, key=default_sort_key)


# ---------------------------------------------------------------------------
# systems

Solution = dict[Symbol, Expr]


def _unknowns_in(e: Expr, unknowns: Sequence[Symbol]) -> list[Symbol]:
    return [u for u in unknowns if e.has(u)]


def _choice(equations: list[Expr], unknowns: list[Symbol]) -> Optional[tuple[int, Symbol, dict[int, Expr]]]:
    """The equation and the unknown to solve for: an equation polynomial
    in an unknown of the lowest degree, the leading coefficients free of
    the other unknowns preferred."""
    best: Optional[tuple[tuple[int, int, int], int, Symbol, dict[int, Expr]]] = None
    for i, e in enumerate(equations):
        present = _unknowns_in(e, unknowns)
        for u in present:
            coefficients = _coefficients(e, u)
            if coefficients is None or any(c.has(u) for c in coefficients.values()):
                continue
            degree = max(coefficients)
            if degree < 1:
                continue
            lead = coefficients[degree]
            constrained = 1 if any(lead.has(v) for v in present) else 0
            key = (constrained, degree, len(present))
            if best is None or key < best[0]:
                best = (key, i, u, coefficients)
    if best is None:
        return None
    return best[1], best[2], best[3]


def _numerators(equations: Sequence[Expr]) -> list[Expr]:
    from sympy_extras.polys.rationaltools import together
    from sympy_extras.simplify.radsimp import fraction
    result: list[Expr] = []
    for e in equations:
        n = expand(fraction(together(e))[0])
        if n != _ZERO and n not in result:
            result.append(n)
    return result


def _nonzero_constant(e: Expr, unknowns: Sequence[Symbol]) -> bool:
    return not _unknowns_in(e, unknowns) and e != _ZERO


def _substitute(equations: list[Expr], u: Symbol, value: Expr) -> list[Expr]:
    from sympy_extras.polys.polytools import cancel
    replaced: list[Expr] = []
    for e in equations:
        r = _subs(e, u, value)
        r = cancel(r) if not r.has(Application) else r
        replaced.append(r)
    return _numerators(replaced)


def _branches(equations: list[Expr], unknowns: list[Symbol], depth: int, out: list[Solution]) -> bool:
    """Solve by substitution, appending the solutions (in the unknowns
    solved, in terms of the free ones) to ``out``; ``False`` when an
    equation cannot be solved."""
    if depth > 3*len(unknowns) + 6 or len(out) > 200:
        return False
    equations = _numerators(equations)
    if any(_nonzero_constant(e, unknowns) for e in equations):
        return True    # inconsistent
    equations = [e for e in equations if _unknowns_in(e, unknowns)]
    if not equations:
        out.append({})
        return True
    choice = _choice(equations, unknowns)
    if choice is None:
        return False
    index, u, coefficients = choice
    e = equations[index]
    others = equations[:index] + equations[index + 1:]
    degree = max(coefficients)
    lead = coefficients[degree]
    remaining = [v for v in unknowns if v != u]
    values = solve_equation(e, u) if degree > 2 or lead.has(Application) else _solve_low(coefficients, u)
    if values is None:
        return False
    complete = True
    for value in values:
        sub: list[Solution] = []
        if not _branches(_substitute(others, u, value), remaining, depth + 1, sub):
            complete = False
            continue
        for s in sub:
            v = value
            for w, wv in s.items():
                v = _subs(v, w, wv)
            from sympy_extras.polys.polytools import cancel
            v = cancel(v) if not v.has(Application) else v
            solution = dict(s)
            solution[u] = v
            out.append(solution)
    if _unknowns_in(lead, unknowns):
        # the branch where the leading coefficient vanishes
        sub_: list[Solution] = []
        if _branches(others + [lead, add(*[c*pow(u, Integer(k)) for k, c in coefficients.items() if k != degree])],
                     unknowns, depth + 1, sub_):
            out.extend(sub_)
        else:
            complete = False
    return complete


def _solve_low(coefficients: dict[int, Expr], u: Symbol) -> Optional[list[Expr]]:
    """The roots of a polynomial of degree one or two with the leading
    coefficient assumed nonzero."""
    from sympy_extras.core.power import sqrt
    from sympy_extras.polys.polytools import cancel
    degree = max(coefficients)
    c = [coefficients.get(k, _ZERO) for k in range(degree, -1, -1)]
    if degree == 1:
        return [cancel(-c[1]/c[0])]
    if degree == 2:
        a, b, c0 = c
        if b == _ZERO:
            r = sqrt(cancel(-c0/a))
            return _distinct([-r, r])
        d = sqrt(cancel(b**2 - 4*a*c0))
        return _distinct([cancel(-b/(2*a)) - d/(2*a), cancel(-b/(2*a)) + d/(2*a)])
    return solve_equation(add(*[ci*pow(u, Integer(k)) for k, ci in coefficients.items()]), u)


def _verified(solution: Solution, equations: Sequence[Expr]) -> bool:
    """Whether a solution is not refuted by substitution into the
    equations."""
    rule: dict[Basic, Basic] = {k: v for k, v in solution.items()}
    return all(zero_truth(_expr(e.subs(rule))) is not False for e in equations)


def _groebner(equations: list[Expr], unknowns: list[Symbol]) -> Optional[list[Expr]]:
    """The reduced lexicographic Gröbner basis of polynomial equations
    with rational coefficients and no other symbol (as SymPy's ``solve``
    computes for a polynomial system), ``None`` for the other systems or
    when it is not computed in time."""
    from fractions import Fraction
    from sympy_extras.polys.domains import QQ
    from sympy_extras.polys.ideals import Ideal
    from sympy_extras.polys.orderings import grevlex, lex
    if not equations or any(e.free_symbols - set(unknowns) for e in equations):
        return None
    if any(e.is_polynomial(*unknowns) is not True for e in equations):
        return None

    def basis() -> list[Expr]:
        ideal: Ideal[Fraction] = Ideal(equations, *unknowns, domain=QQ, order=lex)
        # the inconsistent systems by the cheaper graded basis first
        graded = ideal.groebner_basis(grevlex)
        if any(not any(g.LM) for g in graded):
            return [Integer(1)]
        return [ideal.to_expr(g) for g in ideal.groebner_basis()]
    return attempt(basis, settings.timeout, native=True)


def solve_system(equations: Sequence[Expr], unknowns: Sequence[Symbol]) -> Optional[list[Solution]]:
    """The solutions of the equations (each ``= 0``) for the unknowns, as
    SymPy's ``solve(equations, unknowns, dict=True)``: a list of
    dictionaries, the unknowns missing from one free; ``None`` when an
    equation met cannot be solved (see the module).

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.solvers.solvers import solve_system
    >>> x, y, a = symbols('x y a')
    >>> solve_system([x*y - 1, x - y], [x, y])
    [{x: -1, y: -1}, {x: 1, y: 1}]
    >>> solve_system([x + y - a], [x, y])
    [{x: a - y}]
    >>> solve_system([x**2 + 1, x - 1], [x])
    []
    """
    unknowns_ = list(unknowns)
    system = _numerators(equations)
    basis = _groebner(system, unknowns_)
    if basis is not None:
        if any(_nonzero_constant(g, unknowns_) for g in basis):
            return []
        system = basis
    out: list[Solution] = []
    if not _branches(system, unknowns_, 0, out):
        return None
    result: list[Solution] = []
    keys: list[str] = []
    for s in out:
        if not _verified(s, equations):
            continue
        key = str(sorted((str(k), str(v)) for k, v in s.items()))
        if key in keys:
            continue
        keys.append(key)
        result.append({u: s[u] for u in unknowns_ if u in s})
    result.sort(key=lambda s: [default_sort_key(s.get(u, u)) for u in unknowns_])
    return result
