"""The solutions of the equations of Lambert type in one unknown by the
Lambert W function, on the objects of the core.

An equation which, after taking logarithms, reads
``a*log(b*X + c) + d*X + f = 0`` with ``X = g(x)`` has the solutions
``X = -c/b + (a/d)*W(d/(a*b)*exp((c*d - b*f)/(a*b)))`` (Corless et al.)
on the branches ``k = 0`` and, when real, ``k = -1`` of ``W``. The forms
recognised are those of SymPy's ``_solve_lambert``:
``B**B = R``, ``B*(b*log(B) + c)**a = R``, ``a*log(b*B + c) + d*B = R``,
``(b*B + c)*exp(d*B + g) = R``, ``g*exp(d*B + h) - b*B = c`` and
``d*p**(a*B + g) - b*B = c``.

The module imports no SymPy. It is a port of SymPy's
``sympy/solvers/bivariate.py``.

References
==========

.. [1] R. M. Corless, G. H. Gonnet, D. E. G. Hare, D. J. Jeffrey,
   D. E. Knuth, On the Lambert W function, Adv. Comput. Math. 5 (1996)
   329-359.

Examples
========

>>> from sympy_extras.core import Symbol
>>> from sympy_extras.functions import exp, log
>>> from sympy_extras.solvers.bivariate import solve_lambert
>>> x = Symbol('x')
>>> solve_lambert(x*exp(x) - 1, x)
[LambertW(1)]
>>> solve_lambert(x - exp(x) + 2, x)
[-2 - LambertW(-exp(-2)), -2 - LambertW(-exp(-2), -1)]
>>> solve_lambert(x + log(x) - 2, x)
[LambertW(exp(2))]
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/solvers/bivariate.py (_filtered_gens, _mostfunc,
# _linab, _lambert, _solve_lambert) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: strict typing; None instead of NotImplementedError;
# factor_terms for separatevars; the generators found by a traversal.
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Optional

from sympy_extras.core.add import Add
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr, as_expr
from sympy_extras.core.function import expand, expand_log
from sympy_extras.core.mul import Mul
from sympy_extras.core.numbers import Integer
from sympy_extras.core.power import Pow, pow
from sympy_extras.core.sorting import default_sort_key, ordered
from sympy_extras.core.symbol import Dummy, Symbol
from sympy_extras.functions.elementary.exponential import Exp, Log, exp, lambertw, log

__all__ = ['solve_lambert']

_ZERO = Integer(0)
_ONE = Integer(1)


def _expr(b: Basic) -> Expr:
    return as_expr(b)


def _mexpand(e: Expr) -> Expr:
    from sympy_extras.core.function import expand_mul, expand_multinomial
    return expand_mul(expand_multinomial(e))


# PORTED FROM SYMPY: sympy/solvers/bivariate.py:_mostfunc
def _mostfunc(lhs: Expr, func: type[Expr], x: Symbol) -> Optional[Expr]:
    """The subexpression of class ``func`` containing ``x`` with the most
    nested occurrences of ``func``."""
    found = [_expr(a) for a in lhs.atoms(func) if isinstance(a, Expr) and x in a.free_symbols]
    if not found:
        return None
    if len(found) == 1:
        return found[0]
    candidates = [_expr(a) for a in ordered(found)]
    return max(candidates, key=lambda a: a.count(func))


# PORTED FROM SYMPY: sympy/solvers/bivariate.py:_linab
def _linab(arg: Expr, x: Symbol) -> tuple[Expr, Expr, Expr]:
    """``(a, b, X)`` with ``arg = a*X + b``, ``a`` and ``b`` free of
    ``x``."""
    from sympy_extras.core.exprtools import factor_terms
    arg = factor_terms(expand(arg))
    independent, dependent = arg.as_independent(x)
    if isinstance(arg, Mul) and isinstance(dependent, Add):
        a_, b_, X_ = _linab(dependent, x)
        return independent*a_, independent*b_, X_
    b: Expr = _ZERO
    if not isinstance(arg, Add):
        a, X = independent, dependent
    else:
        b = independent
        a, X = factor_terms(dependent).as_independent(x, as_Add=False)
    if X.could_extract_minus_sign():
        a, X = -a, -X
    return a, b, X


def _real(e: Expr) -> Optional[bool]:
    """Whether a constant expression is real (by the assumptions, then by
    a reliable value)."""
    real = e.is_real
    if real is not None:
        return real
    from sympy_extras._numeric import reliable_value
    from sympy_extras.settings import settings
    if not settings.numerical_checks or not e.is_number:
        return None
    value = reliable_value(e, settings.precision)
    return None if value is None else value.is_real


# PORTED FROM SYMPY: sympy/solvers/bivariate.py:_lambert
def _lambert(eq: Expr, x: Symbol) -> list[Expr]:
    """The solutions of ``a*log(b*X + c) + d*X + f = 0`` with ``X = g(x)``:
    ``x = g^-1(-c/b + (a/d)*W(d/(a*b)*exp(c*d/a/b)*exp(-f/a)))``."""
    from sympy_extras.polys.polyroots import ex_roots
    from sympy_extras.simplify.radsimp import collect
    from .solvers import solve_equation
    # log(exp(u)) is written u (SymPy reaches these equations through its
    # bivariate_type and posify routes): the candidates are checked
    eq = _mexpand(expand_log(eq, force=True))
    mainlog = _mostfunc(eq, Log, x)
    if mainlog is None:
        return []
    other = _expr(eq.subs(mainlog, _ZERO))
    if isinstance(-other, Log):
        eq = _expr((eq - other).subs(mainlog, _expr(mainlog.args[0])))
        inner = _expr(mainlog.args[0])
        if not isinstance(inner, Log):
            return []
        mainlog = inner
        other = -_expr((-other).args[0])
        eq = eq + other
    if x not in other.free_symbols:
        return []
    d, f, X2 = _linab(other, x)
    logterm = collect(eq - other, mainlog)
    a = logterm.as_coefficient(mainlog)
    if a is None or x in a.free_symbols:
        return []
    logarg = _expr(mainlog.args[0])
    b, c, X1 = _linab(logarg, x)
    if X1 != X2:
        return []
    u = Dummy('rhs')
    inverses = solve_equation(X1 - u, x)
    if inverses is None:
        return []
    # the branches k = 0 and k = -1 of W are the only ones which may be
    # real; the argument exp((c*d - b*f)/(a*b)) is taken as the p-th roots
    # of exp(A) for exp(A/p)
    num, den = ((c*d - b*f)/a/b).as_numer_denom()
    p, den = den.as_coeff_Mul()
    e = exp(num/den)
    t = Dummy('t')
    if not isinstance(p, Integer) or p.p < 1:
        return []
    if p == _ONE:
        tees = [e]
    else:
        try:
            tees = ex_roots(pow(t, p) - e, t)
        except NotImplementedError:
            return []
    solutions: list[Expr] = []
    for root in tees:
        argument = d/(a*b)*root
        for k in (-1, 0):
            w = lambertw(argument, Integer(k)) if k else lambertw(argument)
            if k and _real(w) is not True:
                continue
            rhs = -c/b + (a/d)*w
            solutions.extend(_expr(xu.subs(u, rhs)) for xu in inverses)
    return solutions


def _lambert_generators(e: Expr, x: Symbol) -> list[Expr]:
    from .solvers import _generators
    return [g for g in _generators(e, x)
            if isinstance(g, (Exp, Log)) or (isinstance(g, Pow) and isinstance(g.exp, Expr) and g.exp.has(x))]


def _uniq(values: list[Expr]) -> list[Expr]:
    result: list[Expr] = []
    for v in values:
        if v not in result:
            result.append(v)
    return result


# PORTED FROM SYMPY: sympy/solvers/bivariate.py:_solve_lambert._solve_even_degree_expr
def _solve_even_degree_expr(expr: Expr, t: Symbol, x: Symbol, depth: int) -> Optional[list[Expr]]:
    negative, positive = (_expr(expr.xreplace({t: s*x})) for s in (Integer(-1), _ONE))
    solutions = _solve_lambert(negative, x, depth + 1)
    if solutions is None:
        solutions = []
    if positive != negative:
        more = _solve_lambert(positive, x, depth + 1)
        if more is not None:
            solutions.extend(more)
    return _uniq(solutions) if solutions else None


def _even_powers(lhs: Expr, x: Symbol, t: Symbol) -> Expr:
    """``x**(2k)`` replaced by ``t**(2k)``."""
    def query(a: Basic) -> bool:
        return isinstance(a, Pow) and a.base == x and isinstance(a.exp, Expr) and a.exp.is_even is True

    def value(a: Basic) -> Basic:
        assert isinstance(a, Pow)
        return pow(t, _expr(a.exp))
    return _expr(lhs.replace(query, value))


def solve_lambert(f: Expr, x: Symbol) -> Optional[list[Expr]]:
    """The solutions of ``f = 0`` in terms of the Lambert W function, or
    ``None`` when ``f`` is not of one of the Lambert forms (SymPy's
    ``_solve_lambert``, which raises ``NotImplementedError`` then); the
    solutions are not checked."""
    return _solve_lambert(f, x, 0)


# PORTED FROM SYMPY: sympy/solvers/bivariate.py:_solve_lambert
def _solve_lambert(f: Expr, x: Symbol, depth: int) -> Optional[list[Expr]]:
    from sympy_extras.polys.polytools import factor
    from sympy_extras.simplify.powsimp import powsimp
    from sympy_extras.simplify.radsimp import collect
    from .solvers import invert
    if depth > 4:
        return None
    nrhs, lhs = f.as_independent(x, as_Add=True)
    rhs = -nrhs
    if not _lambert_generators(f, x):
        return None
    if isinstance(lhs, (Add, Mul)):
        t = Dummy('t', **x.assumptions0)
        lhs = _even_powers(lhs, x, t)
        if isinstance(lhs, Add) and lhs.has(t):
            t_independent = _expr(lhs.subs(t, _ZERO))
            t_term = lhs - t_independent
            rhs_ = rhs - t_independent
            if not isinstance(t_term, Add) and rhs_ != _ZERO and t_term.is_finite is not False:
                equation = expand_log(log(t_term) - log(rhs_))
                return _solve_even_degree_expr(equation, t, x, depth)
        elif isinstance(lhs, Mul) and rhs != _ZERO:
            lhs = expand_log(log(lhs), force=True)
            rhs = log(rhs)
            if lhs.has(t) and isinstance(lhs, Add):
                return _solve_even_degree_expr(lhs - rhs, t, x, depth)
        back: dict[Basic, Basic] = {t: x}
        lhs = _expr(lhs.xreplace(back))
    lhs = powsimp(factor(lhs, deep=True))
    r = Dummy('r')
    i, lhs = invert(lhs - r, x)
    restore: dict[Basic, Basic] = {r: rhs}
    rhs = _expr(i.xreplace(restore))
    solutions: list[Expr] = []
    mainlog = _mostfunc(lhs, Log, x)
    if mainlog is not None:
        if isinstance(lhs, Mul) and rhs != _ZERO:
            solutions = _lambert(log(lhs) - log(rhs), x)
        elif isinstance(lhs, Add):
            other = _expr(lhs.subs(mainlog, _ZERO))
            if other != _ZERO and not isinstance(other, Add) and \
                    [a for a in other.atoms(Pow) if x in a.free_symbols]:
                if rhs == _ZERO:
                    difference = log(other) - log(other - lhs)
                else:
                    difference = log(lhs - other) - log(rhs - other)
                solutions = _lambert(expand_log(difference), x)
            else:
                solutions = _lambert(lhs - rhs, x)
    if not solutions:
        mainexp = _mostfunc(lhs, Exp, x)
        if mainexp is not None:
            lhs = collect(lhs, mainexp)
            if isinstance(lhs, Mul) and rhs != _ZERO:
                solutions = _lambert(expand_log(log(lhs) - log(rhs)), x)
            elif isinstance(lhs, Add):
                other = _expr(lhs.subs(mainexp, _ZERO))
                mainterm = lhs - other
                rhs = rhs - other
                if mainterm.could_extract_minus_sign() and rhs.could_extract_minus_sign():
                    mainterm, rhs = -mainterm, -rhs
                solutions = _lambert(expand_log(log(mainterm) - log(rhs)), x)
    if not solutions:
        mainpow = _mostfunc(lhs, Pow, x)
        if mainpow is not None and isinstance(mainpow, Pow) and isinstance(mainpow.exp, Expr) \
                and x in mainpow.exp.free_symbols:
            lhs = collect(lhs, mainpow)
            if isinstance(lhs, Mul) and rhs != _ZERO:
                solutions = _lambert(expand_log(log(lhs) - log(rhs)), x)
            elif isinstance(lhs, Add):
                other = _expr(lhs.subs(mainpow, _ZERO))
                mainterm = lhs - other
                rhs = rhs - other
                solutions = _lambert(expand_log(log(mainterm) - log(rhs)), x)
    if not solutions:
        return None
    return sorted(_uniq(solutions), key=default_sort_key)

