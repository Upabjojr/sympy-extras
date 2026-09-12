r"""Indefinite integration: every antiderivative checked before it is returned.

SymPy's ``integrate`` tries its methods in a fixed order and returns the
first result, and a few of them are wrong (a heuristic which mistakes a
component, a branch of a logarithm, a rule applied outside its
hypotheses); here the methods are the typed ones of this package first,
then SymPy's, and a candidate antiderivative `F` is returned only when
it *is* one: `F' - f` cancels to zero, or simplifies to zero, or vanishes
at random points where `f` is defined (:func:`is_antiderivative`). A
candidate carrying `i` or a polar number for a real integrand is first
rewritten to a real form when one checks (:func:`real_form`). When no
candidate checks the ``Integral`` is returned unevaluated.

The order of the methods:

1. rational functions by Hermite reduction and the Lazard–Rioboo–Trager
   logarithmic part (:mod:`.risch.rationaltools`);
2. radicals of a quadratic, `x^n Q^{m/2}` (:mod:`.radicals`);
3. products of powers of trigonometric and hyperbolic functions, their
   rational functions and their products with polynomials, exponentials
   and inverse functions (:mod:`.trigonometric`);
4. the transcendental Risch algorithm (:mod:`.risch`), which also proves
   non-elementarity;
5. the heuristic Risch integrator (:mod:`.heurisch`, then SymPy's);
6. Trager's algorithm for one square root of a polynomial (:mod:`.trager`);
7. SymPy's rule-based ``manualintegrate``, its Meijer G-function route
   and its ``integrate``.

Examples
========

>>> from sympy import symbols, sqrt, exp, sin, Integral
>>> from sympy_extras.integrals.indefinite import indefinite_integral
>>> x = symbols('x')
>>> indefinite_integral(sqrt(1 - x**2), x)
x*sqrt(1 - x**2)/2 + asin(x)/2
>>> indefinite_integral(x/(x**2 + 1), x)
log(x**2 + 1)/2
>>> indefinite_integral(exp(-x**2), x)
sqrt(pi)*erf(x)/2
>>> indefinite_integral(x**x, x)
Integral(x**x, x)
"""
from __future__ import annotations

from typing import Callable, Optional

from sympy.core.expr import Expr
import random

from sympy.core.numbers import I, Rational, nan, oo, zoo
from sympy.core.add import Add
from sympy.core.symbol import Dummy, Symbol
from sympy.logic.boolalg import Boolean, false
from sympy.functions.elementary.complexes import Abs, im
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.trigonometric import atan
from sympy.functions.elementary.exponential import exp_polar
from sympy.integrals.integrals import Integral, integrate
from sympy.integrals.heurisch import heurisch_wrapper
from sympy.integrals.manualintegrate import manualintegrate
from sympy.polys.polytools import cancel
from sympy.simplify.simplify import simplify
from sympy.simplify.simplify import logcombine

from sympy_extras._timeout import attempt
from sympy_extras._typing import ExprLike, as_boolean, as_expr, free_symbols, sorted_symbols
from sympy_extras.assumptions.ask import Assumptions
from sympy_extras.settings import settings
from .conditions import numerically_equal, sample_values

__all__ = ['indefinite_integral', 'verified_antiderivative', 'is_antiderivative', 'real_form', 'conjugate_logarithms']

Method = Callable[[Expr, Symbol], Optional[Expr]]


def _budget() -> Optional[float]:
    return None if settings.timeout is None else settings.timeout / 4


def is_antiderivative(F: ExprLike, f: ExprLike, x: Symbol, assumptions: Assumptions = None) -> Optional[bool]:
    """Whether ``F`` is an antiderivative of ``f``: ``True`` when ``F' - f``
    cancels or simplifies to zero, or vanishes numerically at random
    points satisfying the assumptions; ``False`` when it does not vanish
    there; ``None`` when nothing could be decided (an ``F`` which is not
    an expression in ``x``, an unevaluated ``Integral``).

    >>> from sympy import symbols, log, sin, cos
    >>> from sympy_extras.integrals.indefinite import is_antiderivative
    >>> x = symbols('x')
    >>> is_antiderivative(log(x**2 + 1)/2, x/(x**2 + 1), x)
    True
    >>> is_antiderivative(sin(x), sin(x), x)
    False
    """
    F_, f_ = as_expr(F), as_expr(f)
    if F_.has(Integral, nan, zoo):
        return None
    difference = as_expr(F_.diff(x) - f_)
    reduced = attempt(lambda: as_expr(cancel(difference)), _budget())
    if reduced is not None and reduced == 0:
        return True
    simpler = attempt(lambda: as_expr(simplify(reduced if reduced is not None else difference)), _budget())
    if simpler is not None and simpler == 0:
        return True
    if numerically_equal(as_expr(F_.diff(x)), f_, assumptions) is not True:
        return False
    # the sampler above draws positive points: the census of SymPy's
    # mistakes found nine antiderivatives right for x > 0 and wrong for
    # x < 0 (polar incomplete gammas, -asinh(1/x), Bessel forms), so
    # the difference is evaluated on both sides of 0 as well
    return _vanishes_on_both_sides(difference, x, assumptions)


def _items(assumptions: Assumptions) -> list[Boolean]:
    if assumptions is None:
        return []
    if isinstance(assumptions, (Boolean, bool)):
        return [as_boolean(assumptions)]
    return [as_boolean(a) for a in assumptions]


def _vanishes_on_both_sides(difference: Expr, x: Symbol, assumptions: Assumptions) -> bool:
    """Whether ``difference`` vanishes at fixed real points of both signs
    (the parameters sampled under the assumptions), where it is a finite
    real or complex number; a point where it is undefined is skipped."""
    parameters = sorted_symbols(free_symbols(difference) - {x})
    values: dict[Symbol, Expr] = {}
    if parameters:
        found = sample_values(parameters, assumptions, random.Random(str(difference)))
        if found is None:
            return True
        values = found
    facts = _items(assumptions)
    for point in (Rational(-37, 10), Rational(-13, 10), Rational(-2, 5), Rational(3, 5), Rational(19, 10)):
        # a point the assumptions on x exclude (x > 0) is not a test
        if any(fact.xreplace(values).xreplace({x: point}) == false for fact in facts):
            continue
        value = as_expr(difference.xreplace(values).xreplace({x: point}))
        number = attempt(lambda: value.evalf(30), _budget())
        if number is None or not number.is_number or number.has(nan, zoo, oo, -oo):
            continue
        magnitude = as_expr(Abs(number))
        if magnitude.is_comparable and magnitude > Rational(1, 10**15):
            return False
    return True


def real_form(F: Expr, f: Expr, x: Symbol, assumptions: Assumptions = None) -> Expr:
    """``F`` rewritten without ``I`` and polar numbers when a rewriting
    which is still an antiderivative of ``f`` is found (``logcombine``,
    ``simplify``, the arctangent forms of the logarithms of conjugates);
    ``F`` itself otherwise."""
    if not F.has(I, exp_polar) or f.has(I):
        return F
    candidates: list[Callable[[], Expr]] = [
        lambda: conjugate_logarithms(F, x),
        lambda: as_expr(simplify(conjugate_logarithms(F, x))),
        lambda: as_expr(logcombine(F, force=True)),
        lambda: as_expr(simplify(F)),
        lambda: as_expr(simplify(logcombine(F, force=True))),
    ]
    for candidate in candidates:
        rewritten = attempt(candidate, _budget())
        if rewritten is None or rewritten.has(I, exp_polar):
            continue
        if is_antiderivative(rewritten, f, x, assumptions) is True:
            return rewritten
    return F


def conjugate_logarithms(F: Expr, x: Symbol) -> Expr:
    """The pairs ``a*log(p) - a*log(conjugate(p))`` of ``F``, ``a`` imaginary
    and ``p = u + I*v`` with ``u``, ``v`` real in a real ``x``, written as
    the arctangents ``2*Im(a)*atan(u/v)`` (continuous where ``v`` does not
    vanish): the form the logarithmic part of a rational integral takes
    when its residues are conjugate.

    >>> from sympy import symbols, log, I
    >>> from sympy_extras.integrals.indefinite import conjugate_logarithms
    >>> x = symbols('x')
    >>> conjugate_logarithms(-I*log(x - I)/2 + I*log(x + I)/2, x)
    atan(x)
    """
    real = Dummy('x', real=True)
    terms: list[tuple[Expr, Expr, Expr]] = []                # (coefficient, u, v) of coefficient*log(u + I v)
    others: list[Expr] = []
    for term in Add.make_args(F):
        coefficient, rest = as_expr(term).as_independent(x, as_Add=False)
        coefficient_, rest_ = as_expr(coefficient), as_expr(rest)
        if isinstance(rest_, log) and coefficient_.is_imaginary:
            u, v = as_expr(rest_.args[0]).xreplace({x: real}).as_real_imag()
            terms.append((coefficient_, as_expr(u).xreplace({real: x}), as_expr(v).xreplace({real: x})))
        else:
            others.append(as_expr(term))
    used: set[int] = set()
    for i, (a, u, v) in enumerate(terms):
        if i in used or v == 0 or u == 0:
            continue
        for j in range(i + 1, len(terms)):
            b, u2, v2 = terms[j]
            if j in used or u2 != u or as_expr(v2 + v) != 0 or as_expr(a + b) != 0:
                continue
            # log(u + I v) - log(u - I v) = 2 I atan2(v, u) = 2 I (pi/2 - atan(u/v)) for v > 0:
            # the constant dropped, and atan(u/v) is continuous where v does not vanish
            others.append(as_expr(2 * im(a) * atan(u / v)))
            used.update((i, j))
            break
        else:
            others.append(as_expr(a * log(u + I * v)))
    if not used:
        return F
    return as_expr(Add(*others))


def _rational(f: Expr, x: Symbol) -> Optional[Expr]:
    from .risch.rationaltools import ratint
    if not f.is_rational_function(x):
        return None
    return attempt(lambda: as_expr(ratint(f, x)), _budget())


def _radicals(f: Expr, x: Symbol) -> Optional[Expr]:
    from .radicals import quadratic_radical_antiderivative
    return attempt(lambda: quadratic_radical_antiderivative(f, x), _budget())


def _trigonometric(f: Expr, x: Symbol) -> Optional[Expr]:
    from .trigonometric import trigonometric_antiderivative
    return attempt(lambda: trigonometric_antiderivative(f, x), _budget())


def _risch(f: Expr, x: Symbol) -> Optional[Expr]:
    from .risch import risch_antiderivative
    return attempt(lambda: risch_antiderivative(f, x), _budget())


def _heurisch(f: Expr, x: Symbol) -> Optional[Expr]:
    from .heurisch import heurisch_cases
    found = attempt(lambda: heurisch_cases(f, x), _budget())
    if found is not None:
        return found
    found = attempt(lambda: as_expr(heurisch_wrapper(f, x)), _budget())
    return None if found is None or found.has(Integral) else found


def _trager(f: Expr, x: Symbol) -> Optional[Expr]:
    from .trager import trager_antiderivative
    return attempt(lambda: trager_antiderivative(f, x), _budget())


def _manual(f: Expr, x: Symbol) -> Optional[Expr]:
    found = attempt(lambda: as_expr(manualintegrate(f, x)), _budget())
    return None if found is None or found.has(Integral) else found


def _meijer(f: Expr, x: Symbol) -> Optional[Expr]:
    found = attempt(lambda: as_expr(integrate(f, x, meijerg=True, risch=False)), _budget())
    return None if found is None or found.has(Integral) else found


def _sympy(f: Expr, x: Symbol) -> Optional[Expr]:
    found = attempt(lambda: as_expr(integrate(f, x, risch=False)), _budget())
    return None if found is None or found.has(Integral) else found


#: the methods in the order they are tried
METHODS: list[tuple[str, Method]] = [
    ('rational', _rational), ('radicals', _radicals), ('trigonometric', _trigonometric), ('risch', _risch),
    ('heurisch', _heurisch), ('trager', _trager), ('manual', _manual), ('meijer', _meijer), ('sympy', _sympy)]


def verified_antiderivative(f: ExprLike, x: Symbol, assumptions: Assumptions = None,
                            methods: Optional[list[str]] = None) -> Optional[tuple[Expr, str]]:
    """An antiderivative of ``f`` which checks by :func:`is_antiderivative`,
    with the name of the method which found it; ``None`` when no method
    gives one. ``methods`` restricts the names tried, in the order of
    :data:`METHODS`.

    >>> from sympy import symbols, exp, sin
    >>> from sympy_extras.integrals.indefinite import verified_antiderivative
    >>> x = symbols('x')
    >>> verified_antiderivative(x*exp(x), x)
    ((x - 1)*exp(x), 'risch')
    """
    f_ = as_expr(f)
    for name, method in METHODS:
        if methods is not None and name not in methods:
            continue
        try:
            found = method(f_, x)
        except (AttributeError, ZeroDivisionError, AssertionError, OverflowError, RecursionError):
            # SymPy's internals fail on some inputs
            found = None
        if found is None or found.has(Integral, nan, zoo, oo, -oo):
            continue
        if found.free_symbols - f_.free_symbols - {x}:
            # a constant of integration or a dummy leaked
            continue
        found = real_form(found, f_, x, assumptions)
        if is_antiderivative(found, f_, x, assumptions) is True:
            return (found, name)
    return None


def indefinite_integral(f: ExprLike, x: Symbol, assumptions: Assumptions = None) -> Expr:
    """``Integral(f, x)`` by the methods of the module documentation, each
    result checked before it is returned; the unevaluated ``Integral``
    when none gives an antiderivative which checks.

    Parameters
    ==========

    f : Expr
    x : Symbol
    assumptions : Boolean or list of Booleans, optional
        Assumptions on the parameters, used by the numerical part of the
        check (the points are sampled under them).

    Examples
    ========

    >>> from sympy import symbols, log, tan
    >>> from sympy_extras.integrals.indefinite import indefinite_integral
    >>> x = symbols('x')
    >>> indefinite_integral(1/(x*(log(x)**2 + 1)), x)
    atan(log(x))
    >>> indefinite_integral(tan(x)**3, x)
    log(cos(x)) + tan(x)**2/2
    """
    f_ = as_expr(f)
    found = verified_antiderivative(f_, x, assumptions)
    if found is None:
        return as_expr(Integral(f_, x))
    return found[0]
