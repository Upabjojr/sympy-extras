"""Definite integration: the driver.

:func:`definite_integral` computes ``Integral(f, (x, a, b))`` with
assumptions on the parameters. It does not compute antiderivatives: the
value of a definite integral with a singular integrand, an infinite
range or parameters is obtained from the structure of the integrand,
the way Mathematica's ``Integrate`` and the DEFINT package of REDUCE
work, and the conditions on the parameters under which the integral
converges come out of the computation. The steps are

1. **Splitting.** The range is cut at the points where the integrand is
   not analytic: the zeros of the arguments of ``Abs``, ``sign``,
   ``Heaviside`` and ``Max``/``Min``, the conditions of a ``Piecewise``,
   the zeros of the factors of a radicand or of the argument of a
   logarithm, and the singularities of the integrand inside the range
   (poles, branch points), found with :func:`sympy_extras.assumptions.solve.solve`
   and :func:`sympy_extras.integrals.antiderivative.singularities`. On each piece
   the integrand is replaced by its analytic branch, and a power or a
   logarithm of a product is split over the factors whose sign on the
   piece is known (``(u*v)**r`` is ``u**r * v**r`` for ``u > 0``):
   ``sqrt(x**2 + 2*x + 1)`` is ``x + 1`` or ``-x - 1``, ``(1 - cos(x))**(3/2)``
   is ``2*sqrt(2)*sin(x/2)**3`` on ``(0, 2*pi)``, ``log(sin(x)/x)`` is
   ``log(sin(x)) - log(x)`` on ``(0, pi/2)``. The integral over a piece
   with a singular endpoint is an improper integral, whose convergence is
   part of the conditions.

2. **The Marichev–Adamchik method** (:mod:`.marichev`): the range is
   mapped onto `(0, \\infty)`, `(0, 1)` or `(1, \\infty)` by a linear change
   of variable, and the integral of a power of ``x`` times at most two
   functions of the Mellin table is a Meijer G-function, written as
   hypergeometric functions by Slater's theorem. This is the method for
   integrals with singularities: the strips of the Mellin transforms
   are the convergence conditions, so ``Integral(x**k/(x + 3), (x, 0,
   oo))`` is ``-3**k*pi/sin(pi*k)`` for ``-1 < k < 0`` and nothing else.

3. **Changes of variable** on the way there: ``x = log(u)`` for
   integrands in ``exp(x)`` over the real line; ``x = 1/u`` between
   `(0, 1)` and `(1, \\infty)`; ``x = asin(sqrt(u))`` for trigonometric
   integrands over `(0, \\pi/2)`, which turns powers of ``sin`` and
   ``cos`` into a Beta integral, after cutting a range with endpoints
   at multiples of `\\pi/2` into such pieces.

4. **Residues** (:mod:`.residues`) for rational functions times
   powers, exponentials or trigonometric functions over `(0, \\infty)`
   and over the real line, and for rational functions of ``sin`` and
   ``cos`` over a period.

5. **SymPy's** ``integrate`` as the last resort, under the time limit
   of the settings, and only when its result passes a numerical check
   (``settings.numerical_checks``; an answer the quadrature cannot
   confirm is dropped): ``integrate`` evaluates an antiderivative at
   the endpoints and sometimes misses a singularity in between, a
   condition on the parameters, or a branch cut.

The conditions are decided against the assumptions
(:func:`sympy_extras.assumptions.ask.ask`); what stays undecided is
returned in a ``Piecewise`` with the unevaluated integral
(:class:`~sympy_extras.integrals.integral.Integral`) as the other
branch, as ``integrate`` does.

The module works on the objects of :mod:`sympy_extras.core` and imports
no SymPy: every method is native (the Mellin table, the residues, the
tables, the antiderivatives and the other modules of the package, the
native ``solve``, ``ask``, SAT solver, limits, factorisation and
simplification, the verified antiderivatives of the native
:mod:`.indefinite`, the native cases of the special values of the
parameters), and the numerical checks evaluate the integrands by the
numerical rules of the nodes. The last of the methods, where SymPy's
``integrate`` and ``meijerint`` were, is the native Meijer G-function
integration (:mod:`.meijerint`) with an antiderivative at the endpoints
(the delta functions by :mod:`.deltafunctions`). The functions on SymPy
objects are in :mod:`.definite_sympy`.

Examples
========

>>> from sympy_extras.core import symbols, oo, pi, sqrt
>>> from sympy_extras.functions import exp, sin, log, abs_
>>> from sympy_extras.integrals.definite import definite_integral
>>> x, k = symbols('x k')
>>> a, s = symbols('a s', positive=True)
>>> definite_integral(x**k/(x + 3), (x, 0, oo))
Piecewise((-3**k*pi/sin(pi*k), (k > -1) & (k < 0)), (Integral(x**k/(x + 3), (x, 0, oo)), True))
>>> definite_integral(x**k/(x + 3), (x, 0, oo), (k > -1) & (k < 0))
-3**k*pi/sin(pi*k)
>>> definite_integral(exp(-s*x)*sin(a*x)/x, (x, 0, oo))
atan(a/s)
>>> definite_integral(abs_(x - 1)/sqrt(x), (x, 0, 2))
2*(4 - sqrt(2))/3
>>> definite_integral(log(x)**2/(1 + x**2), (x, 0, oo))
pi**3/8
>>> definite_integral(sqrt(sin(x)), (x, 0, pi/2))
2*sqrt(pi)*gamma(3/4)/gamma(1/4)
"""
from __future__ import annotations

import cmath
import itertools
import random
from fractions import Fraction
from itertools import product
from math import gcd, lcm
from typing import Callable, Final, Mapping, Optional, Sequence

import mpmath

from sympy_extras._numeric import reliable_value
from sympy_extras._special_values import (Point, condition_point, defined_problem, equality_points,
                                          point_assumptions, with_special_values)
from sympy_extras._timeout import attempt
from sympy_extras.assumptions.ask import Assumptions, ask
from sympy_extras.assumptions.facts import element
from sympy_extras.assumptions.sat import satisfiable
from sympy_extras.assumptions.solve import solve
from sympy_extras.core.add import Add, add
from sympy_extras.core.basic import Basic
from sympy_extras.core.coreerrors import PoleError
from sympy_extras.core.evalf import numeric_value, to_float
from sympy_extras.core.expr import Expr, ExprLike, as_expr, to_expr
from sympy_extras.core.exprtools import factor_terms
from sympy_extras.core.function import (Application, DefinedFunction, count_ops, expand, expand_func, expand_log,
                                        expand_trig)
from sympy_extras.core.limits import Limit, limit
from sympy_extras.core.logic import (And, Boolean, BooleanFalse, BooleanTrue, Not, Or, and_, false, not_, or_,
                                     to_boolean, true)
from sympy_extras.core.mul import Mul, mul
from sympy_extras.core.numbers import I, Integer, Rational, nan, number_from_fraction, oo, pi, zoo
from sympy_extras.core.power import Pow, sqrt
from sympy_extras.core.relational import Eq, Ge, Gt, Le, Lt, Ne, Relational, eq, ge, gt, le, lt
from sympy_extras.core.singleton import S
from sympy_extras.core.symbol import Dummy, Symbol
from sympy_extras.functions.combinatorial.factorials import factorial
from sympy_extras.functions.elementary.complexes import Abs, Im, Re, Sign, abs_, sign
from sympy_extras.functions.elementary.exponential import Exp, Log, exp, log
from sympy_extras.functions.elementary.hyperbolic import (Acosh, Acoth, Acsch, Asech, Asinh, Atanh, Cosh, Coth, Csch,
                                                          Sech, Sinh, Tanh)
from sympy_extras.functions.elementary.integers import Ceiling, Floor, Frac
from sympy_extras.functions.elementary.miscellaneous import Max, Min
from sympy_extras.functions.elementary.piecewise import ExprCondPair, Piecewise, piecewise_fold
from sympy_extras.functions.elementary.trigonometric import (Asin, Atan, Cos, Cot, Csc, Sec, Sin, Tan, asin, cos,
                                                             sin, tan)
from sympy_extras.functions.special.delta_functions import DiracDelta, Heaviside
from sympy_extras.functions.special.gamma_functions import gamma
from sympy_extras.functions.special.hyper import hyper
from sympy_extras.functions.special.polynomials import Hermite, Laguerre, OrthogonalPolynomial
from sympy_extras.functions.special.zeta_functions import Zeta, stieltjes
from sympy_extras.polys.cad.indexedroot import IndexedRoot, real_root
from sympy_extras.polys.cad.samplepoints import real_roots_of
from sympy_extras.polys.polyerrors import BasePolynomialError
from sympy_extras.polys.polyroots import roots_linear, roots_quadratic
from sympy_extras.polys.polytools import Poly, cancel, factor
from sympy_extras.polys.rationaltools import together
from sympy_extras.polys.roots import radical_form
from sympy_extras.sets import ConditionSet, EmptySet, FiniteSet, Integers, Intersection, Interval, Reals, Set, Union
from sympy_extras.settings import settings
from sympy_extras.simplify.fu import TR8, TR9
from sympy_extras.simplify.powsimp import powdenest
from sympy_extras.simplify.simplify import nsimplify
from sympy_extras.simplify.trigsimp import trigsimp

from .antiderivative import singularities
from .conditions import ConditionalValue, _items, decide, plain_symbols, sample_values, with_symbol_facts
from .integral import Integral, is_unevaluated
from .marichev import checked_tidy, mellin_integrate, tidy
from .mellin import mellin_transform, monomial
from .parametric import ParametricIntegrator

__all__ = ['definite_integral', 'conditional_integral', 'verify_numerically', 'Limits', 'DefiniteIntegrator',
           'definite_integrator']

#: ``(x, a, b)``
Limits = tuple[Symbol, ExprLike, ExprLike]

#: the depth of the recursive splitting
_MAX_DEPTH: Final = 6
#: the most terms a sum is integrated term by term
_TERMWISE_LIMIT: Final = 8
#: the most points whose position against the endpoints the assumptions
#: do not settle for which the cases of the parameters are enumerated
#: (four positions each when the order of the endpoints is unknown)
_MAX_CASE_POINTS: Final = 2

#: the points of a range which lie strictly inside it, sorted, and those
#: whose position against an endpoint the assumptions do not settle
_Placement = tuple[list[Expr], list[Expr]]

#: the trigonometric functions (SymPy's ``TrigonometricFunction``)
_TRIGONOMETRIC: Final[tuple[type[Expr], ...]] = (Sin, Cos, Tan, Cot, Sec, Csc)
#: the hyperbolic functions (SymPy's ``HyperbolicFunction``)
_HYPERBOLIC: Final[tuple[type[Expr], ...]] = (Sinh, Cosh, Tanh, Coth, Sech, Csch)
#: the inverse hyperbolic functions (SymPy's ``InverseHyperbolicFunction``)
_INVERSE_HYPERBOLIC: Final[tuple[type[Expr], ...]] = (Asinh, Acosh, Atanh, Acoth, Asech, Acsch)
#: the nodes with kinks: their branches change at the zeros of arguments
_KINKS: Final[tuple[type[Expr], ...]] = (Abs, Sign, Heaviside, Max, Min, Piecewise)

_COUNTER: Final = itertools.count()


def _fresh(name: str, **assumptions: bool) -> Dummy:
    """A dummy with a name of its own: the typed polynomials map their
    symbols by name, so the variable ``t`` of a substitution and the
    ``t`` of a substitution made inside it (or a parameter ``t``) would be
    taken for one symbol."""
    return Dummy('%s%d' % (name, next(_COUNTER)), **assumptions)


def _rule(rule: Mapping[Symbol, Expr]) -> dict[Basic, Basic]:
    replacement: dict[Basic, Basic] = {}
    for key, value in rule.items():
        replacement[key] = value
    return replacement


def _xr(e: Expr, rule: Mapping[Symbol, Expr]) -> Expr:
    """``e`` with the symbols replaced by values, the rebuilt nodes
    evaluated (SymPy's ``xreplace``)."""
    return as_expr(e._xreplace_evaluated(_rule(rule)))


def _xr_boolean(b: Boolean, rule: Mapping[Symbol, Expr]) -> Boolean:
    return to_boolean(b._xreplace_evaluated(_rule(rule)))


def _replaced(e: Expr, old: Expr, new: Expr) -> Expr:
    """``e`` with the subtree ``old`` replaced by ``new``, the rebuilt
    nodes evaluated."""
    rule: dict[Basic, Basic] = {old: new}
    return as_expr(e._xreplace_evaluated(rule))


def _subs(e: Expr, old: Expr, new: ExprLike) -> Expr:
    return as_expr(e.subs(old, to_expr(new)))


def _sorted_symbols(symbols: set[Symbol]) -> list[Symbol]:
    """Symbols sorted by name."""
    return sorted(symbols, key=lambda s: s.name)


def _rational(p: int, q: int) -> Rational:
    """``p/q`` as an exact number (an ``Integer`` when ``q`` divides ``p``)."""
    return number_from_fraction(Fraction(p, q))


def _ops(e: Basic) -> int:
    """The number of operations of ``e`` (SymPy's ``count_ops``)."""
    return int(as_expr(count_ops(e)))


def _is(b: Boolean) -> bool:
    """Whether a relation evaluated to ``true``."""
    return isinstance(b, BooleanTrue)


def _has_trigonometric(f: Expr) -> bool:
    return f.has(*_TRIGONOMETRIC)


def definite_integral(f: ExprLike, limits: Limits, assumptions: Assumptions = None,
                      conds: str = 'piecewise', recognize: bool = False,
                      principal_value: bool = False, finite_part: bool = False,
                      numeric: bool = False, digits: int = 15, regularize: bool = False,
                      summability: Optional[str] = None, special_values: bool = True) -> Expr:
    """``Integral(f, (x, a, b))`` under assumptions on the parameters.

    Parameters
    ==========

    f : Expr
        The integrand.
    limits : (Symbol, Expr, Expr)
        The variable and the bounds, ``-oo`` and ``oo`` allowed.
    assumptions : Boolean or list of Booleans, optional
        Assumptions on the parameters as statements (``a > 0``), see
        :mod:`sympy_extras.assumptions`.
    conds : ``'piecewise'`` or ``'none'``
        What to do with the conditions the assumptions do not settle:
        return a ``Piecewise`` with the unevaluated integral as the
        other branch (the default), or assume that they hold.
    recognize : bool
        When every method fails and the integral has no parameters,
        guess a closed form from a high-precision numerical value with
        PSLQ (:mod:`.recognize`): the result is a conjecture checked to
        forty-five digits, not a proof, hence off by default.
    principal_value : bool
        Cauchy's principal value when the integrand has a singularity
        inside the range at which the integral diverges
        (:func:`~sympy_extras.integrals.antiderivative.principal_value_integral`).
    finite_part : bool
        Hadamard's finite part of a divergent integral, the divergent
        terms of the excision at each singularity dropped
        (:func:`~sympy_extras.integrals.antiderivative.finite_part_integral`).
    numeric : bool
        When every symbolic method fails and the integral has no
        parameters, a ``Float`` with ``digits`` correct digits, proved by
        validated integration in interval arithmetic
        (:func:`~sympy_extras.integrals.validated.validated_integral`);
        ``None`` there leaves the integral unevaluated.
    regularize : bool
        Analytic (Riesz) regularisation of the Mellin method: the value
        of the formula, analytic in the exponents, beyond the strips of
        convergence, where the integral itself diverges
        (``Integral(x**(-3/2)*exp(-x), (x, 0, oo))`` gives ``gamma(-1/2)``).
    summability : str or None
        ``'abel'``, ``'cesaro'`` or ``'gaussian'``: the integral taken in
        the sense of that summability method (:mod:`.summability`),
        ``Integral(sin(x), (x, 0, oo))`` being 1 in each; a convergent
        integral keeps its value.
    special_values : bool
        Whether the isolated values of the parameters at which the value
        found is undefined (``0/0``, a division by zero, ``log(0)``) get
        cases of their own, the integral computed again at each
        (:mod:`sympy_extras._special_values`): ``y**x`` over ``(0, 2)``
        is ``(y**2 - 1)/log(y)`` except at ``y = 1``, where it is ``2``;
        a point where the integrand is not defined (``a**(b*x)`` at ``a =
        0``, ``DiracDelta(a*x)`` at ``a = 0``) gets no case.
        On by default; the methods which integrate over the parameters
        afterwards (the regions) turn it off, the values on a set of
        measure zero being of no use there.

    Returns
    =======

    The value, a ``Piecewise`` on the conditions (the isolated special
    values of the parameters first, as ``Eq(p, v)``), ``oo`` or ``-oo`` when
    the integral diverges to a signed infinity (an infinite one-sided
    limit of an antiderivative at an endpoint or a singularity, as for
    ``1/x`` over ``(0, 1)``; an oscillatory divergence, ``cos(x)`` over
    ``(0, oo)``, and infinities of both signs, ``1/x`` over ``(-1, 1)``,
    are left unevaluated), or the unevaluated
    :class:`~sympy_extras.integrals.integral.Integral` when no method
    applies.

    Examples
    ========

    >>> from sympy_extras.core import symbols, oo, Integer
    >>> from sympy_extras.functions import exp, cos, log
    >>> from sympy_extras.assumptions.facts import element
    >>> from sympy_extras.sets import Reals
    >>> from sympy_extras.integrals.definite import definite_integral
    >>> x, a, b = symbols('x a b')
    >>> definite_integral(exp(-a*x)*cos(b*x), (x, 0, oo), (a > 0) & element(b, Reals))
    a/(a**2 + b**2)
    >>> definite_integral(x**a*log(x), (x, 0, 1), a > -1)
    -1/(a + 1)**2
    >>> definite_integral(1/x, (x, 0, 1))
    oo
    """
    x, a, b = limits[0], to_expr(limits[1]), to_expr(limits[2])
    f_ = to_expr(f)
    if conds not in ('piecewise', 'none'):
        raise ValueError("conds must be 'piecewise' or 'none', got %r" % (conds,))
    found: Optional[ConditionalValue]
    if summability is not None:
        from .summability import summable_integral
        found = summable_integral(f_, x, a, b, summability, assumptions)
    else:
        found = conditional_integral(f_, x, a, b, assumptions, principal_value, finite_part, regularize)
    if (found is None or found.value.has(nan)) and recognize:
        from .recognize import recognize_integral
        guessed = recognize_integral(f_, (x, a, b), assumptions)
        if guessed is not None:
            return guessed
    if (found is None or found.value.has(nan)) and numeric:
        from .validated import validated_integral
        enclosed = validated_integral(f_, x, a, b, digits, assumptions)
        if enclosed is not None:
            return enclosed[0]
    if found is None or found.value.has(nan):
        return Integral(f_, x, a, b)
    # the last step, checked like the ones before it: the simplification
    # is kept unless it is found to change the value (case by case for
    # the cases of the parameters)
    found = found.mapped(lambda value, condition: checked_tidy(value, assumptions, condition))
    result = found.value if conds == 'none' else found.as_piecewise(Integral(f_, x, a, b))
    if not special_values:
        return result
    parameters = _sorted_symbols((f_.free_symbols | a.free_symbols | b.free_symbols) - {x})

    def at_point(point: Point, at: Assumptions) -> Optional[Expr]:
        g, lower, upper = _xr(f_, point), _xr(a, point), _xr(b, point)
        value = definite_integral(g, (x, lower, upper), at, conds, recognize, principal_value, finite_part,
                                  numeric, digits, regularize, summability)
        if not is_unevaluated(value):
            return value
        # the limit of the generic value, when the integral at the point
        # is not found: accepted when the quadrature confirms it (the
        # value may differ from the limit)
        return _checked_limit(result, point, g, x, lower, upper, at) or value

    def defined(point: Point) -> bool:
        return defined_problem(_xr(f_, point), [x]) and not any(_xr(e, point).has(nan, zoo) for e in (a, b))

    return with_special_values(result, parameters, assumptions, at_point, defined)


def _checked_limit(value: Expr, point: Point, g: Expr, x: Symbol, a: Expr, b: Expr,
                   assumptions: Assumptions) -> Optional[Expr]:
    """The limit of the branch of ``value`` taken at the point (one
    parameter), when it is finite and agrees with the quadrature of
    ``Integral(g, (x, a, b))``, ``g`` the integrand at the point;
    ``None`` otherwise."""
    if len(point) != 1 or not settings.numerical_checks:
        return None
    ((p, v),) = point.items()
    for e, condition in (_pairs(value) if isinstance(value, Piecewise) else [(value, true)]):
        if is_unevaluated(e) or isinstance(_xr_boolean(condition, point), BooleanFalse):
            continue
        found = attempt(lambda: limit(e, p, v, '+-'), None if settings.timeout is None else settings.timeout / 4,
                        native=True)
        if found is None or found.is_finite is not True or found.has(Limit):
            return None
        return found if verify_numerically(found, g, x, a, b, assumptions) is True else None
    return None


def conditional_integral(f: Expr, x: Symbol, a: Expr, b: Expr,
                         assumptions: Assumptions = None,
                         principal_value: bool = False,
                         finite_part: bool = False,
                         regularize: bool = False) -> Optional[ConditionalValue]:
    """The value of ``Integral(f, (x, a, b))`` with the condition on the
    parameters under which it holds, or ``None``.

    The methods of this package run under half of the time limit of the
    settings, so that the last method (the Meijer G-function theorems and
    an antiderivative at the endpoints) gets the other half when they
    fail; a value they find under a condition the assumptions do not
    settle is kept unless the last method finds an unconditional one.

    >>> from sympy_extras.core import symbols, oo
    >>> from sympy_extras.functions import exp
    >>> from sympy_extras.integrals.definite import conditional_integral
    >>> x, a = symbols('x a')
    >>> conditional_integral(exp(-a*x), x, 0, oo)
    ConditionalValue(1/a, Abs(arg(a)) < pi/2)
    """
    f, a, b = _with_equalities(to_expr(f), to_expr(a), to_expr(b), x, assumptions)
    integrator = _Integrator(assumptions, principal_value=principal_value, finite_part=finite_part,
                             regularize=regularize)
    if not _real_bounds(a, b) or _nested_complex_powers(f):
        return integrator._last(f, x, a, b)
    # a singularity or a kink whose position against the endpoints the
    # assumptions do not settle: the cases of the parameters get the whole
    # time limit, and SymPy's integrate is not asked on the whole range
    # (the bug: 1/x over (a, b) came out as -log(a) + log(b) for plain a
    # and b, integrate's answer checked at positive samples only, where
    # the integral diverges for a < 0 < b)
    parametric = integrator._parametric_points(f, x, a, b)
    budget = None if settings.timeout is None else settings.timeout / (1 if parametric else 2)
    found = attempt(lambda: integrator.integrate(f, x, a, b, 0, False), budget, native=True)
    if found is not None and found.condition is true:
        return found
    if parametric:
        return found
    fallback = integrator._last(f, x, a, b)
    if fallback is not None and found is not None and fallback.condition is true and settings.numerical_checks \
            and not _holds_outside(fallback.value, found.condition, f, x, a, b, assumptions):
        # an unconditional answer of SymPy against a condition of the
        # methods of the package: checked where that condition fails, the
        # samples of the other check falling where both hold (the bug:
        # integrate(x**n, (x, 0, 1), meijerg=True) is gamma(n + 1)/gamma(n + 2)
        # with no condition, which the samples n > 0 confirmed, and it
        # replaced 1/(n + 1) for n > -1: a finite value where the integral
        # diverges)
        return found
    if fallback is not None and (found is None or fallback.condition is true):
        return fallback
    return found


def _assumption_items(assumptions: Assumptions) -> list[Boolean]:
    """The assumptions as a list of statements."""
    return _items(assumptions)


def _with_equalities(f: Expr, a: Expr, b: Expr, x: Symbol, assumptions: Assumptions) -> tuple[Expr, Expr, Expr]:
    """The integrand and bounds with the equalities among the assumptions
    (``Eq(n, m)``, a symbol equal to an expression) substituted: a
    formula valid for generic parameters may fail on the equality (the
    bug: ``sin(m x) sin(n x)`` over a period came out as ``-pi m/(2 n)``
    under ``Eq(n, m)``, from the generic antiderivative ``sin((m - n) x)
    /(m - n)``; the value is ``pi``)."""
    replacement: dict[Symbol, Expr] = {}
    for item in _assumption_items(assumptions):
        atoms: list[Basic] = [item] if isinstance(item, Eq) else list(item.atoms(Eq)) if isinstance(item, And) else []
        for atom in atoms:
            if not isinstance(atom, Eq):
                continue
            lhs, rhs = atom.lhs, atom.rhs
            if isinstance(lhs, Symbol) and lhs != x and not rhs.has(lhs, x):
                replacement[lhs] = rhs
            elif isinstance(rhs, Symbol) and rhs != x and not lhs.has(rhs, x):
                replacement[rhs] = lhs
    if not replacement:
        return f, a, b
    return _xr(f, replacement), _xr(a, replacement), _xr(b, replacement)


def _nested_complex_powers(f: Expr) -> bool:
    """Whether ``f`` has a power of a power with a non-real exponent,
    ``(x**(I/2))**I``: the exponents of a positive base combine
    (``x**(-1/2)``), which the principal branches do not allow when the
    inner argument leaves ``(-pi, pi]``, so the methods which substitute
    a positive variable are not used (the bug: the integral of
    ``(z**(I/2))**I`` over ``(0, 1)`` came out as 2, the value of
    ``z**(-1/2)``; Mathematica's NIntegrate disagrees)."""
    for node in f.atoms(Pow):
        if not isinstance(node, Pow):
            continue
        base, exponent = node.base, node.exp
        if exponent.is_extended_real is False and isinstance(base, Pow) and base.exp.is_extended_real is False:
            return True
    return False


def _real_bounds(a: Expr, b: Expr) -> bool:
    """Whether both bounds are real (or infinite), so that the range is a
    real interval; a complex bound means a segment in the complex plane,
    which only SymPy's antiderivatives handle."""
    for bound in (a, b):
        if bound in (oo, -oo):
            continue
        if bound.is_extended_real is False:
            return False
        if bound.is_extended_real is None and not bound.free_symbols:
            return False
    return True


# ---------------------------------------------------------------------------
# Numerical verification

#: a function of one complex variable evaluated by mpmath
_Numeric = Callable[[mpmath.mpc], mpmath.mpc]
#: a condition of a ``Piecewise`` decided at a value of the variable
_Condition = Callable[[mpmath.mpc], bool]


def _constant(value: mpmath.mpc) -> _Numeric:
    return lambda z: value


def _sum(parts: list[_Numeric]) -> _Numeric:
    def evaluate(z: mpmath.mpc) -> mpmath.mpc:
        total = mpmath.mpc(0)
        for part in parts:
            total = total + part(z)
        return total
    return evaluate


def _product(parts: list[_Numeric]) -> _Numeric:
    def evaluate(z: mpmath.mpc) -> mpmath.mpc:
        total = mpmath.mpc(1)
        for part in parts:
            total = total * part(z)
        return total
    return evaluate


def _power(base: _Numeric, exponent: _Numeric) -> _Numeric:
    def evaluate(z: mpmath.mpc) -> mpmath.mpc:
        u, v = base(z), exponent(z)
        if mpmath.im(u) == 0 and mpmath.im(v) == 0:
            # real arguments as lambdify's: mpmath.power of mpf values
            return mpmath.mpc(mpmath.power(mpmath.re(u), mpmath.re(v)))
        return mpmath.mpc(mpmath.power(u, v))
    return evaluate


def _function(node: DefinedFunction, parts: list[_Numeric]) -> _Numeric:
    def evaluate(z: mpmath.mpc) -> mpmath.mpc:
        value = node.numeric([part(z) for part in parts])
        if value is None:
            # no finite value there (log(0), a pole), as mpmath's -inf
            # through lambdify: the quadrature notes the node
            return mpmath.mpc(mpmath.nan)
        return mpmath.mpc(value)
    return evaluate


def _numeric_integral(integrand: Expr, u: Symbol, lower: Expr, upper: Expr, x: Symbol) -> _Numeric:
    """A definite integral in ``u`` inside an integrand of ``x`` (an
    unevaluated ``Integral``, native or SymPy's), by mpmath's quadrature
    at each value of ``x`` (what lambdify does with ``Integral``)."""
    lo, hi = _numeric_function(lower, x), _numeric_function(upper, x)

    def evaluate(z: mpmath.mpc) -> mpmath.mpc:
        a, b = lo(z), hi(z)
        if mpmath.im(a) != 0 or mpmath.im(b) != 0:
            raise ValueError("a complex bound of an inner integral")
        g = _numeric_function(_subs(integrand, x, _number_of(z)), u)
        return mpmath.mpc(mpmath.quad(lambda w: g(mpmath.mpc(w)), [mpmath.re(a), mpmath.re(b)]))
    return evaluate


def _number_of(z: mpmath.mpc) -> Expr:
    """An mpmath number as an exact number of the core (the rational of
    its binary digits), for an inner integrand at a value of ``x``."""
    value: Expr = number_from_fraction(_fraction(mpmath.re(z)))
    if mpmath.im(z) != 0:
        value = value + I * number_from_fraction(_fraction(mpmath.im(z)))
    return value


def _fraction(x: mpmath.mpf) -> Fraction:
    if not mpmath.isfinite(x):
        raise ValueError("no exact value of %s" % (x,))
    man, exp = mpmath.mpf(x).man_exp
    return Fraction(int(man)) * Fraction(2)**int(exp)


def _integral_parts(e: Expr) -> Optional[tuple[Expr, Symbol, Expr, Expr]]:
    """``(f, u, a, b)`` of a definite integral node: the native one, or
    SymPy's wrapped in a node of the core (``Integral(f, (u, a, b))``)."""
    if isinstance(e, Integral) and e.bounds is not None:
        return e.function, e.variable, e.bounds[0], e.bounds[1]
    if e._is_foreign_node() and e._compare_name() == 'Integral' and len(e.args) == 2:
        f, limits = e.args
        if isinstance(f, Expr) and len(limits.args) == 3:
            u, a, b = limits.args
            if isinstance(u, Symbol) and isinstance(a, Expr) and isinstance(b, Expr):
                return f, u, a, b
    return None


def _real(value: mpmath.mpc) -> mpmath.mpf:
    if mpmath.im(value) != 0:
        raise ValueError("a relation between complex numbers")
    return mpmath.re(value)


def _relation(kind: type[Relational], lhs: _Numeric, rhs: _Numeric) -> _Condition:
    def evaluate(z: mpmath.mpc) -> bool:
        u, v = lhs(z), rhs(z)
        if kind is Eq:
            return bool(u == v)
        if kind is Ne:
            return bool(u != v)
        p, q = _real(u), _real(v)
        if kind is Lt:
            return bool(p < q)
        if kind is Le:
            return bool(p <= q)
        if kind is Gt:
            return bool(p > q)
        return bool(p >= q)
    return evaluate


def _numeric_condition(c: Basic, x: Symbol) -> _Condition:
    """A condition of a ``Piecewise`` as a test on the value of ``x``;
    ``ValueError`` for one without a numerical rule."""
    if isinstance(c, BooleanTrue):
        return lambda z: True
    if isinstance(c, BooleanFalse):
        return lambda z: False
    if isinstance(c, Relational) and type(c) in (Eq, Ne, Lt, Le, Gt, Ge):
        return _relation(type(c), _numeric_function(c.lhs, x), _numeric_function(c.rhs, x))
    if isinstance(c, (And, Or)):
        parts = [_numeric_condition(a, x) for a in c.args]
        if isinstance(c, And):
            return lambda z: all(part(z) for part in parts)
        return lambda z: any(part(z) for part in parts)
    if isinstance(c, Not):
        inner = _numeric_condition(c.args[0], x)
        return lambda z: not inner(z)
    raise ValueError("%s has no numerical rule" % (c,))


def _numeric_piecewise(node: Piecewise, x: Symbol) -> _Numeric:
    branches: list[tuple[_Numeric, _Condition]] = []
    for pair in node.args:
        branches.append((_numeric_function(pair.expr, x), _numeric_condition(pair.cond, x)))

    def evaluate(z: mpmath.mpc) -> mpmath.mpc:
        for value, condition in branches:
            if condition(z):
                return value(z)
        raise ValueError("no branch of %s holds" % (node,))
    return evaluate


def _numeric_function(e: Expr, x: Symbol) -> _Numeric:
    """``e`` as a function of ``x`` evaluated by mpmath at its working
    precision (the role of SymPy's ``lambdify(x, e, 'mpmath')``): the
    numerical rules of the nodes, the branches of a ``Piecewise`` by its
    conditions; ``ValueError`` for a node without a numerical rule."""
    if e == x:
        return lambda z: z
    if not e.has(x):
        value = numeric_value(e, mpmath.mp.prec)
        if value is None:
            raise ValueError("%s has no numerical value" % (e,))
        return _constant(mpmath.mpc(value))
    if isinstance(e, Add):
        return _sum([_numeric_function(as_expr(a), x) for a in e.args])
    if isinstance(e, Mul):
        return _product([_numeric_function(as_expr(a), x) for a in e.args])
    if isinstance(e, Pow):
        return _power(_numeric_function(e.base, x), _numeric_function(e.exp, x))
    if isinstance(e, Piecewise):
        return _numeric_piecewise(e, x)
    parts = _integral_parts(e)
    if parts is not None:
        return _numeric_integral(parts[0], parts[1], parts[2], parts[3], x)
    if isinstance(e, DefinedFunction):
        return _function(e, [_numeric_function(as_expr(a), x) for a in e.args])
    raise ValueError("%s has no numerical rule" % (e,))


def _mpf(e: Expr) -> Optional[mpmath.mpf]:
    """A real number ``e`` as an mpmath number at 20 digits."""
    if e == oo:
        return mpmath.mpf('inf')
    if e == -oo:
        return mpmath.mpf('-inf')
    with mpmath.workdps(20):
        value = numeric_value(e, mpmath.mp.prec)
    if value is None:
        return None
    return mpmath.mpf(mpmath.re(value))


def _quadrature(f: Expr, x: Symbol, a: Expr, b: Expr, inside: Sequence[Expr] = ()) -> Optional[complex]:
    """``Integral(f, (x, a, b))`` by mpmath's quadrature, trusted only when
    two rules agree; a complex bound means the straight segment from
    ``a`` to ``b``, parametrised by ``x = a + (b - a) t``. The numeric
    points ``inside`` a finite real range (singularities, kinks) are
    nodes of the subdivision. ``None`` for an infinite value, which checks
    nothing (the bug: ``log(Abs(x - 1/2))`` over ``(0, 1)`` came out as
    ``-oo``, the midpoint of the rules at the singularity)."""
    if not _real_bounds(a, b):
        if a in (-oo, oo) or b in (-oo, oo) or a.free_symbols or b.free_symbols:
            return None
        t = _fresh('t', real=True)
        g_ = _subs(f, x, a + (b - a) * t) * (b - a)
        return _quadrature(g_, t, S.Zero, S.One, _axis_crossings(g_, t))
    with mpmath.workdps(20):
        try:
            g = _numeric_function(f, x)
        except (ValueError, TypeError, ZeroDivisionError, OverflowError):
            return None
    # the nodes where the integrand is not finite, taken as 0: a node next
    # to an end of a piece rounds the argument of the integrand onto its
    # singularity (the bug: acoth((1 - u**2)/(1 + u**2)), log-singular at
    # u = 0, is acoth(1) = oo at the nodes u < 1e-10 of the rules, and
    # the quadrature of u**2 times it came out oo, so that nothing was
    # checked), which carries no weight; one elsewhere does, and then the
    # quadrature is not trusted
    singular: list[mpmath.mpf] = []

    def finite(u: mpmath.mpf) -> mpmath.mpc:
        value = g(mpmath.mpc(u))
        if mpmath.isfinite(value):
            return value
        singular.append(u)
        return mpmath.mpc(0)
    lo_, hi_ = _mpf(a), _mpf(b)
    if lo_ is None or hi_ is None:
        return None
    lo, hi = lo_, hi_
    points: list[mpmath.mpf] = [lo]
    if lo == mpmath.mpf('-inf') and hi == mpmath.mpf('inf'):
        points = [lo, mpmath.mpf(-10), mpmath.mpf(-1), mpmath.mpf(0), mpmath.mpf(1), mpmath.mpf(10), hi]
    elif hi == mpmath.mpf('inf'):
        points = [lo, lo + 1, lo + 10, lo + 100, hi]
    elif lo == mpmath.mpf('-inf'):
        points = [lo, hi - 100, hi - 10, hi - 1, hi]
    else:
        nodes: list[mpmath.mpf] = []
        for p in inside:
            if p.is_real is True and not p.free_symbols:
                node = _mpf(p)
                if node is not None and min(lo, hi) < node < max(lo, hi) and node not in nodes:
                    nodes.append(node)
        # in the direction of the integration (a reversed range stays so)
        points = [lo] + sorted(nodes, reverse=bool(hi < lo)) + [hi]
    finer: list[mpmath.mpf] = []
    for p, q in zip(points[:-1], points[1:]):
        finer.append(p)
        if p != mpmath.mpf('-inf') and q != mpmath.mpf('inf'):
            finer.append((p + q) / 2)
    finer.append(points[-1])
    try:
        with mpmath.workdps(20):
            first = mpmath.quad(finite, points, method='tanh-sinh', error=True)
            # the same rule on a finer subdivision: an endpoint singularity
            # (a logarithm, a fractional power) which the Gauss-Legendre
            # rule cannot resolve is confirmed this way
            second = mpmath.quad(finite, finer, method='tanh-sinh', error=True)
            third = mpmath.quad(finite, points, method='gauss-legendre', error=True)
    except (ValueError, TypeError, ZeroDivisionError, OverflowError, NameError, AttributeError,
            NotImplementedError, mpmath.libmp.NoConvergence):
        return None
    ends = [p for p in finer if mpmath.isfinite(p)]
    for u in singular:
        if not any(abs(u - p) < mpmath.mpf('1e-10') * (1 + abs(p)) for p in ends) \
                and not (abs(u) > 10**10 and not (mpmath.isfinite(lo) and mpmath.isfinite(hi))):
            return None
    v1, e1 = complex(first[0]), float(abs(first[1]))
    v2, v3 = complex(second[0]), complex(third[0])
    if not (cmath.isfinite(v1) and cmath.isfinite(v2)):
        return None
    scale = 1 + max(abs(v1), abs(v2))
    if abs(v1 - v2) > 1e-8 * scale or e1 > 1e-6 * scale:
        return _oscillatory_quadrature(f, x, lo, hi)
    if abs(v1 - v3) > 1e-6 * scale and abs(v2 - v3) > 1e-6 * scale and e1 > 1e-12 * scale:
        return _oscillatory_quadrature(f, x, lo, hi)
    return v1


def _unit_roots(p: Poly) -> list[Expr]:
    """The real roots of a polynomial over ``QQ`` of degree one to four
    inside ``(0, 1)``: exact for degree one and two, the algebraic numbers
    of the native root isolation otherwise."""
    found: list[Expr] = []
    if p.degree() == 1:
        candidates = roots_linear(p)
    elif p.degree() == 2:
        candidates = roots_quadratic(p)
    else:
        coefficients: list[Fraction] = []
        for c in p.all_coeffs():
            if not isinstance(c, Rational):
                return []
            coefficients.append(Fraction(int(c.p), int(c.q)))
        candidates = [real_root(r, 't') for r in real_roots_of(coefficients)]
    for root in candidates:
        if root.is_extended_real is not True:
            continue
        value = _mpf(root)
        if value is not None and 0 < value < 1 and root not in found:
            found.append(root)
    return found


def _axis_crossings(g: Expr, t: Symbol) -> list[Expr]:
    """The points of ``(0, 1)`` where the argument of a function in
    ``g``, or the base of a fractional power, polynomial in the real ``t``,
    crosses the real or the imaginary axis: the branch points and the
    crossings of the branch cuts on a segment of the complex plane, where
    the integrand jumps, for the subdivision of its quadrature (the bug:
    ``2**log(z)`` from ``-I`` to ``I`` passes through the branch point
    ``0`` at the midpoint, a node of the rules, and its quadrature was
    ``nan``, so that nothing was checked).

    >>> from sympy_extras.core import Dummy, I, Integer
    >>> from sympy_extras.functions import log
    >>> from sympy_extras.integrals.definite import _axis_crossings
    >>> t = Dummy('t', real=True)
    >>> _axis_crossings(2*I*Integer(2)**log(2*I*t - I), t)
    [1/2]
    """
    found: list[Expr] = []
    for node in g.atoms(Application, Pow):
        arguments: list[Expr]
        if isinstance(node, Pow):
            if node.exp.is_integer:
                continue
            arguments = [node.base]
        else:
            arguments = [as_expr(u) for u in node.args if isinstance(u, Expr)]
        for u in arguments:
            if not u.has(t) or not u.is_polynomial(t):
                continue
            for part in expand(u).as_real_imag():
                try:
                    p = Poly(part, t)
                except (BasePolynomialError, ValueError, TypeError):
                    continue
                if p.is_zero or p.degree() < 1 or p.degree() > 4 or p.domain not in ('QQ', 'ZZ'):
                    continue
                for root in _unit_roots(p):
                    if root not in found:
                        found.append(root)
    return found


def _oscillatory_quadrature(f: Expr, x: Symbol, lo: mpmath.mpf, hi: mpmath.mpf) -> Optional[complex]:
    """``Integral(f, (x, lo, hi))`` over a half-line by mpmath's
    ``quadosc`` when ``f`` oscillates (a sine or cosine of ``k*x``): the
    integral summed period by period with Richardson extrapolation,
    trusted when two choices of the period agree (``sin(x)/x`` over
    ``(2, oo)``, where the plain rules do not converge)."""
    if lo == mpmath.mpf('-inf') and hi == mpmath.mpf('inf'):
        left = _oscillatory_quadrature(_subs(f, x, -x), x, mpmath.mpf(0), hi)
        right = _oscillatory_quadrature(f, x, mpmath.mpf(0), hi)
        return None if left is None or right is None else left + right
    if lo == mpmath.mpf('-inf'):
        return _oscillatory_quadrature(_subs(f, x, -x), x, -hi, mpmath.mpf('inf'))
    if hi != mpmath.mpf('inf'):
        return None
    frequencies: list[Rational] = []
    for node in f.atoms(Sin, Cos):
        argument = node.args[0]
        if not isinstance(argument, Expr):
            return None
        found = monomial(argument, x)
        if found is None or found[1] != 1 or not isinstance(found[0], Rational):
            return None
        frequencies.append(found[0])
    if not frequencies:
        return None
    numerator, denominator = 0, 1
    for k in frequencies:
        numerator = gcd(numerator, abs(int(k.p)))
        denominator = lcm(denominator, int(k.q))
    try:
        with mpmath.workdps(20):
            g = _numeric_function(f, x)
            period = 2 * mpmath.pi * denominator / numerator
            first = complex(mpmath.quadosc(lambda u: g(mpmath.mpc(u)), [lo, hi], period=period))
            second = complex(mpmath.quadosc(lambda u: g(mpmath.mpc(u)), [lo, hi], period=2 * period))
    except (ValueError, TypeError, ZeroDivisionError, OverflowError, NameError, AttributeError,
            NotImplementedError, mpmath.libmp.NoConvergence):
        return None
    if abs(first - second) > 1e-8 * (1 + abs(first)):
        return None
    return first


def _quadrature_converges(f: Expr, x: Symbol, a: Expr, b: Expr, assumptions: Assumptions = None) -> bool:
    """Whether the quadrature of ``Integral(f, (x, a, b))`` gives a
    number of a plausible size at some sample of the parameters: the
    check of a claimed divergence (``1/x`` over ``(0, 1)`` gives none;
    ``exp(x)`` over ``(0, oo)`` gives ``inf`` and ``x**2*log(x + 1)`` a
    number of the order of ``1e74``, which are no convergence)."""
    if f.has(DiracDelta):
        return False
    parameters = _sorted_symbols((f.free_symbols | a.free_symbols | b.free_symbols) - {x})
    rng = random.Random(str((f, a, b)))
    for _ in range(2 if parameters else 1):
        values = _sample_values(parameters, assumptions, rng)
        if values is None:
            return False
        g, lower, upper = _xr(f, values), _xr(a, values), _xr(b, values)
        if _grows_like_a_pole(g, x, lower, '+') or _grows_like_a_pole(g, x, upper, '-'):
            # a logarithmic divergence is slow: the quadrature of
            # -sin(x)**2*cot(x - 1) over (0, 1) gives a moderate number
            continue
        found = _quadrature(g, x, lower, upper)
        if found is not None and abs(found) < 1e8:
            return True
    return False


def _grows_like_a_pole(g: Expr, x: Symbol, point: Expr, side: str) -> bool:
    """Whether ``|g(point + h)| * h`` does not shrink with ``h`` on the
    given side of a finite ``point`` (``g`` free of parameters), as it
    does for an integrable singularity ``h**(-p)`` with ``p < 1``: the
    integrand is then not integrable there, whatever a quadrature says."""
    if point in (oo, -oo) or not point.is_number:
        return False
    step = 1 if side == '+' else -1
    sizes: list[float] = []
    for k in (3, 5, 7):
        h = _rational(1, 10**k)
        try:
            with mpmath.workdps(20):
                value = numeric_value(_subs(g, x, point + step * h), mpmath.mp.prec)
        except (TypeError, ValueError, ZeroDivisionError, OverflowError):
            return False
        if value is None or mpmath.im(value) != 0 or not mpmath.isfinite(value):
            return False
        try:
            sizes.append(abs(complex(value)) * float(Fraction(1, 10**k)))
        except (TypeError, ValueError, OverflowError):
            return False
    return sizes[0] > 0 and sizes[1] >= 0.9 * sizes[0] and sizes[2] >= 0.9 * sizes[1]


def verify_numerically(value: Expr, f: Expr, x: Symbol, a: Expr, b: Expr,
                       assumptions: Assumptions = None, samples: int = 2,
                       inside: Sequence[Expr] = ()) -> Optional[bool]:
    """Whether ``value`` agrees with a numerical quadrature of
    ``Integral(f, (x, a, b))`` at ``samples`` random values of the
    parameters satisfying the assumptions: ``False`` when they disagree
    where the quadrature is trusted, ``True`` when they agree there,
    ``None`` when nothing could be checked. The points ``inside`` the
    range (singularities, kinks) subdivide it for the quadrature.

    >>> from sympy_extras.core import symbols, oo, pi, Integer
    >>> from sympy_extras.functions import exp
    >>> from sympy_extras.integrals.definite import verify_numerically
    >>> x, a = symbols('x a', positive=True)
    >>> verify_numerically(1/a, exp(-a*x), x, Integer(0), oo), verify_numerically(2/a, exp(-a*x), x, Integer(0), oo)
    (True, False)
    """
    if f.has(DiracDelta):
        return None
    parameters = _sorted_symbols((f.free_symbols | a.free_symbols | b.free_symbols) - {x})
    rng = random.Random(str((f, a, b)))
    verdict: Optional[bool] = None
    for _ in range(samples if parameters else 1):
        values = _sample_values(parameters, assumptions, rng)
        if values is None:
            return None
        expected = _quadrature(_xr(f, values), x, _xr(a, values), _xr(b, values), [_xr(p, values) for p in inside])
        if expected is None:
            continue
        # a value next to a branch cut on a side which a rounding error
        # chooses checks nothing (sympy-extras#65, see reliable_form); nor
        # does one which cannot be computed: mpmath raises
        # ZeroDivisionError on a hypergeometric series at a pole (SymPy's
        # value of Maxima's specint 174 at an integer sample of the order)
        agrees = _agrees(value, expected, values)
        if agrees is None:
            return None
        if not agrees:
            return False
        verdict = True
    # every branch of a Piecewise value at a sample of its own region: the
    # samples above satisfy the assumptions only, and fall in one branch
    # (the bug: SymPy's value of x*y**x over (0, 2), whose branch for
    # y <= 0 is 2, was accepted from samples y > 0, and the value of
    # y**x over (0, 2) differentiated under the integral sign came out
    # 2*log(y) for y <= 0)
    # (a sample on the boundary of the region, where the branch may not
    # be evaluated, gives way to another)
    for region in _branch_regions(value, parameters):
        for _ in range(_BRANCH_SAMPLES):
            values = _sample_values(parameters, _with(assumptions, [region]), rng)
            if values is None:
                break
            expected = _quadrature(_xr(f, values), x, _xr(a, values), _xr(b, values),
                                   [_xr(p, values) for p in inside])
            agrees = None if expected is None else _agrees(value, expected, values)
            if agrees is False:
                return False
            if agrees is True:
                break
    # the isolated values of the parameters, Eq(y, 1): no sample of a
    # region falls on them, and the value there is checked with the point
    # substituted into the value and the integral (the cases of isolated
    # values the package adds, and those of SymPy's answers)
    for point in equality_points(value, parameters)[:_BRANCHES_CHECKED]:
        at = point_assumptions(point, assumptions)
        if at is None or _xr(f, point).has(nan, zoo):
            continue
        try:
            special = _xr(value, point)
        except TypeError:
            continue
        agrees_there = verify_numerically(special, _xr(f, point), x, _xr(a, point), _xr(b, point), at, 1,
                                          [_xr(p, point) for p in inside])
        if agrees_there is False:
            return False
    return verdict


#: the branches of a ``Piecewise`` value checked at a sample of their own
#: region by :func:`verify_numerically`
_BRANCHES_CHECKED: Final = 4

#: the samples tried in the region of a branch until one checks it
_BRANCH_SAMPLES: Final = 3


def _sample_values(symbols: Sequence[Symbol], assumptions: Assumptions,
                   rng: random.Random) -> Optional[dict[Symbol, Expr]]:
    """:func:`~sympy_extras.integrals.conditions.sample_values` with the
    roots of linear and quadratic polynomials written in radicals (SymPy's
    ``CRootOf`` gives them so): ``IndexedRoot(a**2 - 3, a, 0)`` is
    ``-sqrt(3)``, at which the value ``sqrt(3 - a**2)`` is a zero the
    reliable values prove (the bug: the area of a parameter cell bounded
    by a root was left unevaluated, its check giving nothing)."""
    values = sample_values(symbols, assumptions, rng)
    if values is None:
        return None
    return {s: _radical_sample(v) for s, v in values.items()}


def _radical_sample(v: Expr) -> Expr:
    if not isinstance(v, IndexedRoot):
        return v
    coefficients = _coefficients(v.expr, v.variable)
    if coefficients is None or len(coefficients) > 3:
        return v
    found = radical_form(v)
    return v if found is None or found.has(I) else found


def _agrees(value: Expr, expected: complex, values: dict[Symbol, Expr]) -> Optional[bool]:
    """Whether ``value`` at the sample ``values`` agrees with the
    quadrature ``expected``; ``None`` when its number cannot be trusted
    (:func:`~sympy_extras._numeric.reliable_value`) or computed."""
    number = reliable_value(value, 20, values)
    if number is None:
        return None
    with mpmath.workdps(20):
        computed = numeric_value(number, mpmath.mp.prec)
    if computed is None:
        return None
    try:
        ours = complex(computed)
    except (TypeError, ValueError, OverflowError):
        return None
    return abs(ours - expected) <= 1e-6 * (1 + abs(expected))


def _branch_regions(value: Expr, parameters: Sequence[Symbol]) -> list[Boolean]:
    """The regions of the parameters in which each branch of a
    ``Piecewise`` value is taken (its condition and the negation of the
    earlier ones), for the first ``_BRANCHES_CHECKED`` branches, those
    whose region depends on the parameters; none for a value without a
    ``Piecewise``.

    >>> from sympy_extras.core import symbols, Integer
    >>> from sympy_extras.core.relational import ne
    >>> from sympy_extras.functions import Piecewise, log
    >>> from sympy_extras.integrals.definite import _branch_regions
    >>> y = symbols('y')
    >>> _branch_regions(Piecewise(((y**2 - 1)/log(y), (y > 0) & ne(y, 1)), (Integer(2), True)), [y])
    [(y > 0) & Ne(y, 1), ~((y > 0) & Ne(y, 1))]
    """
    if not value.has(Piecewise):
        return []
    folded = piecewise_fold(value)
    if not isinstance(folded, Piecewise):
        return []
    regions: list[Boolean] = []
    earlier: list[Boolean] = []
    for _, condition in _pairs(folded)[:_BRANCHES_CHECKED]:
        region = and_(condition, *[not_(c) for c in earlier])
        # (a branch at isolated values, Eq(y, 1), has no region to sample:
        # its points are checked apart)
        if region.free_symbols & set(parameters) and not _isolated(condition):
            regions.append(region)
        earlier.append(condition)
    return regions


def _isolated(condition: Boolean) -> bool:
    """Whether a condition holds at isolated points only: equations
    ``Eq(p, v)``, conjunctions and disjunctions of them."""
    disjuncts = [to_boolean(d) for d in condition.args] if isinstance(condition, Or) else [condition]
    return all(bool(condition_point(d)) for d in disjuncts)


# ---------------------------------------------------------------------------
# The integrator

#: the factored argument of a power or a logarithm: the constant, the
#: factors with their multiplicities, whether a square appeared
_Factored = tuple[Expr, list[tuple[Expr, Rational]], bool]


class _Integrator:
    """The strategies, sharing the assumptions; ``parametric`` allows
    differentiation under the integral sign (off inside that method)."""

    def __init__(self, assumptions: Assumptions, parametric: bool = True,
                 principal_value: bool = False, finite_part: bool = False,
                 regularize: bool = False, cases: bool = True) -> None:
        self.assumptions: Final[Assumptions] = assumptions
        self.parametric: Final[bool] = parametric
        self.principal_value: Final[bool] = principal_value
        self.finite_part: Final[bool] = finite_part
        self.regularize: Final[bool] = regularize
        #: whether a singularity or a kink whose position the assumptions
        #: do not settle splits the parameters into cases (:meth:`_cases`);
        #: off in the integrator of a case, which must not split again
        self.cases: Final[bool] = cases
        #: the factored arguments of the powers and logarithms, by argument
        self._factored: Final[dict[Expr, Optional[_Factored]]] = {}
        #: the zeros of the factors placed against the ranges seen
        self._zero_cache: Final[dict[tuple[Expr, Expr, Expr], Optional[_Placement]]] = {}
        #: the singularities of the integrands placed against the ranges seen
        self._singular_cache: Final[dict[tuple[Expr, Expr, Expr], Optional[_Placement]]] = {}

    def ask(self, query: Boolean) -> Optional[bool]:
        return ask(query, self.assumptions)

    def _factors_of(self, base: Expr, x: Symbol) -> Optional[_Factored]:
        if base not in self._factored:
            self._factored[base] = _factored(base, x)
        return self._factored[base]

    def _nodes(self, f: Expr, x: Symbol, full: bool) -> list[tuple[Expr, Expr, list[tuple[Expr, Rational]]]]:
        """The powers and logarithms of ``f`` to split, with the constant
        and the factors of their arguments; without ``full`` only those
        whose argument has a square (``sqrt((x - 1)**2)``, ``(1 -
        cos(x))**(3/2)``), which are rewritten before the other methods;
        with it every one, for the late splitting."""
        found: list[tuple[Expr, Expr, list[tuple[Expr, Rational]]]] = []
        for node in _signed_nodes(f, x):
            base = node.base if isinstance(node, Pow) else as_expr(node.args[0])
            factored = self._factors_of(base, x)
            if factored is not None and (full or factored[2]):
                found.append((node, factored[0], factored[1]))
        return found

    # -- entry point --------------------------------------------------------

    def integrate(self, f: Expr, x: Symbol, a: Expr, b: Expr, depth: int,
                  fallback: bool = True, mapped: bool = False) -> Optional[ConditionalValue]:
        """``Integral(f, (x, a, b))``; without ``fallback`` SymPy's
        ``integrate`` is not tried (the caller tries it on its own form),
        and inside a ``mapped`` range the slow methods which work on the
        original form (creative telescoping, differentiation under the
        integral sign) are skipped."""
        if depth > _MAX_DEPTH:
            return None
        if a == b:
            return ConditionalValue(S.Zero)
        if self.ask(a > b) is True:
            reversed_ = self.integrate(f, x, b, a, depth, fallback, mapped)
            return None if reversed_ is None else reversed_.scaled(S.NegativeOne)
        if f == 0:
            return ConditionalValue(S.Zero)
        if not f.has(x):
            if a in (-oo, oo) or b in (-oo, oo):
                # a constant over an infinite range: its sign's infinity
                if self.ask(f > 0) is True:
                    return ConditionalValue(oo)
                if self.ask(f < 0) is True:
                    return ConditionalValue(-oo)
                return None
            return ConditionalValue(f * (b - a))
        f = piecewise_fold(f) if f.has(Piecewise) else f
        if f.has(DiracDelta):
            return self._last(f, x, a, b)
        f = _combined_exponentials(f, x)
        f = _sums_to_products(f, x)
        f = self._arctangent_pairs(f, x, a, b)
        # constants out, the common factors of sums pulled first
        # (log(t + 1)/(a**2*t**2 + a**2) is log(t + 1)/(t**2 + 1) over a**2)
        constant, rest = factor_terms(f).as_independent(x, as_Add=False)
        if constant != 1:
            found = self.integrate(rest, x, a, b, depth, fallback, mapped)
            return None if found is None else found.scaled(constant)
        if not mapped:
            # before the powers are split into their factors and the range
            # is mapped, which hide the binomial
            found = self._binomial_differential(f, x, a, b, depth)
            if found is not None:
                return found
        parametric, found = self._split_branches(f, x, a, b, depth)
        if found is not None:
            return found
        if parametric:
            # a kink whose position the assumptions do not settle: the cases
            # of the parameters are the answer, or nothing is (a method on
            # the whole range would be checked at samples of one case)
            return None
        if not mapped:
            # the identities which change the integrand act on the whole
            # range, before it is cut at singularities (Glasser's map has
            # poles inside)
            found = self._transformations(f, x, a, b, depth)
            if found is not None:
                return found
        singular, found = self._split_singularities(f, x, a, b, depth)
        if singular and depth == 0 and (self.finite_part or self.principal_value) \
                and (found is None or found.value.has(oo, -oo, zoo)):
            # a singularity inside the range whose pieces diverge, or
            # could not be integrated: the finite part or the principal
            # value which was asked for, and not the infinite value of the
            # integral (the bug: 1/x**2 over (-1, 1) with finite_part=True
            # came out oo since the divergences are reported)
            regular: Optional[ConditionalValue]
            if self.finite_part:
                from .antiderivative import finite_part_integral
                regular = self._finish(finite_part_integral(f, x, a, b, self.assumptions))
            else:
                from .antiderivative import principal_value_integral
                regular = self._finish(principal_value_integral(f, x, a, b, self.assumptions))
            if regular is not None:
                return regular
        if found is not None:
            return found
        if singular:
            # an antiderivative evaluated at the endpoints would be wrong
            # with a singularity inside the range, so SymPy is not asked
            return None
        allowed = (f.free_symbols | a.free_symbols | b.free_symbols) - {x}
        # a divergence read off the leading term at an end of the range,
        # before the methods spend the budget on an integral which has
        # no value
        found = self._asymptotic_divergence(f, x, a, b)
        if found is not None:
            return found
        strategies: list[Callable[[Expr, Symbol, Expr, Expr, int], Optional[ConditionalValue]]] = [
            self._table, self._table_termwise, self._dirichlet, self._radicals,
            self._quick_antiderivative, self._canonical,
            self._regularized_termwise,
            self._mean_value, self._elliptic, self._trigonometric]
        strategies += [self._logarithmic_substitution, self._parameter_exponentials]
        if not mapped and depth == 0:
            # the substitutions on the integral as given only: inside a
            # mapped range, or on a term of a sum, they would be tried again
            # on the same integral, with the constants of the map (the
            # acoth(sqrt(t + 1) - 1) piece of a sum over (1, 2) lost the
            # budget that way)
            strategies += [self._radical_substitution, self._completed_square, self._tangent_substitution]
        # the residues of a rational function before the range is cut at 0
        # (the quartic of Wester's problem 22 lost its budget on the
        # half-lines); a logarithm with a parameter in the keyhole route
        # spent 15 s where the mapped range had the value
        rational = bool(f.is_rational_function(x))
        strategies += [self._residues] if rational else []
        strategies += [self._mapped, self._inversion]
        strategies += [] if rational else [self._residues]
        strategies += [self._contours, self._algebraic]
        if a == -oo and b == oo and f.has(*_HYPERBOLIC):
            # the rectangular contour gives pi**3/4 for x**2/cosh(x) where
            # the Mellin table gives polylogarithms at +-I
            strategies.remove(self._contours)
            strategies.insert(0, self._contours)
        strategies.append(self._split_powers)
        # the antiderivative, bounded by a quarter of the time limit and
        # checked numerically, before the slow methods: 1/(cosh(n*t)**2 + 1)
        # over (0, 1) has one in a second, and differentiation under the
        # integral sign spent the whole budget first
        strategies.append(self._antiderivative)
        if not mapped:
            # the methods which work on the original form only, and the
            # slow ones: not inside a mapped range
            strategies += [self._holonomic, self._laplace, self._parametric, self._series, self._dfinite]
        strategies.append(self._termwise)
        if depth == 0:
            # the heuristics last, on the integral as given only (a quarter
            # of the limit on every piece of every range ate the budget of
            # log(sin(x)/x) over (0, pi/2) before the splitting of the log)
            strategies.append(self._late_antiderivative)
        for strategy in strategies:
            try:
                found = strategy(f, x, a, b, depth)
            except (ZeroDivisionError, AttributeError, AssertionError, OverflowError):
                # the internals fail on some inputs (a division by zero in
                # mpmath inside an evaluation, an assertion of a solver):
                # the method is skipped
                found = None
            if found is not None:
                if found.value.free_symbols - allowed or found.condition.free_symbols - allowed:
                    # a constant of integration or a dummy leaked from a method
                    continue
                return found
        if fallback:
            return self._last(f, x, a, b, depth)
        return None

    # -- splitting ----------------------------------------------------------

    def _points_in(self, points: Sequence[Expr], a: Expr, b: Expr) -> Optional[list[Expr]]:
        """The points which lie inside ``(a, b)``, sorted; ``None`` when a
        point cannot be placed."""
        placed = self._place(points, a, b)
        if placed is None or placed[1]:
            return None
        return placed[0]

    def _place(self, points: Sequence[Expr], a: Expr, b: Expr) -> Optional[_Placement]:
        """The points which lie strictly inside ``(a, b)``, sorted, and
        those whose position against an endpoint the assumptions do not
        settle (``0`` against ``(a, b)`` for plain ``a`` and ``b``); the
        points known to lie outside are dropped, and so are those known
        not real. ``None`` when two points inside cannot be ordered."""
        inside: list[Expr] = []
        undecided: list[Expr] = []
        for p in points:
            if p.is_real is False or p in inside or p in undecided:
                continue
            if p.is_extended_real is not True and not _real_for_real_parameters(p) \
                    and self.ask(element(p, Reals)) is not True:
                # a point not known real (I*y for a plain y, sqrt(-c) for a
                # plain c): neither placed nor a case of the parameters
                # (the regression: x/(x**2 + y**2) over (a, b) spent the
                # budget on the cases of I*y against the endpoints, and the
                # methods never ran); a plain c is a case, the parameters
                # being taken real
                return None
            below = self.ask(p > a) if a != -oo else True
            above = self.ask(p < b) if b != oo else True
            if below is False or above is False:
                continue
            if below is None or above is None:
                undecided.append(p)
            else:
                inside.append(p)
        for i in range(len(inside)):
            for j in range(i + 1, len(inside)):
                if self.ask(inside[i] < inside[j]) is None:
                    return None
        return sorted(inside, key=lambda p: [0 if self.ask(p < q) else 1 for q in inside]), undecided

    def _zeros(self, u: Expr, x: Symbol, a: Expr, b: Expr) -> Optional[list[Expr]]:
        """The zeros of ``u`` in ``(a, b)``; ``None`` when they cannot be
        found or one cannot be placed."""
        placed = self._zero_placement(u, x, a, b)
        if placed is None or placed[1]:
            return None
        return placed[0]

    def _zero_placement(self, u: Expr, x: Symbol, a: Expr, b: Expr) -> Optional[_Placement]:
        """The zeros of ``u`` placed against ``(a, b)`` (:meth:`_place`)."""
        key = (u, a, b)
        if key not in self._zero_cache:
            self._zero_cache[key] = self._solve_zeros(u, x, a, b)
        return self._zero_cache[key]

    def _solve_zeros(self, u: Expr, x: Symbol, a: Expr, b: Expr) -> Optional[_Placement]:
        zeros = _trigonometric_zeros(u, x, a, b)
        if zeros is not None:
            return zeros, []
        bounds: list[Boolean] = []
        if a != -oo:
            bounds.append(x > a)
        if b != oo:
            bounds.append(x < b)
        # the assumptions on the parameters of u only: an unrelated one
        # (a > 0 for the zeros of sin(t/2)) turns the solver's answer from
        # the empty set into a ConditionSet on the bounds
        relevant = [item for item in _items(self.assumptions) if item.free_symbols & u.free_symbols]
        assumptions = _with(relevant, bounds)
        try:
            found = attempt(lambda: solve(u, x, assumptions, domain=Reals), settings.timeout, native=True)
        except (ValueError, TypeError):
            return None
        if found is None:
            return None
        candidates = _candidate_points(found, x, a, b)
        if candidates is None:
            return None
        return self._place(candidates, a, b)

    def _singular_placement(self, f: Expr, x: Symbol, a: Expr, b: Expr) -> Optional[_Placement]:
        """The singularities of ``f`` placed against ``(a, b)``
        (:meth:`_place`); ``None`` when they cannot be found or listed."""
        key = (f, a, b)
        if key not in self._singular_cache:
            found = attempt(lambda: _singular_points(f, x, a, b), settings.timeout, native=True)
            self._singular_cache[key] = None if found is None else self._place(found, a, b)
        return self._singular_cache[key]

    def _kink_placement(self, f: Expr, x: Symbol, a: Expr, b: Expr) -> Optional[_Placement]:
        """The points of ``(a, b)`` where a branch of ``Abs``, ``sign``,
        ``Heaviside``, ``Max``, ``Min`` or ``Piecewise`` in ``f`` changes,
        placed (:meth:`_place`); ``None`` when a zero cannot be found."""
        arguments: list[Expr] = []
        for node in f.atoms(Abs, Sign, Heaviside):
            arguments.append(as_expr(node.args[0]))
        for node in f.atoms(Max, Min):
            if len(node.args) == 2:
                arguments.append(as_expr(node.args[0]) - as_expr(node.args[1]))
        for node in f.atoms(Piecewise):
            if not isinstance(node, Piecewise):
                continue
            for _, condition in _pairs(node):
                for atom in condition.atoms(Relational):
                    if isinstance(atom, Relational):
                        arguments.append(atom.lhs - atom.rhs)
        inside: list[Expr] = []
        undecided: list[Expr] = []
        for u in arguments:
            if not u.has(x):
                continue
            placed = self._zero_placement(u, x, a, b)
            if placed is None:
                return None
            inside.extend(placed[0])
            undecided.extend(p for p in placed[1] if p not in undecided)
        return inside, undecided

    def _parametric_points(self, f: Expr, x: Symbol, a: Expr, b: Expr) -> bool:
        """Whether ``f`` has a kink or, without kinks, a singularity whose
        position against ``(a, b)`` the assumptions do not settle: the
        integral is then a question of cases of the parameters, which
        :meth:`_cases` answers, and no method on the whole range is
        trusted."""
        if not self.cases or not (f.free_symbols | a.free_symbols | b.free_symbols) - {x}:
            return False
        kinks = self._kink_placement(f, x, a, b)
        if kinks is not None and kinks[1]:
            return True
        if f.atoms(*_KINKS):
            return False
        singular = self._singular_placement(f, x, a, b)
        return singular is not None and bool(singular[1])

    def _sample(self, a: Expr, b: Expr) -> Expr:
        if a == -oo and b == oo:
            return S.Zero
        if a == -oo:
            return b - 1
        if b == oo:
            return a + 1
        return (a + b) / 2

    def _branch(self, f: Expr, x: Symbol, a: Expr, b: Expr, unplaced: frozenset[Expr] = frozenset(),
                full: bool = False) -> Optional[Expr]:
        """``f`` with ``Abs``, ``sign``, ``Heaviside``, ``Max``, ``Min`` and
        ``Piecewise`` of arguments depending on ``x`` replaced by their
        branch on ``(a, b)``, decided at a sample point, and the powers
        and logarithms of products split over the factors whose sign on
        ``(a, b)`` is known (:meth:`_signed`); ``None`` when a sign of the
        former is undecided. The factors in ``unplaced`` are those whose
        zeros could not be found, so their sample sign says nothing."""
        sample = self._sample(a, b)
        result = f

        def sign_of(u: Expr) -> Optional[int]:
            value = _subs(u, x, sample)
            if self.ask(value > 0) is True:
                return 1
            if self.ask(value < 0) is True:
                return -1
            return None

        for node, constant, factors in self._nodes(result, x, full):
            rewritten = self._signed(node, constant, factors, sign_of, unplaced, full)
            if rewritten is not None:
                result = _replaced(result, node, rewritten)
        for kink in sorted(result.atoms(Abs, Sign, Heaviside), key=str):
            u = as_expr(kink.args[0])
            if not u.has(x):
                continue
            s = sign_of(u)
            if s is None:
                return None
            if isinstance(kink, Abs):
                result = _replaced(result, as_expr(kink), s * u)
            elif isinstance(kink, Sign):
                result = _replaced(result, as_expr(kink), Integer(s))
            else:
                result = _replaced(result, as_expr(kink), S.One if s > 0 else S.Zero)
        for extremum in sorted(result.atoms(Max, Min), key=str):
            if not extremum.has(x) or len(extremum.args) != 2:
                continue
            u, v = as_expr(extremum.args[0]), as_expr(extremum.args[1])
            s = sign_of(u - v)
            if s is None:
                return None
            larger, smaller = (u, v) if s > 0 else (v, u)
            result = _replaced(result, as_expr(extremum), larger if isinstance(extremum, Max) else smaller)
        for branches in result.atoms(Piecewise):
            if not isinstance(branches, Piecewise) or not branches.has(x):
                continue
            chosen: Optional[Expr] = None
            for value, cond in _pairs(branches):
                if cond is true:
                    chosen = value
                    break
                holds = self.ask(to_boolean(cond.subs(x, sample)))
                if holds is None:
                    return None
                if holds:
                    chosen = value
                    break
            if chosen is None:
                return None
            result = _replaced(result, branches, chosen)
        return result

    def _signed(self, node: Expr, constant: Expr, factors: list[tuple[Expr, Rational]],
                sign_of: Callable[[Expr], Optional[int]], unplaced: frozenset[Expr],
                full: bool) -> Optional[Expr]:
        """A power ``P**r`` or a logarithm ``log(P)`` with ``P`` a product
        of factors of known sign on the piece split over the factors, or
        ``None`` when nothing is known.

        ``(u*v)**r = u**r * v**r`` and ``log(u*v) = log(u) + log(v)`` hold
        for ``u > 0`` and any ``v`` (the argument of the product is the
        argument of ``v``), so the factors known positive on the piece,
        with their multiplicities, and the positive constant come out one
        by one, a factor known negative as ``(-g)**e`` with the sign
        ``(-1)**e`` kept with the remainder, and the factors whose sign is
        unknown stay together in the remainder, which is ``(-1)**r`` at
        most when every sign is known: ``sqrt(c*u**2)`` is ``sqrt(c)*(s*u)``
        with the sign ``s`` of ``u`` on the piece, ``sqrt(x - 2 + 1/x)`` on
        ``(0, 1)`` is ``(1 - x)/sqrt(x)``, ``log(x**2)`` is ``2*log(-x)`` on
        the negative axis. Without ``full`` only the factors of even
        multiplicity and the positive constant come out (the square roots
        of squares), the other factors staying together for the methods
        which read a radicand as a whole."""
        exponent = node.exp if isinstance(node, Pow) else S.One
        positive: list[tuple[Expr, Expr]] = []
        remainder: list[Expr] = []
        total_sign: Expr = S.One
        for g, e in factors:
            if g in unplaced or (not full and not _even(e)):
                remainder.append(g**e)
                continue
            s = sign_of(g)
            if s is None:
                remainder.append(g**e)
                continue
            positive.append((s * g, e))
            total_sign = total_sign * Integer(s)**e
        if constant != 1:
            if self.ask(constant > 0) is True:
                positive.append((constant, S.One))
            elif self.ask(constant < 0) is True:
                positive.append((-constant, S.One))
                total_sign = -total_sign
            else:
                remainder.append(constant)
        if not positive:
            return None
        rest = total_sign * mul(*remainder)
        rewritten: Expr
        if isinstance(node, Pow):
            rewritten = mul(*[g**(e * exponent) for g, e in positive]) * (rest**exponent if rest != 1 else S.One)
        else:
            rewritten = add(*[e * log(g) for g, e in positive]) + (log(rest) if rest != 1 else S.Zero)
        return None if rewritten == node else rewritten

    def _breakpoints(self, f: Expr, x: Symbol, a: Expr, b: Expr,
                     full: bool = False) -> Optional[tuple[list[Expr], frozenset[Expr], list[Expr]]]:
        """The points of ``(a, b)`` where a branch of ``f`` changes: the
        zeros of the arguments of ``Abs`` and the like, and of the factors
        of the powers and logarithms to split (:meth:`_signed`); with the
        factors whose zeros could not be found, which are not split, and
        the zeros of the former whose position against an endpoint the
        assumptions do not settle. ``None`` when a zero of the former
        cannot be found, or two of them cannot be ordered."""
        kinks = self._kink_placement(f, x, a, b)
        if kinks is None:
            return None
        points: list[Expr] = list(kinks[0])
        unplaced: set[Expr] = set()
        for _, _, factors in self._nodes(f, x, full):
            for g, e in factors:
                if not full and not _even(e):
                    continue
                zeros = self._zeros(g, x, a, b)
                if zeros is None:
                    unplaced.add(g)
                else:
                    points.extend(zeros)
        placed = self._points_in(_distinct(points), a, b) if points else []
        if placed is None:
            return None
        return placed, frozenset(unplaced), list(kinks[1])

    def _split_at(self, f: Expr, x: Symbol, a: Expr, b: Expr, points: Sequence[Expr], depth: int,
                  branch: bool, unplaced: frozenset[Expr] = frozenset(),
                  full: bool = False) -> Optional[ConditionalValue]:
        bounds = [a] + list(points) + [b]
        total = ConditionalValue(S.Zero)
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            piece = self._branch(f, x, lo, hi, unplaced, full) if branch else f
            if piece is None:
                return None
            # the late splitting is a rewriting: its pieces skip the slow
            # methods and SymPy, which the original form gets afterwards
            found = self.integrate(piece, x, lo, hi, depth + 1, not full, full)
            if found is None or found.value.has(zoo, nan):
                return None
            # a piece divergent to a signed infinity makes the whole so,
            # unless another cancels it (oo - oo below)
            total = total.add(found)
        if total.value.has(zoo, nan):
            return None
        return total

    def _split_signs(self, f: Expr, x: Symbol, a: Expr, b: Expr, depth: int,
                     full: bool) -> tuple[bool, Optional[ConditionalValue]]:
        """The range cut where a branch of ``f`` changes, each piece
        integrated in its branch; ``None`` when there is nothing to cut or
        to rewrite. A kink whose position against an endpoint the
        assumptions do not settle makes the integral a question of cases
        of the parameters (:meth:`_cases`), which is reported as the first
        item: no method on the whole range answers it."""
        if not f.atoms(*_KINKS) and not self._nodes(f, x, full):
            return False, None
        found = self._breakpoints(f, x, a, b, full)
        if found is None:
            return False, None
        points, unplaced, undecided = found
        if undecided:
            if not self.cases or full:
                return False, None
            return True, self._cases(f, x, a, b, [(p, False) for p in undecided], depth)
        if not points:
            piece = self._branch(f, x, a, b, unplaced, full)
            if piece is None or piece == f:
                return False, None
            return False, self.integrate(piece, x, a, b, depth + 1, not full, full)
        return False, self._split_at(f, x, a, b, points, depth, True, unplaced, full)

    def _signed_form(self, f: Expr, x: Symbol, a: Expr, b: Expr) -> Expr:
        """``f`` with its powers and logarithms split by the signs of
        their factors on ``(a, b)`` when no factor changes sign inside,
        else ``f`` itself: ``(-cos(t))**(2/3)`` on ``(0, pi/2)`` is
        ``(-1)**(2/3)*cos(t)**(2/3)``, which the substitution of the Beta
        integral reads."""
        if not self._nodes(f, x, True):
            return f
        found = self._breakpoints(f, x, a, b, True)
        if found is None or found[0] or found[2]:
            return f
        piece = self._branch(f, x, a, b, found[1], True)
        return f if piece is None else piece

    def _split_branches(self, f: Expr, x: Symbol, a: Expr, b: Expr,
                        depth: int) -> tuple[bool, Optional[ConditionalValue]]:
        """The branches of ``Abs`` and the like, and the square roots of
        squares, before the methods; whether the kinks made the integral
        a question of cases of the parameters, and the value."""
        return self._split_signs(f, x, a, b, depth, False)

    def _split_powers(self, f: Expr, x: Symbol, a: Expr, b: Expr, depth: int) -> Optional[ConditionalValue]:
        """The powers and logarithms of products split over their factors
        of known sign (``sqrt(x*(4 - x))`` as ``sqrt(x)*sqrt(4 - x)`` on
        ``(0, 4)``, ``log(sin(x)/x)`` as ``log(sin(x)) - log(x)`` on
        ``(0, pi/2)``), after the methods which read them whole."""
        return self._split_signs(f, x, a, b, depth, True)[1]

    def _split_singularities(self, f: Expr, x: Symbol, a: Expr, b: Expr,
                             depth: int) -> tuple[bool, Optional[ConditionalValue]]:
        """Whether singularities were found inside the range, or their
        position against an endpoint is a question of cases of the
        parameters (:meth:`_cases`), and the sum of the integrals over the
        pieces between them, or the cases."""
        if f.atoms(*_KINKS):
            return False, None
        placed = self._singular_placement(f, x, a, b)
        if placed is None:
            return False, None
        points, undecided = placed
        if undecided:
            if not self.cases:
                return False, None
            return True, self._cases(f, x, a, b, [(p, True) for p in undecided], depth)
        if not points:
            return False, None
        return True, self._split_at(f, x, a, b, points, depth, False)

    # -- the cases of the parameters ----------------------------------------

    def _cases(self, f: Expr, x: Symbol, a: Expr, b: Expr, points: Sequence[tuple[Expr, bool]],
               depth: int) -> Optional[ConditionalValue]:
        """The integral for every position of the ``points`` (each with
        whether it is a singularity of ``f`` or a kink) against the
        endpoints which the assumptions do not settle: below both, above
        both, or between them, in either order of the endpoints; the
        combinations the assumptions refute are dropped. Each case is
        integrated under its condition added to the assumptions
        (:meth:`_case_value`), checked numerically at samples of the
        parameters within the case, and the cases with the same value, or
        whose value checks under the other's condition, are joined. The
        value is a ``Piecewise`` of the cases (or the value of a single
        one) under the disjunction of their conditions; ``None`` when no
        case has a finite value (the divergent cases, and those no method
        answers, are for the other branch of the ``Piecewise`` of the
        caller)."""
        from .antiderivative import antiderivative
        if len(points) > _MAX_CASE_POINTS:
            return None
        alternatives = [_positions(p, a, b, singular) for p, singular in points]
        cases: list[tuple[Boolean, list[Expr], bool]] = []
        for combination in product(*alternatives):
            conjuncts: list[Boolean] = []
            for c, _ in combination:
                if isinstance(c, And):
                    conjuncts.extend(to_boolean(part) for part in c.args)
                else:
                    conjuncts.append(c)
            condition = self._reduced(conjuncts)
            if condition is None:
                continue
            inside = [p for (p, _), (_, position) in zip(points, combination) if position != 0]
            cases.append((condition, inside, any(position < 0 for _, position in combination)))
        if not cases:
            return None
        # the antiderivative once for every case (its limits differ)
        known = antiderivative(f, x, self.assumptions)
        budget = None if settings.timeout is None else settings.timeout / 4
        results: list[tuple[Boolean, Expr]] = []
        for condition, inside, reversed_ in cases:
            candidates = [value for _, value in sorted(results, key=lambda item: _ops(item[1]))]
            found = attempt(lambda: self._case_value(f, x, a, b, inside, reversed_,
                                                     _with(self.assumptions, [condition]), depth, known, candidates),
                            budget, native=True)
            if found is None or found.value.has(oo, -oo, zoo, nan):
                continue
            case = _canonical_relations(and_(condition, found.condition))
            if found.condition is not true and isinstance(case, And):
                # without the conjuncts the others imply ((c <= 0) & (c < 0)
                # & (1/c < 0) for log(Abs(x - c)) over (0, 1))
                reduced = self._reduced([to_boolean(part) for part in
                                         sorted(case.args, key=_ops, reverse=True)])
                if reduced is None:
                    continue
                case = _canonical_relations(reduced)
            case = _strict_where_undefined(case, found.value)
            if settings.numerical_checks and verify_numerically(
                    found.value, f, x, a, b, _with(self.assumptions, [case]), inside=inside) is False:
                continue
            results.append((case, found.value))
        joined = self._joined(results, f, x, a, b)
        if not joined:
            return None
        if len(joined) == 1:
            return ConditionalValue(joined[0][1], joined[0][0])
        # unevaluated: piecewise simplifies a condition given the negation
        # of the earlier ones, and the conditions of the cases must stay
        # those of their disjunction (ConditionalValue.cases reads them back)
        return ConditionalValue(Piecewise(*[(v, c) for c, v in joined]), or_(*[c for c, _ in joined]))

    def _reduced(self, conjuncts: Sequence[Boolean]) -> Optional[Boolean]:
        """The conjunction of the conditions without those which follow
        from the others and the assumptions (``c > 0`` next to ``c > 1``),
        or ``None`` when the conjunction is unsatisfiable under the
        assumptions (``c < 0`` next to ``c > 1``), both by the SAT solver
        with the theory of the reals (:func:`~sympy_extras.assumptions.sat.satisfiable`,
        which :func:`ask` on a conjunction without assumptions leaves
        undecided, and which answers in half its time)."""
        kept = list(conjuncts)
        # a conjunct refuted on its own: the SAT solver leaves the
        # relations of a radical undecided (the bug: 1/(x**2 - c) over (0,
        # 1) with c > 0 had a case under sqrt(c) < 0, which never holds,
        # with no sample to check its value at), where ask decides them
        if any(self.ask(part) is False for part in kept):
            return None
        if satisfiable(and_(*kept), self.assumptions) is False:
            return None
        for part in list(kept):
            others = [q for q in kept if q != part]
            related: set[Symbol] = set()
            for q in others + _items(self.assumptions):
                related |= q.free_symbols
            if not part.free_symbols & related:
                # nothing relates the part to the others: not implied
                continue
            if satisfiable(not_(part), _with(self.assumptions, others)) is False:
                kept.remove(part)
        return and_(*kept)

    def _case_value(self, f: Expr, x: Symbol, a: Expr, b: Expr, inside: Sequence[Expr], reversed_: bool,
                    assumptions: Assumptions, depth: int, known: Optional[Expr],
                    candidates: Sequence[Expr]) -> Optional[ConditionalValue]:
        """The integral in one case, under its assumptions. A point
        ``inside`` the range (the endpoints in their order of the case)
        at which the ``known`` antiderivative has an infinite one-sided
        limit makes the integral divergent: ``None`` at once. Otherwise
        the values of the earlier cases (``candidates``, the simplest
        first) are tried, each accepted when it agrees with the quadrature
        at samples of the parameters within the case (``-log(a) + log(b)``
        for ``1/x`` holds for negative ``a`` and ``b`` too); then SymPy's
        ``integrate``, checked the same way; then the antiderivative with
        limits on each piece between the points inside; then every method
        on the whole range."""
        from .antiderivative import one_sided_limit, real_logarithms
        if inside and f.is_rational_function(x):
            # a pole of a rational integrand inside the range: divergent,
            # decided at once (the bug: 1/((x - a)*(x - b)) over (0, 1)
            # with a < b spent the time of the finite cases on the limits
            # of an antiderivative whose logarithms had unsimplified
            # arguments, and came back unevaluated)
            denominator = cancel(f).as_numer_denom()[1]
            if any(cancel(_subs(denominator, x, s)) == 0 for s in inside):
                return None
        integrator = _Integrator(assumptions, self.parametric, self.principal_value, self.finite_part,
                                 self.regularize, cases=False)
        lower, upper = (b, a) if reversed_ else (a, b)
        ordered = integrator._place(list(inside), lower, upper)
        if ordered is not None and ordered[0] and known is not None:
            real = real_logarithms(known, x)
            for s in ordered[0]:
                for side in ('-', '+'):
                    if one_sided_limit(real, x, s, side, assumptions) in (oo, -oo):
                        return None
        for candidate in candidates:
            if settings.numerical_checks \
                    and verify_numerically(candidate, f, x, a, b, assumptions, samples=3, inside=inside) is True:
                return ConditionalValue(candidate)
        if not inside and f.atoms(*_KINKS):
            # no kink inside the range in this case: the integrand on its
            # branch there (log(Abs(x - c)) is log(x - c) over (0, 1) for
            # c <= 0, which SymPy integrates at once, where it spent the
            # budget of the case on the absolute value)
            kinks = integrator._kink_placement(f, x, lower, upper)
            if kinks is not None and not kinks[0] and not kinks[1]:
                branch = integrator._branch(f, x, lower, upper)
                if branch is not None:
                    f = branch
        found = integrator._last(f, x, a, b, depth)
        if found is not None and not found.value.has(Piecewise):
            return found
        if ordered is not None and not ordered[1] and known is not None:
            # the antiderivative on the pieces, when there is one: sought
            # again on every piece of every case, it took the budget of the
            # cases (the bug: log(Abs(x - c)) over (0, 1) came back with
            # the case c >= 1 only)
            bounds = [lower] + ordered[0] + [upper]
            total: Optional[ConditionalValue] = ConditionalValue(S.Zero)
            for lo, hi in zip(bounds[:-1], bounds[1:]):
                piece = integrator._antiderivative(f, x, lo, hi, depth, known=known)
                if piece is not None and piece.value.has(oo, -oo, zoo, nan):
                    return None
                total = None if piece is None or total is None else total.add(piece)
            if total is not None:
                return total.scaled(S.NegativeOne) if reversed_ else total
        return integrator.integrate(f, x, a, b, depth)

    def _joined(self, results: Sequence[tuple[Boolean, Expr]], f: Expr, x: Symbol, a: Expr,
                b: Expr) -> list[tuple[Boolean, Expr]]:
        """The cases with equal values joined under the disjunction of
        their conditions, the simplest value first: a case joins an
        earlier one when the values are the same expression, or when the
        earlier value agrees with the quadrature at samples of the
        parameters within the case (``-log(a) + log(b)`` for ``1/x`` over
        ``(a, b)`` holds for negative ``a`` and ``b`` too, where the case
        found ``-log(-a) + log(-b)``)."""
        joined: list[tuple[Boolean, Expr]] = []
        for condition, value in sorted(results, key=lambda item: (_ops(item[1]), str(item[1]))):
            for i, (earlier, candidate) in enumerate(joined):
                if candidate == value or (settings.numerical_checks and verify_numerically(
                        candidate, f, x, a, b, _with(self.assumptions, [condition]), samples=3) is True):
                    joined[i] = (or_(earlier, condition), candidate)
                    break
            else:
                joined.append((condition, value))
        return joined

    # -- the methods on a plain piece ---------------------------------------

    def _finish(self, found: Optional[ConditionalValue]) -> Optional[ConditionalValue]:
        if found is None or found.value.has(zoo, nan):
            return None
        condition = decide(found.condition, self.assumptions)
        if condition is None:
            return None
        return ConditionalValue(found.value, condition)

    def _mellin_on(self, g: Expr, t: Symbol, cutoff: Optional[str]) -> Optional[ConditionalValue]:
        """The Marichev–Adamchik method on ``g``, tried as it is and
        expanded."""
        for candidate in _forms(g, t, self.assumptions):
            found = mellin_integrate(candidate, t, self.assumptions, cutoff, self.regularize)
            if found is not None:
                return found
        return None

    def _canonical(self, f: Expr, x: Symbol, a: Expr, b: Expr, depth: int) -> Optional[ConditionalValue]:
        """The Marichev–Adamchik method on the ranges it works on."""
        t = _fresh('t', positive=True)
        if a == 0 and b == oo:
            found = self._mellin_on(_subs(f, x, t), t, None)
            if found is None and f.has(Exp):
                found = self._exponential_substitution(f, x, a, b)
            if found is None:
                found = self._series_mellin(_subs(f, x, t), t)
            return self._finish(found)
        if a == 0 and b == 1:
            return self._finish(self._mellin_on(_subs(f, x, t), t, 'lower'))
        if a == 1 and b == oo:
            return self._finish(self._mellin_on(_subs(f, x, t), t, 'upper'))
        if a == -oo and b == oo and f.has(Exp):
            return self._finish(self._exponential_substitution(f, x, a, b))
        return None

    def _mapped(self, f: Expr, x: Symbol, a: Expr, b: Expr, depth: int) -> Optional[ConditionalValue]:
        """Other ranges mapped linearly onto `(0, oo)` and `(0, 1)`, and
        the real line cut at 0."""
        t = _fresh('t', positive=True)
        if a == -oo and b == oo:
            right = self.integrate(f, x, S.Zero, oo, depth + 1, False, True)
            if right is None:
                return None
            left = self.integrate(f, x, -oo, S.Zero, depth + 1, False, True)
            if left is None:
                return None
            total = left.add(right)
            # halves which hold on disjoint regions (the bug: the halves of
            # exp(-(x - mu)**2/(2*sigma**2)) hold for mu < 0 and mu > 0,
            # |arg(-mu/sigma**2)| < pi/2 and |arg(mu/sigma**2)| < pi/2,
            # and their sum, undefined at mu = 0, cancelled to 0 under the
            # conjunction, which no sample could check)
            parameters = _sorted_symbols((f.free_symbols | total.condition.free_symbols) - {x})
            if _holds_nowhere(total.condition, parameters, self.assumptions):
                return None
            return total
        if a == -oo:
            return self.integrate(_subs(f, x, b - t), t, S.Zero, oo, depth + 1, False, True)
        if b == oo:
            if a == 0:
                return None
            return self.integrate(_shift(f, x, t, a), t, S.Zero, oo, depth + 1, False, True)
        if a == 0:
            if b == 1:
                return None
            # (0, b): x = b t
            if self.ask(b > 0) is not True:
                return None
            g = _subs(f, x, b * t) * b
            return self.integrate(g, t, S.Zero, S.One, depth + 1, False, True)
        # (a, b): x = a + (b - a) t, and reflected, x = b - (b - a) t, when
        # that fails: the singular end of log(-x)/sqrt(1 - x**2) over
        # (-1, 0) is at 0, which the reflection sends to t = 0 where the
        # Mellin method reads it
        length = b - a
        g = _subs(f, x, a + length * t) * length
        found = self.integrate(g, t, S.Zero, S.One, depth + 1, False, True)
        if found is not None:
            return found
        g = _subs(f, x, b - length * t) * length
        return self.integrate(g, t, S.Zero, S.One, depth + 1, False, True)

    def _series_mellin(self, g: Expr, t: Symbol) -> Optional[ConditionalValue]:
        """``Integral(g, (t, 0, oo))`` as the Mellin transform at ``s = 1``
        found by Ramanujan's master theorem or the method of brackets
        (:mod:`.brackets`), for factors outside the table."""
        from .brackets import mellin_transform_series
        s = _fresh('s')
        found = mellin_transform_series(g, t, s)
        if found is None:
            return None
        transform = found.transform
        value = _subs(transform, s, S.One)
        if value.has(nan, zoo, oo, -oo):
            limited = attempt(lambda: limit(transform, s, S.One), settings.timeout, native=True)
            if limited is None or limited.has(nan, zoo, oo, -oo):
                return None
            value = limited
        lower, upper = found.strip
        parts: list[Boolean] = []
        if lower != -oo:
            parts.append(lower < 1)
        if upper != oo:
            parts.append(upper > 1)
        return ConditionalValue(value, and_(*parts, found.condition))

    def _exponential_substitution(self, f: Expr, x: Symbol, a: Expr, b: Expr) -> Optional[ConditionalValue]:
        """``t = exp(c x)`` for an integrand in ``exp(c_i x)`` with all
        ``c_i`` integer multiples of ``c``: the real line becomes
        ``(0, oo)``, and ``(0, oo)`` becomes ``(0, 1)`` with ``c < 0``."""
        c = _exponential_generator(f, x, negative=(a == 0))
        if c is None:
            return None
        u = _fresh('u', positive=True)
        g = _denest(_powers_of_logarithms(_subs(f, x, log(u) / c) / (c * u), u), self.assumptions)
        if g.has(log(u)) and not _log_powers_only(g, u):
            return None
        negative = c.is_negative is True
        if a == 0:
            cutoff = 'lower' if negative else 'upper'
            found = self._mellin_on(g, u, cutoff)
            return None if found is None else found.scaled(S.NegativeOne if negative else S.One)
        found = self._mellin_on(g, u, None)
        return None if found is None else found.scaled(S.NegativeOne if negative else S.One)

    def _inversion(self, f: Expr, x: Symbol, a: Expr, b: Expr, depth: int) -> Optional[ConditionalValue]:
        """``x = 1/u`` between ``(1, oo)`` and ``(0, 1)``."""
        if (a, b) not in ((S.One, oo), (S.Zero, S.One)):
            return None
        u = _fresh('u', positive=True)
        g = _denest(_subs(f, x, 1 / u) / u**2, self.assumptions)
        cutoff = 'lower' if b == oo else 'upper'
        return self._finish(self._mellin_on(g, u, cutoff))

    def _trigonometric(self, f: Expr, x: Symbol, a: Expr, b: Expr, depth: int) -> Optional[ConditionalValue]:
        """Trigonometric integrands over ranges with endpoints at multiples
        of ``pi/2``: cut into quarter periods, each mapped onto
        ``(0, pi/2)`` and integrated as a Beta integral through
        ``x = asin(sqrt(u))``."""
        if not _has_trigonometric(f) or a in (-oo, oo) or b in (-oo, oo):
            return None
        ka, kb = _quarter(a), _quarter(b)
        if ka is None or kb is None:
            return None
        if kb - ka > 1:
            t = _fresh('t')
            total = ConditionalValue(S.Zero)
            for k in range(ka, kb):
                piece = self.integrate(_subs(f, x, t + k * pi / 2), t, S.Zero, pi / 2, depth + 1, False, True)
                if piece is None:
                    return None
                total = total.add(piece)
            return total
        if ka != 0:
            t = _fresh('t')
            return self.integrate(_subs(f, x, t + ka * pi / 2), t, S.Zero, pi / 2, depth + 1, False, True)
        u = _fresh('u', positive=True)
        f = self._signed_form(f, x, S.Zero, pi / 2)
        g = _arcsine_substitution(f, x, u) / (2 * sqrt(u) * sqrt(1 - u))
        g = _denest(g, self.assumptions)
        # and after the substitution: -log(cos(x)) is -log(sqrt(1 - u)),
        # which is -log(1 - u)/2 on (0, 1)
        g = self._signed_form(g, u, S.Zero, S.One)
        if g.has(Asin):
            return None
        found = self._finish(self._mellin_on(g, u, 'lower'))
        if found is not None:
            return found
        # the other methods on the substituted form: -log(1 - u)/(4*u), from
        # log(1/cos(x))*cos(x)/sin(x), has a dilogarithm antiderivative
        return self.integrate(g, u, S.Zero, S.One, depth + 1, False, True)

    def _mean_value(self, f: Expr, x: Symbol, a: Expr, b: Expr, depth: int) -> Optional[ConditionalValue]:
        """A trigonometric integrand over whole periods as `2 pi k` times
        the constant Laurent coefficient of its form in ``exp(I x)``
        (:mod:`.periodic`)."""
        from .periodic import mean_value_integral
        if not _has_trigonometric(f) or a in (-oo, oo) or b in (-oo, oo):
            return None
        return self._finish(mean_value_integral(f, x, a, b, self.assumptions))

    def _residues(self, f: Expr, x: Symbol, a: Expr, b: Expr, depth: int) -> Optional[ConditionalValue]:
        from .residues import residue_integral
        return self._finish(residue_integral(f, x, a, b, self.assumptions))

    def _real_parts(self, condition: Boolean) -> Boolean:
        """SymPy's conditions with ``re(p)`` replaced by ``p`` and ``im(p)``
        by 0 for the parameters known to be real."""
        replacement: dict[Basic, Basic] = {}
        for node in condition.atoms(Re, Im):
            p = node.args[0]
            if isinstance(p, Symbol) and (p.is_extended_real or self.ask(element(p, Reals)) is True):
                replacement[node] = p if isinstance(node, Re) else S.Zero
        return to_boolean(condition._xreplace_evaluated(replacement)) if replacement else condition

    def _radicals(self, f: Expr, x: Symbol, a: Expr, b: Expr, depth: int) -> Optional[ConditionalValue]:
        """``x**n * Q**(m/2)`` with ``Q`` quadratic (:mod:`.radicals`): the
        real antiderivative of the table evaluated at the endpoints, cheap
        and early, before the methods which spend the budget on the sums
        of radicals the cells of a region leave (the bug: the slices of
        the cylinders reached the antiderivative route, the last one, with
        no time left)."""
        from .radicals import quadratic_radical_antiderivative
        if a.has(x) or b.has(x) or quadratic_radical_antiderivative(f, x) is None:
            return None
        # under a quarter of the time limit: the Euler substitutions give
        # antiderivatives in arctangents of radicals whose discontinuities
        # can take seconds to place (sqrt(1 - x**2)/(x**2 + 1) over (-1, 1),
        # which another route has in a second)
        return self._under_budget(lambda: self._antiderivative(f, x, a, b, depth), 4)

    def _arctangent_pairs(self, f: Expr, x: Symbol, a: Expr, b: Expr) -> Expr:
        """``atan(u) + atan(1/u)`` written ``pi/2`` where ``u`` is positive
        on the range (``-pi/2`` where negative): ``(atan(x**(1/3)) +
        atan(x**(-1/3)))*log(x)/(x**2 + 1)`` over ``(0, oo)`` is
        ``pi/2*log(x)/(x**2 + 1)``, which is 0 (Maxima's ``rtestint`` 75)."""
        if not f.has(Atan):
            return f
        sign_: Optional[int] = None
        if self.ask(a >= 0) is True:
            sign_ = 1
        elif self.ask(b <= 0) is True:
            sign_ = -1
        if sign_ is None:
            return f
        orientation = sign_
        signed = _fresh(x.name, positive=True)

        def pairs(node: Basic) -> Basic:
            terms = [as_expr(term) for term in node.args]
            for i, first in enumerate(terms):
                if not isinstance(first, Atan) or not first.has(x):
                    continue
                u = as_expr(first.args[0])
                for j, second in enumerate(terms):
                    if j <= i or not isinstance(second, Atan):
                        continue
                    v = as_expr(second.args[0])
                    if cancel(u * v - 1) != 0:
                        continue
                    positive = _subs(u, x, orientation * signed).is_positive
                    if positive is None:
                        continue
                    rest = [term for k, term in enumerate(terms) if k not in (i, j)]
                    return add(*rest, pi / 2 if positive else -pi / 2)
            return node

        return as_expr(f.replace(lambda node: isinstance(node, Add) and node.has(Atan), pairs))

    def _quick_antiderivative(self, f: Expr, x: Symbol, a: Expr, b: Expr,
                              depth: int) -> Optional[ConditionalValue]:
        """The antiderivative route early, for the shapes it answers at
        once (a polynomial in elementary functions of linear arguments,
        with a ``log(x)`` factor by parts): ``u**3*exp(-u)*log(u)`` over
        ``(sqrt(x), oo)`` in two seconds, where the mapped range spent
        seven on the shifted form first."""
        from .antiderivative import by_parts_shape, quick_shape
        if a.has(x) or b.has(x) or not quick_shape(f, x):
            return None
        if (a in (-oo, oo) or b in (-oo, oo)) and not by_parts_shape(f, x):
            # a polynomial in exponentials over an infinite range is a
            # Laplace transform, which the Mellin routes settle with their
            # conditions; the limits at infinity of the antiderivative
            # spent the budget on Maxima's specint 32 first
            return None
        return self._under_budget(lambda: self._antiderivative(f, x, a, b, depth), 4)

    def _antiderivative(self, f: Expr, x: Symbol, a: Expr, b: Expr, depth: int,
                        late: bool = False, known: Optional[Expr] = None) -> Optional[ConditionalValue]:
        """An antiderivative evaluated by one-sided limits at the
        endpoints and at its discontinuities (:mod:`.antiderivative`):
        by the exact methods first, by the heuristics ``late``, or the one
        ``known`` already."""
        from .antiderivative import antiderivative_integral
        if a.has(x) or b.has(x):
            return None
        found = antiderivative_integral(f, x, a, b, self.assumptions, late, known)
        if found is None:
            return None
        if found.value in (oo, -oo):
            # a divergence to a signed infinity, from an infinite one-sided
            # limit of the antiderivative: refused when the quadrature
            # converges (a wrong antiderivative, a branch cut taken for a
            # pole), kept otherwise
            if settings.numerical_checks and _quadrature_converges(f, x, a, b, self.assumptions):
                return None
            return self._finish(found)
        if settings.numerical_checks and verify_numerically(found.value, f, x, a, b, self.assumptions) is not True:
            # the antiderivatives sometimes hold for positive parameters
            # or principal branches only: kept when confirmed numerically
            return None
        return self._finish(found)

    def _radical_substitution(self, f: Expr, x: Symbol, a: Expr, b: Expr,
                              depth: int) -> Optional[ConditionalValue]:
        """``u = q(x)**(1/n)`` for the radical ``q(x)**(k/n)`` of ``f``, ``q``
        a polynomial of degree at most two, nonnegative and monotone on
        the range: ``x**(1/3)`` and ``x**(2/3)`` become ``u`` and ``u**2``
        with ``x = u**3``, ``sqrt(z + 1)/(z**2 + 1)`` a rational function of
        ``u = sqrt(z + 1)``, ``z*sqrt(sqrt(z**2 - 1) + 1)`` over ``(1,
        sqrt(2))`` is ``u*sqrt(u + 1)`` over ``(0, 1)`` for ``u = sqrt(z**2
        - 1)``. Taken when the integrand has fewer radicals in ``u`` than
        in ``x``."""
        if isinstance(f, Add):
            # a sum is integrated term by term instead: the substitution
            # in log(z**(1/3) + 1) - acoth(sqrt(z) - 1) took the budget the
            # terms had their values under
            return None
        found = _radical_base(f, x)
        if found is None:
            return None
        q, n = found
        u = _fresh('u', positive=True)
        inverse = self._radical_inverse(q, n, x, u, a, b)
        if inverse is None:
            return None
        x_u, lower, upper = inverse
        # the logarithms expanded for the positive u: log(u**2) is 2*log(u),
        # which the antiderivative route reads at once
        g = _denest(expand_log(_subs(f, x, x_u) * x_u.diff(u)), self.assumptions)
        if g.has(x) or len(_radical_bases(g, u)) >= len(_radical_bases(f, x)):
            return None
        return self._under_budget(lambda: self.integrate(g, u, lower, upper, depth + 1, False, True))

    def _under_budget(self, compute: Callable[[], Optional[ConditionalValue]],
                      share: int = 2) -> Optional[ConditionalValue]:
        """``compute()`` under the time limit divided by ``share``: the
        integral in the new variable of a substitution must not spend the
        whole budget (``u = z**(1/6)`` for ``log(z**(1/3) + 1) -
        acoth(sqrt(z) - 1)`` over ``(1, 2)`` did, where the sum term by
        term had the value)."""
        budget = None if settings.timeout is None else settings.timeout / share
        return attempt(compute, budget, native=True)

    def _radical_inverse(self, q: Expr, n: int, x: Symbol, u: Symbol, a: Expr,
                         b: Expr) -> Optional[tuple[Expr, Expr, Expr]]:
        """``x`` in ``u = q(x)**(1/n)`` on ``(a, b)``, and the range of ``u``;
        ``None`` when ``q`` is not known nonnegative and monotone there."""
        coefficients = _coefficients(q, x)
        if coefficients is None:
            return None
        if len(coefficients) == 2:
            p, r = coefficients[1], coefficients[0]
            ends: list[Expr] = []
            for end in (a, b):
                if end in (oo, -oo):
                    if self.ask(p * end > 0) is not True:
                        return None
                    ends.append(oo)
                    continue
                value = _subs(q, x, end)
                if self.ask(value >= 0) is not True:
                    return None
                ends.append(_simplest_root(value**_rational(1, n)))
            return (u**n - r) / p, ends[0], ends[1]
        if len(coefficients) != 3 or n != 2 or not (a.is_number and b.is_number):
            return None
        # the range on one side of the vertex, where q is monotone, and
        # the branch of the inverse which takes a sample point back
        A, B, C = coefficients[2], coefficients[1], coefficients[0]
        vertex = -B / (2 * A)
        if not (self.ask(a >= vertex) is True or self.ask(b <= vertex) is True):
            return None
        ends = []
        for end in (a, b):
            if end in (oo, -oo):
                if self.ask(A > 0) is not True:
                    return None
                ends.append(oo)
                continue
            value = _subs(q, x, end)
            if self.ask(value >= 0) is not True:
                return None
            ends.append(sqrt(value))
        sample = self._sample(a, b)
        for sign_ in (S.One, S.NegativeOne):
            x_u = (-B + sign_ * sqrt(B**2 - 4 * A * (C - u**2))) / (2 * A)
            back = _subs(x_u, u, sqrt(_subs(q, x, sample)))
            if self.ask(eq(back, sample)) is True:
                return x_u, ends[0], ends[1]
        return None

    def _completed_square(self, f: Expr, x: Symbol, a: Expr, b: Expr,
                          depth: int) -> Optional[ConditionalValue]:
        """``u = x + c`` for a radical of a quadratic ``(x + c)**2 - d``
        with a linear term: ``exp(-s*t)*besselj(0, a*sqrt(t**2 + 2*k*t))``
        over ``(0, oo)`` is ``exp(s*k)*exp(-s*u)*besselj(0, a*sqrt(u**2 -
        k**2))`` over ``(k, oo)``, a table entry."""
        if isinstance(f, Add) or (a not in (-oo, oo) and b not in (-oo, oo)):
            # a finite range is mapped onto (0, 1) instead, where the
            # Mellin method reads log(x)/sqrt(4*x - x**2) over (0, 4)
            return None
        found = _radical_base(f, x)
        if found is None:
            return None
        q, _ = found
        coefficients = _coefficients(q, x)
        if coefficients is None or len(coefficients) != 3:
            return None
        A, B = coefficients[2], coefficients[1]
        if B == 0:
            return None
        c = B / (2 * A)
        u = _fresh('u', real=True)
        g = expand(_subs(f, x, u - c))
        g = as_expr(g.replace(lambda node: isinstance(node, Pow) and node.has(u),
                              lambda node: expand(as_expr(node.args[0]))**as_expr(node.args[1])))
        lower = a + c if a not in (-oo, oo) else a
        upper = b + c if b not in (-oo, oo) else b
        return self._under_budget(lambda: self.integrate(g, u, lower, upper, depth + 1, False, True))

    def _tangent_substitution(self, f: Expr, x: Symbol, a: Expr, b: Expr,
                              depth: int) -> Optional[ConditionalValue]:
        """``t = tan(k*x)`` on a range inside ``(-pi/2, pi/2)/k``, where the
        cosine is positive, and ``t = tan(k*x/2)`` on one inside ``(-pi,
        pi)/k`` (Weierstrass), for an integrand in trigonometric functions
        of ``k*x`` (the arguments expanded first): the integrand becomes
        algebraic in ``t`` over ``(tan(k*a), tan(k*b))``, ``sqrt(tan(x))``
        over ``(0, 1)`` is ``sqrt(t)/(1 + t**2)`` over ``(0, tan(1))`` and
        ``sqrt(tan(x) + sec(x))*sec(x)`` over ``(0, pi/4)`` is
        ``2/(sqrt(1 + t)*(1 - t)**(3/2))`` over ``(0, sqrt(2) - 1)``."""
        if isinstance(f, Add) or not _has_trigonometric(f) or a in (-oo, oo) or b in (-oo, oo):
            return None
        # a rational function of the trigonometric functions has an
        # antiderivative by the trigonometric method, in a better form
        # (-2*log(cos(x))/39 for tan(39*x/2), where the substitution gives
        # log(tan(x)**2 + 1)/39)
        placeholders: dict[Basic, Basic] = {}
        names: list[Symbol] = []
        for node in f.atoms(*_TRIGONOMETRIC):
            if node.has(x):
                names.append(_fresh('w'))
                placeholders[node] = names[-1]
        replaced = as_expr(f._xreplace_evaluated(placeholders))
        if replaced.is_rational_function(x, *names):
            return None
        g = f
        if any(monomial(as_expr(node.args[0]), x) is None for node in f.atoms(*_TRIGONOMETRIC) if node.has(x)):
            g = expand_trig(f)
        k: Optional[Expr] = None
        for node in g.atoms(*_TRIGONOMETRIC):
            if not node.has(x):
                continue
            found = monomial(as_expr(node.args[0]), x)
            if found is None or found[1] != 1 or (k is not None and found[0] != k):
                return None
            k = found[0]
        if k is None or self.ask(k > 0) is not True:
            return None
        frequency = k
        for half in (False, True):
            width = pi if half else pi / 2
            if self.ask(frequency * a >= -width) is not True or self.ask(frequency * b <= width) is not True:
                continue
            t = _fresh('t', positive=True) if self.ask(a >= 0) is True else _fresh('t', real=True)
            sine: Expr
            cosine: Expr
            jacobian: Expr
            if half:
                sine, cosine = 2 * t / (1 + t**2), (1 - t**2) / (1 + t**2)
                jacobian = 2 / (frequency * (1 + t**2))
            else:
                sine, cosine = t / sqrt(1 + t**2), 1 / sqrt(1 + t**2)
                jacobian = 1 / (frequency * (1 + t**2))
            replacement: dict[Basic, Basic] = {}
            for node in g.atoms(*_TRIGONOMETRIC):
                if not node.has(x):
                    continue
                if isinstance(node, Sin):
                    replacement[node] = sine
                elif isinstance(node, Cos):
                    replacement[node] = cosine
                elif isinstance(node, Tan):
                    replacement[node] = sine / cosine
                elif isinstance(node, Cot):
                    replacement[node] = cosine / sine
                elif isinstance(node, Sec):
                    replacement[node] = 1 / cosine
                elif isinstance(node, Csc):
                    replacement[node] = 1 / sine
                else:
                    return None
            lower = _tangent_of(frequency * a / 2 if half else frequency * a)
            upper = _tangent_of(frequency * b / 2 if half else frequency * b)
            h = as_expr(g._xreplace_evaluated(replacement)) * jacobian
            # the radicands factored (sqrt(2*t/(1 - t**2) + (1 + t**2)/(1 - t**2))
            # is sqrt((1 + t)**2/(1 - t**2))), then denested
            variable = t
            h = as_expr(h.replace(lambda node: isinstance(node, Pow) and isinstance(node.exp, Rational)
                                  and node.exp.q != 1 and node.has(variable),
                                  lambda node: factor(together(as_expr(node.args[0])))**as_expr(node.args[1])))
            h = _denest(h, self.assumptions)
            # the nested powers of a base positive on the range of t
            # flattened, sqrt(1/(1 - t**2)) is (1 - t**2)**(-1/2), which the
            # radical table reads (not ((-sin(x))**a)**(1/a), which is not
            # -sin(x) for a general a: the bug gave 0 for its integral)
            h = _flattened_powers(h, t, lower, upper)
            if h.has(x) or not _algebraic(h, t):
                # log(sin(x)) would become log(t/sqrt(1 + t**2)): nothing
                # gained, and the budget lost on it
                continue
            integrand, lo, hi = h, lower, upper
            value = self._under_budget(lambda: self.integrate(integrand, variable, lo, hi, depth + 1, False, True))
            if value is not None:
                return value
        return None

    def _asymptotic_divergence(self, f: Expr, x: Symbol, a: Expr, b: Expr) -> Optional[ConditionalValue]:
        """``oo`` or ``-oo`` when the leading term of ``f`` at a finite end
        of the range is ``c*(x - end)**p`` with ``p <= -1`` and a real ``c``
        of known sign (``-sin(x)*tan(x)*csc(x - 1)`` over ``(0, 1)`` is
        ``sin(1)*tan(1)/(1 - x)`` near 1, whose integral is ``oo``), the
        other end integrable or divergent to the same infinity; ``None``
        otherwise (a divergence at both ends to opposite infinities is
        the principal value's question)."""
        if f.has(Piecewise, Heaviside, Abs, Sign) or self.regularize or self.finite_part:
            # a regularised value or a finite part is another question
            return None
        verdicts: list[Expr] = []
        for end, side in ((a, '+'), (b, '-')):
            if end in (-oo, oo) or not end.is_number or _finite_at(f, x, end):
                continue
            verdict = self._divergence_at(f, x, end, side)
            if verdict is None:
                continue
            verdicts.append(verdict)
        if not verdicts or len(set(verdicts)) > 1:
            return None
        # nothing is claimed while the integrand may be singular elsewhere
        # in the range (the range has been cut at the singularities found;
        # a singularity not found would make the verdict unfounded), the
        # singularities placed against symbolic endpoints under the
        # assumptions
        singular = attempt(lambda: _singular_points(f, x, a, b),
                           settings.timeout / 8 if settings.timeout else None, native=True)
        if singular is None:
            return None
        placed = self._place(singular, a, b)
        if placed is None or placed[0] or placed[1]:
            return None
        if settings.numerical_checks and _quadrature_converges(f, x, a, b, self.assumptions):
            return None
        return ConditionalValue(verdicts[0])

    def _divergence_at(self, f: Expr, x: Symbol, end: Expr, side: str) -> Optional[Expr]:
        """The signed infinity the integral of ``f`` diverges to at ``end``
        by its leading term there, ``None`` for an integrable end or an
        undecided one."""
        h = _fresh('h', positive=True)
        g = _subs(f, x, end + h if side == '+' else end - h)
        budget = None if settings.timeout is None else settings.timeout / 8
        try:
            found = attempt(lambda: g.leadterm(h), budget, native=True)
        except (ValueError, NotImplementedError, PoleError, TypeError):
            return None
        if found is None:
            return None
        coefficient, exponent = found
        if coefficient.has(h) or not exponent.is_number or exponent.is_extended_real is not True \
                or not _is(exponent <= -1):
            return None
        if exponent == -1 and coefficient.has(Log):
            return None
        if self.ask(coefficient > 0) is True:
            return oo
        if self.ask(coefficient < 0) is True:
            return -oo
        return None

    def _binomial_differential(self, f: Expr, x: Symbol, a: Expr, b: Expr,
                               depth: int) -> Optional[ConditionalValue]:
        """A binomial differential ``x**m*(a + b*x**n)**p`` without
        parameters whose integral Chebyshev's criterion proves
        non-elementary, over a finite range, by its hypergeometric
        antiderivative (:func:`.algebraic.binomial_hypergeometric_antiderivative`)
        evaluated by one-sided limits on the pieces of the range between
        the points where ``z = -b*x**n/a`` is 1 (one case of the
        antiderivative on each), under half the time limit: on the
        integrand as given, before the powers are split into their
        factors and the range is mapped to ``(0, 1)``, after which
        ``(2 - t)**(7/2)/sqrt(5*t**2 - 20*t + 16)`` is no binomial any more
        (the regression: ``sqrt(t**7/(1 - 5*t**2))`` over ``(-1, -1/2)``
        spent the time limit on its mapped pieces and stayed
        unevaluated). The recognition is structural and cheap, and the
        table, the Marichev–Adamchik method and Legendre's
        elliptic forms, cheap too, are tried first as in the order of the
        strategies."""
        from .algebraic import binomial_exponents, chebyshev_elementary
        if f.free_symbols != {x} or a.has(x) or b.has(x) or a.is_finite is not True or b.is_finite is not True:
            return None
        if not any(isinstance(node, Pow) and isinstance(node.exp, Rational) and not node.exp.is_integer
                   and node.base.has(x) for node in f.atoms(Pow)):
            return None
        binomial = binomial_exponents(f, x)
        if binomial is None or chebyshev_elementary(binomial):
            return None
        ratio = -binomial.b / binomial.a
        if ratio.is_extended_real is not True:
            return None
        # the points where z = ratio*x**n is 1, where the cases of the
        # antiderivative meet: the range is cut there, so that one case
        # holds on each piece
        boundary = abs_(ratio)**(S.NegativeOne / binomial.n)
        candidates = [side * boundary for side in (1, -1)
                      if (side == 1 or binomial.n.q == 1) and (ratio * Integer(side)**binomial.n).is_extended_positive]
        cuts = self._points_in(_distinct(candidates), a, b)
        if cuts is None:
            return None
        for cheap in (self._table, self._canonical, self._elliptic):
            # in the order of the strategies: 1/sqrt(1 - x**4) over (0, 1)
            # is a beta function by the Marichev–Adamchik method, sqrt(x - x**3)
            # over (0, 1) complete elliptic integrals, at once
            found = cheap(f, x, a, b, depth)
            if found is not None:
                return found
        pieces = cuts

        def compute() -> Optional[ConditionalValue]:
            from .indefinite import verified_antiderivative
            known = verified_antiderivative(f, x, self.assumptions, ['binomial'])
            if known is None:
                return None
            bounds = [a] + pieces + [b]
            total = ConditionalValue(S.Zero)
            for lower, upper in zip(bounds[:-1], bounds[1:]):
                piece = self._antiderivative(f, x, lower, upper, depth, known=known[0])
                if piece is None:
                    return None
                total = total.add(piece)
            return self._finish(total)
        return self._under_budget(compute)

    def _late_antiderivative(self, f: Expr, x: Symbol, a: Expr, b: Expr,
                             depth: int) -> Optional[ConditionalValue]:
        """The antiderivative by the heuristic methods (the Risch–Norman
        method, the substitutions, SymPy's manual and Meijer routes), last
        of all: ``t*exp(-sqrt(t))*log(t)`` over ``(x, oo)`` has one in
        exponential integrals in seconds."""
        return self._antiderivative(f, x, a, b, depth, True)

    def _termwise(self, f: Expr, x: Symbol, a: Expr, b: Expr, depth: int) -> Optional[ConditionalValue]:
        """A sum integrated term by term when the whole defeats every
        method: the antiderivative of ``sqrt(x)*sqrt(2 - x) - sqrt(1 -
        x)*sqrt(x + 1)`` is a complex ``Piecewise`` whose branches give
        ``nan`` at the endpoints, while each term is elementary. The value
        is kept only when every term has a finite value (terms which
        diverge separately may cancel in the sum, ``1/x - 1/(x + x**2)``
        near 0) and the sum passes the numerical check; a product which
        expands to a few terms is tried in the expanded form."""
        terms = list(Add.make_args(f))
        if len(terms) < 2:
            terms = list(Add.make_args(expand(f)))
            if len(terms) < 2 or len(terms) > _TERMWISE_LIMIT:
                return None
        elif len(terms) > _TERMWISE_LIMIT:
            return None
        total: Optional[ConditionalValue] = None
        for term in terms:
            found = self.integrate(term, x, a, b, depth + 1, False, False)
            if found is None or found.value.has(oo, -oo, zoo, nan):
                return None
            total = found if total is None else total.add(found)
        if total is None:
            return None
        if settings.numerical_checks and verify_numerically(total.value, f, x, a, b, self.assumptions) is False:
            return None
        return self._finish(total)

    def _regularized_termwise(self, f: Expr, x: Symbol, a: Expr, b: Expr,
                              depth: int) -> Optional[ConditionalValue]:
        """A sum over ``(0, oo)`` whose terms diverge separately, by the
        analytic (Riesz) regularisation of the Mellin method on each
        term: the value of the formula, analytic in the exponents, beyond
        the strip where the term's own integral converges. The
        divergences cancel in the sum when the whole converges, which the
        numerical check must confirm: ``(exp(-b*t) - exp(-a*t))*exp(-s*t)/
        (2*sqrt(pi)*t**(3/2))`` is ``sqrt(a + s) - sqrt(b + s)``, each term
        alone divergent at 0 and regularised to ``gamma(-1/2)`` times a
        power (Maxima's ``specint`` 45)."""
        if a != 0 or b != oo:
            return None
        terms = list(Add.make_args(expand(f, power_exp=False)))
        if len(terms) < 2 or len(terms) > _TERMWISE_LIMIT:
            return None
        t = _fresh('t', positive=True)
        total: Optional[ConditionalValue] = None
        for term in terms:
            found = None
            for candidate in _forms(_subs(term, x, t), t, self.assumptions):
                found = mellin_integrate(candidate, t, self.assumptions, None, True)
                if found is not None:
                    break
            if found is None:
                total = None
                break
            total = found if total is None else total.add(found)
        at_one = False
        if total is None or total.value.has(oo, -oo, zoo, nan):
            # a term whose transform has a pole at s = 1 (exp(-u)/u is
            # gamma(s - 1), 1/(exp(u) - 1) is gamma(s)*zeta(s)): the
            # transforms summed as functions of s, whose poles cancel, and
            # the limit at 1, with zeta expanded about 1 (there is no
            # series for it there): the Euler-constant integrals
            total = self._transforms_at_one(terms, x, t)
            if total is None:
                return None
            at_one = True
        if not settings.numerical_checks:
            return None
        verdict = verify_numerically(total.value, f, x, a, b, self.assumptions)
        if verdict is not True and not (at_one and verdict is None):
            # the cancellation of the divergences is what the check
            # confirms; the cancelled poles of the transforms confirm it
            # exactly where the quadrature of the sum is inconclusive
            # (-1/u + 1/(1 - exp(-u)) near 0)
            return None
        return self._finish(total)

    def _transforms_at_one(self, terms: Sequence[Expr], x: Symbol, t: Symbol) -> Optional[ConditionalValue]:
        """The limit at ``s = 1`` of the sum of the Mellin transforms of the
        ``terms`` (the integral over ``(0, oo)`` of the sum, where each term
        alone diverges), ``None`` when a transform is missing or the limit
        is not finite."""
        s = _fresh('s')
        summed: Expr = S.Zero
        condition: Boolean = true
        for term in terms:
            transform = None
            for candidate in _forms(_subs(term, x, t), t, self.assumptions):
                transform = mellin_transform(candidate, t, s)
                if transform is not None:
                    break
            if transform is None:
                return None
            summed = summed + transform.transform
            condition = and_(condition, transform.condition)
        expanded = _zeta_about_one(summed, s)
        budget = None if settings.timeout is None else settings.timeout / 4
        value = attempt(lambda: limit(expanded, s, S.One), budget, native=True)
        if value is None or value.has(oo, -oo, zoo, nan, Limit) or value.has(s):
            return None
        return ConditionalValue(value, condition)

    def _logarithmic_substitution(self, f: Expr, x: Symbol, a: Expr, b: Expr,
                                  depth: int) -> Optional[ConditionalValue]:
        """``x = exp(-u)`` for an integrand over ``(0, 1)`` with ``log(x)``
        inside a function or a denominator: ``-log(log(1/x)) + 1/log(x) +
        1/(1 - x)`` becomes ``(-log(u) - 1/u + 1/(1 - exp(-u)))*exp(-u)``
        over ``(0, oo)``, a sum of Euler-constant integrals (Wester's
        problem 30, ``2*EulerGamma``)."""
        if a != 0 or b != 1 or not _logarithm_inside(f, x):
            return None
        u = _fresh('u', positive=True)
        g = expand_log(_subs(f, x, exp(-u)) * exp(-u), force=True)
        found = self._under_budget(lambda: self.integrate(g, u, S.Zero, oo, depth + 1, False, True))
        if found is None:
            return None
        if settings.numerical_checks and verify_numerically(found.value, f, x, a, b, self.assumptions) is False:
            # checked on the form as given, whose quadrature is the sounder
            return None
        return found

    def _parameter_exponentials(self, f: Expr, x: Symbol, a: Expr, b: Expr,
                                depth: int) -> Optional[ConditionalValue]:
        """The exponentials (and hyperbolic functions) of the parameters
        of a rational function of ``x`` made a positive symbol ``y``:
        ``1/(x**4 + 2*x**2*cosh(2*a) + 1)`` is ``1/((x**2 + y)*(x**2 + 1/y))``
        for ``y = exp(2*a)``, whose integral over ``(0, oo)`` the residues
        give as ``pi*sqrt(y)/(2*(y + 1))``; the parameters must be real."""
        # the hyperbolic functions of the parameters only (the rewriting
        # in exponentials on the whole would write x**4 as exp(4*log(x)))
        g = as_expr(f.replace(lambda node: isinstance(node, _HYPERBOLIC) and not node.has(x),
                              lambda node: node.rewrite(Exp)))
        exponentials = [as_expr(node) for node in g.atoms(Exp) if not node.has(x)]
        if not exponentials or not g.is_rational_function(x):
            return None
        arguments = [as_expr(node.args[0]) for node in exponentials]
        symbols: set[Symbol] = set()
        for argument in arguments:
            symbols |= argument.free_symbols
        parameters = _sorted_symbols(symbols)
        if len(parameters) != 1:
            return None
        p = parameters[0]
        coefficients: list[Rational] = []
        for argument in arguments:
            c = argument / p
            if c.has(p) or not isinstance(c, Rational):
                return None
            coefficients.append(c)
        # y = exp(g*p) for the generator g of the coefficients
        numerator = 0
        denominator = 1
        for c in coefficients:
            numerator = gcd(numerator, int(c.p))
            denominator = lcm(denominator, int(c.q))
        generator = _rational(numerator, denominator)
        y = _fresh('y', positive=True)
        rule: dict[Basic, Basic] = {}
        for node, c in zip(exponentials, coefficients):
            rule[node] = y**(c / generator)
        h = as_expr(g._xreplace_evaluated(rule))
        if h.has(p):
            return None
        found = self._under_budget(lambda: self.integrate(h, x, a, b, depth + 1, False, True))
        if found is None:
            return None
        real = self.ask(element(p, Reals))
        condition = and_(found.condition, true if real else element(p, Reals))
        return ConditionalValue(_subs(found.value, y, exp(generator * p)), condition)

    def _dirichlet(self, f: Expr, x: Symbol, a: Expr, b: Expr, depth: int) -> Optional[ConditionalValue]:
        """Trigonometric sums over powers of ``x`` on the half-lines and
        the real line (:mod:`.dirichlet`): the Dirichlet, Frullani and
        Borwein integrals."""
        from .dirichlet import dirichlet_integral
        if a not in (S.Zero, -oo) or b not in (S.Zero, oo) or not f.has(Sin, Cos, Exp):
            return None
        return self._finish(dirichlet_integral(f, x, a, b, self.assumptions))

    def _table(self, f: Expr, x: Symbol, a: Expr, b: Expr, depth: int) -> Optional[ConditionalValue]:
        """The table of Gradshteyn and Ryzhik (:mod:`.tables`), cheap and
        first."""
        from .tables import table_integral
        return self._finish(table_integral(f, x, a, b, self.assumptions))

    def _table_termwise(self, f: Expr, x: Symbol, a: Expr, b: Expr, depth: int) -> Optional[ConditionalValue]:
        """A sum every term of which the table has (up to a constant),
        before the methods: the Laplace transform of ``erfc(k/(2*sqrt(t)))
        - exp(a*k)*exp(a**2*t)*erfc(a*sqrt(t) + k/(2*sqrt(t)))`` (Maxima's
        ``specint`` 107) is the difference of two entries, which the
        term-by-term route at the end of the chain reached with no budget
        left."""
        from .tables import table_integral
        terms = Add.make_args(expand(f, power_exp=False))
        if len(terms) < 2 or len(terms) > _TERMWISE_LIMIT:
            return None
        total = ConditionalValue(S.Zero)
        for term in terms:
            constant, rest = _combined_exponentials(term, x).as_independent(x, as_Add=False)
            found = table_integral(rest, x, a, b, self.assumptions)
            if found is None:
                return None
            total = total.add(found.scaled(constant))
        return self._finish(total)

    def _elliptic(self, f: Expr, x: Symbol, a: Expr, b: Expr, depth: int) -> Optional[ConditionalValue]:
        """Square roots of cubics and quartics reduced to Legendre's
        elliptic integrals (:mod:`.elliptic`)."""
        from .elliptic import elliptic_integral
        if not any(isinstance(node, Pow) and isinstance(node.exp, Rational) and node.exp.q == 2
                   for node in f.atoms(Pow)):
            return None
        return self._finish(elliptic_integral(f, x, a, b, self.assumptions))

    def _algebraic(self, f: Expr, x: Symbol, a: Expr, b: Expr, depth: int) -> Optional[ConditionalValue]:
        """Rationalising substitutions for algebraic integrands of genus
        zero (:mod:`.algebraic`): Euler's substitutions, Möbius roots,
        Chebyshev's binomial differentials."""
        from .algebraic import algebraic_integral
        if depth > 1 or not any(isinstance(node, Pow) and isinstance(node.exp, Rational) and not node.exp.is_integer
                                and node.base.has(x) for node in f.atoms(Pow)):
            return None
        return self._finish(algebraic_integral(f, x, a, b, self.assumptions))

    def _series(self, f: Expr, x: Symbol, a: Expr, b: Expr, depth: int) -> Optional[ConditionalValue]:
        """Series expansion of a factor and termwise integration, the
        series summed in closed form (:mod:`.series`)."""
        from .series import series_integral
        if depth > 0:
            return None
        if a != 0:
            # a Fourier series on a bounded range, (-pi, pi) or (0, t)
            if a in (oo, -oo) or b in (oo, -oo) or not (f.has(Log) or _has_trigonometric(f)):
                return None
        elif b not in (S.One, oo) and not b.is_positive:
            return None
        return self._finish(series_integral(f, x, a, b, self.assumptions))

    def _laplace(self, f: Expr, x: Symbol, a: Expr, b: Expr, depth: int) -> Optional[ConditionalValue]:
        """The operational rules of the Laplace transform
        (:mod:`.laplace`) for ``g(x)*exp(-s*x)`` over ``(0, oo)``."""
        from .laplace import laplace_integral
        if depth > 1 or a != 0 or b != oo or not f.has(Exp):
            return None
        return self._finish(laplace_integral(f, x, a, b, self.assumptions))

    def _contours(self, f: Expr, x: Symbol, a: Expr, b: Expr, depth: int) -> Optional[ConditionalValue]:
        """Rectangular, sector and indented contours (:mod:`.contours`)."""
        from .contours import contour_integral
        return self._finish(contour_integral(f, x, a, b, self.assumptions))

    def _transformations(self, f: Expr, x: Symbol, a: Expr, b: Expr, depth: int) -> Optional[ConditionalValue]:
        """Frullani's theorem and Glasser's master theorem
        (:mod:`.transformations`)."""
        from .transformations import transformation_integral
        if depth > 1:
            return None
        return self._finish(transformation_integral(f, x, a, b, self.assumptions))

    def _holonomic(self, f: Expr, x: Symbol, a: Expr, b: Expr, depth: int) -> Optional[ConditionalValue]:
        """Creative telescoping (:mod:`.telescoping`) for a hyperexponential
        integrand with one parameter: the integral satisfies a linear
        ODE in the parameter, solved with initial conditions."""
        from .reduction import reduction_integral
        from .telescoping import holonomic_integral, is_hyperexponential
        if not self.parametric or depth > 1:
            return None
        parameters = _sorted_symbols(f.free_symbols - {x} - a.free_symbols - b.free_symbols)
        if len(parameters) != 1 or not is_hyperexponential(f, x, parameters[0]):
            return None
        # the reduction-based telescoper first (minimal order, no bound
        # on the certificate), the ansatz as the fallback
        found = reduction_integral(f, x, a, b, parameters[0], self.assumptions)
        if found is None:
            found = holonomic_integral(f, x, a, b, parameters[0], self.assumptions)
        return self._finish(found)

    def _dfinite(self, f: Expr, x: Symbol, a: Expr, b: Expr, depth: int) -> Optional[ConditionalValue]:
        """Chyzak's algorithm (:mod:`.dfinite`): creative telescoping for
        a D-finite integrand with one parameter which is not
        hyperexponential."""
        from .dfinite import dfinite_integral
        from .telescoping import is_hyperexponential
        if not self.parametric or depth > 1:
            return None
        parameters = _sorted_symbols(f.free_symbols - {x} - a.free_symbols - b.free_symbols)
        if len(parameters) != 1 or is_hyperexponential(f, x, parameters[0]):
            return None
        return self._finish(dfinite_integral(f, x, a, b, parameters[0], self.assumptions))

    def _parametric(self, f: Expr, x: Symbol, a: Expr, b: Expr, depth: int) -> Optional[ConditionalValue]:
        """Differentiation under the integral sign (:mod:`.parametric`),
        for each parameter of the integrand in turn."""
        from .parametric import candidate_parameters, parametric_integral
        if not self.parametric or depth > 1:
            return None
        for p in candidate_parameters(f, x):
            found = parametric_integral(f, x, a, b, p, self.assumptions, depth, DefiniteIntegrator(self.assumptions))
            if found is not None:
                return self._finish(found)
        return None

    def _last(self, f: Expr, x: Symbol, a: Expr, b: Expr, depth: int = 0) -> Optional[ConditionalValue]:
        """The last of the methods, where SymPy's ``integrate`` was: the
        Meijer G-function theorems (:func:`.meijerint.meijerint_definite`)
        and an antiderivative evaluated at the endpoints, in SymPy's order
        (the theorems first over an infinite range), each under a quarter
        of the time limit and checked numerically."""
        budget = None if settings.timeout is None else settings.timeout / 4
        infinite = a in (oo, -oo) or b in (oo, -oo)
        routes: list[Callable[[Expr, Symbol, Expr, Expr], Optional[ConditionalValue]]] = \
            [self._meijer_value, self._endpoint_value]
        if not infinite:
            routes.reverse()
        for route in routes:
            try:
                value = attempt(lambda: route(f, x, a, b), budget, native=True)
            except (ZeroDivisionError, AttributeError, AssertionError, OverflowError, IndexError):
                value = None
            if value is None or value.value.has(nan, zoo) or value.condition.has(x):
                continue
            found = self._finish(ConditionalValue(value.value, self._real_parts(value.condition)))
            if found is None:
                continue
            if settings.numerical_checks and not f.has(DiracDelta):
                if found.value.has(oo, -oo):
                    continue
                if verify_numerically(found.value, f, x, a, b, self.assumptions) is not True:
                    # an answer which the quadrature cannot confirm is not
                    # kept: the branch cuts of the antiderivatives and the
                    # conditions of the theorems make such answers wrong
                    # too often (a delta function cannot be integrated
                    # numerically and is integrated symbolically)
                    continue
            tidied = tidy(found.value, self.assumptions)
            if tidied != found.value and settings.numerical_checks and not f.has(DiracDelta) \
                    and verify_numerically(tidied, f, x, a, b, self.assumptions) is False:
                # the simplification is checked too: it turned the
                # log(-exp_polar(I*pi)) of an answer for u**3*exp(-u)*log(u)
                # over (1, oo) into 2*I*pi
                tidied = found.value
            return ConditionalValue(tidied, found.condition)
        return None

    def _meijer_value(self, f: Expr, x: Symbol, a: Expr, b: Expr) -> Optional[ConditionalValue]:
        """The integral by the theorems on Meijer G-functions, with their
        convergence condition."""
        from .meijerint import meijerint_definite
        found = meijerint_definite(f, x, a, b)
        if found is None or isinstance(found[1], BooleanFalse):
            return None
        return ConditionalValue(found[0], found[1])

    def _endpoint_value(self, f: Expr, x: Symbol, a: Expr, b: Expr) -> Optional[ConditionalValue]:
        """An antiderivative (the verified methods of :mod:`.indefinite`,
        the delta functions by :func:`.deltafunctions.deltaintegrate`) at
        the endpoints, by limits, as SymPy's ``integrate`` evaluates it."""
        from .deltafunctions import deltaintegrate
        from .indefinite import verified_antiderivative
        F: Optional[Expr] = None
        if f.has(DiracDelta):
            F = deltaintegrate(f, x)
        else:
            # the exact methods and the rules, which fail fast (those of the
            # antiderivative route of the driver): not the heuristics, which
            # spent the whole budget where SymPy's integrate gave up in
            # seconds (the radicals of the cells of regions), nor the Meijer
            # G route, which the theorems cover and whose antiderivatives
            # took the budget to check (lerchphi of polar arguments for
            # x**k/(x + 3))
            from .antiderivative import _METHODS as EXACT
            found = verified_antiderivative(f, x, self.assumptions, list(EXACT))
            F = None if found is None else found[0]
        if F is None or is_unevaluated(F) or F.has(Integral):
            return None
        if _real_bounds(a, b) and not f.has(DiracDelta):
            # cut at the discontinuities of F inside the range (the
            # antiderivative of 1/(cos(x) + 2) jumps at pi)
            from .antiderivative import antiderivative_integral
            found_value = antiderivative_integral(f, x, a, b, self.assumptions, known=F)
            if found_value is not None and not found_value.value.has(oo, -oo):
                return found_value
        # the values at the endpoints, as SymPy's integrate takes them (the
        # numerical check of the caller rejects a jump inside the range)
        values: list[Expr] = []
        for point, direction in ((b, '-'), (a, '+')):
            value = _endpoint(F, x, point, direction)
            if value is None:
                return None
            values.append(value)
        total = values[0] - values[1]
        if total.has(oo, -oo):
            return None
        return ConditionalValue(total)


def _endpoint(F: Expr, x: Symbol, point: Expr, direction: str) -> Optional[Expr]:
    """``F`` at an endpoint: its value when it is finite there and no step
    function jumps there (the antiderivatives of delta functions are steps:
    the limit of ``E*Heaviside(x - 1)`` at ``0`` does not finish), the
    limit from inside the range otherwise; ``None`` when neither is
    known."""
    if point not in (oo, -oo):
        steps = [h for h in F.atoms(Heaviside) if isinstance(h, Heaviside)]
        value = F.subs(x, point)
        if not value.has(nan, zoo, oo, -oo) and all(_expr_arg(h).subs(x, point).is_zero is False for h in steps):
            return value
        if steps:
            return None
    else:
        # the steps at an infinite endpoint: 1 or 0 by the limit of their
        # argument (the limit of Heaviside(x - 1/2)/4 at oo does not finish)
        rule: dict[Basic, Basic] = {}
        for h in F.atoms(Heaviside):
            if not isinstance(h, Heaviside):
                continue
            side = limit(_expr_arg(h), x, point)
            if side == oo:
                rule[h] = S.One
            elif side == -oo:
                rule[h] = S.Zero
            else:
                return None
        if rule:
            F = as_expr(F.xreplace(rule))
    value = limit(F, x, point) if point in (oo, -oo) or point.is_extended_real is not True \
        else limit(F, x, point, direction)
    if isinstance(value, Limit) or value.has(nan, zoo):
        return None
    return value


def _expr_arg(node: Heaviside) -> Expr:
    return node.args[0]


class DefiniteIntegrator(ParametricIntegrator):
    """The integrator of this module without differentiation under the
    integral sign, and its :func:`verify_numerically`, under the
    assumptions: the integrator of
    :func:`sympy_extras.integrals.parametric.parametric_integral`.

    >>> from sympy_extras.core import symbols, oo, Integer
    >>> from sympy_extras.functions import exp
    >>> from sympy_extras.integrals.definite import DefiniteIntegrator
    >>> x, a = symbols('x a', positive=True)
    >>> DefiniteIntegrator(None).integrate(exp(-a*x), x, Integer(0), oo, 0)
    ConditionalValue(1/a)
    """

    __slots__ = ('assumptions', '_integrator')

    def __init__(self, assumptions: Assumptions) -> None:
        self.assumptions: Final[Assumptions] = assumptions
        self._integrator: Final[_Integrator] = _Integrator(assumptions, parametric=False)

    def integrate(self, f: Expr, x: Symbol, a: Expr, b: Expr, depth: int) -> Optional[ConditionalValue]:
        return self._integrator.integrate(f, x, a, b, depth, True)

    def verify(self, value: Expr, f: Expr, x: Symbol, a: Expr, b: Expr) -> Optional[bool]:
        return verify_numerically(value, f, x, a, b, self.assumptions)


def definite_integrator(assumptions: Assumptions) -> DefiniteIntegrator:
    """The :class:`DefiniteIntegrator` under the assumptions (the default
    integrator of :func:`sympy_extras.integrals.parametric.parametric_integral`)."""
    return DefiniteIntegrator(assumptions)


#: the random points at which a condition must be false before it is
#: taken to hold nowhere
_NOWHERE_POINTS: Final = 8


def _holds_nowhere(condition: Boolean, symbols: Sequence[Symbol], assumptions: Assumptions) -> bool:
    """Whether ``condition`` is false at random real points of every
    sign satisfying the assumptions and the flags of the symbols, at
    ``_NOWHERE_POINTS`` of them at least, where it is decided at every
    one; ``False`` when it holds at one, or cannot be decided.

    >>> from sympy_extras.core import symbols, pi
    >>> from sympy_extras.functions import abs_, arg
    >>> from sympy_extras.integrals.definite import _holds_nowhere
    >>> mu, s = symbols('mu sigma')
    >>> _holds_nowhere((abs_(arg(mu/s**2)) < pi/2) & (abs_(arg(-mu/s**2)) < pi/2), [mu, s], None)
    True
    >>> _holds_nowhere(abs_(arg(mu/s**2)) < pi/2, [mu, s], None)
    False
    """
    if condition is true or not symbols:
        return False
    plain = plain_symbols(symbols)
    facts = with_symbol_facts(assumptions, symbols, plain)
    rng = random.Random(str(condition))
    tested = 0
    for _ in range(4 * _NOWHERE_POINTS):
        values: dict[Symbol, Expr] = {s: _rational(rng.choice((-1, 1)) * rng.randint(1, 19), rng.randint(1, 4))
                                      for s in symbols}
        at = {plain.get(s, s): v for s, v in values.items()}
        if not all(_xr_boolean(fact, at) is true for fact in facts):
            continue
        verdict = _xr_boolean(condition, values)
        if verdict is not false:
            return False
        tested += 1
    return tested >= _NOWHERE_POINTS


#: the samples outside a condition at which a value is checked
_OUTSIDE_SAMPLES: Final = 2


def _holds_outside(value: Expr, condition: Boolean, f: Expr, x: Symbol, a: Expr, b: Expr,
                   assumptions: Assumptions) -> bool:
    """Whether ``value`` agrees with the quadrature of ``Integral(f, (x,
    a, b))`` at random real samples of the parameters satisfying the
    assumptions where ``condition`` is false, at ``_OUTSIDE_SAMPLES`` of
    them: ``False`` at a sample where they disagree, or where the
    quadrature has no finite value (the integral diverges there, or
    cannot be checked); ``True`` when they agree, or when no sample
    falls outside the condition. The samples are drawn at random and
    the condition evaluated at each, since the negation of a condition
    on arguments, ``Abs(arg(-k)) < pi/2``, is beyond the solvers which
    find instances."""
    parameters = _sorted_symbols((f.free_symbols | a.free_symbols | b.free_symbols) - {x})
    if not parameters:
        return True
    plain = plain_symbols(parameters)
    facts = with_symbol_facts(assumptions, parameters, plain)
    rng = random.Random(str((f, condition)))
    checked = 0
    for _ in range(4 * _NOWHERE_POINTS):
        values: dict[Symbol, Expr] = {s: _rational(rng.choice((-1, 1)) * rng.randint(1, 19), rng.randint(1, 4))
                                      for s in parameters}
        at = {plain.get(s, s): v for s, v in values.items()}
        try:
            if not all(_xr_boolean(fact, at) is true for fact in facts) \
                    or _xr_boolean(condition, values) is not false:
                continue
        except TypeError:
            continue
        expected = _quadrature(_xr(f, values), x, _xr(a, values), _xr(b, values))
        if expected is None or _agrees(value, expected, values) is False:
            return False
        checked += 1
        if checked >= _OUTSIDE_SAMPLES:
            break
    return True


def _pairs(node: Piecewise) -> list[tuple[Expr, Boolean]]:
    """The ``(value, condition)`` pairs of a ``Piecewise``."""
    return [(pair.expr, pair.cond) for pair in node.args if isinstance(pair, ExprCondPair)]


def _with(assumptions: Assumptions, extra: Sequence[Boolean]) -> Assumptions:
    """The assumptions together with ``extra``."""
    if not extra:
        return assumptions
    items: list[Boolean] = list(extra)
    items.extend(_items(assumptions))
    return items


def _distinct(points: Sequence[Expr]) -> list[Expr]:
    """The points without repetitions, in their order."""
    found: list[Expr] = []
    for p in points:
        if p not in found:
            found.append(p)
    return found


def _singular_points(f: Expr, x: Symbol, a: Expr, b: Expr) -> Optional[list[Expr]]:
    """The candidate singularities of ``f`` on ``(a, b)``, their position
    against the endpoints left to the placement: the native port of
    SymPy's ``singularities`` under the bounds of the range, and, when
    the solver cannot list them there (a zero ``a`` of ``x - a`` against
    symbolic bounds is a ``ConditionSet``), on the whole real line,
    without the points which are not real for real parameters (the
    complex cube roots of ``x**3 - c**3``); ``None`` when neither lists
    them."""
    bounds: list[Boolean] = []
    if a != -oo:
        bounds.append(x > a)
    if b != oo:
        bounds.append(x < b)
    def zeros(u: Expr) -> Optional[list[Expr]]:
        # the zeros of sin(w*x + c) and cos(w*x + c) on a numeric range
        # enumerated: the native signs of the solver spent seconds on them
        return _trigonometric_zeros(u, x, a, b)

    found = singularities(f, x, bounds, zeros)
    if found is None:
        found = singularities(f, x, None)
        if found is None:
            return None
        found = [p for p in found if not _complex_for_real_parameters(p)]
    return _distinct([_real_odd_roots(p) for p in found])


def _complex_for_real_parameters(p: Expr) -> bool:
    """Whether ``p`` has the imaginary unit and is not known real when its
    symbols are taken real (``c*(-1/2 + sqrt(3)*I/2)``)."""
    if not p.has(I):
        return False
    replacement: dict[Symbol, Expr] = {s: _fresh(s.name, real=True) for s in p.free_symbols}
    return _xr(p, replacement).is_extended_real is not True


def _candidate_points(found: Set, x: Symbol, a: Expr, b: Expr) -> Optional[list[Expr]]:
    """The points of a set of zeros on the range ``(a, b)``, their
    position against the endpoints left to the placement: those of a
    finite set, none of the empty set, and those of the finite part of
    ``Intersection({0}, Interval.open(a, b))`` or of ``ConditionSet(x, (x
    > a) & (x < b), {c})`` (the answer of the solver, the condition being
    the bounds); ``None`` for another kind of set."""
    if isinstance(found, Intersection) and len(found.args) == 2:
        finite = [part for part in found.args if isinstance(part, FiniteSet)]
        intervals = [part for part in found.args if isinstance(part, Interval)]
        if len(finite) == 1 and len(intervals) == 1:
            found = finite[0]
    if isinstance(found, ConditionSet) and isinstance(found.base_set, FiniteSet) \
            and not found.condition.free_symbols - {x} - a.free_symbols - b.free_symbols:
        found = found.base_set
    if isinstance(found, Union):
        # the union of the intersections of the points with the range (the
        # regression: 1/((x - a)*(x - b)) over (0, 1) was left unevaluated)
        points: list[Expr] = []
        for part in found.args:
            if not isinstance(part, Set):
                return None
            listed = _candidate_points(part, x, a, b)
            if listed is None:
                return None
            points.extend(p for p in listed if p not in points)
        return points
    if isinstance(found, FiniteSet):
        return [_real_odd_roots(as_expr(p)) for p in found.args]
    if found is EmptySet:
        return []
    return None


def _real_odd_roots(p: Expr) -> Expr:
    """The point ``p`` with every real odd root ``Abs(w)**(1/n)*sign(w)``
    of a perfect power ``w = v**n`` written ``v``, for real parameters:
    the real zero ``c`` of ``x**3 - c**3`` written so has relations with
    the endpoints which the solvers cannot decide.

    >>> from sympy_extras.core import symbols, Rational
    >>> from sympy_extras.functions import abs_, sign
    >>> from sympy_extras.integrals.definite import _real_odd_roots
    >>> c = symbols('c')
    >>> _real_odd_roots(abs_(8*c**3)**Rational(1, 3)*sign(8*c**3))
    2*c
    >>> _real_odd_roots(abs_(-c**3)**Rational(1, 3)*sign(-c**3))
    -c
    >>> _real_odd_roots(abs_(c + 1)**Rational(1, 3)*sign(c + 1))
    Abs(c + 1)**(1/3)*sign(c + 1)
    """
    def root(node: Basic) -> Optional[Expr]:
        if not isinstance(node, Mul):
            return None
        factors = [as_expr(f) for f in node.args]
        signs = [f for f in factors if isinstance(f, Sign)]
        for power in factors:
            if not (isinstance(power, Pow) and isinstance(power.base, Abs) and isinstance(power.exp, Rational)
                    and power.exp.p == 1 and power.exp.q % 2 == 1):
                continue
            w = as_expr(power.base.args[0])
            n = int(power.exp.q)
            if sign(w) not in signs:
                continue
            for s in (S.One, S.NegativeOne):
                v = powdenest((s * w)**_rational(1, n), force=True)
                if not v.has(Pow) or all(isinstance(e, Pow) and e.exp.is_integer for e in v.atoms(Pow)):
                    rest = [f for f in factors if f != power and f != sign(w)]
                    return mul(s * v, *rest)
        return None

    def rewrite(node: Basic) -> Basic:
        found = root(node)
        return node if found is None else found

    return as_expr(p.replace(lambda node: root(node) is not None, rewrite))


def _positions(s: Expr, a: Expr, b: Expr, singular: bool) -> list[tuple[Boolean, int]]:
    """The positions of the point ``s`` against the endpoints of ``(a, b)``
    as conditions on the parameters, each with ``0`` for outside the
    range, ``1`` for between ``a < s < b`` and ``-1`` for between ``b < s
    < a``: below both endpoints, above both, and between them in either
    order; the bounds on the outside are strict for a ``singular`` point
    (the integral at ``s = a`` is another question, for the other branch
    of the ``Piecewise``) and closed for a kink, where the integrand is
    continuous. An infinite endpoint drops the positions beyond it.

    >>> from sympy_extras.core import symbols, oo, Integer
    >>> from sympy_extras.integrals.definite import _positions
    >>> a, b = symbols('a b')
    >>> _positions(Integer(0), a, b, True)
    [((a > 0) & (b > 0), 0), ((a < 0) & (b < 0), 0), ((b > 0) & (a < 0), 1), ((a > 0) & (b < 0), -1)]
    >>> _positions(Integer(0), a, oo, False)
    [(a >= 0, 0), (a < 0, 1)]
    """
    below_a = lt(s, a) if singular else le(s, a)
    below_b = lt(s, b) if singular else le(s, b)
    above_a = gt(s, a) if singular else ge(s, a)
    above_b = gt(s, b) if singular else ge(s, b)
    found: list[tuple[Boolean, int]] = []
    if a != -oo:
        found.append((and_(below_a, true if b == oo else below_b), 0))
    if b != oo:
        found.append((and_(true if a == -oo else above_a, above_b), 0))
    found.append((and_(true if a == -oo else gt(s, a), true if b == oo else lt(s, b)), 1))
    if a != -oo and b != oo:
        found.append((and_(lt(s, a), gt(s, b)), -1))
    return [(_canonical_relations(condition), position) for condition, position in found]


def _strict_where_undefined(condition: Boolean, value: Expr) -> Boolean:
    """The condition with a closed bound ``u <= v`` on one parameter,
    linear, made strict when ``value`` is not defined at ``u = v``: the
    value of ``log(Abs(x - c))`` over ``(0, 1)`` for ``c <= 0`` has
    ``c*log(-c)``, which is ``nan`` at ``c = 0``.

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.functions import log
    >>> from sympy_extras.integrals.definite import _strict_where_undefined
    >>> c = symbols('c')
    >>> _strict_where_undefined(c <= 0, c*log(-c) - 1)
    c < 0
    >>> _strict_where_undefined(c <= 0, c - 1)
    c <= 0
    """
    replacements: dict[Basic, Basic] = {}
    for atom in condition.atoms(Le, Ge):
        if not isinstance(atom, Relational):
            continue
        lhs, rhs = atom.lhs, atom.rhs
        difference = lhs - rhs
        symbols = _sorted_symbols(difference.free_symbols)
        if len(symbols) != 1:
            continue
        coefficients = _coefficients(difference, symbols[0])
        if coefficients is None or len(coefficients) != 2:
            continue
        point = -coefficients[0] / coefficients[1]
        at = _subs(value, symbols[0], point)
        if at.has(nan, zoo, oo, -oo):
            replacements[atom] = lt(lhs, rhs) if isinstance(atom, Le) else gt(lhs, rhs)
    return to_boolean(condition._xreplace_evaluated(replacements)) if replacements else condition


def _real_for_real_parameters(p: Expr) -> bool:
    """Whether ``p`` is real once its symbols are taken real: a plain
    ``c`` is, ``I*y`` is not, and ``sqrt(-c)`` is not known to be.

    >>> from sympy_extras.core import symbols, I, sqrt
    >>> from sympy_extras.integrals.definite import _real_for_real_parameters
    >>> c, y = symbols('c y')
    >>> _real_for_real_parameters(c), _real_for_real_parameters(I*y), _real_for_real_parameters(sqrt(-c))
    (True, False, False)
    """
    replacement: dict[Symbol, Expr] = {s: _fresh(s.name, real=True) for s in _sorted_symbols(p.free_symbols)}
    return _xr(p, replacement).is_extended_real is True


def _canonical_relations(condition: Boolean) -> Boolean:
    """The relations of a condition in their canonical form (``a >= 0``
    for ``0 <= a``), the form ``Piecewise`` gives its conditions: the
    conditions of the cases and their disjunction must stay the same
    expressions (the bug: the value of ``Abs(x)`` over ``(a, b)`` came
    back as a ``Piecewise`` nested in the ``Piecewise`` on the condition,
    the disjunction written with ``0 <= a``)."""
    def canonical(node: Basic) -> Basic:
        return node.canonical if isinstance(node, Relational) else node

    return to_boolean(condition.replace(lambda node: isinstance(node, Relational), canonical))


def _from_integrate(value: Expr, x: Symbol) -> Optional[ConditionalValue]:
    """A result of ``integrate`` as a conditional value: a ``Piecewise``
    whose last branch is the unevaluated integral gives the first branch
    with its condition."""
    if is_unevaluated(value):
        if isinstance(value, Piecewise) and len(value.args) == 2:
            first, condition = _pairs(value)[0]
            if not is_unevaluated(first) and not condition.has(x):
                return ConditionalValue(first, condition)
        return None
    if value.has(x):
        return None
    return ConditionalValue(value)


def _forms(g: Expr, t: Symbol, assumptions: Assumptions) -> list[Expr]:
    """The integrand as it is and in the forms the Mellin table reads:
    expanded, denested, products and powers of ``sin`` and ``cos``
    written as sums (``sin(x)**2`` is ``1/2 - cos(2 x)/2``), hyperbolic
    functions as exponentials, inverse hyperbolic functions as
    logarithms, orthogonal polynomials expanded."""
    forms = [g]

    def add_form(e: Expr) -> None:
        if e not in forms:
            forms.append(e)

    # the exponentials kept whole: expand writes exp(-(a + s)*t) as
    # exp(-a*t)*exp(-s*t), two kernels for Parseval's formula with the
    # condition a > 0 on each where a + s > 0 is the one; and a sum of
    # products of exponentials (the census's three-exponential Laplace
    # transform) has each term's exponentials combined
    add_form(_combined_exponentials(expand(g, power_exp=False)))
    add_form(_denest(g, assumptions))
    if g.has(Sin, Cos):
        add_form(expand(TR8(g)))
    if g.has(*_HYPERBOLIC):
        add_form(expand(as_expr(g.rewrite(Exp))))
    if g.has(Exp):
        # exp(-u)/(1 - exp(-u)) is 1/(exp(u) - 1), a kernel of the table
        add_form(cancel(g))
    if g.has(*_INVERSE_HYPERBOLIC):
        rewritten = expand(as_expr(g.rewrite(Log)))
        add_form(rewritten)
        add_form(expand_log(rewritten, force=True))
    if g.has(OrthogonalPolynomial):
        add_form(expand(expand_func(g)))
        # a Laguerre polynomial of symbolic degree by Kummer's transformation,
        # laguerre(n, u) = 1F1(-n; 1; u) = exp(u)*1F1(n + 1; 1; -u): the
        # confluent kernel of the Mellin table with a negative argument
        # (the Laplace transform of laguerre(n, t) is (s - 1)**n/s**(n + 1))
        add_form(_combined_exponentials(as_expr(g.replace(
            lambda node: isinstance(node, Laguerre) and not as_expr(node.args[0]).is_Integer,
            lambda node: exp(as_expr(node.args[1])) * hyper([as_expr(node.args[0]) + 1], [1],
                                                             -as_expr(node.args[1]))))))
        # and a Hermite polynomial of symbolic degree 2n or 2n + 1, by
        # H_{2n}(u) = (-1)**n (2n)!/n! 1F1(-n; 1/2; u**2) and
        # H_{2n+1}(u) = (-1)**n (2n + 1)!/n! 2u 1F1(-n; 3/2; u**2), then Kummer
        if g.has(Hermite):
            add_form(_combined_exponentials(as_expr(g.replace(
                lambda node: isinstance(node, Hermite) and not as_expr(node.args[0]).is_Integer,
                lambda node: _hermite_as_kummer(node, assumptions)))))
    return forms


def _simplest_root(e: Expr) -> Expr:
    """A number ``e`` with a root of higher order written through the
    square roots it contains when it is such a combination, checked to
    50 digits: ``(sqrt(5) - 2)**(1/3)`` is ``(sqrt(5) - 1)/2`` (the end of
    the range of ``log(1 - z**(1/3) - z**(2/3))`` in ``u = z**(1/3)``)."""
    if not e.is_number or not e.has(Pow):
        return e
    square_roots = [as_expr(node) for node in e.atoms(Pow)
                    if isinstance(node, Pow) and node.exp == S.Half and node.base.is_Integer]
    higher = [as_expr(node) for node in e.atoms(Pow)
              if isinstance(node, Pow) and isinstance(node.exp, Rational) and node.exp.q > 2]
    if not higher or not square_roots:
        return e
    try:
        candidate = nsimplify(e, tuple(square_roots), tolerance=1e-30, full=True)
    except (TypeError, ValueError, RecursionError, ZeroDivisionError):
        return e
    if candidate.has(*higher) or any(isinstance(node, Pow) and isinstance(node.exp, Rational) and node.exp.q > 2
                                     for node in candidate.atoms(Pow)):
        return e
    difference = reliable_value(candidate - e, 50)
    if difference is None:
        return e
    with mpmath.workdps(60):
        number = numeric_value(difference, mpmath.mp.prec)
    if number is not None and abs(complex(number)) < 1e-45:
        return candidate
    return e


def _logarithm_inside(f: Expr, x: Symbol) -> bool:
    """Whether ``log(x)`` (or ``log(1/x)``) sits inside another function
    or a denominator of ``f``: ``log(log(1/x))``, ``1/log(x)`` (not
    ``log(1 + x)``, nor ``log(x)**2``, which the Mellin method reads)."""
    logarithms = (log(x), log(1 / x))
    for node in f.atoms(Application):
        if any(node.has(item) for item in logarithms) and node not in logarithms:
            return True
    for node in f.atoms(Pow):
        if not isinstance(node, Pow):
            continue
        exponent = node.exp
        if node.base in logarithms and (exponent.is_negative or exponent.is_Integer is False):
            return True
    return False


def _zeta_about_one(e: Expr, s: Symbol) -> Expr:
    """``zeta(s)`` in ``e`` replaced by its Laurent expansion about 1,
    ``1/(s - 1) + sum((-1)**n*stieltjes(n)*(s - 1)**n/n!, n >= 0)`` to the
    second order (the series of ``zeta`` at 1 is not taken): enough for a
    limit at 1 where a simple pole cancels."""
    expansion: Expr = 1 / (s - 1)
    for n in range(3):
        expansion = expansion + Integer(-1)**n * stieltjes(Integer(n)) * (s - 1)**n / factorial(Integer(n))
    return as_expr(e.replace(lambda node: isinstance(node, Zeta) and node.args == (s,), lambda node: expansion))


def _finite_at(f: Expr, x: Symbol, point: Expr) -> bool:
    """Whether ``f`` takes a finite value at ``point`` (no divergence to
    look for there): the leading term is computed at the other points."""
    try:
        value = _subs(f, x, point)
    except (TypeError, ValueError, ZeroDivisionError):
        return False
    if value.has(oo, -oo, zoo, nan):
        return False
    if value.is_number:
        try:
            number = numeric_value(value, 53)
        except (TypeError, ValueError, ZeroDivisionError, OverflowError):
            return False
        return number is not None and bool(mpmath.isfinite(number))
    return value.is_finite is True


def _flattened_powers(h: Expr, t: Symbol, lower: Expr, upper: Expr) -> Expr:
    """``(u**p)**q`` written ``u**(p*q)`` for rational ``p``, ``q`` and a
    base ``u`` positive on ``(lower, upper)``, and ``(-1/(t - 1))**q`` as
    ``(1 - t)**(-q)``: the identity holds for a positive base, not for
    ``((-sin(x))**a)**(1/a)`` (which is ``-sin(x)`` only for ``|a| <= 1``)."""
    facts: list[Boolean] = [t > lower] if lower not in (-oo, oo) else []
    facts += [t < upper] if upper not in (-oo, oo) else []

    def positive(u: Expr) -> bool:
        return u.is_positive is True or ask(u > 0, facts) is True

    def flatten(node: Basic) -> Basic:
        if not isinstance(node, Pow):
            return node
        base, exponent = node.base, node.exp
        if isinstance(base, Mul) and len(base.args) == 2 and base.args[0] == S.NegativeOne:
            inner = base.args[1]
            if isinstance(inner, Pow) and inner.exp.is_Integer and int(as_expr(inner.exp)) % 2 == 1 \
                    and isinstance(inner.base, Add):
                base = expand(-inner.base)**inner.exp
        if isinstance(base, Pow) and isinstance(exponent, Rational) and isinstance(base.exp, Rational) \
                and positive(base.base):
            return base.base**(base.exp * exponent)
        return node

    return as_expr(h.replace(lambda node: isinstance(node, Pow) and isinstance(node.base, (Pow, Mul)), flatten))


def _algebraic(h: Expr, t: Symbol) -> bool:
    """Whether ``h`` is a rational function of ``t`` and of powers of
    polynomials in ``t`` with rational exponents."""
    replacement: dict[Basic, Basic] = {}
    placeholders: list[Symbol] = []
    for node in h.atoms(Pow):
        if not isinstance(node, Pow):
            continue
        base, exponent = node.base, node.exp
        if base.has(t) and isinstance(exponent, Rational) and exponent.q != 1:
            if not base.is_rational_function(t):
                return False
            placeholders.append(_fresh('r'))
            replacement[node] = placeholders[-1]
    return bool(as_expr(h._xreplace_evaluated(replacement)).is_rational_function(t, *placeholders))


def _tangent_of(v: Expr) -> Expr:
    """``tan(v)``, the infinities at ``+-pi/2`` (where ``tan`` gives ``zoo``)."""
    if v == pi / 2:
        return oo
    if v == -pi / 2:
        return -oo
    return tan(v)


def _coefficients(q: Expr, x: Symbol) -> Optional[list[Expr]]:
    """The coefficients of ``q`` as a polynomial in ``x``, by the power
    (any expressions free of ``x``: the typed ``Poly`` refuses irrational
    ones), the leading one nonzero; ``None`` when ``q`` is not a
    polynomial in ``x``."""
    if not q.is_polynomial(x):
        return None
    terms: dict[int, Expr] = {}
    for term in Add.make_args(expand(q)):
        c, rest = term.as_independent(x, as_Add=False)
        k: int
        if rest == 1:
            k = 0
        elif rest == x:
            k = 1
        elif isinstance(rest, Pow) and rest.base == x and isinstance(rest.exp, Integer) and int(rest.exp) > 0:
            k = int(rest.exp)
        else:
            return None
        terms[k] = terms.get(k, S.Zero) + c
    nonzero = [k for k, c in terms.items() if c != 0]
    top = max(nonzero) if nonzero else 0
    return [terms.get(k, S.Zero) for k in range(top + 1)]


def _radical_bases(f: Expr, x: Symbol) -> list[Expr]:
    """The distinct bases of the powers of ``f`` with a rational exponent
    which is not an integer, depending on ``x``."""
    bases: list[Expr] = []
    for node in f.atoms(Pow):
        if not isinstance(node, Pow):
            continue
        base, exponent = node.base, node.exp
        if base.has(x) and isinstance(exponent, Rational) and exponent.q != 1 and base not in bases:
            bases.append(base)
    return bases


def _radical_base(f: Expr, x: Symbol) -> Optional[tuple[Expr, int]]:
    """``(q, n)`` for the substitution ``u = q**(1/n)``: the one radical
    base of ``f`` which is a polynomial in ``x`` of degree at most two,
    with ``n`` the common denominator of its exponents."""
    bases = _radical_bases(f, x)
    polynomial = [base for base in bases if base.is_polynomial(x)]
    if len(polynomial) != 1:
        return None
    q = polynomial[0]
    coefficients = _coefficients(q, x)
    if coefficients is None or len(coefficients) > 3:
        return None
    n = 1
    for node in f.atoms(Pow):
        if isinstance(node, Pow) and node.base == q and isinstance(node.exp, Rational):
            n = lcm(n, int(node.exp.q))
    return (q, n) if n > 1 else None


def _denest(g: Expr, assumptions: Assumptions) -> Expr:
    """``powdenest(g, force=True)`` done honestly: the forced denesting
    takes every symbol positive (``sqrt(c**2)`` becomes ``c``), so a
    parameter known negative is first written ``-d`` with ``d`` positive,
    and a parameter of unknown sign blocks the forced form (the bug: the
    area of the disc ``x**2 + y**2 < c**2`` came out as 0 for ``c < 0``)."""
    replacement: dict[Symbol, Expr] = {}
    back: dict[Symbol, Expr] = {}
    for p in _sorted_symbols(g.free_symbols):
        if p.is_positive:
            continue
        if ask(p > 0, assumptions) is True:
            d = _fresh(p.name, positive=True)
            replacement[p] = d
            back[d] = p
        elif ask(p < 0, assumptions) is True:
            d = _fresh(p.name, positive=True)
            replacement[p] = -d
            back[d] = -p
        else:
            return powdenest(g)
    denested = powdenest(_xr(g, replacement), force=True)
    return _xr(denested, back)


def _arcsine_substitution(f: Expr, x: Symbol, u: Symbol) -> Expr:
    """``f`` at ``x = asin(sqrt(u))`` with the trigonometric functions of
    ``x`` written in ``u`` (``sin(x)`` is ``sqrt(u)``, ``cos(x)`` is
    ``sqrt(1 - u)``): the compositions ``sin(asin(y))`` and the like,
    which SymPy evaluates, stay unevaluated on the core.

    >>> from sympy_extras.core import Symbol, Rational
    >>> from sympy_extras.functions import sin, cos
    >>> from sympy_extras.integrals.definite import _arcsine_substitution
    >>> x, u = Symbol('x'), Symbol('u', positive=True)
    >>> _arcsine_substitution(sin(x)**Rational(1, 3)*cos(x)**Rational(1, 3), x, u)
    u**(1/6)*(1 - u)**(1/6)
    """
    s, c = sqrt(u), sqrt(1 - u)
    rule: dict[Basic, Basic] = {Sin(x): s, Cos(x): c, Tan(x): s / c, Cot(x): c / s, Sec(x): 1 / c, Csc(x): 1 / s}
    return _subs(as_expr(f._xreplace_evaluated(rule)), x, asin(s))


def _shift(f: Expr, x: Symbol, t: Symbol, a: Expr) -> Expr:
    return _subs(f, x, t + a)


def _quarter(e: Expr) -> Optional[int]:
    """``k`` when ``e == k*pi/2``."""
    k = 2 * e / pi
    if isinstance(k, Integer):
        return int(k)
    return None


def _exponential_generator(f: Expr, x: Symbol, negative: bool) -> Optional[Expr]:
    """``c`` such that every ``exp(c_i x)`` of ``f`` has ``c_i`` an integer
    multiple of ``c`` (the rational ``c_i`` reduced to their common
    fraction), with the sign of the ``c_i`` when they share one and
    ``negative`` deciding otherwise; ``None`` without such exponentials."""
    coefficients: list[Rational] = []
    symbolic = 0
    for node in f.atoms(Exp):
        argument = as_expr(node.args[0])
        if argument.has(Exp) and any(inner.has(x) for inner in argument.atoms(Exp)):
            # exp(-a*exp(-u)): the inner exponential is the one substituted,
            # the outer follows (exp(-a*t) for t = exp(-u))
            continue
        found = monomial(argument, x)
        if found is None or found[1] != 1:
            return None
        if isinstance(found[0], Rational):
            coefficients.append(found[0])
        else:
            symbolic += 1
    if not coefficients and not symbolic:
        return None
    numerator = 0
    denominator = 1
    for c in coefficients:
        numerator = gcd(numerator, int(c.p))
        denominator = lcm(denominator, int(c.q))
    generator: Expr = _rational(numerator, denominator) if coefficients else S.One
    if coefficients and all(k.is_negative for k in coefficients):
        return -generator
    if coefficients and all(k.is_positive for k in coefficients):
        return generator
    return -generator if negative else generator


def _powers_of_logarithms(g: Expr, u: Symbol) -> Expr:
    """``exp(c*log(u) + rest)`` written ``u**c*exp(rest)`` for ``c`` free of
    ``u``, which the evaluation leaves as it is: ``exp(-v*x)`` at ``x =
    -log(u)`` is ``u**v``."""
    logarithm = log(u)

    def rewrite(node: Basic) -> Basic:
        argument = as_expr(node.args[0])
        power: Expr = S.One
        rest: list[Expr] = []
        for term in Add.make_args(argument):
            coefficient, factor_ = term.as_independent(logarithm, as_Add=False)
            if factor_ == logarithm and not coefficient.has(u):
                power = power * u**coefficient
            else:
                rest.append(term)
        return power * exp(add(*rest))

    return as_expr(g.replace(lambda node: isinstance(node, Exp) and node.has(logarithm), rewrite))


def _log_powers_only(g: Expr, u: Symbol) -> bool:
    """Whether ``log(u)`` occurs only as ``log(u)**n``."""
    logarithm = log(u)
    for node in g.atoms(Log):
        if node.has(u) and node != logarithm:
            return False
    return True


def _trigonometric_zeros(u: Expr, x: Symbol, a: Expr, b: Expr) -> Optional[list[Expr]]:
    """The zeros of ``sin(w*x + c)`` or ``cos(w*x + c)`` with numeric
    ``w`` and ``c`` inside a bounded numeric range ``(a, b)``, enumerated
    (``solve`` spends seconds on them, and answers with a ConditionSet
    under an assumption on another symbol); ``None`` for any other
    ``u`` or range."""
    if not isinstance(u, (Sin, Cos)) or not (a.is_number and b.is_number) or a in (-oo, oo) or b in (-oo, oo):
        return None
    argument = as_expr(u.args[0])
    coefficients = _coefficients(argument, x)
    if coefficients is None or len(coefficients) != 2:
        return None
    w, c = coefficients[1], coefficients[0]
    if not (w.is_number and c.is_number and w.is_real and c.is_real) or w == 0:
        return None
    # the zeros are at w x + c = k pi, or k pi + pi/2 for the cosine
    shift = S.Zero if isinstance(u, Sin) else pi / 2
    try:
        ends = sorted([to_float((w * a + c - shift) / pi), to_float((w * b + c - shift) / pi)])
    except (TypeError, ValueError):
        return None
    if ends[1] - ends[0] > 1000:
        return None
    zeros: list[Expr] = []
    for k in range(int(ends[0]) - 1, int(ends[1]) + 2):
        point = (k * pi + shift - c) / w
        above, below = (point - a).is_positive, (b - point).is_positive
        if above is None or below is None:
            return None
        if above and below:
            zeros.append(point)
    return sorted(zeros, key=to_float)


def _hermite_as_kummer(node: Basic, assumptions: Assumptions) -> Basic:
    """``hermite(2*n, u)`` or ``hermite(2*n + 1, u)`` with ``n`` an integer
    under the assumptions, through Kummer's function of a negative
    argument: ``exp(u**2)`` times ``1F1(n + 1/2; 1/2; -u**2)`` or
    ``2*u*1F1(n + 3/2; 3/2; -u**2)`` with the factor ``(-1)**n*(2n)!/n!``
    or ``(-1)**n*(2n + 1)!/n!``. The node itself when the parity of the
    degree is not known (the bug: ``hermite(n, u)`` was read as the even
    degree ``2*(n/2)``, and Maxima's ``specint`` 105 came out wrong)."""
    degree_, u = as_expr(node.args[0]), as_expr(node.args[1])
    for shift in (S.Zero, S.One):
        n = (degree_ - shift) / 2
        if n.is_integer is not True and ask(element(n, Integers), assumptions) is not True:
            continue
        if shift == 0:
            return (S.NegativeOne**n * gamma(2 * n + 1) / gamma(n + 1) * exp(u**2)
                    * hyper([n + S.Half], [S.Half], -u**2))
        return (S.NegativeOne**n * gamma(2 * n + 2) / gamma(n + 1) * 2 * u * exp(u**2)
                * hyper([n + _rational(3, 2)], [_rational(3, 2)], -u**2))
    return node


def _combined_exponentials(f: Expr, x: Optional[Symbol] = None) -> Expr:
    """The exponential factors of each product combined into one,
    ``exp(-a*t)*exp(-s*t)`` into ``exp(-(a + s)*t)``: the identity
    ``exp(u)*exp(v) = exp(u + v)`` holds everywhere, and the methods read
    one exponential (the bug: the product went through Parseval's formula
    as two kernels, with the condition ``a > 0`` on each instead of
    ``a + s > 0`` on the sum). With ``x`` the exponentials free of it are
    left alone: a constant ``exp(a*k)`` is taken out of the integral, and
    the table's patterns do not carry it."""
    def combine(node: Basic) -> Basic:
        exponentials = [as_expr(part) for part in node.args
                        if isinstance(part, Exp) and (x is None or part.has(x))]
        if len(exponentials) < 2:
            return node
        rest = [as_expr(part) for part in node.args if as_expr(part) not in exponentials]
        return mul(*rest) * exp(add(*[as_expr(e.args[0]) for e in exponentials]))

    if not f.has(Exp):
        return f
    return as_expr(f.replace(lambda n: isinstance(n, Mul), combine))


def _sums_to_products(f: Expr, x: Symbol) -> Expr:
    """A quotient of sums of sines and cosines of multiples of ``x`` with
    the sums written as products (the sum-to-product formulas, ``TR9``)
    and cancelled: ``(sin(19*x) + sin(20*x))/(cos(19*x) + cos(20*x))`` is
    ``tan(39*x/2)``; ``f`` itself when nothing cancels."""
    numerator, denominator = f.as_numer_denom()

    def trigonometric_sum(e: Expr) -> bool:
        if not isinstance(e, Add):
            return False
        for term in e.args:
            _, rest = as_expr(term).as_independent(x, as_Add=False)
            if not isinstance(rest, (Sin, Cos)) or not as_expr(rest.args[0]).is_polynomial(x):
                return False
        return True

    if not (trigonometric_sum(numerator) or trigonometric_sum(denominator)):
        return f
    rewritten = cancel(TR9(numerator) / TR9(denominator))
    return rewritten if _ops(rewritten) < _ops(f) else f


def _trigonometric_squares(base: Expr) -> Expr:
    """The trigonometric sums which are squares written as such, so that
    a power of the sum splits over the factors: ``c - c*cos(u)`` is
    ``2*c*sin(u/2)**2`` and ``c + c*cos(u)`` is ``2*c*cos(u/2)**2`` (the
    half-angle formulas), and a sum which ``trigsimp`` turns into a product
    or a power is taken in that form (``tan(u)**2 + 1`` is ``cos(u)**(-2)``,
    the cardioid's ``a**2*(1 - cos(t))**2 + a**2*sin(t)**2`` is
    ``2*a**2*(1 - cos(t))``)."""
    def half_angle(node: Basic) -> Basic:
        if not isinstance(node, Add) or len(node.args) != 2:
            return node
        terms = [as_expr(t) for t in node.args]
        for constant, other in ((terms[0], terms[1]), (terms[1], terms[0])):
            if _has_trigonometric(constant):
                continue
            cosines = [part for part in Mul.make_args(other) if isinstance(part, Cos)]
            if len(cosines) != 1:
                continue
            coefficient = other / cosines[0]
            u = as_expr(cosines[0].args[0])
            if coefficient == constant:
                return 2 * constant * cos(u / 2)**2
            if coefficient == -constant:
                return 2 * constant * sin(u / 2)**2
        return node

    def squares(node: Expr) -> Expr:
        # the outer sums first, the constants factored out or not (the
        # cardioid's a**2*((1 - cos(t))**2 + sin(t)**2) as well as the
        # expanded sum): trigsimp gives 2 - 2*cos(t), a square by the
        # half-angle formula, only while its 1 - cos(t) is not yet written
        # as one
        if isinstance(node, Add) and _has_trigonometric(node):
            found = as_expr(trigsimp(node).replace(lambda n: isinstance(n, Add), half_angle))
            if not isinstance(found, Add):
                return found
        if not node.args:
            return node
        return as_expr(node._evaluated(tuple(squares(as_expr(argument)) for argument in node.args)))

    return as_expr(squares(base).replace(lambda n: isinstance(n, Add), half_angle))


#: the largest base of a power factored for its signs
_FACTOR_LIMIT: Final = 200


def _factored(base: Expr, x: Symbol) -> Optional[_Factored]:
    """``base`` as a constant times powers of factors depending on ``x``
    with integer or rational exponents (a radical factor, ``sqrt(1 - v)``
    in ``log(sqrt(1 - v))``, keeps its base and its exponent): the
    trigonometric squares written as such, the numerator and the
    denominator factored over the rationals; and whether a square
    appeared (a factor of integer multiplicity two or more, a
    trigonometric square). Each factor is continuous, so that it keeps
    one sign between its zeros; ``None`` when a factor has poles
    (``tan``, a negative power of ``x`` inside it) or the base is too
    large to factor."""
    if _ops(base) > _FACTOR_LIMIT:
        return None
    g = _trigonometric_squares(base) if _has_trigonometric(base) else base
    squares = g != base
    try:
        factored = attempt(lambda: factor(together(g)), settings.timeout, native=True)
    except (BasePolynomialError, ValueError, TypeError):
        return None
    if factored is None:
        return None
    constant, rest = factored.as_independent(x, as_Add=False)
    factors: list[tuple[Expr, Rational]] = []
    for part in Mul.make_args(rest):
        b_, e_ = part.as_base_exp()
        if not isinstance(e_, Rational):
            # a symbolic power, continuous where it is real, as one factor
            b_, e_ = part, S.One
        if b_.has(Tan, Cot, Sec, Csc, Piecewise, Floor, Ceiling, Frac, Sign, Heaviside) or any(
                isinstance(node, Pow) and node.base.has(x) and node.exp.is_negative
                for node in b_.atoms(Pow)):
            return None
        if not isinstance(e_, Rational):
            return None
        factors.append((b_, e_))
        squares = squares or (isinstance(e_, Integer) and abs(int(e_)) >= 2)
    return constant, factors, squares


def _even(e: Rational) -> bool:
    """Whether a multiplicity is an even integer (the square roots of
    squares are rewritten before the other methods)."""
    return isinstance(e, Integer) and int(e) % 2 == 0


def _signed_nodes(f: Expr, x: Symbol) -> list[Expr]:
    """The powers with a non-integer exponent and the logarithms whose
    argument depends on ``x``, the candidates of :meth:`_Integrator._signed`."""
    found: list[Expr] = []
    for node in sorted(f.atoms(Pow, Log), key=str):
        if isinstance(node, Pow):
            if isinstance(node.exp, Integer) or not node.base.has(x):
                continue
        elif not node.args[0].has(x):
            continue
        found.append(as_expr(node))
    return found
