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
candidate checks the unevaluated integral (:class:`.integral.Integral`)
is returned.

The order of the methods:

1. rational functions by Hermite reduction and the Lazard–Rioboo–Trager
   logarithmic part (:mod:`.risch.rationaltools`);
2. radicals of a quadratic, `x^n Q^{m/2}` (:mod:`.radicals`);
3. powers times exponentials of powers, `x^{v-1} e^{a x^n + b}`, by the
   incomplete gamma function and its elementary and error-function
   cases, polynomials times `e^{a x^2 + b x + c}`, and rational functions
   and binomials of an exponential (:mod:`.exponential`);
4. products of powers of trigonometric and hyperbolic functions, their
   rational functions and their products with polynomials, exponentials
   and inverse functions (:mod:`.trigonometric`);
5. the transcendental Risch algorithm (:mod:`.risch`), which also proves
   non-elementarity;
6. Trager's algorithm for one square root or one root of any index of a
   rational function, and on the curve of several or nested radicals
   (:mod:`.trager`, :mod:`.curves`), and right after it the Gauss
   hypergeometric antiderivative of a binomial differential
   `x^m (a + b x^n)^p` whose integral Chebyshev's criterion proves
   non-elementary (:mod:`.algebraic`): it cannot preempt an elementary
   answer, and saves the heuristics their budget;
7. the methods above again on the canonical forms of the integrand and
   on the integrands of its substitutions (:mod:`.rewriting`), before
   the heuristics: `x = t**3` makes an integrand in `x**(1/3)`, `exp(x)`
   and `log(x)` a tower the Risch algorithm settles at once;
8. the heuristic Risch integrator (:mod:`.heurisch`), on the integrand
   and then on its canonical forms (on the exponentials of hyperbolic
   functions already in 7, after the substitutions);
9. SymPy's heuristic Risch integrator, the rule-based
   ``manualintegrate`` (the native port :mod:`.manualintegrate`, in the
   place SymPy's had), SymPy's Meijer G-function route and its
   ``integrate``.

The module works on the objects of :mod:`sympy_extras.core` and imports
no SymPy: the methods 1 to 8, ``manualintegrate``, the checks (the native
``cancel``, ``simplify``, the structure theorem of
:mod:`sympy_extras.simplify.structure` and the reliable values of
:mod:`sympy_extras._numeric`) and the cases of the special values of the
parameters (:mod:`sympy_extras._special_values`) are native. SymPy's
other routes of 9 have no native counterpart: they are
reached last, through the lazily imported ``*_of_core`` functions of the
face :mod:`.indefinite_sympy`, so that SymPy is loaded only when every
native method has failed. The functions on SymPy objects are in
:mod:`.indefinite_sympy`.

Examples
========

>>> from sympy_extras.core import symbols
>>> from sympy_extras.functions import sqrt, exp
>>> from sympy_extras.integrals.indefinite import indefinite_integral
>>> x = symbols('x')
>>> indefinite_integral(sqrt(1 - x**2), x)
x*sqrt(1 - x**2)/2 + asin(x)/2
>>> indefinite_integral(x/(x**2 + 1), x)
log(x**2 + 1)/2
>>> indefinite_integral(exp(-x**2), x)
sqrt(pi)*erf(x)/2
"""
from __future__ import annotations

import itertools
import random
from typing import Callable, Final, Optional

import mpmath

from sympy_extras._numeric import reliable_value
from sympy_extras._special_values import Point, defined_problem, with_special_values
from sympy_extras._timeout import attempt
from sympy_extras.assumptions.ask import Assumptions
from sympy_extras.core.add import Add, add
from sympy_extras.core.basic import Basic
from sympy_extras.core.evalf import numeric_value
from sympy_extras.core.expr import Expr, ExprLike, as_expr, to_expr
from sympy_extras.core.function import Derivative
from sympy_extras.core.logic import Boolean, BooleanFalse, and_, to_boolean, true
from sympy_extras.core.numbers import I, NegativeInfinity, Rational, nan, oo, zoo
from sympy_extras.core.power import Pow
from sympy_extras.core.relational import gt
from sympy_extras.core.singleton import S
from sympy_extras.core.symbol import Dummy, Symbol
from sympy_extras.functions.elementary.complexes import PolarLift, Sign, im
from sympy_extras.functions.elementary.exponential import ExpPolar, Log, log
from sympy_extras.functions.elementary.hyperbolic import (Acosh, Acoth, Acsch, Asech, Asinh, Atanh, Cosh, Coth, Csch,
                                                          Sech, Sinh, Tanh)
from sympy_extras.functions.elementary.piecewise import piecewise
from sympy_extras.functions.elementary.trigonometric import atan
from sympy_extras.functions.special.delta_functions import DiracDelta
from sympy_extras.polys.polytools import cancel
from sympy_extras.settings import settings
from sympy_extras.simplify.simplify import logcombine, simplify

from .conditions import sample_values
from .integral import Integral, is_unevaluated
from .rewriting import Substitution

__all__ = ['indefinite_integral', 'verified_antiderivative', 'is_antiderivative', 'real_form', 'conjugate_logarithms',
           'closed_antiderivative', 'METHODS', 'TYPED', 'SYMPY_METHODS']

Method = Callable[[Expr, Symbol, Assumptions], Optional[Expr]]

_COUNTER: Final = itertools.count()


def _fresh(name: str, **assumptions: bool) -> Dummy:
    """A dummy with a name of its own: the typed polynomials map their
    symbols by name, so a dummy ``x`` beside the variable ``x`` would be
    taken for it."""
    return Dummy('%s_indefinite_%d' % (name, next(_COUNTER)), **assumptions)


def _budget() -> Optional[float]:
    """A quarter of the limit for a typed method (they answer or decline
    fast) and half for SymPy's routes, which need it: under a 20 s limit
    the elliptic and Piecewise answers of ``integrate`` missed a quarter."""
    return None if settings.timeout is None else settings.timeout / 4


def _long_budget() -> Optional[float]:
    return None if settings.timeout is None else settings.timeout / 2


def _replaced(e: Expr, rule: dict[Basic, Basic]) -> Expr:
    """``e`` with the replacements of ``rule``, evaluated (the core's
    ``xreplace`` builds the nodes as they are)."""
    return as_expr(e._xreplace_evaluated(rule))


def is_antiderivative(F: ExprLike, f: ExprLike, x: Symbol, assumptions: Assumptions = None) -> Optional[bool]:
    """Whether ``F`` is an antiderivative of ``f``: ``True`` when ``F' - f``
    cancels or simplifies to zero, or vanishes numerically at random
    points satisfying the assumptions; ``False`` when it does not vanish
    there; ``None`` when nothing could be decided (an ``F`` which is not
    an expression in ``x``, an unevaluated ``Integral``).

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.functions import log, sin
    >>> from sympy_extras.integrals.indefinite import is_antiderivative
    >>> x = symbols('x')
    >>> is_antiderivative(log(x**2 + 1)/2, x/(x**2 + 1), x)
    True
    >>> is_antiderivative(sin(x), sin(x), x)
    False
    """
    F_, f_ = to_expr(F), to_expr(f)
    if is_unevaluated(F_) or F_.has(nan, zoo):
        return None
    difference = _almost_everywhere(F_.diff(x)) - f_
    reduced = attempt(lambda: cancel(difference), _budget(), native=True)
    if reduced is not None and reduced == S.Zero:
        return True
    # exactly, by the structure theorem: a proof where the heuristics of
    # simplify and the sample points below are evidence
    if attempt(lambda: _exactly_zero(difference, assumptions), _EXACT_SECONDS, native=True) is True:
        return True
    simpler = attempt(lambda: simplify(reduced if reduced is not None else difference), _budget(), native=True)
    if simpler is not None and simpler == S.Zero:
        return True
    # numerically, at points of both signs and several scales where the
    # integrand is real: the census of SymPy's mistakes found nine
    # antiderivatives right for x > 0 and wrong for x < 0 (polar
    # incomplete gammas, -asinh(1/x), Bessel forms), and a complex
    # integrand (1/sqrt(1 - x**2) beyond 1, log(x) asin(x) for x < 0) is
    # met with branches the two sides need not share. A candidate still
    # in complex form (the real-form rewriting failed) is trusted only
    # when the points reach both sides: a non-integer sample of the r of
    # exp(a*x**r) makes x < 0 complex, hides it from the check, and let
    # the polar incomplete gamma of the Meijer route through
    return _vanishes_on_both_sides(difference, f_, x, assumptions,
                                   both_signs=F_.has(I, ExpPolar, PolarLift))


#: the budget of the exact zero test of :func:`is_antiderivative`
_EXACT_SECONDS = 2.0


def _exactly_zero(difference: Expr, assumptions: Assumptions) -> bool:
    """Whether ``difference`` is zero in the tower of the structure
    theorem (:mod:`sympy_extras.simplify.structure`); ``False`` when it is
    not, or not elementary."""
    from sympy_extras.simplify.structure import ElementaryTower, NotElementary
    tower = ElementaryTower(assumptions)
    try:
        return tower.vanishes(tower.element(difference))
    except (NotElementary, ZeroDivisionError):
        return False


def _almost_everywhere(derivative: Expr) -> Expr:
    """``derivative`` away from the points where a ``sign`` factor jumps:
    its derivative (an unevaluated ``Derivative`` for an argument not
    known real, a ``DiracDelta`` otherwise) is zero there, and a numeric
    check at random points never meets the jump.

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.functions import sign
    >>> from sympy_extras.integrals.indefinite import _almost_everywhere
    >>> x = symbols('x')
    >>> _almost_everywhere((x**2*sign(x)/2).diff(x))
    x*sign(x)
    """
    def jump(node: Basic) -> bool:
        return isinstance(node, DiracDelta) or (isinstance(node, Derivative) and isinstance(node.expr, Sign))

    def zero(node: Basic) -> Basic:
        return S.Zero

    if not derivative.has(DiracDelta, Derivative):
        return derivative
    return as_expr(derivative.replace(jump, zero))


def _items(assumptions: Assumptions) -> list[Boolean]:
    if assumptions is None:
        return []
    if isinstance(assumptions, (Boolean, bool)):
        return [to_boolean(assumptions)]
    return [to_boolean(a) for a in assumptions]


#: the points where the difference of the derivatives is evaluated: both
#: signs, and the scales 1/20, 1/2, 2 and 4, so that a domain such as
#: (-1, 1) or (0, 1) gets several
_POINTS: Final[tuple[Rational, ...]] = (
    Rational(-37, 10), Rational(-19, 10), Rational(-13, 10), Rational(-2, 5), Rational(-1, 20),
    Rational(1, 20), Rational(3, 5), Rational(13, 10), Rational(19, 10), Rational(41, 10))


def _vanishes_on_both_sides(difference: Expr, f: Expr, x: Symbol, assumptions: Assumptions,
                            both_signs: bool = False) -> Optional[bool]:
    """Whether ``difference`` vanishes at the fixed real points of
    :data:`_POINTS` (the parameters sampled under the assumptions) where
    ``f`` is a finite real number; a point where ``f`` is undefined, not
    real, or excluded by the assumptions on ``x`` is skipped, and
    ``None`` is the verdict when fewer than two points remain, the
    parameters sampled three times, every sample tested: ``False`` when
    one of them sees a difference, ``True`` when one at least decides
    and none sees one (one sample can miss where the candidate goes
    wrong: the Meijer G antiderivative of ``log(Abs(c - x))`` is right for
    ``x < c`` only, and a ``c`` drawn large leaves every point there).
    With ``both_signs`` the verdict ``True`` also needs tested points of
    both signs."""
    parameters = sorted((s for s in difference.free_symbols if s != x), key=lambda s: s.name)
    facts = _items(assumptions)
    # a sample of the parameters may leave nothing to test (the integrand
    # real nowhere: c**(d*z) with c drawn negative), or nothing wrong: three samples
    samples = 3 if parameters else 1
    points: list[Expr] = list(_POINTS)
    decided = False
    for index in range(samples):
        values: dict[Symbol, Expr] = {}
        if parameters:
            seed = str(difference) if index == 0 else '%s#%d' % (difference, index)
            found = sample_values(parameters, assumptions, random.Random(seed))
            if found is None:
                break
            values = found
        verdict, signs = _tested_points(difference, f, x, values, facts, points)
        if verdict is None:
            # an integrand real on a parameter-dependent interval the fixed
            # points miss: sqrt(-2*k*s**2 + 2*h*s - a**2)
            extra = _domain_points(f, x, values, random.Random(str(difference)))
            if extra:
                verdict, signs = _tested_points(difference, f, x, values, facts, points + extra)
        if verdict is False:
            return False
        if verdict is True and (not both_signs or len(signs) == 2):
            decided = True
    return True if decided else None


def _domain_points(f: Expr, x: Symbol, values: dict[Symbol, Expr], rng: random.Random,
                   count: int = 6) -> list[Expr]:
    """Points where the radicands and the arguments of the logarithms of
    ``f`` (the parameters at ``values``) are positive, from the sampler of
    the assumptions; none when ``f`` has neither."""
    rule: dict[Basic, Basic] = {s: v for s, v in values.items()}
    conditions: list[Boolean] = []
    for node in f.atoms(Pow):
        if not isinstance(node, Pow):
            continue
        exponent = node.exp
        if isinstance(exponent, Rational) and exponent.is_integer is not True and node.base.has(x):
            conditions.append(gt(_replaced(node.base, rule), S.Zero))
    for node in f.atoms(Log):
        argument = as_expr(node.args[0])
        if argument.has(x):
            conditions.append(gt(_replaced(argument, rule), S.Zero))
    if not conditions:
        return []
    points: list[Expr] = []
    for _ in range(count):
        found = attempt(lambda: sample_values([x], conditions, rng), _budget(), native=True)
        if not found or x not in found:
            break
        point = found[x]
        if point not in points:
            points.append(point)
    return points


def _sign_of(point: Expr) -> int:
    value = numeric_value(point, 64)
    return 1 if value is not None and mpmath.re(value) > 0 else -1


def _tested_points(difference: Expr, f: Expr, x: Symbol, values: dict[Symbol, Expr],
                   facts: list[Boolean], points: list[Expr]) -> tuple[Optional[bool], set[int]]:
    """The verdict of :func:`_vanishes_on_both_sides` at one sample of the
    parameters and the given points, with the signs of the points which
    were tested."""
    tested = 0
    signs: set[int] = set()
    rule: dict[Basic, Basic] = {s: v for s, v in values.items()}
    integrand = _replaced(f, rule) if rule else f
    conditions = [to_boolean(fact._xreplace_evaluated(rule)) for fact in facts] if rule else facts
    for point in points:
        at: dict[Basic, Basic] = {x: point}
        # a point the assumptions on x exclude (x > 0) is not a test
        if any(isinstance(fact._xreplace_evaluated(at), BooleanFalse) for fact in conditions):
            continue
        at_point = _replaced(integrand, at)
        sample = attempt(lambda: at_point.evalf(30), _budget(), native=True)
        if sample is None or not sample.is_number or sample.has(nan, zoo, oo, NegativeInfinity):
            continue
        with mpmath.workprec(128):
            value = numeric_value(sample, 128)
            if value is None or not mpmath.isfinite(value):
                continue
            size, imaginary = abs(value), abs(mpmath.im(value))
            # a negligible value of the integrand is no test (its derivative
            # candidate is negligible too), and the imaginary part must be
            # negligible both in itself and next to the value: exp(a*x**r) at
            # x = -37/10 is -2e-48 + 4e-49*I, below an absolute threshold
            # while a fifth of the value, and bessely(11, x) at x < 0 is
            # 2.6e13 + 2e-15*I, negligible next to the value while not real
            tiny = mpmath.mpf(10)**-20
            if size < tiny or imaginary > tiny or imaginary > size * tiny:
                continue
        # a point where the rounding chooses the side of a branch cut is no
        # test (see reliable_form)
        at_both = dict(values)
        at_both[x] = point
        number = attempt(lambda: reliable_value(difference, 30, at_both), _budget(), native=True)
        if number is None or number.has(nan, zoo, oo, NegativeInfinity):
            continue
        with mpmath.workprec(128):
            residual = numeric_value(number, 128)
            if residual is None or not mpmath.isfinite(residual):
                continue
            if abs(residual) > max(mpmath.mpf(1), size) / mpmath.mpf(10)**15:
                return False, signs
        tested += 1
        signs.add(_sign_of(point))
    return (True if tested >= 2 else None), signs


def real_form(F: Expr, f: Expr, x: Symbol, assumptions: Assumptions = None) -> Expr:
    """``F`` rewritten without ``I`` and polar numbers when a rewriting
    which is still an antiderivative of ``f`` is found (``logcombine``,
    ``simplify``, the arctangent forms of the logarithms of conjugates);
    ``F`` itself otherwise.

    >>> from sympy_extras.core import symbols, I
    >>> from sympy_extras.functions import log
    >>> from sympy_extras.integrals.indefinite import real_form
    >>> x = symbols('x')
    >>> real_form(-I*log(x - I)/2 + I*log(x + I)/2, 1/(x**2 + 1), x)
    atan(x)
    """
    if not F.has(I, ExpPolar) or f.has(I):
        return F
    candidates: list[Callable[[], Expr]] = [
        lambda: conjugate_logarithms(F, x),
        lambda: simplify(conjugate_logarithms(F, x)),
        lambda: logcombine(F, force=True),
        lambda: simplify(F),
        lambda: simplify(logcombine(F, force=True)),
    ]
    for candidate in candidates:
        rewritten = attempt(candidate, _budget(), native=True)
        if rewritten is None or rewritten.has(I, ExpPolar):
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

    >>> from sympy_extras.core import symbols, I
    >>> from sympy_extras.functions import log
    >>> from sympy_extras.integrals.indefinite import conjugate_logarithms
    >>> x = symbols('x')
    >>> conjugate_logarithms(-I*log(x - I)/2 + I*log(x + I)/2, x)
    atan(x)
    """
    real = _fresh(x.name, real=True)
    to_real: dict[Basic, Basic] = {x: real}
    back: dict[Basic, Basic] = {real: x}
    terms: list[tuple[Expr, Expr, Expr]] = []                # (coefficient, u, v) of coefficient*log(u + I v)
    others: list[Expr] = []
    for term in Add.make_args(F):
        coefficient, rest = term.as_independent(x, as_Add=False)
        if isinstance(rest, Log) and coefficient.is_imaginary is True:
            u, v = _replaced(as_expr(rest.args[0]), to_real).as_real_imag()
            terms.append((coefficient, _replaced(u, back), _replaced(v, back)))
        else:
            others.append(term)
    used: set[int] = set()
    for i, (a, u, v) in enumerate(terms):
        if i in used or v == S.Zero or u == S.Zero:
            continue
        for j in range(i + 1, len(terms)):
            b, u2, v2 = terms[j]
            if j in used or u2 != u or v2 + v != S.Zero or a + b != S.Zero:
                continue
            # log(u + I v) - log(u - I v) = 2 I atan2(v, u) = 2 I (pi/2 - atan(u/v)) for v > 0:
            # the constant dropped, and atan(u/v) is continuous where v does not vanish
            others.append(2 * im(a) * atan(u / v))
            used.update((i, j))
            break
        else:
            others.append(a * log(u + I * v))
    if not used:
        return F
    return add(*others)


def _rational(f: Expr, x: Symbol, assumptions: Assumptions) -> Optional[Expr]:
    from .risch.rationaltools import ratint
    if f.is_rational_function(x) is not True:
        return None
    return attempt(lambda: ratint(f, x), _budget(), native=True)


def _radicals(f: Expr, x: Symbol, assumptions: Assumptions) -> Optional[Expr]:
    from .radicals import quadratic_radical_antiderivative
    return attempt(lambda: quadratic_radical_antiderivative(f, x, assumptions), _budget(), native=True)


def _exponential(f: Expr, x: Symbol, assumptions: Assumptions) -> Optional[Expr]:
    from sympy_extras.functions.elementary.exponential import Exp
    if not f.has(Exp):
        # the first test of the module, without importing it
        return None
    from .exponential import exponential_antiderivative
    return attempt(lambda: exponential_antiderivative(f, x, assumptions), _budget(), native=True)


def _trigonometric(f: Expr, x: Symbol, assumptions: Assumptions) -> Optional[Expr]:
    from .trigonometric import trigonometric_antiderivative
    return attempt(lambda: trigonometric_antiderivative(f, x), _budget(), native=True)


def _risch(f: Expr, x: Symbol, assumptions: Assumptions) -> Optional[Expr]:
    from .risch.risch import risch_integrate
    found = attempt(lambda: risch_integrate(f, x), _budget(), native=True)
    return None if found is None or is_unevaluated(found) else found


def _heurisch(f: Expr, x: Symbol, assumptions: Assumptions) -> Optional[Expr]:
    from .heurisch import heurisch_cases
    return attempt(lambda: heurisch_cases(f, x), _budget(), native=True)


def _trager(f: Expr, x: Symbol, assumptions: Assumptions) -> Optional[Expr]:
    from .trager import trager_antiderivative
    return attempt(lambda: trager_antiderivative(f, x), _budget(), native=True)


def _binomial(f: Expr, x: Symbol, assumptions: Assumptions) -> Optional[Expr]:
    from .algebraic import binomial_hypergeometric_antiderivative
    return attempt(lambda: binomial_hypergeometric_antiderivative(f, x), _budget(), native=True)


#: the methods tried on a rewritten form or a substituted integrand
TYPED: Final[list[str]] = ['rational', 'radicals', 'exponential', 'trigonometric', 'risch', 'heurisch', 'trager']

#: the typed methods without the heuristic, which answer or decline fast
_FAST_TYPED: Final[list[str]] = [name for name in TYPED if name != 'heurisch']

#: the hyperbolic functions (without the inverse ones)
_HYPERBOLIC: Final[tuple[type[Basic], ...]] = (Sinh, Cosh, Tanh, Coth, Sech, Csch)

#: the inverse hyperbolic functions
_INVERSE_HYPERBOLIC: Final[tuple[type[Basic], ...]] = (Asinh, Acosh, Atanh, Acoth, Asech, Acsch)


def _hyperbolic(f: Expr) -> bool:
    """Whether ``f`` has hyperbolic functions and no inverse ones: its
    canonical forms are then exponentials, on which the heuristic is
    tried in :func:`_rewriting` rather than in :func:`_rewritten_heurisch`."""
    return f.has(*_HYPERBOLIC) and not f.has(*_INVERSE_HYPERBOLIC)


def _rewriting(f: Expr, x: Symbol, assumptions: Assumptions) -> Optional[Expr]:
    """The typed methods on the canonical forms of ``f`` and on the
    integrands of its substitutions (:mod:`.rewriting`): a power of a
    base not known non-positive as an exponential, hyperbolic functions
    as exponentials, inverse hyperbolic functions as logarithms; ``x = t**k``
    for fractional powers, ``u = exp(c*x)`` for rational functions of an
    exponential, ``x = exp(t)`` for rational functions of a logarithm."""
    from .rewriting import implied_assumptions, power_substitutions, rewritten_forms
    forms = attempt(lambda: rewritten_forms(f, x, assumptions), _budget(), native=True) or []
    # the forms hold under the facts they assume (c > 0 for c**(d*z)), and
    # so does an antiderivative found through them
    implied = _items(assumptions) + (attempt(lambda: implied_assumptions(f, x, assumptions), _budget(),
                                             native=True) or [])
    # the heuristic comes after the substitutions, and after itself on
    # the integrand (:func:`_rewritten_heurisch`): on the logarithms of
    # acoth(sqrt(z)) or acosh(z) it spent up to 15 s and failed, where
    # the substitution z = t**2, or the heuristic on the integrand as
    # given, answer in a tenth of that (the bug: the antiderivative of
    # log(1 - sqrt(z)) - acoth(sqrt(z)), FriCAS's mapleok in2485a, ran out
    # of the budget of the definite integrator)
    for form in forms:
        found = verified_antiderivative(form, x, implied, methods=_FAST_TYPED)
        if found is not None:
            return found[0]
    # the substitutions of f and of its first canonical form: sqrt(a + b*c**(d*z))
    # has its exponential, for u = exp(d*z*log(c)), only once c**(d*z) is written as one
    substitutions = list(attempt(lambda: power_substitutions(f, x), _budget(), native=True) or [])
    for form in forms[:1]:
        for extra in attempt(lambda: power_substitutions(form, x), _budget(), native=True) or []:
            if all(extra.back != s.back for s in substitutions):
                substitutions.append(extra)
    # the facts about the parameters hold in the new variable, those about x do not
    inner = [item for item in implied if not item.has(x)]
    for substitution in substitutions:
        if substitution.facts:
            continue
        F = _substituted(f, x, substitution, inner)
        if F is not None:
            return F
    # hyperbolic functions (without inverse ones) are the exception: the
    # heuristic on their exponentials answers, where on the integrand as
    # given it spends both of its budgets and fails (the bug:
    # sinh(a + b*log(c*x**n))**2, FriCAS integ 10, ran out of 20 s once the
    # heuristic on the forms came after the one on the integrand)
    if _hyperbolic(f):
        for form in forms:
            found = verified_antiderivative(form, x, implied, methods=['heurisch'])
            if found is not None:
                return found[0]
    # the substitutions holding on a region each (a radicand's factor
    # extracted with its sign): the antiderivatives assembled piecewise
    # when every region has one
    pieces: list[tuple[Expr, Boolean]] = []
    for substitution in substitutions:
        if not substitution.facts:
            continue
        F = _substituted(f, x, substitution, inner)
        if F is None or is_antiderivative(F, f, x, implied + substitution.facts) is not True:
            return None
        pieces.append((F, and_(*substitution.facts)))
    if pieces:
        return piecewise(*pieces[:-1], (pieces[-1][0], true))
    return None


def _rewritten_heurisch(f: Expr, x: Symbol, assumptions: Assumptions) -> Optional[Expr]:
    """The heuristic Risch integrator on the canonical forms of ``f``
    (:mod:`.rewriting`), after it failed on ``f`` itself; not again on
    the exponential forms of hyperbolic functions, which
    :func:`_rewriting` tried."""
    from .rewriting import implied_assumptions, rewritten_forms
    if _hyperbolic(f):
        return None
    forms = attempt(lambda: rewritten_forms(f, x, assumptions), _budget(), native=True) or []
    implied = _items(assumptions) + (attempt(lambda: implied_assumptions(f, x, assumptions), _budget(),
                                             native=True) or [])
    for form in forms:
        found = verified_antiderivative(form, x, implied, methods=['heurisch'])
        if found is not None:
            return found[0]
    return None


def _substituted(f: Expr, x: Symbol, substitution: Substitution, inner: list[Boolean]) -> Optional[Expr]:
    """The antiderivative of ``f`` through ``substitution``, the typed
    methods on the substituted integrand and on its canonical forms:
    ``d**(a*z + b*sqrt(z))`` becomes ``2*t*d**(a*t**2 + b*t)``, an
    exponential of a quadratic once the power of ``d`` is written as one."""
    from .rewriting import rewritten_forms, substitute_back
    g, t = substitution.integrand, substitution.variable
    candidates = [g] + (attempt(lambda: rewritten_forms(g, t, inner), _budget(), native=True) or [])
    for candidate in candidates:
        found = verified_antiderivative(candidate, t, inner, methods=TYPED)
        if found is not None:
            return substitute_back(found[0], t, substitution.back)
    return None


# ---------------------------------------------------------------------------
# SymPy's routes, which have no native counterpart: through the face, last
# (with the native manualintegrate in the place of SymPy's)

def _sympy_heurisch(f: Expr, x: Symbol, assumptions: Assumptions) -> Optional[Expr]:
    from .indefinite_sympy import heurisch_of_core
    return heurisch_of_core(f, x, _budget())


def _manual(f: Expr, x: Symbol, assumptions: Assumptions) -> Optional[Expr]:
    from .manualintegrate import manualintegrate
    found = attempt(lambda: manualintegrate(f, x), _long_budget(), native=True)
    return None if found is None or found.has(Integral) else found


def _meijer(f: Expr, x: Symbol, assumptions: Assumptions) -> Optional[Expr]:
    from .indefinite_sympy import meijerint_of_core
    return meijerint_of_core(f, x, _long_budget())


def _sympy(f: Expr, x: Symbol, assumptions: Assumptions) -> Optional[Expr]:
    from .indefinite_sympy import integrate_of_core
    return integrate_of_core(f, x, _long_budget())


#: the methods in the order they are tried
METHODS: list[tuple[str, Method]] = [
    ('rational', _rational), ('radicals', _radicals), ('exponential', _exponential), ('trigonometric', _trigonometric),
    ('risch', _risch), ('trager', _trager), ('binomial', _binomial),
    ('rewriting', _rewriting), ('heurisch', _heurisch), ('rewritten heurisch', _rewritten_heurisch),
    ('sympy heurisch', _sympy_heurisch), ('manual', _manual), ('meijer', _meijer), ('sympy', _sympy)]

#: the methods which load SymPy (through :mod:`.indefinite_sympy`); the
#: native 'manual' among them keeps the place of SymPy's manualintegrate
SYMPY_METHODS: Final[tuple[str, ...]] = ('sympy heurisch', 'meijer', 'sympy')


def verified_antiderivative(f: ExprLike, x: Symbol, assumptions: Assumptions = None,
                            methods: Optional[list[str]] = None) -> Optional[tuple[Expr, str]]:
    """An antiderivative of ``f`` which checks by :func:`is_antiderivative`,
    with the name of the method which found it; ``None`` when no method
    gives one. ``methods`` restricts the names tried, in the order of
    :data:`METHODS`.

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.functions import exp
    >>> from sympy_extras.integrals.indefinite import verified_antiderivative
    >>> x = symbols('x')
    >>> verified_antiderivative(x*exp(x), x)
    (x*exp(x) - exp(x), 'exponential')
    """
    f_ = to_expr(f)
    for name, method in METHODS:
        if methods is not None and name not in methods:
            continue
        try:
            found = method(f_, x, assumptions)
        except (AttributeError, ZeroDivisionError, AssertionError, OverflowError, RecursionError, IndexError):
            # the internals fail on some inputs (an IndexError in SymPy's
            # rational integrator on nested algebraic coefficients)
            found = None
        if found is None or is_unevaluated(found) or found.has(nan, zoo, oo, NegativeInfinity):
            continue
        if found.free_symbols - f_.free_symbols - {x}:
            # a constant of integration or a dummy leaked
            continue
        found = real_form(found, f_, x, assumptions)
        if is_antiderivative(found, f_, x, assumptions) is True:
            return (found, name)
    return None


def indefinite_integral(f: ExprLike, x: Symbol, assumptions: Assumptions = None,
                        special_values: bool = True) -> Expr:
    """``Integral(f, x)`` by the methods of the module documentation, each
    result checked before it is returned; the unevaluated
    :class:`~.integral.Integral` when none gives an antiderivative which
    checks.

    Parameters
    ==========

    f : Expr
    x : Symbol
    assumptions : Boolean or list of Booleans, optional
        Assumptions on the parameters: the typed methods which decide
        signs take them (the exponential and radical tables), and the
        numerical part of the check samples its points under them.
    special_values : bool
        Whether the isolated values of the parameters at which the
        antiderivative found is undefined get cases of their own, the
        integrand integrated again at each
        (:mod:`sympy_extras._special_values`), in the form of SymPy's
        ``integrate``: ``exp(k*x)`` gives ``Piecewise((exp(k*x)/k, Ne(k,
        0)), (x, True))``; a value with several cases lists them first,
        each under its equations. A point where the integrand is not
        defined (``a**(b*x)`` at ``a = 0``), or whose antiderivative is
        not found, gets no case.

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.functions import log, tan, exp
    >>> from sympy_extras.integrals.indefinite import indefinite_integral
    >>> x, k = symbols('x k')
    >>> indefinite_integral(1/(x*(log(x)**2 + 1)), x)
    atan(log(x))
    >>> indefinite_integral(tan(x)**3, x)
    log(cos(x)) + tan(x)**2/2
    >>> indefinite_integral(x*exp(k*x), x)
    Piecewise((x*exp(k*x)/k - exp(k*x)/k**2, Ne(k, 0)), (x**2/2, True))
    """
    return _indefinite(to_expr(f), x, assumptions, special_values, True)


def closed_antiderivative(f: ExprLike, x: Symbol, cases: bool = False, closed: bool = False,
                          seconds: Optional[float] = None) -> Optional[Expr]:
    """An antiderivative of ``f`` by :func:`indefinite_integral` within
    ``seconds`` (the time limit of the settings when ``None``), with the
    cases of the special values of the parameters when ``cases`` (the
    generic antiderivative alone otherwise, as SymPy's ``integrate(f, x,
    conds='none')``); ``None`` when no antiderivative checks or the time
    is up, and with ``closed`` also when it has a ``Piecewise`` or a polar
    number. The quadratures of the native ODE solvers.

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.functions import exp
    >>> from sympy_extras.integrals.indefinite import closed_antiderivative
    >>> x, k = symbols('x k')
    >>> closed_antiderivative(exp(k*x), x), closed_antiderivative(exp(k*x), x, cases=True)
    (exp(k*x)/k, Piecewise((exp(k*x)/k, Ne(k, 0)), (x, True)))
    >>> closed_antiderivative(exp(k*x), x, cases=True, closed=True) is None
    True
    """
    from sympy_extras.functions.elementary.piecewise import Piecewise
    f_ = to_expr(f)
    limit = settings.timeout if seconds is None else seconds
    found = attempt(lambda: _indefinite(f_, x, None, cases, True), limit, native=True)
    if found is None or is_unevaluated(found):
        return None
    if closed and found.has(Piecewise, ExpPolar):
        return None
    return found


def _indefinite(f: Expr, x: Symbol, assumptions: Assumptions, special_values: bool, sympy_style: bool) -> Expr:
    """:func:`indefinite_integral`; the cases of the special values in
    SymPy's form with ``sympy_style``, and each under its equations
    otherwise (inside the case of another value, where the cases are
    flattened into the outer ``Piecewise``)."""
    found = verified_antiderivative(f, x, assumptions)
    if found is None:
        return Integral(f, x)
    if not special_values:
        return found[0]
    parameters = sorted((s for s in f.free_symbols if s != x), key=lambda s: s.name)

    def at_point(point: Point, at: Assumptions) -> Optional[Expr]:
        return _indefinite(_replaced(f, {s: v for s, v in point.items()}), x, at, True, False)

    def defined(point: Point) -> bool:
        return defined_problem(_replaced(f, {s: v for s, v in point.items()}), [x])

    # (a point whose antiderivative is not found gets no case: an
    # unevaluated Integral in a case made the whole answer unevaluated,
    # the integral of a**(b*z)/z at a = 0 among them)
    return with_special_values(found[0], parameters, assumptions, at_point, defined, sympy_style=sympy_style,
                               keep_unevaluated=False)
