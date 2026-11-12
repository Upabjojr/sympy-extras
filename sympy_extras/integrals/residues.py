"""Definite integrals by the residue theorem.

The residue theorem, `\\oint_\\gamma f(z)\\, dz = 2\\pi i \\sum \\operatorname{Res} f`
over the poles inside a closed contour `\\gamma` [Ahlfors]_, chapter 4.5,
evaluates the classical families of definite integrals of rational
functions; Mathematica's ``Integrate`` uses it for the same families
[Wolfram]_. Four of them are implemented here, each with the conditions
under which the contour integral equals the real integral:

1. **Rational functions over the real line**, `\\int_{-\\infty}^\\infty R(x)\\, dx`
   with `R = P/Q`, `\\deg Q \\ge \\deg P + 2` and no real poles: the
   contour is the real segment closed by a large semicircle in the upper
   half plane, whose contribution vanishes, so the integral is `2\\pi i`
   times the sum of the residues of `R` in the upper half plane
   ([Ahlfors]_, 4.5.3, example 1).

2. **Fourier integrals** `\\int_{-\\infty}^\\infty R(x) e^{ikx}\\, dx` with
   `\\deg Q \\ge \\deg P + 1` and `k > 0`: Jordan's lemma makes the
   semicircle vanish again, the value is `2\\pi i` times the residues of
   `R(z) e^{ikz}` in the upper half plane, and `k < 0` uses the lower one
   with the opposite sign. The integrals with `\\cos kx` and `\\sin kx` are
   the half sum and half difference of the two, so they stay exact for
   parametric `R`. The same integrals over `(0, \\infty)` are computed when
   the integrand is even.

3. **Powers of** `x` **times rational functions over** `(0, \\infty)`,
   `\\int_0^\\infty x^\\alpha R(x)\\, dx` with `\\alpha` not an integer, the
   integral converging at both ends and `R` without poles on the positive
   real axis: the keyhole contour around the branch cut of `z^\\alpha`
   along the positive axis gives

   .. math::

       \\int_0^\\infty x^\\alpha R(x)\\, dx =
       \\frac{2\\pi i}{1 - e^{2\\pi i \\alpha}} \\sum_{z_k \\ne 0}
       \\operatorname{Res}_{z = z_k} z^\\alpha R(z),
       \\qquad 0 < \\arg z < 2\\pi,

   ([Ahlfors]_, 4.5.3, example 3; [Marsden]_, section 4.3). For an
   integer `\\alpha` the integrand is rational and the same contour with
   `R(z) \\log z` gives `\\int_0^\\infty R(x)\\, dx = -\\sum \\operatorname{Res} R(z) \\log z`
   with the branch `0 < \\arg z < 2\\pi`. A factor `\\log^n x` is the `n`-th
   derivative of the power formula with respect to `\\alpha`.

4. **Rational functions of** `\\sin x` **and** `\\cos x` **over a period**:
   with `z = e^{ix}`, `\\sin x = (z - 1/z)/2i`, `\\cos x = (z + 1/z)/2` and
   `dx = dz/iz`, the integral over `(a, a + 2\\pi)` is the integral over
   the unit circle of a rational function of `z`, that is `2\\pi i` times
   the residues at the poles inside the circle ([Ahlfors]_, 4.5.3,
   example 2); over `(a, a + 2\\pi k)` it is `k` times that.

The poles are the roots of the denominators (the native
:func:`~sympy_extras.polys.polyroots.roots`, the irrational constants of
the coefficients held by placeholders, since the typed polynomials have
no domain ``EX``), and their position -- in the upper half plane, inside
the unit circle, off the positive real axis -- is decided exactly for
numbers and by :func:`sympy_extras.assumptions.ask.ask` under the
assumptions for parameters; a pole whose position cannot be decided
makes the function give up, except that, with the numerical checks of
the settings on, the position may be read at a sample of the parameters
satisfying the assumptions and the value is then verified numerically
(by mpmath's rules of :mod:`.recognize`, or the validated quadrature of
:mod:`.validated`). The residues at simple poles are `P(z_k)/Q'(z_k)`,
at multiple poles the derivative formula, the pole divided out of `Q`
by synthetic division; the branch of the logarithm is kept as a symbol
`L` with the derivation `d/dz + (1/z) d/dL` so that the derivatives are
branch-independent.

The module works on the expressions of the typed core
(:mod:`sympy_extras.core`) and imports no SymPy. One step stays on
SymPy: the complex roots of a numerical denominator whose roots in
radicals are unwieldy (SymPy's ``CRootOf``, which the core has no node
for) come from :func:`sympy_extras.integrals.residues_sympy.complex_roots_of_core`,
imported only then; the verification of a value placed by sampling,
when no native quadrature settles it, goes to the quadratures of
:func:`~sympy_extras.integrals.definite.verify_numerically`. :mod:`.residues_sympy` is the
face of the module on SymPy objects.

Examples
========

>>> from sympy_extras.core import symbols, Symbol, oo, pi, sqrt
>>> from sympy_extras.functions import cos
>>> from sympy_extras.integrals.residues import residue_integral
>>> x = symbols('x')
>>> a, b = symbols('a b', positive=True)
>>> residue_integral(1/(x**4 + 1), x, -oo, oo)
ConditionalValue(sqrt(2)*pi/2)
>>> residue_integral(cos(b*x)/(x**2 + a**2), x, -oo, oo)
ConditionalValue(pi*exp(-a*b)/a)
>>> residue_integral(1/(2 + cos(x)), x, 0, 2*pi)
ConditionalValue(2*sqrt(3)*pi/3)
>>> residue_integral(sqrt(x)/(x**2 + 1), x, 0, oo)
ConditionalValue(sqrt(2)*pi/2)

References
==========

.. [Ahlfors] L. V. Ahlfors, *Complex analysis*, 3rd edition, McGraw-Hill,
   1979, chapter 4.5 (The calculus of residues).
.. [Marsden] J. E. Marsden, M. J. Hoffman, *Basic complex analysis*, 3rd
   edition, W. H. Freeman, 1999, section 4.3 (Evaluation of definite
   integrals).
.. [Wolfram] Wolfram Research, *Some notes on internal implementation*,
   https://reference.wolfram.com/language/tutorial/SomeNotesOnInternalImplementation.html
"""
from __future__ import annotations

import itertools
import random
from typing import Final, Iterable, Optional, Sequence

import mpmath

from sympy_extras._numeric import reliable_complex, reliable_value
from sympy_extras._timeout import Deadline, TimeLimitExceeded, attempt, time_limit
from sympy_extras.assumptions.ask import Assumptions, ask
from sympy_extras.assumptions.facts import Facts, element
from sympy_extras.assumptions.sat import find_instance
from sympy_extras.core.add import Add, add
from sympy_extras.core.basic import Basic
from sympy_extras.core.evalf import numeric_value
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.function import Application, count_ops, expand, expand_trig
from sympy_extras.core.limits import Limit, limit
from sympy_extras.core.logic import Boolean, BooleanFalse, BooleanTrue, and_, to_boolean
from sympy_extras.core.mul import Mul, mul
from sympy_extras.core.numbers import (ComplexInfinity, Float, I, Infinity, Integer, NaN, NegativeInfinity, Rational,
                                       oo, pi)
from sympy_extras.core.power import Pow, pow
from sympy_extras.core.relational import Eq, Ge, Gt, Le, Lt, Ne, Relational, gt, lt, ne
from sympy_extras.core.singleton import S
from sympy_extras.core.symbol import Dummy, Symbol
from sympy_extras.functions.combinatorial.factorials import factorial
from sympy_extras.functions.elementary.complexes import Im, Re, abs_, arg, im, re
from sympy_extras.functions.elementary.exponential import Exp, Log, exp, log
from sympy_extras.functions.elementary.trigonometric import Cos, Cot, Csc, Sec, Sin, Tan, cos, sin
from sympy_extras.polys.polyerrors import BasePolynomialError
from sympy_extras.polys.polyroots import roots
from sympy_extras.polys.polytools import Poly, cancel, factor
from sympy_extras.sets.sets import Reals
from sympy_extras.settings import settings
from sympy_extras.simplify.simplify import simplify
from .conditions import ConditionalValue, sample_values

__all__ = ['residue_integral', 'rational_function', 'poles', 'real_line_integral',
           'fourier_integral', 'keyhole_integral', 'period_integral']

#: the trigonometric functions (SymPy's ``TrigonometricFunction``)
_TRIGONOMETRIC: Final[tuple[type, ...]] = (Sin, Cos, Tan, Cot, Sec, Csc)

_COUNTER = itertools.count()


def _fresh(name: str, real: bool = False) -> Dummy:
    """A dummy with a name of its own: the typed polynomials map their
    symbols by name, so a dummy ``z`` beside a parameter ``z`` would be
    taken for it."""
    if real:
        return Dummy('%s_%d' % (name, next(_COUNTER)), real=True)
    return Dummy('%s_%d' % (name, next(_COUNTER)))


def _expr(b: Basic) -> Expr:
    if not isinstance(b, Expr):
        raise TypeError("expected an expression, got %s" % (b,))
    return b


def _replace(e: Expr, rule: dict[Basic, Basic]) -> Expr:
    """``e`` with the replacements made and the rebuilt nodes evaluated
    (the core's ``xreplace`` does not evaluate)."""
    if not rule:
        return e
    return _expr(e._xreplace_evaluated(rule))


def _replace_boolean(b: Boolean, rule: dict[Basic, Basic]) -> Boolean:
    if not rule:
        return b
    return to_boolean(b._xreplace_evaluated(rule))


def _budget() -> Optional[float]:
    return settings.timeout


def _is_bad(e: Expr) -> bool:
    """Whether ``e`` contains ``nan`` or ``zoo``."""
    return e.has(NaN, ComplexInfinity)


# ---------------------------------------------------------------------------
# Polynomials with irrational constants in their coefficients

class _Placeholders:
    """The irrational constants of expressions (``sqrt(3)``, ``pi``,
    ``exp(a)``) replaced by dummies, so that the typed polynomials, which
    have no domain ``EX``, take them as parameters (SymPy's ``cancel``
    takes them as generators too); :meth:`restore` puts them back, the
    rebuilt nodes evaluated. Every subexpression free of the variable
    which is not a polynomial in symbols (``sqrt(a)``) is a placeholder,
    and ``I`` too when ``gaussian`` is false (the typed ``Poly`` has no
    domain ``QQ_I``)."""

    def __init__(self, variable: Symbol, gaussian: bool = False) -> None:
        self.variable: Final[Symbol] = variable
        self.gaussian: Final[bool] = gaussian
        self.forward: Final[dict[Expr, Dummy]] = {}
        self.back: Final[dict[Basic, Basic]] = {}

    def hide(self, e: Expr) -> Expr:
        if isinstance(e, (Rational, Symbol)):
            return e
        if self.gaussian and e == I:
            return e
        if isinstance(e, Add):
            return add(*[self.hide(_expr(a)) for a in e.args])
        if isinstance(e, Mul):
            return mul(*[self.hide(_expr(a)) for a in e.args])
        if isinstance(e, Pow) and isinstance(e.exp, Integer):
            return pow(self.hide(e.base), e.exp)
        if e.has(self.variable):
            return e
        if isinstance(e, Pow) and isinstance(e.exp, Rational) and e.exp.p != 1:
            # a**(p/q) is (a**(1/q))**p, as SymPy's generators are
            return pow(self._placeholder(pow(e.base, Rational(1, e.exp.q))), Integer(e.exp.p))
        if isinstance(e, Exp):
            # exp(k*t) is exp(t)**k for an integer k, as SymPy's generators
            # are: the discriminant of u**2 + (exp(a) + exp(-a))*u + 1 is
            # then the square (exp(a)**2 - 1)**2/exp(a)**2
            coefficient, rest = e.args[0].as_coeff_Mul()
            if isinstance(coefficient, Integer) and coefficient.p not in (0, 1):
                return pow(self._placeholder(exp(rest)), coefficient)
        return self._placeholder(e)

    def _placeholder(self, e: Expr) -> Expr:
        found = self.forward.get(e)
        if found is None:
            found = _fresh('c')
            self.forward[e] = found
            self.back[found] = e
        return found

    def restore(self, e: Expr) -> Expr:
        return _replace(e, self.back)


def _coefficients(e: Expr, s: Symbol) -> Optional[dict[int, Expr]]:
    """The coefficients of the polynomial ``e`` in ``s`` by degree, read
    from its expansion (``None`` when ``e`` is not a polynomial in
    ``s``); the zero coefficients are left out."""
    collected: dict[int, list[Expr]] = {}
    for term in Add.make_args(expand(e)):
        if term == 0:
            continue
        coefficient, power = term.as_independent(s, as_Add=False)
        if power == 1:
            degree = 0
        elif power == s:
            degree = 1
        elif isinstance(power, Pow) and power.base == s and isinstance(power.exp, Integer) and power.exp.p > 0:
            degree = power.exp.p
        else:
            return None
        collected.setdefault(degree, []).append(coefficient)
    result: dict[int, Expr] = {}
    for degree, parts in collected.items():
        total = add(*parts)
        if total != 0:
            result[degree] = total
    return result


class _Polynomial:
    """A polynomial in ``x`` with coefficients free of ``x`` (any
    expressions: radicals, ``I``, parameters), dense, the highest degree
    first and no leading zero; the zero polynomial has no coefficients
    and the degree ``-1`` (SymPy's ``Poly`` over ``EX``, which the typed
    polynomials do not have)."""

    __slots__ = ('coefficients', 'x')

    def __init__(self, coefficients: Sequence[Expr], x: Symbol) -> None:
        stripped = list(coefficients)
        while stripped and stripped[0] == 0:
            stripped.pop(0)
        self.coefficients: Final[tuple[Expr, ...]] = tuple(stripped)
        self.x: Final[Symbol] = x

    @staticmethod
    def from_expr(e: Expr, x: Symbol) -> Optional[_Polynomial]:
        """The polynomial of an expression polynomial in ``x``, ``None``
        otherwise."""
        found = _coefficients(e, x)
        if found is None or any(c.has(x) for c in found.values()):
            return None
        top = max(found, default=-1)
        return _Polynomial([found.get(d, S.Zero) for d in range(top, -1, -1)], x)

    def __repr__(self) -> str:
        return "_Polynomial(%s, %s)" % (self.as_expr(), self.x)

    def degree(self) -> int:
        return len(self.coefficients) - 1

    @property
    def is_zero(self) -> bool:
        return not self.coefficients

    def all_coeffs(self) -> list[Expr]:
        return list(self.coefficients)

    def coeffs(self) -> list[Expr]:
        """The nonzero coefficients."""
        return [c for c in self.coefficients if c != 0]

    def as_expr(self) -> Expr:
        n = self.degree()
        return add(*[mul(c, pow(self.x, Integer(n - i))) for i, c in enumerate(self.coefficients)])

    def diff(self) -> _Polynomial:
        n = self.degree()
        return _Polynomial([mul(Integer(n - i), c) for i, c in enumerate(self.coefficients[:-1])], self.x)

    def eval(self, value: Expr) -> Expr:
        """The value at ``value`` (Horner's scheme)."""
        total: Expr = S.Zero
        for c in self.coefficients:
            total = add(mul(total, value), c)
        return total

    def times_power(self, k: int) -> _Polynomial:
        """The polynomial multiplied by ``x**k``."""
        return _Polynomial(list(self.coefficients) + [S.Zero] * k, self.x)

    def quotient_by_root(self, point: Expr, m: int) -> _Polynomial:
        """The quotient by ``(x - point)**m`` by synthetic division, for a
        root ``point`` of multiplicity at least ``m``: the remainders,
        zero, are dropped (SymPy's division over ``EX``)."""
        current = list(self.coefficients)
        for _ in range(m):
            quotient: list[Expr] = []
            carry: Expr = S.Zero
            for c in current[:-1]:
                carry = expand(add(c, mul(point, carry)))
                quotient.append(carry)
            current = quotient
        return _Polynomial(current, self.x)

    def to_poly(self) -> Optional[tuple[Poly, _Placeholders]]:
        """The typed ``Poly`` of the polynomial, the constants held by
        placeholders; ``None`` when it does not take it."""
        hidden = _Placeholders(self.x)
        e = hidden.hide(self.as_expr())
        if any(isinstance(p, Pow) and p.exp.is_negative for p in e.atoms(Pow)):
            # the negative powers of the placeholders (exp(-a)) cleared: a
            # constant factor, which leaves the roots alone
            e = cancel(e).as_numer_denom()[0]
        try:
            return (Poly(e, self.x), hidden)
        except (BasePolynomialError, NotImplementedError, ValueError, TypeError):
            return None


class _Fraction:
    """A rational function ``numerator/denominator`` of ``x``, as
    polynomials with coefficients free of ``x``."""

    __slots__ = ('numerator', 'denominator', 'x')

    def __init__(self, numerator: _Polynomial, denominator: _Polynomial, x: Symbol) -> None:
        self.numerator: Final[_Polynomial] = numerator
        self.denominator: Final[_Polynomial] = denominator
        self.x: Final[Symbol] = x

    def __repr__(self) -> str:
        return "_Fraction(%s, %s)" % (self.as_expr(), self.x)

    def as_expr(self) -> Expr:
        return mul(self.numerator.as_expr(), pow(self.denominator.as_expr(), Integer(-1)))

    @property
    def excess(self) -> int:
        """``deg Q - deg P``."""
        return self.denominator.degree() - self.numerator.degree()


def rational_function(f: ExprLike, x: Symbol) -> Optional[_Fraction]:
    """``f`` as a quotient of polynomials in ``x`` with coefficients free of
    ``x`` and the common factors cancelled, or ``None``.

    >>> from sympy_extras.core import symbols, sqrt
    >>> from sympy_extras.integrals.residues import rational_function
    >>> x, a = symbols('x a')
    >>> rational_function((x + 1)/((x + 1)*(x**2 + sqrt(2)*a)), x)
    _Fraction(1/(sqrt(2)*a + x**2), x)
    """
    e = to_expr(f)
    if not e.has(x):
        return None
    if e.is_rational_function(x) is not True:
        return None
    hidden = _Placeholders(x, gaussian=True)
    h = hidden.hide(e)
    reduced = attempt(lambda: cancel(h), _budget(), native=True)
    if reduced is None:
        return None
    numerator, denominator = reduced.as_numer_denom()
    p = _Polynomial.from_expr(numerator, x)
    q = _Polynomial.from_expr(denominator, x)
    if p is None or q is None:
        return None
    p = _Polynomial([hidden.restore(c) for c in p.coefficients], x)
    q = _Polynomial([hidden.restore(c) for c in q.coefficients], x)
    if any(c.has(x) for c in p.coefficients + q.coefficients):
        return None
    if q.is_zero:
        return None
    return _Fraction(p, q, x)


class _Pole:
    """A pole with its multiplicity; ``numeric`` for a root of a numerical
    polynomial found as SymPy's ``CRootOf`` (whose position is read from
    its value)."""

    __slots__ = ('point', 'multiplicity', 'numeric')

    def __init__(self, point: Expr, multiplicity: int, numeric: bool = False) -> None:
        self.point: Final[Expr] = point
        self.multiplicity: Final[int] = multiplicity
        self.numeric: Final[bool] = numeric

    def __repr__(self) -> str:
        return "_Pole(%s, %s)" % (self.point, self.multiplicity)


def _gaussian_factors(q: _Polynomial) -> list[tuple[_Polynomial, int]]:
    """``q`` split into its factors over ``ZZ_I`` (the native ``factor``)
    when ``I`` is among its coefficients: a placeholder for ``I`` would
    hide the square ``(z**2 + 6*I*z - 1)**2`` of the denominator of
    ``1/(3 + sin(x))**2`` in ``z = exp(I*x)`` (SymPy's polynomials over
    ``QQ_I`` see it); ``q`` itself otherwise."""
    if not any(c.has(I) for c in q.coefficients):
        return [(q, 1)]
    factored = attempt(lambda: factor(q.as_expr()), _budget(), native=True)
    if factored is None:
        return [(q, 1)]
    result: list[tuple[_Polynomial, int]] = []
    for part in Mul.make_args(factored):
        base, multiplicity = part, 1
        if isinstance(part, Pow) and isinstance(part.exp, Integer) and part.exp.p > 0:
            base, multiplicity = part.base, part.exp.p
        if not base.has(q.x):
            continue
        p = _Polynomial.from_expr(base, q.x)
        if p is None:
            return [(q, 1)]
        result.append((p, multiplicity))
    if sum(p.degree() * m for p, m in result) != q.degree():
        return [(q, 1)]
    return result


def _roots_in_radicals(q: _Polynomial) -> Optional[dict[Expr, int]]:
    """The roots of ``q`` with their multiplicities by the native ``roots``
    (the constants held by placeholders, put back afterwards, equal roots
    merged; the factors over ``ZZ_I`` first when ``I`` is a coefficient);
    ``None`` when the native polynomials do not take ``q``."""
    result: dict[Expr, int] = {}
    for part, multiplicity in _gaussian_factors(q):
        converted = part.to_poly()
        if converted is None:
            return None
        poly, hidden = converted
        try:
            found = roots(poly)
        except (BasePolynomialError, NotImplementedError, ValueError, TypeError):
            return None
        if not isinstance(found, dict):
            return None
        for r, m in found.items():
            point = hidden.restore(r)
            result[point] = result.get(point, 0) + m * multiplicity
    return result


def poles(q: _Polynomial) -> Optional[list[_Pole]]:
    """The roots of ``q`` with multiplicities, or ``None`` when they cannot
    all be found (``roots`` for parametric coefficients, SymPy's
    ``CRootOf`` through the face for numerical ones whose radicals are
    unwieldy).

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.integrals.residues import rational_function, poles
    >>> x = symbols('x')
    >>> fraction = rational_function(1/((x**2 + 1)**2*(x - 3)), x)
    >>> sorted((str(p.point), p.multiplicity) for p in poles(fraction.denominator))
    [('-I', 2), ('3', 1), ('I', 2)]
    """
    if q.degree() <= 0:
        return []
    parametric = any(c.free_symbols for c in q.coeffs())
    found = attempt(lambda: _roots_in_radicals(q), _budget(), native=True)
    if found is not None and sum(found.values()) == q.degree() \
            and (parametric or not any(_unwieldy(r) for r in found)):
        return [_Pole(r, m) for r, m in found.items()]
    if parametric or not all(isinstance(c, Rational) for c in q.coefficients):
        return None
    try:
        _, factors = Poly(q.as_expr(), q.x).factor_list()
    except (BasePolynomialError, NotImplementedError, ValueError, TypeError):
        return None
    result: list[_Pole] = []
    for g, e in factors:
        if g.degree() <= 0:
            continue
        if g.degree() <= 2:
            low = _roots_in_radicals(_Polynomial(g.all_coeffs(), q.x))
            if low is None:
                return None
            result.extend(_Pole(r, m * e) for r, m in low.items())
            continue
        from .residues_sympy import complex_roots_of_core
        all_roots = complex_roots_of_core(g.as_expr(), q.x, _budget())
        if all_roots is None:
            return None
        result.extend(_Pole(r, e, True) for r in all_roots)
    if sum(p.multiplicity for p in result) != q.degree():
        return None
    return result


def _unwieldy(root: Expr) -> bool:
    """Whether a root in radicals is too large to work with: the roots
    of x**4 + x**2 + x + 1 are nested cube roots of complex numbers, for
    which ``CRootOf`` is the better form."""
    return _size(root) > 60


# ---------------------------------------------------------------------------
# Deciding where a pole lies

class _Locator:
    """Answers ``query`` about a pole: exactly, or under the assumptions,
    or (with the numerical checks on) at a sample of the parameters."""

    def __init__(self, assumptions: Assumptions) -> None:
        self.assumptions: Final[Assumptions] = assumptions
        self.sampled = False
        self._sample: Optional[dict[Symbol, Expr]] = None
        self._no_sample = False
        self._slow = False

    def decide(self, query: Boolean) -> Optional[bool]:
        q = query
        if isinstance(q, BooleanTrue):
            return True
        if isinstance(q, BooleanFalse):
            return False
        if not q.free_symbols:
            # a question about numbers (the imaginary part of a root of
            # x**4 + x**2 + x + 1 in radicals, or a CRootOf): the sides
            # evaluated
            verdict = _numerically(q)
            if verdict is not None:
                return verdict
        # a question about radicals of the parameters is expensive for the
        # exact decision procedures: the sample is consulted first, and
        # the value found with it is verified numerically afterwards
        if settings.numerical_checks and _has_radicals(q):
            verdict = self._at_sample(q)
            if verdict is not None:
                return verdict
        if self._slow:
            return None
        deadline: Optional[Deadline] = None
        try:
            with time_limit(settings.timeout, native=True) as deadline:
                verdict = ask(q, self.assumptions)
        except TimeLimitExceeded as expiry:
            # an enclosing limit which expires is not this question being
            # slow: it goes on to the code which set it
            if not expiry.expired(deadline):
                raise
            self._slow = True
            return None
        if verdict is not None:
            return verdict
        if not settings.numerical_checks:
            return None
        return self._at_sample(q)

    def _at_sample(self, q: Boolean) -> Optional[bool]:
        sample = self.sample(_sorted_symbols(q.free_symbols))
        if sample is None:
            return None
        at_sample = _replace_boolean(q, {s: v for s, v in sample.items()})
        if isinstance(at_sample, BooleanTrue):
            self.sampled = True
            return True
        if isinstance(at_sample, BooleanFalse):
            self.sampled = True
            return False
        verdict = ask(at_sample)
        if verdict is not None:
            self.sampled = True
        return verdict

    def sample(self, symbols: Sequence[Symbol]) -> Optional[dict[Symbol, Expr]]:
        """A point of the parameter space satisfying the assumptions."""
        if self._no_sample:
            return None
        if self._sample is not None and all(s in self._sample for s in symbols):
            return self._sample
        facts = Facts(self.assumptions, symbols=symbols)
        wanted = _sorted_symbols(set(symbols) | facts.formula.free_symbols)
        # plain symbols, so that the flags of the symbols (positive=True)
        # are relations the SAT solver sees
        names = [s.name for s in wanted]
        distinct = len(set(names)) == len(names)
        plain: dict[Symbol, Symbol] = {s: Dummy(s.name) if distinct else _fresh(s.name) for s in wanted}
        flags: list[Boolean] = []
        for s in wanted:
            if s.is_positive:
                flags.append(gt(plain[s], 0))
            elif s.is_negative:
                flags.append(lt(plain[s], 0))
            elif s.is_nonnegative:
                flags.append(plain[s] >= 0)
            elif s.is_nonzero:
                flags.append(ne(plain[s], 0))
        formula = and_(_replace_boolean(facts.formula, {s: p for s, p in plain.items()}), *flags)
        if isinstance(formula, BooleanTrue):
            # nothing constrains the parameters: a sample says nothing
            self._no_sample = True
            return None
        witnesses = find_instance(formula, [plain[s] for s in wanted])
        if not witnesses:
            self._no_sample = True
            return None
        self._sample = {s: witnesses[0].get(plain[s], Rational(3, 2)) for s in wanted}
        return self._sample

    def imaginary_sign(self, pole: _Pole) -> Optional[int]:
        """The sign of the imaginary part of a pole: ``1``, ``-1``, ``0``
        (real) or ``None``."""
        point = pole.point
        if pole.numeric:
            if point.is_extended_real is True:
                return 0
            value = numeric_value(point, 100)
            if value is None:
                return None
            return 1 if value.imag > 0 else -1
        imaginary = im(point)
        if self.decide(gt(imaginary, 0)):
            return 1
        if self.decide(lt(imaginary, 0)):
            return -1
        real = self.decide(element(point, Reals))
        if real is True:
            return 0
        return None

    def inside_unit_circle(self, pole: _Pole) -> Optional[bool]:
        """Whether ``|point| < 1``; ``None`` on the circle or undecided."""
        point = pole.point
        if pole.numeric or not point.free_symbols:
            number = reliable_value(point, 30)
            if number is None:
                return None
            value = numeric_value(number, 120)
            if value is None:
                return None
            with mpmath.workprec(120):
                distance = abs(value) - 1
            if abs(distance) < mpmath.mpf(10)**(-20):
                return None
            return bool(distance < 0)
        modulus = abs_(point)
        inside = self.decide(lt(modulus, 1))
        if inside is not None:
            return inside if inside else (None if self.decide(gt(modulus, 1)) is not True else False)
        if self.decide(element(point, Reals)):
            square = pow(point, Integer(2))
            inside = self.decide(lt(square, 1))
            if inside is True:
                return True
            if self.decide(gt(square, 1)) is True:
                return False
        return None


def _sorted_symbols(symbols: Iterable[Basic]) -> list[Symbol]:
    """The symbols among ``symbols``, sorted by name (and by the order of
    creation of the dummies of one name)."""
    found = [s for s in symbols if isinstance(s, Symbol)]
    return sorted(found, key=lambda s: (s.name, s.dummy_index if isinstance(s, Dummy) else -1))


def _numerically(q: Boolean) -> Optional[bool]:
    """A relation between numbers decided by evaluation: ``None`` when
    the difference of the sides is not clearly of one sign (an equality
    is never confirmed this way, only refuted)."""
    if not isinstance(q, Relational):
        return None
    lhs, rhs = q.args[0], q.args[1]
    if not isinstance(lhs, Expr) or not isinstance(rhs, Expr):
        return None
    # not by a value whose branch a rounding error chooses (see
    # reliable_form): the sign of im(log(-2*sqrt(2) - 2*sqrt(z))) with z a
    # sum which is exactly zero is the sign of that error
    difference = reliable_value(add(lhs, mul(Integer(-1), rhs)), 50)
    if difference is None or not isinstance(difference, Float):
        return None
    value = numeric_value(difference, 200)
    if value is None or value.imag != 0:
        return None
    if abs(value.real) < mpmath.mpf('1e-40'):
        return None
    positive = bool(value.real > 0)
    if isinstance(q, (Gt, Ge)):
        return positive
    if isinstance(q, (Lt, Le)):
        return not positive
    if isinstance(q, Eq):
        return False
    if isinstance(q, Ne):
        return True
    return None


def _has_radicals(q: Boolean) -> bool:
    """Whether the question involves non-polynomial functions of the
    parameters (radicals, absolute values, ...)."""
    for node in q.atoms(Pow):
        if isinstance(node, Pow) and not isinstance(node.exp, Integer):
            return True
    return bool(q.atoms(Application))


# ---------------------------------------------------------------------------
# Residues

def _derivative(h: Expr, x: Symbol, branch: Optional[Symbol]) -> Expr:
    """``dh/dx``, the symbol ``branch`` standing for ``log(x)``
    (``d branch/dx = 1/x``)."""
    found = h.diff(x)
    if branch is not None and h.has(branch):
        found = add(found, mul(h.diff(branch), pow(x, Integer(-1))))
    return found


def _residue(fraction: _Fraction, pole: _Pole, factor: Expr, z: Symbol,
             branch: Optional[Symbol] = None) -> Optional[Expr]:
    """The residue of ``fraction(z) * factor(z)`` at the pole, ``factor``
    holomorphic there: ``P(z_k) factor(z_k)/Q'(z_k)`` at a simple pole,
    the derivative formula otherwise (the branch of a logarithm written
    as the symbol ``branch``, with the derivation ``d/dz + (1/z) d/dL``,
    is left in place)."""
    x = fraction.x
    point = pole.point
    g = _replace(factor, {z: x})
    if pole.multiplicity == 1:
        derivative = fraction.denominator.diff().eval(point)
        at = _replace(g, {x: point})
        return mul(fraction.numerator.eval(point), at, pow(derivative, Integer(-1)))
    m = pole.multiplicity
    quotient = fraction.denominator.quotient_by_root(point, m)
    h = mul(fraction.numerator.as_expr(), pow(quotient.as_expr(), Integer(-1)), g)
    for _ in range(m - 1):
        h = _derivative(h, x, branch)
    value = mul(_replace(h, {x: point}), pow(factorial(Integer(m - 1)), Integer(-1)))
    if _is_bad(value):
        found = attempt(lambda: limit(h, x, point), _budget(), native=True)
        if found is None or isinstance(found, Limit) or _is_bad(found) or found.has(Infinity, NegativeInfinity):
            return None
        value = mul(found, pow(factorial(Integer(m - 1)), Integer(-1)))
    return value


def _branch_log(point: Expr, imaginary_sign: int) -> Expr:
    """``log|point| + i theta`` with ``0 < theta < 2 pi`` (``theta = pi``
    for a negative real pole)."""
    if imaginary_sign == 0:
        return add(log(mul(Integer(-1), point)), mul(I, pi))
    theta = arg(point)
    if imaginary_sign < 0:
        theta = add(theta, mul(Integer(2), pi))
    return add(log(abs_(point)), mul(I, theta))


def _tidy(value: Expr, assumptions: Assumptions) -> Expr:
    """The value simplified, with the imaginary parts which cancel for
    real parameters removed; each simplification under an eighth of the
    time limit (``simplify`` on the roots of x**4 + x**2 + x + 1 took the
    whole budget of the definite driver)."""
    result = value
    budget = None if settings.timeout is None else settings.timeout / 8
    parameters = _sorted_symbols(result.free_symbols)
    reals: dict[Basic, Basic] = {s: _fresh(s.name, real=True) for s in parameters
                                 if s.is_extended_real or ask(element(s, Reals), assumptions)}
    if reals and result.has(I):
        back: dict[Basic, Basic] = {d: s for s, d in reals.items()}
        replaced = _replace(result, reals)
        expanded = attempt(lambda: expand(replaced, complex=True), budget, native=True)
        if expanded is not None and not expanded.has(I):
            result = _replace(expanded, back)
    current = result
    simpler = attempt(lambda: simplify(cancel(current)), budget, native=True)
    if simpler is not None and _size(simpler) <= _size(result):
        result = simpler
    if result.has(I) and not reals and not parameters:
        current = result
        simpler = attempt(lambda: simplify(expand(current, complex=True)), budget, native=True)
        if simpler is not None and not simpler.has(I):
            result = simpler
    return result


def _size(e: Expr) -> int:
    counted = count_ops(e, visual=False)
    return counted.p if isinstance(counted, Integer) else 0


# ---------------------------------------------------------------------------
# 1. and 2.: the real line

def _sum_residues(fraction: _Fraction, factor: Expr, z: Symbol, locator: _Locator,
                  upper: bool) -> Optional[Expr]:
    """The sum of the residues of ``fraction * factor`` in the upper (or
    lower) half plane; ``None`` with a real pole or an undecided one."""
    found = poles(fraction.denominator)
    if found is None:
        return None
    total: Expr = S.Zero
    for pole in found:
        s = locator.imaginary_sign(pole)
        if s is None or s == 0:
            return None
        if (s > 0) == upper:
            value = _residue(fraction, pole, factor, z)
            if value is None:
                return None
            total = add(total, value)
    return total


def real_line_integral(f: ExprLike, x: Symbol, assumptions: Assumptions = None) -> Optional[ConditionalValue]:
    """``Integral(R(x), (x, -oo, oo))`` for a rational function with
    ``deg Q >= deg P + 2`` and no real poles.

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.integrals.residues import real_line_integral
    >>> x = symbols('x')
    >>> real_line_integral(1/(x**2 + 1)**2, x)
    ConditionalValue(pi/2)
    """
    e = to_expr(f)
    fraction = rational_function(e, x)
    if fraction is None or fraction.excess < 2:
        return None
    locator = _Locator(assumptions)
    z = _fresh('z')
    total = _sum_residues(fraction, S.One, z, locator, True)
    if total is None:
        return None
    return _checked(ConditionalValue(_tidy(mul(Integer(2), pi, I, total), assumptions)),
                    e, x, S.NegativeInfinity, oo, assumptions, locator)


def _fourier_factor(f: Expr, x: Symbol) -> Optional[tuple[Expr, Expr, str]]:
    """``(R, k, kind)`` when ``f == R(x) * g(k x)`` with ``g`` one of
    ``exp(I*.)``, ``cos``, ``sin`` (``kind`` ``'exp'``, ``'cos'``,
    ``'sin'``) and ``R`` rational."""
    rest: list[Expr] = []
    found: Optional[tuple[Expr, str]] = None
    for e in Mul.make_args(f):
        kind: Optional[str] = None
        if isinstance(e, Exp):
            kind = 'exp'
        elif isinstance(e, Cos):
            kind = 'cos'
        elif isinstance(e, Sin):
            kind = 'sin'
        if kind is None or found is not None or not e.has(x):
            rest.append(e)
            continue
        inner = _expr(e.args[0])
        coefficient, variable = inner.as_independent(x, as_Add=False)
        if variable != x:
            return None
        k = coefficient
        if kind == 'exp':
            k = mul(k, pow(I, Integer(-1)))
            if k.is_extended_real is not True and not (k.is_number and im(k) == 0 and k != 0):
                return None
        found = (k, kind)
    if found is None:
        return None
    return (mul(*rest), found[0], found[1])


def _real_parameters(e: Expr, assumptions: Assumptions) -> Optional[dict[Basic, Basic]]:
    """Real dummies for the parameters of ``e``, or ``None`` when one is
    not known to be real."""
    reals: dict[Basic, Basic] = {}
    for s in _sorted_symbols(e.free_symbols):
        if s.is_extended_real or ask(element(s, Reals), assumptions):
            reals[s] = _fresh(s.name, real=True)
        else:
            return None
    return reals


def _part(e: Expr, real: bool, assumptions: Assumptions) -> Optional[Expr]:
    """The real (or imaginary) part of ``e`` for real parameters, when it
    comes out free of ``re``, ``im`` and ``I``."""
    reals = _real_parameters(e, assumptions)
    if reals is None:
        return None
    back: dict[Basic, Basic] = {d: s for s, d in reals.items()}
    replaced = _replace(e, reals)
    found = attempt(lambda: expand(re(replaced) if real else im(replaced), complex=True), _budget(), native=True)
    if found is None or found.has(Re, Im, I):
        return None
    return _replace(found, back)


def fourier_integral(f: ExprLike, x: Symbol, assumptions: Assumptions = None,
                     half: bool = False) -> Optional[ConditionalValue]:
    """``Integral(R(x)*exp(I*k*x), (x, -oo, oo))``, and with ``cos(k*x)`` or
    ``sin(k*x)``, for a rational ``R`` with ``deg Q >= deg P + 1`` and no
    real poles, by Jordan's lemma; with ``half=True`` the integral over
    ``(0, oo)`` of an even integrand.

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.functions import cos, sin
    >>> from sympy_extras.integrals.residues import fourier_integral
    >>> x, k = symbols('x k')
    >>> fourier_integral(cos(2*x)/(x**2 + 1), x)
    ConditionalValue(pi*exp(-2))
    >>> fourier_integral(x*sin(k*x)/(x**2 + 4), x)
    ConditionalValue(pi*exp(-2*k), k > 0)
    """
    e = to_expr(f)
    found = _fourier_factor(e, x)
    if found is None:
        return None
    r, k, kind = found
    fraction = rational_function(r, x)
    if fraction is None or fraction.excess < 1:
        return None
    if half:
        mirrored = _replace(e, {x: mul(Integer(-1), x)})
        even = attempt(lambda: cancel(add(e, mul(Integer(-1), mirrored))), _budget(), native=True)
        if even is None or even != 0:
            return None
    locator = _Locator(assumptions)
    z = _fresh('z')
    condition: Boolean = S.true
    sign_k = 0
    if k.is_positive:
        sign_k = 1
    elif k.is_negative:
        sign_k = -1
    else:
        verdict = ask(gt(k, 0), assumptions)
        if verdict is True:
            sign_k = 1
        elif verdict is False and ask(lt(k, 0), assumptions) is True:
            sign_k = -1
        elif verdict is None:
            condition = gt(k, 0)
            sign_k = 1
        else:
            return None
    kappa = k if sign_k > 0 else mul(Integer(-1), k)
    # E(kappa) = 2 pi i sum_upper Res R e^{i kappa z}, E(-kappa) = -2 pi i sum_lower Res R e^{-i kappa z}
    upper = _sum_residues(fraction, exp(mul(I, kappa, z)), z, locator, True)
    if upper is None:
        return None
    plus = mul(Integer(2), pi, I, upper)
    value: Optional[Expr] = None
    if kind == 'exp':
        if sign_k > 0:
            value = plus
        else:
            lower = _sum_residues(fraction, exp(mul(Integer(-1), I, kappa, z)), z, locator, False)
            if lower is None:
                return None
            value = mul(Integer(-2), pi, I, lower)
    else:
        # the real and imaginary parts of E(kappa) when the coefficients are real
        value = _part(plus, kind == 'cos', assumptions)
        if value is None:
            lower = _sum_residues(fraction, exp(mul(Integer(-1), I, kappa, z)), z, locator, False)
            if lower is None:
                return None
            minus = mul(Integer(-2), pi, I, lower)
            if kind == 'cos':
                value = mul(add(plus, minus), S.Half)
            else:
                value = mul(add(plus, mul(Integer(-1), minus)), pow(mul(Integer(2), I), Integer(-1)))
        if kind == 'sin' and sign_k < 0:
            value = mul(Integer(-1), value)
    if half:
        value = mul(value, S.Half)
    return _checked(ConditionalValue(_tidy(value, assumptions), condition), e, x,
                    S.Zero if half else S.NegativeInfinity, oo, assumptions, locator)


# ---------------------------------------------------------------------------
# 3.: the keyhole contour

def _power_and_logs(f: Expr, x: Symbol) -> Optional[tuple[Expr, int, Expr]]:
    """``(alpha, n, R)`` when ``f == x**alpha * log(x)**n * R(x)``."""
    alpha: Expr = S.Zero
    n = 0
    rest: list[Expr] = []
    for e in Mul.make_args(f):
        if e == x:
            alpha = add(alpha, S.One)
        elif isinstance(e, Pow) and e.base == x and not e.exp.has(x):
            alpha = add(alpha, e.exp)
        elif isinstance(e, Log) and e.args[0] == x:
            n += 1
        elif isinstance(e, Pow) and isinstance(e.base, Log) and e.base.args[0] == x \
                and isinstance(e.exp, Integer) and e.exp.p > 0:
            n += e.exp.p
        else:
            rest.append(e)
    return (alpha, n, mul(*rest))


def _place_off_axis(pole: _Pole, locator: _Locator) -> Optional[int]:
    """The sign of the imaginary part of a pole which is neither 0 nor on
    the positive real axis, ``None`` otherwise or when undecided."""
    if pole.point == 0:
        return None
    s = locator.imaginary_sign(pole)
    if s is None:
        return None
    if s == 0 and locator.decide(gt(re(pole.point), 0)) is not False:
        return None
    return s


def _keyhole_sum(fraction: _Fraction, alpha: Expr, locator: _Locator, z: Symbol) -> Optional[Expr]:
    """``sum Res z^alpha R(z)`` over the poles off the positive real axis,
    with ``z^alpha = exp(alpha L(z))``, ``0 < arg z < 2 pi``."""
    found = poles(fraction.denominator)
    if found is None:
        return None
    branch = _fresh('L')
    total: Expr = S.Zero
    for pole in found:
        s = _place_off_axis(pole, locator)
        if s is None:
            return None
        value = _residue(fraction, pole, exp(mul(alpha, branch)), z, branch)
        if value is None:
            return None
        total = add(total, _replace(value, {branch: _branch_log(pole.point, s)}))
    return total


def _log_sum(fraction: _Fraction, locator: _Locator, z: Symbol) -> Optional[Expr]:
    """``-sum Res R(z) L(z)`` over the poles, ``0 < arg z < 2 pi``."""
    found = poles(fraction.denominator)
    if found is None:
        return None
    branch = _fresh('L')
    total: Expr = S.Zero
    for pole in found:
        s = _place_off_axis(pole, locator)
        if s is None:
            return None
        value = _residue(fraction, pole, branch, z, branch)
        if value is None:
            return None
        total = add(total, _replace(value, {branch: _branch_log(pole.point, s)}))
    return mul(Integer(-1), total)


def _absorb_zero(fraction: _Fraction, alpha: Expr) -> tuple[_Fraction, Expr]:
    """The factor ``x**m`` of the numerator or the denominator moved into
    the power."""
    x = fraction.x
    p, q = fraction.numerator, fraction.denominator
    while p.degree() > 0 and p.coefficients[-1] == 0:
        p = _Polynomial(p.coefficients[:-1], x)
        alpha = add(alpha, S.One)
    while q.degree() > 0 and q.coefficients[-1] == 0:
        q = _Polynomial(q.coefficients[:-1], x)
        alpha = add(alpha, Integer(-1))
    return _Fraction(p, q, x), alpha


def keyhole_integral(f: ExprLike, x: Symbol, assumptions: Assumptions = None) -> Optional[ConditionalValue]:
    """``Integral(x**alpha * log(x)**n * R(x), (x, 0, oo))`` for a rational
    ``R`` without poles on the positive real axis, by the keyhole contour
    (a plain rational function, ``alpha`` an integer, by the logarithm).

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.functions import log
    >>> from sympy_extras.integrals.residues import keyhole_integral
    >>> x, a = symbols('x a')
    >>> keyhole_integral(x**a/(x + 1), x)
    ConditionalValue(-pi/sin(pi*a), (a > -1) & (a < 0))
    >>> keyhole_integral(1/(x**3 + 1), x)
    ConditionalValue(2*sqrt(3)*pi/9)
    >>> keyhole_integral(log(x)/(x**2 + 1), x)
    ConditionalValue(0)
    """
    e = to_expr(f)
    found = _power_and_logs(e, x)
    if found is None:
        return None
    alpha, n, r = found
    fraction = rational_function(r, x)
    if fraction is None:
        return None
    fraction, alpha = _absorb_zero(fraction, alpha)
    locator = _Locator(assumptions)
    z = _fresh('z')
    lower_bound = gt(alpha, -1)
    upper_bound = lt(alpha, fraction.excess - 1)
    condition = and_(lower_bound, upper_bound)
    if isinstance(condition, BooleanFalse):
        return None
    if n > 0:
        return _keyhole_with_logs(fraction, alpha, n, locator, z, condition, e, x, assumptions)
    if alpha.is_integer:
        if not isinstance(alpha, Integer):
            return None
        p, q = fraction.numerator, fraction.denominator
        shift = alpha.p
        if shift >= 0:
            p = p.times_power(shift)
        else:
            q = q.times_power(-shift)
        folded = _Fraction(p, q, x)
        if folded.excess < 2 or shift < 0:
            return None
        total = _log_sum(folded, locator, z)
        if total is None:
            return None
        return _checked(ConditionalValue(_tidy(total, assumptions)), e, x, S.Zero, oo, assumptions, locator)
    total = _keyhole_sum(fraction, alpha, locator, z)
    if total is None:
        return None
    value = mul(Integer(2), pi, I, total, pow(add(S.One, mul(Integer(-1), exp(mul(Integer(2), pi, I, alpha)))),
                                              Integer(-1)))
    return _checked(ConditionalValue(_tidy(value, assumptions), condition), e, x, S.Zero, oo,
                    assumptions, locator)


def _keyhole_with_logs(fraction: _Fraction, alpha: Expr, n: int, locator: _Locator, z: Symbol,
                       condition: Boolean, f: Expr, x: Symbol, assumptions: Assumptions) -> Optional[ConditionalValue]:
    """The power formula with a symbolic exponent, differentiated ``n``
    times; the limit at an integer exponent."""
    a = _fresh('alpha', real=True)
    total = _keyhole_sum(fraction, a, locator, z)
    if total is None:
        return None
    value = mul(Integer(2), pi, I, total, pow(add(S.One, mul(Integer(-1), exp(mul(Integer(2), pi, I, a)))),
                                              Integer(-1)))
    for _ in range(n):
        value = value.diff(a)
    if alpha.is_integer:
        current = value
        at = attempt(lambda: limit(current, a, alpha), _budget(), native=True)
        if at is None or isinstance(at, Limit) or at.has(Infinity, NegativeInfinity):
            return None
        value = at
    else:
        value = _replace(value, {a: alpha})
    return _checked(ConditionalValue(_tidy(value, assumptions), condition), f, x, S.Zero, oo,
                    assumptions, locator)


# ---------------------------------------------------------------------------
# 4.: a period

def _periods(a: Expr, b: Expr) -> Optional[int]:
    """``k`` when ``b - a == 2*pi*k`` for a positive integer ``k``."""
    k = mul(add(b, mul(Integer(-1), a)), pow(mul(Integer(2), pi), Integer(-1)))
    if isinstance(k, Integer) and k.p > 0:
        return k.p
    return None


def period_integral(f: ExprLike, x: Symbol, a: ExprLike, b: ExprLike,
                    assumptions: Assumptions = None) -> Optional[ConditionalValue]:
    """``Integral(f, (x, a, b))`` over ``k`` periods, ``b - a = 2*pi*k``,
    for a rational function of ``sin(x)`` and ``cos(x)`` (also of
    multiple angles), through ``z = exp(I*x)`` and the residues inside
    the unit circle.

    Examples
    ========

    >>> from sympy_extras.core import symbols, Symbol, pi
    >>> from sympy_extras.functions import cos
    >>> from sympy_extras.integrals.residues import period_integral
    >>> x = symbols('x')
    >>> a = Symbol('a', positive=True)
    >>> period_integral(1/(1 - 2*a*cos(x) + a**2), x, 0, 2*pi, a < 1)
    ConditionalValue(2*pi/(1 - a**2))
    """
    e, lower, upper = to_expr(f), to_expr(a), to_expr(b)
    k = _periods(lower, upper)
    if k is None or not e.has(*_TRIGONOMETRIC):
        return None
    z = _fresh('z')
    g = expand_trig(e.rewrite(Cos).rewrite(Sin))
    for node in g.atoms(*_TRIGONOMETRIC):
        if node.has(x) and not (isinstance(node, (Sin, Cos)) and node.args[0] == x):
            return None
    inverse = pow(z, Integer(-1))
    rule: dict[Basic, Basic] = {sin(x): mul(add(z, mul(Integer(-1), inverse)), pow(mul(Integer(2), I), Integer(-1))),
                                cos(x): mul(add(z, inverse), S.Half)}
    substituted = mul(_replace(g, rule), pow(mul(I, z), Integer(-1)))
    if substituted.has(x):
        return None
    fraction = rational_function(substituted, z)
    if fraction is None:
        return None
    found = poles(fraction.denominator)
    if found is None:
        return None
    locator = _Locator(assumptions)
    total: Expr = S.Zero
    for pole in found:
        inside = locator.inside_unit_circle(pole)
        if inside is None:
            return None
        if inside:
            value = _residue(fraction, pole, S.One, z)
            if value is None:
                return None
            total = add(total, value)
    value = mul(Integer(2), pi, I, Integer(k), total)
    return _checked(ConditionalValue(_tidy(value, assumptions)), e, x, lower, upper, assumptions, locator)


# ---------------------------------------------------------------------------
# Verification and the entry point

#: the samples of the parameters at which a value placed by sampling is
#: checked (those of definite's verify_numerically)
_SAMPLES = 2


def _quadrature(f: Expr, x: Symbol, a: Expr, b: Expr) -> Optional[tuple[complex, float]]:
    """``Integral(f, (x, a, b))`` of an integrand free of parameters with
    an error bound: mpmath's two rules when they agree to ten digits
    (:func:`.recognize.quadrature` at fifteen), else the validated
    quadrature (a proved enclosure to eight digits, slower: a minute for a
    rational function over the real line to ten); ``None`` when neither
    settles it."""
    from .recognize import quadrature
    rough = attempt(lambda: quadrature(f, x, a, b, 15), _budget(), native=True)
    if rough is not None:
        return (complex(rough), 1e-10 * (1 + abs(float(rough))))
    from .validated import validated_integral
    budget = None if settings.timeout is None else settings.timeout / 4
    found = attempt(lambda: validated_integral(f, x, a, b, 8), budget, native=True)
    if found is None:
        return None
    value, error = found
    center = numeric_value(value, 60)
    bound = numeric_value(error, 60)
    if center is None or bound is None:
        return None
    return (complex(center), float(abs(bound)))


def _verified(value: Expr, f: Expr, x: Symbol, a: Expr, b: Expr, assumptions: Assumptions) -> Optional[bool]:
    """Whether ``value`` agrees with a quadrature of ``Integral(f, (x, a,
    b))`` at samples of the parameters satisfying the assumptions:
    ``False`` when they disagree, ``True`` when they agree, ``None`` when
    nothing could be checked."""
    parameters = _sorted_symbols((f.free_symbols | a.free_symbols | b.free_symbols) - {x})
    rng = random.Random(str((f, a, b)))
    verdict: Optional[bool] = None
    for _ in range(_SAMPLES if parameters else 1):
        values = sample_values(parameters, assumptions, rng)
        if values is None:
            return None
        rule: dict[Basic, Basic] = {s: v for s, v in values.items()}
        integrand, lower, upper = _replace(f, rule), _replace(a, rule), _replace(b, rule)
        if integrand.free_symbols - {x} or lower.free_symbols or upper.free_symbols:
            return None
        expected = _quadrature(integrand, x, lower, upper)
        if expected is None:
            return None
        ours = reliable_complex(value, 20, values)
        if ours is None:
            return None
        center, error = expected
        if abs(ours - center) > error + 1e-6 * (1 + abs(center)):
            return False
        verdict = True
    return verdict


def _checked(result: ConditionalValue, f: Expr, x: Symbol, a: Expr, b: Expr,
             assumptions: Assumptions, locator: _Locator) -> Optional[ConditionalValue]:
    """The result, verified numerically when a pole was placed by
    sampling (the numerical checks are then on): by the native
    quadratures, else by those of :mod:`.definite` through the face."""
    if not locator.sampled:
        return result
    verdict = _verified(result.value, f, x, a, b, assumptions)
    if verdict is None:
        from .definite import verify_numerically
        verdict = verify_numerically(result.value, f, x, a, b, assumptions)
    if verdict is False:
        return None
    return result


def residue_integral(f: ExprLike, x: Symbol, a: ExprLike, b: ExprLike,
                     assumptions: Assumptions = None) -> Optional[ConditionalValue]:
    """``Integral(f, (x, a, b))`` by the residue theorem when ``f`` is one
    of the families of the module documentation; ``None`` otherwise.

    Parameters
    ==========

    f : Expr
    x : Symbol
    a, b : Expr
        The bounds: ``(-oo, oo)``, ``(0, oo)`` or a range of length
        ``2*pi*k``.
    assumptions : Boolean or list of Booleans, optional
        Assumptions on the parameters, used to place the poles.

    Returns
    =======

    A :class:`~.conditions.ConditionalValue`: the value and the condition
    on the parameters (the sign of ``k`` in a Fourier integral, the
    convergence conditions on the exponent of a power).

    Examples
    ========

    >>> from sympy_extras.core import symbols, oo, pi
    >>> from sympy_extras.functions import sin, log
    >>> from sympy_extras.integrals.residues import residue_integral
    >>> x = symbols('x')
    >>> residue_integral(x*sin(x)/(x**2 + 1), x, -oo, oo)
    ConditionalValue(pi*exp(-1))
    >>> residue_integral(log(x)**2/(x**2 + 1), x, 0, oo)
    ConditionalValue(pi**3/8)
    >>> residue_integral(1/(5 + 3*sin(x)), x, 0, 2*pi)
    ConditionalValue(pi/2)
    >>> residue_integral(1/(x**2 + 1), x, 0, 1) is None
    True
    """
    e, lower, upper = to_expr(f), to_expr(a), to_expr(b)
    if not e.has(x):
        return None
    constant, rest = e.as_independent(x, as_Add=False)
    if constant != 1:
        found = residue_integral(rest, x, lower, upper, assumptions)
        return None if found is None else found.scaled(constant)
    if lower == S.NegativeInfinity and upper == oo:
        if rest.has(Exp, Sin, Cos):
            return fourier_integral(rest, x, assumptions)
        return real_line_integral(rest, x, assumptions)
    if lower == 0 and upper == oo:
        if rest.has(Exp, Sin, Cos):
            return fourier_integral(rest, x, assumptions, half=True)
        if rest.has(*_TRIGONOMETRIC):
            return None
        return keyhole_integral(rest, x, assumptions)
    if lower.is_number and upper.is_number and _periods(lower, upper) is not None:
        return period_integral(rest, x, lower, upper, assumptions)
    return None
