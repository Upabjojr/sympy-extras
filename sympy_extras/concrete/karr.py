r"""Karr's algorithm for indefinite summation.

:func:`karr_sum` and :func:`karr_term` extend Gosper's algorithm
(:func:`sympy.concrete.gosper.gosper_sum`) from hypergeometric terms to
summands built from rational functions, hypergeometric terms (factorials,
binomials, powers) and *sums* like harmonic numbers or nested sums, that is
to summands living in a $\Pi\Sigma$-field (see
:mod:`sympy_extras.concrete.pisigma`). The summand is analysed, a
$\Pi\Sigma$-field containing it is built by adjoining a generator for every
product or sum which is not already expressible in the field, and the
telescoping equation $g(k + 1) - g(k) = f(k)$ is solved with Karr's
algorithm.

Examples
========

>>> from sympy import harmonic, factorial, binomial
>>> from sympy.abc import k, n
>>> from sympy_extras.concrete import karr_sum
>>> karr_sum(harmonic(k), (k, 1, n))
n*harmonic(n) - n + harmonic(n)
>>> karr_sum(k*harmonic(k), (k, 1, n))
n*(2*n*harmonic(n) - n + 2*harmonic(n) + 1)/4
>>> karr_sum(harmonic(k)/k, (k, 1, n))
(harmonic(n)**2 + harmonic(n, 2))/2
>>> karr_sum(k*factorial(k), (k, 1, n))
n*factorial(n) + factorial(n) - 1
>>> karr_sum(binomial(2*k, k)/4**k, (k, 0, n))
(2*n + 1)*binomial(2*n, n)/4**n
>>> karr_sum(2**k/k, (k, 1, n)) is None
True

The last example is a *proof* that $\sum_{k=1}^n 2^k/k$ has no closed form
in terms of rational functions of $n$ and $2^n$.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence, Union

from sympy.concrete.products import Product
from sympy.concrete.summations import Sum, summation as _sympy_summation
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.traversal import preorder_traversal
from sympy.core.expr import Expr
from sympy.core.exprtools import factor_terms
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.numbers import harmonic
from sympy.polys.polytools import Poly
from sympy.core.relational import Eq
from sympy.logic.boolalg import Boolean, true
from sympy.core.numbers import Integer, Rational, nan, zoo
from sympy.polys.rationaltools import together
from sympy.polys.polyerrors import PolynomialError
from sympy.polys.polytools import cancel, factor_list
from sympy.simplify.simplify import hypersimp

from sympy.polys.fields import FracElement

from sympy_extras._numeric import reliable_value
from sympy_extras._special_values import Point, with_special_values
from sympy_extras._timeout import attempt
from sympy_extras.settings import settings
from sympy_extras.assumptions.ask import Assumptions

from sympy_extras._typing import as_boolean, as_expr, as_symbol, free_symbols, sorted_symbols

from .pisigma import PiSigmaField

__all__ = ['karr_term', 'karr_sum', 'summation', 'build_pisigma_field']


def _atoms(expr: Expr, k: Symbol) -> set[Expr]:
    """The subexpressions of ``expr`` depending on ``k`` which are not
    rational operations: functions of ``k``, powers with ``k`` in the
    exponent or with a non-integer exponent, sums and products."""
    expr = sympify(expr)
    if not expr.has(k) or expr == k:
        return set()
    if isinstance(expr, (Add, Mul)):
        return set().union(*[_atoms(a, k) for a in expr.args])
    if isinstance(expr, Pow):
        if expr.exp.is_Integer and not expr.exp.has(k):
            return _atoms(expr.base, k)
        return {expr}
    return {expr}


def _linear_shift(arg: Expr, k: Symbol) -> Optional[int]:
    """``c`` if ``arg == k + c`` with ``c`` an integer, else ``None``."""
    c = arg - k
    if c.is_Integer:
        return int(c)
    return None


def _linear(arg: Expr, k: Symbol) -> Optional[tuple[int, int]]:
    """``(a, b)`` if ``arg == a*k + b`` with ``a`` a positive integer and
    ``b`` an integer, else ``None``."""
    arg = sympify(arg).expand()
    a = arg.coeff(k)
    b = (arg - a*k).expand()
    if a.is_Integer and a > 0 and b.is_Integer:
        return int(a), int(b)
    return None


def _evaluate_at(expr: Expr, k: Symbol, k0: int) -> Optional[Expr]:
    value = expr.subs(k, k0)
    try:
        value = value.doit()
    except Exception:
        return None
    if value.has(S.NaN, S.ComplexInfinity, S.Infinity, S.NegativeInfinity):
        return None
    return value


class _Builder:
    """Builds a $\\Pi\\Sigma$-field containing given expressions."""

    def __init__(self, k: Symbol, params: Sequence[Symbol]) -> None:
        self.k = k
        self.field = PiSigmaField(k, params)
        # atom -> expression in k and the generator symbols
        self.known: dict[Expr, Expr] = {}

    def convert(self, expr: Union[Expr, int]) -> FracElement:
        """The element of the field equal to ``expr``, adjoining
        generators as needed."""
        expr = sympify(expr)
        atoms = sorted(_atoms(expr, self.k), key=lambda a: (a.count_ops(), str(a)))
        for atom in atoms:
            if atom not in self.known:
                self.known[atom] = self._represent(atom)
        rewritten = expr.xreplace(self.known)
        try:
            return self.field.from_expr(rewritten)
        except (ValueError, PolynomialError, TypeError) as e:
            raise ValueError("%s is not a rational expression in the summation "
                             "index, the parameters and the sequences of the "
                             "field: %s" % (expr, e))

    def _represent(self, atom: Expr) -> Expr:
        k = self.k
        # sums and products with a shifted upper limit are expressed through
        # the unshifted ones
        canonical, shift = self._canonical(atom)
        if shift:
            if canonical not in self.known:
                self.known[canonical] = self._represent(canonical)
            element = self.field.from_expr(self.known[canonical])
            return self.field.to_expr(self.field.sigma(element, shift), substitute=False)
        if isinstance(atom, harmonic):
            order = atom.args[1] if len(atom.args) > 1 else S.One
            linear = _linear(atom.args[0], k)
            if linear is not None and not order.has(k):
                a, b = linear
                # H_{a(k+1)+b} - H_{ak+b} = sum of the a reciprocals in between
                beta = sum(1/(a*k + b + i)**order for i in range(1, a + 1))
                return self._sigma(atom, beta)
        if isinstance(atom, Sum) and len(atom.limits) == 1:
            j, lo, hi = atom.limits[0]
            if hi == k and not lo.has(k) and not atom.function.has(k):
                return self._sigma(atom, atom.function.subs(j, k + 1))
        if isinstance(atom, Product) and len(atom.limits) == 1:
            j, lo, hi = atom.limits[0]
            if hi == k and not lo.has(k) and not atom.function.has(k):
                return self._pi(atom, atom.function.subs(j, k + 1))
        ratio = hypersimp(atom, k)
        if ratio is not None and ratio != 0:
            return self._pi(atom, ratio)
        raise ValueError("%s is not a hypergeometric term or an indefinite sum "
                         "in %s" % (atom, k))

    def _canonical(self, atom: Expr) -> tuple[Expr, int]:
        """``(atom with k in place of k + c, c)`` for harmonic numbers, sums
        and products with argument or upper limit ``k + c``."""
        k = self.k
        if isinstance(atom, harmonic):
            linear = _linear(atom.args[0], k)
            if linear is not None:
                a, b = linear
                b0 = b % a
                if b != b0:
                    return atom.func(a*k + b0, *atom.args[1:]), (b - b0)//a
        if isinstance(atom, (Sum, Product)) and len(atom.limits) == 1:
            j, lo, hi = atom.limits[0]
            shift = _linear_shift(hi, k)
            if shift and not lo.has(k) and not atom.function.has(k):
                return atom.func(atom.function, (j, lo, k)), shift
        return atom, 0

    def _initial_constant(self, atom: Expr, w_expr: Expr, ratio: bool = False) -> Optional[Expr]:
        """The constant ``atom - w`` (or ``atom/w``) as an element of the
        constants, by evaluation at a point, or ``None``."""
        for k0 in range(0, 12):
            a0 = _evaluate_at(atom, self.k, k0)
            w0 = _evaluate_at(w_expr, self.k, k0)
            if a0 is None or w0 is None:
                continue
            if ratio:
                if w0 == 0:
                    continue
                c = cancel(a0/w0)
            else:
                c = cancel(a0 - w0)
            try:
                self.field.C.from_sympy(c)
            except Exception:
                return None
            # the identity must hold for all k: check at another point
            k1 = k0 + 1
            a1 = _evaluate_at(atom, self.k, k1)
            w1 = _evaluate_at(w_expr, self.k, k1)
            if a1 is not None and w1 is not None:
                check = cancel(a1/w1 - c) if ratio else cancel(a1 - w1 - c)
                if check != 0:
                    return None
            return c
        return None

    def _sigma(self, atom: Expr, beta: Expr) -> Expr:
        b = self.convert(beta)
        w = self.field.telescope(b)
        if w is not None:
            c = self._initial_constant(atom, self.field.to_expr(w))
            if c is not None:
                return self.field.to_expr(w, substitute=False) + c
        return self.field.add_sigma(self.field.to_expr(b, substitute=False), atom)

    def _pi(self, atom: Expr, alpha: Expr) -> Expr:
        a = self.convert(alpha)
        sols = self.field.solve(a, [])
        if sols:
            for _, w in sols:
                if w != self.field.zero:
                    c = self._initial_constant(atom, self.field.to_expr(w), ratio=True)
                    if c is not None:
                        return c*self.field.to_expr(w, substitute=False)
        return self.field.add_pi(self.field.to_expr(a, substitute=False), atom)


def _auto_extensions(f: Expr, k: Symbol) -> list[Expr]:
    """Harmonic numbers to adjoin when the telescoping fails: for a summand
    with harmonic numbers or nested sums of total degree ``d`` and highest
    order ``m``, the harmonic numbers of orders up to ``m + d``."""
    f = sympify(f)
    atoms = [a for a in _atoms(f, k) if isinstance(a, (harmonic, Sum))]
    if not atoms:
        # a rational summand with no closed form in the rational functions
        # sums to harmonic numbers: partial fractions give orders up to
        # the multiplicity of the poles (sum 1/k is harmonic(n),
        # sympy-extras#34)
        numerator, denominator = f.as_numer_denom()
        if not denominator.has(k) or not f.is_rational_function(k):
            return []
        multiplicity = max((int(m) for _, m in Poly(denominator, k).factor_list()[1]), default=0)
        return [harmonic(k, m) for m in range(1, multiplicity + 1)]
    order = 1
    for a in atoms:
        if isinstance(a, harmonic) and len(a.args) > 1 and a.args[1].is_Integer:
            order = max(order, int(a.args[1]))
    degree = 0
    for a in atoms:
        degree = max(degree, _degree_in(f, a))
    return [harmonic(k, m) for m in range(1, order + max(degree, 1) + 1)]


def _degree_in(expr: Expr, atom: Expr) -> int:
    """The degree of ``expr`` in ``atom`` (an upper bound)."""
    expr = sympify(expr)
    if expr == atom:
        return 1
    if not expr.has(atom):
        return 0
    if isinstance(expr, Add):
        return max(_degree_in(a, atom) for a in expr.args)
    if isinstance(expr, Mul):
        return sum(_degree_in(a, atom) for a in expr.args)
    if isinstance(expr, Pow) and expr.exp.is_Integer:
        return abs(int(expr.exp))*_degree_in(expr.base, atom)
    return 1


def build_pisigma_field(f: Union[Expr, int], k: Symbol, extensions: Sequence[Expr] = ()) -> tuple[PiSigmaField, FracElement]:
    r"""A $\Pi\Sigma$-field containing ``f`` and the ``extensions``, and
    the element of the field equal to ``f``.

    Returns ``(field, element)``. The symbols of ``f`` other than ``k`` are
    parameters (constants).

    Examples
    ========

    >>> from sympy import harmonic, factorial
    >>> from sympy.abc import k
    >>> from sympy_extras.concrete.karr import build_pisigma_field
    >>> F, f = build_pisigma_field(harmonic(k + 1)*factorial(k), k)
    >>> F.extensions
    [Extension(pi, factorial(k), k + 1), Extension(sigma, harmonic(k), 1/(k + 1))]
    >>> F.to_expr(f)
    (k*harmonic(k) + harmonic(k) + 1)*factorial(k)/(k + 1)
    """
    f_ = as_expr(f)
    k_ = as_symbol(k)
    symbols = free_symbols(f_)
    for e in extensions:
        symbols |= free_symbols(sympify(e))
    params = sorted_symbols(symbols - {k_})
    builder = _Builder(k_, params)
    for e in extensions:
        builder.convert(e)
    element = builder.convert(f_)
    field = builder.field
    field.known_atoms = list(builder.known)
    return field, element


def _telescope(f: Union[Expr, int], k: Symbol, extensions: Sequence[Expr], auto: bool
               ) -> tuple[PiSigmaField, FracElement, Optional[FracElement]]:
    """The field, the element and a telescoper for ``f``, trying the
    automatic extensions if the first attempt fails."""
    f_ = as_expr(f)
    field, element = build_pisigma_field(f_, k, extensions)
    g = field.telescope(element)
    if g is None and auto:
        extra = [e for e in _auto_extensions(f_, k)
                 if e not in extensions and e not in field.known_atoms]
        if extra:
            field, element = build_pisigma_field(f_, k, list(extensions) + extra)
            g = field.telescope(element)
    return field, element, g


def karr_term(f: Union[Expr, int], k: Symbol, extensions: Sequence[Expr] = (), auto: bool = True,
              special_values: bool = True) -> Optional[Expr]:
    r"""Indefinite sum of ``f`` by Karr's algorithm: ``g`` with
    ``g.subs(k, k + 1) - g == f``, or ``None`` if there is none in the
    $\Pi\Sigma$-field built from ``f``.

    Parameters
    ==========

    f : Expr
        The summand: a rational expression in ``k``, hypergeometric terms
        (factorials, binomials, powers, ``RisingFactorial``, ...),
        harmonic numbers ``harmonic(k + c, m)`` and sums or products with
        upper limit ``k + c`` (``c`` an integer). Other symbols are
        parameters.
    k : Symbol
        The summation index.
    extensions : iterable of Expr, optional
        Sequences to adjoin to the field even if they do not occur in ``f``,
        when the closed form needs them.
    auto : bool
        If the telescoping fails and ``f`` contains harmonic numbers or
        nested sums, retry with the harmonic numbers of the orders up to
        the highest order plus the degree of ``f`` in the sums adjoined
        (for instance $\sum_k H_k/k = (H_k^2 + H_k^{(2)})/2$ needs
        $H_k^{(2)}$).
    special_values : bool
        Add a case for each isolated value of the parameters at which the
        indefinite sum is undefined, the sum computed again there
        (:func:`with_sum_special_values`); the cases whose sum has no
        closed form there are left out.

    Examples
    ========

    >>> from sympy import harmonic, factorial
    >>> from sympy.abc import k
    >>> from sympy_extras.concrete import karr_term
    >>> karr_term(harmonic(k), k)
    k*(harmonic(k) - 1)
    >>> karr_term(k*factorial(k), k)
    factorial(k)
    >>> karr_term(factorial(k)/k, k) is None
    True
    >>> karr_term(harmonic(k)/(k + 1), k)
    (harmonic(k)**2 - harmonic(k, 2))/2

    The value of a parameter at which the indefinite sum is undefined
    gets a case of its own:

    >>> from sympy.abc import y
    >>> karr_term(y**k, k)
    Piecewise((k, Eq(y, 1)), (y**k/(y - 1), True))
    """
    f_ = as_expr(f)
    field, element, g = _telescope(f_, k, extensions, auto)
    if g is None:
        return None
    value = field.to_expr(g)
    if not special_values:
        return value

    def at_point(point: Point, at: Assumptions) -> Optional[Expr]:
        return karr_term(as_expr(f_.xreplace(point)), k, [as_expr(e.xreplace(point)) for e in extensions], auto)

    return with_sum_special_values(value, f_, [k], [], at_point, keep_unevaluated=False)


def karr_sum(f: Union[Expr, int], k: Union[Symbol, Sequence[Union[Symbol, Expr, int]]],
             extensions: Sequence[Expr] = (), auto: bool = True, special_values: bool = True) -> Optional[Expr]:
    r"""Definite sum $\sum_{k=a}^{b} f(k)$ by Karr's algorithm.

    ``k`` is ``(k, a, b)``; the result is $g(b + 1) - g(a)$ where ``g`` is
    the indefinite sum of :func:`karr_term`, following Karr's summation
    convention (:class:`sympy.concrete.summations.Sum`). ``None`` is
    returned if there is no closed form in the $\Pi\Sigma$-field built from
    the summand (and the ``extensions``). With ``k`` a symbol the
    indefinite sum is returned (:func:`karr_term`). With
    ``special_values`` (the default) the isolated values of the
    parameters of the summand at which the closed form is undefined get
    cases of their own (:func:`with_sum_special_values`).

    Examples
    ========

    >>> from sympy import harmonic, factorial, binomial
    >>> from sympy.abc import k, n
    >>> from sympy_extras.concrete import karr_sum
    >>> karr_sum(harmonic(k)**2, (k, 1, n))
    n*harmonic(n)**2 - 2*n*harmonic(n) + 2*n + harmonic(n)**2 - harmonic(n)
    >>> karr_sum(1/(k*(k + 1)), (k, 1, n))
    n/(n + 1)
    >>> karr_sum(k**2*2**k, (k, 0, n))
    2*(2**n*n**2 - 2*2**n*n + 3*2**n - 3)
    """
    f_ = as_expr(f)
    a: Optional[Expr] = None
    b: Optional[Expr] = None
    if isinstance(k, Symbol):
        index = k
    else:
        index_, a_, b_ = k
        index = as_symbol(index_)
        a, b = as_expr(a_), as_expr(b_)
    if a is None or b is None:
        return karr_term(f_, index, extensions, auto, special_values)
    field, element, g = _telescope(f_, index, extensions, auto)
    if g is None:
        return None
    # g(b + 1) is sigma(g) at b, which keeps the generators unshifted
    upper = field.to_expr(field.sigma(g)).subs(index, b)
    lower = field.to_expr(g).subs(index, a)
    closed = as_expr(factor_terms(cancel(upper.doit() - lower.doit())))
    repaired = _repaired_at_poles(closed, field.to_expr(g), f_, index, a, b)
    if not special_values:
        return repaired
    lower_, upper_ = a, b

    def at_point(point: Point, at: Assumptions) -> Optional[Expr]:
        return karr_sum(as_expr(f_.xreplace(point)), (index, lower_, upper_),
                        [as_expr(e.xreplace(point)) for e in extensions], auto)

    return with_sum_special_values(repaired, f_, [index], [a, b], at_point)


def with_sum_special_values(value: Expr, f: Expr, indices: Sequence[Symbol], limits: Sequence[Expr],
                            compute: Callable[[Point, Assumptions], Optional[Expr]],
                            keep_unevaluated: bool = True,
                            defined: Optional[Callable[[Point], bool]] = None) -> Expr:
    """The value of a sum of ``f`` with a case for each isolated value of
    the parameters of the summand (not the ``indices``, nor the symbols
    of the ``limits``, whose small values the algorithms settle
    themselves) at which it is undefined, the sum computed again there
    by ``compute`` (:mod:`sympy_extras._special_values`): the sum of
    ``y**k`` is ``n + 1`` at ``y = 1``.

    ``indices`` starts with the index of summation and ``limits`` are its
    bounds ``(lower, upper)``, or are empty for an indefinite sum (an
    antidifference in the index). A point where the sum itself is not
    defined gets no case: one where the summand is ``nan`` or ``zoo``, or
    has a pole at an integer of the range of summation (``1/((k + a)*(k +
    b))`` from ``k = 0`` at ``a = 0``), or which ``defined`` refutes (a
    point outside the region of convergence of a series). A summand with
    powers of ``0`` at the point, which the algorithms do not take
    (``binomial(n, k)*y**k/(k + 1)`` at ``y = 0``), is summed by the
    values of ``0**(c*k + d)``: ``1`` where the exponent is zero, at the
    lower limit, and ``0`` beyond it. With the
    numerical checks of the settings each case is compared with the sum
    computed term by term, for a few values of the upper limit (or, for
    an antidifference, with the summand at a few values of the index),
    the other parameters at sample values, through
    :func:`~sympy_extras._numeric.reliable_value`; a case which disagrees
    is left out. ``keep_unevaluated`` is that of
    :func:`~sympy_extras._special_values.with_special_values`.

    >>> from sympy.abc import k, n, y
    >>> from sympy_extras.concrete import karr_sum
    >>> karr_sum(y**k, (k, 0, n))
    Piecewise((n + 1, Eq(y, 1)), ((y*y**n - 1)/(y - 1), True))
    """
    excluded: set[Symbol] = set(indices)
    for limit in limits:
        excluded |= free_symbols(limit)
    parameters = sorted(free_symbols(f) - excluded, key=lambda s: s.name)
    if not parameters:
        return value
    index = indices[0]
    bounds = (limits[0], limits[1]) if len(limits) == 2 else None
    indefinite = not limits and len(indices) == 1

    def defined_at(point: Point) -> bool:
        term = as_expr(f.xreplace(point))
        if term.has(nan, zoo):
            return False
        if bounds is not None and _pole_in_range(term, index, bounds[0], bounds[1]):
            return False
        return defined is None or defined(point)

    def checked(point: Point, at: Assumptions) -> Optional[Expr]:
        term = as_expr(f.xreplace(point))
        if bounds is not None and _zero_powers(term, index):
            special = _sum_of_zero_powers(term, index, bounds[0], bounds[1])
        else:
            special = compute(point, at)
        if special is None or not settings.numerical_checks:
            return special
        if _disagrees(special, as_expr(f.xreplace(point)), index, bounds, indefinite):
            return None
        return special

    return with_special_values(value, parameters, None, checked, defined_at, keep_unevaluated=keep_unevaluated)


#: the seconds given to the search for the poles of a summand at a point
_POLE_SECONDS = 2.0
#: the seconds given to the evaluation of a case at one sample point
_CHECK_SECONDS = 2.0
#: the sample values of the parameters left free by a point, in the checks
_SAMPLES = (Rational(17, 7), Rational(23, 9), Rational(31, 11), Rational(41, 13))
#: the number of values of the upper limit (or of the index) checked
_CHECKS = 3
#: the relative difference beyond which a case disagrees with the direct sum
_TOLERANCE = 1e-9


def _zero_powers(term: Expr, k: Symbol) -> list[Pow]:
    """The powers of ``0`` with an exponent in ``k`` in ``term``."""
    return [node for node in preorder_traversal(term)
            if isinstance(node, Pow) and as_expr(node.base).is_zero and as_expr(node.exp).has(k)]


def _sum_of_zero_powers(term: Expr, k: Symbol, lower: Expr, upper: Expr) -> Optional[Expr]:
    """The sum of ``term`` from ``lower`` to ``upper`` when its powers
    ``0**(c*k + d)`` (``c`` a positive integer, ``d`` a rational number)
    are ``0`` over the range except at the lower limit, where the exponent
    may vanish: the sum of ``term`` with them written ``0``, and the term
    at the lower limit, where they are ``1``, for its own; ``None`` when a
    power is not of this form or is infinite somewhere in the range."""
    if not isinstance(lower, Integer):
        return None
    zeroed: dict[Basic, Basic] = {}
    at_lower = False
    for power in _zero_powers(term, k):
        exponent = as_expr(power.exp)
        if not exponent.is_polynomial(k):
            return None
        polynomial = Poly(exponent, k)
        c, d = as_expr(polynomial.nth(1)), as_expr(polynomial.nth(0))
        if polynomial.degree() != 1 or not isinstance(c, Integer) or c <= 0 or not isinstance(d, Rational):
            return None
        vanishing = as_expr(-d / c)
        if vanishing > lower:
            # 0 to a negative power inside the range
            return None
        at_lower = at_lower or vanishing == lower
        zeroed[power] = S.Zero
    rest = as_expr(term.xreplace(zeroed))
    value = summation(rest, (k, lower, upper)) if rest.has(k) else as_expr(rest * (upper - lower + 1))
    if at_lower:
        value = as_expr(value + term.subs(k, lower) - rest.subs(k, lower))
    return value


def _pole_in_range(term: Expr, k: Symbol, lower: Expr, upper: Expr) -> bool:
    """Whether ``term`` has a pole at an integer ``k`` of the range from
    ``lower`` (an integer) to ``upper`` (an integer, else unbounded): a
    factor ``c*k + d`` of its denominator with an integer root there. For
    another lower limit, or a denominator which is no polynomial in ``k``,
    ``False``: nothing is known."""
    if not isinstance(lower, Integer):
        return False
    denominator = as_expr(together(term).as_numer_denom()[1])
    if not denominator.has(k) or not denominator.is_polynomial(k):
        return False
    factored = attempt(lambda: factor_list(denominator, k), _POLE_SECONDS)
    if factored is None:
        return False
    for factor_, _ in factored[1]:
        polynomial = Poly(factor_, k)
        if polynomial.degree() != 1:
            continue
        root = as_expr(cancel(-polynomial.nth(0) / polynomial.LC()))
        if isinstance(root, Integer) and root >= lower and (not isinstance(upper, Integer) or root <= upper):
            return True
    return False


def _upper_values(lower: Integer, upper: Expr) -> Optional[list[tuple[Point, int]]]:
    """The values of the symbol of ``upper`` (of the form ``c*m + d``,
    ``c`` a positive and ``d`` any integer, or an integer) at which a sum
    from ``lower`` is checked, with the upper limit there; ``None`` for
    another upper limit."""
    if isinstance(upper, Integer):
        return [({}, int(upper))]
    symbols = sorted_symbols(free_symbols(upper))
    if len(symbols) != 1 or not upper.is_polynomial(symbols[0]):
        return None
    m = symbols[0]
    polynomial = Poly(upper, m)
    c, d = polynomial.nth(1), polynomial.nth(0)
    if polynomial.degree() != 1 or not isinstance(c, Integer) or not isinstance(d, Integer) or c <= 0:
        return None
    # the first value of m with an upper limit of at least lower - 1 (the
    # empty sum) and nonnegative (the limits of definite_sum are for m >= 0)
    first = max(0, -((d - int(lower) + 1) // int(c)))
    values: list[tuple[Point, int]] = [({m: Integer(v)}, int(c) * v + int(d))
                                                   for v in range(first, first + _CHECKS)]
    return values


def _disagrees(value: Expr, term: Expr, k: Symbol, bounds: Optional[tuple[Expr, Expr]], indefinite: bool) -> bool:
    """Whether ``value``, the sum of ``term`` from ``bounds[0]`` to
    ``bounds[1]`` (or its antidifference in ``k``, when ``indefinite``),
    is refuted by the sum computed term by term at a few values of the
    upper limit (by the difference of the antidifference at a few values
    of ``k``), the other symbols at sample values. A value which cannot
    be evaluated reliably there (:func:`~sympy_extras._numeric.reliable_value`
    gives ``None``) is not refuted."""
    checks: list[tuple[Expr, Expr]] = []
    if bounds is not None:
        lower, upper = bounds
        if not isinstance(lower, Integer):
            return False
        uppers = _upper_values(lower, upper)
        if uppers is None:
            return False
        for at, top in uppers:
            direct = as_expr(Add(*[term.xreplace(at).subs(k, i) for i in range(int(lower), top + 1)]))
            checks.append((as_expr(value.xreplace(at)), direct))
    elif indefinite:
        for i in range(2, 2 + _CHECKS):
            checks.append((as_expr(value.subs(k, i + 1) - value.subs(k, i)), as_expr(term.subs(k, i))))
    else:
        return False
    for case, direct in checks:
        others = sorted_symbols((free_symbols(case) | free_symbols(direct)) - {k})
        samples: Point = {s: _SAMPLES[i % len(_SAMPLES)] + i // len(_SAMPLES)
                                      for i, s in enumerate(others)}
        evaluated = attempt(lambda: as_expr(case.xreplace(samples).doit()), _CHECK_SECONDS)
        if evaluated is None:
            continue
        difference = reliable_value(as_expr(evaluated - direct.xreplace(samples)), 15)
        size = reliable_value(as_expr(direct.xreplace(samples)), 15)
        if difference is None or size is None:
            continue
        if float(abs(difference)) > _TOLERANCE * (1 + float(abs(size))):
            return True
    return False


def _repaired_at_poles(closed: Expr, antidifference: Expr, f: Expr, index: Symbol,
                       a: Expr, b: Expr) -> Expr:
    """The closed form, with the values where it is false restored.

    The antidifference may carry a denominator in a *parameter* of the
    summand; at its nonnegative integer roots the antidifference is
    undefined and the closed form need not hold -- ``sum (-1)**k k**2
    binomial(n, k)`` is 0 for every ``n`` except 1 and 2, where the
    unconditional 0 was returned (sympy-extras#35). Those points are
    evaluated directly and written as a ``Piecewise``.
    """
    from sympy.functions.elementary.piecewise import Piecewise
    from sympy.polys.polyerrors import PolynomialError
    from sympy.polys.polyroots import roots
    parameters = sorted(free_symbols(antidifference) - {index}, key=lambda s: s.name)
    if len(parameters) != 1 or not isinstance(b, Symbol) or b != parameters[0]:
        return closed
    parameter = parameters[0]
    denominator = as_expr(together(antidifference).as_numer_denom()[1])
    try:
        poles = roots(Poly(denominator, parameter))
    except PolynomialError:
        return closed
    branches: list[tuple[Expr, Boolean]] = []
    for pole in sorted(poles, key=lambda r: str(r)):
        if not (isinstance(pole, Integer) and int(pole) >= 0):
            continue
        direct = Sum(f.subs(parameter, pole), (index, a, pole)).doit()
        if isinstance(direct, Sum) or not isinstance(direct, Expr):
            continue
        if cancel(as_expr(direct) - closed.subs(parameter, pole)) != 0:
            branches.append((as_expr(direct), as_boolean(Eq(parameter, pole))))
    if not branches:
        return closed
    return as_expr(Piecewise(*branches, (closed, true)))


def summation(f: Union[Expr, int], *symbols: Union[Symbol, Sequence[Union[Symbol, Expr, int]]],
              extensions: Sequence[Expr] = (), auto: bool = True, special_values: bool = True,
              **kwargs: object) -> Expr:
    r"""Summation with SymPy's :func:`~sympy.summation` and Karr's
    algorithm.

    The sum is first evaluated by SymPy; if it is not, and it is a sum over
    a single index of a summand in the scope of :func:`karr_sum`, Karr's
    algorithm is tried, and when that fails for a summand depending on a
    limit, creative telescoping (:func:`~.creative.definite_sum`). With
    ``special_values`` (the default) the isolated values of the
    parameters of the summand at which the result is undefined get cases
    of their own (:func:`with_sum_special_values`).

    Examples
    ========

    >>> from sympy import harmonic
    >>> from sympy.abc import k, n
    >>> from sympy_extras.concrete import summation
    >>> summation(k**2, (k, 1, n))
    n**3/3 + n**2/2 + n/6
    >>> summation(harmonic(k), (k, 1, n))
    n*harmonic(n) - n + harmonic(n)
    >>> summation(harmonic(k)/k, (k, 1, n))
    (harmonic(n)**2 + harmonic(n, 2))/2
    >>> summation(harmonic(k)/(n + 1 - k), (k, 1, n))
    harmonic(n + 1)**2 - harmonic(n + 1, 2)
    """
    result = as_expr(_sympy_summation(f, *symbols, **kwargs))
    if result.has(Sum):
        result = _karr_on_sums(result, extensions, auto)
    if not special_values:
        return result
    indices: list[Symbol] = []
    limits: list[Expr] = []
    for spec in symbols:
        if isinstance(spec, Symbol):
            indices.append(spec)
        else:
            indices.append(as_symbol(spec[0]))
            limits.extend(as_expr(e) for e in spec[1:])

    def at_point(point: Point, at: Assumptions) -> Optional[Expr]:
        at_symbols = [spec if isinstance(spec, Symbol) else tuple(as_expr(sympify(e)).xreplace(point) for e in spec)
                      for spec in symbols]
        return summation(as_expr(sympify(f)).xreplace(point), *at_symbols, extensions=extensions, auto=auto,
                         special_values=True, **kwargs)

    return with_sum_special_values(result, as_expr(sympify(f)), indices, limits, at_point)


def _karr_on_sums(expr: Expr, extensions: Sequence[Expr], auto: bool) -> Expr:
    def evaluate(s: Sum) -> Expr:
        if len(s.limits) != 1:
            return s
        k, a, b = s.limits[0]
        try:
            value = karr_sum(s.function, (k, a, b), extensions, auto, special_values=False)
        except ValueError:
            value = None
        if value is None and free_symbols(s.function) & (free_symbols(a) | free_symbols(b)) - {k}:
            # the summand depends on a limit: creative telescoping
            from .creative import definite_sum
            value = attempt(lambda: definite_sum(s.function, (k, a, b), extensions=extensions,
                                                 special_values=False),
                            settings.timeout)
        return s if value is None else value
    return as_expr(expr.replace(lambda e: isinstance(e, Sum), evaluate))
