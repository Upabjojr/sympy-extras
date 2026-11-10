r"""Karr's algorithm for indefinite summation.

:func:`karr_sum` and :func:`karr_term` extend Gosper's algorithm
(:func:`~sympy_extras.concrete.gosper.gosper_sum`) from hypergeometric
terms to summands built from rational functions, hypergeometric terms
(factorials, binomials, powers) and *sums* like harmonic numbers or nested
sums, that is to summands living in a $\Pi\Sigma$-field (see
:mod:`sympy_extras.concrete.pisigma`). The summand is analysed, a
$\Pi\Sigma$-field containing it is built by adjoining a generator for every
product or sum which is not already expressible in the field, and the
telescoping equation $g(k + 1) - g(k) = f(k)$ is solved with Karr's
algorithm.

The module is written on the core (:mod:`sympy_extras.core`, the native
functions, polynomials and sums) and takes and returns its expressions;
:mod:`~sympy_extras.concrete.karr_sympy` is its face for SymPy
expressions, used by the modules still written on SymPy. One step is
not native yet and goes through SymPy for the call: the cases of the
isolated values of the parameters (:mod:`sympy_extras._special_values`),
for summands with parameters only. Creative telescoping
(:func:`~sympy_extras.concrete.creative.definite_sum`, native too) is the
last resort of :func:`summation`.

Examples
========

>>> from sympy_extras.core import symbols
>>> from sympy_extras.functions import harmonic, factorial, binomial
>>> from sympy_extras.concrete.karr import karr_sum
>>> k, n = symbols('k n')
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

from sympy_extras._timeout import attempt
from sympy_extras.core.add import Add
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.exprtools import factor_terms
from sympy_extras.core.logic import Boolean, true
from sympy_extras.core.mul import Mul
from sympy_extras.core.numbers import ComplexInfinity, Infinity, Integer, NaN, NegativeInfinity
from sympy_extras.core.power import Pow
from sympy_extras.core.relational import Eq
from sympy_extras.core.symbol import Symbol
from sympy_extras.functions.combinatorial.numbers import Harmonic, harmonic
from sympy_extras.functions.elementary.piecewise import piecewise
from sympy_extras.polys.polyerrors import BasePolynomialError
from sympy_extras.polys.polyroots import roots
from sympy_extras.polys.polytools import Poly, cancel
from sympy_extras.polys.rationaltools import together
from sympy_extras.settings import settings

from .expr_with_limits import Product, Sum
from .gosper import hypersimp
from .pisigma import PiSigmaElement, PiSigmaTower, parametric_pisigma_field, rational_pisigma_field
from .summations import summation as _native_summation

__all__ = ['karr_term', 'karr_sum', 'summation', 'build_pisigma_field', 'SumPoint', 'SumCompute']

#: a point of the parameters: their values
SumPoint = dict[Symbol, Expr]
#: the sum computed again at a point of the parameters, ``None`` if it has
#: no closed form there
SumCompute = Callable[[SumPoint], Optional[Expr]]
#: a summation index with its bounds, or the index alone (an indefinite sum)
SumSpec = Union[Symbol, Sequence[Union[Symbol, ExprLike]]]

ZERO = Integer(0)
ONE = Integer(1)


def _symbol(s: object) -> Symbol:
    if not isinstance(s, Symbol):
        raise TypeError("%s is not a symbol" % (s,))
    return s


def _bound(e: object) -> Expr:
    if isinstance(e, Expr):
        return e
    if isinstance(e, int):
        return Integer(e)
    raise TypeError("%s is not an expression" % (e,))


def _limits(spec: SumSpec) -> tuple[Symbol, Optional[tuple[Expr, Expr]]]:
    """The index and the bounds (``None`` for an indefinite sum)."""
    if isinstance(spec, Symbol):
        return spec, None
    if len(spec) != 3:
        raise ValueError("a summation limit is (k, a, b), not %s" % (spec,))
    return _symbol(spec[0]), (_bound(spec[1]), _bound(spec[2]))


def _sorted_symbols(symbols: set[Symbol]) -> list[Symbol]:
    return sorted(symbols, key=lambda s: s.name)


def _expr_args(e: Basic) -> list[Expr]:
    return [a for a in e.args if isinstance(a, Expr)]


def _ops(e: Expr) -> int:
    n = e.count_ops()
    return n.p if isinstance(n, Integer) else 0


def _atoms(expr: Expr, k: Symbol) -> set[Expr]:
    """The subexpressions of ``expr`` depending on ``k`` which are not
    rational operations: functions of ``k``, powers with ``k`` in the
    exponent or with a non-integer exponent, sums and products."""
    if not expr.has(k) or expr == k:
        return set()
    if isinstance(expr, (Add, Mul)):
        result: set[Expr] = set()
        for a in _expr_args(expr):
            result |= _atoms(a, k)
        return result
    if isinstance(expr, Pow) and isinstance(expr.exp, Integer):
        return _atoms(expr.base, k)
    return {expr}


def _linear_shift(arg: Expr, k: Symbol) -> Optional[int]:
    """``c`` if ``arg == k + c`` with ``c`` an integer, else ``None``."""
    c = arg - k
    if isinstance(c, Integer):
        return c.p
    return None


def _linear(arg: Expr, k: Symbol) -> Optional[tuple[int, int]]:
    """``(a, b)`` if ``arg == a*k + b`` with ``a`` a positive integer and
    ``b`` an integer, else ``None``."""
    arg = arg.expand()
    a = arg.coeff(k)
    b = (arg - a*k).expand()
    if isinstance(a, Integer) and a.p > 0 and isinstance(b, Integer):
        return a.p, b.p
    return None


def _evaluate_at(expr: Expr, k: Symbol, k0: int) -> Optional[Expr]:
    value = expr.subs(k, Integer(k0))
    try:
        value = value.doit()
    except Exception:
        return None
    if value.has(NaN, ComplexInfinity, Infinity, NegativeInfinity):
        return None
    return value


def _new_field(k: Symbol, params: Sequence[Symbol]) -> PiSigmaTower:
    if params:
        return parametric_pisigma_field(k, params)
    return rational_pisigma_field(k)


class _Builder:
    """Builds a $\\Pi\\Sigma$-field containing given expressions: the
    field grows (``field`` is replaced by the extended one) as the atoms
    of the expressions are represented in it."""

    def __init__(self, k: Symbol, params: Sequence[Symbol]) -> None:
        self.k: Symbol = k
        self.field: PiSigmaTower = _new_field(k, params)
        #: atom -> expression in k and the generator symbols
        self.known: dict[Expr, Expr] = {}

    def convert(self, expr: Expr) -> PiSigmaElement:
        """The element of the field equal to ``expr``, adjoining
        generators as needed."""
        atoms = sorted(_atoms(expr, self.k), key=lambda a: (_ops(a), str(a)))
        for atom in atoms:
            if atom not in self.known:
                self.known[atom] = self._represent(atom)
        rewritten = expr.xreplace(self.known)
        try:
            return self.field.from_expr(rewritten)
        except (ValueError, TypeError, ZeroDivisionError, BasePolynomialError) as e:
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
        if isinstance(atom, Harmonic):
            order = atom.args[1] if len(atom.args) > 1 else ONE
            linear = _linear(atom.args[0], k)
            if linear is not None and not order.has(k):
                a, b = linear
                # H_{a(k+1)+b} - H_{ak+b} = sum of the a reciprocals in between
                beta: Expr = ZERO
                for i in range(1, a + 1):
                    beta = beta + ONE/(a*k + b + i)**order
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
        if ratio is not None and ratio != ZERO:
            return self._pi(atom, ratio)
        raise ValueError("%s is not a hypergeometric term or an indefinite sum "
                         "in %s" % (atom, k))

    def _canonical(self, atom: Expr) -> tuple[Expr, int]:
        """``(atom with k in place of k + c, c)`` for harmonic numbers, sums
        and products with argument or upper limit ``k + c``."""
        k = self.k
        if isinstance(atom, Harmonic):
            linear = _linear(atom.args[0], k)
            if linear is not None:
                a, b = linear
                b0 = b % a
                if b != b0:
                    argument = a*k + b0
                    shifted = Harmonic(argument, *atom.args[1:]) if len(atom.args) > 1 else Harmonic(argument)
                    return shifted, (b - b0)//a
        if isinstance(atom, (Sum, Product)) and len(atom.limits) == 1:
            j, lo, hi = atom.limits[0]
            shift = _linear_shift(hi, k)
            if shift and not lo.has(k) and not atom.function.has(k):
                unshifted: Expr = (Sum(atom.function, (j, lo, k)) if isinstance(atom, Sum)
                                   else Product(atom.function, (j, lo, k)))
                return unshifted, shift
        return atom, 0

    def _is_constant_expr(self, c: Expr) -> bool:
        """Whether ``c`` is an element of the constants (a rational
        function of the parameters)."""
        try:
            return self.field.is_constant(self.field.from_expr(c))
        except (ValueError, TypeError, ZeroDivisionError):
            return False

    def _initial_constant(self, atom: Expr, w_expr: Expr, ratio: bool = False) -> Optional[Expr]:
        """The constant ``atom - w`` (or ``atom/w``) as an element of the
        constants, by evaluation at a point, or ``None``."""
        for k0 in range(0, 12):
            a0 = _evaluate_at(atom, self.k, k0)
            w0 = _evaluate_at(w_expr, self.k, k0)
            if a0 is None or w0 is None:
                continue
            if ratio:
                if w0 == ZERO:
                    continue
                c = cancel(a0/w0)
            else:
                c = cancel(a0 - w0)
            if not self._is_constant_expr(c):
                return None
            # the identity must hold for all k: check at another point
            k1 = k0 + 1
            a1 = _evaluate_at(atom, self.k, k1)
            w1 = _evaluate_at(w_expr, self.k, k1)
            if a1 is not None and w1 is not None:
                check = cancel(a1/w1 - c) if ratio else cancel(a1 - w1 - c)
                if check != ZERO:
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
        self.field, symbol = self.field.extend('sigma', b, atom)
        return symbol

    def _pi(self, atom: Expr, alpha: Expr) -> Expr:
        a = self.convert(alpha)
        sols = self.field.solve_embedded(a, [])
        if sols:
            for _, w in sols:
                if w != self.field.zero:
                    c = self._initial_constant(atom, self.field.to_expr(w), ratio=True)
                    if c is not None:
                        return c*self.field.to_expr(w, substitute=False)
        self.field, symbol = self.field.extend('pi', a, atom)
        return symbol


def _auto_extensions(f: Expr, k: Symbol) -> list[Expr]:
    """Harmonic numbers to adjoin when the telescoping fails: for a summand
    with harmonic numbers or nested sums of total degree ``d`` and highest
    order ``m``, the harmonic numbers of orders up to ``m + d``."""
    atoms = [a for a in _atoms(f, k) if isinstance(a, (Harmonic, Sum))]
    if not atoms:
        # a rational summand with no closed form in the rational functions
        # sums to harmonic numbers: partial fractions give orders up to
        # the multiplicity of the poles (sum 1/k is harmonic(n),
        # sympy-extras#34)
        numerator, denominator = f.as_numer_denom()
        if not denominator.has(k) or f.is_rational_function(k) is not True:
            return []
        try:
            factors = Poly(denominator, k).factor_list()[1]
        except (BasePolynomialError, ValueError, TypeError):
            return []
        multiplicity = max((m for _, m in factors), default=0)
        return [harmonic(k, Integer(m)) for m in range(1, multiplicity + 1)]
    order = 1
    for a in atoms:
        if isinstance(a, Harmonic) and len(a.args) > 1:
            m_ = a.args[1]
            if isinstance(m_, Integer):
                order = max(order, m_.p)
    degree = 0
    for a in atoms:
        degree = max(degree, _degree_in(f, a))
    return [harmonic(k, Integer(m)) for m in range(1, order + max(degree, 1) + 1)]


def _degree_in(expr: Expr, atom: Expr) -> int:
    """The degree of ``expr`` in ``atom`` (an upper bound)."""
    if expr == atom:
        return 1
    if not expr.has(atom):
        return 0
    if isinstance(expr, Add):
        return max(_degree_in(a, atom) for a in _expr_args(expr))
    if isinstance(expr, Mul):
        return sum(_degree_in(a, atom) for a in _expr_args(expr))
    if isinstance(expr, Pow) and isinstance(expr.exp, Integer):
        return abs(expr.exp.p)*_degree_in(expr.base, atom)
    return 1


def _build(f: Expr, k: Symbol, extensions: Sequence[Expr]) -> tuple[PiSigmaTower, PiSigmaElement, list[Expr]]:
    """The field, the element equal to ``f`` and the atoms represented in
    the field."""
    symbols = set(f.free_symbols)
    for e in extensions:
        symbols |= e.free_symbols
    builder = _Builder(k, _sorted_symbols(symbols - {k}))
    for e in extensions:
        builder.convert(e)
    element = builder.convert(f)
    return builder.field, element, list(builder.known)


def build_pisigma_field(f: ExprLike, k: Symbol, extensions: Sequence[Expr] = ()) -> tuple[PiSigmaTower, PiSigmaElement]:
    r"""A $\Pi\Sigma$-field containing ``f`` and the ``extensions``, and
    the element of the field equal to ``f``.

    Returns ``(field, element)``, the field a
    :class:`~sympy_extras.concrete.pisigma.PiSigmaField` (over the
    rationals, or over the rational functions of the parameters). The
    symbols of ``f`` other than ``k`` are parameters (constants).

    Examples
    ========

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.functions import harmonic, factorial
    >>> from sympy_extras.concrete.karr import build_pisigma_field
    >>> k = Symbol('k')
    >>> F, f = build_pisigma_field(harmonic(k + 1)*factorial(k), k)
    >>> F.extensions
    (Extension(pi, factorial(k), k + 1), Extension(sigma, harmonic(k), 1/(k + 1)))
    >>> F.to_expr(f)
    (k*harmonic(k) + harmonic(k) + 1)*factorial(k)/(k + 1)
    """
    field, element, _ = _build(to_expr(f), _symbol(k), extensions)
    return field, element


def _telescope(f: Expr, k: Symbol, extensions: Sequence[Expr], auto: bool
               ) -> tuple[PiSigmaTower, PiSigmaElement, Optional[PiSigmaElement]]:
    """The field, the element and a telescoper for ``f``, trying the
    automatic extensions if the first attempt fails."""
    field, element, known = _build(f, k, extensions)
    g = field.telescope(element)
    if g is None and auto:
        extra = [e for e in _auto_extensions(f, k) if e not in extensions and e not in known]
        if extra:
            field, element, _ = _build(f, k, list(extensions) + extra)
            g = field.telescope(element)
    return field, element, g


def karr_term(f: ExprLike, k: Symbol, extensions: Sequence[Expr] = (), auto: bool = True,
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
        (:func:`~sympy_extras.concrete.karr_sympy.with_sum_special_values`);
        the cases whose sum has no closed form there are left out.

    Examples
    ========

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.functions import harmonic, factorial
    >>> from sympy_extras.concrete.karr import karr_term
    >>> k = Symbol('k')
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

    >>> y = Symbol('y')
    >>> karr_term(y**k, k)
    Piecewise((k, Eq(y, 1)), (y**k/(y - 1), True))
    """
    f_ = to_expr(f)
    k_ = _symbol(k)
    field, element, g = _telescope(f_, k_, extensions, auto)
    if g is None:
        return None
    value = field.to_expr(g)
    if not special_values:
        return value

    def at_point(point: SumPoint) -> Optional[Expr]:
        return karr_term(f_.xreplace(point), k_, [e.xreplace(point) for e in extensions], auto)

    return _with_special_values(value, f_, [k_], [], at_point, keep_unevaluated=False)


def karr_sum(f: ExprLike, k: SumSpec, extensions: Sequence[Expr] = (), auto: bool = True,
             special_values: bool = True) -> Optional[Expr]:
    r"""Definite sum $\sum_{k=a}^{b} f(k)$ by Karr's algorithm.

    ``k`` is ``(k, a, b)``; the result is $g(b + 1) - g(a)$ where ``g`` is
    the indefinite sum of :func:`karr_term`, following Karr's summation
    convention (:class:`~sympy_extras.concrete.expr_with_limits.Sum`).
    ``None`` is returned if there is no closed form in the
    $\Pi\Sigma$-field built from the summand (and the ``extensions``).
    With ``k`` a symbol the indefinite sum is returned (:func:`karr_term`).
    With ``special_values`` (the default) the isolated values of the
    parameters of the summand at which the closed form is undefined get
    cases of their own
    (:func:`~sympy_extras.concrete.karr_sympy.with_sum_special_values`).

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.functions import harmonic
    >>> from sympy_extras.concrete.karr import karr_sum
    >>> k, n, y = symbols('k n y')
    >>> karr_sum(harmonic(k)**2, (k, 1, n))
    n*harmonic(n)**2 - 2*n*harmonic(n) + 2*n + harmonic(n)**2 - harmonic(n)
    >>> karr_sum(1/(k*(k + 1)), (k, 1, n))
    n/(n + 1)
    >>> karr_sum(k**2*2**k, (k, 0, n))
    2*(2**n*n**2 - 2*2**n*n + 3*2**n - 3)
    >>> karr_sum(y**k, (k, 0, n))
    Piecewise((n + 1, Eq(y, 1)), ((y*y**n - 1)/(y - 1), True))
    """
    f_ = to_expr(f)
    index, bounds = _limits(k)
    if bounds is None:
        return karr_term(f_, index, extensions, auto, special_values)
    a, b = bounds
    field, element, g = _telescope(f_, index, extensions, auto)
    if g is None:
        return None
    # g(b + 1) is sigma(g) at b, which keeps the generators unshifted
    upper = field.to_expr(field.sigma(g)).subs(index, b)
    lower = field.to_expr(g).subs(index, a)
    closed = factor_terms(cancel(upper.doit() - lower.doit()))
    repaired = _repaired_at_poles(closed, field.to_expr(g), f_, index, a, b)
    if not special_values:
        return repaired

    def at_point(point: SumPoint) -> Optional[Expr]:
        return karr_sum(f_.xreplace(point), (index, a, b), [e.xreplace(point) for e in extensions], auto)

    return _with_special_values(repaired, f_, [index], [a, b], at_point)


def _with_special_values(value: Expr, f: Expr, indices: Sequence[Symbol], limits: Sequence[Expr],
                         compute: SumCompute, keep_unevaluated: bool = True,
                         defined: Optional[Callable[[SumPoint], bool]] = None) -> Expr:
    """:func:`~sympy_extras.concrete.karr_sympy.with_sum_special_values`
    for core expressions. A summand without parameters (symbols other
    than the ``indices`` and those of the ``limits``) has no such values;
    the others go through SymPy (:mod:`sympy_extras._special_values` is
    not ported). ``defined`` refutes the points where the sum is not
    defined (outside the region of convergence of a series)."""
    excluded: set[Symbol] = set(indices)
    for limit in limits:
        excluded |= limit.free_symbols
    if not f.free_symbols - excluded:
        return value
    from .karr_sympy import sum_special_values_of_core
    return sum_special_values_of_core(value, f, indices, limits, compute, keep_unevaluated, defined)


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
    parameters = _sorted_symbols(antidifference.free_symbols - {index})
    if len(parameters) != 1 or not isinstance(b, Symbol) or b != parameters[0]:
        return closed
    parameter = parameters[0]
    denominator = together(antidifference).as_numer_denom()[1]
    try:
        poles = roots(Poly(denominator, parameter))
    except (BasePolynomialError, ValueError, TypeError):
        return closed
    if not isinstance(poles, dict):
        return closed
    branches: list[tuple[Expr, Boolean]] = []
    for pole in sorted(poles, key=lambda r: str(r)):
        if not (isinstance(pole, Integer) and pole.p >= 0):
            continue
        direct = Sum(f.subs(parameter, pole), (index, a, pole)).doit()
        if isinstance(direct, Sum):
            continue
        if cancel(direct - closed.subs(parameter, pole)) != ZERO:
            branches.append((direct, Eq(parameter, pole)))
    if not branches:
        return closed
    return piecewise(*branches, (closed, true))


def summation(f: ExprLike, *symbols: tuple[Symbol, ExprLike, ExprLike],
              extensions: Sequence[Expr] = (), auto: bool = True, special_values: bool = True) -> Expr:
    r"""Summation with :func:`~sympy_extras.concrete.summations.summation`
    (SymPy's ``summation``, ported) and Karr's algorithm.

    The sum is first evaluated by the ported ``summation``; if it is not,
    and it is a sum over a single index of a summand in the scope of
    :func:`karr_sum`, Karr's algorithm is tried, and when that fails for a
    summand depending on a limit, creative telescoping
    (:func:`~sympy_extras.concrete.creative.definite_sum`). With
    ``special_values`` (the default) the isolated values of the
    parameters of the summand at which the result is undefined get
    cases of their own
    (:func:`~sympy_extras.concrete.karr_sympy.with_sum_special_values`).

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.functions import harmonic
    >>> from sympy_extras.concrete.karr import summation
    >>> k, n = symbols('k n')
    >>> summation(k**2, (k, 1, n))
    n**3/3 + n**2/2 + n/6
    >>> summation(harmonic(k), (k, 1, n))
    n*harmonic(n) - n + harmonic(n)
    >>> summation(harmonic(k)/k, (k, 1, n))
    (harmonic(n)**2 + harmonic(n, 2))/2
    >>> summation(harmonic(k)/(n + 1 - k), (k, 1, n))
    harmonic(n + 1)**2 - harmonic(n + 1, 2)
    """
    f_ = to_expr(f)
    limits: list[tuple[Symbol, Expr, Expr]] = [(_symbol(s[0]), to_expr(s[1]), to_expr(s[2])) for s in symbols]
    result = _native_summation(f_, *limits)
    if result.has(Sum):
        result = _karr_on_sums(result, extensions, auto)
    if not special_values:
        return result
    indices = [s[0] for s in limits]
    bounds = [e for s in limits for e in s[1:]]

    def at_point(point: SumPoint) -> Optional[Expr]:
        at_limits = [(s[0], s[1].xreplace(point), s[2].xreplace(point)) for s in limits]
        return summation(f_.xreplace(point), *at_limits, extensions=extensions, auto=auto, special_values=True)

    return _with_special_values(result, f_, indices, bounds, at_point)


def _karr_on_sums(expr: Expr, extensions: Sequence[Expr], auto: bool) -> Expr:
    def evaluate(s: Basic) -> Basic:
        if not isinstance(s, Sum) or len(s.limits) != 1:
            return s
        k, a, b = s.limits[0]
        try:
            value = karr_sum(s.function, (k, a, b), extensions, auto, special_values=False)
        except ValueError:
            value = None
        if value is None and s.function.free_symbols & (a.free_symbols | b.free_symbols) - {k}:
            # the summand depends on a limit: creative telescoping, within
            # the time limit of the settings (imported here: creative is
            # built on this module)
            from .creative import definite_sum
            function = s.function
            value = attempt(lambda: definite_sum(function, (k, a, b), extensions=extensions, special_values=False),
                            settings.timeout, native=True)
        return s if value is None else value
    replaced = expr.replace(lambda e: isinstance(e, Sum), evaluate)
    assert isinstance(replaced, Expr)
    return replaced
