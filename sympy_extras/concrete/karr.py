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

from typing import Optional, Sequence, Union

from sympy.concrete.products import Product
from sympy.concrete.summations import Sum, summation as _sympy_summation
from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.exprtools import factor_terms
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.numbers import harmonic
from sympy.polys.polyerrors import PolynomialError
from sympy.polys.polytools import cancel
from sympy.simplify.simplify import hypersimp

from sympy.polys.fields import FracElement

from sympy_extras._typing import as_expr, as_symbol, free_symbols, sorted_symbols

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
        return []
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


def karr_term(f: Union[Expr, int], k: Symbol, extensions: Sequence[Expr] = (), auto: bool = True) -> Optional[Expr]:
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
    """
    field, element, g = _telescope(f, k, extensions, auto)
    if g is None:
        return None
    return field.to_expr(g)


def karr_sum(f: Union[Expr, int], k: Union[Symbol, Sequence[Union[Symbol, Expr, int]]],
             extensions: Sequence[Expr] = (), auto: bool = True) -> Optional[Expr]:
    r"""Definite sum $\sum_{k=a}^{b} f(k)$ by Karr's algorithm.

    ``k`` is ``(k, a, b)``; the result is $g(b + 1) - g(a)$ where ``g`` is
    the indefinite sum of :func:`karr_term`, following Karr's summation
    convention (:class:`sympy.concrete.summations.Sum`). ``None`` is
    returned if there is no closed form in the $\Pi\Sigma$-field built from
    the summand (and the ``extensions``). With ``k`` a symbol the
    indefinite sum is returned.

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
    field, element, g = _telescope(f_, index, extensions, auto)
    if g is None:
        return None
    if a is None or b is None:
        return field.to_expr(g)
    # g(b + 1) is sigma(g) at b, which keeps the generators unshifted
    upper = field.to_expr(field.sigma(g)).subs(index, b)
    lower = field.to_expr(g).subs(index, a)
    return factor_terms(cancel(upper.doit() - lower.doit()))


def summation(f: Union[Expr, int], *symbols: Union[Symbol, Sequence[Union[Symbol, Expr, int]]],
              extensions: Sequence[Expr] = (), auto: bool = True, **kwargs: object) -> Expr:
    r"""Summation with SymPy's :func:`~sympy.summation` and Karr's
    algorithm.

    The sum is first evaluated by SymPy; if it is not, and it is a sum over
    a single index of a summand in the scope of :func:`karr_sum`, Karr's
    algorithm is tried.

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
    """
    result = as_expr(_sympy_summation(f, *symbols, **kwargs))
    if not result.has(Sum):
        return result
    return _karr_on_sums(result, extensions, auto)


def _karr_on_sums(expr: Expr, extensions: Sequence[Expr], auto: bool) -> Expr:
    def evaluate(s: Sum) -> Expr:
        if len(s.limits) != 1:
            return s
        k, a, b = s.limits[0]
        try:
            value = karr_sum(s.function, (k, a, b), extensions, auto)
        except ValueError:
            return s
        return s if value is None else value
    return as_expr(expr.replace(lambda e: isinstance(e, Sum), evaluate))
