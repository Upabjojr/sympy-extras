"""Indexed real roots of polynomials as nodes of the core.

``IndexedRoot(f, t, k)`` is the ``k``-th distinct real root, counted from
0 in increasing order, of the polynomial ``f`` in ``t``, the bound
variable, whose coefficients may depend on other symbols: a root function
of a cylindrical description (the extended Tarski formulas of [Brown]_,
Mathematica's parametric ``Root``), and with rational coefficients the
exact real algebraic number which the decompositions of
:mod:`sympy_extras.polys.cad` write their sample points and the ends of
their intervals with (SymPy's ``CRootOf``).

:class:`IndexedRoot` is the node, which never evaluates;
:func:`indexed_root` is the evaluating function: it writes the bound
variable as the dummy kept for its name (so that equal roots are equal
expressions, and a substitution of a symbol of that name around the root
does not reach it), and a root whose coefficients are rational as a
rational number, in radicals when :func:`~sympy_extras.polys.roots.radical_form`
allows, or as the canonical node ``IndexedRoot(m, t, i)`` of its minimal
polynomial ``m`` and its index ``i`` among the real roots of ``m``
(:func:`real_root`). The numerical value of such a node is that of the
real algebraic number (:func:`algebraic_of`), refined exactly.

The module is native: it imports no SymPy. The SymPy counterpart of the
node is :class:`sympy_extras.polys.cad.indexedroot_sympy.IndexedRoot`
(:mod:`sympy_extras.interop` converts them; a canonical node with
rational coefficients becomes SymPy's ``CRootOf``).

Examples
========

>>> from sympy_extras.core import symbols
>>> from sympy_extras.polys.cad.indexedroot import IndexedRoot, indexed_root
>>> x, y = symbols('x y')
>>> r = indexed_root(y**3 + x*y - 1, y, 0); r
IndexedRoot(x*y + y**3 - 1, y, 0)
>>> r.subs(x, 0), r.subs(x, 1).evalf(10), r.free_symbols
(1, 0.6823278038, {x})
>>> indexed_root(y**3 - 3*y + x, y, 2).subs(x, 5)     # one real root only
IndexedRoot(y**3 - 3*y + 5, y, 2)
>>> indexed_root(y**2 - 2, y, 1), indexed_root(y**5 - y - 1, y, 0).evalf(10)
(sqrt(2), 1.167303978)

References
==========

.. [Brown] C. W. Brown, *Solution formula construction for truth
   invariant CAD's*, PhD thesis, University of Delaware, 1999.
"""
from __future__ import annotations

from collections import OrderedDict
from fractions import Fraction
from typing import Final, Optional, Union

import mpmath

from sympy_extras.core.basic import Basic, ReplacementRule
from sympy_extras.core.evalf import numeric_value
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.numbers import Integer, Rational, number_from_fraction
from sympy_extras.core.printing import StrPrinter
from sympy_extras.core.symbol import Dummy, Symbol
from sympy_extras.polys.numberfields.qqbar import Algebraic
from sympy_extras.polys.polyerrors import BasePolynomialError
from sympy_extras.polys.polytools import Poly, _decompose_with_gens
from sympy_extras.polys.rootisolation import dup_count_real_roots

from .samplepoints import rational_real_roots

__all__ = ['IndexedRoot', 'indexed_root', 'real_root', 'real_number', 'algebraic_of', 'bound_variable']

#: the bound variables of the roots, by the name of the variable: one per
#: name, so that equal roots are equal expressions
_BOUND: dict[str, Dummy] = {}


def bound_variable(name: str) -> Dummy:
    """The bound variable of the roots in the variable called ``name``."""
    found = _BOUND.get(name)
    if found is None:
        found = Dummy(name, real=True)
        _BOUND[name] = found
    return found


class IndexedRoot(Expr):
    """The ``index``-th distinct real root, from 0 in increasing order, of
    the polynomial ``f`` in the bound variable ``t``, whose coefficients
    may depend on other symbols. The constructor checks and stores its
    arguments (:func:`indexed_root` evaluates).

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.polys.cad.indexedroot import IndexedRoot
    >>> x, y = symbols('x y')
    >>> r = IndexedRoot(y**2 - x, y, 1)
    >>> r, r.args, r.free_symbols, r.bound_symbols
    (IndexedRoot(-x + y**2, y, 1), (-x + y**2, y, 1), {x}, (y,))
    >>> r.subs(y, 3) == r, r.subs(x, 4)
    (True, 2)
    """

    __slots__ = ('_polynomial', '_variable', '_index')

    def __init__(self, f: Expr, t: Symbol, k: Union[int, Integer]) -> None:
        if not isinstance(f, Expr):
            raise TypeError("an expression is expected, got %s" % (f,))
        if not isinstance(t, Symbol):
            raise ValueError("IndexedRoot(f, t, k) takes a symbol t and an index k, from 0")
        index = k.p if isinstance(k, Integer) else k
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError("IndexedRoot(f, t, k) takes a symbol t and an index k, from 0")
        self._polynomial: Final[Expr] = f
        self._variable: Final[Symbol] = t
        self._index: Final[int] = index
        Basic.__init__(self)

    @property
    def args(self) -> tuple[Basic, ...]:
        return (self._polynomial, self._variable, Integer(self._index))

    @property
    def expr(self) -> Expr:
        """The polynomial, in the bound variable."""
        return self._polynomial

    @property
    def variable(self) -> Symbol:
        """The bound variable."""
        return self._variable

    @property
    def index(self) -> int:
        """The index of the root among the distinct real roots, from 0."""
        return self._index

    @property
    def free_symbols(self) -> set[Symbol]:
        return self._polynomial.free_symbols - {self._variable}

    @property
    def bound_symbols(self) -> tuple[Symbol, ...]:
        return (self._variable,)

    def _is_number_node(self) -> Optional[bool]:
        return not self.free_symbols

    def _rebuild(self, args: tuple[Basic, ...]) -> Basic:
        f, t, k = args
        if not (isinstance(f, Expr) and isinstance(t, Symbol) and isinstance(k, Integer)):
            raise TypeError("invalid arguments of an IndexedRoot: %s" % (args,))
        return IndexedRoot(f, t, k)

    def _evaluated(self, args: tuple[Basic, ...]) -> Basic:
        f, t, k = args
        if not (isinstance(f, Expr) and isinstance(t, Symbol) and isinstance(k, Integer)):
            raise TypeError("invalid arguments of an IndexedRoot: %s" % (args,))
        return indexed_root(f, t, k.p)

    def _subs(self, old: Basic, new: Basic) -> Basic:
        # the variable is bound: a substitution of it leaves the root alone
        if self == old:
            return new
        if old == self._variable:
            return self
        f = self._polynomial._subs(old, new)
        if f is self._polynomial:
            return self
        if not isinstance(f, Expr):
            raise TypeError("the substitution made %s a non-expression" % (self,))
        return indexed_root(f, self._variable, self._index)

    def _xreplace(self, rule: ReplacementRule) -> Basic:
        if self in rule:
            return rule[self]
        return self._replaced_polynomial(rule, False)

    def _xreplace_evaluated(self, rule: ReplacementRule) -> Basic:
        if self in rule:
            return rule[self]
        return self._replaced_polynomial(rule, True)

    def _replaced_polynomial(self, rule: ReplacementRule, evaluated: bool) -> Basic:
        """The root with ``rule`` applied to its polynomial, the bound
        variable left alone (it is a dummy of its own: no rule built from
        the symbols of a formula names it)."""
        f = self._polynomial._xreplace_evaluated(rule) if evaluated else self._polynomial._xreplace(rule)
        if f is self._polynomial:
            return self
        if not isinstance(f, Expr):
            raise TypeError("the replacement made %s a non-expression" % (self,))
        if evaluated:
            return indexed_root(f, self._variable, self._index)
        return IndexedRoot(f, self._variable, self._index)

    def _eval_facts(self) -> dict[str, bool]:
        facts = {'commutative': True, 'extended_real': True}
        if self.free_symbols:
            return facts
        a = algebraic_of(self)
        if a is None:
            return facts
        s = a.sign()
        facts.update({'real': True, 'finite': True, 'zero': s == 0, 'positive': s > 0, 'negative': s < 0})
        return facts

    def _numeric_node(self, prec: int) -> Optional[mpmath.mpc]:
        if self.free_symbols:
            return None
        a = algebraic_of(self)
        if a is not None:
            return _algebraic_value(a, prec)
        return _numerical_value(self._polynomial, self._variable, self._index, prec)

    def polynomial_root(self) -> Optional[tuple[Poly, int]]:
        """The polynomial and the index of the root as SymPy's ``CRootOf``
        has them (the minimal polynomial and the index among its real
        roots), for a root with rational coefficients which is not
        rational; ``None`` otherwise."""
        a = algebraic_of(self) if not self.free_symbols else None
        if a is None or a.is_rational:
            return None
        node = real_root(a, self._variable.name)
        if not isinstance(node, IndexedRoot):
            return None  # pragma: no cover
        # in a plain symbol, as SymPy's CRootOf shows its polynomial
        plain = Symbol(node.variable.name)
        return Poly(node.expr.xreplace({node.variable: plain}), plain), node.index

    def _sympystr(self, printer: StrPrinter) -> str:
        shown = Symbol(self._variable.name)
        # evaluated: the terms in the printing order of the plain symbol
        # (x**3 - 2*x**2 + x - 1 was printed -1 + x**3 - 2*x**2 + x)
        return "IndexedRoot(%s, %s, %s)" % (printer.doprint(self._polynomial._xreplace_evaluated({self._variable: shown})),
                                            printer.doprint(shown), self._index)


def _algebraic_value(a: Algebraic, prec: int) -> mpmath.mpc:
    """The value of a real algebraic number at ``prec`` bits: the middle of
    its isolating interval refined to a relative width below ``2**-prec``."""
    eps = Fraction(1, 2**(prec + 4))
    while True:
        lo, hi, _, _ = a.enclosure(eps)
        middle = (lo + hi)/2
        if middle == 0 or hi - lo <= abs(middle)*Fraction(1, 2**(prec + 2)) or eps < Fraction(1, 2**(4*prec + 64)):
            break
        eps = eps*abs(middle) if abs(middle) < 1 else eps/2
    with mpmath.workprec(prec):
        return mpmath.mpc(mpmath.mpf(middle.numerator)/middle.denominator)


def _coefficients(f: Expr, t: Symbol) -> Optional[list[Expr]]:
    """The coefficients of ``f`` as a polynomial in ``t``, the leading one
    first; ``None`` when ``f`` is not a polynomial in ``t``."""
    try:
        decomposed = _decompose_with_gens(f, [t])
    except BasePolynomialError:
        return None
    from sympy_extras.core.add import add
    by_degree = {m[0]: add(*cs) for m, cs in decomposed.items()}
    by_degree = {d: c for d, c in by_degree.items() if not (isinstance(c, Rational) and c.p == 0)}
    if not by_degree:
        return []
    n = max(by_degree)
    return [by_degree.get(n - i, Integer(0)) for i in range(n + 1)]


def _numerical_value(f: Expr, t: Symbol, index: int, prec: int) -> Optional[mpmath.mpc]:
    """The ``index``-th distinct real root of a polynomial with numerical
    coefficients which are not all rational, from its numerical roots. A
    multiple root (which is what the polynomial has over a section of its
    discriminant) is found with half of the working digits, slowly: twice
    the digits asked for, and as many steps as it takes."""
    coefficients = _coefficients(f, t)
    if coefficients is None or len(coefficients) < 2:
        return None
    digits = max(15, mpmath.libmp.prec_to_dps(prec))
    working = 2*digits + 10
    bits = mpmath.libmp.dps_to_prec(working + 10)
    values: list[mpmath.mpc] = []
    for c in coefficients:
        value = numeric_value(c, bits)
        if value is None:
            return None
        values.append(value)
    with mpmath.workdps(working):
        try:
            found = mpmath.polyroots(values, maxsteps=5000, extraprec=bits)
        except (mpmath.libmp.libhyper.NoConvergence, ValueError, ZeroDivisionError):
            return None
        tolerance = mpmath.mpf(10)**(3 - digits)
        reals: list[mpmath.mpf] = []
        for root in found:
            z = mpmath.mpc(root)
            if abs(z.imag) < tolerance*(1 + abs(z.real)):
                reals.append(z.real)
        reals.sort()
        distinct: list[mpmath.mpf] = []
        for value_ in reals:
            if not distinct or abs(value_ - distinct[-1]) > tolerance*(1 + abs(value_)):
                distinct.append(value_)
        if index >= len(distinct):
            return None
        result = distinct[index]
    with mpmath.workprec(prec):
        return mpmath.mpc(+result)


#: the real algebraic numbers of the canonical roots with rational
#: coefficients (the refinements of their intervals are kept)
_ALGEBRAIC: OrderedDict[IndexedRoot, Algebraic] = OrderedDict()
_KEPT = 20000


def _remember(node: IndexedRoot, a: Algebraic) -> None:
    _ALGEBRAIC[node] = a
    while len(_ALGEBRAIC) > _KEPT:
        _ALGEBRAIC.popitem(last=False)


def _rational_poly(f: Expr, t: Symbol) -> Optional[Poly]:
    """``f`` as a polynomial in ``t`` with rational coefficients, of
    positive degree; ``None`` otherwise."""
    try:
        poly = Poly(f, t)
    except BasePolynomialError:
        return None
    if poly.domain not in ('ZZ', 'QQ') or poly.degree() < 1:
        return None
    return poly


def algebraic_of(e: Expr) -> Optional[Algebraic]:
    """The real algebraic number of a rational number or of an
    :class:`IndexedRoot` with rational coefficients (which has that many
    real roots); ``None`` for any other expression.

    >>> from sympy_extras.core import Symbol, Rational
    >>> from sympy_extras.polys.cad.indexedroot import IndexedRoot, algebraic_of
    >>> y = Symbol('y')
    >>> algebraic_of(IndexedRoot(y**3 - 2*y, y, 2)), algebraic_of(Rational(1, 2))
    (Algebraic([1, 0, -2], 1.414213562), Algebraic([2, -1], 1/2))
    >>> algebraic_of(IndexedRoot(y**2 + 1, y, 0)) is None
    True
    """
    if isinstance(e, Rational):
        return Algebraic.from_rational(e.as_fraction())
    if not isinstance(e, IndexedRoot) or e.free_symbols:
        return None
    found = _ALGEBRAIC.get(e)
    if found is not None:
        return found
    poly = _rational_poly(e.expr, e.variable)
    if poly is None:
        return None
    roots = rational_real_roots(poly)
    if e.index >= len(roots):
        return None
    a = roots[e.index][0]
    _remember(e, a)
    return a


def real_root(a: Algebraic, name: str) -> Expr:
    """The real algebraic number ``a`` as an expression: a rational
    number, or the canonical :class:`IndexedRoot` of its minimal
    polynomial in the variable called ``name`` (no radicals: SymPy's
    ``CRootOf``).

    >>> from sympy_extras.polys.cad.samplepoints import real_roots_of
    >>> from sympy_extras.polys.cad.indexedroot import real_root
    >>> [real_root(r, 'x') for r in real_roots_of([1, 0, -3, 0, 2])]
    [IndexedRoot(x**2 - 2, x, 0), -1, 1, IndexedRoot(x**2 - 2, x, 1)]
    """
    if a.is_rational:
        return number_from_fraction(a.rational_value())
    t = bound_variable(name)
    minpoly = a.minpoly
    lo, _ = a.real_interval()
    index = dup_count_real_roots([Fraction(c) for c in minpoly], None, lo)
    d = len(minpoly) - 1
    poly = Poly.from_dict({(d - i,): c for i, c in enumerate(minpoly) if c}, t)
    node = IndexedRoot(poly.as_expr(), t, index)
    if node not in _ALGEBRAIC:
        _remember(node, a)
    return node


def real_number(a: Algebraic, name: str) -> Expr:
    """:func:`real_root` written in radicals when
    :func:`~sympy_extras.polys.roots.radical_form` allows (the roots of a
    quadratic polynomial, of a binomial, ...).

    >>> from sympy_extras.polys.cad.samplepoints import real_roots_of
    >>> from sympy_extras.polys.cad.indexedroot import real_number
    >>> [real_number(r, 'x') for r in real_roots_of([1, 0, -3, 0, 2]) + real_roots_of([1, 0, -1, -1])]
    [-sqrt(2), -1, 1, sqrt(2), IndexedRoot(x**3 - x - 1, x, 0)]
    """
    node = real_root(a, name)
    if not isinstance(node, IndexedRoot):
        return node
    from sympy_extras.polys.roots import radical_form
    form = radical_form(node)
    return node if form is None else form


def indexed_root(f: ExprLike, t: Symbol, k: int) -> Expr:
    """``IndexedRoot(f, t, k)`` evaluated: the bound variable written as
    the dummy of its name, and a root with rational coefficients written
    as :func:`real_number` writes it, when the polynomial has that many
    real roots (a polynomial made constant by the values of its
    parameters, or with fewer roots, stays a node).

    >>> from sympy_extras.core import symbols, Rational
    >>> from sympy_extras.polys.cad.indexedroot import indexed_root
    >>> x, y = symbols('x y')
    >>> indexed_root(y**2 - x, y, 0), indexed_root(y**2 - 4, y, 0), indexed_root(y - 2*x, y, 0).subs(x, Rational(1, 3))
    (IndexedRoot(-x + y**2, y, 0), -2, 2/3)
    """
    if not isinstance(t, Symbol) or isinstance(k, bool) or not isinstance(k, int) or k < 0:
        raise ValueError("IndexedRoot(f, t, k) takes a symbol t and an index k, from 0")
    expression = to_expr(f)
    bound = bound_variable(t.name)
    if t != bound:
        expression = expression.xreplace({t: bound})
    if expression.has(bound) and expression.free_symbols == {bound}:
        value = _numerical_root(expression, bound, k)
        if value is not None:
            return value
    return IndexedRoot(expression, bound, k)


def _numerical_root(expression: Expr, t: Symbol, index: int) -> Optional[Expr]:
    """The ``index``-th distinct real root of a polynomial with rational
    coefficients, when it has that many."""
    poly = _rational_poly(expression, t)
    if poly is None:
        return None
    roots = rational_real_roots(poly)
    if index >= len(roots):
        return None
    return real_number(roots[index][0], t.name)
