"""The Almkvist–Zeilberger algorithm: creative telescoping for definite
integrals of hyperexponential functions, the continuous analogue of
Zeilberger's algorithm (:mod:`sympy_extras.concrete.zeilberger`).

A function `F(x, t)` is *hyperexponential* in `x` and `t` when both
logarithmic derivatives

.. math::

    q = \\frac{\\partial_x F}{F}, \\qquad p = \\frac{\\partial_t F}{F}

are rational functions: products of powers of polynomials and of
exponentials of rational functions, such as `e^{-x^2 - t^2/x^2}`,
`x^a (1 - x)^b (1 - t x)^c` or `e^{-x t}/(x^2 + t^2)`. The algorithm
looks for rational functions `a_0(t), \\ldots, a_J(t)` of `t` alone, not
all zero, and a rational function `R(x, t)` with

.. math::

    \\sum_{j=0}^{J} a_j(t)\\, \\partial_t^j F = \\partial_x \\bigl(R F\\bigr).

Dividing by `F` and writing `p_j = \\partial_t^j F / F` (rational, by the
recursion `p_0 = 1`, `p_{j+1} = \\partial_t p_j + p\\, p_j`) this is the
first order linear differential equation

.. math::

    R' + q R = \\sum_j a_j p_j

for a rational `R` with the `a_j` as unknowns. As in Gosper's algorithm,
the denominator of `R` is bounded by a local analysis at every
irreducible factor `d` of the denominators of `q` and of the `p_j`: with
`k` the order of the pole of `q` at `d` and `m` the order of the pole of
`R`, the pole of `R' + q R` has order `m + 1` (`k \\le 1`, unless `k = 1`
and the residue of `q` is the positive integer `m`, when the leading terms
cancel) or `m + k` (`k \\ge 2`), which must not exceed the pole order of
the right hand side. The degree of the numerator is bounded likewise at
infinity. What remains is a linear system over `\\mathbb{Q}(t)` for the
coefficients of the numerator and the `a_j`; the order `J` is increased
until it has a solution.

Integrating over `x` gives, for the definite integral
`I(t) = \\int_a^b F\\, dx`, the linear differential equation

.. math::

    \\sum_j a_j(t)\\, I^{(j)}(t) = \\bigl[R F\\bigr]_{x=a}^{x=b},

whose right hand side vanishes at natural boundaries. The equation is
solved by the linear ODE solvers of :mod:`sympy_extras.solvers.linear_ode`
(SymPy's ``dsolve`` when they do not give a basis) and the constants are
fixed by the values of the integral (and its derivatives) at a convenient
point, computed by
:func:`~sympy_extras.integrals.definite.definite_integral`. This is the
*holonomic* method of definite integration: it applies to integrands
outside the tables of Mellin transforms, and the differential equation
is an answer in itself when it has no closed form solution. Chyzak's
algorithm, which does the same for general D-finite integrands (Bessel
functions, orthogonal polynomials, ...), is in
:mod:`sympy_extras.integrals.dfinite`.

The module works on the expressions of the core and imports no SymPy:
the polynomials in `x` have their coefficients in the field
`\\mathbb{Q}(i)(t, \\ldots)` of the typed polynomial layer (factored over
`\\mathbb{Q}` or `\\mathbb{Q}(i)`), the limits at the bounds are the native
ones. The steps which stay on SymPy are reached through the functions
ending in ``_of_core`` of :mod:`.telescoping_sympy`, imported when they are
needed: the values of the integral at a point
(:func:`~sympy_extras.integrals.definite.definite_integral`, not ported
yet), its numerical check (``verify_numerically``) and SymPy's ``dsolve``
of an equation which the native solvers leave unsolved.
:mod:`.telescoping_sympy` is the old interface on SymPy objects.

Examples
========

>>> from sympy_extras.core import symbols, oo
>>> from sympy_extras.functions import exp
>>> from sympy_extras.integrals.telescoping import almkvist_zeilberger, holonomic_ode
>>> x = symbols('x')
>>> t = symbols('t', positive=True)
>>> almkvist_zeilberger(exp(-x**2 - t**2/x**2), x, t)
DifferentialTelescoper([-4, 0, 1], 2/x)
>>> holonomic_ode(exp(-x**2)*exp(2*x*t), x, -oo, oo, t)
Eq(-2*t*I(t) + Derivative(I(t), t), 0)

References
==========

.. [AZ] G. Almkvist, D. Zeilberger, The method of differentiating under
   the integral sign, Journal of Symbolic Computation 10 (1990), 571–591.
.. [Koutschan] C. Koutschan, Advanced applications of the holonomic
   systems approach, PhD thesis, RISC, Johannes Kepler University Linz,
   2009, chapter 2.
.. [PWZ] M. Petkovšek, H. Wilf, D. Zeilberger, A = B, A K Peters, 1996,
   chapter 6 (the discrete algorithm, of which this is the analogue).
.. [Chyzak] F. Chyzak, An extension of Zeilberger's fast algorithm to
   general holonomic functions, Discrete Mathematics 217 (2000), 115–134.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Final, Optional, Sequence

from sympy_extras._timeout import attempt
from sympy_extras.assumptions.ask import Assumptions
from sympy_extras.assumptions.limits import limit
from sympy_extras.core.add import Add, add
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr, ExprLike, as_expr, to_expr
from sympy_extras.core.function import AppliedUndef, Function, diff, expand
from sympy_extras.core.mul import Mul, mul
from sympy_extras.core.numbers import I, Integer, NaN, ComplexInfinity, Number, Rational, nan, oo, zoo
from sympy_extras.core.power import Pow
from sympy_extras.core.relational import Eq
from sympy_extras.core.symbol import Dummy, Symbol
from sympy_extras.functions.elementary.exponential import Exp
from sympy_extras.functions.elementary.hyperbolic import Cosh, Coth, Csch, Sech, Sinh, Tanh
from sympy_extras.functions.elementary.piecewise import Piecewise
from sympy_extras.functions.elementary.trigonometric import Cos, Cot, Csc, Sec, Sin, Tan
from sympy_extras.polys.domains.gaussiandomains import QQ_I, GaussianRational
from sympy_extras.polys.domains.sparsefractionfield import SparseFracElement, SparseFractionField
from sympy_extras.polys.domains.sparsepolynomialring import RationalPolynomialRing, SparsePolynomialRing
from sympy_extras.polys.polyclasses import DensePoly
from sympy_extras.polys.polyerrors import BasePolynomialError
from sympy_extras.polys.polytools import Poly, cancel, gcd_list, lcm_list
from sympy_extras.polys.rationaltools import together
from sympy_extras.polys.rings import PolyElement, PolyRing
from sympy_extras.settings import settings
from sympy_extras.simplify.simplify import simplify
from .conditions import ConditionalValue

__all__ = ['DifferentialTelescoper', 'logarithmic_derivative', 'is_hyperexponential',
           'almkvist_zeilberger', 'holonomic_ode', 'holonomic_integral']

_ZERO = Integer(0)
_ONE = Integer(1)

#: the trigonometric and hyperbolic functions written as exponentials
_TRIGONOMETRIC: Final[tuple[type, ...]] = (Sin, Cos, Tan, Cot, Sec, Csc, Sinh, Cosh, Tanh, Coth, Sech, Csch)

#: what the polynomial layer raises on an expression it cannot take
_POLYNOMIAL_FAILURES = (BasePolynomialError, ValueError, TypeError, NotImplementedError, ZeroDivisionError)


class DifferentialTelescoper:
    """The output of the Almkvist–Zeilberger algorithm for ``F(x, t)``.

    Attributes
    ==========

    coefficients : list of Expr
        The polynomials ``a_0(t), ..., a_J(t)`` of the differential
        equation, without common factors.
    certificate : Expr
        The function ``R(x, t)`` with ``G = R*F`` (a rational function of
        ``x`` for a hyperexponential ``F``, and for a sum of such terms
        the sum of their certificates weighted by the terms).
    term, x, t
        The input.

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.integrals.telescoping import almkvist_zeilberger
    >>> x, t = symbols('x t')
    >>> telescoper = almkvist_zeilberger(1/(x**2 + t**2), x, t)
    >>> telescoper.order, telescoper.operator()
    (1, t*Derivative(I(t), t) + I(t))
    >>> telescoper.check()
    True
    """

    def __init__(self, term: Expr, x: Symbol, t: Symbol, coefficients: Sequence[Expr], certificate: Expr) -> None:
        self.term: Final[Expr] = term
        self.x: Final[Symbol] = x
        self.t: Final[Symbol] = t
        self.coefficients: Final[list[Expr]] = list(coefficients)
        self.certificate: Final[Expr] = certificate

    @property
    def order(self) -> int:
        return len(self.coefficients) - 1

    def operator(self, name: str = 'I') -> Expr:
        """``sum_j a_j(t) I^(j)(t)``, the left hand side of the equation
        satisfied by the integral over ``x``, in the function ``I(t)``."""
        f = Function(name)(self.t)
        return add(*[mul(a, diff(f, (self.t, j))) for j, a in enumerate(self.coefficients)])

    def check(self) -> bool:
        """Verify ``sum_j a_j d^j F/dt^j = d(R F)/dx`` by simplification."""
        F = self.term
        lhs = add(*[mul(a, diff(F, (self.t, j))) for j, a in enumerate(self.coefficients)])
        difference = (lhs - diff(self.certificate*F, self.x))/F
        if difference.has(*_TRIGONOMETRIC):
            difference = difference.rewrite(Exp)
        difference = cancel(expand(difference))
        return difference == _ZERO or simplify(difference) == _ZERO

    def __repr__(self) -> str:
        return "DifferentialTelescoper(%s, %s)" % (self.coefficients, self.certificate)


def logarithmic_derivative(F: Expr, x: Symbol) -> Optional[Expr]:
    """``diff(F, x)/F`` as a rational function of ``x`` (cancelled), or
    ``None`` when it is not one.

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.functions import exp
    >>> from sympy_extras.integrals.telescoping import logarithmic_derivative
    >>> x, t = symbols('x t')
    >>> logarithmic_derivative(x**t*exp(-x**2), x)
    (t - 2*x**2)/x
    >>> logarithmic_derivative(exp(exp(x)), x) is None
    True
    """
    if F == _ZERO:
        return None
    try:
        ratio = cancel(diff(F, x)/F)
        if ratio.has(Exp) and ratio.is_rational_function(x) is not True:
            # powers written as exponentials of logarithms
            ratio = cancel(expand(ratio, power_exp=True, log=True))
    except _POLYNOMIAL_FAILURES:
        return None
    if ratio.is_rational_function(x) is not True:
        return None
    return ratio


def is_hyperexponential(F: ExprLike, x: Symbol, t: Symbol) -> bool:
    """Whether both logarithmic derivatives of ``F`` are rational
    functions of ``x`` and ``t``.

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.functions import exp, sin
    >>> from sympy_extras.integrals.telescoping import is_hyperexponential
    >>> x, t = symbols('x t')
    >>> is_hyperexponential(exp(-x*t)*x**3/(x**2 + t**2), x, t)
    True
    >>> is_hyperexponential(exp(-x*t)*sin(x), x, t)
    False
    """
    F_ = to_expr(F)
    q = logarithmic_derivative(F_, x)
    p = logarithmic_derivative(F_, t)
    return (q is not None and p is not None and q.is_rational_function(x, t) is True
            and p.is_rational_function(x, t) is True)


# ---------------------------------------------------------------------------
# Rational functions of x over the field of the parameters: the domain
# QQ_I(t, ...) of the typed polynomial layer (one element type whether I
# occurs or not), factored over QQ or over QQ_I as SymPy's QQ(t) and
# QQ_I(t) factor

Element = SparseFracElement[GaussianRational]
KPoly = DensePoly[Element]


def _real(f: PolyElement[GaussianRational]) -> bool:
    return all(not c.y for c in f.terms.values())


class _GaussianPolynomialRing(SparsePolynomialRing[GaussianRational]):
    """``QQ_I[y1, ..., yk]`` whose fractions are cancelled as SymPy's
    ``QQ(y1, ...)`` cancels them when the coefficients are rational (the
    integer contents, the sign of the denominator), and with a monic
    denominator otherwise."""

    def __init__(self, names: Sequence[str]) -> None:
        super().__init__(QQ_I, names)
        self.rational: Final[RationalPolynomialRing] = RationalPolynomialRing(names)

    def _to_rational(self, f: PolyElement[GaussianRational]) -> PolyElement[Fraction]:
        return self.rational.ring.from_dict({m: c.x for m, c in f.terms.items()})

    def _from_rational(self, f: PolyElement[Fraction]) -> PolyElement[GaussianRational]:
        return self.ring.from_dict({m: GaussianRational(c) for m, c in f.terms.items()})

    def gcd(self, a: PolyElement[GaussianRational], b: PolyElement[GaussianRational]) -> PolyElement[GaussianRational]:
        if _real(a) and _real(b):
            return self._from_rational(self.rational.gcd(self._to_rational(a), self._to_rational(b)))
        return super().gcd(a, b)

    def cancel(self, f: PolyElement[GaussianRational], g: PolyElement[GaussianRational]
               ) -> tuple[PolyElement[GaussianRational], PolyElement[GaussianRational]]:
        if not f.terms:
            return f, self.one
        if _real(f) and _real(g):
            p, q = self.rational.cancel(self._to_rational(f), self._to_rational(g))
            return self._from_rational(p), self._from_rational(q)
        h = super().gcd(f, g)
        numerator, denominator = f.exquo(h), g.exquo(h)
        inverse = QQ_I.invert(denominator.LC)
        return numerator.mul_ground(inverse), denominator.mul_ground(inverse)


class _FunctionField(SparseFractionField[GaussianRational]):
    """The field ``QQ_I(y1, ..., yk)`` of the coefficients, with the
    factorisation of the polynomials over it in one more variable: over
    ``QQ`` (Gauss's lemma: the factors of the numerator over ``QQ[x, y]``
    which depend on ``x``) unless ``gaussian``, over ``QQ_I`` then."""

    def __init__(self, names: Sequence[str], gaussian: bool) -> None:
        super().__init__(_GaussianPolynomialRing(names))
        self.variables: Final[list[str]] = list(names)
        self.gaussian: Final[bool] = gaussian

    def factor_dense(self, f: list[Element]) -> tuple[Element, list[tuple[list[Element], int]]]:
        if not f:
            return self.zero, []
        ring = self.ring
        common = ring.one
        for c in f:
            common = ring.lcm(common, c.denom)
        n = len(f) - 1
        terms: dict[tuple[int, ...], GaussianRational] = {}
        for i, c in enumerate(f):
            for m, value in (c.numer*common.exquo(c.denom)).terms.items():
                terms[(n - i,) + m] = value
        variables = ['_kfx'] + self.variables
        found: list[tuple[dict[tuple[int, ...], GaussianRational], int]] = []
        if self.gaussian or not all(not c.y for c in terms.values()):
            from sympy_extras.polys.gaussianfactor import gaussian_factor_list
            G = PolyRing(QQ_I, variables)
            _, factors = gaussian_factor_list(G.from_dict(terms), False)
            found = [(dict(g.terms), k) for g, k in factors]
        else:
            rational = Poly.from_dict({m: c.x for m, c in terms.items()}, *[Symbol(v) for v in variables])
            _, poly_factors = rational.factor_list()
            found = [({m: GaussianRational(c) for m, c in g.rep.terms.items()}, k) for g, k in poly_factors]
        result: list[tuple[list[Element], int]] = []
        product = self.one
        for g, k in found:
            degree = max(m[0] for m in g)
            if degree == 0:
                continue
            by_degree: dict[int, dict[tuple[int, ...], GaussianRational]] = {}
            for m, value in g.items():
                by_degree.setdefault(m[0], {})[m[1:]] = value
            dense = [self.from_ring(ring.ring.from_dict(by_degree.get(degree - i, {}))) for i in range(degree + 1)]
            result.append((dense, k))
            product = self.mul(product, self.pow(dense[0], k))
        return self.quo(f[0], product), result


def _generators(e: Expr, x: Symbol, found: list[Expr]) -> None:
    """The generators of ``e`` as a rational function over ``QQ_I``
    (its symbols, and the other atoms such as ``pi`` or ``sqrt(2)``, taken
    as transcendental as SymPy's polynomial domains take them); ``x``
    excluded, ``ValueError`` when it occurs inside one."""
    if isinstance(e, (Add, Mul)):
        for a in e.args:
            if isinstance(a, Expr):
                _generators(a, x, found)
        return
    if isinstance(e, Pow) and isinstance(e.exp, Integer):
        _generators(e.base, x, found)
        return
    if isinstance(e, Rational) or e == I:
        return
    if e == x:
        return
    if isinstance(e, Number) or e.has(x):
        raise ValueError("%s is not a rational function of %s" % (e, x))
    if e not in found:
        found.append(e)


class _Context:
    """Rational functions of ``x`` with their coefficients in the field of
    the generators of the given expressions (the typed polynomials know
    their variables by name: every generator is renamed)."""

    def __init__(self, exprs: Sequence[Expr], x: Symbol, generators: Sequence[Expr] = (),
                 gaussian: bool = False) -> None:
        found: list[Expr] = list(generators)
        for e in exprs:
            _generators(e, x, found)
        found = sorted(found, key=lambda g: g.sort_key()) if not generators else found
        self.x: Final[Symbol] = x
        self.generators: Final[list[Expr]] = found
        self.gaussian: Final[bool] = gaussian or any(e.has(I) for e in exprs)
        names = ['_k%d' % i for i in range(len(found))] or ['_k']
        self.renamed: Final[dict[Basic, Basic]] = {g: Symbol(name) for g, name in zip(found, names)}
        self.restored: Final[dict[Basic, Basic]] = {Symbol(name): g for g, name in zip(found, names)}
        self.variable: Final[Symbol] = Symbol('_kx')
        self.renamed[x] = self.variable
        self.restored[self.variable] = x
        self.field: Final[_FunctionField] = _FunctionField(names, self.gaussian)

    def covers(self, exprs: Sequence[Expr]) -> bool:
        """Whether the expressions are rational functions over the field
        (whose elements may have Gaussian coefficients in any case)."""
        found: list[Expr] = list(self.generators)
        for e in exprs:
            _generators(e, self.x, found)
        return len(found) == len(self.generators)

    def extended(self, exprs: Sequence[Expr], gaussian: bool = False) -> _Context:
        """The context of the generators of ``exprs`` too."""
        found: list[Expr] = list(self.generators)
        for e in exprs:
            _generators(e, self.x, found)
        return _Context(exprs, self.x, found, self.gaussian or gaussian)

    def _internal(self, e: Expr) -> Expr:
        return as_expr(e.xreplace(self.renamed))

    def restore(self, e: Expr) -> Expr:
        return as_expr(e._xreplace_evaluated(self.restored))

    def element(self, e: Expr) -> Element:
        """An expression free of ``x`` as an element of the field."""
        return self.field.from_expr(self._internal(e))

    def element_expr(self, c: Element) -> Expr:
        return self.restore(self.field.to_expr(c))

    def poly(self, e: Expr) -> KPoly:
        """A polynomial in ``x`` with coefficients in the field."""
        return DensePoly.from_expr(expand(self._internal(e)), self.variable, self.field)

    def from_elements(self, coefficients: list[Element]) -> KPoly:
        return DensePoly(coefficients, self.field, self.variable)

    def constant(self, c: Element) -> KPoly:
        return self.from_elements([c])

    def poly_expr(self, p: KPoly) -> Expr:
        return self.restore(p.to_expr())

    def fraction(self, e: Expr) -> tuple[KPoly, KPoly]:
        """Numerator and denominator of a rational function of ``x``."""
        numerator, denominator = together(e).as_numer_denom()
        return self.poly(numerator), self.poly(denominator)

    def fraction_expr(self, numerator: KPoly, denominator: KPoly) -> Expr:
        return cancel(self.poly_expr(numerator)/self.poly_expr(denominator))

    def convert(self, p: KPoly, other: _Context) -> KPoly:
        """A polynomial of the context ``other`` in this one."""
        return self.poly(other.poly_expr(p)) if other is not self else p


def _context(exprs: Sequence[Expr], x: Symbol) -> Optional[_Context]:
    try:
        return _Context(exprs, x)
    except ValueError:
        return None


def _nullspace(rows: Sequence[Sequence[Expr]], columns: int) -> Optional[list[list[Expr]]]:
    """The canonical basis of the null space of a matrix of rational
    functions (SymPy's ``Matrix.nullspace``: a vector per free column of
    the reduced row echelon form, its free coordinate one), over the
    field of the generators of the entries; ``None`` when an entry is not
    a rational function."""
    K = _context([e for row in rows for e in row], Dummy('_kn'))
    if K is None:
        return None
    try:
        converted = [[K.element(e) for e in row] for row in rows]
    except _POLYNOMIAL_FAILURES:
        return None
    return [[K.element_expr(v) for v in vector] for vector in K.field.nullspace(converted, columns)]


def _positive_integer(e: Expr) -> Optional[int]:
    e_ = cancel(e)
    if isinstance(e_, Integer) and e_.p > 0:
        return e_.p
    return None


def _irreducibles(P: KPoly) -> list[tuple[KPoly, int]]:
    """``(factor, multiplicity)`` for the monic irreducible factors of
    ``P`` which depend on the variable."""
    _, factors = P.factor_list()
    return [(d.monic(), k) for d, k in factors if d.degree > 0]


def _denominator_bound(K: _Context, A: KPoly, B: KPoly, denominators: Sequence[KPoly]) -> KPoly:
    """The polynomial ``D`` the denominator of ``R`` divides, from the
    local analysis of ``R' + q R = S`` at the irreducible factors of the
    denominators of ``q = A/B`` and of the ``p_j`` (see the module
    documentation)."""
    # factor -> (order in B, max order in the p_j)
    factors: list[tuple[KPoly, int, int]] = [(d, k, 0) for d, k in _irreducibles(B)]
    for L in denominators:
        for d, k in _irreducibles(L):
            for i, (e, order_b, order_s) in enumerate(factors):
                if e == d:
                    factors[i] = (e, order_b, max(order_s, k))
                    break
            else:
                factors.append((d, 0, k))
    D = K.constant(K.field.one)
    for d, k, order_s in factors:
        if k <= 1:
            m = order_s - 1
            if k == 1:
                # a simple pole of q with a positive integer residue rho
                # allows a pole of order rho of R (the leading terms cancel)
                cofactor = B.quo(d)
                try:
                    residue: Optional[KPoly] = (A*cofactor.invert(d)).rem(d)
                except _POLYNOMIAL_FAILURES:
                    residue = None
                if residue is not None and residue.degree <= 0:
                    rho = _positive_integer(K.element_expr(residue.LC) if not residue.is_zero else _ZERO)
                    if rho is not None:
                        m = max(m, rho)
        else:
            m = order_s - k
        if m > 0:
            D = D*d**m
    return D


def _degree(P: KPoly) -> Optional[int]:
    """The degree, ``None`` (minus infinity) for the zero polynomial."""
    return None if P.is_zero else P.degree


def _numerator_degree(K: _Context, A: KPoly, B: KPoly, D: KPoly, degree_s: int) -> int:
    """The bound on the degree of the numerator ``N`` of ``R = N/D``
    from the behaviour at infinity."""
    degree_a = _degree(A)
    degree_q = None if degree_a is None else degree_a - B.degree
    if degree_q is not None and degree_q >= 0:
        n = degree_s + D.degree - degree_q
    elif degree_q == -1:
        n = degree_s + D.degree + 1
        # R' + q R ~ (n - deg D + rho_oo) x^(n - deg D - 1) may cancel
        rho = K.element_expr(K.field.quo(A.LC, B.LC))
        zero = _positive_integer(-rho)
        if zero is not None:
            n = max(n, D.degree + zero)
        elif rho == _ZERO:
            n = max(n, D.degree)
    else:
        n = degree_s + D.degree + 1
    return max(n, 0)


def _coefficient(P: KPoly, i: int) -> Element:
    return P.coeff(i)


def _telescope(q: Expr, p: Expr, x: Symbol, t: Symbol, order: int) -> Optional[tuple[list[Expr], Expr]]:
    """``(a_0, ..., a_J, R)`` for the given order ``J``, or ``None``."""
    ps: list[Expr] = [_ONE]
    for _ in range(order):
        ps.append(cancel(diff(ps[-1], t) + p*ps[-1]))
    K = _context([q] + ps, x)
    if K is None:
        return None
    try:
        A, B = K.fraction(q)
        fractions = [K.fraction(pj) for pj in ps]
    except _POLYNOMIAL_FAILURES:
        return None
    denominators = [Lj for _, Lj in fractions]
    D = _denominator_bound(K, A, B, denominators)
    degree_s = max(P.degree - Lj.degree for P, Lj in fractions if not P.is_zero)
    n = _numerator_degree(K, A, B, D, degree_s)
    L = K.constant(K.field.one)
    for _, Lj in fractions:
        L = L.lcm(Lj)
    # B L (N' D - N D') + A L N D = B D^2 sum_j a_j (L / L_j) P_j: a
    # column per coefficient of N = sum c_i x**i and per a_j
    columns: list[KPoly] = []
    x_ = K.from_elements([K.field.one, K.field.zero])
    D_prime = D.diff()
    for i in range(n + 1):
        monomial = x_**i
        derivative = K.from_elements([K.field.from_int(i)])*x_**(i - 1) if i else K.constant(K.field.zero)
        columns.append(B*L*(derivative*D - monomial*D_prime) + A*L*monomial*D)
    for P, Lj in fractions:
        columns.append(-(B*D*D*L.quo(Lj)*P))
    height = max(c.degree for c in columns) + 1
    if height <= 0:
        return None
    rows = [[_coefficient(c, height - 1 - r) if r >= height - 1 - c.degree else K.field.zero for c in columns]
            for r in range(height)]
    for vector in K.field.nullspace(rows, len(columns)):
        a_values = [K.element_expr(v) for v in vector[n + 1:]]
        if all(v == _ZERO for v in a_values):
            continue
        N_value = K.from_elements(list(reversed(vector[:n + 1])))
        R = K.fraction_expr(N_value, D)
        normalized, factor_ = _normalized(a_values, t)
        return normalized, cancel(R*factor_)
    return None


def _normalized(values: list[Expr], t: Symbol) -> tuple[list[Expr], Expr]:
    """The coefficients cleared of denominators and common factors, and
    the factor they were multiplied by (which the certificate gets too)."""
    denominators = [together(v).as_numer_denom()[1] for v in values]
    try:
        scale = lcm_list(denominators)
        scaled = [cancel(v*scale) for v in values]
        common = gcd_list([v for v in scaled if v != _ZERO])
    except _POLYNOMIAL_FAILURES:
        scale, scaled, common = _ONE, values, _ONE
    result = [cancel(v/common) for v in scaled]
    factor_ = cancel(scale/common)
    # a positive leading coefficient of the highest order term
    lead = result[-1]
    negative = lead.is_negative is True
    if not negative and lead.is_number is False and lead.is_polynomial(t) is True:
        try:
            negative = Poly(lead, t).LC().is_negative is True
        except _POLYNOMIAL_FAILURES:
            negative = False
    if negative:
        result = [-v for v in result]
        factor_ = -factor_
    return result, factor_


def _terms(F: Expr, x: Symbol, t: Symbol) -> list[Expr]:
    """``F`` as a sum of hyperexponential terms (trigonometric and
    hyperbolic functions rewritten as exponentials), or ``[]``."""
    G = F
    if G.has(*_TRIGONOMETRIC):
        G = G.rewrite(Exp)
    G = expand(G)
    terms = list(Add.make_args(G))
    if all(is_hyperexponential(term, x, t) for term in terms):
        return terms
    if is_hyperexponential(F, x, t):
        return [F]
    return []


def almkvist_zeilberger(F: ExprLike, x: Symbol, t: Symbol, max_order: int = 4,
                        min_order: int = 0) -> Optional[DifferentialTelescoper]:
    """The Almkvist–Zeilberger algorithm for a hyperexponential ``F(x, t)``,
    or a sum of hyperexponential terms sharing the same equation.

    Parameters
    ==========

    F : Expr
        The integrand, hyperexponential in ``x`` and ``t`` (see
        :func:`is_hyperexponential`); ``sin``, ``cos``, ``sinh``, ``cosh``
        are rewritten as exponentials and the terms telescoped one by one.
    x : Symbol
        The integration variable.
    t : Symbol
        The parameter.
    max_order, min_order : int
        The orders ``J`` of the differential equation tried.

    Returns
    =======

    A :class:`DifferentialTelescoper` with the coefficients ``a_j(t)`` and
    the certificate ``R``, or ``None`` when no equation of order at most
    ``max_order`` exists (or the terms of a sum have different ones).

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.functions import exp
    >>> from sympy_extras.integrals.telescoping import almkvist_zeilberger
    >>> x, t = symbols('x t')
    >>> telescoper = almkvist_zeilberger(1/(x**2 + t**2), x, t)
    >>> telescoper.coefficients, telescoper.certificate
    ([1, t], -x)
    >>> telescoper.check()
    True
    >>> almkvist_zeilberger(exp(-x**2)*exp(2*x*t), x, t).coefficients
    [-2*t, 1]
    """
    F_ = to_expr(F)
    terms = _terms(F_, x, t)
    if not terms:
        return None
    found: list[tuple[list[Expr], Expr, Expr]] = []
    for term in terms:
        q = logarithmic_derivative(term, x)
        p = logarithmic_derivative(term, t)
        if q is None or p is None:
            return None
        result: Optional[tuple[list[Expr], Expr]] = None
        for order in range(min_order, max_order + 1):
            result = _telescope(q, p, x, t, order)
            if result is not None:
                break
        if result is None:
            return None
        found.append((result[0], result[1], term))
    coefficients = found[0][0]
    for other, _, _ in found[1:]:
        if len(other) != len(coefficients) or any(cancel(a - b) != _ZERO for a, b in zip(other, coefficients)):
            # the terms of the sum satisfy different equations: the
            # least common left multiple is not computed
            return None
    if len(found) == 1:
        certificate = found[0][1]
    else:
        certificate = cancel(add(*[R*term for _, R, term in found])/F_)
    return DifferentialTelescoper(F_, x, t, coefficients, certificate)


# ---------------------------------------------------------------------------
# The differential equation of the integral, and its solution

def _boundary(G: Expr, x: Symbol, point: Expr, direction: str, assumptions: Assumptions) -> Optional[Expr]:
    value = attempt(lambda: limit(G, x, point, direction, assumptions), settings.timeout, native=True)
    if value is None or value.has(nan, zoo, oo, -oo, NaN, ComplexInfinity, Piecewise) or value.has(x):
        return None
    return value


def _equation(coefficients: Sequence[Expr], G: Expr, x: Symbol, a: Expr, b: Expr, t: Symbol,
              assumptions: Assumptions, name: str) -> Optional[Eq]:
    """``sum_j a_j I^(j) = [G]_a^b`` (the boundary values as limits),
    ``None`` when one of them is not finite."""
    upper = _boundary(G, x, b, '-', assumptions)
    lower = _boundary(G, x, a, '+', assumptions)
    if upper is None or lower is None:
        return None
    f = Function(name)(t)
    operator = add(*[mul(c, diff(f, (t, j))) for j, c in enumerate(coefficients)])
    return Eq(operator, simplify(upper - lower))


def holonomic_ode(F: ExprLike, x: Symbol, a: ExprLike, b: ExprLike, t: Symbol,
                  assumptions: Assumptions = None, name: str = 'I') -> Optional[Eq]:
    """The linear differential equation ``sum_j a_j(t) I^(j)(t) = [R F]_a^b``
    satisfied by ``I(t) = Integral(F, (x, a, b))``, with the boundary
    values taken as limits under the assumptions; ``None`` when no
    telescoper is found or a boundary limit is not finite.

    Examples
    ========

    >>> from sympy_extras.core import symbols, oo
    >>> from sympy_extras.functions import exp
    >>> from sympy_extras.integrals.telescoping import holonomic_ode
    >>> x, t = symbols('x t')
    >>> holonomic_ode(exp(-x**2)*exp(2*x*t), x, -oo, oo, t)
    Eq(-2*t*I(t) + Derivative(I(t), t), 0)
    """
    F_ = to_expr(F)
    telescoper = almkvist_zeilberger(F_, x, t)
    if telescoper is None:
        return None
    return _equation(telescoper.coefficients, telescoper.certificate*F_, x, to_expr(a), to_expr(b), t, assumptions,
                     name)


def _initial_values(F: Expr, x: Symbol, a: Expr, b: Expr, t: Symbol, t0: Expr, order: int,
                    assumptions: Assumptions) -> Optional[list[Expr]]:
    """``I^(j)(t0)`` for ``j < order`` by integrating ``d^j F/dt^j`` at
    ``t = t0`` (differentiation under the integral sign), or ``None``."""
    from sympy_extras.solvers.linear_ode import has_integral
    from .telescoping_sympy import definite_integral_of_core
    values: list[Expr] = []
    for j in range(order):
        integrand = as_expr(diff(F, (t, j)).subs(t, t0))
        value = definite_integral_of_core(integrand, x, a, b, assumptions, False, settings.timeout)
        if value is None or has_integral(value) or value.has(Piecewise, nan, zoo, oo, -oo):
            return None
        values.append(value)
    return values


def _general_solution(coefficients: Sequence[Expr], rhs: Expr, t: Symbol,
                      equation: Eq) -> Optional[tuple[Expr, list[Symbol]]]:
    """The general solution of ``sum_j a_j I^(j) = rhs`` with its
    constants: a basis of the homogeneous equation and a particular
    solution by variation of parameters (the native linear solvers),
    SymPy's ``dsolve`` when they do not find them."""
    from sympy_extras.solvers.factorization import variation_of_parameters
    from sympy_extras.solvers.linear_ode import LinearOperator, dsolve_linear, has_integral
    order = len(coefficients) - 1
    if order == 0:
        return cancel(rhs/coefficients[0]), []
    y = Function('_y')(t)
    homogeneous = add(*[mul(c, diff(y, (t, j))) for j, c in enumerate(coefficients)])
    basis = attempt(lambda: dsolve_linear(homogeneous, y, use_dsolve=False, special_values=False),
                    settings.timeout, native=True)
    if basis is not None and len(basis) == order and not any(has_integral(s) for s in basis):
        L = LinearOperator(list(coefficients), t)
        particular: Optional[Expr] = _ZERO
        if rhs != _ZERO:
            particular = attempt(lambda: variation_of_parameters(L, basis, rhs), settings.timeout, native=True)
        if particular is not None and not has_integral(particular):
            constants = [Dummy('_C%d' % (j + 1)) for j in range(order)]
            return add(particular, *[mul(c, s) for c, s in zip(constants, basis)]), list(constants)
    from .telescoping_sympy import dsolve_of_core
    return dsolve_of_core(equation, Function('I')(t))


def _solve_constants(equations: list[Expr], constants: list[Symbol]) -> Optional[list[Expr]]:
    """The unique solution of linear equations in the constants."""
    from sympy_extras.solvers.linsolve import solve_linear_unique
    from sympy_extras.solvers.recurr import _is_zero, _row_reduce
    found = solve_linear_unique(equations, constants)
    if found is not None:
        return found
    zeros: dict[Basic, Basic] = {c: _ZERO for c in constants}
    rows: list[list[Expr]] = []
    for e in equations:
        row = [diff(e, c) for c in constants]
        constant = as_expr(e._xreplace_evaluated(zeros))
        if any(v.has(*constants) for v in row):
            return None
        rows.append(row + [-constant])
    try:
        reduced, pivots = _row_reduce(rows)
    except _POLYNOMIAL_FAILURES:
        return None
    if pivots != list(range(len(constants))):
        return None
    if any(not _is_zero(row[-1]) for row in reduced[len(pivots):]):
        return None
    return [reduced[i][-1] for i in range(len(constants))]


def _solve(coefficients: Sequence[Expr], equation: Eq, F: Expr, x: Symbol, a: Expr, b: Expr, t: Symbol,
           assumptions: Assumptions) -> Optional[Expr]:
    """The solution of the equation of the integral with its constants
    fixed by the initial values at ``t0`` (``1``, ``2``, ``1/2``, ``3``,
    the first at which they are found), ``None`` when a step fails."""
    rhs = as_expr(equation.rhs)
    general = _general_solution(coefficients, rhs, t, equation)
    if general is None:
        return None
    solution, constants = general
    order = len(coefficients) - 1
    if len(constants) != order:
        return None
    if not constants:
        return solution
    for t0 in (_ONE, Integer(2), Rational(1, 2), Integer(3)):
        values = _initial_values(F, x, a, b, t, t0, order, assumptions)
        if values is None:
            continue
        equations = [as_expr(diff(solution, (t, j)).subs(t, t0)) - v for j, v in enumerate(values)]
        solved = attempt(lambda: _solve_constants(equations, constants), settings.timeout, native=True)
        if solved is None:
            continue
        assignment: dict[Basic, Basic] = dict(zip(constants, solved))
        return as_expr(solution._xreplace_evaluated(assignment))
    return None


def _verified(value: Expr, F: Expr, x: Symbol, a: Expr, b: Expr, assumptions: Assumptions) -> bool:
    """Whether the numerical check (when it is on) does not reject the
    value."""
    if not settings.numerical_checks:
        return True
    from .telescoping_sympy import verify_numerically_of_core
    return verify_numerically_of_core(value, F, x, a, b, assumptions) is not False


def holonomic_integral(F: ExprLike, x: Symbol, a: ExprLike, b: ExprLike, t: Symbol,
                       assumptions: Assumptions = None) -> Optional[ConditionalValue]:
    """``Integral(F, (x, a, b))`` as a function of the parameter ``t`` by
    the holonomic method: the differential equation of
    :func:`holonomic_ode` solved by the linear ODE solvers (SymPy's
    ``dsolve`` when they do not solve it), its constants fixed by the
    values of the integral and its derivatives at a point ``t0`` (``1``,
    ``2``, ``1/2``, ``3``, the first at which
    :func:`~sympy_extras.integrals.definite.definite_integral` gives them).

    The value holds where the integral converges and may be
    differentiated under the integral sign; it is checked numerically
    when ``settings.numerical_checks`` is on. ``None`` when a step fails.

    Examples
    ========

    >>> from sympy_extras.core import symbols, oo
    >>> from sympy_extras.functions import exp
    >>> from sympy_extras.integrals.telescoping import holonomic_integral
    >>> from sympy_extras.settings import configure
    >>> x = symbols('x')
    >>> t = symbols('t', positive=True)
    >>> with configure(numerical_checks=False):
    ...     holonomic_integral(x*exp(-t*x**2), x, 0, oo, t)
    ConditionalValue(1/(2*t))
    """
    from .marichev import tidy
    F_ = to_expr(F)
    a_, b_ = to_expr(a), to_expr(b)
    telescoper = almkvist_zeilberger(F_, x, t)
    if telescoper is None:
        return None
    equation = _equation(telescoper.coefficients, telescoper.certificate*F_, x, a_, b_, t, assumptions, 'I')
    if equation is None:
        return None
    solution = _solve(telescoper.coefficients, equation, F_, x, a_, b_, t, assumptions)
    if solution is None:
        return None
    value = tidy(solution, assumptions)
    if not _verified(value, F_, x, a_, b_, assumptions):
        return None
    return ConditionalValue(value)


def _order(lhs: Expr, unknown: AppliedUndef, t: Symbol) -> int:
    """The order of a linear differential expression in ``unknown``."""
    order = 0
    for j in range(1, 8):
        if lhs.has(diff(unknown, (t, j))):
            order = j
    return order
