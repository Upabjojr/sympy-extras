"""Chyzak's algorithm: creative telescoping for definite integrals of
D-finite integrands, the general case of the Almkvist–Zeilberger
algorithm of :mod:`sympy_extras.integrals.telescoping`.

A function `F(x, t)` is *D-finite* in `x` and in `t` when it satisfies
linear differential equations with polynomial coefficients in each
variable,

.. math::

    L_x F = \\sum_{i=0}^{r} c_i(x, t)\\, \\partial_x^i F = 0, \\qquad
    L_t F = \\sum_{j=0}^{s} d_j(x, t)\\, \\partial_t^j F = 0:

exponentials, Bessel functions, orthogonal polynomials, hypergeometric
functions and their products and compositions with linear arguments.
Every mixed derivative `\\partial_x^i \\partial_t^j F` is then a combination,
with rational coefficients, of finitely many functions. Two ways to
that closure are implemented: the *normal forms* modulo the two
annihilators (`i \\ge r` reduced with `L_x`, `j \\ge s` with `L_t`,
Leibniz's rule taking the derivatives of the coefficients; each
annihilator is the first linear dependency among the derivatives of `F`
in the closure below, with the other variable in the coefficient field,
so of least order there), which is exact
when the ideal of the two operators is rectangular; and, for the
algorithm itself, the closure of a *product of D-finite factors*
`f_i(\\text{argument}_i)` with polynomial arguments: the basis is the set
of products `\\prod_i f_i^{(m_i)}(\\text{argument}_i)`, `m_i < n_i`, and the
chain rule with the equation of each factor gives the derivatives, so
that the rank is right even when a factor depends on `x t` (where the
two annihilators overcount).

Chyzak's algorithm [Chyzak]_ looks for a *telescoper*, an operator
`P = \\sum_{k \\le J} a_k(t)\\, \\partial_t^k` in `t` alone, and a *certificate*
`G = \\sum_b g_b(x, t)\\, b` over the basis `b` of the closure, with
rational `g_b`, such that

.. math::

    P F = \\partial_x G.

Both sides are reduced to the normal forms, and the coefficients of the
basis derivatives are compared: with the ansatz of Koutschan
[Koutschan]_, `g_b = p_b(x)/D(x)^e` with `D` the product of the
irreducible factors of the denominators met in the derivatives of the
basis, `e = 1, 2`, and `p_b` polynomials of degree at most `N`, the
comparison is a linear system over `\\mathbb{Q}(t)` for the coefficients
of the `p_b` and the `a_k`; `J` and `N` are increased
until it has a solution with some `a_k \\ne 0` (a solution with all
`a_k = 0` is a certificate of nothing). Integrating over `x`, the
definite integral `I(t) = \\int_a^b F\\, dx` satisfies

.. math::

    \\sum_k a_k(t)\\, I^{(k)}(t) = \\bigl[G\\bigr]_{x=a}^{x=b},

which is solved as in
:func:`~sympy_extras.integrals.telescoping.holonomic_integral`, the
constants fixed by the values of the integral at a point
(:func:`~sympy_extras.integrals.definite.definite_integral`).
For a hyperexponential `F` (`r = s = 1`) the certificate is the single
rational function of the Almkvist–Zeilberger algorithm.

The module works on the expressions of the core and imports no SymPy:
the equations of the factors ``exp``, ``sin``, ``cos``, ``sinh``,
``cosh``, ``erf``, ``erfc``, ``erfi``, ``log``, the Bessel and Hankel
functions and the Airy functions are written down (as SymPy's
``expr_to_holonomic`` gives them where it has them), the equation of
another factor `f(u)` whose derivative is in the closure (``Ei``, ``Si``,
``Ci``, ``atan``, ``asin``) is that of `f'` composed with `d/du`, and for
:func:`annihilators` the closure also takes sums, rational arguments and
the powers `b^r` of rational functions with constant exponents (`b^r`
satisfies `b y' = r b' y`). The linear systems are solved over the field
of the parameters of the typed polynomial layer. :mod:`.dfinite_sympy`
is the old interface on SymPy objects.

Examples
========

>>> from sympy_extras.core import symbols, oo
>>> from sympy_extras.functions import exp, besselj
>>> from sympy_extras.integrals.dfinite import chyzak, dfinite_ode
>>> x = symbols('x')
>>> t = symbols('t', positive=True)
>>> chyzak(exp(-t*x)*besselj(0, x), x, t)
DFiniteTelescoper([t, t**2 + 1], {exp(-t*x)*besselj(0, x): t*x, -exp(-t*x)*besselj(1, x): x})
>>> dfinite_ode(exp(-t*x)*besselj(0, x), x, 0, oo, t)
Eq(t*I(t) + (t**2 + 1)*Derivative(I(t), t), 0)

References
==========

.. [Chyzak] F. Chyzak, An extension of Zeilberger's fast algorithm to
   general holonomic functions, Discrete Mathematics 217 (2000),
   115–134.
.. [Koutschan] C. Koutschan, A fast approach to creative telescoping,
   Mathematics in Computer Science 4 (2010), 259–266.
.. [AZ] G. Almkvist, D. Zeilberger, The method of differentiating under
   the integral sign, Journal of Symbolic Computation 10 (1990),
   571–591 (the hyperexponential case).
"""
from __future__ import annotations

from functools import lru_cache
from itertools import count
from math import comb
from typing import Callable, Final, Optional, Sequence

from sympy_extras._timeout import attempt
from sympy_extras.assumptions.ask import Assumptions
from sympy_extras.core.add import Add, add
from sympy_extras.core.expr import Expr, ExprLike, as_expr, to_expr
from sympy_extras.core.function import Application, Derivative, Function, diff, expand_func
from sympy_extras.core.mul import Mul, mul
from sympy_extras.core.numbers import Integer, nan, oo, zoo
from sympy_extras.core.power import Pow, pow
from sympy_extras.core.relational import Eq
from sympy_extras.core.symbol import Dummy, Symbol
from sympy_extras.functions.elementary.exponential import Exp, Log
from sympy_extras.functions.elementary.hyperbolic import Cosh, Sinh
from sympy_extras.functions.elementary.piecewise import Piecewise
from sympy_extras.functions.elementary.trigonometric import Cos, Sin
from sympy_extras.functions.special.bessel import (AiryAi, AiryAiPrime, AiryBi, AiryBiPrime, BesselI, BesselJ, BesselK,
                                                    BesselY, Hankel1, Hankel2)
from sympy_extras.functions.special.error_functions import Erf, Erfc, Erfi
from sympy_extras.polys.polytools import Poly, cancel, factor_list
from sympy_extras.polys.rationaltools import together
from sympy_extras.settings import settings
from sympy_extras.simplify.simplify import simplify
from .conditions import ConditionalValue
from .telescoping import _POLYNOMIAL_FAILURES, Element, _Context, _equation, _solve, _verified

__all__ = ['Annihilator', 'DFiniteTelescoper', 'Factor', 'annihilators', 'normal_form', 'closure', 'chyzak',
           'dfinite_ode', 'dfinite_integral']

_ZERO = Integer(0)
_ONE = Integer(1)

#: a mixed derivative ``d^i/dx^i d^j/dt^j F`` by its orders
Derivative_ = tuple[int, int]
#: a combination of the basis derivatives with rational coefficients
NormalForm = dict[Derivative_, Expr]

#: unique names for the auxiliary symbols (the typed polynomials know
#: their variables by name)
_names = count()


def _fresh(prefix: str) -> Dummy:
    return Dummy('_%s%d' % (prefix, next(_names)))


class Annihilator:
    """A linear differential operator ``sum_i c_i(x, t) d^i/dv^i`` in the
    variable ``v`` with coefficients polynomial in ``v``, ``c_r != 0``.

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.integrals.dfinite import Annihilator
    >>> x, t = symbols('x t')
    >>> L_x = Annihilator(x, [2*t*x, 1])
    >>> L_x, L_x.order, L_x.reduction()
    (Annihilator(x, [2*t*x, 1]), 1, [-2*t*x])
    """

    def __init__(self, variable: Symbol, coefficients: Sequence[Expr]) -> None:
        stripped = list(coefficients)
        while stripped and stripped[-1] == _ZERO:
            stripped.pop()
        if len(stripped) < 2:
            raise ValueError("an annihilator has order at least one")
        self.variable: Final[Symbol] = variable
        self.coefficients: Final[list[Expr]] = stripped

    @property
    def order(self) -> int:
        return len(self.coefficients) - 1

    @property
    def leading(self) -> Expr:
        return self.coefficients[-1]

    def reduction(self) -> list[Expr]:
        """The coefficients ``e_i`` with ``d^r F = sum_{i < r} e_i d^i F``."""
        return [cancel(-c/self.leading) for c in self.coefficients[:-1]]

    def apply(self, f: Expr) -> Expr:
        """``sum_i c_i d^i f/dv^i``."""
        return add(*[mul(c, diff(f, (self.variable, i))) for i, c in enumerate(self.coefficients)])

    def __repr__(self) -> str:
        return "Annihilator(%s, %s)" % (self.variable, self.coefficients)


class DFiniteTelescoper:
    """The output of Chyzak's algorithm for ``F(x, t)``: the operator
    ``P = sum_k a_k(t) d^k/dt^k`` and the certificate ``G``, a combination
    of the basis of the closure with rational coefficients, with
    ``P F = dG/dx``.

    Attributes
    ==========

    coefficients : list of Expr
        ``a_0(t), ..., a_J(t)``, polynomials without common factors.
    certificate : dict
        The basis functions of the closure (products of derivatives of
        the D-finite factors) with their rational coefficients in ``G``.
    term, x, t
        The input.

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.functions import exp
    >>> from sympy_extras.integrals.dfinite import chyzak
    >>> x, t = symbols('x t')
    >>> found = chyzak(exp(-x**2)*exp(2*t*x), x, t)
    >>> found.order, found.operator(), found.function(), found.check()
    (1, -2*t*I(t) + Derivative(I(t), t), -exp(-x**2)*exp(2*t*x), True)
    """

    def __init__(self, term: Expr, x: Symbol, t: Symbol, coefficients: Sequence[Expr],
                 certificate: dict[Expr, Expr]) -> None:
        self.term: Final[Expr] = term
        self.x: Final[Symbol] = x
        self.t: Final[Symbol] = t
        self.coefficients: Final[list[Expr]] = list(coefficients)
        self.certificate: Final[dict[Expr, Expr]] = dict(certificate)

    @property
    def order(self) -> int:
        return len(self.coefficients) - 1

    def operator(self, name: str = 'I') -> Expr:
        """``sum_k a_k(t) I^(k)(t)`` in the function ``I(t)``."""
        f = Function(name)(self.t)
        return add(*[mul(a, diff(f, (self.t, k))) for k, a in enumerate(self.coefficients)])

    def function(self) -> Expr:
        """The certificate ``G`` as an expression."""
        return add(*[mul(g, b) for b, g in self.certificate.items()])

    def check(self) -> bool:
        """Verify ``P F = dG/dx`` by simplification."""
        lhs = add(*[mul(a, diff(self.term, (self.t, k))) for k, a in enumerate(self.coefficients)])
        difference = lhs - diff(self.function(), self.x)
        return simplify(difference.doit()) == _ZERO

    def __repr__(self) -> str:
        return "DFiniteTelescoper(%s, %s)" % (self.coefficients, self.certificate)


# ---------------------------------------------------------------------------
# Annihilators

def _minimal_annihilator(F: Expr, v: Symbol, parameter: Symbol, depth: int = 0) -> Optional[Annihilator]:
    """The annihilator of least order of ``F`` in ``v`` within the closure
    of its terms: each term is a rational coefficient times D-finite
    factors (:func:`_closure_of`, the arguments rational in ``v`` and
    ``parameter``), its derivatives are vectors in the basis of its
    factors, and the first linear dependency over the rational functions
    of ``v`` and the parameters among ``F, F', F'', ...`` (the terms in
    the direct sum of their spaces) gives the coefficients, cleared of
    denominators and common factors; ``None`` when a term is not of this
    kind."""
    if F == _ZERO:
        return None
    terms = [a for a in F.args if isinstance(a, Expr)] if isinstance(F, Add) else [F]
    spaces: list[list[Factor]] = []
    current: list[Vector] = []
    keys: list[tuple[str, ...]] = []
    dimension = 0
    for term in terms:
        found = _closure_of(term, v, parameter, True, depth)
        if found is None:
            return None
        coefficient, factors_ = found
        factors = sorted(factors_, key=repr)
        zero: Basis = tuple(0 for _ in factors)
        # the terms with the same factors share their space (the rational
        # terms among them: x**2 + 1 is of order one)
        key = tuple(repr(f) for f in factors)
        if key in keys:
            vector = current[keys.index(key)]
            vector[zero] = add(vector.get(zero, _ZERO), coefficient)
            continue
        keys.append(key)
        spaces.append(factors)
        current.append({zero: coefficient})
        size = 1
        for factor in factors:
            size *= factor.order
        dimension += size
    columns: list[list[Vector]] = [current]
    for _ in range(dimension):
        columns.append([_derive(vector, v, factors) for factors, vector in zip(spaces, columns[-1])])
        relation = _relation(columns, v)
        if relation is not None:
            try:
                return Annihilator(v, relation)
            except ValueError:
                return None
    return None


def _relation(columns: Sequence[Sequence[Vector]], v: Symbol) -> Optional[list[Expr]]:
    """The coefficients of a linear dependency of the derivatives (one
    column per order, each a vector per term), polynomials in ``v``
    without common factors, the leading one nonzero; ``None`` when the
    columns are independent."""
    keys: list[tuple[int, Basis]] = []
    for column in columns:
        for s, vector in enumerate(column):
            for basis in vector:
                if (s, basis) not in keys:
                    keys.append((s, basis))
    entries = [value for column in columns for vector in column for value in vector.values()]
    width = len(columns)
    try:
        K = _Context(entries, _fresh('z'))
        rows = [[K.element(columns[k][s].get(basis, _ZERO)) for k in range(width)] for s, basis in keys]
        null = K.field.nullspace(rows, width)
    except _POLYNOMIAL_FAILURES:
        return None
    if not null:
        return None
    coefficients, _ = _normalized([K.element_expr(c) for c in null[0]], v)
    if coefficients[-1] == _ZERO:
        return None
    return coefficients


def annihilators(F: ExprLike, x: Symbol, t: Symbol) -> Optional[tuple[Annihilator, Annihilator]]:
    """The annihilators ``L_x`` and ``L_t`` of ``F`` of least order in the
    closure of its terms (see the module documentation and
    :func:`closure`; the arguments of the factors rational in ``x`` and
    ``t``, sums, products and algebraic powers of rational functions), or
    ``None`` when a term is not a product of D-finite factors.

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.functions import exp, besselj
    >>> from sympy_extras.integrals.dfinite import annihilators
    >>> x, t = symbols('x t')
    >>> L_x, L_t = annihilators(exp(-x**2)*besselj(0, t*x), x, t)
    >>> L_x.order, L_t.order
    (2, 2)
    >>> annihilators(exp(exp(x))*t, x, t) is None
    True
    """
    F_ = to_expr(F)
    L_x = _minimal_annihilator(F_, x, t)
    if L_x is None:
        return None
    L_t = _minimal_annihilator(F_, t, x)
    if L_t is None:
        return None
    return L_x, L_t


# ---------------------------------------------------------------------------
# Normal forms modulo the two annihilators

def _add(total: NormalForm, key: Derivative_, value: Expr) -> None:
    total[key] = add(total.get(key, _ZERO), value)


def _leibniz(coefficient: Expr, i: int, m: int, base: Derivative_, x: Symbol, t: Symbol,
             reduce: Callable[[int, int], NormalForm]) -> NormalForm:
    """``d^i/dx^i d^m/dt^m (coefficient * d^base F)`` in normal form."""
    result: NormalForm = {}
    for a in range(i + 1):
        for b in range(m + 1):
            binomials = Integer(comb(i, a)*comb(m, b))
            factor = mul(binomials, diff(coefficient, (x, i - a), (t, m - b))) if (i - a or m - b) \
                else mul(binomials, coefficient)
            if factor == _ZERO:
                continue
            inner = reduce(base[0] + a, base[1] + b)
            for key, value in inner.items():
                _add(result, key, mul(factor, value))
    return result


def normal_form(L_x: Annihilator, L_t: Annihilator, x: Symbol, t: Symbol) -> Callable[[int, int], NormalForm]:
    """The reduction ``(i, j) -> d^i/dx^i d^j/dt^j F`` as a combination of
    the basis derivatives ``i < r``, ``j < s``, memoised.

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.integrals.dfinite import Annihilator, normal_form
    >>> x, t = symbols('x t')
    >>> L_x, L_t = Annihilator(x, [2*t*x, 1]), Annihilator(t, [x**2, 1])
    >>> reduce = normal_form(L_x, L_t, x, t)
    >>> reduce(2, 0)
    {(0, 0): 4*t**2*x**2 - 2*t}
    """
    r, s = L_x.order, L_t.order
    e_x = L_x.reduction()
    e_t = L_t.reduction()

    @lru_cache(maxsize=None)
    def reduce(i: int, j: int) -> NormalForm:
        if i < r and j < s:
            return {(i, j): _ONE}
        result: NormalForm = {}
        if j >= s:
            # d^i_x d^j_t F = d^i_x d^(j-s)_t sum_k e_k d^k_t F
            for k, e in enumerate(e_t):
                if e == _ZERO:
                    continue
                for key, value in _leibniz(e, i, j - s, (0, k), x, t, reduce).items():
                    _add(result, key, value)
        else:
            # i >= r: d^i_x d^j_t F = d^(i-r)_x d^j_t sum_k e_k d^k_x F
            for k, e in enumerate(e_x):
                if e == _ZERO:
                    continue
                for key, value in _leibniz(e, i - r, j, (k, 0), x, t, reduce).items():
                    _add(result, key, value)
        cancelled = {key: cancel(value) for key, value in result.items()}
        return {key: value for key, value in cancelled.items() if value != _ZERO}

    return reduce


# ---------------------------------------------------------------------------
# The closure of the derivatives of a product of D-finite factors

class Factor:
    """A D-finite factor ``f(u)`` of the integrand at ``u = argument``, a
    polynomial in ``x`` and ``t``: its order ``n`` and the reduction
    ``f^(n)(u) = sum_{m < n} e_m(u) f^(m)(u)``.

    The basis of the closure of ``F`` under differentiation is the set
    of products of the ``f_i^(m_i)(argument_i)``, ``m_i < n_i``, with
    coefficients rational in ``x`` and ``t``: the derivative of a product
    is computed by the chain rule and the reductions.

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.functions import besselj
    >>> from sympy_extras.integrals.dfinite import closure
    >>> x, t = symbols('x t')
    >>> J = closure(besselj(0, t*x), x, t)[1][0]
    >>> J, J.order, J.derivative(1)
    (Factor(besselj(0, t*x), 2), 2, -besselj(1, t*x))
    """

    def __init__(self, function: Expr, u: Symbol, argument: Expr, reduction: Sequence[Expr]) -> None:
        self.function: Final[Expr] = function
        self.u: Final[Symbol] = u
        self.argument: Final[Expr] = argument
        self.reduction: Final[list[Expr]] = list(reduction)

    @property
    def order(self) -> int:
        return len(self.reduction)

    def derivative(self, m: int) -> Expr:
        """``f^(m)(argument)`` as an expression."""
        return as_expr(diff(self.function, (self.u, m)).subs(self.u, self.argument))

    def __repr__(self) -> str:
        return "Factor(%s, %s)" % (self.function.subs(self.u, self.argument), self.order)


#: a basis element: the orders of the derivatives of the factors
Basis = tuple[int, ...]
#: an element of the closure: basis element -> rational coefficient
Vector = dict[Basis, Expr]


def _table(function: Expr, u: Symbol) -> Optional[list[Expr]]:
    """The coefficients of the annihilator of ``function`` in ``u``
    (lowest order first), written down for the common factors from their
    differential equations (the forms of SymPy's ``expr_to_holonomic``
    for those it has; the second order Bessel and Airy equations, where
    SymPy's Meijer G route gives a third order one for ``besseli`` and
    ``bessely`` and none for the Airy functions); ``None`` for the
    others."""
    if not isinstance(function, Application) or not function.args or function.args[-1] != u:
        return None
    if isinstance(function, Exp):
        return [Integer(-1), _ONE]
    if isinstance(function, (Sin, Cos)):
        return [_ONE, _ZERO, _ONE]
    if isinstance(function, (Sinh, Cosh)):
        return [Integer(-1), _ZERO, _ONE]
    if isinstance(function, (Erf, Erfc)):
        return [_ZERO, 2*u, _ONE]
    if isinstance(function, Erfi):
        return [_ZERO, -2*u, _ONE]
    if isinstance(function, Log) and len(function.args) == 1:
        return [_ZERO, _ONE, u]
    if isinstance(function, (AiryAi, AiryBi)):
        return [-u, _ZERO, _ONE]
    if isinstance(function, (AiryAiPrime, AiryBiPrime)):
        # w = Ai': w' = u Ai, w'' = Ai + u w = w'/u + u w
        return [-u**2, Integer(-1), u]
    if isinstance(function, (BesselJ, BesselY, Hankel1, Hankel2, BesselI, BesselK)):
        nu = function.args[0]
        if isinstance(nu, Expr):
            if isinstance(function, (BesselI, BesselK)):
                return [-u**2 - nu**2, u, u**2]
            return [u**2 - nu**2, u, u**2]
    return None


#: the depth of the derivatives through which the annihilator of a factor
#: outside the table is sought (``Si(u)`` through ``sin(u)/u``)
_MAX_DEPTH: Final[int] = 3


def _derivative_annihilator(function: Expr, u: Symbol, depth: int) -> Optional[list[Expr]]:
    """The annihilator of ``function`` in ``u`` as ``L o d/du`` with ``L``
    the annihilator of its derivative, when the derivative is D-finite in
    the closure (the exponential, sine, cosine and logarithmic integrals,
    the arctangent; ``asin`` through an algebraic power)."""
    if depth >= _MAX_DEPTH:
        return None
    derivative = diff(function, u)
    if derivative.has(Derivative) or function in derivative.atoms(Application):
        return None
    inner = _minimal_annihilator(derivative, u, _fresh('w'), depth + 1)
    if inner is None:
        return None
    return [_ZERO] + inner.coefficients


def _dfinite_factor(f: Expr, x: Symbol, t: Symbol, rational: bool = False, depth: int = 0) -> Optional[Factor]:
    """``f`` as a D-finite function of a polynomial argument (rational in
    ``x`` and ``t`` with ``rational``, which also takes the powers
    ``b**r`` of rational functions ``b`` with a constant exponent)."""
    if rational and isinstance(f, Pow) and not f.exp.has(x, t) and f.base.is_rational_function(x, t) is True:
        w = _fresh('u')
        return Factor(pow(w, f.exp), w, f.base, [f.exp/w])
    if not isinstance(f, Application) or not f.args:
        return None
    arguments = [a for a in f.args if isinstance(a, Expr)]
    if len(arguments) != len(f.args):
        return None
    argument = arguments[-1]
    if any(a.has(x, t) for a in arguments[:-1]):
        return None
    if rational:
        if argument.is_rational_function(x, t) is not True:
            return None
    elif argument.is_polynomial(x, t) is not True:
        return None
    u = _fresh('u')
    function = as_expr(f._evaluated(tuple(arguments[:-1]) + (u,)))
    coefficients = _table(function, u)
    if coefficients is None:
        coefficients = _derivative_annihilator(function, u, depth)
    if coefficients is None or len(coefficients) < 2 or coefficients[-1] == _ZERO:
        return None
    leading = coefficients[-1]
    try:
        reduction = [cancel(-c/leading) for c in coefficients[:-1]]
    except _POLYNOMIAL_FAILURES:
        return None
    return Factor(function, u, argument, reduction)


def closure(F: ExprLike, x: Symbol, t: Symbol) -> Optional[tuple[Expr, list[Factor]]]:
    """The integrand as a rational coefficient times D-finite factors
    ``f_i(argument_i)`` (the argument polynomial in ``x`` and ``t``), or
    ``None`` when a factor is not of this kind.

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.functions import exp, besselj
    >>> from sympy_extras.integrals.dfinite import closure
    >>> x, t = symbols('x t')
    >>> closure(x*exp(-x**2)*besselj(0, t*x), x, t)
    (x, [Factor(besselj(0, t*x), 2), Factor(exp(-x**2), 1)])
    """
    return _closure_of(to_expr(F), x, t, False, 0)


def _closure_of(F: Expr, x: Symbol, t: Symbol, rational: bool, depth: int) -> Optional[tuple[Expr, list[Factor]]]:
    """:func:`closure`, with the factors of :func:`_dfinite_factor` under
    ``rational``."""
    coefficient: Expr = _ONE
    factors: list[Factor] = []
    parts = [a for a in F.args if isinstance(a, Expr)] if isinstance(F, Mul) else [F]
    for part in parts:
        e = part
        if e.is_rational_function(x, t) is True:
            coefficient = mul(coefficient, e)
            continue
        multiplicity = 1
        if isinstance(e, Pow) and isinstance(e.exp, Integer) and e.exp.p > 0:
            multiplicity = e.exp.p
            e = e.base
        factor = _dfinite_factor(e, x, t, rational, depth)
        if factor is None:
            expanded = expand_func(e)
            if expanded != e:
                inner = _closure_of(expanded, x, t, rational, depth)
                if inner is None:
                    return None
                coefficient = mul(coefficient, pow(inner[0], Integer(multiplicity)))
                factors.extend(inner[1]*multiplicity)
                continue
            return None
        factors.extend([factor]*multiplicity)
    return coefficient, factors


def _cancelled(vector: Vector) -> Vector:
    cancelled = {basis: cancel(value) for basis, value in vector.items()}
    return {basis: value for basis, value in cancelled.items() if value != _ZERO}


def _derive(vector: Vector, var: Symbol, factors: Sequence[Factor]) -> Vector:
    """The derivative of an element of the closure with respect to
    ``var`` (``x`` or ``t``), in the basis."""
    result: Vector = {}
    for basis, coefficient in vector.items():
        derived = diff(coefficient, var)
        if derived != _ZERO:
            _add_basis(result, basis, derived)
        for k, factor in enumerate(factors):
            chain = diff(factor.argument, var)
            if chain == _ZERO:
                continue
            m = basis[k] + 1
            if m < factor.order:
                _add_basis(result, basis[:k] + (m,) + basis[k + 1:], mul(coefficient, chain))
            else:
                # f^(n)(argument) reduced by the differential equation of f
                for m_, e in enumerate(factor.reduction):
                    if e == _ZERO:
                        continue
                    _add_basis(result, basis[:k] + (m_,) + basis[k + 1:],
                               mul(coefficient, chain, as_expr(e.subs(factor.u, factor.argument))))
    return _cancelled(result)


def _add_basis(total: Vector, key: Basis, value: Expr) -> None:
    total[key] = add(total.get(key, _ZERO), value)


def _denominators(vectors: Sequence[Vector], x: Symbol) -> Expr:
    """The product of the irreducible factors in ``x`` of the
    denominators of the coefficients."""
    factors: list[Expr] = []
    for vector in vectors:
        for value in vector.values():
            denominator = together(value).as_numer_denom()[1]
            if not denominator.has(x):
                continue
            try:
                _, parts = factor_list(denominator, x)
            except _POLYNOMIAL_FAILURES:
                continue
            for part, _ in parts:
                if part.has(x) and part not in factors:
                    factors.append(part)
    return mul(*factors)


class _Closure:
    """The integrand in its closure: the vector of ``F`` and its
    derivatives in ``t``, and the derivatives in ``x`` of the basis."""

    def __init__(self, F: Expr, x: Symbol, t: Symbol, coefficient: Expr, factors: Sequence[Factor]) -> None:
        self.F: Final[Expr] = F
        self.x: Final[Symbol] = x
        self.t: Final[Symbol] = t
        self.factors: Final[list[Factor]] = list(factors)
        basis: list[Basis] = [()]
        for factor in factors:
            basis = [b + (m,) for b in basis for m in range(factor.order)]
        self.basis: Final[list[Basis]] = basis
        zero: Basis = tuple(0 for _ in factors)
        self.vector: Final[Vector] = {zero: coefficient}
        self.t_derivatives: Final[list[Vector]] = [self.vector]
        self.x_derivatives: Final[dict[Basis, Vector]] = {b: _derive({b: _ONE}, x, self.factors) for b in basis}

    def t_derivative(self, k: int) -> Vector:
        while len(self.t_derivatives) <= k:
            self.t_derivatives.append(_derive(self.t_derivatives[-1], self.t, self.factors))
        return self.t_derivatives[k]

    def element(self, basis: Basis) -> Expr:
        """The basis element as an expression."""
        return mul(*[factor.derivative(m) for factor, m in zip(self.factors, basis)])


def _telescope(closure_: _Closure, order: int, degree: int, power: int) -> Optional[DFiniteTelescoper]:
    """The telescoper of the given order with the certificate of the
    given degree over ``D**power``: the comparison of the coefficients of
    ``sum_k a_k d^k_t F - d_x(sum_b g_b b)`` on the basis, a linear system
    over the field of the parameters (one row per basis element and power
    of ``x`` of the numerators over a common denominator)."""
    x, t = closure_.x, closure_.t
    lhs = [closure_.t_derivative(k) for k in range(order + 1)]
    D = pow(_denominators(lhs + list(closure_.x_derivatives.values()) + [closure_.vector], x), Integer(power))
    width = order + 1 + len(closure_.basis)*(degree + 1)

    def unknown(b: int, m: int) -> int:
        return order + 1 + b*(degree + 1) + m

    # sum_k a_k d^k_t F - d_x (sum_b g_b b), collected on the basis: the
    # coefficient of each unknown at each basis element
    total: dict[Basis, dict[int, Expr]] = {}

    def contribute(key: Basis, column: int, value: Expr) -> None:
        entries = total.setdefault(key, {})
        entries[column] = add(entries.get(column, _ZERO), value)

    for k, vector in enumerate(lhs):
        for key, value in vector.items():
            contribute(key, k, value)
    for i, b in enumerate(closure_.basis):
        for m in range(degree + 1):
            g = x**m/D
            column = unknown(i, m)
            contribute(b, column, -diff(g, x))
            for key, value in closure_.x_derivatives[b].items():
                contribute(key, column, -g*value)
    entries = [v for row in total.values() for v in row.values()]
    try:
        K = _Context(entries, x)
        rows: list[list[Element]] = []
        for row in total.values():
            fractions = {column: K.fraction(value) for column, value in row.items()}
            common = K.constant(K.field.one)
            for _, Q in fractions.values():
                common = common.lcm(Q)
            numerators = {column: P*common.quo(Q) for column, (P, Q) in fractions.items()}
            height = max((P.degree for P in numerators.values() if not P.is_zero), default=-1) + 1
            for power_ in range(height):
                rows.append([numerators[c].coeff(power_) if c in numerators else K.field.zero
                             for c in range(width)])
    except _POLYNOMIAL_FAILURES:
        return None
    for null_vector in K.field.nullspace(rows, width):
        solution = [K.element_expr(v) for v in null_vector]
        if all(v == _ZERO for v in solution[:order + 1]):
            continue
        coefficients, scale = _normalized(solution[:order + 1], t)
        certificate: dict[Expr, Expr] = {}
        for i, b in enumerate(closure_.basis):
            g = add(*[mul(solution[unknown(i, m)], x**m) for m in range(degree + 1)])/D
            g_ = cancel(g*scale)
            if g_ != _ZERO:
                certificate[closure_.element(b)] = g_
        return DFiniteTelescoper(closure_.F, x, t, coefficients, certificate)
    return None


def _normalized(values: Sequence[Expr], t: Symbol) -> tuple[list[Expr], Expr]:
    """The coefficients with denominators cleared and common factors
    removed, and the factor they were multiplied by."""
    combined = together(add(*[mul(v, _fresh('n')) for v in values]))
    _, denominator = combined.as_numer_denom()
    scaled = [cancel(v*denominator) for v in values]
    scale = denominator
    try:
        polys = [Poly(v, t) for v in scaled]
        common = polys[0]
        for p in polys[1:]:
            common = common.gcd(p)
        if common.degree() > 0 or common.LC() != _ONE:
            scaled = [p.quo(common).as_expr() for p in polys]
            scale = denominator/common.as_expr()
    except _POLYNOMIAL_FAILURES:
        pass
    if scaled and scaled[-1].could_extract_minus_sign():
        scaled = [-v for v in scaled]
        scale = -scale
    return scaled, scale


def chyzak(F: ExprLike, x: Symbol, t: Symbol, max_order: int = 3, max_degree: int = 6) -> Optional[DFiniteTelescoper]:
    """A telescoper and certificate for ``F(x, t)`` by Chyzak's algorithm
    with Koutschan's ansatz (see the module documentation), the order of
    the telescoper and the degree of the certificate increased up to the
    bounds; ``None`` when none is found or ``F`` is not a product of
    D-finite factors of polynomial arguments.

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.functions import exp
    >>> from sympy_extras.integrals.dfinite import chyzak
    >>> x, t = symbols('x t')
    >>> chyzak(exp(-x**2)*exp(2*t*x), x, t)
    DFiniteTelescoper([-2*t, 1], {exp(-x**2)*exp(2*t*x): -1})
    >>> chyzak(exp(-t*x**2), x, t).coefficients
    [1, 2*t]
    """
    F_ = to_expr(F)
    found = closure(F_, x, t)
    if found is None:
        return None
    coefficient, factors = found
    closure_ = _Closure(F_, x, t, coefficient, factors)
    for order in range(max_order + 1):
        for power in (1, 2):
            for degree in range(max_degree + 1):
                telescoper = attempt(lambda: _telescope(closure_, order, degree, power), settings.timeout,
                                     native=True)
                if telescoper is not None:
                    return telescoper
    return None


# ---------------------------------------------------------------------------
# The differential equation of the integral, and its solution

def dfinite_ode(F: ExprLike, x: Symbol, a: ExprLike, b: ExprLike, t: Symbol,
                assumptions: Assumptions = None, name: str = 'I') -> Optional[Eq]:
    """The linear differential equation ``sum_k a_k(t) I^(k)(t) = [G]_a^b``
    satisfied by ``I(t) = Integral(F, (x, a, b))``, the boundary values
    of the certificate taken as limits under the assumptions; ``None``
    when no telescoper is found or a limit is not finite.

    Examples
    ========

    >>> from sympy_extras.core import symbols, oo
    >>> from sympy_extras.functions import exp, sin
    >>> from sympy_extras.integrals.dfinite import dfinite_ode
    >>> x = symbols('x')
    >>> t = symbols('t', positive=True)
    >>> dfinite_ode(exp(-t*x)*sin(x)/x, x, 0, oo, t)
    Eq(Derivative(I(t), t), -1/(t**2 + 1))
    """
    F_ = to_expr(F)
    telescoper = chyzak(F_, x, t)
    if telescoper is None:
        return None
    return _equation(telescoper.coefficients, telescoper.function().doit(), x, to_expr(a), to_expr(b), t,
                     assumptions, name)


def dfinite_integral(F: ExprLike, x: Symbol, a: ExprLike, b: ExprLike, t: Symbol,
                     assumptions: Assumptions = None) -> Optional[ConditionalValue]:
    """``Integral(F, (x, a, b))`` as a function of the parameter ``t`` by
    Chyzak's algorithm: the equation of :func:`dfinite_ode` solved by the
    linear ODE solvers (SymPy's ``dsolve`` when they do not solve it), its
    constants fixed by the values of the integral and its derivatives at
    a point (``1``, ``2``, ``1/2``, ``3``) computed by
    :func:`~sympy_extras.integrals.definite.definite_integral`; checked
    numerically when ``settings.numerical_checks`` is on. ``None`` when
    a step fails.

    Examples
    ========

    >>> from sympy_extras.core import symbols, oo
    >>> from sympy_extras.functions import exp, besselj
    >>> from sympy_extras.integrals.dfinite import dfinite_integral
    >>> x = symbols('x')
    >>> t = symbols('t', positive=True)
    >>> dfinite_integral(x*exp(-x**2)*besselj(0, t*x), x, 0, oo, t)
    ConditionalValue(exp(-t**2/4)/2)
    """
    from .marichev import tidy
    F_ = to_expr(F)
    a_, b_ = to_expr(a), to_expr(b)
    telescoper = chyzak(F_, x, t)
    if telescoper is None:
        return None
    equation = _equation(telescoper.coefficients, telescoper.function().doit(), x, a_, b_, t, assumptions, 'I')
    if equation is None:
        return None
    solution = _solve(telescoper.coefficients, equation, F_, x, a_, b_, t, assumptions)
    if solution is None:
        return None
    if solution.free_symbols - F_.free_symbols - {x}:
        return None
    value = tidy(solution, assumptions)
    if value.has(nan, zoo, oo, -oo, Piecewise):
        return None
    if not _verified(value, F_, x, a_, b_, assumptions):
        return None
    return ConditionalValue(value)
