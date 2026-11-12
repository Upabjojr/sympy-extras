"""The classical hints of SymPy's ``dsolve`` on the objects of the core.

:func:`dsolve` solves an ordinary differential equation in one unknown
function by the methods of the textbooks, as SymPy's
:func:`sympy.solvers.ode.dsolve` does with its hints:

* **linear equations** ``a_n y^(n) + ... + a_0 y = b``: a basis of the
  homogeneous equation, from the roots of the characteristic polynomial
  for constant coefficients (``nth_linear_constant_coeff_homogeneous``:
  ``x**k exp(a x) cos(b x)``, ``x**k exp(a x) sin(b x)`` for a pair of
  conjugate roots ``a +- b I``), of the indicial polynomial for Euler's
  equations ``a_i = c_i x**i`` (``nth_linear_euler_eq_homogeneous``:
  ``x**a log(x)**k cos(b log(x))``, ...), by a quadrature for the first
  order (``1st_linear``), by reduction to the lowest derivative present
  (``nth_order_reducible``) and otherwise by the algorithms of
  :mod:`~sympy_extras.solvers.linear_ode` for rational coefficients
  (rational and hyperexponential solutions, Kovacic's algorithm); the
  particular solution of an inhomogeneous equation by variation of
  parameters (:func:`~sympy_extras.solvers.factorization.variation_of_parameters`,
  which gives what SymPy's undetermined coefficients give when the
  quadratures are elementary);
* **first order equations** ``y' = R(x, y)``: ``separable`` ones
  (``R = g(x) h(y)``), ``Bernoulli`` ones (``R = P y + Q y**n``),
  ``1st_exact`` ones (``M + N y' = 0`` with ``M_y = N_x``) and those with
  homogeneous coefficients (``R(x, x u) = R(1, u)``,
  ``1st_homogeneous_coeff_subs_dep_div_indep``), the implicit solutions
  made explicit when the relation is inverted by elementary steps;
* **factorable** equations, whose left hand side is a product: the
  solutions of each factor.

The general solution is written with the constants ``C1, C2, ...``
(those not already in the equation), and is a list of equations
``Eq(f(x), ...)`` or implicit ones, as SymPy's are. The cases of the
isolated values of the parameters are not made here: the callers make
them (:mod:`sympy_extras.solvers._special_values`).

The module works on the objects of the core and imports no SymPy: the
quadratures come from the native indefinite integration
(:func:`~sympy_extras.integrals.indefinite.closed_antiderivative`, an
:class:`~sympy_extras.integrals.integral.Integral` left where none is
found), the roots of the characteristic polynomials from the native
:func:`~sympy_extras.polys.polyroots.roots`.

Ported from SymPy's ``sympy.solvers.ode.single`` (the solvers of the
hints listed above), rewritten on the typed core.

References
==========

.. [Tenenbaum] M. Tenenbaum, H. Pollard, Ordinary Differential Equations,
   Dover 1963, lessons 6-11 (first order), 19-21 (constant coefficients),
   38 (Euler's equation), 40 (variation of parameters).

Examples
========

>>> from sympy_extras.core import Function, Symbol
>>> from sympy_extras.solvers.dsolve import dsolve
>>> x = Symbol('x')
>>> y = Function('y')(x)
>>> dsolve(y.diff(x, 2) + y, y)
[Eq(y(x), C1*sin(x) + C2*cos(x))]
>>> dsolve(x**2*y.diff(x, 2) - 2*y, y)
[Eq(y(x), C1/x + C2*x**2)]
>>> dsolve(y.diff(x) - x*y**2, y)
[Eq(y(x), 1/(-C1 - x**2/2))]
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/solvers/ode/single.py, sympy/solvers/ode/ode.py (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: the solvers of the hints as typed functions on the core, no
# classification table; variation of parameters for the particular solutions.
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Final, Optional, Sequence

from sympy_extras.core.add import Add, add
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr
from sympy_extras.core.function import AppliedUndef, Derivative, diff, expand
from sympy_extras.core.mul import Mul, mul
from sympy_extras.core.numbers import I, Integer, Number
from sympy_extras.core.power import Pow, sqrt
from sympy_extras.core.relational import Eq
from sympy_extras.core.symbol import Dummy, Symbol
from sympy_extras.functions.elementary.exponential import Exp, Log, exp, log
from sympy_extras.functions.elementary.trigonometric import cos, sin
from sympy_extras.polys.polyerrors import BasePolynomialError
from sympy_extras.polys.polytools import cancel, factor
from sympy_extras._timeout import attempt
from sympy_extras.settings import settings

__all__ = ['dsolve', 'explicit_solution', 'solution_parts', 'linear_coefficients', 'linear_basis']

_ZERO: Final[Expr] = Integer(0)
_ONE: Final[Expr] = Integer(1)

#: what the polynomial routines raise on coefficients they cannot take
_FAILURES = (BasePolynomialError, NotImplementedError, ValueError, TypeError, ZeroDivisionError)


# ---------------------------------------------------------------------------
# helpers

def _sum(terms: Sequence[Expr]) -> Expr:
    return add(*terms) if terms else _ZERO


def _replaced(e: Expr, rule: dict[Basic, Basic]) -> Expr:
    value = e._xreplace_evaluated(rule)
    if not isinstance(value, Expr):
        raise TypeError("%s is not an expression" % (value,))
    return value


def _normal(e: Expr) -> Expr:
    try:
        return cancel(e)
    except _FAILURES:
        return e


def _as_zero(equation: Basic) -> Expr:
    if isinstance(equation, Eq):
        lhs, rhs = equation.lhs, equation.rhs
        if isinstance(lhs, Expr) and isinstance(rhs, Expr):
            return lhs - rhs
        raise TypeError("%s is not an equation of expressions" % (equation,))
    if not isinstance(equation, Expr):
        raise TypeError("%s is not an equation" % (equation,))
    return equation


def _variable(f: AppliedUndef) -> Symbol:
    if len(f.args) != 1 or not isinstance(f.args[0], Symbol):
        raise ValueError("%s is not a function of one symbol" % (f,))
    return f.args[0]


def _order(e: Expr, f: AppliedUndef) -> int:
    """The order of the highest derivative of ``f`` in ``e`` (``0`` for none)."""
    x = _variable(f)
    n = 0
    for d in e.atoms(Derivative):
        if isinstance(d, Derivative) and d.expr == f:
            n = max(n, sum(int(count) for v, count in d.variable_count if v == x))
    return n


def _constants(taken: set[Symbol], n: int) -> list[Symbol]:
    """``n`` constants ``C1, C2, ...`` not in ``taken`` (SymPy's
    ``iter_numbered_constants``)."""
    names = {s.name for s in taken}
    result: list[Symbol] = []
    k = 1
    while len(result) < n:
        name = 'C%d' % k
        if name not in names:
            result.append(Symbol(name))
        k += 1
    return result


def _symbols(e: Basic) -> set[Symbol]:
    return {s for s in e.free_symbols if isinstance(s, Symbol)}


def _integral(e: Expr, x: Symbol) -> Expr:
    """An antiderivative by the native indefinite integration, the
    unevaluated ``Integral`` when none is found."""
    from sympy_extras.integrals.indefinite import closed_antiderivative
    from sympy_extras.integrals.integral import Integral
    found = closed_antiderivative(e, x)
    return Integral(e, x) if found is None else found


def _closed_integral(e: Expr, x: Symbol) -> Optional[Expr]:
    from sympy_extras.integrals.indefinite import closed_antiderivative
    return closed_antiderivative(e, x)


# ---------------------------------------------------------------------------
# linear equations

def linear_coefficients(equation: Basic, f: AppliedUndef) -> Optional[tuple[list[Expr], Expr]]:
    """``([a_0, ..., a_n], b)`` with ``a_n y^(n) + ... + a_0 y = b`` the
    equation, when it is linear in ``f`` and its derivatives (``None``
    otherwise; ``a_n`` nonzero).

    >>> from sympy_extras.core import Function, Symbol
    >>> from sympy_extras.solvers.dsolve import linear_coefficients
    >>> x = Symbol('x')
    >>> y = Function('y')(x)
    >>> linear_coefficients(x*y.diff(x, 2) + y - x**2, y)
    ([1, 0, x], x**2)
    """
    e = _as_zero(equation)
    x = _variable(f)
    n = _order(e, f)
    expanded = expand(e)
    coefficients: list[Expr] = []
    rest = expanded
    for i in range(n, -1, -1):
        term: Expr = f.diff(x, i) if i > 0 else f
        c = rest.coeff(term)
        if c != _ZERO:
            rest = expand(rest - c*term)
        coefficients.append(c)
    if rest.has(f) or any(c.has(f) for c in coefficients):
        return None
    coefficients.reverse()
    if coefficients[-1] == _ZERO:
        return None
    return coefficients, -rest


def _cleared(p: list[Expr]) -> list[Expr]:
    """The coefficients times the product of their distinct denominators,
    polynomial in the parameters (the native ``Poly`` takes no fractions
    of the parameters as coefficients)."""
    denominators: list[Expr] = []
    for c in p:
        d = _normal(c).as_numer_denom()[1]
        if d != _ONE and not any(expand(d - e) == _ZERO for e in denominators):
            denominators.append(d)
    if not denominators:
        return p
    common = mul(*denominators)
    return [_normal(c*common) for c in p]


def _cleared_polynomial(e: Expr, m: Symbol) -> Expr:
    """``e``, a polynomial in ``m``, with the denominators of its
    coefficients cleared."""
    from .kovacic import _coefficients
    terms = _coefficients(e, m)
    degree = max(terms) if terms else 0
    cleared = _cleared([terms.get(i, _ZERO) for i in range(degree + 1)])
    return expand(_sum([mul(c, m**i) for i, c in enumerate(cleared)]))


def _characteristic_roots(p: list[Expr]) -> Optional[list[tuple[Expr, int]]]:
    """The roots of ``sum p_i m**i`` with their multiplicities, ``None``
    when they are not all found in radicals."""
    from sympy_extras.polys.polyroots import roots
    m = Dummy('m')
    polynomial = _sum([mul(c, m**i) for i, c in enumerate(p)])
    try:
        found = roots(polynomial, m)
    except _FAILURES:
        return None
    if not isinstance(found, dict):
        return None
    pairs: list[tuple[Expr, int]] = []
    for r, k in found.items():
        if isinstance(r, Expr):
            pairs.append((r, k))
    if sum(k for _, k in pairs) != len(p) - 1:
        return None
    from sympy_extras.core.sorting import default_sort_key
    return sorted(pairs, key=lambda pair: default_sort_key(pair[0]))


def _real_imaginary(r: Expr) -> Optional[tuple[Expr, Expr]]:
    """``(a, b)`` with ``r = a + b I``, ``b`` a nonzero number times
    real-looking factors, when ``r`` has an explicit imaginary part."""
    if not r.has(I):
        return None
    a = _replaced(r, {I: _ZERO})
    b = expand(r - a).coeff(I)
    if b == _ZERO or b.has(I) or expand(a + b*I - r) != _ZERO:
        return None
    return expand(a), b


def _root_basis(pairs: list[tuple[Expr, int]], x: Symbol, t: Expr) -> list[Expr]:
    """The solutions ``x**k exp(r t)`` (``t = x``) or ``x**r log(x)**k``
    (``t = log(x)``) for the roots ``r``; a conjugate pair ``a +- b I`` as
    the real ``exp(a t) cos(b t)``, ``exp(a t) sin(b t)``."""
    euler = t != x
    basis: list[Expr] = []
    done: list[Expr] = []

    def power(k: int) -> Expr:
        return t**k if k else _ONE

    def growth(a: Expr) -> Expr:
        return x**a if euler else exp(a*x)

    for r, k in pairs:
        if any(expand(r - d) == _ZERO for d in done):
            continue
        parts = _real_imaginary(r)
        if parts is not None:
            a, b = parts
            conjugate = a - b*I
            partner = [q for q, j in pairs if j == k and expand(q - conjugate) == _ZERO]
            if partner:
                done.append(conjugate)
                if isinstance(b, Number) and b.is_negative is True:
                    b = -b
                for j in range(k):
                    basis.append(mul(power(j), growth(a), sin(b*t)))
                    basis.append(mul(power(j), growth(a), cos(b*t)))
                continue
        done.append(r)
        for j in range(k):
            basis.append(mul(power(j), growth(r)))
    return basis


def linear_basis(coefficients: Sequence[Expr], x: Symbol, general: bool = True) -> Optional[list[Expr]]:
    """A basis of the solutions of ``sum a_i y^(i) = 0`` (the coefficients
    from ``a_0``), ``None`` when not all of them are found: constant
    coefficients, Euler's equations, the first order, the reduction to
    the lowest derivative and (with ``general``) the algorithms for
    rational coefficients of :mod:`~sympy_extras.solvers.linear_ode`.

    >>> from sympy_extras.core import Integer, Symbol
    >>> from sympy_extras.solvers.dsolve import linear_basis
    >>> x = Symbol('x')
    >>> linear_basis([Integer(1), Integer(2), Integer(1)], x)
    [exp(-x), x*exp(-x)]
    """
    n = len(coefficients) - 1
    leading = coefficients[-1]
    p = [_normal(c/leading) for c in coefficients]
    if n == 0:
        return []
    if all(not c.has(x) for c in p):
        pairs = _characteristic_roots(_cleared(p))
        return None if pairs is None else _root_basis(pairs, x, x)
    scaled = [_normal(c*x**(n - i)) for i, c in enumerate(p)]
    if all(not c.has(x) for c in scaled):
        # Euler: x**n y^(n) + ... with y = x**m: sum c_i m (m - 1) ... (m - i + 1)
        m = Dummy('m')
        indicial = _ZERO
        for i, c in enumerate(scaled):
            falling: Expr = _ONE
            for j in range(i):
                falling = falling*(m - j)
            indicial = indicial + c*falling
        from sympy_extras.polys.polyroots import roots
        try:
            found = roots(_cleared_polynomial(indicial, m), m)
        except _FAILURES:
            found = {}
        if isinstance(found, dict):
            pairs = [(r, k) for r, k in found.items() if isinstance(r, Expr)]
            if sum(k for _, k in pairs) == n:
                from sympy_extras.core.sorting import default_sort_key
                pairs.sort(key=lambda pair: default_sort_key(pair[0]))
                return _root_basis(pairs, x, log(x))
    if n == 1:
        exponent = _closed_integral(-p[0], x)
        if exponent is None:
            from sympy_extras.integrals.integral import Integral
            return [exp(Integral(-p[0], x))]
        return [_simplified_exp(exponent)]
    k = 0
    while p[k] == _ZERO:
        k += 1
    if k > 0:
        reduced = linear_basis(p[k:], x, general)
        if reduced is None:
            return None
        basis: list[Expr] = [x**j if j else _ONE for j in range(k)]
        for v in reduced:
            w = v
            for _ in range(k):
                integral = _closed_integral(w, x)
                if integral is None:
                    return None
                w = integral
            basis.append(w)
        return basis
    if not general:
        return None
    from sympy_extras.core.function import Function
    from .linear_ode import dsolve_linear
    y = Function('_y')(x)
    equation = _sum([mul(c, y.diff(x, i)) if i else mul(c, y) for i, c in enumerate(p)])
    try:
        found_basis = attempt(lambda: dsolve_linear(equation, y, use_dsolve=False, special_values=False),
                              settings.timeout, native=True)
    except _FAILURES:
        return None
    if found_basis is None or len(found_basis) != n:
        return None
    return found_basis


def _simplified_exp(exponent: Expr) -> Expr:
    """``exp(exponent)`` with the logarithms of the exponent taken out
    (``exp(2 log(x)) = x**2``)."""
    from sympy_extras.simplify.powsimp import powsimp
    value = exp(exponent)
    try:
        return powsimp(value)
    except _FAILURES:
        return value


def _linear(coefficients: list[Expr], b: Expr, f: AppliedUndef, taken: set[Symbol]) -> Optional[list[Eq]]:
    """The general solution of a linear equation, ``None`` when no basis
    or particular solution is found."""
    x = _variable(f)
    n = len(coefficients) - 1
    found_basis = linear_basis(coefficients, x)
    if found_basis is None or len(found_basis) != n:
        return None
    # a basis is defined up to constant factors: the numbers are left out
    # of the solutions, as SymPy's are (C2*erf(z/2), not sqrt(pi)*C2*erf(z/2))
    basis = [_without_number(b, x) for b in found_basis]
    particular: Expr = _ZERO
    if b != _ZERO:
        from .factorization import variation_of_parameters
        from .linear_ode import LinearOperator
        L = LinearOperator(list(coefficients), x)
        try:
            found = attempt(lambda: variation_of_parameters(L, basis, b), settings.timeout, native=True)
        except _FAILURES:
            return None
        if found is None:
            return None
        particular = _tidy(found, x)
    constants = _constants(taken, n)
    general = add(particular, *[mul(c, s) for c, s in zip(constants, basis)])
    if n == 1 and particular != _ZERO:
        # SymPy's form of the first order (C1 + Integral(Q/y1)) y1 when it
        # is shorter: (C1 + x) exp(x) rather than C1 exp(x) + x exp(x)
        quotient = _tidy(particular/basis[0], x)
        factored = mul(add(constants[0], quotient), basis[0])
        if _ops(factored) < _ops(general):
            general = factored
    return [Eq(f, general)]


def _ops(e: Expr) -> int:
    from sympy_extras.core.function import count_ops
    found = count_ops(e)
    return int(found) if isinstance(found, Integer) else 0


def _without_number(b: Expr, x: Symbol) -> Expr:
    """``b`` without a constant factor which is a nonzero number."""
    if not isinstance(b, Mul):
        return b
    numbers = [f for f in b.args if isinstance(f, Expr) and not f.free_symbols]
    if not numbers:
        return b
    rest = [f for f in b.args if isinstance(f, Expr) and f.free_symbols]
    return mul(*rest) if rest else b


def _tidy(e: Expr, x: Symbol) -> Expr:
    """A particular solution in a short form: expanded and cancelled term
    by term."""
    from sympy_extras.simplify.simplify import simplify
    found = attempt(lambda: simplify(e), settings.timeout, native=True)
    return e if found is None else found


# ---------------------------------------------------------------------------
# first order equations

def _invert(g: Expr, y: Symbol, value: Expr) -> list[Expr]:
    """The ``y`` with ``g(y) = value`` by elementary inversions (all the
    branches of a square, the principal one of other powers), ``[]`` when
    ``g`` is not inverted so."""
    if g == y:
        return [value]
    if not g.has(y):
        return []
    if isinstance(g, Add):
        independent, dependent = g.as_independent(y, as_Add=True)
        if independent != _ZERO:
            return _invert(dependent, y, value - independent)
    elif isinstance(g, Mul):
        independent, dependent = g.as_independent(y, as_Add=False)
        if independent != _ONE:
            return _invert(dependent, y, value/independent)
    elif isinstance(g, Pow):
        base, exponent = g.base, g.exp
        if not exponent.has(y):
            if exponent == Integer(2):
                return _invert(base, y, sqrt(value)) + _invert(base, y, -sqrt(value))
            return _invert(base, y, value**(1/exponent))
        if not base.has(y):
            return _invert(exponent, y, log(value)/log(base))
    elif isinstance(g, Exp):
        argument = g.args[0]
        if isinstance(argument, Expr) and argument.could_extract_minus_sign():
            # SymPy's form: exp(-y) = v gives y = log(1/v)
            return _invert(-argument, y, log(_normal(1/value)))
        return _invert(argument, y, log(value))
    elif isinstance(g, Log):
        return _invert(g.args[0], y, exp(value))
    # a polynomial of degree two in y
    from .kovacic import _coefficients
    try:
        terms = _coefficients(g - value, y)
    except ValueError:
        return []
    if max(terms) == 2 and min(terms) >= 0:
        a, b_, c = terms.get(2, _ZERO), terms.get(1, _ZERO), terms.get(0, _ZERO)
        d = sqrt(b_**2 - 4*a*c)
        return [_normal((-b_ + d)/(2*a)), _normal((-b_ - d)/(2*a))]
    return []


def _implicit_or_explicit(lhs: Expr, rhs: Expr, u: Symbol, f: AppliedUndef) -> list[Eq]:
    """``lhs(u) = rhs`` solved for ``u = f`` when inverted, else as the
    implicit relation in ``f``."""
    values = _invert(lhs, u, rhs)
    if values:
        return [Eq(f, _replaced(v, {u: f})) for v in values]
    return [Eq(_replaced(lhs, {u: f}), _replaced(rhs, {u: f}))]


def _separated(R: Expr, x: Symbol, u: Symbol) -> Optional[tuple[Expr, Expr]]:
    """``(g(x), h(u))`` with ``R = g h``, ``None`` when ``R`` does not
    separate."""
    if not R.has(u):
        return R, _ONE
    # the constant factors go with g, as in SymPy: C1 exp(y) is g = C1, h = exp(y)
    candidates = [R]
    try:
        factored = factor(R)
        candidates.append(factored)
    except _FAILURES:
        pass
    for candidate in candidates:
        g_parts: list[Expr] = []
        h_parts: list[Expr] = []
        for factor_ in Mul.make_args(candidate):
            if not factor_.has(u):
                g_parts.append(factor_)
            elif not factor_.has(x):
                h_parts.append(factor_)
            else:
                break
        else:
            return mul(*g_parts) if g_parts else _ONE, mul(*h_parts) if h_parts else _ONE
    return None


def _separable(R: Expr, x: Symbol, u: Symbol, f: AppliedUndef, C: Symbol) -> Optional[list[Eq]]:
    parts = _separated(R, x, u)
    if parts is None:
        return None
    g, h = parts
    left = _closed_integral(_normal(1/h), u)
    right = _closed_integral(g, x)
    if left is None or right is None:
        return None
    return _implicit_or_explicit(left, right + C, u, f)


def _bernoulli(R: Expr, x: Symbol, u: Symbol, f: AppliedUndef, C: Symbol) -> Optional[list[Eq]]:
    """``y' = P y + Q y**n``: ``v = y**(1 - n)`` solves ``v' = (1 - n)(P v + Q)``."""
    P: Expr = _ZERO
    Q: Expr = _ZERO
    n: Optional[Expr] = None
    for term in Add.make_args(expand(R)):
        coefficient, rest = term.as_independent(u, as_Add=False)
        if rest == u:
            P = P + coefficient
            continue
        if rest == _ONE:
            return None
        if isinstance(rest, Pow) and rest.base == u and not rest.exp.has(x, u):
            if n is None or expand(n - rest.exp) == _ZERO:
                n = rest.exp
                Q = Q + coefficient
                continue
        return None
    if n is None or Q == _ZERO or n == _ONE or n == _ZERO:
        return None
    k = 1 - n
    integrating = _closed_integral(-k*P, x)
    if integrating is None:
        return None
    mu = _simplified_exp(integrating)
    quadrature = _closed_integral(_normal(k*Q*mu), x)
    if quadrature is None:
        return None
    v = (quadrature + C)/mu
    return [Eq(f, v**(1/k))]


def _exact(M: Expr, N: Expr, x: Symbol, u: Symbol, f: AppliedUndef, C: Symbol) -> Optional[list[Eq]]:
    """``M + N y' = 0`` with ``M_y = N_x``: the potential ``F`` with
    ``F_x = M``, ``F_y = N`` is constant."""
    if _normal(diff(M, u) - diff(N, x)) != _ZERO:
        return None
    first = _closed_integral(M, x)
    if first is None:
        return None
    rest = _normal(N - diff(first, u))
    if rest.has(x):
        return None
    second = _closed_integral(rest, u) if rest != _ZERO else _ZERO
    if second is None:
        return None
    potential = first + second
    return _implicit_or_explicit(potential, C, u, f)


def _homogeneous(R: Expr, x: Symbol, u: Symbol, f: AppliedUndef, C: Symbol) -> Optional[list[Eq]]:
    """``R(x, x w) = R(1, w)``: ``y = x w`` gives ``x w' = R(1, w) - w``,
    separable."""
    w = Dummy('w')
    scaled = _normal(_replaced(R, {u: x*w}))
    if scaled.has(x):
        return None
    denominator = _normal(scaled - w)
    if denominator == _ZERO:
        return None
    left = _closed_integral(_normal(1/denominator), w)
    if left is None:
        return None
    relation = _replaced(left, {w: u/x}) - log(x)
    return _implicit_or_explicit(relation, C, u, f)


def _first_order(e: Expr, f: AppliedUndef, taken: set[Symbol]) -> Optional[list[Eq]]:
    x = _variable(f)
    u, p = Dummy('y'), Dummy('p')
    replaced = _replaced(_replaced(e, {Derivative(f, x): p}), {f: u})
    if replaced.has(f) or replaced.atoms(Derivative):
        return None
    from .kovacic import _coefficients
    try:
        terms = _coefficients(replaced, p)
    except ValueError:
        return None
    if set(terms) - {0, 1} or 1 not in terms:
        return None
    N, M = terms[1], terms.get(0, _ZERO)
    [C] = _constants(taken, 1)
    R = _normal(-M/N)
    for method in (_separable, _bernoulli):
        try:
            found = method(R, x, u, f, C)
        except _FAILURES:
            found = None
        if found is not None:
            return found
    try:
        exact = _exact(M, N, x, u, f, C)
    except _FAILURES:
        exact = None
    if exact is not None:
        return exact
    try:
        return _homogeneous(R, x, u, f, C)
    except _FAILURES:
        return None


# ---------------------------------------------------------------------------
# the dispatcher

def _factors(e: Expr, f: AppliedUndef) -> list[Expr]:
    """The factors of ``e`` with ``f`` in them, when ``e`` is a product of
    several (SymPy's hint ``factorable``)."""
    x = _variable(f)
    n = _order(e, f)
    jets = [Dummy('y%d' % i) for i in range(n + 1)]
    rule: dict[Basic, Basic] = {}
    for i in range(n, 0, -1):
        rule[Derivative(f, (x, i))] = jets[i]
    rule[f] = jets[0]
    replaced = _replaced(e, rule)
    try:
        factored = factor(replaced)
    except _FAILURES:
        return []
    if not isinstance(factored, Mul):
        return []
    back: dict[Basic, Basic] = {jets[i]: (f.diff(x, i) if i else f) for i in range(n + 1)}
    found = [_replaced(g, back) for g in factored.args if g.has(*jets)]
    return found if len(found) > 1 else []


def explicit_solution(equation: Basic, f: AppliedUndef) -> Optional[tuple[Expr, list[Symbol]]]:
    """The right hand side of the explicit general solution of
    :func:`dsolve` and its constants ``C1, C2, ...`` in order, ``None``
    when it does not give one explicit solution.

    >>> from sympy_extras.core import Function, Symbol
    >>> from sympy_extras.solvers.dsolve import explicit_solution
    >>> t = Symbol('t')
    >>> I = Function('I')(t)
    >>> explicit_solution(I.diff(t) + I/t - 1, I)
    (C1/t + t/2, [C1])
    """
    found = dsolve(equation, f)
    if len(found) != 1 or found[0].lhs != f:
        return None
    rhs = found[0].rhs
    if not isinstance(rhs, Expr):
        return None
    taken = {s.name for s in _symbols(_as_zero(equation))}
    constants = sorted((s for s in _symbols(rhs) if s.name not in taken and s.name.startswith('C')
                        and s.name[1:].isdigit()), key=lambda s: int(s.name[1:]))
    return rhs, constants


def solution_parts(equation: Basic, f: AppliedUndef) -> Optional[list[Expr]]:
    """Independent solutions of a homogeneous linear equation by the
    methods of :func:`linear_basis` which need no algorithm for rational
    coefficients (constant coefficients, Euler's equations, the first
    order and the reduction to the lowest derivative), each free of
    unevaluated integrals; ``None`` when there are none.

    >>> from sympy_extras.core import Function, Symbol
    >>> from sympy_extras.solvers.dsolve import solution_parts
    >>> x = Symbol('x')
    >>> y = Function('y')(x)
    >>> solution_parts(y.diff(x, 2) - y, y)
    [exp(-x), exp(x)]
    """
    from .linear_ode import has_integral
    linear = linear_coefficients(equation, f)
    if linear is None or linear[1] != _ZERO:
        return None
    basis = linear_basis(linear[0], _variable(f), general=False)
    if basis is None:
        return None
    parts = [b for b in basis if not has_integral(b)]
    return parts or None


def dsolve(equation: Basic, f: AppliedUndef) -> list[Eq]:
    """The general solution of an ordinary differential equation in
    ``f`` by the classical methods (see the module docstring), as a list
    of equations (explicit ``Eq(f, ...)`` or implicit); ``[]`` when none
    of the methods applies.

    Parameters
    ==========

    equation : Eq or Expr
        The equation, an expression being equated to zero.
    f : AppliedUndef
        The unknown function ``f(x)``.

    Examples
    ========

    >>> from sympy_extras.core import Function, Symbol
    >>> from sympy_extras.functions import exp
    >>> from sympy_extras.solvers.dsolve import dsolve
    >>> x = Symbol('x')
    >>> y = Function('y')(x)
    >>> dsolve(y.diff(x, 2) - 2*y.diff(x) + y - exp(2*x), y)
    [Eq(y(x), C1*exp(x) + C2*x*exp(x) + exp(2*x))]
    >>> dsolve(y.diff(x) - y/x, y)
    [Eq(y(x), C1*x)]
    """
    e = _as_zero(equation)
    x = _variable(f)
    n = _order(e, f)
    if n == 0:
        return []
    taken = _symbols(e)
    linear = linear_coefficients(e, f)
    if linear is not None:
        coefficients, b = linear
        found = _linear(coefficients, b, f, taken)
        return [] if found is None else found
    factors = _factors(e, f)
    if factors:
        solutions: list[Eq] = []
        for g in factors:
            if _order(g, f) == 0:
                continue
            for s in dsolve(g, f):
                if s not in solutions:
                    solutions.append(s)
        if solutions:
            return solutions
    if n == 1:
        found = _first_order(e, f, taken)
        return [] if found is None else found
    del x
    return []
