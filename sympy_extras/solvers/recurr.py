r"""Linear recurrences with polynomial coefficients: polynomial,
hypergeometric and d'Alembertian solutions.

A linear recurrence of order `d`,

.. math:: p_d(n)\, y(n + d) + \cdots + p_1(n)\, y(n + 1) + p_0(n)\, y(n) = f(n),

with polynomial coefficients `p_i` is written here as the list of its
coefficients ``[p_0, ..., p_d]``. SymPy's ``rsolve`` finds the polynomial,
rational and hypergeometric solutions (Petkovšek's ``Hyper``) but no other:
for `y(n + 1) = (n + 1)\, y(n) + 1` it returns nothing, and for
`(n + 2)\, y(n + 2) = (2n + 3)\, y(n + 1) - (n + 1)\, y(n)` it returns the
constants and misses the harmonic numbers. This module adds the
*d'Alembertian* solutions of Abramov and Petkovšek [1]_: the nested
indefinite sums of hypergeometric terms,

.. math:: y(n) = h_1(n) \sum_{k_1 < n} h_2(k_1) \sum_{k_2 < k_1} h_3(k_2) \cdots,

which are exactly the solutions of the operators that factor into first
order factors over the rational functions. They include `n!`, the
harmonic numbers, the derangement numbers `n! \sum (-1)^k / k!` and the
solutions of every recurrence with constant coefficients.

The algorithms
==============

* :func:`rsolve_polynomial`: the degree of a polynomial solution is
  bounded by the nonnegative integer roots of the first nonzero of the
  polynomials `b_k(e)` in the expansion
  `L[n^e] = \sum_k b_k(e)\, n^{D + e - k}` (`D` the largest degree of the
  coefficients), and for a right-hand side `f` also by `\deg f - D + k`;
  the coefficients of the solution are then found by linear algebra.
* :func:`rsolve_hypergeometric`, Petkovšek's algorithm [2]_: a
  hypergeometric solution has `y(n + 1)/y(n) = z\, a(n)\, c(n + 1) /
  (b(n)\, c(n))` with `a, b, c` monic, `\gcd(a(n), b(n + h)) = 1` for every
  nonnegative integer `h` (the Gosper–Petkovšek form, :func:`gosper_form`),
  and then `a` divides `p_0(n)`, `b` divides `p_d(n - d + 1)`, `z` is a
  root of the polynomial made of the leading coefficients of
  `P_i = p_i(n) \prod_{j < i} a(n + j) \prod_{i \le j < d} b(n + j)`, and
  `c` is a polynomial solution of `\sum_i z^i P_i(n)\, c(n + i) = 0`. The
  terms found are grouped into similarity classes (two hypergeometric
  terms are similar when their quotient is a rational function, decided
  by the Gosper–Petkovšek form of the quotient of their ratios) so that
  the result is a basis.
* :func:`rsolve_dalembertian`, [1]_: a hypergeometric solution `h` with
  ratio `r` gives `L = \tilde L \circ (N - r)`, and with `y = h \sum z`
  the equation becomes `\tilde L' z = 0` of order `d - 1`, whose
  d'Alembertian solutions are found recursively; the solutions of `L` are
  `h` and the `h \sum_{k < n} z(k)`. Every d'Alembertian solution is
  found, whichever `h` is chosen.
* :func:`rsolve_linear`: the equation written with `y(n + i)`, the
  d'Alembertian solutions as a general solution with constants
  `C_0, C_1, \ldots`, a particular solution when the right-hand side is a
  sum of hypergeometric terms (each term `f_j` has an annihilator
  `N - f_j(n + 1)/f_j(n)`, the operator `(N - \rho_j) \circ L` has a
  d'Alembertian solution `u` with `L u = \kappa f_j`, `\kappa` a constant
  found by evaluating at one point), and initial values.

The solutions are exact expressions in ``RisingFactorial``, ``factorial``,
``Product`` (for the irreducible factors of degree two and more),
``Sum`` and powers `z^n`; they satisfy the recurrence for every `n` at
which they are defined, which is every `n \ge 0` unless a factor of the
products vanishes at a nonnegative integer, in which case the products
and sums start after it. Every solution is checked by substitution at
several integers before it is returned.

References
==========

.. [1] S. A. Abramov, M. Petkovšek, D'Alembertian solutions of linear
   differential and difference equations, ISSAC '94, 169–174.
.. [2] M. Petkovšek, Hypergeometric solutions of linear recurrences with
   polynomial coefficients, Journal of Symbolic Computation 14 (1992)
   243–264.
.. [3] M. Petkovšek, H. S. Wilf, D. Zeilberger, A = B, A K Peters (1996),
   chapters 5 and 8.
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

from sympy import (Add, Dummy, Eq, Expr, Integer, Matrix, Mul, Piecewise, Poly, Product, Rational, RisingFactorial, S,
    Sum, Symbol, cancel, factorial, fraction, gamma, gammasimp, hyper, lcm, lerchphi, lowergamma, numer, polylog, roots,
    simplify, sympify, together, uppergamma, zeta)
from sympy.core.function import AppliedUndef
from sympy.polys.matrices import DomainMatrix
from sympy.polys.polyerrors import CoercionFailed, GeneratorsError, PolynomialError

from sympy_extras._timeout import attempt
from sympy_extras._typing import as_expr, as_symbol, free_symbols

__all__ = ['rsolve_polynomial', 'rsolve_hypergeometric', 'rsolve_dalembertian', 'rsolve_linear',
           'gosper_form', 'is_recurrence_solution', 'RecurrenceSolution']

#: the names of the summation indices, by depth of the nested sums
_INDEX_NAMES = ('k', 'j', 'i', 'l', 'm')
#: the seconds given to SymPy for the closed form of a sum in a solution
CLOSED_FORM_SECONDS = 2.0
#: a solution and its ratio, with the first index from which it is defined
RecurrenceSolution = tuple[Expr, Expr, int]


# ---------------------------------------------------------------------------
# polynomials in n

def _poly(expr: Expr, n: Symbol) -> Poly:
    """``expr`` as a polynomial in ``n``, over an algebraic field when its
    coefficients are algebraic numbers."""
    p = Poly(expr, n)
    if p.domain.is_EX:
        try:
            p = Poly(expr, n, extension=True)
        except (CoercionFailed, GeneratorsError, NotImplementedError, PolynomialError):
            pass
    return p


def _coefficient_polys(coefficients: Sequence[Union[Expr, int]], n: Symbol) -> list[Poly]:
    """The coefficients as polynomials in ``n``, the zero leading ones
    dropped; rational functions of ``n`` are brought to a common
    denominator first."""
    exprs = [as_expr(sympify(c)) for c in coefficients]
    if not all(e.is_rational_function(n) for e in exprs):
        raise ValueError("the coefficients must be rational functions of %s: %s" % (n, exprs))
    common: Expr = S.One
    for e in exprs:
        common = as_expr(lcm(common, as_expr(fraction(cancel(e))[1])))
    polys = [_poly(as_expr(cancel(e*common)), n) for e in exprs]
    while polys and polys[-1].is_zero:
        polys.pop()
    return polys


def _normalized(polys: Sequence[Poly], n: Symbol) -> list[Poly]:
    """The recurrence with its trailing coefficient nonzero: when
    ``p_0 = ... = p_{s-1} = 0`` the equation is one in ``y(n + s), ...``,
    the same recurrence at the index ``n + s``."""
    polys = list(polys)
    s = 0
    while polys and polys[0].is_zero:
        polys.pop(0)
        s += 1
    return [p.shift(-s) for p in polys] if s else polys


def _nonnegative_integer_roots(expr: Expr, x: Symbol) -> list[int]:
    """The nonnegative integer roots of a polynomial in ``x``, whose
    coefficients may be algebraic numbers or parameters."""
    found: set[int] = set()
    try:
        p = _poly(expr, x)
        if p.is_zero:
            return []
        ground = p.ground_roots()
    except (CoercionFailed, GeneratorsError, NotImplementedError, PolynomialError):
        ground = roots(expr, x, filter='Z')
    for root in ground:
        value = as_expr(sympify(root))
        if value.is_Integer and value >= 0:
            found.add(int(value))
    return sorted(found)


def _linear_system(rows: list[list[Expr]], rhs: Optional[list[Expr]], columns: int
                   ) -> tuple[Optional[list[Expr]], list[list[Expr]]]:
    """A particular solution of ``rows * x = rhs`` (``None`` when there is
    none, or when ``rhs`` is not given) and a basis of the solutions of
    the homogeneous system in ``columns`` unknowns."""
    if columns == 0:
        return ([] if rhs is None or all(v == 0 for v in rhs) else None), []
    if not rows:
        identity: list[list[Expr]] = [[S.One if i == j else S.Zero for j in range(columns)] for i in range(columns)]
        zeros: list[Expr] = [S.Zero]*columns
        return (zeros if rhs is not None else None), identity
    if rhs is not None:
        rows = [row + [v] for row, v in zip(rows, rhs)]
    matrix = Matrix(rows)
    try:
        reduced, pivots = DomainMatrix.from_Matrix(matrix, extension=True).to_field().rref()
        R = reduced.to_Matrix()
    except (CoercionFailed, GeneratorsError, NotImplementedError, ValueError):
        R, pivots_ = matrix.rref(simplify=True)
        pivots = tuple(pivots_)
    pivot_columns = list(pivots)
    if rhs is not None and columns in pivot_columns:
        return None, _nullspace_from_rref(R, pivot_columns, columns)
    particular: Optional[list[Expr]] = None
    if rhs is not None:
        particular = [S.Zero for _ in range(columns)]
        for row, column in enumerate(pivot_columns):
            particular[column] = as_expr(R[row, columns])
    return particular, _nullspace_from_rref(R, pivot_columns, columns)


def _nullspace_from_rref(R: Matrix, pivot_columns: list[int], columns: int) -> list[list[Expr]]:
    basis: list[list[Expr]] = []
    for free in range(columns):
        if free in pivot_columns:
            continue
        vector: list[Expr] = [S.Zero for _ in range(columns)]
        vector[free] = S.One
        for row, column in enumerate(pivot_columns):
            vector[column] = as_expr(-R[row, free])
        basis.append(vector)
    return basis


# ---------------------------------------------------------------------------
# polynomial solutions

def _degree_bound(polys: Sequence[Poly], n: Symbol, rhs_degree: Optional[int]) -> Optional[int]:
    """The largest possible degree of a polynomial solution of the
    recurrence (with a right-hand side of degree ``rhs_degree``), or
    ``None`` when there is none. ``L[n**e]`` has, generically, the degree
    ``D + e - k`` where ``k`` is the index of the first of the polynomials
    ``b_k(e) = sum_j (sum_i a_{i, D-k+j} i**j) binomial(e, j)`` which is
    not identically zero (``a_{ij}`` the coefficient of ``n**j`` in
    ``p_i``); the degree is lower exactly at the roots of ``b_k``."""
    D = max(p.degree() for p in polys)
    order = len(polys) - 1
    e = Dummy('e')
    table = [[as_expr(c) for c in reversed(p.all_coeffs())] for p in polys]

    def a(i: int, j: int) -> Expr:
        return table[i][j] if 0 <= j < len(table[i]) else S.Zero

    for k in range(D + order + 2):
        total = S.Zero
        for j in range(k + 1):
            m = D - k + j
            if m < 0:
                continue
            s = Add(*[a(i, m)*Integer(i)**j for i in range(order + 1)])
            total += s*_binomial_polynomial(e, j)
        b = as_expr(total.expand())
        if b != 0:
            candidates = _nonnegative_integer_roots(b, e)
            if rhs_degree is not None:
                candidates.append(rhs_degree - D + k)
            candidates = [c for c in candidates if c >= 0]
            return max(candidates) if candidates else None
    return None  # the zero operator


def _binomial_polynomial(e: Symbol, j: int) -> Expr:
    """``binomial(e, j)`` as a polynomial in ``e``."""
    result: Expr = S.One
    for i in range(j):
        result = as_expr(result*(e - i)/(i + 1))
    return as_expr(result.expand())


def _image_of_power(polys: Sequence[Poly], n: Symbol, j: int) -> Poly:
    """``L[n**j] = sum p_i(n) (n + i)**j``."""
    result = Poly(S.Zero, n, domain=polys[0].domain)
    for i, p in enumerate(polys):
        result = result + p*Poly(n + i, n)**j
    return result


def _polynomial_solutions(polys: Sequence[Poly], n: Symbol, rhs: Expr) -> tuple[Optional[Expr], list[Expr]]:
    """A particular polynomial solution of ``L y = rhs`` (``None`` when
    there is none) and a basis of the polynomial solutions of
    ``L y = 0``."""
    rhs_degree: Optional[int] = None
    if rhs != 0:
        rhs_degree = _poly(rhs, n).degree()
    bound = _degree_bound(polys, n, rhs_degree)
    if bound is None:
        return (S.Zero if rhs == 0 else None), []
    # the images L[n**j] as polynomials: the columns of the linear system
    images = [_image_of_power(polys, n, j) for j in range(bound + 1)]
    rhs_poly = _poly(rhs, n) if rhs != 0 else None
    height = max([int(p.degree()) for p in images if not p.is_zero]
                 + ([int(rhs_poly.degree())] if rhs_poly is not None else []) + [0]) + 1
    rows = [[as_expr(image.nth(e)) for image in images] for e in range(height)]
    constants = [as_expr(rhs_poly.nth(e)) if rhs_poly is not None else S.Zero for e in range(height)]
    particular, basis = _linear_system(rows, constants if rhs != 0 else None, bound + 1)
    polynomials = [as_expr(Add(*[v*n**i for i, v in enumerate(vector)])) for vector in basis]
    if rhs == 0:
        return S.Zero, polynomials
    if particular is None:
        return None, polynomials
    return as_expr(Add(*[v*n**i for i, v in enumerate(particular)])), polynomials


def rsolve_polynomial(coefficients: Sequence[Union[Expr, int]], n: Symbol) -> list[Expr]:
    """A basis of the polynomial solutions of
    ``sum coefficients[i] * y(n + i) = 0``.

    Examples
    ========

    >>> from sympy.abc import n
    >>> from sympy_extras.solvers.recurr import rsolve_polynomial
    >>> rsolve_polynomial([1, -2, 1], n)
    [1, n]
    >>> rsolve_polynomial([n + 1, -n], n)
    [n]
    >>> rsolve_polynomial([(n + 1)**2, -n**2 - 1], n)
    []
    """
    polys = _normalized(_coefficient_polys(coefficients, n), n)
    if not polys:
        raise ValueError("the recurrence is trivial")
    return _polynomial_solutions(polys, n, S.Zero)[1]


# ---------------------------------------------------------------------------
# the Gosper–Petkovšek form

def gosper_form(r: Expr, n: Symbol) -> tuple[Expr, Poly, Poly, Poly]:
    """The Gosper–Petkovšek form ``(z, a, b, c)`` of a rational function:
    ``r = z * a(n)/b(n) * c(n + 1)/c(n)`` with ``a``, ``b``, ``c`` monic
    polynomials and ``gcd(a(n), b(n + h)) = 1`` for every nonnegative
    integer ``h``. ``r`` is the quotient ``q(n + 1)/q(n)`` of a rational
    function if and only if ``z = 1`` and ``a = b = 1``.

    Examples
    ========

    >>> from sympy.abc import n
    >>> from sympy_extras.solvers.recurr import gosper_form
    >>> gosper_form((n + 2)/n, n)
    (1, Poly(1, n, domain='QQ'), Poly(1, n, domain='QQ'), Poly(n**2 + n, n, domain='QQ'))
    >>> gosper_form(2*(n + 1)/(n + 3), n)
    (2, Poly(n + 1, n, domain='QQ'), Poly(n + 3, n, domain='QQ'), Poly(1, n, domain='QQ'))

    References
    ==========

    .. [1] M. Petkovšek, Hypergeometric solutions of linear recurrences
       with polynomial coefficients, J. Symbolic Computation 14 (1992),
       Lemma 3.1 and Algorithm 3.1.
    """
    numerator, denominator = fraction(cancel(as_expr(r)))
    P = _poly(as_expr(numerator), n)
    Q = _poly(as_expr(denominator), n)
    if P.is_zero:
        raise ValueError("the ratio is zero")
    z = as_expr(cancel(as_expr(P.LC())/as_expr(Q.LC())))
    a = P.monic()
    b = Q.monic()
    c = Poly(S.One, n, domain=a.domain)
    for root in _shift_coincidences(a, b):
        g = a.gcd(b.shift(root))
        if g.degree() <= 0:
            continue
        a = a.quo(g).monic()
        b = b.quo(g.shift(-root)).monic()
        for i in range(1, root + 1):
            c = c*g.shift(-i)
    return z, a, b, c.monic()


def _is_shift_quotient(r: Expr, n: Symbol) -> Optional[Poly]:
    """The monic polynomial ``q`` with ``r = q(n + 1)/q(n)``, or ``None``
    when ``r`` is not such a quotient."""
    z, a, b, c = gosper_form(r, n)
    if z == 1 and a.degree() == 0 and b.degree() == 0:
        return c
    return None


# ---------------------------------------------------------------------------
# hypergeometric solutions

def _shift_coincidences(a: Poly, b: Poly) -> list[int]:
    """The nonnegative integers ``h`` with ``gcd(a(n), b(n + h)) != 1``:
    an irreducible factor ``f`` of ``a`` and one ``g`` of ``b`` have a
    common root when ``g(n + h) = f(n)``, and ``h`` is then read off the
    second coefficients (the resultant in ``n`` of ``a(n)`` and ``b(n + h)``
    is a polynomial in ``h`` with these roots, but far more expensive)."""
    if a.degree() <= 0 or b.degree() <= 0:
        return []
    found: set[int] = set()
    factors_a = [f.monic() for f, _ in a.factor_list()[1]]
    factors_b = [g.monic() for g, _ in b.factor_list()[1]]
    for f in factors_a:
        m = f.degree()
        for g in factors_b:
            if g.degree() != m:
                continue
            h = as_expr(cancel((as_expr(f.nth(m - 1)) - as_expr(g.nth(m - 1)))/m))
            if h.is_Integer and h >= 0 and g.shift(int(h)) == f:
                found.add(int(h))
    return sorted(found)


def _products(p: Poly, n: Symbol, lo: int, hi: int) -> Poly:
    """``prod_{j = lo}^{hi - 1} p(n + j)``."""
    result = Poly(S.One, n, domain=p.domain)
    for j in range(lo, hi):
        result = result*p.shift(j)
    return result


def _divisor_pairs(p0: Poly, pd: Poly, n: Symbol) -> list[tuple[Poly, Poly]]:
    """The pairs ``(a, b)`` of monic divisors of ``p0`` and ``pd`` with
    ``gcd(a(n), b(n + h)) = 1`` for every nonnegative integer ``h``: a
    pair of irreducible factors, one of each, is incompatible when one is
    a shift of the other by a nonnegative integer, and the divisors are
    built from the compatible factors."""
    factors_a = [(f.monic(), int(e)) for f, e in p0.factor_list()[1]]
    factors_b = [(g.monic(), int(e)) for g, e in pd.factor_list()[1]]
    incompatible = {(i, j) for i, (f, _) in enumerate(factors_a) for j, (g, _) in enumerate(factors_b)
                    if _shift_coincidences(f, g)}
    one = Poly(S.One, n, domain=p0.domain.unify(pd.domain))

    def divisors(factors: list[tuple[Poly, int]]) -> list[tuple[Poly, frozenset[int]]]:
        found: list[tuple[Poly, frozenset[int]]] = [(one, frozenset())]
        for index, (f, multiplicity) in enumerate(factors):
            found = [(d*f**e, used | ({index} if e else frozenset()))
                     for d, used in found for e in range(multiplicity + 1)]
        return found

    pairs: list[tuple[Poly, Poly]] = []
    for a, used_a in divisors(factors_a):
        for b, used_b in divisors(factors_b):
            if any((i, j) in incompatible for i in used_a for j in used_b):
                continue
            pairs.append((a, b))
    return pairs


def _index(depth: int, avoid: set[Symbol]) -> Symbol:
    """A summation index for the given depth which is none of the
    symbols of the problem."""
    names = list(_INDEX_NAMES) + ['%s%d' % (c, i) for i in range(1, 100) for c in _INDEX_NAMES]
    taken = {s.name for s in avoid}
    free = [name for name in names if name not in taken]
    return Symbol(free[depth], integer=True)


def _start_of(a: Poly, b: Poly, n: Symbol) -> int:
    """The first index from which ``prod a(k)/b(k)`` has no vanishing
    factor: one more than the largest nonnegative integer root of ``a b``."""
    roots_ = _nonnegative_integer_roots(as_expr(a.as_expr()*b.as_expr()), n)
    return roots_[-1] + 1 if roots_ else 0


def _product_expression(p: Poly, n: Symbol, start: int, k: Symbol) -> Expr:
    """``prod_{k = start}^{n - 1} p(k)`` with the linear factors written as
    rising factorials."""
    content, factors = p.factor_list()
    result: Expr = as_expr(as_expr(content)**(n - start))
    for f, multiplicity in factors:
        lead = as_expr(f.LC())
        if f.degree() == 1:
            root = as_expr(-f.nth(0)/lead)
            piece: Expr = as_expr(lead**(n - start)*RisingFactorial(start - root, n - start))
        else:
            piece = as_expr(Product(as_expr(f.as_expr().subs(n, k)), (k, start, n - 1)))
        result = as_expr(result*piece**int(multiplicity))
    return result


def _hypergeometric_expression(c: Expr, z: Expr, a: Poly, b: Poly, n: Symbol, start: int, k: Symbol) -> Expr:
    """``c(n) z**n prod_{k = start}^{n - 1} a(k)/b(k)``."""
    core = as_expr(z**n*_product_expression(a, n, start, k)/_product_expression(b, n, start, k))
    return _simplified(as_expr(c*core))


def _simplified(expr: Expr) -> Expr:
    """The quotients of rising factorials cancelled (``gammasimp``), when
    that keeps the expression free of gamma functions."""
    candidate = as_expr(gammasimp(expr))
    if candidate.has(gamma, Piecewise):
        candidate = expr
    return as_expr(candidate.replace(RisingFactorial, _factorial_form))


def _factorial_form(a: Expr, m: Expr) -> Expr:
    """``RisingFactorial(a, m)`` as ``(a + m - 1)!/(a - 1)!`` for a positive
    integer ``a``."""
    if a.is_Integer and a >= 1:
        return as_expr(factorial(a + m - 1)/factorial(a - 1))
    return as_expr(RisingFactorial(a, m))


def _closed_form(expr: Expr) -> Expr:
    """The sums evaluated by SymPy when that gives an expression without
    sums, products or special functions."""
    if not expr.has(Sum):
        return expr
    if any(s.function.has(Sum) for s in expr.atoms(Sum)):
        return expr  # SymPy's symbolic summation of nested sums took minutes
    for form in (expr, as_expr(expr.replace(Sum, _from_one))):
        try:
            candidate = attempt(lambda: as_expr(form.doit()), CLOSED_FORM_SECONDS)
        except (ValueError, TypeError, NotImplementedError):
            continue
        if candidate is not None and not candidate.has(Sum, Product, gamma, uppergamma, lowergamma, hyper, polylog,
                                                       lerchphi, Piecewise, zeta):
            return candidate
    return expr


def _from_one(function: Expr, *limits: tuple[Symbol, Expr, Expr]) -> Expr:
    """``Sum(f(k), (k, 0, m))`` as ``Sum(f(k - 1), (k, 1, m + 1))``, the form
    in which SymPy recognises the harmonic numbers."""
    if len(limits) == 1 and limits[0][1] == 0:
        k, _, hi = limits[0]
        return as_expr(Sum(function.subs(k, k - 1), (k, 1, hi + 1)))
    return as_expr(Sum(function, *limits))


class _Class:
    """A similarity class of hypergeometric terms: a core
    ``z**n prod a/b`` and the polynomials multiplying it."""

    def __init__(self, z: Expr, a: Poly, b: Poly, ratio: Expr, start: int) -> None:
        self.z, self.a, self.b, self.ratio, self.start = z, a, b, ratio, start
        self.polynomials: list[Poly] = []


def _hypergeometric_records(polys: Sequence[Poly], n: Symbol, depth: int, avoid: set[Symbol]
                            ) -> list[RecurrenceSolution]:
    """The hypergeometric solutions with their ratios (Petkovšek)."""
    d = len(polys) - 1
    if d == 0:
        return []
    k = _index(depth, avoid)
    classes: list[_Class] = []
    z_symbol = Dummy('z')
    for a, b in _divisor_pairs(polys[0], polys[d].shift(1 - d), n):
        P = [polys[i]*_products(a, n, 0, i)*_products(b, n, i, d) for i in range(d + 1)]
        m = max(p.degree() for p in P)
        leading = as_expr(Add(*[as_expr(p.LC())*z_symbol**i for i, p in enumerate(P) if p.degree() == m]))
        for z in _nonzero_roots(leading, z_symbol):
            aux = _normalized([_scaled(p, as_expr(z**i), n) for i, p in enumerate(P)], n)
            if not aux:
                continue
            for c in _polynomial_solutions(aux, n, S.Zero)[1]:
                cp = Poly(c, n)
                ratio = as_expr(cancel(z*a.as_expr()*cp.shift(1).as_expr()/(b.as_expr()*cp.as_expr())))
                _classify(classes, z, a, b, cp, ratio, n)
    records: list[RecurrenceSolution] = []
    for cls in classes:
        for member in _polynomial_basis(cls.polynomials, n):
            ratio = as_expr(cancel(cls.ratio*member.shift(1).as_expr()/member.as_expr()))
            expr = _hypergeometric_expression(member.as_expr(), cls.z, cls.a, cls.b, n, cls.start, k)
            records.append((expr, ratio, cls.start))
    return records


def _scaled(p: Poly, factor: Expr, n: Symbol) -> Poly:
    """``factor * p`` for a constant ``factor``, possibly algebraic."""
    try:
        return p.mul_ground(factor)
    except (CoercionFailed, GeneratorsError, TypeError, ValueError):
        return _poly(as_expr((factor*p.as_expr()).expand()), n)


def _nonzero_roots(expr: Expr, z: Symbol) -> list[Expr]:
    """The nonzero roots of a polynomial in ``z``, exactly."""
    p = Poly(expr, z)
    if p.degree() <= 0:
        return []
    found = roots(p, filter=None)
    values = [as_expr(sympify(r)) for r in found]
    if sum(found.values()) < p.degree() and p.domain.is_Numerical:
        values = [as_expr(sympify(r)) for r in p.all_roots()]
    return [v for v in dict.fromkeys(values) if v != 0]


def _classify(classes: list[_Class], z: Expr, a: Poly, b: Poly, c: Poly, ratio: Expr, n: Symbol) -> None:
    """Put the term ``c z**n prod a/b`` into its similarity class."""
    for cls in classes:
        q = _is_shift_quotient(as_expr(cancel(ratio/cls.ratio)), n)
        if q is not None:
            cls.polynomials.append(q)
            return
    core_ratio = as_expr(cancel(z*a.as_expr()/b.as_expr()))
    cls = _Class(z, a, b, core_ratio, _start_of(a, b, n))
    cls.polynomials.append(c.monic())
    classes.append(cls)


def _polynomial_basis(polynomials: Sequence[Poly], n: Symbol) -> list[Poly]:
    """A basis of the span of the polynomials."""
    if not polynomials:
        return []
    degree = max(p.degree() for p in polynomials)
    rows = [[as_expr(p.nth(j)) for j in range(degree + 1)] for p in polynomials]
    try:
        reduced, pivots = DomainMatrix.from_Matrix(Matrix(rows), extension=True).to_field().rref()
        R = reduced.to_Matrix()
    except (CoercionFailed, GeneratorsError, NotImplementedError, ValueError):
        R, pivots_ = Matrix(rows).rref(simplify=True)
        pivots = tuple(pivots_)
    basis: list[Poly] = []
    for row in range(len(pivots)):
        basis.append(Poly(Add(*[as_expr(R[row, j])*n**j for j in range(degree + 1)]), n))
    return basis


def rsolve_hypergeometric(coefficients: Sequence[Union[Expr, int]], n: Symbol) -> list[Expr]:
    """A basis of the hypergeometric solutions of
    ``sum coefficients[i] * y(n + i) = 0``, the solutions ``y`` whose ratio
    ``y(n + 1)/y(n)`` is a rational function of ``n``.

    Examples
    ========

    >>> from sympy.abc import n
    >>> from sympy_extras.solvers.recurr import rsolve_hypergeometric
    >>> rsolve_hypergeometric([-(n + 1), 1], n)
    [factorial(n)]
    >>> rsolve_hypergeometric([-1, -1, 1], n)
    [(1/2 - sqrt(5)/2)**n, (1/2 + sqrt(5)/2)**n]
    >>> rsolve_hypergeometric([n**2 + 1, -(n + 1)], n)
    [Product(k**2 + 1, (k, 0, n - 1))/factorial(n)]
    >>> rsolve_hypergeometric([n + 1, -(n + 3), 1], n)
    [factorial(n)]

    References
    ==========

    .. [1] M. Petkovšek, Hypergeometric solutions of linear recurrences
       with polynomial coefficients, J. Symbolic Computation 14 (1992).
    """
    polys = _normalized(_coefficient_polys(coefficients, n), n)
    if not polys:
        raise ValueError("the recurrence is trivial")
    avoid = {n} | free_symbols(Add(*[p.as_expr() for p in polys]))
    records = _hypergeometric_records(polys, n, 0, avoid)
    return [expr for expr, _, start in records if _checked(polys, n, expr, S.Zero, start)]


# ---------------------------------------------------------------------------
# d'Alembertian solutions

def _reduced_operator(polys: Sequence[Poly], n: Symbol, ratio: Expr) -> list[Poly]:
    """The operator ``L'`` of order ``d - 1`` with ``L[h sum z] =
    h L'[z]`` for the hypergeometric solution ``h`` of ratio ``r``:
    ``L' z = sum_j (sum_{i > j} p_i(n) rho_i(n)) z(n + j)`` with
    ``rho_i = prod_{m < i} r(n + m)``, cleared of denominators."""
    d = len(polys) - 1
    rho: list[Expr] = [S.One]
    for i in range(1, d + 1):
        rho.append(as_expr(cancel(rho[-1]*ratio.subs(n, n + i - 1))))
    coefficients = [as_expr(cancel(Add(*[polys[i].as_expr()*rho[i] for i in range(j + 1, d + 1)])))
                    for j in range(d)]
    return _normalized(_coefficient_polys(coefficients, n), n)


def _dalembertian_records(polys: Sequence[Poly], n: Symbol, depth: int, avoid: set[Symbol]
                          ) -> list[RecurrenceSolution]:
    hyper = _hypergeometric_records(polys, n, depth, avoid)
    if not hyper:
        return []
    records = list(hyper)
    order = len(polys) - 1
    if order <= 1 or len(hyper) >= order:
        return records
    # the hypergeometric solutions come first; the sums h sum z add the
    # directions the hypergeometric ones do not span
    h, ratio, start = hyper[0]
    reduced = _reduced_operator(polys, n, ratio)
    k = _index(depth, avoid)
    for z, _, z_start in _dalembertian_records(reduced, n, depth + 1, avoid):
        lower = max(start, z_start)
        total = as_expr(Sum(z.subs(n, k), (k, lower, n - 1)))
        records.append((as_expr(h*total), S.Zero, lower))
    return _independent(records, polys, n)


def _independent(records: list[RecurrenceSolution], polys: Sequence[Poly], n: Symbol) -> list[RecurrenceSolution]:
    """A maximal independent subset of the solutions, the earlier ones
    preferred: solutions of a recurrence of order ``d`` are independent
    when their values at ``d`` consecutive points, past the roots of the
    leading coefficient, are."""
    d = len(polys) - 1
    start = max([r[2] for r in records] + _nonnegative_integer_roots(polys[d].as_expr(), n) + [-1]) + 1
    chosen: list[RecurrenceSolution] = []
    rows: list[list[Expr]] = []
    for record in records:
        if len(chosen) >= d:
            break
        values = [_value(record[0], n, at) for at in range(start, start + d)]
        if any(v.is_finite is False for v in values):
            continue
        if _rank(rows + [values]) > len(rows):
            rows.append(values)
            chosen.append(record)
    return chosen


def _rank(rows: list[list[Expr]]) -> int:
    matrix = Matrix(rows)
    try:
        return DomainMatrix.from_Matrix(matrix, extension=True).to_field().rank()
    except (CoercionFailed, GeneratorsError, NotImplementedError, ValueError):
        return matrix.rank(simplify=True)


def rsolve_dalembertian(coefficients: Sequence[Union[Expr, int]], n: Symbol) -> list[Expr]:
    """A basis of the d'Alembertian solutions of
    ``sum coefficients[i] * y(n + i) = 0``: the nested indefinite sums of
    hypergeometric terms (Abramov and Petkovšek). When their number is the
    order of the recurrence, they are all the solutions.

    Examples
    ========

    >>> from sympy.abc import n
    >>> from sympy_extras.solvers.recurr import rsolve_dalembertian
    >>> rsolve_dalembertian([n + 1, -(2*n + 3), n + 2], n)
    [1, harmonic(n)]
    >>> rsolve_dalembertian([n + 1, -(n + 3), 1], n)
    [factorial(n), factorial(n)*Sum(1/factorial(k + 1), (k, 0, n - 1))]
    >>> rsolve_dalembertian([1, -2, 1], n)
    [1, n]
    >>> rsolve_dalembertian([1, -(n + 1), 1], n)
    []

    References
    ==========

    .. [1] S. A. Abramov, M. Petkovšek, D'Alembertian solutions of linear
       differential and difference equations, ISSAC '94.
    """
    polys = _normalized(_coefficient_polys(coefficients, n), n)
    if not polys:
        raise ValueError("the recurrence is trivial")
    avoid = {n} | free_symbols(Add(*[p.as_expr() for p in polys]))
    records = _dalembertian_records(polys, n, 0, avoid)
    return [_closed_form(expr) for expr, _, start in records if _checked(polys, n, expr, S.Zero, start)]


# ---------------------------------------------------------------------------
# verification

def _value(expr: Expr, n: Symbol, at: int) -> Expr:
    """``expr`` at ``n = at``, the sums and products with integer limits
    written out (SymPy's ``doit`` looks for closed forms first, which is
    slow and may fail)."""
    return _evaluated(as_expr(expr.subs(n, at)))


def _evaluated(expr: Expr) -> Expr:
    if not expr.has(Sum, Product):
        return as_expr(expr.doit()) if expr.has(RisingFactorial, factorial) else expr
    if isinstance(expr, (Sum, Product)):
        function = as_expr(expr.function)
        for limit in expr.limits:
            index, lo, hi = limit
            lo_, hi_ = as_expr(lo), as_expr(hi)
            if not (lo_.is_Integer and hi_.is_Integer):
                return as_expr(expr.doit())
            values = [_evaluated(as_expr(function.subs(index, i))) for i in range(int(lo_), int(hi_) + 1)]
            function = as_expr(Add(*values)) if isinstance(expr, Sum) else as_expr(Mul(*values))
        return function
    if not expr.args:
        return expr
    return as_expr(expr.func(*[_evaluated(as_expr(a)) if isinstance(a, Expr) else a for a in expr.args]))


def _is_zero(value: Expr) -> bool:
    simplified = as_expr(simplify(value))
    if simplified == 0:
        return True
    if simplified.is_number:
        approximation = as_expr(simplified.evalf(30))
        return bool(abs(approximation) < Rational(1, 10**20))
    return False


def is_recurrence_solution(coefficients: Sequence[Union[Expr, int]], n: Symbol, solution: Expr,
                           rhs: Union[Expr, int] = 0, start: int = 0, count: int = 6) -> bool:
    """Whether ``solution`` satisfies ``sum coefficients[i] y(n + i) =
    rhs`` at ``n = start, ..., start + count - 1``, exactly.

    Examples
    ========

    >>> from sympy import factorial, harmonic
    >>> from sympy.abc import n
    >>> from sympy_extras.solvers.recurr import is_recurrence_solution
    >>> is_recurrence_solution([-(n + 1), 1], n, factorial(n))
    True
    >>> is_recurrence_solution([n + 1, -(2*n + 3), n + 2], n, harmonic(n))
    True
    >>> is_recurrence_solution([-2, 1], n, 3**n)
    False
    """
    polys = _coefficient_polys(coefficients, n)
    return _checked(polys, n, as_expr(sympify(solution)), as_expr(sympify(rhs)), start, count)


def _checked(polys: Sequence[Poly], n: Symbol, solution: Expr, rhs: Expr, start: int, count: int = 6) -> bool:
    for at in range(start, start + count):
        value = as_expr(Add(*[_value(as_expr(p.as_expr()), n, at)*_value(solution, n, at + i)
                              for i, p in enumerate(polys)]) - _value(rhs, n, at))
        if value.is_finite is False or not _is_zero(value):
            return False
    return True


# ---------------------------------------------------------------------------
# equations

def _parse(equation: Union[Expr, Eq], y: AppliedUndef) -> tuple[list[Expr], Expr, Symbol]:
    """``(coefficients, rhs, n)`` of a linear recurrence written with
    ``y(n + i)``: ``sum coefficients[i] y(n + i) = rhs``, the index
    shifted so that the least shift is zero (the same recurrence)."""
    n = as_symbol(y.args[0])
    expr = as_expr(equation.lhs - equation.rhs) if isinstance(equation, Eq) else as_expr(equation)
    terms = [t for t in expr.atoms(AppliedUndef) if t.func == y.func]
    if not terms:
        raise ValueError("no %s in the equation" % y)
    shifts: dict[AppliedUndef, int] = {}
    for t in terms:
        offset = as_expr(t.args[0] - n)
        if not offset.is_Integer:
            raise ValueError("%s is not a shift of %s by an integer" % (t, y))
        shifts[t] = int(offset)
    least = min(shifts.values())
    expr = as_expr(numer(together(expr.subs(n, n - least))))
    terms = sorted((t for t in expr.atoms(AppliedUndef) if t.func == y.func), key=lambda t: as_expr(t.args[0] - n))
    poly = Poly(expr, *terms)
    if poly.total_degree() > 1:
        raise ValueError("the equation is not linear in %s" % y)
    order = max(int(as_expr(t.args[0] - n)) for t in terms)
    coefficients = [as_expr(poly.coeff_monomial(y.func(n + i))) if y.func(n + i) in terms else S.Zero
                    for i in range(order + 1)]
    rhs = as_expr(-poly.coeff_monomial(S.One))
    for c in coefficients + [rhs]:
        if c.has(y.func):
            raise ValueError("the equation is not linear in %s" % y)
    return coefficients, rhs, n


def _hypergeometric_ratio(f: Expr, n: Symbol) -> Optional[Expr]:
    """``f(n + 1)/f(n)`` when it is a rational function of ``n``."""
    if f == 0:
        return None
    ratio = as_expr(cancel(as_expr(simplify(as_expr(f.subs(n, n + 1)/f)))))
    if ratio.is_rational_function(n) and not ratio.has(Sum, Product):
        return ratio
    return None


def _particular(polys: Sequence[Poly], n: Symbol, f: Expr, avoid: set[Symbol]) -> Optional[tuple[Expr, int]]:
    """A d'Alembertian particular solution of ``L y = f`` for a
    hypergeometric ``f``: a solution ``u`` of ``(N - rho) L y = 0``,
    ``rho = f(n + 1)/f(n)``, with ``L u = kappa f`` for a nonzero
    constant ``kappa``, scaled."""
    rho = _hypergeometric_ratio(f, n)
    if rho is None:
        return None
    d = len(polys) - 1
    extended: list[Expr] = []
    for i in range(d + 2):
        previous = polys[i - 1].shift(1).as_expr() if 1 <= i <= d + 1 else S.Zero
        current = polys[i].as_expr() if i <= d else S.Zero
        extended.append(as_expr(cancel(previous - rho*current)))
    operator = _normalized(_coefficient_polys(extended, n), n)
    for u, _, start in _dalembertian_records(operator, n, 0, avoid):
        for at in range(start, start + 4):
            value = as_expr(Add(*[_value(as_expr(p.as_expr()), n, at)*_value(u, n, at + i)
                                  for i, p in enumerate(polys)]))
            f_value = _value(f, n, at)
            if f_value == 0 or value.is_finite is False or f_value.is_finite is False:
                continue
            kappa = as_expr(simplify(value/f_value))
            if _is_zero(kappa):
                break
            candidate = as_expr(u/kappa)
            if _checked(polys, n, candidate, f, start):
                return candidate, start
            break
    return None


def rsolve_linear(equation: Union[Expr, Eq], y: AppliedUndef, init: Optional[dict[AppliedUndef, Union[Expr, int]]] = None
                  ) -> Optional[Expr]:
    """The general solution of a linear recurrence with polynomial (or
    rational) coefficients, written with ``y(n + i)``, as a d'Alembertian
    expression with the constants ``C0, C1, ...``; ``None`` when the
    d'Alembertian solutions do not span the solutions (fewer than the
    order), or when the right-hand side is not a sum of hypergeometric
    terms with a d'Alembertian particular solution. With ``init``, the
    values ``{y(0): ..., y(1): ...}`` determine the constants.

    Examples
    ========

    >>> from sympy import Function, Eq, factorial
    >>> from sympy.abc import n
    >>> from sympy_extras.solvers.recurr import rsolve_linear
    >>> y = Function('y')
    >>> rsolve_linear(y(n + 1) - (n + 1)*y(n) - 1, y(n))
    C0*factorial(n) + factorial(n)*Sum(1/factorial(k + 1), (k, 0, n - 1))
    >>> rsolve_linear(Eq((n + 2)*y(n + 2), (2*n + 3)*y(n + 1) - (n + 1)*y(n)), y(n), {y(0): 0, y(1): 1})
    harmonic(n)
    >>> rsolve_linear(y(n + 2) - y(n + 1) - y(n), y(n), {y(0): 0, y(1): 1})
    -sqrt(5)*(1/2 - sqrt(5)/2)**n/5 + sqrt(5)*(1/2 + sqrt(5)/2)**n/5
    >>> rsolve_linear(y(n + 1) - 2*y(n) - n, y(n))
    2**n*C0 - n - 1
    >>> rsolve_linear(y(n + 2) - (n + 1)*y(n + 1) + y(n), y(n)) is None
    True

    The derangement numbers, ``y(n + 1) = n (y(n) + y(n - 1))``:

    >>> d = rsolve_linear(y(n + 1) - n*(y(n) + y(n - 1)), y(n), {y(0): 1, y(1): 0})
    >>> d
    -factorial(n)*Sum((-1)**k/factorial(k + 1), (k, 0, n - 1)) + factorial(n)
    >>> [d.subs(n, i).doit() for i in range(7)]
    [1, 0, 1, 2, 9, 44, 265]
    """
    coefficients, rhs, n = _parse(equation, y)
    polys = _normalized(_coefficient_polys(coefficients, n), n)
    if not polys:
        raise ValueError("the recurrence is trivial")
    order = len(polys) - 1
    avoid = {n} | free_symbols(Add(*[p.as_expr() for p in polys])) | free_symbols(rhs)
    homogeneous = [_closed_form(expr) for expr, _, start in _dalembertian_records(polys, n, 0, avoid)
                   if _checked(polys, n, expr, S.Zero, start)]
    if len(homogeneous) < order:
        return None
    particular: Expr = S.Zero
    if rhs != 0:
        for term in Add.make_args(as_expr(rhs.expand())):
            found = _particular(polys, n, as_expr(term), avoid)
            if found is None:
                return None
            particular = as_expr(particular + _closed_form(found[0]))
    constants = [Symbol('C%d' % i) for i in range(order)]
    general = as_expr(particular + Add(*[c*h for c, h in zip(constants, homogeneous)]))
    if init is None:
        return general
    rows: list[list[Expr]] = []
    values: list[Expr] = []
    for point, value in init.items():
        at = as_expr(point.args[0])
        if not at.is_Integer:
            raise ValueError("the initial values must be given at integers: %s" % point)
        evaluated = _value(general, n, int(at))
        rows.append([as_expr(evaluated.diff(c)) for c in constants])
        values.append(as_expr(as_expr(sympify(value)) - evaluated.subs({c: 0 for c in constants})))
    solution, _ = _linear_system(rows, values, len(constants))
    if solution is None:
        raise ValueError("the initial values are inconsistent with the recurrence")
    return as_expr(general.subs(dict(zip(constants, solution))))
