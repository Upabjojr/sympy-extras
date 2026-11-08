"""Gosper's algorithm for the indefinite summation of hypergeometric terms
(SymPy's ``gosper_normal``, ``gosper_term``, ``gosper_sum``), with
``hypersimp``.

Ported from SymPy's ``sympy.concrete.gosper``. SymPy works with
polynomials over a field of the parameters; here a polynomial in ``n`` is
the list of its coefficients, expressions kept cancelled (the elements of
that field), with the resultant from the Sylvester determinant and its
nonnegative integer roots from its linear factors.

Examples
========

>>> from sympy_extras.core import Symbol
>>> from sympy_extras.functions import binomial, factorial
>>> from sympy_extras.concrete.gosper import gosper_sum, gosper_term
>>> i, m, n = Symbol('i', integer=True), Symbol('m', integer=True), Symbol('n', integer=True)
>>> gosper_sum(i*factorial(i), (i, 0, n)), gosper_term(1/(i*(i + 1)), i)
(n*factorial(n) + factorial(n) - 1, -i - 1)
>>> gosper_sum(binomial(m, i)*(-1)**i, (i, 0, n))
(-1)**n*(m - n)*binomial(m, n)/m
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/concrete/gosper.py (gosper_normal, gosper_term,
# gosper_sum), sympy/simplify/simplify.py (hypersimp) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: typed; the polynomials are coefficient lists over the cancelled
# expressions; the linear system of the coefficients is solved by
# elimination.
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Optional, Sequence, Union

from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.numbers import Integer, NaN
from sympy_extras.core.symbol import Dummy, Symbol

__all__ = ['hypersimp', 'gosper_normal', 'gosper_term', 'gosper_sum']

ZERO = Integer(0)
ONE = Integer(1)


def _c(e: Expr) -> Expr:
    from sympy_extras.polys.polytools import cancel
    return cancel(e)


class _UPoly:
    """A polynomial in ``n``: the coefficients, the constant first, each
    cancelled (an element of the field of the parameters)."""

    __slots__ = ('c',)

    def __init__(self, coeffs: Sequence[Expr]) -> None:
        cs = [_c(e) for e in coeffs]
        while cs and cs[-1] == 0:
            cs.pop()
        self.c: list[Expr] = cs

    @classmethod
    def of(cls, e: Expr, n: Symbol) -> _UPoly:
        """``e``, a polynomial in ``n`` (``ValueError`` otherwise)."""
        from sympy_extras.core.add import Add
        from sympy_extras.core.function import expand
        by: dict[int, list[Expr]] = {}
        ex = expand(e)
        for t in (ex.args if isinstance(ex, Add) else (ex,)):
            assert isinstance(t, Expr)
            coeff, k = t.as_coeff_exponent(n)
            if coeff.has(n) or not isinstance(k, Integer) or k.p < 0:
                raise ValueError('%s is not a polynomial in %s' % (e, n))
            by.setdefault(k.p, []).append(coeff)
        from sympy_extras.core.add import add
        top = max(by) if by else 0
        return cls([add(*by.get(k, [])) for k in range(top + 1)])

    def is_zero(self) -> bool:
        return not self.c

    def degree(self) -> int:
        return len(self.c) - 1

    def lc(self) -> Expr:
        return self.c[-1] if self.c else ZERO

    def nth(self, k: int) -> Expr:
        return self.c[k] if 0 <= k < len(self.c) else ZERO

    def as_expr(self, n: Expr) -> Expr:
        from sympy_extras.core.add import add
        return add(*[c * n**k for k, c in enumerate(self.c)])

    def __add__(self, other: _UPoly) -> _UPoly:
        size = max(len(self.c), len(other.c))
        return _UPoly([self.nth(k) + other.nth(k) for k in range(size)])

    def __sub__(self, other: _UPoly) -> _UPoly:
        size = max(len(self.c), len(other.c))
        return _UPoly([self.nth(k) - other.nth(k) for k in range(size)])

    def __mul__(self, other: _UPoly) -> _UPoly:
        if self.is_zero() or other.is_zero():
            return _UPoly([])
        out: list[Expr] = [ZERO] * (len(self.c) + len(other.c) - 1)
        for i, a in enumerate(self.c):
            for j, b in enumerate(other.c):
                out[i + j] = out[i + j] + a * b
        return _UPoly(out)

    def scale(self, k: Expr) -> _UPoly:
        return _UPoly([c * k for c in self.c])

    def monic(self) -> _UPoly:
        return self.scale(1 / self.lc()) if self.c else self

    def divmod(self, other: _UPoly) -> tuple[_UPoly, _UPoly]:
        rem = list(self.c)
        q: list[Expr] = [ZERO] * max(len(rem) - len(other.c) + 1, 1)
        lc = other.lc()
        while len(rem) >= len(other.c) and rem:
            k = len(rem) - len(other.c)
            factor = _c(rem[-1] / lc)
            q[k] = factor
            for j, b in enumerate(other.c):
                rem[j + k] = _c(rem[j + k] - factor * b)
            while rem and rem[-1] == 0:
                rem.pop()
        return _UPoly(q), _UPoly(rem)

    def quo(self, other: _UPoly) -> _UPoly:
        return self.divmod(other)[0]

    def gcd(self, other: _UPoly) -> _UPoly:
        """The monic gcd (SymPy's gcd over a field)."""
        a, b = self, other
        while not b.is_zero():
            a, b = b, a.divmod(b)[1]
        return a.monic() if not a.is_zero() else _UPoly([ONE])

    def shift(self, s: Expr) -> _UPoly:
        """``p(n + s)``."""
        from sympy_extras.functions.combinatorial.factorials import binomial
        out: list[Expr] = [ZERO] * len(self.c)
        for k, c in enumerate(self.c):
            for j in range(k + 1):
                out[j] = out[j] + c * binomial(Integer(k), Integer(j)) * s**(k - j)
        return _UPoly(out)


def _determinant(rows: list[list[Expr]]) -> Expr:
    """The determinant by fraction-free (Bareiss) elimination."""
    m = [list(r) for r in rows]
    size = len(m)
    if size == 0:
        return ONE
    sign = 1
    previous: Expr = ONE
    for k in range(size - 1):
        if m[k][k] == 0:
            pivot = next((r for r in range(k + 1, size) if m[r][k] != 0), None)
            if pivot is None:
                return ZERO
            m[k], m[pivot] = m[pivot], m[k]
            sign = -sign
        for i in range(k + 1, size):
            for j in range(k + 1, size):
                m[i][j] = _c((m[i][j] * m[k][k] - m[i][k] * m[k][j]) / previous)
        previous = m[k][k]
    return _c(sign * m[size - 1][size - 1])


def _resultant(a: _UPoly, b: _UPoly) -> Expr:
    """The resultant of ``a`` and ``b`` (the Sylvester determinant)."""
    da, db = a.degree(), b.degree()
    if da < 0 or db < 0:
        return ZERO
    size = da + db
    if size == 0:
        return ONE
    rows: list[list[Expr]] = []
    for k in range(db):
        row: list[Expr] = [ZERO] * size
        for j, c in enumerate(reversed(a.c)):
            row[k + j] = c
        rows.append(row)
    for k in range(da):
        row = [ZERO] * size
        for j, c in enumerate(reversed(b.c)):
            row[k + j] = c
        rows.append(row)
    return _determinant(rows)


def _nonnegative_integer_roots(r: Expr, h: Symbol) -> list[int]:
    """The nonnegative integer roots of ``r`` in ``h`` (SymPy's integer
    ``ground_roots``), from the linear factors of its factorization."""
    from sympy_extras.core.mul import Mul
    from sympy_extras.core.power import Pow
    from sympy_extras.polys.polytools import factor
    from sympy_extras.polys.rationaltools import together
    numer, _ = together(r).as_numer_denom()
    roots: set[int] = set()
    f = factor(numer)
    for t in (f.args if isinstance(f, Mul) else (f,)):
        assert isinstance(t, Expr)
        base = t.args[0] if isinstance(t, Pow) else t
        assert isinstance(base, Expr)
        if not base.has(h):
            continue
        try:
            p = _UPoly.of(base, h)
        except ValueError:
            continue
        if p.degree() == 1:
            root = _c(-p.nth(0) / p.nth(1))
            if isinstance(root, Integer) and root.p >= 0:
                roots.add(root.p)
    return sorted(roots)


# PORTED FROM SYMPY: sympy/concrete/gosper.py:gosper_normal
def gosper_normal(f: Expr, g: Expr, n: Symbol) -> tuple[_UPoly, _UPoly, _UPoly]:
    """The Gosper normal form ``f/g = Z*A*C(n + 1)/(B*C(n))`` of two
    polynomials: ``(Z*A, B, C)``, ``gcd(A(n), B(n + h)) = 1`` for every
    nonnegative integer ``h``."""
    p, q = _UPoly.of(f, n), _UPoly.of(g, n)
    a, A = p.lc(), p.monic()
    b, B = q.lc(), q.monic()
    C = _UPoly([ONE])
    Z = _c(a / b)
    h = Dummy('h')
    shifted = B.shift(h)
    R = _resultant(A, shifted)
    for i in _nonnegative_integer_roots(R, h):
        d = A.gcd(B.shift(Integer(i)))
        A = A.quo(d)
        B = B.quo(d.shift(Integer(-i)))
        for j in range(1, i + 1):
            C = C * d.shift(Integer(-j))
    return A.scale(Z), B, C


# PORTED FROM SYMPY: sympy/simplify/simplify.py:hypersimp
def hypersimp(f: ExprLike, k: Symbol) -> Optional[Expr]:
    """The ratio ``f(k + 1)/f(k)`` simplified when it is a rational function
    of ``k`` (``f`` a hypergeometric term), ``None`` otherwise."""
    from sympy_extras.core.function import expand_func
    from sympy_extras.functions.elementary.piecewise import Piecewise, piecewise_fold
    from sympy_extras.functions.special.gamma_functions import Gamma
    from sympy_extras.simplify.powsimp import powsimp
    from sympy_extras.simplify.simplify import simplify
    e = to_expr(f)
    shifted = e.subs(k, k + 1)
    assert isinstance(shifted, Expr)
    g = shifted / e
    rewritten = g.rewrite(Gamma)
    assert isinstance(rewritten, Expr)
    g = rewritten
    if g.has(Piecewise):
        g = piecewise_fold(g)
        assert isinstance(g, Piecewise)
        g = g.args[-1].expr
    g = expand_func(g)
    g = powsimp(g, deep=True, combine='exp')
    if g.is_rational_function(k):
        return simplify(g, ratio=float('inf'))
    return None


def _solve_linear(equations: list[Expr], unknowns: list[Dummy]) -> Optional[dict[Symbol, Expr]]:
    """SymPy's ``solve`` of a linear system: a solution of the consistent
    system (free unknowns left out), ``None`` when inconsistent."""
    rows: list[list[Expr]] = []
    for eq in equations:
        row = [_c(eq.diff(u)) for u in unknowns]
        constant = eq
        for u in unknowns:
            subbed = constant.subs(u, ZERO)
            assert isinstance(subbed, Expr)
            constant = subbed
        rows.append(row + [_c(-constant)])
    pivots: list[int] = []
    r = 0
    width = len(unknowns)
    for col in range(width):
        pivot = next((i for i in range(r, len(rows)) if rows[i][col] != 0), None)
        if pivot is None:
            continue
        rows[r], rows[pivot] = rows[pivot], rows[r]
        lead = rows[r][col]
        rows[r] = [_c(v / lead) for v in rows[r]]
        for i in range(len(rows)):
            if i != r and rows[i][col] != 0:
                factor = rows[i][col]
                rows[i] = [_c(v - factor * w) for v, w in zip(rows[i], rows[r])]
        pivots.append(col)
        r += 1
    for row in rows[r:]:
        if row[-1] != 0:
            return None
    solution: dict[Symbol, Expr] = {}
    for i, col in enumerate(pivots):
        rest = rows[i][-1]
        for j in range(width):
            if j != col and j not in pivots and rows[i][j] != 0:
                rest = rest - rows[i][j] * unknowns[j]
        solution[unknowns[col]] = _c(rest)
    return solution


# PORTED FROM SYMPY: sympy/concrete/gosper.py:gosper_term
def gosper_term(f: ExprLike, n: Symbol) -> Optional[Expr]:
    """The rational function ``g`` for which ``g*f`` is an antidifference
    of ``f`` (``g(n + 1)*f(n + 1) - g(n)*f(n) = f(n)``); ``None`` when ``f``
    is not Gosper-summable."""
    r = hypersimp(f, n)
    if r is None:
        return None
    p, q = r.as_numer_denom()
    A, B, C = gosper_normal(p, q, n)
    B = B.shift(Integer(-1))
    N, M, K = A.degree(), B.degree(), C.degree()
    D: set[Expr]
    if N != M or A.lc() != B.lc():
        D = {Integer(K - max(N, M))}
    elif not N:
        D = {Integer(K - N + 1), ZERO}
    else:
        D = {Integer(K - N + 1), _c((B.nth(N - 1) - A.nth(N - 1)) / A.lc())}
    candidates = [d.p for d in D if isinstance(d, Integer) and d.p >= 0]
    if not candidates:
        return None
    d = max(candidates)
    coeffs = [Dummy('c%d' % k) for k in range(d + 1)]
    # SymPy's Poly(coeffs, n): the leading coefficient first
    x = _UPoly(list(reversed(coeffs)))
    H = A * x.shift(ONE) - B * x - C
    solution = _solve_linear(H.c, coeffs)
    if solution is None:
        return None
    xe = x.as_expr(n)
    subbed = xe.subs(solution)
    assert isinstance(subbed, Expr)
    xe = subbed
    for coeff in coeffs:
        if coeff not in solution:
            zeroed = xe.subs(coeff, ZERO)
            assert isinstance(zeroed, Expr)
            xe = zeroed
    if xe.is_zero is True or xe == 0:
        return None
    return B.as_expr(n) * xe / C.as_expr(n)


# PORTED FROM SYMPY: sympy/concrete/gosper.py:gosper_sum
def gosper_sum(f: ExprLike, k: Union[Symbol, tuple[Symbol, Expr, Expr]]) -> Optional[Expr]:
    """The sum of the hypergeometric term ``f`` by Gosper's algorithm:
    indefinite for a symbol, definite over ``(k, a, b)``; ``None`` when
    ``f`` is not Gosper-summable."""
    from sympy_extras.polys.polytools import factor
    e = to_expr(f)
    if isinstance(k, Symbol):
        g = gosper_term(e, k)
        if g is None:
            return None
        return factor(e * g)
    var, a, b = k
    g = gosper_term(e, var)
    if g is None:
        return None
    upper = (e * (g + 1)).subs(var, b)
    lower = (e * g).subs(var, a)
    assert isinstance(upper, Expr) and isinstance(lower, Expr)
    result: Optional[Expr] = upper - lower
    if isinstance(result, NaN):
        from sympy_extras.core.limits import limit
        try:
            result = limit(e * (g + 1), var, b) - limit(e * g, var, a)
        except NotImplementedError:
            result = None
    return factor(result) if result is not None else None
