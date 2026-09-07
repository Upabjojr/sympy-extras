r"""$\Pi\Sigma$-fields and Karr's algorithm for first order difference equations.

A $\Pi\Sigma$-field (Karr, 1981) is a tower of fields

.. math::

    \mathbb{C} \subset \mathbb{C}(k) \subset \mathbb{C}(k)(t_1) \subset
    \cdots \subset \mathbb{C}(k)(t_1, \ldots, t_m)

with an automorphism $\sigma$ (the shift $k \to k + 1$) such that
$\sigma(k) = k + 1$ and every generator is either a *$\Pi$-extension*,
$\sigma(t) = \alpha t$ with $\alpha \neq 0$ in the field below (products
like $k!$, $2^k$, $\binom{2k}{k}$), or a *$\Sigma$-extension*,
$\sigma(t) = t + \beta$ with $\beta$ in the field below (sums like the
harmonic numbers $H_k$, $H_k^{(2)}$ or nested sums). The constants
$\mathbb{C}$ are the rational numbers, possibly with parameters (symbols
different from the summation index) adjoined.

Indefinite summation of $f$ in such a field means finding $g$ with
$\sigma(g) - g = f$, so that $\sum_{k=a}^{b} f(k) = g(b + 1) - g(a)$.
Karr's algorithm solves, more generally, the *parameterized first order
equation*

.. math::

    \sigma(g) - a\, g = c_1 f_1 + \cdots + c_r f_r

for $g$ in the field and constants $c_i$, by recursion over the tower:
in the top generator $t$ the unknown $g$ is a polynomial ($\Sigma$-case) or a
Laurent polynomial ($\Pi$-case) in $t$ whose degree is bounded, and the
coefficients are found from the top down by solving equations of the same
shape in the field below; at the bottom, in $\mathbb{C}(k)$, the rational
solutions are found with Abramov's universal denominator and a degree
bound for the polynomial part. The linear systems are solved over the
constants, so the constants $c_i$ come out together with $g$.

This implementation has one restriction with respect to the general
theory: in a $\Sigma$-extension $t$ only right hand sides which are
polynomial in $t$ are handled, and in a $\Pi$-extension only Laurent
polynomials in $t$ (denominators like $1/(2^k + 1)$ or $1/H_k$ are not).
For $\Sigma$-extensions this is no restriction on the solutions: by a
theorem of Karr the solution $g$ is then a polynomial in $t$ as well, so
an answer ``no solution`` is a proof that no closed form exists in the
field.

References
==========

.. [1] M. Karr, Summation in finite terms, J. ACM 28 (1981) 305-350.
.. [2] M. Karr, Theory of summation in finite terms, J. Symbolic
       Computation 1 (1985) 303-315.
.. [3] C. Schneider, Symbolic summation assists combinatorics, Sem. Lothar.
       Combin. 56 (2007), B56b.
.. [4] S. A. Abramov, Rational solutions of linear difference and
       q-difference equations with polynomial coefficients, ISSAC 1995.
"""
from __future__ import annotations

from math import comb
from typing import Callable, Iterable, Optional, Sequence, Union

from sympy.core.expr import Expr
from sympy.core.symbol import Dummy, Symbol
from sympy.core.sympify import sympify
from sympy.polys.domains import QQ
from sympy.polys.domains.domain import Domain
from sympy.polys.fields import field as _field, FracElement, FracField
from sympy.polys.rings import PolyElement
from sympy.polys.matrices import DomainMatrix
from sympy.core.exprtools import factor_terms
from sympy.polys.polytools import Poly, factor_list

from sympy_extras._typing import DomainElement

__all__ = ['PiSigmaField', 'Extension', 'Solution', 'Solutions']

#: an element of the constant field
Constant = DomainElement
#: a solution (c_1, ..., c_r, g) of the parameterized first order equation
Solution = tuple[list[Constant], FracElement]
#: a basis of solutions, or None when the equation is out of scope
Solutions = Optional[list[Solution]]
#: the right hand sides of a step of the coefficientwise recursion, as a
#: function of the current expression of the constants and coefficients
Rhs = Callable[[list[list[Constant]], dict[int, list[FracElement]], int], list[FracElement]]


class Extension:
    """A generator of a $\\Pi\\Sigma$-field.

    Attributes
    ==========

    kind : ``'sigma'`` or ``'pi'``
    symbol : Dummy
        The generator as a symbol.
    value : Expr
        The defining expression of $\\alpha$ ($\\Pi$) or $\\beta$ ($\\Sigma$)
        in terms of the summation index and the previous generators.
    expr : Expr
        The sequence the generator stands for, as a SymPy expression in the
        summation index (for instance ``harmonic(k)``).
    """

    __slots__ = ('kind', 'symbol', 'value', 'expr')

    def __init__(self, kind: str, symbol: Dummy, value: Expr, expr: Expr) -> None:
        self.kind = kind
        self.symbol = symbol
        self.value = value
        self.expr = expr

    def __repr__(self) -> str:
        return "Extension(%s, %s, %s)" % (self.kind, self.expr, self.value)


class PiSigmaField:
    r"""A $\Pi\Sigma$-field $\mathbb{C}(k)(t_1, \ldots, t_m)$ with its shift.

    Parameters
    ==========

    k : Symbol
        The summation index, with $\sigma(k) = k + 1$.
    params : iterable of Symbol
        Parameters adjoined to the constants.

    Elements are elements of the SymPy fraction field in ``k`` and the
    generator symbols over ``QQ`` (with the parameters). Extensions are added
    with :meth:`add_sigma` and :meth:`add_pi`, and the parameterized first
    order equation is solved with :meth:`solve`; :meth:`telescope` is the
    special case $\sigma(g) - g = f$.

    Examples
    ========

    >>> from sympy import harmonic, factorial
    >>> from sympy.abc import k
    >>> from sympy_extras.concrete.pisigma import PiSigmaField
    >>> F = PiSigmaField(k)
    >>> h = F.add_sigma(1/(k + 1), harmonic(k))
    >>> g = F.telescope(F.from_expr(h))
    >>> F.to_expr(g)
    k*(harmonic(k) - 1)
    >>> F.sigma(g) - g == F.from_expr(h)
    True
    >>> t = F.add_pi(k + 1, factorial(k))
    >>> F.to_expr(F.telescope(F.from_expr(k*t)))
    factorial(k)
    >>> F.telescope(F.from_expr(t/k)) is None
    True
    """

    def __init__(self, k: Symbol, params: Iterable[Symbol] = ()) -> None:
        k_ = sympify(k)
        if not isinstance(k_, Symbol):
            raise TypeError("the summation index must be a symbol")
        self.k: Symbol = k_
        self.params = tuple(sympify(p) for p in params)
        self.C: Domain = QQ.frac_field(*self.params) if self.params else QQ
        self.extensions: list[Extension] = []
        self.field: FracField
        self.gens: tuple[FracElement, ...]
        self.symbols: tuple[Symbol, ...]
        self._sigma_gens: list[FracElement]
        self.zero: FracElement
        self.one: FracElement
        self._sigma_cache: dict[tuple[int, int], FracElement]
        self._sigma_inv_gens: Optional[list[FracElement]]
        self._sigma_inv_cache: dict[tuple[int, int], FracElement]
        #: the atoms represented in the field, filled by build_pisigma_field
        self.known_atoms: list[Expr] = []
        self._rebuild()

    # ------------------------------------------------------------------
    # the field

    def _rebuild(self) -> None:
        symbols: list[Symbol] = [self.k] + [e.symbol for e in self.extensions]
        self.field, *gens = _field(symbols, self.C)
        self.gens = tuple(gens)
        self.zero = getattr(self.field, 'zero')
        self.one = getattr(self.field, 'one')
        self.symbols = tuple(symbols)
        self._sigma_gens = [self.gens[0] + 1]
        for i, e in enumerate(self.extensions, start=1):
            value = self.from_expr(e.value)
            if e.kind == 'pi':
                self._sigma_gens.append(value*self.gens[i])
            else:
                self._sigma_gens.append(self.gens[i] + value)
        self._sigma_cache = {}
        self._sigma_inv_gens = None
        self._sigma_inv_cache = {}

    @property
    def level(self) -> int:
        """The number of extensions."""
        return len(self.extensions)

    def from_expr(self, expr: Union[Expr, int]) -> FracElement:
        """Convert an expression in the index, the parameters and the
        generator symbols to an element of the field."""
        return self.field.from_expr(sympify(expr))

    def to_expr(self, f: FracElement, substitute: bool = True) -> Expr:
        """Convert an element to an expression; with ``substitute`` the
        generators are replaced by the sequences they stand for."""
        expr = f.as_expr()
        if substitute:
            if self.extensions:
                expr = expr.xreplace({e.symbol: e.expr for e in self.extensions})
            expr = factor_terms(expr)
        return expr

    def add_sigma(self, beta: Union[Expr, int], expr: Union[Expr, int], name: Optional[str] = None) -> Dummy:
        r"""Adjoin a $\Sigma$-extension $t$ with $\sigma(t) = t + \beta$.

        ``beta`` is an expression in the index and the existing generators
        and ``expr`` the sequence $t$ stands for. Returns the symbol of the
        new generator. No check is made that the extension is proper.
        """
        return self._add('sigma', beta, expr, name)

    def add_pi(self, alpha: Union[Expr, int], expr: Union[Expr, int], name: Optional[str] = None) -> Dummy:
        r"""Adjoin a $\Pi$-extension $t$ with $\sigma(t) = \alpha t$."""
        return self._add('pi', alpha, expr, name)

    def _add(self, kind: str, value: Union[Expr, int], expr: Union[Expr, int], name: Optional[str]) -> Dummy:
        value = sympify(value)
        if kind == 'pi' and value == 0:
            raise ValueError("alpha must be nonzero")
        symbol = Dummy(name or 't%d' % (len(self.extensions) + 1))
        self.extensions.append(Extension(kind, symbol, value, sympify(expr)))
        self._rebuild()
        return symbol

    # ------------------------------------------------------------------
    # the shift

    def sigma(self, f: FracElement, power: int = 1) -> FracElement:
        """The shift $\\sigma^{j}(f)$ for an integer ``j = power`` (negative
        powers are the inverse shift)."""
        if power == 0:
            return f
        if power < 0:
            images, cache = self._inverse_gens(), self._sigma_inv_cache
        else:
            images, cache = self._sigma_gens, self._sigma_cache
        for _ in range(abs(power)):
            f = self._sigma_poly(f.numer, images, cache)/self._sigma_poly(f.denom, images, cache)
        return f

    def _inverse_gens(self) -> list[FracElement]:
        """Images of the generators under the inverse shift, computed on
        first use: $\\sigma^{-1}(k) = k - 1$, $\\sigma^{-1}(t) = t/\\sigma^{-1}(\\alpha)$
        or $t - \\sigma^{-1}(\\beta)$."""
        if self._sigma_inv_gens is None:
            self._sigma_inv_gens = [self.gens[0] - 1]
            for i, e in enumerate(self.extensions, start=1):
                value = self.sigma(self.from_expr(e.value), -1)
                if e.kind == 'pi':
                    self._sigma_inv_gens.append(self.gens[i]/value)
                else:
                    self._sigma_inv_gens.append(self.gens[i] - value)
        return self._sigma_inv_gens

    def _sigma_poly(self, p: PolyElement, images: Optional[list[FracElement]] = None,
                    cache: Optional[dict[tuple[int, int], FracElement]] = None) -> FracElement:
        if images is None or cache is None:
            images, cache = self._sigma_gens, self._sigma_cache
        result = self.zero
        for monom, coeff in p.terms():
            term = self.field(coeff)
            for i, e in enumerate(monom):
                if e:
                    key = (i, e)
                    value = cache.get(key)
                    if value is None:
                        value = images[i]**e
                        cache[key] = value
                    term *= value
            result += term
        return result

    def shift_poly(self, p: PolyElement, j: int) -> PolyElement:
        """$\\sigma^j$ of a polynomial in ``k`` alone (a ring element),
        for any integer ``j``."""
        x = self.field.ring.gens[0]
        return p.compose(x, x + j)

    # ------------------------------------------------------------------
    # levels and decompositions

    def level_of(self, f: FracElement) -> int:
        """The index of the highest generator occurring in ``f``: ``-1`` for
        a constant, ``0`` for an element of $\\mathbb{C}(k)$, ``i`` for an
        element involving $t_i$."""
        level = -1
        for p in (f.numer, f.denom):
            for monom in p.monoms():
                for i in range(len(monom) - 1, level, -1):
                    if monom[i]:
                        level = max(level, i)
                        break
        return level

    def is_constant(self, f: FracElement) -> bool:
        return self.level_of(f) == -1

    def to_constant(self, f: FracElement) -> Constant:
        """The element of the constant field equal to the constant ``f``."""
        if not self.is_constant(f):
            raise ValueError("%s is not a constant" % (f,))
        return self.C.convert(f.numer.LC)/self.C.convert(f.denom.LC)

    def from_constant(self, c: Constant) -> FracElement:
        return self.field(c)

    def as_poly_in(self, f: FracElement, i: int, laurent: bool = False) -> Optional[dict[int, FracElement]]:
        """Decompose ``f`` as a polynomial in the generator of index ``i``
        with coefficients in the field without it: a dict mapping exponents
        to coefficients. With ``laurent`` the denominator may be a power of
        the generator; otherwise it must not involve it. Returns ``None``
        if ``f`` has no such decomposition."""
        ring = self.field.ring
        num, den = f.numer, f.denom
        shift = 0
        den_exps = {m[i] for m in den.monoms()}
        if len(den_exps) != 1:
            return None
        [d] = den_exps
        if d:
            if not laurent:
                return None
            shift = d
            den = ring.from_dict({m[:i] + (0,) + m[i + 1:]: c for m, c in den.terms()})
        result: dict[int, PolyElement] = {}
        for monom, coeff in num.terms():
            e = monom[i] - shift
            rest = monom[:i] + (0,) + monom[i + 1:]
            result[e] = result.get(e, ring.zero) + ring.from_dict({rest: coeff})
        den = self.field(den)
        return {e: self.field(p)/den for e, p in result.items()}

    # ------------------------------------------------------------------
    # linear algebra over the constants

    def _nullspace(self, rows: Sequence[Sequence[Constant]], ncols: int) -> list[list[Constant]]:
        """Basis of the null space of a matrix over the constants given as a
        list of rows (lists of constants)."""
        if not rows:
            return [[self.C.one if i == j else self.C.zero for i in range(ncols)]
                    for j in range(ncols)]
        M = DomainMatrix([[self.C.convert(c) for c in row] for row in rows],
                         (len(rows), ncols), self.C)
        N = M.nullspace().to_Matrix()
        return [[self.C.from_sympy(N[r, c]) for c in range(N.cols)] for r in range(N.rows)]

    # ------------------------------------------------------------------
    # the solver

    def solve(self, a: FracElement, fs: Sequence[FracElement], level: Optional[int] = None) -> Solutions:
        r"""Solve $\sigma(g) - a\,g = c_1 f_1 + \cdots + c_r f_r$.

        Returns a basis of the space of solutions $(c_1, \ldots, c_r, g)$
        over the constants, as a list of pairs ``(c, g)`` with ``c`` a list
        of constants and ``g`` an element of the field, or ``None`` if the
        equation is outside the scope of the implementation (a right hand
        side with a $\Sigma$-generator in a denominator, or ``a`` involving
        the top generator). An empty list means that there is no solution
        other than $g = 0$, $c = 0$.

        ``level`` is the field in which ``g`` is searched, by default the
        whole field: the solution may need generators which do not occur
        in the $f_i$, for instance $\sum_k H_k/k = (H_k^2 + H_k^{(2)})/2$.
        """
        if level is None:
            level = self.level
        if level == -1:
            return self._solve_constants(a, fs)
        if level == 0:
            return self._solve_rational(a, fs)
        ext = self.extensions[level - 1]
        if self.level_of(a) >= level:
            return None
        if ext.kind == 'sigma':
            return self._solve_sigma(level, a, fs)
        return self._solve_pi(level, a, fs)

    def telescope(self, f: FracElement) -> Optional[FracElement]:
        r"""A solution $g$ of $\sigma(g) - g = f$, or ``None`` if there is
        none in the field (or it could not be decided, see :meth:`solve`)."""
        sols = self.solve(self.one, [f])
        if not sols:
            return None
        for c, g in sols:
            if c[0]:
                return g/self.field(c[0])
        return None

    # constants: sigma is the identity

    def _solve_constants(self, a_: FracElement, fs_: Sequence[FracElement]) -> Solutions:
        a = self.to_constant(a_)
        fs = [self.to_constant(f) for f in fs_]
        s = len(fs)
        one, zero = self.C.one, self.C.zero
        if a != one:
            return [([one if i == l else zero for i in range(s)],
                     self.field(fs[l]/(one - a))) for l in range(s)]
        sols: list[Solution] = [(v, self.zero) for v in self._nullspace([fs], s)] if s else []
        sols.append(([zero]*s, self.one))
        return sols

    # the rational level C(k): Abramov's universal denominator and a
    # degree bound for the polynomial part

    def _integer_roots(self, p: Poly, h: Symbol) -> set[int]:
        """Non-negative integer roots of a polynomial in ``h`` (a Poly)."""
        roots: set[int] = set()
        if p.is_zero:
            return roots
        for factor, _ in factor_list(p.as_expr(), h)[1]:
            factor = Poly(factor, h)
            if factor.degree() == 1:
                a1, a0 = factor.all_coeffs()
                r = -a0/a1
                if r.is_Integer and r >= 0:
                    roots.add(int(r))
        return roots

    def _dispersion(self, A: PolyElement, B: PolyElement) -> int:
        """The largest ``h >= 0`` with ``gcd(A(k), B(k + h)) != 1`` for
        univariate polynomials ``A``, ``B`` (ring elements), or ``-1``."""
        k, h = self.k, Dummy('h')
        pa = Poly(A.as_expr(), k)
        pb = Poly(B.as_expr().subs(k, k + h), k)
        if pa.degree() <= 0 or pb.degree() <= 0:
            return -1
        res = Poly(pa.resultant(pb), h)
        roots = self._integer_roots(res, h)
        return max(roots) if roots else -1

    def _universal_denominator(self, p1: PolyElement, p0: PolyElement) -> PolyElement:
        """Abramov's universal denominator of the rational solutions of
        ``p1(k) y(k + 1) + p0(k) y(k) = q(k)`` with polynomial ``q``."""
        ring = self.field.ring
        A = self.shift_poly(p1, -1)
        B = p0
        U = ring.one
        N = self._dispersion(A, B)
        for i in range(N, -1, -1):
            d = A.gcd(self.shift_poly(B, i))
            if d.degree(0) <= 0:
                continue
            A = A.quo(d)
            B = B.quo(self.shift_poly(d, -i))
            for j in range(i + 1):
                U *= self.shift_poly(d, -j)
        return U

    def _coeff_list(self, p: PolyElement) -> list[Constant]:
        """Coefficients of a univariate polynomial in ``k`` (ring element)
        as a list indexed by the degree, as constants."""
        coeffs: dict[int, Constant] = {}
        for monom, c in p.terms():
            coeffs[monom[0]] = self.C.convert(c)
        d = max(coeffs) if coeffs else -1
        return [coeffs.get(i, self.C.zero) for i in range(d + 1)]

    def _solve_rational(self, a: FracElement, fs: Sequence[FracElement]) -> Solutions:
        ring = self.field.ring
        x = ring.gens[0]
        A, B = a.numer, a.denom
        # B sigma(y) - A y = B (u_1 f_1 + ... + u_s f_s), then clear the
        # denominators of the right hand side
        bfs = [self.field(B)*f for f in fs]
        W = ring.one
        for f in bfs:
            W = W.lcm(f.denom)
        S: list[PolyElement] = []
        for f in bfs:
            wf = self.field(W)*f
            if wf.denom.degree(0) > 0:
                raise RuntimeError("denominator not cleared: %s" % (wf,))
            # the denominator may still be a constant
            S.append(wf.numer.quo_ground(wf.denom.LC))
        p1, p0 = W*B, -W*A
        U = self._universal_denominator(p1, p0)
        sU = self.shift_poly(U, 1)
        P1, P0 = p1*U, p0*sU
        Q = [U*sU*q for q in S]
        # degree bound for the polynomial part
        c1, c0 = self._coeff_list(P1), self._coeff_list(P0)
        d1, d0 = len(c1) - 1, len(c0) - 1
        m = max(d1, d0)
        dq = max([len(self._coeff_list(q)) - 1 for q in Q] + [-1])
        D = max(dq - m, 0)
        if d1 == d0 and c1[-1] + c0[-1] == self.C.zero:
            D = max(dq - m + 1, 0)
            lower = (c1[m - 1] if m >= 1 else self.C.zero) + (c0[m - 1] if m >= 1 else self.C.zero)
            root = -lower/c1[-1]
            rs = self.C.to_sympy(root)
            if rs.is_Integer and rs >= 0:
                D = max(D, int(rs))
        # linear system for the coefficients y_0..y_D and the constants
        columns: list[list[Constant]] = []
        for j in range(D + 1):
            columns.append(self._coeff_list(P1*(x + 1)**j + P0*x**j))
        for q in Q:
            columns.append([-c for c in self._coeff_list(q)])
        nrows = max([len(col) for col in columns] + [0])
        ncols = len(columns)
        rows: list[list[Constant]] = []
        for r in range(nrows):
            rows.append([col[r] if r < len(col) else self.C.zero for col in columns])
        sols: list[Solution] = []
        for v in self._nullspace(rows, ncols):
            p = ring.zero
            for j in range(D + 1):
                if v[j]:
                    p += ring.from_dict({(j,) + (0,)*(ring.ngens - 1): v[j]})
            g = self.field(p)/self.field(U)
            sols.append((list(v[D + 1:]), g))
        return sols

    # Sigma and Pi extensions: coefficientwise recursion with a running set
    # of unknown constants

    def _is_sigma_ratio(self, a: FracElement, level: int) -> bool:
        """Whether $\\sigma(h) = a h$ has a nonzero solution in the field of
        the given level (``a == 1`` included)."""
        if a == self.one:
            return True
        sols = self.solve(a, [], level=level)
        if not sols:
            return False
        return any(g != self.zero for _, g in sols)

    def _combination(self, coefficients: Sequence[Constant], elements: Sequence[FracElement]) -> FracElement:
        """The linear combination of ``elements`` with constant ``coefficients``."""
        result = self.zero
        for c, e in zip(coefficients, elements):
            if c:
                result += self.field(c)*e
        return result

    def _solve_steps(self, level: int, steps: Sequence[tuple[int, FracElement, Rhs]], r: int) -> Solutions:
        """Common driver of the $\\Sigma$ and $\\Pi$ cases.

        ``steps`` is a list of ``(j, a_j, rhs)`` where ``rhs(cmap, G)``
        returns the right hand sides for the current unknowns: the
        coefficient $g_j$ satisfies $\\sigma(g_j) - a_j g_j = \\sum_l u_l R_l$.
        ``cmap`` is the matrix expressing the original constants in terms
        of the current unknowns and ``G`` maps the exponents already
        processed to the coefficients of $g_j$ per unknown.
        """
        one, zero = self.C.one, self.C.zero
        s = r
        cmap: list[list[Constant]] = [[one if i == l else zero for l in range(s)] for i in range(r)]
        G: dict[int, list[FracElement]] = {}
        for j, a_j, rhs in steps:
            R = rhs(cmap, G, s)
            sols = self.solve(a_j, R, level=level - 1)
            if sols is None:
                return None
            if not sols:
                return []
            p = len(sols)
            V = [v for v, _ in sols]
            G[j] = [g for _, g in sols]
            for jj in G:
                if jj != j:
                    G[jj] = [self._combination([V[m][l] for l in range(s)], G[jj]) for m in range(p)]
            cmap = [[sum((cmap[i][l]*V[m][l] for l in range(s)), zero) for m in range(p)]
                    for i in range(r)]
            s = p
        t = self.gens[level]
        result: list[Solution] = []
        for m in range(s):
            g = self.zero
            for j, coeffs in G.items():
                g += coeffs[m]*t**j
            result.append(([cmap[i][m] for i in range(r)], g))
        return result

    def _solve_sigma(self, level: int, a: FracElement, fs: Sequence[FracElement]) -> Solutions:
        beta = self.from_expr(self.extensions[level - 1].value)
        polys: list[dict[int, FracElement]] = []
        for f in fs:
            p = self.as_poly_in(f, level)
            if p is None:
                return None
            polys.append(p)
        maxdeg = max([max(p) for p in polys if p] + [0])
        D = maxdeg + (1 if self._is_sigma_ratio(a, level - 1) else 0)
        r = len(fs)
        betas: dict[int, FracElement] = {}

        def make_rhs(j: int) -> Rhs:
            def rhs(cmap: list[list[Constant]], G: dict[int, list[FracElement]], s: int) -> list[FracElement]:
                R: list[FracElement] = []
                for l in range(s):
                    value = self.zero
                    for i in range(r):
                        fij = polys[i].get(j)
                        if fij is not None and cmap[i][l]:
                            value += self.field(cmap[i][l])*fij
                    for jj, coeffs in G.items():
                        if jj > j:
                            if jj - j not in betas:
                                betas[jj - j] = beta**(jj - j)
                            value -= comb(jj, j)*betas[jj - j]*self.sigma(coeffs[l])
                    R.append(value)
                return R
            return rhs

        steps: list[tuple[int, FracElement, Rhs]] = [(j, a, make_rhs(j)) for j in range(D, -1, -1)]
        return self._solve_steps(level, steps, r)

    def _pi_exponents(self, a: FracElement, alpha: FracElement) -> set[int]:
        r"""Candidate exponents ``j`` such that $\sigma(c)\, \alpha^j = a\, c$
        may have a solution $c$ in the field below: for elements of
        $\mathbb{C}(k)$ the leading coefficients and the degrees of ``a``
        and ``alpha`` must match, which fixes ``j`` in most cases; otherwise
        a few small exponents are tried."""
        if a == self.one:
            return set()
        if self.level_of(a) > 0 or self.level_of(alpha) > 0:
            return set(range(-3, 4))

        def lc_and_degree(f: FracElement) -> tuple[Constant, int]:
            num, den = f.numer, f.denom
            return (self.C.convert(num.LC)/self.C.convert(den.LC),
                    num.degree(0) - den.degree(0))

        la, da = lc_and_degree(a)
        lal, dal = lc_and_degree(alpha)
        if dal != 0:
            if da % dal == 0:
                return {da//dal}
            return set()
        if lal not in (self.C.one, -self.C.one):
            power = self.C.one
            for j in range(0, 40):
                if power == la:
                    return {j}
                power *= lal
            power = self.C.one
            for j in range(0, 40):
                if power == la:
                    return {-j}
                power /= lal
            return set()
        return set(range(-3, 4))

    def _solve_pi(self, level: int, a: FracElement, fs: Sequence[FracElement]) -> Solutions:
        alpha = self.from_expr(self.extensions[level - 1].value)
        polys: list[dict[int, FracElement]] = []
        for f in fs:
            p = self.as_poly_in(f, level, laurent=True)
            if p is None:
                return None
            polys.append(p)
        exponent_set: set[int] = set().union(*[set(p) for p in polys]) | {0}
        exponent_set |= self._pi_exponents(a, alpha)
        exponents = sorted(exponent_set, reverse=True)
        r = len(fs)

        def make_rhs(j: int, alpha_j: FracElement) -> Rhs:
            def rhs(cmap: list[list[Constant]], G: dict[int, list[FracElement]], s: int) -> list[FracElement]:
                R: list[FracElement] = []
                for l in range(s):
                    value = self.zero
                    for i in range(r):
                        fij = polys[i].get(j)
                        if fij is not None and cmap[i][l]:
                            value += self.field(cmap[i][l])*fij
                    R.append(value/alpha_j)
                return R
            return rhs

        steps: list[tuple[int, FracElement, Rhs]] = []
        for j in exponents:
            alpha_j = alpha**j
            steps.append((j, a/alpha_j, make_rhs(j, alpha_j)))
        return self._solve_steps(level, steps, r)
