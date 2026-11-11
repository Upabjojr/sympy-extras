"""Linear differential-algebraic equations with constant coefficients,
solved through the core-nilpotent decomposition of the pencil.

A system `A y' + B y = f(x)` with constant square matrices `A`, `B` is a
*differential-algebraic* equation when `A` is singular: some of the
equations contain no derivative. The pencil `\\lambda A + B` is *regular*
when `\\det(\\lambda A + B)` is not identically zero; then for a scalar
`\\lambda` with `\\det(\\lambda A + B) \\ne 0` the system is equivalent to

.. math::

    \\hat A y' + (I - \\lambda \\hat A) y = \\hat f, \\qquad
    \\hat A = (\\lambda A + B)^{-1} A, \\quad \\hat f = (\\lambda A + B)^{-1} f.

The **core-nilpotent decomposition** of `\\hat A` (its Drazin
decomposition) is a change of basis `T` with

.. math::

    T^{-1} \\hat A T = \\begin{pmatrix} C & 0 \\\\ 0 & N \\end{pmatrix},

`C` invertible and `N` nilpotent of index `k` (the *index* of the
differential-algebraic equation); the columns of `T` span the range and
the null space of `\\hat A^{k}`. In the coordinates `y = T z` the system
splits into

* a regular part `C z_1' + (I - \\lambda C) z_1 = g_1`, an ordinary linear
  system with constant coefficients solved by the matrix exponential,
  whose solutions carry the free constants;
* a nilpotent part `N z_2' + (I - \\lambda N) z_2 = g_2`, which has the
  unique solution
  `z_2 = \\sum_{i=0}^{k-1} (-Q N)^i Q\\, g_2^{(i)}` with
  `Q = (I - \\lambda N)^{-1}`, obtained by substituting the equation into
  itself `k` times (the nilpotent part isolates the purely algebraic
  constraints, which involve derivatives of the right-hand side up to
  order `k - 1`).

The matrix exponential `e^{M x}` is computed by Putzer's algorithm: with
the eigenvalues `\\lambda_1, \\dots, \\lambda_n` of `M` (repeated by their
multiplicities), `e^{M x} = \\sum_{k=0}^{n-1} r_{k+1}(x) P_k` where
`P_0 = I`, `P_k = \\prod_{j=1}^{k} (M - \\lambda_j I)`, `r_1 = e^{\\lambda_1 x}`
and `r_{k+1}(x) = \\int_0^x e^{\\lambda_{k+1}(x - s)} r_k(s)\\, ds`, sums of
polynomials times exponentials integrated in closed form. For a real
matrix the complex exponentials are then written with sines and cosines.

SymPy's ``dsolve`` handles systems of ordinary differential equations
only (`A` invertible); a singular `A` is rejected.

The module imports no SymPy: the matrices are
:class:`~sympy_extras.matrices.dense.ExprMatrix` of core expressions, the
linear algebra (reduced row echelon forms, ranks, null spaces, inverses,
determinants) is exact Gaussian elimination with the native ``cancel``,
the eigenvalues are the native polynomial
:func:`~sympy_extras.polys.polyroots.roots` and the results are put in
shape by the native ``simplify``. One step has no native counterpart and
is taken from SymPy at run time through the face
:mod:`sympy_extras.solvers.dae_sympy`: ``integrate`` for the particular
solution of the regular part when the integrand is not a sum of powers of
the variable (``integral_of_core``, an unevaluated integral when SymPy
finds none). The face also keeps the old API on SymPy matrices.

References
==========

.. [Campbell] S. L. Campbell, Singular Systems of Differential Equations,
   Pitman (1980), chapter 2.
.. [Kunkel] P. Kunkel, V. Mehrmann, Differential-Algebraic Equations:
   Analysis and Numerical Solution, EMS (2006), chapter 2 (the
   Weierstrass canonical form).
.. [Putzer] E. J. Putzer, Avoiding the Jordan canonical form in the
   discussion of linear systems with constant coefficients, The American
   Mathematical Monthly 73 (1966).
"""
from __future__ import annotations

from math import factorial
from typing import Final, Optional, Sequence

from sympy_extras.core.add import add
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr
from sympy_extras.core.function import AppliedUndef, Derivative, diff
from sympy_extras.core.mul import mul
from sympy_extras.core.numbers import Integer
from sympy_extras.core.power import pow
from sympy_extras.core.relational import Eq
from sympy_extras.core.sorting import ordered
from sympy_extras.core.symbol import Dummy, Symbol
from sympy_extras.functions.elementary.exponential import exp
from sympy_extras.functions.elementary.trigonometric import Cos
from sympy_extras.matrices.dense import ExprMatrix
from sympy_extras.polys.polyroots import roots
from sympy_extras.polys.polytools import Poly, cancel
from sympy_extras.polys.rationaltools import together
from sympy_extras.simplify.radsimp import fraction
from sympy_extras.simplify.simplify import simplify

from .charpit import _power_antiderivative, _replaced

__all__ = ['core_nilpotent_decomposition', 'dae_index', 'dsolve_dae', 'dae_matrices', 'check_dae', 'DAESolution',
           'matrix_exponential']


# ---------------------------------------------------------------------------
# exact linear algebra on matrices of expressions

def _normal(e: Expr) -> Expr:
    """``e`` as a rational function in lowest terms (``cancel``)."""
    return cancel(e)


def _is_zero(e: Expr) -> bool:
    """Whether ``e`` is zero: in lowest terms for a rational function
    (a symbol is not zero, as in SymPy's matrices), by ``simplify`` for
    the other expressions."""
    c = _normal(e)
    if c == 0:
        return True
    if c.is_rational_function() is True:
        return False
    return simplify(c) == 0


def _block(M: ExprMatrix, rows: range, cols: range) -> ExprMatrix:
    """The submatrix of the rows and columns of the ranges."""
    return ExprMatrix(len(rows), len(cols), [M[i, j] for i in rows for j in cols])


def _hstack(columns: Sequence[ExprMatrix], rows: int) -> ExprMatrix:
    """The columns side by side (``rows`` rows)."""
    return ExprMatrix(rows, len(columns), [c[i, 0] for i in range(rows) for c in columns])


def _vstack(top: ExprMatrix, bottom: ExprMatrix) -> ExprMatrix:
    return ExprMatrix(top.rows + bottom.rows, top.cols if top.rows else bottom.cols,
                      list(top.entries) + list(bottom.entries))


def _product(A: ExprMatrix, B: ExprMatrix) -> ExprMatrix:
    """``A*B`` with the entries in lowest terms."""
    return (A * B).applyfunc(_normal)


def _rref(M: ExprMatrix) -> tuple[ExprMatrix, list[int]]:
    """The reduced row echelon form of ``M`` and its pivot columns."""
    rows = [[M[i, j] for j in range(M.cols)] for i in range(M.rows)]
    pivots: list[int] = []
    r = 0
    for j in range(M.cols):
        if r == len(rows):
            break
        pivot = next((i for i in range(r, len(rows)) if not _is_zero(rows[i][j])), None)
        if pivot is None:
            continue
        rows[r], rows[pivot] = rows[pivot], rows[r]
        inverse = pow(rows[r][j], Integer(-1))
        rows[r] = [_normal(e * inverse) for e in rows[r]]
        for i in range(len(rows)):
            if i != r and not _is_zero(rows[i][j]):
                factor = rows[i][j]
                rows[i] = [_normal(a - factor * b) for a, b in zip(rows[i], rows[r])]
        pivots.append(j)
        r += 1
    return ExprMatrix(M.rows, M.cols, [_cleaned(e) for row in rows for e in row]), pivots


def _cleaned(e: Expr) -> Expr:
    return Integer(0) if _is_zero(e) else e


def _rank(M: ExprMatrix) -> int:
    return len(_rref(M)[1])


def _columnspace(M: ExprMatrix) -> list[ExprMatrix]:
    """The columns of ``M`` at the pivots of its echelon form (a basis
    of its range, as SymPy's ``columnspace``)."""
    _, pivots = _rref(M)
    return [_block(M, range(M.rows), range(j, j + 1)) for j in pivots]


def _nullspace(M: ExprMatrix) -> list[ExprMatrix]:
    """A basis of the null space of ``M``: one vector for each free
    column of the reduced echelon form, ``1`` there (as SymPy's
    ``nullspace``)."""
    R, pivots = _rref(M)
    basis: list[ExprMatrix] = []
    for free in (j for j in range(M.cols) if j not in pivots):
        entries: list[Expr] = [Integer(0)] * M.cols
        entries[free] = Integer(1)
        for i, j in enumerate(pivots):
            entries[j] = -R[i, free]
        basis.append(ExprMatrix.column(entries))
    return basis


def _inverse(M: ExprMatrix) -> ExprMatrix:
    """The inverse of a square matrix by Gauss-Jordan elimination;
    ``ValueError`` when it is singular."""
    n = M.rows
    augmented = ExprMatrix(n, 2 * n, [e for i in range(n) for e in
                                      list(M.row(i)) + [Integer(1) if i == j else Integer(0) for j in range(n)]])
    R, pivots = _rref(augmented)
    if pivots[:n] != list(range(n)):
        raise ValueError("the matrix is not invertible")
    return _block(R, range(n), range(n, 2 * n))


def _determinant(M: ExprMatrix) -> Expr:
    """The determinant by Gaussian elimination, in lowest terms."""
    rows = [[M[i, j] for j in range(M.cols)] for i in range(M.rows)]
    n = len(rows)
    result: Expr = Integer(1)
    for j in range(n):
        pivot = next((i for i in range(j, n) if not _is_zero(rows[i][j])), None)
        if pivot is None:
            return Integer(0)
        if pivot != j:
            rows[j], rows[pivot] = rows[pivot], rows[j]
            result = -result
        result = _normal(result * rows[j][j])
        inverse = pow(rows[j][j], Integer(-1))
        for i in range(j + 1, n):
            factor = _normal(rows[i][j] * inverse)
            rows[i] = [_normal(a - factor * b) for a, b in zip(rows[i], rows[j])]
    return result


def _power(M: ExprMatrix, k: int) -> ExprMatrix:
    result = ExprMatrix.eye(M.rows)
    for _ in range(k):
        result = _product(result, M)
    return result


def _charpoly(M: ExprMatrix, t: Symbol) -> Expr:
    """``det(t I - M)`` by the Faddeev-LeVerrier recurrence."""
    n = M.rows
    coefficients: list[Expr] = [Integer(1)]
    current = ExprMatrix.zeros(n, n)
    for k in range(1, n + 1):
        current = _product(M, current + ExprMatrix.eye(n) * coefficients[-1])
        trace = add(*[current[i, i] for i in range(n)])
        coefficients.append(_normal(-trace / Integer(k)))
    return add(*[mul(c, pow(t, Integer(n - i))) for i, c in enumerate(coefficients)])


def _eigenvalues(M: ExprMatrix) -> list[Expr]:
    """The eigenvalues of ``M`` repeated by their multiplicities, equal
    ones together; ``NotImplementedError`` when the characteristic
    polynomial has roots which are not found."""
    t = Dummy('t')
    numerator, _ = fraction(together(_charpoly(M, t)))
    found = roots(Poly(numerator, t))
    if not isinstance(found, dict) or sum(found.values()) != M.rows:
        raise NotImplementedError("the eigenvalues of %s are not found" % (M,))
    values: list[Expr] = []
    for value in ordered(list(found)):
        assert isinstance(value, Expr)
        values.extend([value] * found[value])
    return values


#: a sum of polynomials times exponentials: the exponent ``mu`` to the
#: coefficients of the powers ``x**j`` (lowest first) of its polynomial
ExponentialPolynomial = dict[Expr, list[Expr]]


def _add_term(target: ExponentialPolynomial, mu: Expr, j: int, c: Expr) -> None:
    coefficients = target.setdefault(mu, [])
    while len(coefficients) <= j:
        coefficients.append(Integer(0))
    coefficients[j] = _normal(coefficients[j] + c)


def _convolution(r: ExponentialPolynomial, lam: Expr) -> ExponentialPolynomial:
    """``int_0^x exp(lam (x - s)) r(s) ds``."""
    result: ExponentialPolynomial = {}
    for mu, coefficients in r.items():
        d = _normal(mu - lam)
        for j, c in enumerate(coefficients):
            if _is_zero(c):
                continue
            if _is_zero(d):
                _add_term(result, lam, j + 1, c / Integer(j + 1))
                continue
            # int_0^x s**j exp(d s) ds
            #   = exp(d x) sum_i (-1)**i j!/(j - i)! x**(j - i)/d**(i + 1) - (-1)**j j!/d**(j + 1)
            for i in range(j + 1):
                coefficient = Integer((-1) ** i * factorial(j) // factorial(j - i))
                _add_term(result, mu, j - i, c * coefficient * pow(d, Integer(-(i + 1))))
            _add_term(result, lam, 0, -c * Integer((-1) ** j * factorial(j)) * pow(d, Integer(-(j + 1))))
    return result


def matrix_exponential(M: ExprMatrix, x: Symbol) -> ExprMatrix:
    """``exp(M*x)`` for a constant square matrix ``M``, by Putzer's
    algorithm (see the module); ``NotImplementedError`` when the
    eigenvalues of ``M`` are not found.

    Examples
    ========

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.matrices.dense import ExprMatrix
    >>> from sympy_extras.solvers.dae import matrix_exponential
    >>> x = Symbol('x')
    >>> matrix_exponential(ExprMatrix.from_rows([[1, 1], [0, 1]]), x)
    ExprMatrix([[exp(x), x*exp(x)], [0, exp(x)]])
    >>> matrix_exponential(ExprMatrix.from_rows([[0, 1], [-1, 0]]), x)[0, 0]
    exp(I*x)/2 + exp(-I*x)/2
    """
    n = M.rows
    if M.cols != n:
        raise ValueError("a square matrix is expected, got shape %s" % (M.shape,))
    if n == 0:
        return ExprMatrix.zeros(0, 0)
    eigenvalues = _eigenvalues(M)
    r: ExponentialPolynomial = {eigenvalues[0]: [Integer(1)]}
    P = ExprMatrix.eye(n)
    # the entries as exponent -> polynomial coefficients
    entries: list[ExponentialPolynomial] = [{} for _ in range(n * n)]
    for k in range(n):
        if k:
            P = _product(P, M - ExprMatrix.eye(n) * eigenvalues[k - 1])
            r = _convolution(r, eigenvalues[k])
        for index, p in enumerate(P.entries):
            if _is_zero(p):
                continue
            for mu, coefficients in r.items():
                for j, c in enumerate(coefficients):
                    _add_term(entries[index], mu, j, c * p)
    result: list[Expr] = []
    for entry in entries:
        terms: list[Expr] = []
        for mu, coefficients in entry.items():
            polynomial = add(*[mul(c, pow(x, Integer(j))) for j, c in enumerate(coefficients)])
            if not _is_zero(polynomial):
                terms.append(mul(polynomial, exp(mul(mu, x))))
        result.append(add(*terms))
    return ExprMatrix(n, n, result)


# ---------------------------------------------------------------------------
# the decomposition and the solver

def _index_of_nilpotency(M: ExprMatrix) -> int:
    """The smallest ``k`` with ``rank(M**k) == rank(M**(k + 1))``."""
    n = M.rows
    power = ExprMatrix.eye(n)
    previous = n
    for k in range(n + 1):
        current = _rank(power)
        if current == previous and k > 0:
            return k - 1
        previous = current
        power = _product(power, M)
    return n


def core_nilpotent_decomposition(M: ExprMatrix) -> tuple[ExprMatrix, ExprMatrix, ExprMatrix, int]:
    """The core-nilpotent (Drazin) decomposition of a square matrix:
    ``T``, ``C``, ``N`` and the index ``k`` with ``T**-1 * M * T`` block
    diagonal with blocks ``C`` (invertible) and ``N`` (nilpotent,
    ``N**k == 0``, ``N**(k-1) != 0``); ``k`` is ``0`` when ``M`` is
    invertible.

    Examples
    ========

    >>> from sympy_extras.matrices.dense import ExprMatrix
    >>> from sympy_extras.solvers.dae import core_nilpotent_decomposition
    >>> T, C, N, k = core_nilpotent_decomposition(ExprMatrix.from_rows([[1, 1], [0, 0]]))
    >>> T, C, N, k
    (ExprMatrix([[1, -1], [0, 1]]), ExprMatrix([[1]]), ExprMatrix([[0]]), 1)
    """
    if M.rows != M.cols:
        raise ValueError("a square matrix is expected, got shape %s" % (M.shape,))
    n = M.rows
    k = _index_of_nilpotency(M)
    power = _power(M, k)
    core_columns = _columnspace(power)
    null_columns = _nullspace(power)
    if len(core_columns) + len(null_columns) != n:
        raise ValueError("the range and the null space of M**k do not span the space")
    T = _hstack(core_columns + null_columns, n)
    r = len(core_columns)
    block = _product(_product(_inverse(T), M), T)
    C = _block(block, range(r), range(r))
    N = _block(block, range(r, n), range(r, n))
    return T, C, N, k


def _regular_lambda(A: ExprMatrix, B: ExprMatrix) -> Optional[Integer]:
    """An integer ``lambda`` with ``det(lambda*A + B) != 0``, or ``None``
    for a singular pencil (the determinant is a polynomial in ``lambda``
    of degree at most ``n``, so ``n + 1`` values suffice)."""
    for value in range(A.rows + 1):
        candidate = Integer(value)
        if not _is_zero(_determinant(A * candidate + B)):
            return candidate
    return None


def dae_index(A: ExprMatrix, B: ExprMatrix) -> int:
    """The index of the regular pencil ``lambda*A + B``: the index of
    nilpotency of the nilpotent part of ``(lambda*A + B)**-1 * A``
    (``0`` for an ordinary differential equation).

    Examples
    ========

    >>> from sympy_extras.matrices.dense import ExprMatrix
    >>> from sympy_extras.solvers.dae import dae_index
    >>> dae_index(ExprMatrix.from_rows([[1, 0], [0, 0]]), ExprMatrix.from_rows([[0, 1], [1, 0]]))
    2
    >>> dae_index(ExprMatrix.eye(2), ExprMatrix.from_rows([[0, 1], [1, 0]]))
    0
    """
    value = _regular_lambda(A, B)
    if value is None:
        raise ValueError("the pencil lambda*A + B is singular")
    return _index_of_nilpotency(_product(_inverse(A * value + B), A))


class DAESolution:
    """The general solution of ``A y' + B y = f``.

    Attributes
    ==========

    solution : ExprMatrix
        The column of the components of ``y``, in the free constants.
    constants : list of Symbols
        The free constants (as many as the dimension of the regular part).
    index : int
        The index of the pencil.
    """

    def __init__(self, solution: ExprMatrix, constants: list[Symbol], index: int) -> None:
        self.solution: Final[ExprMatrix] = solution
        self.constants: Final[list[Symbol]] = constants
        self.index: Final[int] = index

    def __repr__(self) -> str:
        return "DAESolution(%s, constants=%s, index=%s)" % (self.solution, self.constants, self.index)


def _integrate(e: Expr, x: Symbol) -> Expr:
    """An antiderivative of ``e``: term by term for a sum of powers of
    ``x``, by SymPy's ``integrate`` otherwise (an unevaluated integral
    when it finds none)."""
    native = _power_antiderivative(e, x)
    if native is not None:
        return native
    from .dae_sympy import integral_of_core
    return integral_of_core(e, x)


def _real_exponential(M: ExprMatrix, x: Symbol) -> ExprMatrix:
    """``exp(M*x)`` with the complex exponentials of a real matrix written
    with sines and cosines."""
    result = matrix_exponential(M, x)
    if all((e * x).is_extended_real is not False for e in M.entries):
        result = result.applyfunc(lambda e: simplify(e.rewrite(Cos)))
    return result


def _constants(count: int, taken: set[Basic]) -> list[Symbol]:
    result: list[Symbol] = []
    i = 1
    while len(result) < count:
        c = Symbol('C%d' % i)
        if c not in taken:
            result.append(c)
        i += 1
    return result


def dsolve_dae(A: ExprMatrix, B: ExprMatrix, f: Optional[ExprMatrix], x: Symbol) -> DAESolution:
    """The general solution of ``A y' + B y = f(x)`` with constant square
    matrices ``A`` and ``B`` (``f`` a column of functions of ``x``, or
    ``None`` for the homogeneous system).

    ``ValueError`` is raised when the pencil is singular (the system has
    either no solution or infinitely many for some right-hand sides).

    Examples
    ========

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.functions import sin
    >>> from sympy_extras.matrices.dense import ExprMatrix
    >>> from sympy_extras.solvers.dae import dsolve_dae
    >>> x = Symbol('x')
    >>> A = ExprMatrix.from_rows([[1, 0], [0, 0]])
    >>> B = ExprMatrix.from_rows([[0, -1], [1, 0]])          # y1' = y2, y1 = sin(x)
    >>> result = dsolve_dae(A, B, ExprMatrix.column([0, sin(x)]), x)
    >>> result.solution, result.constants, result.index
    (ExprMatrix([[sin(x)], [cos(x)]]), [], 2)
    >>> result = dsolve_dae(A, ExprMatrix.from_rows([[0, 0], [1, -1]]), None, x)
    >>> result.solution                                      # y1' = 0, y1 = y2
    ExprMatrix([[C1], [C1]])
    """
    n = A.rows
    if A.shape != (n, n) or B.shape != (n, n):
        raise ValueError("square matrices of the same size are expected")
    if f is not None and f.shape != (n, 1):
        raise ValueError("the right-hand side must be a column of length %d" % n)
    if any(x in e.free_symbols for e in A.entries + B.entries):
        raise ValueError("the coefficients must be constant")
    value = _regular_lambda(A, B)
    if value is None:
        raise ValueError("the pencil lambda*A + B is singular")
    P = _inverse(A * value + B)
    A_hat = _product(P, A)
    f_hat = P * f if f is not None else ExprMatrix.zeros(n, 1)
    T, C, N, k = core_nilpotent_decomposition(A_hat)
    r = C.rows
    g = _inverse(T) * f_hat
    g1, g2 = _block(g, range(r), range(1)), _block(g, range(r, n), range(1))
    taken: set[Basic] = set()
    if f is not None:
        for e in f_hat.entries:
            taken |= e.free_symbols
    constants = _constants(r, taken)
    # regular part: z1' = M z1 + h with M = -C**-1 (I - lambda C)
    z1 = ExprMatrix.zeros(0, 1)
    if r:
        C_inv = _inverse(C)
        M = -_product(C_inv, ExprMatrix.eye(r) - C * value)
        h = C_inv * g1
        Phi = _real_exponential(M, x)
        Phi_inv = _real_exponential(-M, x)
        particular = Phi_inv * h
        integrals = ExprMatrix.column([_integrate(e, x) for e in particular.entries])
        z1 = (Phi * (ExprMatrix.column(constants) + integrals)).applyfunc(simplify)
    # nilpotent part: z2 = sum_i (-Q N)**i Q g2**(i)
    m = n - r
    z2 = ExprMatrix.zeros(m, 1)
    if m:
        Q = _inverse(ExprMatrix.eye(m) - N * value)
        step = ExprMatrix.eye(m)
        derivative = g2
        for _ in range(k):
            z2 = z2 + step * Q * derivative
            step = _product(step, -_product(Q, N))
            derivative = derivative.diff(x)
        z2 = z2.applyfunc(simplify)
    z = _vstack(z1, z2)
    y = (T * z).applyfunc(simplify)
    return DAESolution(y, constants, k)


def _row(equation: Basic) -> Expr:
    if isinstance(equation, Eq):
        return equation.lhs - equation.rhs
    if not isinstance(equation, Expr):
        raise TypeError("an equation or an expression is expected, got %s" % (equation,))
    return equation


def dae_matrices(equations: Sequence[Basic], functions: Sequence[AppliedUndef]
                 ) -> tuple[ExprMatrix, ExprMatrix, ExprMatrix, Symbol]:
    """The matrices ``A``, ``B`` and the column ``f`` of a system of linear
    equations with constant coefficients in the functions and their first
    derivatives, ``A y' + B y = f``.

    Examples
    ========

    >>> from sympy_extras.core import Function, Symbol, Eq
    >>> from sympy_extras.functions import sin
    >>> from sympy_extras.solvers.dae import dae_matrices
    >>> x = Symbol('x')
    >>> y1, y2 = Function('y1')(x), Function('y2')(x)
    >>> A, B, f, _ = dae_matrices([Eq(y1.diff(x), y2), Eq(y1, sin(x))], [y1, y2])
    >>> A, B, f
    (ExprMatrix([[1, 0], [0, 0]]), ExprMatrix([[0, -1], [1, 0]]), ExprMatrix([[0], [sin(x)]]))
    """
    if not functions:
        raise ValueError("no unknown function")
    x_ = functions[0].args[0]
    if not isinstance(x_, Symbol):
        raise ValueError("the functions must depend on a symbol")
    x = x_
    derivatives = [Dummy('d%d' % i) for i in range(len(functions))]
    values = [Dummy('u%d' % i) for i in range(len(functions))]
    first: dict[Basic, Basic] = {diff(fn, x): d for fn, d in zip(functions, derivatives)}
    functions_rule: dict[Basic, Basic] = {fn: u for fn, u in zip(functions, values)}
    unknowns: list[Symbol] = []
    unknowns.extend(derivatives)
    unknowns.extend(values)
    zero: dict[Basic, Basic] = {s: Integer(0) for s in unknowns}
    n = len(functions)
    A_entries: list[Expr] = []
    B_entries: list[Expr] = []
    rhs: list[Expr] = []
    for equation in equations:
        # the other derivatives are looked for before the functions are
        # replaced (a derivative of a symbol would evaluate to 0)
        replaced = _replaced(_row(equation), first)
        if replaced.has(Derivative):
            raise ValueError("%s is not linear in the functions and their first derivatives" % (equation,))
        replaced = _replaced(replaced, functions_rule)
        if any(replaced.has(fn) for fn in functions):
            raise ValueError("%s is not linear in the functions and their first derivatives" % (equation,))
        coefficients = [diff(replaced, s) for s in unknowns]
        constant = _replaced(replaced, zero)
        if any(c.has(*unknowns) for c in coefficients) or constant.has(*unknowns):
            raise ValueError("%s is not linear in the functions and their first derivatives" % (equation,))
        A_entries.extend(coefficients[:n])
        B_entries.extend(coefficients[n:])
        rhs.append(-constant)
    rows = len(rhs)
    A, B = ExprMatrix(rows, n, A_entries), ExprMatrix(rows, n, B_entries)
    if any(x in e.free_symbols for e in A.entries + B.entries):
        raise ValueError("the coefficients must be constant")
    return A, B, ExprMatrix.column(rhs), x


def check_dae(A: ExprMatrix, B: ExprMatrix, f: Optional[ExprMatrix], y: ExprMatrix, x: Symbol) -> bool:
    """Whether ``y`` solves ``A y' + B y = f``.

    Examples
    ========

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.functions import sin, cos
    >>> from sympy_extras.matrices.dense import ExprMatrix
    >>> from sympy_extras.solvers.dae import check_dae
    >>> x = Symbol('x')
    >>> A, B = ExprMatrix.eye(2), ExprMatrix.from_rows([[0, -1], [1, 0]])
    >>> check_dae(A, B, None, ExprMatrix.column([cos(x), -sin(x)]), x)
    True
    """
    residual = A * y.diff(x) + B * y - (f if f is not None else ExprMatrix.zeros(A.rows, 1))
    return all(simplify(e.doit()) == 0 for e in residual.entries)
