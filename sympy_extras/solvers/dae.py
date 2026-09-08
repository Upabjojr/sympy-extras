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

SymPy's ``dsolve`` handles systems of ordinary differential equations
only (`A` invertible); a singular `A` is rejected.

References
==========

.. [Campbell] S. L. Campbell, Singular Systems of Differential Equations,
   Pitman (1980), chapter 2.
.. [Kunkel] P. Kunkel, V. Mehrmann, Differential-Algebraic Equations:
   Analysis and Numerical Solution, EMS (2006), chapter 2 (the
   Weierstrass canonical form).
"""
from __future__ import annotations

from typing import Optional, Sequence

from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.function import AppliedUndef, Derivative
from sympy.core.numbers import Integer
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.functions.elementary.trigonometric import cos
from sympy.integrals.integrals import Integral, integrate
from sympy.matrices.dense import Matrix, MutableDenseMatrix, eye, zeros
from sympy.simplify.simplify import simplify
from sympy.solvers.solveset import linear_eq_to_matrix

from sympy_extras._timeout import attempt
from sympy_extras._typing import as_expr, free_symbols
from sympy_extras.settings import settings

__all__ = ['core_nilpotent_decomposition', 'dae_index', 'dsolve_dae', 'dae_matrices', 'check_dae', 'DAESolution']


def _index_of_nilpotency(M: MutableDenseMatrix) -> int:
    """The smallest ``k`` with ``rank(M**k) == rank(M**(k + 1))``."""
    n = M.rows
    power: MutableDenseMatrix = eye(n)
    previous = n
    for k in range(n + 1):
        current = power.rank()
        if current == previous and k > 0:
            return k - 1
        previous = current
        power = power*M
    return n


def core_nilpotent_decomposition(M: MutableDenseMatrix) -> tuple[MutableDenseMatrix, MutableDenseMatrix,
                                                                  MutableDenseMatrix, int]:
    """The core-nilpotent (Drazin) decomposition of a square matrix:
    ``T``, ``C``, ``N`` and the index ``k`` with ``T**-1 * M * T`` block
    diagonal with blocks ``C`` (invertible) and ``N`` (nilpotent,
    ``N**k == 0``, ``N**(k-1) != 0``); ``k`` is ``0`` when ``M`` is
    invertible.

    Examples
    ========

    >>> from sympy import Matrix
    >>> from sympy_extras.solvers.dae import core_nilpotent_decomposition
    >>> T, C, N, k = core_nilpotent_decomposition(Matrix([[1, 1], [0, 0]]))
    >>> C, N, k
    (Matrix([[1]]), Matrix([[0]]), 1)
    >>> T.inv()*Matrix([[1, 1], [0, 0]])*T == Matrix([[1, 0], [0, 0]])
    True
    """
    if M.rows != M.cols:
        raise ValueError("a square matrix is expected, got shape %s" % (M.shape,))
    n = M.rows
    k = _index_of_nilpotency(M)
    power = M**k
    core_columns = [Matrix(c) for c in power.columnspace()]
    null_columns = [Matrix(c) for c in power.nullspace()]
    if len(core_columns) + len(null_columns) != n:
        raise ValueError("the range and the null space of M**k do not span the space")
    T: MutableDenseMatrix = Matrix.hstack(*core_columns, *null_columns) if n else zeros(0, 0)
    r = len(core_columns)
    block = T.inv()*M*T
    C = block[:r, :r]
    N = block[r:, r:]
    return T, Matrix(C), Matrix(N), k


def _regular_lambda(A: MutableDenseMatrix, B: MutableDenseMatrix) -> Optional[Integer]:
    """An integer ``lambda`` with ``det(lambda*A + B) != 0``, or ``None``
    for a singular pencil (the determinant is a polynomial in ``lambda``
    of degree at most ``n``, so ``n + 1`` values suffice)."""
    n = A.rows
    for value in range(n + 1):
        candidate = Integer(value)
        d = simplify((candidate*A + B).det())
        if d != 0:
            return candidate
    return None


def dae_index(A: MutableDenseMatrix, B: MutableDenseMatrix) -> int:
    """The index of the regular pencil ``lambda*A + B``: the index of
    nilpotency of the nilpotent part of ``(lambda*A + B)**-1 * A``
    (``0`` for an ordinary differential equation).

    Examples
    ========

    >>> from sympy import Matrix
    >>> from sympy_extras.solvers.dae import dae_index
    >>> dae_index(Matrix([[1, 0], [0, 0]]), Matrix([[0, 1], [1, 0]]))
    2
    >>> dae_index(Matrix([[1, 0], [0, 1]]), Matrix([[0, 1], [1, 0]]))
    0
    """
    value = _regular_lambda(A, B)
    if value is None:
        raise ValueError("the pencil lambda*A + B is singular")
    A_hat = (value*A + B).inv()*A
    return _index_of_nilpotency(A_hat)


class DAESolution:
    """The general solution of ``A y' + B y = f``.

    Attributes
    ==========

    solution : Matrix
        The column of the components of ``y``, in the free constants.
    constants : list of Symbols
        The free constants (as many as the dimension of the regular part).
    index : int
        The index of the pencil.
    """

    def __init__(self, solution: MutableDenseMatrix, constants: list[Symbol], index: int) -> None:
        self.solution = solution
        self.constants = constants
        self.index = index

    def __repr__(self) -> str:
        return "DAESolution(%s, constants=%s, index=%s)" % (self.solution, self.constants, self.index)


def _integrate(e: Expr, x: Symbol) -> Expr:
    value = attempt(lambda: as_expr(integrate(e, x)), settings.timeout)
    if value is None or value.has(Integral):
        return as_expr(Integral(e, x))
    return value


def _real_exponential(M: MutableDenseMatrix) -> MutableDenseMatrix:
    """``exp(M)`` with the complex exponentials of a real matrix written
    with sines and cosines."""
    result = M.exp()
    if all(as_expr(e).is_extended_real is not False for e in M.flat()):
        result = result.applyfunc(lambda e: simplify(as_expr(e).rewrite(cos)))
    return Matrix(result)


def _constants(count: int, taken: set[Symbol]) -> list[Symbol]:
    result: list[Symbol] = []
    i = 1
    while len(result) < count:
        c = Symbol('C%d' % i)
        if c not in taken:
            result.append(c)
        i += 1
    return result


def dsolve_dae(A: MutableDenseMatrix, B: MutableDenseMatrix, f: Optional[MutableDenseMatrix], x: Symbol) -> DAESolution:
    """The general solution of ``A y' + B y = f(x)`` with constant square
    matrices ``A`` and ``B`` (``f`` a column of functions of ``x``, or
    ``None`` for the homogeneous system).

    ``ValueError`` is raised when the pencil is singular (the system has
    either no solution or infinitely many for some right-hand sides).

    Examples
    ========

    >>> from sympy import Matrix, sin, Symbol
    >>> from sympy_extras.solvers.dae import dsolve_dae
    >>> x = Symbol('x')
    >>> A = Matrix([[1, 0], [0, 0]])
    >>> B = Matrix([[0, -1], [1, 0]])          # y1' = y2, y1 = sin(x)
    >>> result = dsolve_dae(A, B, Matrix([0, sin(x)]), x)
    >>> result.solution.T, result.constants, result.index
    (Matrix([[sin(x), cos(x)]]), [], 2)
    >>> result = dsolve_dae(Matrix([[1, 0], [0, 0]]), Matrix([[0, 0], [1, -1]]), None, x)
    >>> result.solution.T                      # y1' = 0, y1 = y2
    Matrix([[C1, C1]])
    """
    n = A.rows
    if A.shape != (n, n) or B.shape != (n, n):
        raise ValueError("square matrices of the same size are expected")
    if f is not None and f.shape != (n, 1):
        raise ValueError("the right-hand side must be a column of length %d" % n)
    if any(x in free_symbols(as_expr(e)) for e in A.flat() + B.flat()):
        raise ValueError("the coefficients must be constant")
    value = _regular_lambda(A, B)
    if value is None:
        raise ValueError("the pencil lambda*A + B is singular")
    P = (value*A + B).inv()
    A_hat = P*A
    f_hat: MutableDenseMatrix = P*f if f is not None else zeros(n, 1)
    T, C, N, k = core_nilpotent_decomposition(A_hat)
    r = C.rows
    g = T.inv()*f_hat
    g1, g2 = Matrix(g[:r, :]), Matrix(g[r:, :])
    taken: set[Symbol] = set()
    if f is not None:
        for e in f_hat.flat():
            taken |= free_symbols(as_expr(e))
    constants = _constants(r, taken)
    # regular part: z1' = M z1 + h with M = -C**-1 (I - lambda C)
    z1: MutableDenseMatrix = zeros(0, 1)
    if r:
        C_inv = C.inv()
        M = -C_inv*(eye(r) - value*C)
        h = C_inv*g1
        Phi = _real_exponential(M*x)
        Phi_inv = _real_exponential(-M*x)
        particular = Phi_inv*h
        z1 = Phi*(Matrix(constants) + Matrix([_integrate(as_expr(e), x) for e in particular]))
        z1 = z1.applyfunc(lambda e: simplify(e))
    # nilpotent part: z2 = sum_i (-Q N)**i Q g2**(i)
    m = n - r
    z2: MutableDenseMatrix = zeros(m, 1)
    if m:
        Q = (eye(m) - value*N).inv()
        step: MutableDenseMatrix = eye(m)
        derivative = g2
        for _ in range(k):
            z2 = z2 + step*Q*derivative
            step = step*(-Q*N)
            derivative = derivative.diff(x)
        z2 = z2.applyfunc(lambda e: simplify(e))
    z = Matrix.vstack(z1, z2) if r and m else (z1 if r else z2)
    y = (T*z).applyfunc(lambda e: simplify(e))
    return DAESolution(Matrix(y), constants, k)


def dae_matrices(equations: Sequence[Basic], functions: Sequence[AppliedUndef]
                 ) -> tuple[MutableDenseMatrix, MutableDenseMatrix, MutableDenseMatrix, Symbol]:
    """The matrices ``A``, ``B`` and the column ``f`` of a system of linear
    equations with constant coefficients in the functions and their first
    derivatives, ``A y' + B y = f``.

    Examples
    ========

    >>> from sympy import Function, Symbol, Eq, sin
    >>> from sympy_extras.solvers.dae import dae_matrices
    >>> x = Symbol('x')
    >>> y1, y2 = Function('y1')(x), Function('y2')(x)
    >>> A, B, f, _ = dae_matrices([Eq(y1.diff(x), y2), Eq(y1, sin(x))], [y1, y2])
    >>> A, B, f.T
    (Matrix([
    [1, 0],
    [0, 0]]), Matrix([
    [0, -1],
    [1,  0]]), Matrix([[0, sin(x)]]))
    """
    if not functions:
        raise ValueError("no unknown function")
    x_ = functions[0].args[0]
    if not isinstance(x_, Symbol):
        raise ValueError("the functions must depend on a symbol")
    x = x_
    derivatives = [Dummy('d%d' % i) for i in range(len(functions))]
    values = [Dummy('u%d' % i) for i in range(len(functions))]
    replacements: dict[Basic, Basic] = {}
    for fn, d, u in zip(functions, derivatives, values):
        replacements[Derivative(fn, x)] = d
    for fn, d, u in zip(functions, derivatives, values):
        replacements[fn] = u
    rows: list[Expr] = []
    for equation in equations:
        lhs = equation.lhs - equation.rhs if isinstance(equation, Eq) else as_expr(equation)
        replaced = as_expr(as_expr(lhs).xreplace(replacements))
        if replaced.has(Derivative) or any(replaced.has(fn) for fn in functions):
            raise ValueError("%s is not linear in the functions and their first derivatives" % (equation,))
        rows.append(replaced)
    M, rhs = linear_eq_to_matrix(rows, derivatives + values)
    n = len(functions)
    A, B = Matrix(M[:, :n]), Matrix(M[:, n:])
    if any(x in free_symbols(as_expr(e)) for e in A.flat() + B.flat()):
        raise ValueError("the coefficients must be constant")
    return A, B, Matrix(rhs), x


def check_dae(A: MutableDenseMatrix, B: MutableDenseMatrix, f: Optional[MutableDenseMatrix],
              y: MutableDenseMatrix, x: Symbol) -> bool:
    """Whether ``y`` solves ``A y' + B y = f``."""
    residual = A*y.diff(x) + B*y - (f if f is not None else zeros(A.rows, 1))
    return all(simplify(as_expr(e).doit()) == S.Zero for e in residual)
