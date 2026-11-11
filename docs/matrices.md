# Linear algebra on matrices of expressions

`sympy_extras.matrices` has two layers:

* `DomainMatrix` (`sympy_extras.matrices.domainmatrix`): a dense matrix
  over a typed `Domain[T]` (`ZZ`, `QQ`, `GF(p)`, a polynomial ring), with
  fraction-free algorithms;
* `sympy_extras.matrices.eigen`: eigenvalues, eigenvectors,
  diagonalisation and the Jordan form; `sympy_extras.matrices.functions`:
  the matrix exponential, symbolic powers, functions of matrices;
  `sympy_extras.matrices.svd`: the singular value decomposition;
* `sympy_extras.matrices.linalg`: the same operations on an `ExprMatrix`
  of expressions of the core, also available as methods (`M.det()`,
  `M.inv()`, `M.rank()`, `M.rref()`, `M.nullspace()`, `M.adjugate()`,
  `M.charpoly(x)`, `M.solve(B)`).

## Why fraction-free

Gaussian elimination on expressions adds and multiplies rational
functions at every step; the entries grow, and keeping them in lowest
terms costs a polynomial gcd per operation. Here the matrix is first
scaled to one over the ring $\mathbb{Z}[k_1, \dots, k_m]$ of polynomials
in its *kernels* (the symbols, and every other subexpression which is not
a sum, a product or an integer power), and the elimination stays in that
ring:

* **Bareiss' update** $a_{ij} \leftarrow (p_k a_{ij} - a_{ik} a_{kj}) /
  p_{k-1}$ divides exactly by the previous pivot (every entry produced is
  a minor of the matrix, by Sylvester's identity); the last pivot is the
  determinant;
* applied above the pivot as well, it gives a matrix $R$ and one
  denominator $d$ with $R/d$ the reduced row echelon form: the rank, the
  null space, the inverse and the solutions of systems are read off it;
* **Berkowitz' algorithm** computes the characteristic polynomial without
  any division; the adjugate of a singular matrix follows by the
  Cayley–Hamilton theorem.

Results are quotients of expanded polynomials in lowest terms.

```python
>>> from sympy_extras.core import Symbol
>>> from sympy_extras.matrices import ExprMatrix
>>> a, b, c, d, x = (Symbol(s) for s in 'abcdx')
>>> M = ExprMatrix.from_rows([[a, b], [c, d]])
>>> M.det(), M.inv()
(a*d - b*c, ExprMatrix([[d/(a*d - b*c), -b/(a*d - b*c)], [-c/(a*d - b*c), a/(a*d - b*c)]]))
>>> M.charpoly(x)
a*d - b*c + x**2 + x*(-a - d)
>>> S = ExprMatrix.from_rows([[1, a, 2], [a, 1, 3]])
>>> S.rref()
(ExprMatrix([[1, 0, (3*a - 2)/(a**2 - 1)], [0, 1, (2*a - 3)/(a**2 - 1)]]), (0, 1))
>>> S.nullspace()
[ExprMatrix([[(2 - 3*a)/(a**2 - 1)], [(3 - 2*a)/(a**2 - 1)], [1]])]

```

Linear systems of any shape, with the general solution:

```python
>>> from sympy_extras.matrices.linalg import linear_solve
>>> linear_solve(ExprMatrix.from_rows([[1, 1, 1], [1, 2, a]]), ExprMatrix.column([1, 2]))
LinearSolution(ExprMatrix([[0], [1], [0]]), [ExprMatrix([[a - 2], [1 - a], [1]])])
>>> linear_solve(ExprMatrix.from_rows([[1, 1], [1, 1]]), ExprMatrix.column([1, 2])) is None
True

```

The Moore–Penrose pseudoinverse of a real matrix comes from a full rank
factorisation `M = B*C` (the pivot columns and the nonzero rows of the
echelon form): `M.pinv()` is `C.T*(C*C.T)**-1*(B.T*B)**-1*B.T`, and
`M.pinv()*b` is the least squares solution of least norm.

```python
>>> ExprMatrix.from_rows([[1, 0], [0, 1], [1, 1]]).pinv()
ExprMatrix([[2/3, -1/3, 1/3], [-1/3, 2/3, 1/3]])
>>> ExprMatrix.from_rows([[1, a], [1, a]]).pinv()
ExprMatrix([[1/(2*a**2 + 2), 1/(2*a**2 + 2)], [a/(2*a**2 + 2), a/(2*a**2 + 2)]])

```

## What the answers mean

The determinant, the adjugate and the characteristic polynomial are
polynomial identities in the entries: they hold whatever relations there
are between the kernels.

```python
>>> from sympy_extras.functions import sin, cos
>>> t = Symbol('t')
>>> R = ExprMatrix.from_rows([[cos(t), -sin(t)], [sin(t), cos(t)]])
>>> R.det()
sin(t)**2 + cos(t)**2

```

The rank, the echelon form, the null space, the inverse and the solutions
of systems depend on which quantities vanish.

* With symbols only, the answer is the *generic* one: it holds wherever
  no pivot vanishes (`a**2 - 1` above). Case distinctions on the
  parameters are the business of `solve(..., cases=True)` and `resolve`
  (see `docs/reduce.md`).
* With the imaginary unit `I` and square roots of rational numbers (and
  symbols) the arithmetic is that of the field $K(i, \sqrt{p_1}, \dots)$
  over the rational functions $K$, for the primes involved
  (`sympy_extras.polys.domains.multiquadratic.MultiquadraticField`):
  exact as well.

```python
>>> from sympy_extras.core import I
>>> from sympy_extras.functions import sqrt
>>> P = ExprMatrix.from_rows([[1, I], [I, -1]])
>>> P.rank(), P.nullspace()
(1, [ExprMatrix([[-I], [1]])])
>>> ExprMatrix.from_rows([[1 + I, a], [2, 1 - I]]).inv()
ExprMatrix([[-1/(2*a - 2) + I/(2*a - 2), a/(2*a - 2)], [1/(a - 1), -1/(2*a - 2) - I/(2*a - 2)]])
>>> Q = ExprMatrix.from_rows([[sqrt(2), sqrt(3)], [sqrt(6), 3]])
>>> Q.det(), Q.rank(), Q.nullspace()
(0, 1, [ExprMatrix([[-sqrt(6)/2], [1]])])
>>> W = ExprMatrix.from_rows([[1, 1], [1, -1]])/sqrt(2)
>>> W.inv() == W, W.eigenvals()
(True, {-1: 1, 1: 1})

```

* With other kernels (`sin(t)`, `2**(1/3)`) a nonzero polynomial may
  be a zero quantity. Every pivot is then submitted to the zero test of
  `sympy_extras.simplify.structure`, and `NotImplementedError` is raised
  unless all are proved nonzero, so that no wrong rank is returned:

```python
>>> R.inv()
ExprMatrix([[cos(t)/(sin(t)**2 + cos(t)**2), sin(t)/(sin(t)**2 + cos(t)**2)], [-sin(t)/(sin(t)**2 + cos(t)**2), cos(t)/(sin(t)**2 + cos(t)**2)]])
>>> H = ExprMatrix.from_rows([[sin(t)**2 + cos(t)**2, 1], [1, 1]])
>>> H.rank()
Traceback (most recent call last):
...
NotImplementedError: the pivot -sin(t)**2 - cos(t)**2 + 1 is not proved nonzero: simplify the entries of the matrix first

```

## Over a typed domain

```python
>>> from sympy_extras.polys.domains import ZZ, GF
>>> from sympy_extras.matrices import DomainMatrix
>>> A = DomainMatrix([[2, 1, 1], [1, 3, 2], [1, 0, 0]], ZZ)
>>> A.det(), A.rank(), A.charpoly()
(-1, 3, [1, -5, 4, 1])
>>> A.inv_den()
(DomainMatrix([[0, 0, 1], [-2, 1, 3], [3, -1, -5]], ZZ), 1)
>>> DomainMatrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]], ZZ).rref_den()
(DomainMatrix([[3, 0, -3], [0, 3, 6], [0, 0, 0]], ZZ), 3, [0, 1])
>>> DomainMatrix([[1, 2], [3, 4]], GF(7)).inv()
DomainMatrix([[5, 1], [5, 3]], GF(7))

```

`rref_den`, `inv_den`, `solve_den` and `nullspace` return numerators with
a single denominator and need only exact division, so they work over a
ring; `rref` and `inv` are for fields.

## Timings

Seconds, on one machine, against `sympy.Matrix` (SymPy 1.14); the entries
of the symbolic matrices are `p*x + q*y + r` with random small integers.

| operation | here | SymPy |
|---|---|---|
| determinant, 6 x 6, entries linear in `x, y` | 0.03 | 3.9 |
| determinant, 8 x 8, entries linear in `x, y` | 0.12 | 27 |
| determinant, 10 x 10, entries linear in `x, y` | 0.6 | 129 |
| determinant, 5 x 5 of distinct symbols | 0.02 | 0.09 |
| determinant, 60 x 60 of two-digit integers | 0.04 | 3.7 |
| inverse, 6 x 6, entries linear in `x, y` (in lowest terms here, not in SymPy) | 0.2 | 0.3 |
| inverse, 60 x 60 of two-digit integers | 0.4 | 0.5 |

## Eigenvalues and eigenvectors

`sympy_extras.matrices.eigen` factors the characteristic polynomial over
the field $K$ of rational functions in the symbols of the matrix, and
computes the eigenvectors *once for all the roots of each irreducible
factor* $f$: the null space of $M - \lambda I$ is found by elimination
over the field $K[\lambda]/(f)$
(`sympy_extras.polys.domains.quotientfield.QuotientField`), in which
$\lambda$ is a root of $f$ by construction. No expression in radicals is
ever tested for zero, and the structure does not depend on solving $f$.

```python
>>> from sympy_extras.matrices.eigen import eigenspaces, is_diagonalizable
>>> F = ExprMatrix.from_rows([[1, 1], [1, 0]])
>>> eigenspaces(F, x)
[Eigenspace(x**2 - x - 1, 1, [ExprMatrix([[x], [1]])])]
>>> F.eigenvals()
{1/2 - sqrt(5)/2: 1, 1/2 + sqrt(5)/2: 1}
>>> F.eigenvects()
[(1/2 - sqrt(5)/2, 1, [ExprMatrix([[1/2 - sqrt(5)/2], [1]])]), (1/2 + sqrt(5)/2, 1, [ExprMatrix([[1/2 + sqrt(5)/2], [1]])])]
>>> ExprMatrix.from_rows([[a, b], [b, a]]).diagonalize()
(ExprMatrix([[-1, 1], [1, 1]]), ExprMatrix([[a - b, 0], [0, a + b]]))
>>> J = ExprMatrix.from_rows([[a, 1], [0, a]])
>>> eigenspaces(J, x), J.is_diagonalizable()
([Eigenspace(-a + x, 2, [ExprMatrix([[1], [0]])])], False)

```

An `Eigenspace` has the irreducible `factor`, its `multiplicity` in the
characteristic polynomial, and a `basis` of the eigenspace in terms of
the eigenvalue; `values()` gives the roots in radicals when they are
found, `vectors(value)` the basis for one of them. When a factor is not
solved, the structure is still complete (and `is_diagonalizable` is
decided), while `eigenvals`, `eigenvects` and `diagonalize` raise
`NotImplementedError`:

```python
>>> C = ExprMatrix.from_rows([[0, 0, 0, 0, 1], [1, 0, 0, 0, 1], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0]])
>>> [space] = eigenspaces(C, x)
>>> space.factor, space.values() is None, space.basis
(x**5 - x - 1, True, (ExprMatrix([[x**4 - 1], [x**3], [x**2], [x], [1]]),))
>>> C.is_diagonalizable()
True

```

The entries are rational functions of symbols, possibly with `I` and
square roots of rationals; the characteristic polynomial is factored
over $\mathbb{Q}$ or $\mathbb{Q}(i)$ (its coefficients must not contain
square roots):

```python
>>> Y = ExprMatrix.from_rows([[0, -I], [I, 0]])
>>> Y.eigenvects()
[(-1, 1, [ExprMatrix([[I], [1]])]), (1, 1, [ExprMatrix([[-I], [1]])])]
>>> ExprMatrix.from_rows([[2, 1 - I], [1 + I, 3]]).eigenvals()
{1: 1, 4: 1}

```

### Pencils

`sympy_extras.matrices.pencils` solves the generalised problem $A v =
\lambda B v$ in the same way: $\det(A - \lambda B)$ is factored and the
null space of $A - \lambda B$ is computed over $K[\lambda]/(f)$ for each
factor, without inverting $B$. A singular $B$ gives the eigenvalue
`oo`, with the multiplicity by which the degree of the determinant
drops.

```python
>>> from sympy_extras.matrices.pencils import generalized_eigenvalues, generalized_eigenspaces
>>> generalized_eigenvalues(ExprMatrix.from_rows([[0, 1], [1, 0]]), ExprMatrix.from_rows([[1, 0], [0, 2]]))
{-sqrt(2)/2: 1, sqrt(2)/2: 1}
>>> generalized_eigenvalues(ExprMatrix.from_rows([[1, 2], [3, 4]]), ExprMatrix.from_rows([[1, 0], [0, 0]]))
{-1/2: 1, oo: 1}
>>> generalized_eigenspaces(ExprMatrix.from_rows([[0, 1], [1, 0]]), ExprMatrix.from_rows([[1, 0], [0, 2]]), x)
[Eigenspace(2*x**2 - 1, 1, [ExprMatrix([[2*x], [1]])])]

```

### The Jordan form

`jordan_chains(M, x)` computes the chains of generalised eigenvectors in
the same way: for each irreducible factor, the kernels of the powers of
$M - \lambda I$ over $K[\lambda]/(f)$ give the sizes of the blocks, and
the chains are built from the longest down, each level completing the
kernel of the previous power. `jordan_form(M)` (or `M.jordan_form()`)
substitutes the roots and returns `(P, J)` with `M P = P J`.

```python
>>> from sympy_extras.matrices.eigen import jordan_chains
>>> G = ExprMatrix.from_rows([[3, 1, 0], [-1, 1, 0], [0, 0, 2]])
>>> jordan_chains(G, x)
[JordanChains(x - 2, 3, blocks [2, 1])]
>>> G.jordan_form()
(ExprMatrix([[1, 1, 0], [-1, 0, 0], [0, 0, 1]]), ExprMatrix([[2, 1, 0], [0, 2, 0], [0, 0, 2]]))
>>> ExprMatrix.from_rows([[a, 1], [0, a]]).jordan_form()
(ExprMatrix([[1, 0], [0, 1]]), ExprMatrix([[a, 1], [0, a]]))

```

## Functions of matrices

`sympy_extras.matrices.functions` computes $f(M) = \sum_\lambda \sum_j
f^{(j)}(\lambda) C_{\lambda, j}$ from the *component matrices*
$C_{\lambda, j} = (M - \lambda)^j E_\lambda / j!$, with $E_\lambda$ the
projector on the generalised eigenspace, a polynomial in $M$. The
components are computed once for all the roots of an irreducible factor
of the characteristic polynomial, over the field it defines: no matrix of
eigenvectors is inverted. `matrix_exp(M, t)` (or `M.exp(t)`) is the
fundamental matrix of $y' = M y$, with cosines and sines for complex
pairs of eigenvalues; `matrix_power(M, n)` (or `M.power(n)`) is a closed
form of the powers; `matrix_function(M, f, x)` takes any expression `f`
in `x`.

```python
>>> from sympy_extras.functions import sqrt
>>> from sympy_extras.matrices.functions import matrix_function, matrix_components
>>> t, n = Symbol('t'), Symbol('n')
>>> ExprMatrix.from_rows([[0, 1], [-1, 0]]).exp(t)
ExprMatrix([[cos(t), sin(t)], [-sin(t), cos(t)]])
>>> ExprMatrix.from_rows([[a, 1], [0, a]]).exp(t)
ExprMatrix([[exp(a*t), t*exp(a*t)], [0, exp(a*t)]])
>>> ExprMatrix.from_rows([[1, -2], [2, 1]]).exp(t)
ExprMatrix([[exp(t)*cos(2*t), -exp(t)*sin(2*t)], [exp(t)*sin(2*t), exp(t)*cos(2*t)]])
>>> ExprMatrix.from_rows([[2, 1], [0, 2]]).power(n)
ExprMatrix([[2**n, 2**n*n/2], [0, 2**n]])
>>> matrix_function(ExprMatrix.from_rows([[5, 4], [4, 5]]), sqrt(x), x)
ExprMatrix([[2, 1], [1, 2]])
>>> matrix_components(ExprMatrix.from_rows([[1, 1], [1, 0]]), x)
[MatrixComponents(x**2 - x - 1, [ExprMatrix([[x/5 + 2/5, 2*x/5 - 1/5], [2*x/5 - 1/5, 3/5 - x/5]])])]

```

The eigenvalues must be found in radicals (`NotImplementedError`
otherwise; `matrix_components` always works).

## Singular values

`sympy_extras.matrices.svd` computes the singular value decomposition of
a matrix of rational numbers exactly. The orthonormal eigenvectors of
$A^T A$ come from the eigenspaces above; inside an eigenspace of
dimension greater than one they are made orthogonal by a Gram–Schmidt
process without square roots, in the field $\mathbb{Q}[\lambda]/(f)$,
and the square roots are taken once at the end. The three real roots of
an irreducible cubic are written in trigonometric form, their radical
form going through complex numbers. `svd` returns the compact
decomposition `(U, S, V)` with `A = U * S * V.transpose()`, the nonzero
singular values in decreasing order.

```python
>>> B = ExprMatrix.from_rows([[3, 0], [4, 5]])
>>> B.singular_values()
[3*sqrt(5), sqrt(5)]
>>> U, S, V = B.svd()
>>> U, V
(ExprMatrix([[sqrt(10)/10, -3*sqrt(10)/10], [3*sqrt(10)/10, sqrt(10)/10]]), ExprMatrix([[sqrt(2)/2, -sqrt(2)/2], [sqrt(2)/2, sqrt(2)/2]]))
>>> (U*S*V.transpose()).applyfunc(lambda e: e.expand()) == B
True
>>> ExprMatrix.from_rows([[1, 1], [1, 1]]).svd()
(ExprMatrix([[sqrt(2)/2], [sqrt(2)/2]]), ExprMatrix([[2]]), ExprMatrix([[sqrt(2)/2], [sqrt(2)/2]]))

```

`singular_values` also accepts symbols (taken to be real); `svd` raises
`NotImplementedError` for them and when the singular values are not
expressed in radicals.

## Decompositions

`sympy_extras.matrices.decompositions` factors matrices exactly, in the
field of the entries (rational functions of symbols, with `I` and square
roots of rationals): every zero test of the eliminations is exact. With
`I` in the entries the symbols are real and the transposes are conjugate
transposes.

```python
>>> from sympy_extras.matrices.decompositions import (lu, ldl, cholesky, qr, congruence_diagonalize, inertia,
...     leading_principal_minors)
>>> F = lu(ExprMatrix.from_rows([[0, 1, 2], [1, x, 0], [2, 2*x, 1]]))
>>> F.L, F.U, F.permutation
(ExprMatrix([[1, 0, 0], [0, 1, 0], [2, 0, 1]]), ExprMatrix([[1, x, 0], [0, 1, 2], [0, 0, 1]]), (1, 0, 2))
>>> S = ExprMatrix.from_rows([[4, 2, 2], [2, 5, 3], [2, 3, 6]])
>>> ldl(S)
(ExprMatrix([[1, 0, 0], [1/2, 1, 0], [1/2, 1/2, 1]]), ExprMatrix([[4, 0, 0], [0, 4, 0], [0, 0, 4]]))
>>> S.cholesky()
ExprMatrix([[2, 0, 0], [1, 2, 0], [1, 1, 2]])
>>> ExprMatrix.from_rows([[1, 1], [1, 0], [0, 1]]).qr()
(ExprMatrix([[sqrt(2)/2, sqrt(6)/6], [sqrt(2)/2, -sqrt(6)/6], [0, sqrt(6)/3]]), ExprMatrix([[sqrt(2), sqrt(2)/2], [0, sqrt(6)/2]]))

```

* `lu` takes any shape and rank: $P M = L U$ with $U$ in row echelon
  form, and a row is moved only when the entry at the pivot position is
  zero.
* `ldl` ($M = L D L^*$) needs nonzero leading principal minors;
  `cholesky` multiplies $L$ by the square roots of the pivots, and
  rejects a rational pivot which is not positive.
* `qr` and `gram_schmidt` orthogonalise without square roots and take
  the norms at the end; for dependent columns $Q$ has as many columns as
  the rank and $R$ is in row echelon form, so that $M = Q R$ always.
* `congruence_diagonalize` finds $P$ with $P^* M P$ diagonal for *any*
  symmetric or Hermitian matrix (Lagrange's reduction of the quadratic
  form, with a step for a zero diagonal). By Sylvester's law of inertia
  the signs of that diagonal are those of the eigenvalues: `inertia`,
  `is_positive_definite`, `is_positive_semidefinite` decide the
  definiteness of a matrix of numbers without computing an eigenvalue
  (the signs of the pivots are those of rational numbers, or the ones
  the assumptions of the core decide: `sqrt(2) - 1`, a symbol declared
  positive). With symbols, `leading_principal_minors` gives the
  quantities of Sylvester's criterion.

```python
>>> congruence_diagonalize(ExprMatrix.from_rows([[0, 1], [1, 0]]))
(ExprMatrix([[1, -1/2], [1, 1/2]]), ExprMatrix([[2, 0], [0, -1/2]]))
>>> inertia(ExprMatrix.from_rows([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])), inertia(ExprMatrix.from_rows([[0, 1], [1, 0]]))
((3, 0, 0), (1, 1, 0))
>>> leading_principal_minors(ExprMatrix.from_rows([[2, a, 0], [a, 2, a], [0, a, 2]]))
[2, 4 - a**2, 8 - 4*a**2]

```

## Matrix equations

`sympy_extras.matrices.equations` solves equations which are linear in
an unknown matrix, $\sum_i A_i X B_i = C$, through the identity
$\operatorname{vec}(A X B) = (B^T \otimes A) \operatorname{vec}(X)$ and
the elimination above. Sylvester's equation $A X + X B = C$ and the
Lyapunov equations of stability theory are the usual cases; when the
solution is not unique the solutions of the homogeneous equation come
with it.

```python
>>> from sympy_extras.matrices.equations import sylvester_solve, lyapunov_solve, commuting_matrices
>>> k = Symbol('k')
>>> lyapunov_solve(ExprMatrix.from_rows([[0, 1], [-1, -k]]), ExprMatrix.eye(2)).particular
ExprMatrix([[(k**2 + 2)/(2*k), -1/2], [-1/2, 1/k]])
>>> A = ExprMatrix.from_rows([[1, 2], [3, 4]])
>>> sylvester_solve(A, ExprMatrix.from_rows([[0, 1], [1, 1]]), ExprMatrix.eye(2))
MatrixSolution(ExprMatrix([[-26/41, 23/41], [22/41, -10/41]]), [])
>>> sylvester_solve(A, -A, ExprMatrix.eye(2)) is None
True
>>> commuting_matrices(ExprMatrix.from_rows([[0, 1], [0, 0]]))
[ExprMatrix([[0, 1], [0, 0]]), ExprMatrix([[1, 0], [0, 1]])]

```

The first result is the matrix of the Lyapunov function of the damped
oscillator $x'' + k x' + x = 0$: it is positive definite exactly for
$k > 0$. The system solved has $mn$ unknowns for an $m \times n$
unknown matrix; the Bartels–Stewart reduction to triangular form, which
avoids it, needs the eigenvalues and is not used.

The algebraic Riccati equation $A^T X + X A - X B R^{-1} B^T X + Q = 0$
of the linear-quadratic regulator is quadratic in $X$:
`sympy_extras.matrices.riccati` solves it as a polynomial system for
rational data and selects the stabilising solution by the criterion of
Routh and Hurwitz, certified in interval arithmetic and, when a sign
stays undecided, in exact arithmetic on algebraic numbers.

```python
>>> from sympy_extras.matrices.riccati import care_solve, lqr, riccati_solutions
>>> A2, B2 = ExprMatrix.from_rows([[0, 1], [0, 0]]), ExprMatrix.column([0, 1])
>>> riccati_solutions(A2, B2, ExprMatrix.eye(2), ExprMatrix.eye(1))
[ExprMatrix([[-sqrt(3), 1], [1, -sqrt(3)]]), ExprMatrix([[sqrt(3), 1], [1, sqrt(3)]])]
>>> lqr(A2, B2, ExprMatrix.eye(2), ExprMatrix.eye(1))
(ExprMatrix([[1, sqrt(3)]]), ExprMatrix([[sqrt(3), 1], [1, sqrt(3)]]))
>>> from sympy_extras.matrices.riccati import dare_solve, dlqr
>>> dare_solve(ExprMatrix.eye(1), ExprMatrix.eye(1), ExprMatrix.eye(1), ExprMatrix.eye(1))        # discrete time
ExprMatrix([[1/2 + sqrt(5)/2]])

```

Systems in state space form $x' = A x + B u$, $y = C x + D u$ have their
tools in `sympy_extras.matrices.statespace`: Kalman's matrices and rank
tests, the transfer function in lowest terms, pole placement by
Ackermann's formula and the controllable canonical realisation of a
rational function. The stability conditions with parameters are in
`sympy_extras.solvers.stability` (see [solvers.md](solvers.md)).

```python
>>> from sympy_extras.matrices.statespace import is_controllable, transfer_function, pole_placement, realization
>>> s, k, m = Symbol('s'), Symbol('k'), Symbol('m')
>>> A3, B3 = ExprMatrix.from_rows([[0, 1], [-k/m, 0]]), ExprMatrix.column([0, 1/m])
>>> is_controllable(A3, B3), transfer_function(A3, B3, ExprMatrix.from_rows([[1, 0]]), s)
(True, ExprMatrix([[1/(k + m*s**2)]]))
>>> pole_placement(A3, B3, (s + 1)*(s + 2), s)
ExprMatrix([[-k + 2*m, 3*m]])
>>> realization((s + 3)/(s**2 + 3*s + 2), s)
(ExprMatrix([[0, 1], [-2, -3]]), ExprMatrix([[0], [1]]), ExprMatrix([[3, 1]]), ExprMatrix([[0]]))
>>> from sympy_extras.matrices.statespace import minimal_realization
>>> minimal_realization(ExprMatrix.from_rows([[-1, 0], [0, -2]]), ExprMatrix.column([1, 1]), ExprMatrix.from_rows([[1, 0]]))
(ExprMatrix([[-1]]), ExprMatrix([[1]]), ExprMatrix([[1]]))

```

## Normal forms without eigenvalues

`sympy_extras.matrices.normalforms` computes the Smith normal form of a
matrix of polynomials over $K[x]$, with the unimodular $U$, $V$ such
that $U M V$ is diagonal, each diagonal entry dividing the next. Applied
to $x I - A$ it gives the *invariant factors* of a square matrix $A$:
the last is the minimal polynomial, their product the characteristic
polynomial, and two matrices are similar exactly when they have the
same ones.

```python
>>> from sympy_extras.matrices.normalforms import (smith_normal_form, invariant_factors, matrix_minimal_polynomial,
...     frobenius_form, elementary_divisors, is_similar)
>>> smith_normal_form(ExprMatrix.from_rows([[x, 0], [0, x + 1]]), x).S
ExprMatrix([[1, 0], [0, x**2 + x]])
>>> D = ExprMatrix.from_rows([[2, 0, 0], [0, 2, 0], [0, 0, 3]])
>>> invariant_factors(D, x), matrix_minimal_polynomial(D, x)
([x - 2, x**2 - 5*x + 6], x**2 - 5*x + 6)
>>> P, F = frobenius_form(D)
>>> F, D*P == P*F
(ExprMatrix([[2, 0, 0], [0, 0, -6], [0, 1, 5]]), True)
>>> elementary_divisors(ExprMatrix.from_rows([[a, 1, 0], [0, a, 0], [0, 0, a]]), x)
[(-a + x, 1), (-a + x, 2)]
>>> is_similar(ExprMatrix.from_rows([[1, 1], [0, 1]]), ExprMatrix.from_rows([[1, 0], [5, 1]]))
True

```

With row operations only, `hermite_normal_form` gives the canonical form
of the module generated by the rows (`U M = H` in echelon form, monic
pivots, lower degrees above them), and `is_unimodular` tells whether a
matrix of polynomials has a polynomial inverse:

```python
>>> from sympy_extras.matrices.normalforms import hermite_normal_form, is_unimodular
>>> hermite_normal_form(ExprMatrix.from_rows([[x**2, x + 1], [x, 1]]), x)
(ExprMatrix([[x, 0], [0, 1]]), ExprMatrix([[-1, x + 1], [1, -x]]))
>>> is_unimodular(ExprMatrix.from_rows([[x, x + 1], [x - 1, x]]), x)
True

```

The Frobenius (rational canonical) form is the block diagonal matrix of
the companion matrices of the invariant factors. It exists over the
field of the entries: where the Jordan form needs the roots of the
characteristic polynomial, `frobenius_form` only divides polynomials.
The change of basis comes from the reduction itself: $K^n$ is the
quotient of $K[x]^n$ by the columns of $x I - A$, so the columns of
$U^{-1}$, evaluated at $A$, generate the cyclic summands. The
elementary divisors (the invariant factors split into powers of
irreducible polynomials) give the sizes of the Jordan blocks.

## Step by step

The fraction-free methods above produce intermediate matrices which mean
nothing to a reader. `sympy_extras.matrices.steps` does the elimination
of the textbooks instead and returns every elementary row operation with
the matrix it gives: `rref_steps`, `echelon_steps` (forward elimination
only), `inverse_steps` (on $[M \mid I]$), `solve_steps` (on the
augmented matrix) and `determinant_steps` (the product of the pivots,
with the sign of the row exchanges).

```python
>>> from sympy_extras.matrices.steps import rref_steps, inverse_steps, determinant_steps, format_steps
>>> E = ExprMatrix.from_rows([[0, 2, 4], [1, 1, 1]])
>>> print(format_steps(E, rref_steps(E)))
[0, 2, 4]
[1, 1, 1]
R1 <-> R2
[1, 1, 1]
[0, 2, 4]
R2 <- (1/2)*R2
[1, 1, 1]
[0, 1, 2]
R1 <- R1 - R2
[1, 0, -1]
[0, 1,  2]
>>> [str(step.operation) for step in inverse_steps(ExprMatrix.from_rows([[1, 2], [3, 7]]))]
['R2 <- R2 - (3)*R1', 'R1 <- R1 - (2)*R2']
>>> steps = determinant_steps(ExprMatrix.from_rows([[1, a], [a, 1]]))
>>> steps.steps[0].operation, steps.diagonal, steps.value
(R2 <- R2 - (a)*R1, (1, 1 - a**2), 1 - a**2)

```

A `RowOperation` knows its elementary matrix (`operation.matrix(n)`),
so the product of the steps is the transformation.

## Derivatives

`sympy_extras.matrices.calculus` builds the matrices of derivatives:
`jacobian`, `gradient`, `hessian` (bordered by the Jacobian of
constraints when they are given) and the `wronskian` determinant, which
is reduced modulo $\sin^2 + \cos^2 = 1$ so that dependent functions give
zero. `classify_critical_point` is the second derivative test: the signs
of the eigenvalues of the Hessian at the point come from `inertia`, with
no eigenvalue computed.

```python
>>> from sympy_extras.functions import cos, exp, sin
>>> from sympy_extras.matrices.calculus import jacobian, hessian, wronskian, classify_critical_point
>>> y = Symbol('y')
>>> jacobian([x**2*y, x + y], [x, y]), hessian(x**3 + x*y**2, [x, y])
(ExprMatrix([[2*x*y, x**2], [1, 1]]), ExprMatrix([[6*x, 2*y], [2*y, 2*x]]))
>>> wronskian([cos(x), sin(x)], x), wronskian([sin(x)**2, cos(x)**2, 1], x)
(1, 0)
>>> f = x**3 + y**3 - 3*x*y
>>> classify_critical_point(f, [x, y], [1, 1]), classify_critical_point(f, [x, y], [0, 0])
('minimum', 'saddle')

```

## Not done

* eigenvalues when square roots remain in the coefficients of the
  characteristic polynomial, or with other kernels (`sin(t)`): the
  factorisation over the field they generate is needed; eigenvalues as
  `RootOf` objects when a factor is not solved in radicals;
* the singular value decomposition with parameters and the full (square
  `U`, `V`) decomposition;
* the Popov form of matrices of polynomials; a Smith form
  which controls the growth of the coefficients (modular, or by the
  determinantal divisors) for large matrices;
* the inertia when the core does not decide the sign of a pivot (a
  general real algebraic number), pivoting strategies for `ldl`
  (Bunch–Kaufman);
* case distinctions on the parameters inside these functions;
* a sparse representation, and modular or evaluation-interpolation
  methods for large matrices of polynomials.

## References

* E. H. Bareiss, *Sylvester's identity and multistep integer-preserving
  Gaussian elimination*, Mathematics of Computation 22 (1968), 565-578.
* S. J. Berkowitz, *On computing the determinant in small parallel time
  using a small number of processors*, Information Processing Letters 18
  (1984), 147-150.
* K. O. Geddes, S. R. Czapor, G. Labahn, *Algorithms for Computer
  Algebra*, Kluwer, 1992, Chapter 9.
