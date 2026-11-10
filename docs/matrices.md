# Linear algebra on matrices of expressions

`sympy_extras.matrices` has two layers:

* `DomainMatrix` (`sympy_extras.matrices.domainmatrix`): a dense matrix
  over a typed `Domain[T]` (`ZZ`, `QQ`, `GF(p)`, a polynomial ring), with
  fraction-free algorithms;
* `sympy_extras.matrices.eigen`: eigenvalues, eigenvectors,
  diagonalisation and the Jordan form; `sympy_extras.matrices.svd`: the
  singular value decomposition;
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
* With the imaginary unit `I` (and symbols) the arithmetic is that of the
  field $K[i]/(i^2 + 1)$ over the rational functions $K$: exact as well.

```python
>>> from sympy_extras.core import I
>>> P = ExprMatrix.from_rows([[1, I], [I, -1]])
>>> P.rank(), P.nullspace()
(1, [ExprMatrix([[-I], [1]])])
>>> ExprMatrix.from_rows([[1 + I, a], [2, 1 - I]]).inv()
ExprMatrix([[-1/(2*a - 2) + I/(2*a - 2), a/(2*a - 2)], [1/(a - 1), -1/(2*a - 2) - I/(2*a - 2)]])

```

* With other kernels (`sin(t)`, `sqrt(2)`) a nonzero polynomial may
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
| inverse, 6 x 6, entries linear in `x, y` (in lowest terms here, not in SymPy) | 0.6 | 0.3 |

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

The entries are rational functions of symbols, possibly with `I`; the
characteristic polynomial is then factored over $\mathbb{Q}(i)$:

```python
>>> Y = ExprMatrix.from_rows([[0, -I], [I, 0]])
>>> Y.eigenvects()
[(-1, 1, [ExprMatrix([[I], [1]])]), (1, 1, [ExprMatrix([[-I], [1]])])]
>>> ExprMatrix.from_rows([[2, 1 - I], [1 + I, 3]]).eigenvals()
{1: 1, 4: 1}

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

## Not done

* eigenvalues of matrices with kernels other than symbols and `I`
  (`sqrt(2)`, `sin(t)`): the
  factorisation over the field they generate is needed; eigenvalues as
  `RootOf` objects when a factor is not solved in radicals;
* the Smith and rational canonical forms of symbolic matrices; the
  singular value decomposition with parameters, the full (square `U`,
  `V`) decomposition and the pseudoinverse;
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
