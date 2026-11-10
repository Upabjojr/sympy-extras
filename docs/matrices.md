# Linear algebra on matrices of expressions

`sympy_extras.matrices` has two layers:

* `DomainMatrix` (`sympy_extras.matrices.domainmatrix`): a dense matrix
  over a typed `Domain[T]` (`ZZ`, `QQ`, `GF(p)`, a polynomial ring), with
  fraction-free algorithms;
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
* With other kernels (`sin(t)`, `sqrt(2)`, `I`) a nonzero polynomial may
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

## Not done

* eigenvalues and eigenvectors (the roots of the characteristic
  polynomial), Jordan and Smith forms of symbolic matrices;
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
