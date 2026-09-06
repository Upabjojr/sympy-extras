# Cylindrical algebraic decomposition

Module: `sympy_extras.polys.cad`

A cylindrical algebraic decomposition (CAD) of $\mathbb{R}^n$ adapted to a
set of polynomials in $x_1, \ldots, x_n$ is a partition of $\mathbb{R}^n$
into finitely many connected *cells* on each of which every polynomial has a
constant sign. The decomposition is *cylindrical*: the projections of the
cells onto the first $k$ coordinates form a decomposition of $\mathbb{R}^k$
for every $k$. Each cell comes with an exact sample point, so any property
which only depends on the signs of the polynomials can be decided by looking
at finitely many points. This is the basis of the decision procedure and of
quantifier elimination for the first order theory of the real numbers
introduced by Collins.

The implementation follows the classical two phases:

1. **Projection.** Starting from a squarefree basis of the input, a
   projection operator computes polynomials in $x_1, \ldots, x_{n-1}$ whose
   sign invariance guarantees that the real roots of the input polynomials
   in $x_n$ are delineable, i.e. given by finitely many continuous non
   crossing functions over every cell. This is repeated down to univariate
   polynomials in $x_1$. Two operators are available, McCallum's (the
   default, small but requiring the polynomials to be *well-oriented*) and
   Hong's (always valid).

2. **Lifting.** The real line is decomposed at the real roots of the
   univariate polynomials into points (*sections*) and open intervals
   (*sectors*). Over the sample point of every cell the polynomials of the
   next level become univariate; their real roots split the cylinder over
   the cell into sections and sectors again. Sample points are exact: they
   are rational numbers for sectors and real algebraic numbers, kept in a
   single algebraic number field, for sections.

## Examples

The circle $x^2 + y^2 = 1$ gives 13 cells: the two sides of the vertical
lines $x = \pm 1$, the lines themselves split at the circle, and the strip
in between split by the two arcs of the circle.

```python
>>> from sympy.abc import x, y
>>> from sympy_extras.polys.cad import cylindrical_algebraic_decomposition
>>> cad = cylindrical_algebraic_decomposition([x**2 + y**2 - 1], [x, y])
>>> cad
CAD(13 cells, x, y)
>>> [cell.point for cell in cad if cell.signs == (0,)]
[(-1, 0), (0, -1), (0, 1), (1, 0)]
>>> [cell.point for cell in cad if cell.signs == (-1,)]
[(0, 0)]

```

Cells are indexed like in Collins' work: the $k$-th entry of the index is
odd for a sector and even for a section of the cylinder over the parent
cell. The dimension of a cell is the number of odd entries.

```python
>>> cell = cad.cells[7]
>>> cell.index, cell.point, cell.dimension
((3, 4), (0, 1), 1)

```

The signs of the input polynomials on each cell are read off the sample
point, so a system of polynomial equations and inequalities is satisfiable
if and only if it holds at one of the sample points:

```python
>>> from sympy_extras.polys.cad import sample_points
>>> sample_points((x**2 + y**2 < 1) & (x > y), [x, y])
[{x: 0, y: -1/2}, {x: CRootOf(2*x**2 - 1, 1), y: 0}, {x: 3/4, y: 0}]
>>> sample_points((x**2 + y**2 < 1) & (x > 1), [x, y])
[]

```

Quantifiers are eliminated by propagating the truth of a formula from the
cells of the full decomposition down to the cells of the space of the free
variables. With a single free variable the result is a union of intervals
with exact endpoints; with more free variables it is a formula in the
polynomials computed by the projection.

```python
>>> from sympy import Eq
>>> from sympy.abc import a, b, c
>>> from sympy_extras.polys.cad import quantifier_elimination, decide, solution_set
>>> quantifier_elimination(x**2 + b*x + c > 0, [('forall', x)])
b**2 - 4*c < 0
>>> quantifier_elimination(Eq(x**2 + a*x + b, 0), [('exists', x)])
a**2 - 4*b >= 0
>>> solution_set(Eq(x**2 + y**2, 1) & (y > x), x, [('exists', y)])
Interval.Ropen(-1, CRootOf(2*x**2 - 1, 1))
>>> decide(Eq(y, x**2), [('forall', x), ('exists', y)])
True
>>> decide(Eq(x, y**2), [('forall', x), ('exists', y)])
False

```

The number of cells grows quickly with the number of variables and the
degrees: this implementation is meant for problems with a few variables
and moderate degrees.

## Reference

All the names below are importable from `sympy_extras.polys.cad`. The
docstrings of the functions contain the details and more examples.

### Decomposition (`sympy_extras.polys.cad.lifting`)

- `cylindrical_algebraic_decomposition(polys, gens, method=None)`: the CAD
  of $\mathbb{R}^n$ sign-invariant for `polys`, with `gens` giving the
  order of the variables. `method` is `'mccallum'`, `'hong'` or `None`
  (try McCallum's projection, fall back to Hong's when the input is not
  well-oriented).
- `CAD`: the result, with the attributes `gens`, `polys`, `projection`
  (the projection factor sets $P_1, \ldots, P_n$), `method` and `cells`
  (the cells of $\mathbb{R}^n$ in lexicographic order of their indices),
  and the methods `cells_at(level)` and `children(cell)`.
- `CADCell`: a cell, with `index`, `point` (the exact sample point as
  SymPy expressions), `sample` (the same point as a `SamplePoint`),
  `parent`, `signs` (the signs of the input polynomials on the cell),
  `level`, `dimension` and `is_section`.
- `NotWellOriented`: raised with `method='mccallum'` when a projection
  factor vanishes identically on a cell of positive dimension.

### Formulas (`sympy_extras.polys.cad.qe`)

Formulas are Boolean combinations (`And`, `Or`, `Not`, `Implies`,
`Equivalent`, `Xor`) of polynomial equations and inequalities with
rational coefficients. Quantifier prefixes are lists of pairs
`(kind, variables)` with `kind` equal to `'exists'` or `'forall'`,
outermost first.

- `quantifier_elimination(formula, quantifiers=(), free=None, method=None)`:
  a quantifier-free formula in the free variables equivalent to the input
  over the reals, or `S.true`/`S.false` when every variable is quantified.
  With more than one free variable the result is written with sign
  conditions on the projection factors, and `NotImplementedError` is raised
  when those are not enough to describe the solution set.
- `decide(formula, quantifiers, method=None)`: the truth value of a formula
  with all its variables quantified.
- `solution_set(formula, x, quantifiers=(), method=None)`: the set of values
  of the free variable `x` for which the quantified formula holds, as a
  union of intervals and points with exact endpoints.
- `sample_points(formula, gens, method=None)`: one exact point in every
  cell on which a quantifier-free formula holds.

### Projection (`sympy_extras.polys.cad.projection`)

- `projection_sets(polys, gens, method='mccallum')`: the projection factor
  sets $P_1, \ldots, P_n$ for all levels.
- `mccallum_projection(polys, x)`: McCallum's projection of a squarefree
  basis with respect to `x` (coefficients, discriminants, pairwise
  resultants).
- `hong_projection(polys, x)`: Hong's projection (leading coefficients and
  principal subresultant coefficients of the reducta).
- `squarefree_basis(polys, gens)`: the finest squarefree basis of the input,
  split by level.

### Sample points (`sympy_extras.polys.cad.samplepoints`)

- `SamplePoint`: a point with real algebraic coordinates kept in a single
  field $\mathbb{Q}(\theta)$, built one coordinate at a time with
  `extend(root)`; `sign(poly, gens)` and `real_roots(poly, gens)` evaluate
  polynomials at the point exactly.
- `compare_real(a, b)`: exact comparison of real algebraic numbers given as
  rationals or real `CRootOf` roots.
- `rational_between(a, b)`, `rational_below(a)`, `rational_above(a)`,
  `simplest_between(lo, hi, lo_open=True, hi_open=True)`: canonical simple
  rational numbers between, below and above real algebraic numbers.

### Principal subresultant coefficients (`sympy_extras.polys.euclidtools`)

- `dup_psc(f, g, K)`, `dmp_psc(f, g, u, K)`: the principal subresultant
  coefficients of two polynomials in dense representation.
- `psc(f, g)`: the same for elements of a `PolyRing`.

## References

- G. E. Collins, *Quantifier elimination for real closed fields by
  cylindrical algebraic decomposition*, Automata Theory and Formal
  Languages, Lecture Notes in Computer Science 33, Springer, 1975,
  pp. 134-183.
- S. McCallum, *An improved projection operation for cylindrical algebraic
  decomposition*, in: Quantifier Elimination and Cylindrical Algebraic
  Decomposition, Springer, 1998, pp. 242-268.
- H. Hong, *An improvement of the projection operator in cylindrical
  algebraic decomposition*, ISSAC 1990, pp. 261-264.
