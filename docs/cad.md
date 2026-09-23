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
polynomials computed by the projection (a *solution formula*, see below).

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

### Solution formulas

With several free variables the answer is a Boolean combination of sign
conditions on the projection factors of the free levels: the true cells
of the space of the free variables have sign vectors which no false cell
has, and a formula which holds on those sign vectors and on no other is
found by merging them (a greedy minimisation: cells differing in the
sign of one factor are joined, redundant conditions dropped, and covered
terms removed). The signs of the projection factors do not always tell
the true cells from the false ones: for `Exists z: z**2 = x and z > y`
the condition is `y < sqrt(x)` on `x >= 0`, the resultant `x - y**2` is
positive on both sides of the parabola and no factor vanishes on
`y = 0`. The projection is then *augmented*, after Hong (ISSAC 1992):
as long as a true and a false cell have the same sign vector, the
derivatives of the projection factors which vanish between the two (at
the first level where they lie in different cells of one stack; the
choice of Brown's thesis) are added to the projection factor sets, the
levels below are projected again, and the space of the free variables is
decomposed again, every cell of the finer decomposition taking the truth
value of the cell which contains it. Two cells of one stack are told
apart by the signs of a factor and of its derivatives of all orders
(Thom's lemma), so the rounds end; with an equational constraint the
first round adds the resultants and discriminants which the reduced
projection left out. Here the derivative `y` of `x - y**2` does it:

```python
>>> from sympy.abc import z
>>> quantifier_elimination(Eq(z**2, x) & (z > y), [('exists', z)], free=[x, y])
((x >= 0) & (y < 0)) | (x - y**2 > 0)
>>> quantifier_elimination(Eq(z**4, x) & (z > y), [('exists', z)], free=[x, y])
((x >= 0) & (y < 0)) | (x - y**4 > 0)

```

The formula is verified against `decide` on the formula with rational
values put for the free variables, at random points (hundreds of points
per test formula, and on random formulas with two free and one or two
bound variables of degree two). When the rounds do not end (they should,
by Thom's lemma; a bound of sixteen guards against a mistake), the answer
is written with the root functions of the cylindrical description below.

### Cylindrical descriptions

The set where a formula holds is a union of cells, and a cell is
cylindrical: its first coordinate lies between two real algebraic numbers
(or is one), its second coordinate between two consecutive real roots of
the projection polynomials of level two over the point below (or is one
of them), and so on. The $k$-th real root of a projection polynomial is a
continuous function on the cell below (delineability), so that the true
cells are described by bounds which are *root functions*: this is the
output of Mathematica's `CylindricalDecomposition` and `Reduce`, and a
description of the solution set for any number of free variables which
needs no augmented projection (`quantifier_elimination` writes its answer
this way only when the augmentation does not end).

```python
>>> from sympy.abc import z
>>> from sympy_extras.polys.cad import cylindrical_formula, cylindrical_set, IndexedRoot
>>> cylindrical_formula(x**2 + y**2 <= 1, [x, y])
(x >= -1) & (x <= 1) & (y <= sqrt(1 - x**2)) & (y >= -sqrt(1 - x**2))
>>> cylindrical_formula(Eq(z**2, x) & (z > y), [x, y], [('exists', z)])
(x >= 0) & (y < sqrt(x))
>>> cylindrical_set((x**2 + y**2 <= 1) & (x + y >= 1), [x, y])
ConditionSet((x, y), (x >= 0) & (x <= 1) & (y >= 1 - x) & (y <= sqrt(1 - x**2)), ProductSet(Reals, Reals))

```

A root function is written explicitly when the polynomial has degree one
or two in its variable on the cell (its coefficients have constant signs
there, which tell the degree and the branch of the quadratic formula), and
as `IndexedRoot(f, t, k)` otherwise: the $k$-th distinct real root of $f$
in $t$, from 0, the counterpart of Mathematica's parametric `Root`. It
becomes a number when the other symbols are given values:

```python
>>> cylindrical_formula((y**3 - 3*y + x > 0) & (x > 2), [x, y])
(x > 2) & (y > IndexedRoot(x + y**3 - 3*y, y, 0))
>>> IndexedRoot(y**3 - 3*y + x, y, 0).subs(x, 3).evalf(10)       # a CRootOf
-2.103803403

```

Consecutive cells of a stack with the same description of their higher
coordinates are joined, and a section joins a neighbour whose description,
at the section, is its own (the closed disc above is one piece: at
`x = -1` and `x = 1` the two bounds of `y` meet, and what the open
interval of `x` says there is the point which the section is; a bound
written with an `IndexedRoot` is not extended to the boundary this way,
since roots may meet there and the index changes). At a section which is a
number the higher bounds are evaluated. `cylindrical_set` returns the
points with numerical coordinates as a finite set and the rest as a
`ConditionSet`.

Verified on random formulas: the description has the truth value of the
formula at the sample point of every cell of the decomposition and at the
points of a rational grid (three batches of random formulas in two and
three variables, of degree up to four in the last one), and Mathematica
proves the equivalence of the formula and its description
(`Resolve[ForAll[vars, Equivalent[...]], Reals]`) for 60 random formulas
whose bounds are explicit.

### Partial decompositions

The functions on formulas (`decide`, `quantifier_elimination`,
`solution_set`, `sample_points`, `truth_tables`, `cylindrical_formula`,
`cylindrical_set`, `cylindrical_cases`) build a *partial* decomposition, after Collins and
Hong: the stacks are lifted one at a time, on demand, and a stack is only
built when the truth value of the formula over its base cell is not yet
known. The formula is evaluated as soon as the signs of the projection
factors of the levels reached determine it (trial evaluation, in
three-valued logic), an `exists` is settled by one true cell of the
stack and a `forall` by one false one, the sectors being tried first
(their sample points are rational, lifting over a section takes an
algebraic extension). The space of the free variables is decomposed as
before (more coarsely with an equational constraint, see below), so
that the solution formulas are the same. The signs of the
projection factors on a stack are read off the real roots and their
multiplicities over the base cell, so that a sample point is only
computed in its algebraic field when a stack is built over it.

`partial=False` builds the full decomposition and propagates the truth
values from its top cells, as `cylindrical_algebraic_decomposition`
always does: it is the reference implementation, kept for checking, and
gives the same answers.

When the formula implies that a polynomial vanishes (it is an equation,
or an equation is one of the terms of its conjunction) and that
polynomial has positive degree in the last variable, which is
quantified, the projection of the last level is McCallum's reduced one
(ISSAC 1999): the coefficients and discriminants of the factors of the
constraint, and their resultants with the other polynomials, instead of
the discriminants of the others and all the pairwise resultants. The
other polynomials are then sign-invariant on the sections of the
constraint, where the formula can hold, and not on the sectors, where
it is false. Fewer polynomials are projected, so the space of the free
variables is decomposed more coarsely and a solution formula may be
written with fewer factors: it describes the same set. `Exists x, y:
x**2 + y**2 = 1 and y = a*x + b` takes 646 cells instead of 3638.

The real roots of a polynomial over a sample point with algebraic
coordinates are found among the real roots of its norm (a polynomial
over the rationals, whose roots are `CRootOf` objects): each candidate
is kept or dropped by counting the roots of the polynomial in its
isolating interval with a Sturm sequence over the field, whose signs are
decided exactly by interval arithmetic on the generator. The roots of
the polynomials of a stack are then merged by their isolating intervals,
and two roots are compared exactly (by refining the intervals) only when
the intervals overlap. On a benchmark of 95 random and classical
formulas in two and three variables of degree two and three, these
changes together take the time from 951 to 53 CPU seconds, the number
of formulas which a minute does not settle from 11 to 0, and, on the 84
formulas answered by both, the time from 293 to 13 seconds and the
number of cells from 29,600 to 16,800; the answers are the same.

The liftings of the last 64 questions are kept, keyed on the projection
factor sets, the variables and the projection operator: a question about
the same polynomials (the theory checks of `satisfiable`, the handlers
of `refine` and `ask` do this many times) reuses the stacks built so far.
A stack is the same whenever it is built, so the answers do not depend
on the questions asked before.

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
- `Lifting(projection, gens, method)`: the lifting phase one stack at a
  time: `stack(cell)` builds (once) the cells over `cell`, `lift_all()`
  every stack, `levels()` lists the cells built so far and `lifted`
  counts them; `lifting_for(projection, gens, method)` is the lifting
  kept from an earlier question with the same sets, or a new one.
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

- `quantifier_elimination(formula, quantifiers=(), free=None, method=None, partial=True)`:
  a quantifier-free formula in the free variables equivalent to the input
  over the reals, or `S.true`/`S.false` when every variable is quantified.
  With more than one free variable the result is written with sign
  conditions on the projection factors, augmented by Hong's method with
  the derivatives of the factors when those are not enough to describe
  the solution set (and with their root functions, `cylindrical_formula`,
  should the augmentation not end).
- `decide(formula, quantifiers, method=None, partial=True)`: the truth
  value of a formula with all its variables quantified.
- `solution_set(formula, x, quantifiers=(), method=None, partial=True)`:
  the set of values of the free variable `x` for which the quantified
  formula holds, as a union of intervals and points with exact endpoints.
- `sample_points(formula, gens, method=None, partial=True)`: one exact
  point in every cell on which a quantifier-free formula holds.

`partial=False` builds the full decomposition instead of the partial one
(see above).

### Cylindrical descriptions (`sympy_extras.polys.cad.cylindrical`)

- `cylindrical_formula(formula, gens, quantifiers=(), method=None, partial=True)`: the
  set of the points `gens` at which the (quantified) formula holds, as a
  disjunction of conjunctions which bound the first variable by numbers,
  the second by root functions of the first, and so on.
- `cylindrical_set(formula, gens, method=None, partial=True)`: the same as a set, the
  points with numerical coordinates in a `FiniteSet` and the rest in a
  `ConditionSet`.
- `cylindrical_cases(formula, parameters, unknowns, method=None, partial=True)`: the real
  solutions in the unknowns for every real value of the parameters, as
  pairs of a cylindrical condition on the parameters and the set of the
  solutions under it (a finite set, a union of intervals for one unknown,
  a `ConditionSet` otherwise); the cases with one set are joined.
- `IndexedRoot(f, t, k)`: the `k`-th distinct real root (from 0) of the
  polynomial `f` in `t`, a function of the other symbols of `f`.

### Projection (`sympy_extras.polys.cad.projection`)

- `projection_sets(polys, gens, method='mccallum', equational=None)`: the
  projection factor sets $P_1, \ldots, P_n$ for all levels; `equational`
  is one of `polys` which the formula implies to vanish, for the reduced
  projection of the last level.
- `augmented_projection_sets(projection, gens, extra, method='mccallum')`:
  the sets with the factors of the polynomials `extra` added at their
  levels and the levels below projected again (every set contains the
  one it comes from), for the solution formulas.
- `mccallum_projection(polys, x, equational=())`: McCallum's projection
  of a squarefree basis with respect to `x` (coefficients, discriminants,
  pairwise resultants); with `equational`, the factors of an equational
  constraint among `polys`, the reduced projection of McCallum (1999).
- `hong_projection(polys, x)`: Hong's projection (leading coefficients and
  principal subresultant coefficients of the reducta).
- `squarefree_basis(polys, gens)`: the finest squarefree basis of the input,
  split by level.

### Sample points (`sympy_extras.polys.cad.samplepoints`)

- `SamplePoint`: a point with real algebraic coordinates kept in a single
  field $\mathbb{Q}(\theta)$, built one coordinate at a time with
  `extend(root)`; `sign(poly, gens)` and `real_roots(poly, gens)` evaluate
  polynomials at the point exactly, and `specialization(poly, gens)` gives
  the degree, the sign of the leading coefficient and the real roots with
  their multiplicities of a polynomial in one more variable over the point
  (a `Specialization`, kept on the point).
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
- G. E. Collins, H. Hong, *Partial cylindrical algebraic decomposition
  for quantifier elimination*, J. Symbolic Comput. 12 (1991), pp. 299-328.
- S. McCallum, *On projection in CAD-based quantifier elimination with
  equational constraint*, ISSAC 1999, pp. 145-149.
- H. Hong, *Simple solution formula construction in cylindrical algebraic
  decomposition based quantifier elimination*, ISSAC 1992, pp. 177-188.
- C. W. Brown, *Solution formula construction for truth invariant CAD's*,
  PhD thesis, University of Delaware, 1999.
- G. E. Collins, R. Loos, *Real zeros of polynomials*, in: Computer
  Algebra: Symbolic and Algebraic Computation, Springer, 1982, pp. 83-94.
