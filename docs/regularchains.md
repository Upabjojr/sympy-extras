# Triangular decompositions: regular chains

`sympy_extras.polys.regularchains` solves systems of polynomial equations
by *triangular decomposition*: the solutions are written as a union of
triangular systems, each of which is solved one variable after the other,
like a linear system in row echelon form.

## What SymPy has and what is added

SymPy solves polynomial systems through a lexicographic Gröbner basis
(`solve_poly_system`, `solve`), which works when the solutions are
finitely many and in general position. It gives up on a system with
infinitely many solutions, and a Gröbner basis separates neither the
components of a solution set nor the values of the parameters for which
the solutions change:

```python
>>> from sympy import groebner, solve_poly_system
>>> from sympy.abc import x, y, z
>>> groebner([x*y, x*z], x, y, z, order='lex').exprs
[x*y, x*z]
>>> try:
...     solve_poly_system([x*z - y, y*z - x], x, y, z)
... except NotImplementedError as error:
...     print(str(error).strip())
only zero-dimensional systems supported (finite number of solutions)

```

Here:

- `triangularize(equations, *symbols, inequations=..., mode=...)`: the
  decomposition of the zeros of a system into regular chains, in the sense
  of Lazard (all the zeros) or of Kalkbrener (the generic zeros of every
  component);
- `RegularChain`: a triangular system with `polys`, `main_variables`,
  `free_variables`, `initials`, `dimension`, `degree`; membership in its
  saturated ideal (`reduce`, `contains`), `is_regular`, `regularize`,
  `intersect`, `saturated_ideal`, and its isolated points exactly
  (`solutions`) and numerically (`numerical_solutions`);
- `regular_gcd(p, q, x, chain)`: greatest common divisors of polynomials
  whose coefficients are algebraic over the chain, with the case
  distinctions they need.

```python
>>> from sympy_extras.polys.regularchains import triangularize
>>> triangularize([x*y, x*z], x, y, z)
[RegularChain([x], x, y, z), RegularChain([z, y], x, y, z)]
>>> triangularize([x*z - y, y*z - x], x, y, z)
[RegularChain([y, x], x, y, z), RegularChain([z + 1, x + y], x, y, z), RegularChain([z - 1, x - y], x, y, z)]

```

A plane and a line; then three lines: `x = y = 0`, and `x = -y`, `x = y`
in the planes `z = -1` and `z = 1`.

## Regular chains

The variables are ordered, **the first one being the greatest**, as for
the lexicographic Gröbner bases of SymPy: the last variables are solved
first. A polynomial is seen as a polynomial in its greatest variable, its
*main variable*; its leading coefficient, a polynomial in the smaller
variables, is its *initial*.

A *regular chain* `T` is a set of polynomials with distinct main variables
such that the initial of each polynomial is not a zero divisor modulo the
*saturated ideal* `sat(T') = (T') : h^oo` of the polynomials `T'` below it,
where `h` is the product of their initials. It stands for its
*quasi-component* `W(T) = V(T) \ V(h)`: its zeros where no initial
vanishes. The variables which are the main variable of no polynomial are
free, their number is the dimension of `W(T)`, and for every value of them
(outside a hypersurface) the chain has `degree` solutions, found by
solving univariate polynomials one after the other.

```python
>>> from sympy_extras.polys.regularchains import RegularChain
>>> T = RegularChain([y**2 - z, (z - 1)*x**2 - y], x, y, z)
>>> T.main_variables, T.free_variables, T.initials
([y, x], [z], [1, z - 1])
>>> T.dimension, T.degree
(1, 4)

```

The zeros of the closure of `W(T)` are those of the saturated ideal, and a
polynomial is in it exactly when its pseudo-remainder by the chain is zero:
no Gröbner basis is needed to decide it.

```python
>>> T.contains((z - 1)**2*x**4 - z), T.contains(x**2 - y)
(True, False)
>>> T.reduce(x**4)
z
>>> T.saturated_ideal().reduced().exprs
[x**2*z - x**2 - y, y**2 - z]

```

(`reduce` multiplies by initials: `(z - 1)**2 x**4 = z` modulo the chain.)
The chains of this module are *squarefree*: no polynomial has a multiple
root modulo the ones below it, so that the saturated ideal is radical and
every solution is counted once. The constructor checks the definition:

```python
>>> RegularChain([y**2 - z*y, (y - z)*x - 1], x, y, z)
Traceback (most recent call last):
...
ValueError: the initial y - z is a zero divisor modulo the polynomials below

```

## The two decompositions

```python
>>> from sympy.abc import p, q
>>> f = x**3 + p*x + q
>>> triangularize([f, f.diff(x)], x, p, q, mode='kalkbrener')
[RegularChain([4*p**3 + 27*q**2, 2*p*x + 3*q], x, p, q)]
>>> triangularize([f, f.diff(x)], x, p, q)
[RegularChain([4*p**3 + 27*q**2, 2*p*x + 3*q], x, p, q), RegularChain([q, p, x], x, p, q)]

```

The cubic has a double root when its discriminant vanishes, and the root
is `-3q/(2p)`. This is the decomposition *in the sense of Kalkbrener*: the
zeros of the system are the union of the **closures** of the
quasi-components, that is the radical of the ideal of the system is the
intersection of the saturated ideals. The chain says nothing about
`p = 0`, where its initial vanishes. The decomposition *in the sense of
Lazard* (the default) adds what is missing: the zeros of the system are
exactly the union of the quasi-components.

The Kalkbrener decomposition is smaller and faster; it is the one to use
for the dimension and the generic points of the components, for radical
membership, or when the points where an initial vanishes are known not to
matter. It is computed by discarding on the way the chains with more
polynomials than the system has equations: by Krull's theorem no
component has a smaller dimension.

## Parameters

The symbols which are not given as variables are put below them, and the
decomposition discusses their values:

```python
>>> from sympy.abc import a, b, c
>>> for chain in triangularize([a*x**2 + b*x + c], x):
...     print(chain.polys, chain.initials)
[a*x**2 + b*x + c] [a]
[a, b*x + c] [1, b]
[c, b, a] [1, 1, 1]
>>> system = [a*x + y + z - 1, x + a*y + z - 1, x + y + a*z - 1]
>>> for chain in triangularize(system, x, y, z):
...     print(chain.polys, chain.free_variables)
[a - 1, x + y + z - 1] [y, z]
[a*z + 2*z - 1, a*y + 2*y - 1, a*x + 2*x - 1] [a]

```

The linear system has the solution `x = y = z = 1/(a + 2)` when the
initial `a + 2` does not vanish, a plane of solutions for `a = 1` and none
for `a = -2`: no chain contains `a + 2`.

## Inequations

```python
>>> triangularize([x**2 - y**2, x*(y - 1)], x, y, inequations=[x])
[RegularChain([y - 1, x + 1], x, y), RegularChain([y - 1, x - 1], x, y)]

```

The solutions are the points of the quasi-components where no inequation
vanishes. The chains on which an inequation vanishes identically are
removed and the others are split so that every inequation is regular
modulo every chain: it vanishes at most on a subset of smaller dimension
of the quasi-component, and nowhere on a chain without free variables.

## Isolated solutions

A chain without free variables has finitely many points, which
`numerical_solutions` computes from the bottom of the chain. The chains
being squarefree and disjoint, every solution of the system comes once,
whatever its multiplicity in the system, and the degrees add up to their
number:

```python
>>> system = [x**3 + y*z - 2, y**3 + x*z - 3, z**2 + x*y - 1]
>>> chains = triangularize(system, x, y, z)
>>> [chain.degree for chain in chains]
[18]
>>> solutions = chains[0].numerical_solutions(10)
>>> max(abs(complex(f.xreplace(s))) for f in system for s in solutions) < 1e-8
True
>>> any(s[z].is_real for s in solutions)
False
>>> system = [x**2 + y + z - 1, x + y**2 + z - 1, x + y + z**2 - 1]
>>> for chain in triangularize(system, x, y, z):
...     print(chain.polys, chain.numerical_solutions(6))
[z**2 + 2*z - 1, y - z, x - z] [{z: -2.41421, y: -2.41421, x: -2.41421}, {z: 0.414214, y: 0.414214, x: 0.414214}]
[z, y, x - 1] [{z: 0, y: 0, x: 1.00000}]
[z, y - 1, x] [{z: 0, y: 1.00000, x: 0}]
[z - 1, y, x] [{z: 1.00000, y: 0, x: 0}]

```

`solutions` gives the same points exactly, and with `real=True` the real
ones, a root object being real or not exactly. The chain is solved from the
bottom. The roots of a polynomial whose coefficients at the point found so
far are rational are root objects (`CRootOf`), in radicals where
`sympy_extras.polys.roots` writes them so; a polynomial of degree one in its
main variable gives its coordinate by a division (every coordinate but the
first one, for a system in shape position), written as a root object of its
own polynomial when it involves root objects; the roots of another polynomial,
whose coefficients are algebraic numbers, are roots of a polynomial with
rational coefficients, an iterated resultant of the chain, and they are
told from its other roots numerically: the chain being squarefree they are
simple roots, which are matched with the numerical roots of the resultant,
themselves matched with the root objects by their isolating intervals
(`NotImplementedError` when the match is not clear at the working
precision). SymPy does not solve the second system below:
`solve_poly_system` answers `[]` and `nonlinsolve` answers `{(x, -sqrt(2)),
(x, sqrt(2))}`, the unknown left free.

```python
>>> for chain in triangularize([x**2 + y + z - 1, x + y**2 + z - 1, x + y + z**2 - 1], x, y, z):
...     print(chain.solutions())
[{z: -sqrt(2) - 1, y: -sqrt(2) - 1, x: -sqrt(2) - 1}, {z: -1 + sqrt(2), y: -1 + sqrt(2), x: -1 + sqrt(2)}]
[{z: 0, y: 0, x: 1}]
[{z: 0, y: 1, x: 0}]
[{z: 1, y: 0, x: 0}]
>>> [chain] = triangularize([x**5 - x - 1 - y, y**2 - 2], x, y)
>>> len(chain.solutions())
10
>>> for solution in chain.solutions(real=True):
...     print(solution)
{y: -sqrt(2), x: CRootOf(x**10 - 2*x**6 - 2*x**5 + x**2 + 2*x - 1, 0)}
{y: -sqrt(2), x: CRootOf(x**10 - 2*x**6 - 2*x**5 + x**2 + 2*x - 1, 1)}
{y: -sqrt(2), x: CRootOf(x**10 - 2*x**6 - 2*x**5 + x**2 + 2*x - 1, 2)}
{y: sqrt(2), x: CRootOf(x**10 - 2*x**6 - 2*x**5 + x**2 + 2*x - 1, 3)}

```

`sympy_extras.assumptions.solve` solves this way the polynomial systems
with rational coefficients, no parameter and finitely many solutions. For
the other systems it substitutes the points of `nonlinsolve` back in the
equations, and a polynomial system they do not satisfy goes through the
chains too (the family of a chain with free variables is returned when its
polynomials have degree one in their main variables, the free variables
standing for themselves as in `nonlinsolve`).

## Computing modulo a chain

Modulo the saturated ideal of a squarefree regular chain the ring of
fractions is a product of fields. The algorithms compute in it as in a
field and split the chain when an element turns out to be a zero divisor
(the *D5 principle*). `regularize` makes a polynomial zero or regular:

```python
>>> S = RegularChain([y**2 - 1, x**2 - y], x, y)
>>> S.is_regular(x - 2), S.is_regular(x - 1)
(True, False)
>>> zero, regular = S.regularize(x - 1)
>>> zero
[RegularChain([y - 1, x - 1], x, y)]
>>> regular
[RegularChain([y + 1, x**2 - y], x, y), RegularChain([y - 1, x + 1], x, y)]
>>> S.intersect(x - 1)
[RegularChain([y - 1, x - 1], x, y)]

```

A *regular gcd* of two polynomials in a variable above the chain is a
polynomial with a regular leading coefficient which is in the ideal of
both and pseudo-divides both, modulo the saturated ideal; it may differ
from one part of the chain to another:

```python
>>> from sympy_extras.polys.regularchains import regular_gcd
>>> regular_gcd(x**2 - y, x**2 - 3*x + 2*y, x, RegularChain([y**2 - 1], x, y))
[(1, RegularChain([y + 1], x, y)), (x - 1, RegularChain([y - 1], x, y))]

```

## The algorithm

The decomposition is incremental ([ChenMorenoMaza]): the equations are
intersected one after the other with the chains obtained so far, with four
operations which call each other.

- `regular_gcd(p, q, v, T)` takes the subresultant chain of `p` and `q` in
  `v`, computed once over the integers, and looks for the subresultant of
  lowest index whose principal coefficient is regular modulo the chain,
  those below being zero: subresultants commute with the specializations
  which keep the degrees, and modulo a prime ideal that subresultant is the
  gcd.
- `regularize(p, T)` reduces `p` by the chain, makes its initial regular
  or zero, and computes the regular gcd `g` of `p` and the polynomial
  `T_v` of the chain with the same main variable: `T_v` is replaced by `g`
  (where `p` is zero) and by `T_v/g` (where `p` is regular, the chain
  being squarefree).
- `intersect(p, T)` regularizes `p`; where it is regular, its common zeros
  with `T_v` lie over the zeros of their resultant, which is intersected
  with the chain below, or `p` is put in the chain if `v` is free, the
  zeros of its initial being treated apart.
- `extend` puts polynomials back on top of a chain which has changed
  below them, where their initials stay regular, keeping the chain
  squarefree by the regular gcd of each polynomial with its derivative.

Every time a leading coefficient `h` is needed regular, the points of the
quasi-component where it vanishes are treated by a recursive call on
`intersect(h, T)`: this is what makes the decomposition one in the sense
of Lazard. A chain of the same height as the one it refines inherits what
is regular modulo it (both saturated ideals are unmixed of the same
dimension); on a chain of greater height nothing is inherited and the
computation is restarted, which terminates because the height is at most
the number of variables.

Chains whose quasi-component is contained in another one are removed by a
sufficient test (the other chain is in the saturated ideal, and its
initials vanish nowhere on the quasi-component), and the chains without
free variables are made disjoint.

## Verification

The tests check the decompositions against an independent oracle, Gröbner
bases (`sympy_extras.polys.ideals.Ideal`): every chain is a squarefree
regular chain (the initials and the derivatives are not zero divisors
modulo the saturated ideals, which are not the whole ring), the equations
are in every saturated ideal, the intersection of the saturated ideals is
in the radical of the system, and every zero of the system is in a
quasi-component (for every way of leaving all the quasi-components, by a
zero of an initial or a polynomial which does not vanish, the system
becomes inconsistent). `regularize`, `intersect` and `regular_gcd` are
checked against their definitions in the same way, on fixed and random
inputs; a random audit of 184 systems in two and three variables found no
difference. Published decompositions (the example of the RegularChains
library of Maple, the cyclic and Katsura systems, discriminants) are in
`tests/test_regularchains_known.py`.

The decompositions were also checked with Mathematica 12.2, which decides
by quantifier elimination over the complex numbers whether the zeros of the
system are the union of the quasi-components (`Resolve[ForAll[vars,
Equivalent[system, chains]], Complexes]`): true for the 63 systems tried
(the systems of the tests, among them those with parameters, and random
ones with components of several dimensions, 13 with inequations). In the
sense of Kalkbrener, for 35 systems: every quasi-component is in the zero
set, and the products of generators of the saturated ideals (computed by
Mathematica as elimination ideals) vanish on it (34 systems, one with too
many products). The exact solutions: for 30 systems with finitely many
solutions the number of complex solutions (`NSolve`) and the real points
(`Solve` over the reals, to nine digits) agree, as does `solve` over the
reals.

## Limitations

- The coefficients are rational numbers; parameters are variables. There
  are no algebraic number coefficients and no computation over a field of
  rational functions in the parameters (the generic case only).
- The chains are not *normalized*: the initials are not made free of the
  main variables, so that the coefficients of the polynomials at the top
  of a chain can be large.
- In the sense of Kalkbrener a chain may remain whose quasi-component is
  in the closure of another one: the test of inclusion is only sufficient.
- There is no modular or evaluation/interpolation method: the subresultant
  chains are computed over the integers by SymPy's subresultant sequence,
  and the intermediate resultants are large for systems with more than a
  handful of solutions per variable. Large polynomials are not factored
  (SymPy's factorization stalls on them), only divided by the irreducible
  factors met before.
- The exact solutions tell the roots over an algebraic point apart
  numerically (simple roots, matched at forty digits and more), not by a
  computation in the algebraic number field; a chain with free variables
  has no closed form unless its polynomials have degree one in their main
  variables, and parameters are not supported in `solve` through chains.
- The real points of a chain without free variables are selected
  (`solutions(real=True)`); semi-algebraic systems (`RealTriangularize`),
  sample points of the real components of positive dimension and the
  comprehensive triangular decomposition are not implemented.

## References

- [ChenMorenoMaza] C. Chen, M. Moreno Maza, *Algorithms for computing
  triangular decomposition of polynomial systems*, Journal of Symbolic
  Computation 47 (2012), 610-642.
- P. Aubry, D. Lazard, M. Moreno Maza, *On the theories of triangular
  sets*, Journal of Symbolic Computation 28 (1999), 105-124.
- M. Kalkbrener, *A generalized Euclidean algorithm for computing
  triangular representations of algebraic varieties*, Journal of Symbolic
  Computation 15 (1993), 143-167.
- D. Lazard, *A new method for solving algebraic systems of positive
  dimension*, Discrete Applied Mathematics 33 (1991), 147-160.
- M. Moreno Maza, *On triangular decompositions of algebraic varieties*,
  MEGA 2000.
- F. Boulier, F. Lemaire, M. Moreno Maza, *Well known theorems on
  triangular systems and the D5 principle*, Transgressive Computing 2006.
- J. Della Dora, C. Dicrescenzo, D. Duval, *About a new method for
  computing in algebraic number fields*, EUROCAL 1985.
- L. Ducos, *Optimizations of the subresultant algorithm*, Journal of Pure
  and Applied Algebra 145 (2000), 149-163.
