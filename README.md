# sympy-extras

Extensions to [SymPy](https://www.sympy.org): algorithms built on top of
SymPy which are not (yet) part of SymPy itself.

SymPy is a large, conservative project: getting a new algorithm merged means
meeting its review standards, keeping every corner case of a stable public
API working and waiting for a release cycle. **sympy-extras** is the place
for algorithms that are useful today but do not fit that process yet: they
live here, mirror the layout of SymPy's own modules, are tested against
released SymPy versions, and may move into SymPy proper later.

## Policy

- **An extension, not a fork.** `sympy-extras` only depends on SymPy and
  adds new functionality on top of it. It does not patch or replace anything
  in SymPy.
- **A lax policy on AI-generated algorithms.** Code written with the help of
  AI models (or entirely by them) is welcome here, and a large part of the
  code in this repository was generated that way. What is required is the
  same as for any other code: a clear description of the algorithm with
  references, docstrings with examples, and tests which check the results
  against independent sources (hand computations, known results, other
  computer algebra systems). Provenance of AI-generated code is stated in
  commit messages, not hidden.
- **No support against breaking changes as of now.** The project is at
  version 0.x. Any release may rename, move or remove public functions and
  change their results; there is no deprecation policy yet. Pin the exact
  version if you depend on it. Changes are listed in
  [CHANGELOG.md](CHANGELOG.md).

## Installation

```
pip install sympy-extras

```

`sympy-extras` requires Python 3.10 or later and SymPy 1.14 or later.

To work on the code, clone the repository and install it in editable mode
with the test dependencies:

```
pip install -e ".[test]"
python -m pytest

```

The test command runs both the unit tests and the doctests in the
docstrings. The package is fully type annotated and checked with mypy in
strict mode (`pip install -e ".[dev]"` then `python -m mypy`); it ships a
`py.typed` marker.

## Contents

### Cylindrical algebraic decomposition (`sympy_extras.polys.cad`)

A cylindrical algebraic decomposition (CAD) of $\mathbb{R}^n$ adapted to a
set of polynomials in $x_1, \ldots, x_n$ is a partition of $\mathbb{R}^n$
into finitely many connected cells on each of which every polynomial has a
constant sign. Each cell comes with an exact sample point, so any property
that only depends on the signs of the polynomials can be decided by looking
at finitely many points. This is the basis of Collins' decision procedure
and quantifier elimination for the first order theory of the real numbers.

```python
>>> from sympy import Eq
>>> from sympy.abc import a, b, c, x, y
>>> from sympy_extras.polys.cad import cylindrical_algebraic_decomposition
>>> cad = cylindrical_algebraic_decomposition([x**2 + y**2 - 1], [x, y])
>>> cad
CAD(13 cells, x, y)
>>> [cell.point for cell in cad if cell.signs == (0,)]
[(-1, 0), (0, -1), (0, 1), (1, 0)]

```

Quantifier elimination, decision of closed formulas, solution sets and
sample points of systems of polynomial equations and inequalities are built
on top of the decomposition:

```python
>>> from sympy_extras.polys.cad import quantifier_elimination, decide, solution_set, sample_points
>>> quantifier_elimination(x**2 + b*x + c > 0, [('forall', x)])
b**2 - 4*c < 0
>>> quantifier_elimination(Eq(x**2 + a*x + b, 0), [('exists', x)])
a**2 - 4*b >= 0
>>> solution_set(Eq(x**2 + y**2, 1) & (y > x), x, [('exists', y)])
Interval.Ropen(-1, CRootOf(2*x**2 - 1, 1))
>>> decide(Eq(y, x**2), [('forall', x), ('exists', y)])
True
>>> sample_points((x**2 + y**2 < 1) & (x > y), [x, y])
[{x: 0, y: -1/2}, {x: CRootOf(2*x**2 - 1, 1), y: 0}, {x: 3/4, y: 0}]

```

The implementation follows the classical two phases, projection (McCallum's
operator by default, Hong's as a fallback when the input is not
well-oriented) and lifting with exact real algebraic sample points kept in a
single algebraic number field. See [docs/cad.md](docs/cad.md) for a longer
description and the API reference.

The number of cells grows quickly with the number of variables and the
degrees: the implementation is meant for problems with a few variables and
moderate degrees.

### Assumptions as mathematical statements (`sympy_extras.assumptions`)

An alternative front end to SymPy's assumptions, modelled on Mathematica's
user interface with Python names. Assumptions are written as ordinary
statements instead of predicates: `x > 0` for `Q.positive(x)`,
`element(n, S.Integers)` (that is `Contains(n, S.Integers)`) for
`Q.integer(n)`, intervals and other sets, combined with `&`, `|`, `~`, and
quantified with `ForAll` and `Exists`. SymPy's `ask`, `refine`, `simplify`
and SAT solver are the backends, together with the cylindrical algebraic
decomposition above for everything polynomial over the reals. `refine` and
`simplify` decide the signs of the symbols and of the relevant
subexpressions, hand SymPy's algorithms symbols carrying the equivalent
assumptions, and rewrite with the CAD what SymPy cannot decide.

```python
>>> from sympy import S, Abs, sqrt, Eq, log
>>> from sympy.abc import b, c, x, y, n
>>> from sympy_extras.assumptions import ask, refine, simplify, element, resolve, satisfiable, ForAll, Exists
>>> ask(x**2 - 2*x + 1 >= 0, x > 0)
True
>>> ask(x > 1, x > 2)
True
>>> refine(Abs(x - 1) + sqrt(x**2), x > 2)
2*x - 1
>>> simplify(sqrt(x**2 - 2*x + 1) + log(x) + log(y), (x > 1) & (y > 0))
x + log(x*y) - 1
>>> ask(element(n**2 + n, S.Integers), element(n, S.Integers))
True
>>> resolve(ForAll(x, x**2 + b*x + c > 0))
b**2 < 4*c
>>> resolve(Exists(y, Eq(x**2 + y**2, 1) & (y > x)))
(x >= -1) & (x < CRootOf(2*x**2 - 1, 1))
>>> satisfiable((x**2 + y**2 < 1) & (x + y > 1))
{x: 1/2, y: 2/3}

```

Relations between non-polynomial expressions of one real variable
(`sin(x) > 0` for `0 < x < pi`, `x*exp(x) > 1` for `x > 1`) are decided by
calculus (exact zeros, continuity, monotonicity, certified numerics); the
numerical checks and the solver time limits are governed by
`sympy_extras.settings`.

`refine`, `simplify`, `ask`, `assuming`/`global_assumptions`, `resolve`,
`satisfiable`, `tautology` and `find_instance` correspond to Mathematica's
`Refine`, `Simplify`, `Assuming`/`$Assumptions`, `Resolve`, `SatisfiableQ`,
`TautologyQ` and `FindInstance`. See
[docs/assumptions.md](docs/assumptions.md).

### Summation: Karr, Zeilberger, q-analogues (`sympy_extras.concrete`)

SymPy sums hypergeometric terms with Gosper's algorithm but has no
implementation of Karr's algorithm, its extension to summands containing
sums such as harmonic numbers, nested sums and their products with
factorials and powers. `karr_sum` implements it in a ΠΣ-field built from
the summand, and decides when no closed form exists in that field.

```python
>>> from sympy import harmonic, factorial
>>> from sympy.abc import k, n
>>> from sympy_extras.concrete import karr_sum, summation
>>> karr_sum(harmonic(k)**2, (k, 1, n))
n*harmonic(n)**2 - 2*n*harmonic(n) + 2*n + harmonic(n)**2 - harmonic(n)
>>> karr_sum(harmonic(k)/k, (k, 1, n))
(harmonic(n)**2 + harmonic(n, 2))/2
>>> karr_sum(k*factorial(k), (k, 1, n))
n*factorial(n) + factorial(n) - 1
>>> karr_sum(2**k/k, (k, 1, n)) is None
True

```

`summation` runs SymPy's summation first and Karr's algorithm on what is
left. See [docs/karr.md](docs/karr.md).

### Polynomial ideals and the Gröbner walk (`sympy_extras.polys.ideals`)

SymPy computes Gröbner bases and converts them with FGLM for
zero-dimensional ideals, but its ideal class leaves saturation, radicals,
primality and dimension unimplemented. The `Ideal` class adds elimination
ideals (with hashable block orders, which SymPy's product orders are not),
intersections, quotients, saturations, radical membership, the Krull
dimension, Hilbert series and polynomial, the degree, and for
zero-dimensional ideals the standard monomials, multiplication matrices,
the radical and the tests for radical, prime and maximal ideals. The
Gröbner walk converts bases between orders for ideals of any dimension.

```python
>>> from sympy.abc import x, y, z, t
>>> from sympy_extras.polys.ideals import Ideal
>>> I = Ideal([x*z - y**2, x**2 - y*z], x, y, z)
>>> I.dimension(), I.degree(), I.hilbert_series(t)
(1, 4, (t**2 + 2*t + 1)/(1 - t))
>>> I.saturate(Ideal([y], x, y, z))
Ideal([x**2 - y*z, x*y - z**2, -x*z + y**2], x, y, z)
>>> [p.as_expr() for p in I.change_order('lex')]
[x**2 - y*z, x*y**2 - y*z**2, x*z - y**2, y**4 - y*z**3]
>>> Ideal([x**2 + y**2 - 1, x - y**2], x, y).is_maximal()
True

```

See [docs/ideals.md](docs/ideals.md).

### Lie symmetries of differential equations (`sympy_extras.solvers`)

Point symmetries of ODEs, PDEs and systems from the determining equations
with a polynomial ansatz (every symmetry found is verified by
substitution), similarity reductions of PDEs to ODEs and their group
invariant solutions, and ODEs of any order solved by reduction of order in
canonical coordinates. SymPy's `dsolve` only has a `lie_group` hint for
first order equations and `pdsolve` has no symmetry analysis.

```python
>>> from sympy import Function, symbols
>>> from sympy_extras.solvers import pde_symmetries, pdsolve_lie, dsolve_lie
>>> x, t = symbols('x t')
>>> u = Function('u')(x, t)
>>> burgers = u.diff(t) + u*u.diff(x) - u.diff(x, 2)
>>> for X in pde_symmetries(burgers, u):
...     print(X.generator())
d/dt
d/dx
t*d/dx + d/du
x*d/dx + 2*t*d/dt - u*d/du
t*x*d/dx + t**2*d/dt + (-t*u + x)*d/du
>>> pdsolve_lie(u.diff(t) - u.diff(x, 2), u)[2]
Eq(u(x, t), C1 + C2*erf(x/(2*sqrt(t))))
>>> f = Function('y')(x)
>>> dsolve_lie(f.diff(x, 2) - f.diff(x)**2/f - f.diff(x)/x, f)
[Eq(y(x), exp(C1*x**2/2 + C2))]

```

`sympy_extras.assumptions.solve` is `Solve` with assumptions: polynomial
equations and inequalities in one real unknown go to the CAD, the rest to
`solveset`/`nonlinsolve` with the parameters carrying the assumptions and
the solutions filtered by `ask`. See [docs/solvers.md](docs/solvers.md).

### Exact equation solving and reduction (`resolve`, `solve`, `sympy_extras.polys.comprehensive`, `sympy_extras.solvers.integers`, `sympy_extras.solvers.transcendental`)

The algorithms of Mathematica's `Reduce`/`Resolve`/`Solve` that SymPy
lacks (see [docs/reduce.md](docs/reduce.md) for the full mapping): linear
quantifier elimination by virtual substitution (Loos–Weispfenning), used
by `resolve` before the CAD; reduction over the complex numbers with
comprehensive Gröbner systems (Kapur–Sun–Wang); linear Diophantine
systems by the Hermite normal form, minimal nonnegative solutions by
Contejean–Devie, and Presburger arithmetic by Cooper's algorithm;
transcendental equations reduced to polynomial ones through kernels with
side conditions and inverted with a database of inverse images, the
parameters carrying their assumptions.

```python
>>> from sympy import S, Eq, Mod, exp, sin, cos
>>> from sympy.abc import a, b, x, y
>>> from sympy_extras.assumptions import resolve, solve, Exists, ForAll
>>> resolve(Exists(y, Eq(a*y + b, 0) & (y > 0)))
(Eq(a, 0) & Eq(b, 0)) | ((a > 0) & (b < 0)) | ((b > 0) & (a < 0))
>>> resolve(Exists(x, Eq(x**2 + a*x + b, 0) & Eq(2*x + a, 0)), domain=S.Complexes)
Eq(a**2 - 4*b, 0)
>>> resolve(Exists(x, Eq(2*x, y)), domain=S.Integers)
Eq(Mod(y, 2), 0)
>>> solve(Eq(3*x + 5*y, 22), [x, y], (x >= 0) & (y >= 0), domain=S.Integers)
{(4, 2)}
>>> solve(Eq(sin(x) + cos(x), 1), x, (x > 0) & (x < 3))
{pi/2}
>>> solve(Eq(exp(x), a), x, domain=S.Reals)
ConditionSet(x, a > 0, {log(a)})

```

### Linear ODEs, first order PDEs, definite sums, limits (`sympy_extras.solvers`, `sympy_extras.concrete`, `sympy_extras.assumptions`)

The algorithms of Mathematica's `DSolve`, `Sum` and `Limit` that SymPy
lacks (mapping in [docs/reduce.md](docs/reduce.md)): Kovacic's
algorithm for the Liouvillian solutions of second order linear ODEs (all
three cases); polynomial, rational and hyperexponential solutions,
first order right factors and reduction of order for linear ODEs of any
order with polynomial coefficients (Abramov–Bronstein–Petkovšek,
Singer, Beke); complete integrals of first order nonlinear PDEs by
Charpit's method; Zeilberger's algorithm and Wilf–Zeilberger
certificates for definite hypergeometric sums; limits and series with
statement assumptions and case distinctions.

```python
>>> from sympy import Function, binomial, exp, oo, symbols
>>> from sympy_extras.solvers import dsolve_kovacic, dsolve_linear, complete_integral
>>> from sympy_extras.concrete import zeilberger_sum, wz_prove
>>> from sympy_extras.assumptions import limit
>>> x, y = symbols('x y')
>>> f = Function('y')(x)
>>> dsolve_kovacic(f.diff(x, 2) + f.diff(x)/x + (1 - 1/(4*x**2))*f, f)
[exp(I*x)/sqrt(x), exp(-I*x)/sqrt(x)]
>>> dsolve_linear(x*f.diff(x, 2) - (x + 2)*f.diff(x) + 2*f, f)
[x**2 + 2*x + 2, exp(x)]
>>> u = Function('u')(x, y)
>>> complete_integral(u.diff(x)*u.diff(y) - 1, u)
Eq(u(x, y), a*x + b + y/a)
>>> n, k = symbols('n k', integer=True)
>>> zeilberger_sum((-1)**k*binomial(2*n, k)**3, (k, 0, 2*n))
(-1)**n*factorial(3*n)/factorial(n)**3
>>> wz_prove(binomial(n, k)**2, binomial(2*n, n), n, k)
True
>>> limit(exp(a*x), x, oo)
Piecewise((oo, a > 0), (1, Eq(a, 0)), (0, a < 0))

```

### Linear and first order ODEs beyond `dsolve` (`sympy_extras.solvers`)

Kovacic's algorithm (`dsolve_kovacic`), polynomial, rational and
hyperexponential solutions of linear equations of any order with
exponential parts from the Newton polygon (`dsolve_linear`), Bessel,
Whittaker and hypergeometric solutions recognised through the normal
form (`special_solutions`), systems through a cyclic vector
(`dsolve_linear_system`), and Abel, Chini and d'Alembert–Lagrange first
order equations (`dsolve_first_order`). See
[docs/solvers.md](docs/solvers.md); the Kamke benchmark results are in
[benchmarks/README.md](benchmarks/README.md).

```python
>>> from sympy import Function
>>> from sympy.abc import x, a, n
>>> from sympy_extras.solvers import dsolve_linear, special_solutions
>>> y = Function('y')(x)
>>> dsolve_linear(x*y.diff(x, 2) - (x + 2)*y.diff(x) + 2*y, y)
[x**2 + 2*x + 2, exp(x)]
>>> special_solutions(x**2*y.diff(x, 2) + x*y.diff(x) + (x**2 - n**2)*y, y)
[besselj(n, x), bessely(n, x)]

```

### Thue equations, second order integrating factors, Abel invariants, polygamma series

The last items of the notes: `thue` solves Thue equations `F(x, y) = m`
completely (Baker's method with the reduction of Tzanakis and de Weger),
`dsolve_second_order` finds integrating factors and linearising
transformations of second order equations (Lie's test, the fibre
preserving construction, and the rectification of two commuting
symmetries), `abel_ode` recognises the integrable Abel classes through
their invariants, and `polygamma_series` sums Euler sums and series of
polygamma functions in zeta values.

```python
>>> from sympy import harmonic
>>> from sympy_extras.solvers import thue, dsolve_second_order
>>> from sympy_extras.concrete import polygamma_series
>>> thue(x**3 + x**2*y - 2*x*y**2 - y**3, 1, x, y)
[(-9, 5), (-1, -1), (-1, 1), (-1, 2), (0, -1), (1, 0), (2, -1), (4, -9), (5, 4)]
>>> from sympy import Function
>>> f = Function('y')(x)
>>> dsolve_second_order(f.diff(x, 2) + 3*f*f.diff(x) + f**3, f)
[Eq(y(x), 2*(C2 + x)/(2*C1 + 2*C2*x + x**2))]
>>> polygamma_series(harmonic(n)/n**2, n)
2*zeta(3)

```

### Transcendental roots, differential-algebraic equations, convergence (`sympy_extras.solvers`, `sympy_extras.concrete`)

More of the algorithms of Mathematica's notes (see
[docs/reduce.md](docs/reduce.md)): real roots of transcendental functions
isolated exactly and returned by `solve` as `TranscendentalRoot` objects
(`sympy_extras.solvers.isolation`); linear differential-algebraic
equations with constant coefficients through the core-nilpotent
decomposition of the pencil (`sympy_extras.solvers.dae`); convergence
of series and infinite products with conditions on the parameters by the
tests of d'Alembert, Raabe, Bertrand, Cauchy and Leibniz
(`sympy_extras.concrete.convergence`); Dirichlet series of the
arithmetic functions by pattern matching (`sympy_extras.concrete.dirichlet`);
sequence limits with assumptions (`limit_seq`); linear questions in `ask`
decided by virtual substitution and the parity of integer polynomials in
`refine`.

```python
>>> from sympy import cos, exp, Eq, S, Matrix, sin, mobius, Mod
>>> from sympy.abc import x, n, p
>>> from sympy_extras.assumptions import solve, refine, limit_seq, element
>>> from sympy_extras.solvers import dsolve_dae
>>> from sympy_extras.concrete import sum_convergence, dirichlet_series
>>> solve(Eq(x, cos(x)), x, domain=S.Reals)
{TranscendentalRoot(x - cos(x), x, 5/8, 3/4)}
>>> dsolve_dae(Matrix([[1, 0], [0, 0]]), Matrix([[0, -1], [1, 0]]), Matrix([0, sin(x)]), x).solution.T
Matrix([[sin(x), cos(x)]])
>>> sum_convergence(x**n/n, n)
(x >= -1) & (x < 1)
>>> sum_convergence(1/n**p, n)
p > 1
>>> dirichlet_series(mobius(n)/n**p, n)
(1/zeta(p), re(p) > 1)
>>> limit_seq(p**n, n, assumptions=(p > -1) & (p < 0))
0
>>> refine((-1)**(n**2 + n), element(n, S.Integers))
1

```

### Definite integration: the Marichev–Adamchik method, integrals over regions (`sympy_extras.integrals`)

`definite_integral` computes definite integrals without antiderivatives,
the way Mathematica and REDUCE's DEFINT do: the range is cut at the kinks
and the singularities of the integrand, mapped onto `(0, oo)`, `(0, 1)` or
`(1, oo)`, and the integral of a power of `x` times at most two functions
of a table of Mellin transforms is a Meijer G-function (Parseval's formula
for the Mellin transform), written as hypergeometric functions by Slater's
theorem. The strips of the transforms are the convergence conditions on
the parameters, decided against the assumptions or reported in a
`Piecewise`. Creative telescoping (Almkvist–Zeilberger), differentiation
under the integral sign, an antiderivative from the Risch algorithm (ported
from Aaron Meurer's unmerged SymPy pull requests) evaluated by one-sided
limits at every discontinuity, Ramanujan's master theorem and the method
of brackets come next; SymPy's `integrate` is the last resort, and its
answer is accepted only when a numerical check passes. `IntegralByRanges` integrates
over a region described by polynomial inequalities, decomposed into stacks
of intervals by the cylindrical algebraic decomposition. See
[docs/integrals.md](docs/integrals.md).

```python
>>> from sympy import symbols, exp, sin, cos, sqrt, Abs, log, oo, pi, S
>>> from sympy_extras.assumptions import element
>>> from sympy_extras.integrals import definite_integral, IntegralByRanges, mellin_transform
>>> x, y, k, s = symbols('x y k s')
>>> a, b = symbols('a b', positive=True)
>>> definite_integral(x**k/(x + 3), (x, 0, oo))
Piecewise((-3**k*pi/sin(pi*k), (k > -1) & (k < 0)), (Integral(x**k/(x + 3), (x, 0, oo)), True))
>>> definite_integral(exp(-a*x)*sin(b*x)/x, (x, 0, oo))
atan(b/a)
>>> definite_integral(exp(-a*x)*cos(k*x), (x, 0, oo), element(k, S.Reals))
a/(a**2 + k**2)
>>> definite_integral(x**k*(1 - x)**k, (x, 0, 1), k > -1)
gamma(k + 1)**2/gamma(2*k + 2)
>>> definite_integral(log(x)**2/(1 + x**2), (x, 0, oo))
pi**3/8
>>> definite_integral(x**(S(1)/3)/sqrt(-log(x)), (x, 0, 1))
sqrt(3)*sqrt(pi)/2

```

Singular integrands, kinks and trigonometric powers, where an
antiderivative evaluated at the endpoints goes wrong or does not exist:

```python
>>> definite_integral(1/(x*sqrt((x + 1)**2)), (x, -oo, -2))
-log(2)
>>> definite_integral(1/x, (x, -1, 2))
Integral(1/x, (x, -1, 2))
>>> definite_integral(Abs(x - 1)/sqrt(x), (x, 0, 2))
2*(4 - sqrt(2))/3
>>> definite_integral(sqrt(sin(x)), (x, 0, pi/2))
2*sqrt(pi)*gamma(3/4)/gamma(1/4)
>>> definite_integral(sqrt(1 - cos(x)), (x, 0, 2*pi))
4*sqrt(2)
>>> definite_integral(x*exp(x)*exp(k*x)/(exp(x) + 3), (x, -oo, oo), (k > -1) & (k < 0))
3**k*pi*(polygamma(0, -k) - polygamma(0, k + 1) - log(3))/sin(pi*k)

```

The Mellin transforms come with their strips, and the region integrals
with a case distinction on the parameters:

```python
>>> mellin_transform(1/(1 + x), x, s)
MellinTransform(gamma(s)*gamma(1 - s), (0, 1))
>>> IntegralByRanges(x*y, (x > 0) & (y > 0) & (x + y < 1)).doit()
1/24
>>> IntegralByRanges(1, x**2 + y**2 < b**2, [x, y]).doit()
pi*b**2
>>> from sympy import Eq
>>> IntegralByRanges(1, Eq(x**2 + y**2, 1), measure='hausdorff').doit()
2*pi

```

Three kernels with a trigonometric factor go through exponentials with
complex scales, `regularize=True` gives Hadamard's finite part of a
divergent Mellin-type integral, and the Fourier series of the classical
table are integrated termwise:

```python
>>> from sympy import besselj, I
>>> definite_integral(exp(-2*x)*cos(x)*besselj(0, x), (x, 0, oo))
2**(3/4)*sqrt(sqrt(2) + 2)/8
>>> definite_integral(x**(-S(3)/2)*exp(-x), (x, 0, oo), regularize=True)
-2*sqrt(pi)
>>> definite_integral(x*log(sin(x)), (x, 0, pi/2))
-pi**2*log(2)/8 + 7*zeta(3)/16

```

### Principal subresultant coefficients (`sympy_extras.polys.euclidtools`)

`dup_psc`, `dmp_psc` and `psc` compute the principal subresultant
coefficients of two polynomials, the leading coefficients of the
subresultant sequence, which are needed by Hong's projection operator.

```python
>>> from sympy import ring, ZZ
>>> from sympy_extras.polys.euclidtools import psc
>>> R, x, y = ring("x,y", ZZ)
>>> psc(x**2*y + x, x + y)
[y**3 - y, 1]

```

## What this adds to SymPy

[docs/comparison.md](docs/comparison.md) puts seventy questions to SymPy
alone and to sympy-extras on top of it, with the answers of both: 37 which
SymPy cannot answer at all, 20 it returns unevaluated, 9 it answers
partially (a truncated power series where there is a closed form, one
branch of two) and 4 where it is wrong. The examples of sympy-extras
there are doctests, and
[benchmarks/comparison.py](benchmarks/comparison.py) regenerates the whole
comparison against the installed SymPy.

## Layout

The package mirrors the layout of SymPy: code extending `sympy.polys` lives
in `sympy_extras/polys`, and so on. Tests live next to the code in `tests`
subdirectories and use the same conventions as SymPy's tests.

```
sympy_extras/
    assumptions/
        facts.py             assumptions as statements, translation to predicates and polynomials
        quantifiers.py       ForAll, Exists, prenex normal form
        context.py           assuming, global_assumptions
        ask.py               ask
        refine.py            refine, simplify
        resolve.py           resolve (quantifier elimination over the reals, integers, complexes)
        sat.py               satisfiable, tautology, find_instance
        solve.py             solve with assumptions and a domain
        limits.py            limit and series with assumptions and case distinctions
    concrete/
        pisigma.py           ΠΣ-fields and Karr's solver for first order difference equations
        karr.py              karr_sum, karr_term, summation
        zeilberger.py        Zeilberger's algorithm, WZ certificates, definite sums
        qhyper.py            q-Pochhammer symbols, q-Gosper, q-Zeilberger
        rational.py          Abramov's decomposition of rational summands
        zeilberger.py        Zeilberger's algorithm, WZ certificates, definite sums
    integrals/
        mellin.py            Mellin transforms as gamma quotients, the table of kernels
        slater.py            Mellin–Barnes integrals as Meijer G-functions, Slater's theorem
        marichev.py          the Marichev–Adamchik method over (0, oo), (0, 1), (1, oo)
        residues.py          rational, Fourier and trigonometric integrals by residues
        definite.py          definite_integral: splitting, range mappings, verified fallback
        regions.py           IntegralByRanges: integrals over semialgebraic regions (CAD)
        telescoping.py       Almkvist–Zeilberger creative telescoping (holonomic integrals)
        parametric.py        differentiation under the integral sign
        antiderivative.py    antiderivatives evaluated by one-sided limits
        brackets.py          Ramanujan's master theorem, method of brackets
        recognize.py         PSLQ recognition of integrals (an oracle)
        risch/               Risch algorithm ported from Aaron Meurer's SymPy pull requests
    solvers/
        lie.py               jet spaces, prolongation, determining equations, symmetries
        pde.py               pde_symmetries, similarity_reduction, pdsolve_lie
        ode.py               ode_symmetries, canonical_coordinates, reduce_order, dsolve_lie, solve_ode
        integers.py          Hermite normal form, Contejean-Devie, Cooper's algorithm
        transcendental.py    transcendental equations reduced to polynomial ones
        kovacic.py           Kovacic's algorithm
        linear_ode.py        polynomial/rational/hyperexponential solutions, reduction of order
        special.py           Bessel, Whittaker, hypergeometric solutions
        linear_systems.py    cyclic vector, systems Y' = A Y
        first_order.py       Abel, Chini, d'Alembert-Lagrange equations
        charpit.py           complete integrals of first order PDEs
        kovacic.py           Kovacic's algorithm (Liouvillian solutions of second order linear ODEs)
        linear_ode.py        polynomial, rational, hyperexponential solutions; reduction of order
        charpit.py           complete integrals of first order nonlinear PDEs
    polys/
        euclidtools.py       principal subresultant coefficients
        ideals.py            Ideal: elimination, saturation, dimension, Hilbert series, radicals
        groebnerwalk.py      Gröbner walk (order conversion for any ideal)
        orderings.py         WeightOrder, BlockOrder
        virtual_substitution.py  linear and quadratic quantifier elimination by virtual substitution
        comprehensive.py     comprehensive Gröbner systems, reduction over the complex numbers
        roots.py             roots in radicals through functional decomposition
        cad/
            projection.py    projection operators (McCallum, Hong)
            samplepoints.py  exact real algebraic sample points
            lifting.py       lifting phase, cylindrical_algebraic_decomposition
            qe.py            quantifier elimination and decision

```

`benchmarks/` holds the drivers which run the algorithms on external
collections (the Kamke ODEs from Maxima's test suite, classical PDE
symmetry algebras, random polynomial equations against an oracle); they
are not part of the test suite and download their data on first use.

## Releasing

Releases are published to PyPI by the `Release` GitHub Actions workflow
when a tag `vX.Y.Z` is pushed. To release:

1. Update `__version__` in `sympy_extras/__init__.py` and add a section to
   `CHANGELOG.md`.
2. Commit, then tag and push:
   ```
   git tag v0.0.1
   git push origin master v0.0.1
   ```
3. The workflow checks that the tag matches the version, runs the tests,
   builds the sdist and wheel, publishes them to PyPI with trusted
   publishing and creates a GitHub release.

The workflow needs a one-time setup: a `pypi` environment in the repository
settings, and the workflow registered as a trusted publisher for the
`sympy-extras` project on PyPI (see the comment at the top of
`.github/workflows/release.yml`). A release can also be built and uploaded
by hand with `python -m build` and `python -m twine upload dist/*`.

## License

BSD 3-Clause, see [LICENSE](LICENSE).
