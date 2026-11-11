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

A set in several variables is described cylindrically, the first variable
between numbers, the second between root functions of the first, and so
on, as Mathematica's `CylindricalDecomposition` and `Reduce` do (explicit
roots up to degree two, `IndexedRoot` beyond; `solve` answers systems of
inequalities in several unknowns this way):

```python
>>> from sympy_extras.polys.cad import cylindrical_formula
>>> cylindrical_formula((x**2 + y**2 <= 1) & (x + y >= 1), [x, y])
(x >= 0) & (x <= 1) & (y >= 1 - x) & (y <= sqrt(1 - x**2))

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
>>> from sympy_extras.assumptions.ask import ask
>>> from sympy_extras.assumptions.refine import refine, simplify
>>> from sympy_extras.assumptions.facts_sympy import element
>>> from sympy_extras.assumptions.resolve import resolve
>>> from sympy_extras.assumptions.sat import satisfiable
>>> from sympy_extras.assumptions.quantifiers_sympy import ForAll, Exists
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
>>> from sympy_extras.core import symbols
>>> from sympy_extras.functions import harmonic, factorial
>>> from sympy_extras.concrete.karr import karr_sum, summation
>>> k, n = symbols('k n')
>>> karr_sum(harmonic(k)**2, (k, 1, n))
n*harmonic(n)**2 - 2*n*harmonic(n) + 2*n + harmonic(n)**2 - harmonic(n)
>>> karr_sum(harmonic(k)/k, (k, 1, n))
(harmonic(n)**2 + harmonic(n, 2))/2
>>> karr_sum(k*factorial(k), (k, 1, n))
n*factorial(n) + factorial(n) - 1
>>> karr_sum(2**k/k, (k, 1, n)) is None
True

```

`summation` runs SymPy's summation (ported) first and Karr's algorithm on what is
left. Definite sums whose summand depends on the upper limit are evaluated
by creative telescoping in the same ΠΣ-fields (`definite_sum`, Schneider's
extension of Zeilberger's algorithm), with harmonic numbers in the summand:

```python
>>> from sympy_extras.core import symbols
>>> from sympy_extras.functions import binomial, harmonic
>>> from sympy_extras.concrete.creative import definite_sum
>>> k, n = symbols('k n')
>>> definite_sum(binomial(n, k)**2*harmonic(k), (k, 0, n))
(2*harmonic(n) - harmonic(2*n))*binomial(2*n, n)

```

See [docs/karr.md](docs/karr.md).

### Polynomial ideals and the Gröbner walk (`sympy_extras.polys.ideals`)

SymPy computes Gröbner bases and converts them with FGLM for
zero-dimensional ideals, but its ideal class leaves saturation, radicals,
primality and dimension unimplemented. The `Ideal` class adds elimination
ideals (with hashable block orders, which SymPy's product orders are not),
intersections, quotients, saturations, radical membership, the Krull
dimension, Hilbert series and polynomial (the numerator by Bigatti's
algorithm, which handles hundreds of generators where the textbook
recursion takes minutes), the degree, and for zero-dimensional ideals
the standard monomials, multiplication matrices and the rational
univariate representation of Rouillier (every zero as rational functions
of one algebraic number, the real zeros as the real roots of one
polynomial); in any dimension, through regular chains, the radical, its
equidimensional parts, the minimal primes and the tests for radical and
prime ideals, and the primary decomposition with the embedded primes
(Gianni–Trager–Zacharias). The Gröbner walk converts bases between orders for ideals of
any dimension.
The real solutions of a zero-dimensional system, and the signs of other
polynomials at them, are counted without solving the system by the
signature of Hermite's quadratic form on the quotient algebra
(`sympy_extras.polys.hermite`, Pedersen–Roy–Szpirglas, with the sign
determination of Ben-Or–Kozen–Reif); `satisfiable` uses it to prove that
such a system has no real solution before trying a CAD.
The bases are computed by Faugère's F4 (`sympy_extras.polys.f4`: the
S-polynomials of many critical pairs reduced at once by one sparse row
reduction, two to twenty times faster than SymPy's Buchberger on the
Katsura and cyclic systems, the same reduced basis) and, over the
rationals, those which the direct computation does not finish in a quarter
of a second by the modular algorithm of Arnold
(`sympy_extras.polys.modulargroebner`): modulo large primes, on the
homogenised input, with a verification which proves the result. Strong
Gröbner bases over the integers (`sympy_extras.polys.stronggroebner`)
decide membership in ideals of `ZZ[x]`, where SymPy's `groebner` with
`domain=ZZ` gives the basis over the rationals: `<2x, 3y>` contains `xy`
and neither `x` nor `y`.

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
>>> Ideal([x**2, x*y], x, y).primary_decomposition()
[(Ideal([x], x, y), Ideal([x], x, y)), (Ideal([x**2, y], x, y), Ideal([x, y], x, y))]
>>> from sympy_extras.polys.hermite import count_real_solutions, sign_determination
>>> count_real_solutions([x**2 + y**2 - 1, x - y], x, y)
2
>>> sign_determination([x**2 - 1, y**2 - 1], [x + y, x], x, y)
{(-1, -1): 1, (0, -1): 1, (0, 1): 1, (1, 1): 1}
>>> from sympy import QQ, ring, lex, symbols
>>> u, v, w, a, b = symbols('u v w a b')
>>> from sympy_extras.polys.domains import QQ as TQQ, ZZ as TZZ
>>> from sympy_extras.polys.rings import PolyRing, lex as typed_lex
>>> from sympy_extras.polys.f4 import f4
>>> from sympy_extras.polys.modulargroebner import modular_groebner
>>> T = PolyRing(TQQ, ['u', 'v', 'w'], typed_lex)
>>> from sympy_extras.polys.interop import ring_from_sympy, ring_to_sympy
>>> F = [ring_from_sympy(T, e) for e in (u**2 + v**2 + w**2 - 1, u*v - w/3, u + v - 2*w)]
>>> [ring_to_sympy(g) for g in f4(F, T)]
[u + v - 2*w, v**2 - 2*v*w + w/3, w**2 - 2*w/15 - 1/5]
>>> R, U, V, W = ring("u,v,w", QQ, lex)
>>> modular_groebner([U**2 + V**2 + W**2 - 1, U*V - W/3, U + V - 2*W], R)
[u + v - 2*w, v**2 - 2*v*w + 1/3*w, w**2 - 2/15*w - 1/5]
>>> from sympy_extras.polys.stronggroebner import strong_groebner
>>> Z = PolyRing(TZZ, ['a', 'b'], typed_lex)
>>> [str(g) for g in strong_groebner([ring_from_sympy(Z, 2*a), ring_from_sympy(Z, 3*b)], Z)]
['a*b', '2*a', '3*b']

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
>>> from sympy_extras.solvers.pde import pde_symmetries, pdsolve_lie
>>> from sympy_extras.solvers.ode import dsolve_lie
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
>>> from sympy_extras.assumptions.resolve import resolve
>>> from sympy_extras.assumptions.solve import solve
>>> from sympy_extras.assumptions.quantifiers_sympy import Exists, ForAll
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
>>> from sympy_extras.solvers.kovacic import dsolve_kovacic
>>> from sympy_extras.solvers.linear_ode import dsolve_linear
>>> from sympy_extras.solvers.charpit import complete_integral
>>> from sympy_extras.concrete.zeilberger import zeilberger_sum, wz_prove
>>> from sympy_extras.assumptions.limits import limit
>>> x, y = symbols('x y')
>>> f = Function('y')(x)
>>> dsolve_kovacic(f.diff(x, 2) + f.diff(x)/x + (1 - 1/(4*x**2))*f, f)
[exp(I*x)/sqrt(x), exp(-I*x)/sqrt(x)]
>>> dsolve_linear(x*f.diff(x, 2) - (x + 2)*f.diff(x) + 2*f, f)
[x**2 + 2*x + 2, exp(x)]
>>> u = Function('u')(x, y)
>>> complete_integral(u.diff(x)*u.diff(y) - 1, u)
Eq(u(x, y), a*x + b + y/a)
>>> from sympy_extras.core import Symbol as CoreSymbol
>>> from sympy_extras.functions import binomial as core_binomial
>>> n, k = CoreSymbol('n', integer=True), CoreSymbol('k', integer=True)
>>> zeilberger_sum((-1)**k*core_binomial(2*n, k)**3, (k, 0, 2*n))
(-1)**n*factorial(3*n)/factorial(n)**3
>>> wz_prove(core_binomial(n, k)**2, core_binomial(2*n, n), n, k)
True
>>> limit(exp(a*x), x, oo)
Piecewise((oo, a > 0), (1, Eq(a, 0)), (0, a < 0))

```

### Linear and first order ODEs beyond `dsolve` (`sympy_extras.solvers`)

Kovacic's algorithm (`dsolve_kovacic`), polynomial, rational and
hyperexponential solutions of linear equations of any order with
exponential parts from the Newton polygon (`dsolve_linear`), Bessel,
Whittaker and hypergeometric solutions recognised through the normal
form (`special_solutions`), factorisation of linear differential
operators by Beke's algorithm (`factor_operator`, used by
`dsolve_linear` for equations of order three and more), systems
`Y' = A Y + b` through a cyclic vector, with rational solutions by
Barkatou's method and variation of constants (`dsolve_system`), Abel,
Chini and d'Alembert–Lagrange first order equations
(`dsolve_first_order`), and first order equations `y' = P(x, y)/Q(x, y)`
with polynomial `P`, `Q` solved by the Prelle–Singer procedure
(`prelle_singer`): the Darboux polynomials and exponential factors of the
field up to a degree bound, a Darboux integrating factor from a linear
system for their exponents, and the first integral by quadrature
(`darboux_polynomials`, `darboux_integrating_factor`,
`darboux_first_integral`). See
[docs/solvers.md](docs/solvers.md); the Kamke benchmark results are in
[benchmarks/README.md](benchmarks/README.md).

```python
>>> from sympy import Function
>>> from sympy.abc import x, a, n
>>> from sympy_extras.solvers.linear_ode import dsolve_linear
>>> from sympy_extras.solvers.special import special_solutions
>>> y = Function('y')(x)
>>> dsolve_linear(x*y.diff(x, 2) - (x + 2)*y.diff(x) + 2*y, y)
[x**2 + 2*x + 2, exp(x)]
>>> special_solutions(x**2*y.diff(x, 2) + x*y.diff(x) + (x**2 - n**2)*y, y)
[besselj(n, x), bessely(n, x)]
>>> from sympy_extras.solvers.darboux import prelle_singer
>>> prelle_singer(x*(1 - y)*y.diff(x) - y*(x - 1), y)          # Lotka–Volterra
Eq(-x - y(x) + log(x) + log(y(x)), C1)

```

Linear recurrences with polynomial coefficients are solved beyond SymPy's
`rsolve` (`sympy_extras.solvers.recurr`): Petkovšek's hypergeometric
solutions and the d'Alembertian solutions of Abramov and Petkovšek, the
nested sums of hypergeometric terms, found by reduction of order; the
harmonic numbers, the derangement numbers and `y(n + 1) = (n + 1) y(n) + 1`,
on which `rsolve` returns nothing or misses a solution, are solved, with
particular solutions for hypergeometric right-hand sides and initial
values. The module is native (written on the core, without SymPy).

```python
>>> from sympy_extras.core import Eq, Function, Symbol
>>> from sympy_extras.solvers.recurr import rsolve_linear
>>> n = Symbol('n')
>>> u = Function('u')
>>> rsolve_linear(u(n + 1) - (n + 1)*u(n) - 1, u(n))
C0*factorial(n) + factorial(n)*Sum(1/factorial(k + 1), (k, 0, n - 1))
>>> rsolve_linear(Eq((n + 2)*u(n + 2), (2*n + 3)*u(n + 1) - (n + 1)*u(n)), u(n), {u(0): 0, u(1): 1})
harmonic(n)
>>> rsolve_linear(u(n + 1) - n*(u(n) + u(n - 1)), u(n), {u(0): 1, u(1): 0})   # derangements
-factorial(n)*Sum((-1)**k/factorial(k + 1), (k, 0, n - 1)) + factorial(n)

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
>>> from sympy_extras.solvers.thue_equation import thue
>>> from sympy_extras.solvers.second_order import dsolve_second_order
>>> from sympy_extras.concrete.eulersums import polygamma_series
>>> thue(x**3 + x**2*y - 2*x*y**2 - y**3, 1, x, y)
[(-9, 5), (-1, -1), (-1, 1), (-1, 2), (0, -1), (1, 0), (2, -1), (4, -9), (5, 4)]
>>> from sympy import Function
>>> f = Function('y')(x)
>>> dsolve_second_order(f.diff(x, 2) + 3*f*f.diff(x) + f**3, f)
[Eq(y(x), 2*(C2 + x)/(2*C1 + 2*C2*x + x**2))]
>>> from sympy_extras.core import Symbol
>>> from sympy_extras.functions import harmonic
>>> m = Symbol('m')
>>> polygamma_series(harmonic(m)/m**2, m)
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
>>> from sympy_extras.assumptions.solve import solve
>>> from sympy_extras.assumptions.refine import refine
>>> from sympy_extras.assumptions.limits import limit_seq
>>> from sympy_extras.assumptions.facts_sympy import element
>>> from sympy_extras.solvers.dae import dsolve_dae
>>> from sympy_extras.concrete.convergence import sum_convergence
>>> solve(Eq(x, cos(x)), x, domain=S.Reals)
{TranscendentalRoot(x - cos(x), x, 5/8, 3/4)}
>>> dsolve_dae(Matrix([[1, 0], [0, 0]]), Matrix([[0, -1], [1, 0]]), Matrix([0, sin(x)]), x).solution.T
Matrix([[sin(x), cos(x)]])
>>> sum_convergence(x**n/n, n)
(x >= -1) & (x < 1)
>>> sum_convergence(1/n**p, n)
p > 1
>>> from sympy_extras.interop import expr_from_sympy
>>> from sympy_extras.concrete.dirichlet import dirichlet_series
>>> dirichlet_series(expr_from_sympy(mobius(n)/n**p), expr_from_sympy(n))
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
>>> from sympy_extras.assumptions.facts_sympy import element
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

### A zero test for elementary expressions (`sympy_extras.simplify`)

`is_zero`, `equal` and `canonical_form` decide identities between
elementary expressions by the Risch–Rosenlicht structure theorem: the
expression is written as a rational function of exponentials, logarithms
and roots which are proven algebraically independent, so it vanishes
exactly when its numerator does. `True` is a proof, and the branches are
part of the question: an identity between logarithms or roots holds on the
region of the assumptions.

```python
>>> from sympy import symbols, exp, log, sqrt, sin, cos, atan, pi, Rational
>>> from sympy_extras.simplify.structure import is_zero, canonical_form
>>> x, y = symbols('x y')
>>> is_zero(sin(x + y) - sin(x)*cos(y) - cos(x)*sin(y)), is_zero(exp(x) - x - 1)
(True, False)
>>> is_zero(log(x**2) - 2*log(x)), is_zero(log(x**2) - 2*log(x), x > 0)
(False, True)
>>> is_zero(atan(x) + atan(1/x) - pi/2, x > 0)
True
>>> is_zero(pi/4 - 4*atan(Rational(1, 5)) + atan(Rational(1, 239)))
True
>>> canonical_form((sqrt(x) + 1)**2 - x - 1)
2*sqrt(x)

```

SymPy's `simplify` leaves Machin's formula as it is and its `equals` samples
it. See [docs/simplify.md](docs/simplify.md).

### Differential elimination: Janet bases and Rosenfeld–Gröbner (`sympy_extras.polys.differential`)

The differential counterpart of Gröbner bases. `janet_basis` completes a
linear system of partial differential equations with all its
integrability conditions: it decides consistency and the membership of a
consequence, and counts the solutions without solving. `rosenfeld_groebner`
splits a polynomial system of ordinary or partial differential equations
into regular differential systems, which separates the general solution
from the singular ones, finds the hidden constraints of
differential-algebraic systems and eliminates unknowns.

```python
>>> from sympy import Function, symbols
>>> from sympy_extras.polys.differential import janet_basis, rosenfeld_groebner
>>> x, y, z = symbols('x y z')
>>> u = Function('u')(x, y, z)
>>> J = janet_basis([u.diff(z, 2) + y*u.diff(x, 2), u.diff(y, 2)], [u])
>>> J.dimension, J.contains(u.diff(z, 4))
(12, True)
>>> p = Function('p')(x)
>>> I = rosenfeld_groebner([p.diff(x)**2 - 4*p], [p])
>>> [c.equations for c in I.components]
[[-4*p(x) + Derivative(p(x), x)**2], [p(x)]]
>>> I.components[0].contains(p.diff(x, 2) - 2)
True

```

The size of the symmetry algebra of a differential equation follows from
the Janet basis of its determining equations:

```python
>>> from sympy_extras.solvers.lie import symmetry_janet_basis
>>> symmetry_janet_basis(p.diff(x, 2), p).dimension
8

```

See [docs/differential.md](docs/differential.md).

### Triangular decompositions: regular chains (`sympy_extras.polys.regularchains`)

`triangularize` solves a system of polynomial equations by writing its
zeros as a union of *regular chains*: triangular systems which are solved
one variable after the other. Unlike a lexicographic Gröbner basis it works
in any dimension, separates the components, and discusses the parameters
(the symbols put last); `solve_poly_system` refuses the first system below.

```python
>>> from sympy.abc import a, b, c, p, q, x, y, z
>>> from sympy_extras.polys.regularchains import triangularize
>>> triangularize([x*z - y, y*z - x], x, y, z)
[RegularChain([y, x], x, y, z), RegularChain([z + 1, x + y], x, y, z), RegularChain([z - 1, x - y], x, y, z)]
>>> for chain in triangularize([a*x**2 + b*x + c], x, a, b, c):
...     print(chain.polys, chain.initials)
[a*x**2 + b*x + c] [a]
[a, b*x + c] [1, b]
[c, b, a] [1, 1, 1]

```

The solutions of a chain are its zeros where no initial (leading
coefficient) vanishes. The decomposition in the sense of Lazard (the
default) describes all the zeros; the one in the sense of Kalkbrener only
the generic points of every component, here the double roots of a cubic:

```python
>>> f = x**3 + p*x + q
>>> triangularize([f, f.diff(x)], x, p, q, mode='kalkbrener')
[RegularChain([4*p**3 + 27*q**2, 2*p*x + 3*q], x, p, q)]

```

`RegularChain` decides the membership in its saturated ideal by
pseudo-division (`contains`, `reduce`), splits itself where a polynomial is
a zero divisor (`regularize`, `intersect`, `regular_gcd`: the D5 principle)
and computes its isolated solutions, exactly (`solutions`, the real ones
with `real=True`) and numerically (`numerical_solutions`); `solve` uses
them for the polynomial systems `nonlinsolve` answers wrongly. See
[docs/regularchains.md](docs/regularchains.md).

### An experimental core: unevaluating classes, evaluating functions (`sympy_extras.core`)

A proposal for a replacement of `sympy.core`, written from scratch: the
classes build exactly the node they are asked for (`Add(x, x)` has the
arguments `(x, x)`), so that their types are honest and the package is
checked by mypy in strict mode; the evaluation of SymPy's constructors is
done by functions of the same name in lower case (`add(x, x)` is `2*x`),
which the operators call and which `doit` applies to a tree. Everything
else keeps SymPy's names, evaluated forms, assumptions and printing: the
pattern matching (`Wild`, `match`, `replace`), `Mod`, `Subs`, `Order`,
the full set of `is_*` predicates, the connectives (`And`, `Or`, `Not`,
... with `and_`, `or_`, `not_`, ...), `Q` and `ask` (conjunctive
assumptions answered by the old assumptions, the rest by SymPy's solver),
`refine`, the methods of `Expr` (the structural ones written here,
`series`, `simplify`, `integrate`, ... through the conversion to SymPy),
`factor_terms`, the order of the arguments of sums and products, the
context managers, the cache, the traversals and the integer helpers;
`to_sympy`/`from_sympy` convert in both directions, every object of SymPy
included; the conventions of the methods (`srepr`, `sort_key`, `evalf`,
`count_ops`, `extract_multiplicatively`, ...) are SymPy's, checked on
random expressions, and the arithmetic runs within a factor of two of
`sympy.core`. See `docs/core.md`.

```python
>>> from sympy_extras.core import Symbol, Add, add, sqrt, Eq, eq
>>> from sympy_extras.interop import to_sympy
>>> x = Symbol('x')
>>> Add(x, x), Add(x, x).args, add(x, x), x + x
(x + x, (x, x), 2*x, 2*x)
>>> Eq(x, x), eq(x, x), sqrt(8)*sqrt(2), (x**2 + 2*x + 1).diff(x)
(Eq(x, x), True, 4, 2*x + 2)
>>> Add(x, 1, x).doit(), to_sympy(Add(x, x)), to_sympy(Add(x, x), evaluate=False)
(2*x + 1, 2*x, x + x)

```

### Principal subresultant coefficients (`sympy_extras.polys.subresultants`)

`dup_psc`, `dmp_psc` and `psc` compute the principal subresultant
coefficients of two polynomials, the leading coefficients of the
subresultant sequence, which are needed by Hong's projection operator.

```python
>>> from sympy import ring, ZZ
>>> from sympy_extras.polys.subresultants import psc
>>> R, x, y = ring("x,y", ZZ)
>>> psc(x**2*y + x, x + y)
[y**3 - y, 1]

```

### Lattices, roots, algebraic numbers and exact solving on the typed layer

A group of algorithms written on the typed polynomial layer (no SymPy
inside), each described in `docs/dense.md` or `docs/solvers.md`.

**Lattice reduction and integer linear algebra** (`polys.lattice`): the
LLL algorithm in all-integer form, Hermite and Smith normal forms,
integer kernels and linear Diophantine systems. On top of it, van
Hoeij's knapsack factorisation replaces the exponential subset search of
Zassenhaus' algorithm in `polys.factortools`.

```python
>>> from sympy_extras.polys.lattice import lll, solve_integer_system, smith_normal_form
>>> lll([[1, 0, 0, 1000000], [0, 1, 0, 1414214], [0, 0, 1, 2828427]])[0]      # sqrt(8) = 2 sqrt(2)
[0, -2, 1, -1]
>>> solve_integer_system([[1, 2, 3], [4, 5, 6]], [6, 15], 3)
IntegerSolutions([1, 1, 1], [[1, -2, 1]])
>>> smith_normal_form([[2, 4, 4], [-6, 6, 12], [10, -4, -16]])
[2, 6, 12]

```

**Complex roots** (`polys.complexroots`): isolation in rectangles by
exact winding numbers, certified refinement, root counts in half-planes
and discs (the criteria of Routh–Hurwitz and Schur–Cohn).

```python
>>> from fractions import Fraction
>>> from sympy_extras.polys.complexroots import dup_isolate_complex_roots_sqf, dup_hurwitz_stable_p
>>> quartic = [Fraction(c) for c in (1, 0, 0, 0, 4)]                          # roots 1 + i, 1 - i, -1 + i, -1 - i
>>> dup_isolate_complex_roots_sqf(quartic, Fraction(1, 1000))[1].contains(Fraction(1), Fraction(1))
True
>>> dup_hurwitz_stable_p([Fraction(c) for c in (1, 3, 3, 1)]), dup_hurwitz_stable_p(quartic)
(True, False)

```

**Algebraic numbers** (`polys.numberfields`): exact arithmetic in the
field of all algebraic numbers (`qqbar.Algebraic`: a minimal polynomial
and an isolating region), principal roots, primitive elements,
recognition of a number from its decimals by LLL, integral bases and
field discriminants (Round 2).

```python
>>> from sympy_extras.polys.numberfields.qqbar import Algebraic, algebraic_roots
>>> from sympy_extras.polys.numberfields.recognize import recognize_algebraic
>>> from sympy_extras.polys.numberfields.basis import field_discriminant
>>> sqrt2, sqrt3 = Algebraic.from_rational(2).sqrt(), Algebraic.from_rational(3).sqrt()
>>> sqrt2 + sqrt3
Algebraic([1, 0, -10, 0, 1], 3.14626437)
>>> (5 + 2*sqrt2*sqrt3).sqrt() == sqrt2 + sqrt3, 1/(sqrt3 - sqrt2) == sqrt2 + sqrt3
(True, True)
>>> recognize_algebraic(Fraction(3146264369941972, 10**15), digits=15)
Algebraic([1, 0, -10, 0, 1], 3.14626437)
>>> field_discriminant([1, -1, -2, -8]), field_discriminant([1, 0, -10, 0, 1])
(-503, 2304)

```

**Polynomial systems** (`polys.fglm`, `polys.zerodim`): the change of
monomial order for zero-dimensional ideals, and all the solutions of a
system with finitely many, with exact coordinates.

```python
>>> from sympy_extras.polys.domains import QQ
>>> from sympy_extras.polys.rings import PolyRing, lex
>>> from sympy_extras.polys.zerodim import solve_zero_dimensional
>>> R = PolyRing(QQ, ['x', 'y'], lex)
>>> x, y = R.gens
>>> points = solve_zero_dimensional([x*x + y*y - R.one, y - x*x], R)          # a circle and a parabola
>>> len(points), [p for p in points if p[0].is_real and p[0] > 0]
(4, [[Algebraic([1, 0, 1, 0, -1], 0.7861513778), Algebraic([1, 1, -1], 0.6180339887)]])

```

**Finite fields** (`polys.domains.galoisfield`, `polys.galoisfactor`):
`GF(p**n)` as a coefficient domain and factorisation over it.

```python
>>> from sympy_extras.polys.domains.galoisfield import GaloisField
>>> from sympy_extras.polys.galoisfactor import gfq_factor
>>> gfq_factor([[1], [1], [1]], GaloisField(2, 2))                            # x**2 + x + 1 over GF(4)
([1], [([[1], [1, 0]], 1), ([[1], [1, 1]], 1)])

```

**Guessing** (`concrete.guess`): the recurrence, differential or
algebraic equation behind the first terms of a sequence, and closed forms
through the recurrence solver.

```python
>>> from sympy_extras.core import Function, Symbol
>>> from sympy_extras.concrete.guess import guess_recurrence, guess_closed_form
>>> n = Symbol('n')
>>> guess_recurrence([1, 1, 2, 5, 14, 42, 132, 429, 1430, 4862, 16796]).as_expr(Function('a'), n)
(-4*n - 2)*a(n) + (n + 2)*a(n + 1)
>>> guess_closed_form([1, 0, 1, 2, 9, 44, 265, 1854, 14833, 133496, 1334961, 14684570], n)
-factorial(n)*Sum((-1)**k/factorial(k + 1), (k, 0, n - 1)) + factorial(n)

```

**Diophantine equations and linear arithmetic** (`solvers`): all the
solutions of `r a**x - s b**y = c` (Baker's method with reduction), the
integer zeros of exponential polynomials, exact linear and integer
programming with certificates, Fourier–Motzkin elimination.

```python
>>> from sympy_extras.solvers.exponential_diophantine import pillai
>>> from sympy_extras.solvers.simplex import linear_program, integer_program
>>> pillai(3, 2, 1), pillai(2, 3, 13)
([(1, 1), (2, 3)], [(4, 1), (8, 5)])
>>> best = linear_program([5, 4], [[6, 4], [1, 2], [-1, 0], [0, -1]], [24, 6, 0, 0])
>>> best.value == 21, best.point == [3, Fraction(3, 2)], best.dual == [Fraction(3, 4), Fraction(1, 2), 0, 0]
(True, True, True)
>>> integer_program([5, 4], [[6, 4], [1, 2], [-1, 0], [0, -1]], [24, 6, 0, 0])
IntegerProgramResult('optimal', value=20, point=[4, 0])

```

### The Z-transform (`sympy_extras.concrete.ztransform`)

The unilateral Z-transform of sequences and the inverse of every rational
function whose poles are found in radicals (see
[docs/solvers.md](docs/solvers.md)).

```python
>>> from sympy_extras.core import Symbol
>>> from sympy_extras.functions import sin
>>> from sympy_extras.concrete.ztransform import z_transform, inverse_z_transform
>>> n, z, w = Symbol('n'), Symbol('z'), Symbol('w')
>>> z_transform(n*2**n, n, z), z_transform(sin(w*n), n, z)
(2*z/(z**2 - 4*z + 4), z*sin(w)/(z**2 - 2*z*cos(w) + 1))
>>> inverse_z_transform(z/(z**2 - z - 1), z, n)
-sqrt(5)*(1/2 - sqrt(5)/2)**n/5 + sqrt(5)*(1/2 + sqrt(5)/2)**n/5
>>> inverse_z_transform(z/(z**2 - z + 1), z, n)
2*sqrt(3)*sin(pi*n/3)/3

```

### Linear algebra on matrices of expressions (`sympy_extras.matrices`)

Determinants, inverses, ranks, null spaces, linear systems and
characteristic polynomials of matrices with symbolic entries, by
fraction-free elimination (Bareiss) and Berkowitz' algorithm over the
polynomial ring of the entries: no rational function is added to another
one on the way, and the results are in lowest terms. An 8 x 8 determinant
with entries linear in two symbols takes 0.1 s (27 s with
`sympy.Matrix.det`). See [docs/matrices.md](docs/matrices.md).

```python
>>> from sympy_extras.core import Symbol
>>> from sympy_extras.matrices import ExprMatrix
>>> from sympy_extras.matrices.linalg import linear_solve
>>> a, b, c, x = (Symbol(s) for s in 'abcx')
>>> V = ExprMatrix.from_rows([[1, a, a**2], [1, b, b**2], [1, c, c**2]])
>>> V.det()
-a**2*b + a**2*c + a*b**2 - a*c**2 - b**2*c + b*c**2
>>> ExprMatrix.from_rows([[1, a], [a, 1]]).inv()
ExprMatrix([[-1/(a**2 - 1), a/(a**2 - 1)], [a/(a**2 - 1), -1/(a**2 - 1)]])
>>> ExprMatrix.from_rows([[2, 1], [1, 2]]).charpoly(x)
x**2 - 4*x + 3
>>> linear_solve(ExprMatrix.from_rows([[1, 1, 1], [1, 2, a]]), ExprMatrix.column([1, 2]))
LinearSolution(ExprMatrix([[0], [1], [0]]), [ExprMatrix([[a - 2], [1 - a], [1]])])

```

Eigenvalues and eigenvectors are computed per irreducible factor of the
characteristic polynomial, by elimination over the extension field it
defines, so that diagonalisability is decided even when the eigenvalues
have no closed form:

```python
>>> M = ExprMatrix.from_rows([[a, b], [b, a]])
>>> M.eigenvals(), M.diagonalize()
({a - b: 1, a + b: 1}, (ExprMatrix([[-1, 1], [1, 1]]), ExprMatrix([[a - b, 0], [0, a + b]])))
>>> ExprMatrix.from_rows([[a, 1], [0, a]]).is_diagonalizable()
False

```

The Jordan form (`M.jordan_form()`), the exact singular value
decomposition of rational matrices (`M.svd()`), the matrix exponential
and symbolic powers are built on the same construction:

```python
>>> t, n = Symbol('t'), Symbol('n')
>>> ExprMatrix.from_rows([[1, -2], [2, 1]]).exp(t)
ExprMatrix([[exp(t)*cos(2*t), -exp(t)*sin(2*t)], [exp(t)*sin(2*t), exp(t)*cos(2*t)]])
>>> ExprMatrix.from_rows([[2, 1], [0, 2]]).power(n)
ExprMatrix([[2**n, 2**n*n/2], [0, 2**n]])

```

Exact decompositions (LU, LDL, Cholesky, QR, the inertia of symmetric
matrices), matrix equations (Sylvester, Lyapunov), the normal forms which
need no eigenvalue (Smith over `K[x]`, Frobenius, minimal polynomial,
similarity) and step-by-step elimination:

```python
>>> from sympy_extras.matrices.decompositions import inertia
>>> from sympy_extras.matrices.equations import lyapunov_solve
>>> from sympy_extras.matrices.normalforms import frobenius_form, matrix_minimal_polynomial
>>> from sympy_extras.matrices.steps import rref_steps
>>> ExprMatrix.from_rows([[4, 2], [2, 3]]).cholesky(), inertia(ExprMatrix.from_rows([[0, 1], [1, 0]]))
(ExprMatrix([[2, 0], [1, sqrt(2)]]), (1, 1, 0))
>>> lyapunov_solve(ExprMatrix.from_rows([[0, 1], [-1, -a]]), ExprMatrix.eye(2)).particular
ExprMatrix([[(a**2 + 2)/(2*a), -1/2], [-1/2, 1/a]])
>>> matrix_minimal_polynomial(ExprMatrix.from_rows([[2, 0, 0], [0, 2, 0], [0, 0, 3]]), x)
x**2 - 5*x + 6
>>> frobenius_form(ExprMatrix.from_rows([[1, 2], [3, 4]]))[1]
ExprMatrix([[0, 2], [1, 5]])
>>> for step in rref_steps(ExprMatrix.from_rows([[2, 4], [1, 3]])):
...     print(step.operation, ' ', step.matrix)
R1 <- (1/2)*R1   ExprMatrix([[1, 2], [1, 3]])
R2 <- R2 - R1   ExprMatrix([[1, 2], [0, 1]])
R1 <- R1 - (2)*R2   ExprMatrix([[1, 0], [0, 1]])

```

The rank-dependent operations return the generic answer for symbols; with
other kernels (`sin(t)`, `sqrt(2)`) every pivot must be proved nonzero by
the zero test, otherwise `NotImplementedError` is raised.

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
    core/                    an experimental core: unevaluating classes, evaluating functions (docs/core.md)
    functions/
        elementary/
            exponential.py   Exp, Log, LambertW and exp, log, lambertw on the typed core (ported from SymPy)
            complexes.py     Re, Im, Abs, Sign, Arg, Conjugate, Atan2; as_real_imag, conjugation (ported from SymPy)
            integers.py      Floor, Ceiling, Frac and floor, ceiling, frac (ported from SymPy)
            miscellaneous.py Max, Min and max_, min_; root, cbrt, real_root (ported from SymPy)
            piecewise.py     Piecewise, ExprCondPair and piecewise (ported from SymPy)
            trigonometric.py Sin, ..., Csc, Asin, ..., Acsc and sin, ..., acsc (ported from SymPy)
            hyperbolic.py    Sinh, ..., Csch, Asinh, ..., Acsch and sinh, ..., acsch (ported from SymPy)
        combinatorial/
            factorials.py    Factorial, Binomial, RisingFactorial, FallingFactorial (ported from SymPy)
            numbers.py       catalan, fibonacci, lucas, harmonic, bernoulli, euler, bell, genocchi, partition (ported from SymPy)
        special/
            gamma_functions.py Gamma, LogGamma, PolyGamma, DiGamma, TriGamma, LowerGamma, UpperGamma (ported from SymPy)
            delta_functions.py DiracDelta, Heaviside and dirac_delta, heaviside (ported from SymPy)
            zeta_functions.py Zeta, DirichletEta, PolyLog and zeta, dirichlet_eta, polylog (ported from SymPy)
            error_functions.py Erf, ..., Expint, EiNode, SiNode, ..., OffsetLiNode and erf, ..., Ei, Si, ..., Li (ported from SymPy)
            beta_functions.py Beta and beta (ported from SymPy)
            bessel.py        Bessel, Hankel, spherical Bessel and Airy functions (ported from SymPy)
            polynomials.py   Legendre, Chebyshev, Hermite, Laguerre, Jacobi, Gegenbauer polynomials (ported from SymPy)
            spherical_harmonics.py Ynm, Znm, Ynm_c (ported from SymPy)
            tensor_functions.py KroneckerDelta, LeviCivita (ported from SymPy)
            mathieu_functions.py mathieus, mathieuc and their derivatives (ported from SymPy)
            singularity_functions.py SingularityFunction (ported from SymPy)
            hyper.py         Hyper, MeijerG, AppellF1 and hyper, meijerg, appellf1 (ported from SymPy)
            bsplines.py      bspline_basis, bspline_basis_set (ported from SymPy)
            elliptic_integrals.py EllipticK, EllipticF, EllipticE, EllipticPi and elliptic_k, ... (ported from SymPy)
    concrete/
        pisigma.py           ΠΣ-fields and Karr's solver for first order difference equations
        karr.py              karr_sum, karr_term, summation
        creative.py          creative_telescoping, definite_sum
        guess.py             guessing recurrences, differential and algebraic equations, closed forms from the first terms
        zeilberger.py        Zeilberger's algorithm, WZ certificates, definite sums
        qhyper.py            q-Pochhammer symbols, q-Gosper, q-Zeilberger
        rational.py          Abramov's decomposition of rational summands
        zeilberger_sympy.py  the SymPy face of zeilberger: zeilberger_sum, wz_certificate, wz_prove
        qhyper_sympy.py      the SymPy face of qhyper: QPochhammer as a SymPy function, qgosper_sum, qzeilberger, ...
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
        trager.py            Trager's algorithm for algebraic functions (one square root)
        summability.py       Abel, Cesàro and Gaussian means of divergent oscillatory integrals
        axisymmetric.py      region integrals with a rotational symmetry reduced to their profile
        radicals.py          real antiderivatives of x**n * Q**(m/2), Q quadratic
        indefinite.py        indefinite integration by the typed methods in turn, every antiderivative checked
        trigonometric.py     products and rational functions of trigonometric and hyperbolic functions
        heurisch.py          the heuristic Risch integrator, typed, verified, with an extended candidate table
        rewriting.py         canonical forms and substitutions tried before SymPy's routes
        exponential.py       powers times exponentials of powers by the incomplete gamma function, exponential-rational integrands
    solvers/
        lie.py               jet spaces, prolongation, determining equations, symmetries
        pde.py               pde_symmetries, similarity_reduction, pdsolve_lie
        ode.py               ode_symmetries, canonical_coordinates, reduce_order, dsolve_lie, solve_ode
        integers.py          Hermite normal form, Contejean-Devie, Cooper's algorithm
        transcendental.py    transcendental equations reduced to polynomial ones
        kovacic.py           Kovacic's algorithm
        linear_ode.py        polynomial/rational/hyperexponential solutions, reduction of order, operator arithmetic
        irregular.py         exponential parts at irregular finite singular points (local Newton polygons)
        factorization.py     factorisation of linear differential operators (Beke), solving through the factors
        special.py           Bessel, Whittaker, hypergeometric solutions
        linear_systems.py    systems Y' = A Y + b: cyclic vectors, rational solutions, variation of constants
        first_order.py       Abel, Chini, d'Alembert-Lagrange equations
        exponential_diophantine.py  Pillai's equation a**x - b**y = c (Baker–Wüstholz, continued fractions), integer zeros of exponential polynomials (native)
        simplex.py           exact linear programming over QQ with certificates (duality, Farkas, rays), strict inequalities (native)
        recurr.py            linear recurrences: polynomial, hypergeometric (Petkovšek), d'Alembertian (Abramov–Petkovšek) solutions (native)
        charpit.py           complete integrals of first order PDEs
        kovacic.py           Kovacic's algorithm (Liouvillian solutions of second order linear ODEs)
        linear_ode.py        polynomial, rational, hyperexponential solutions; reduction of order
        charpit.py           complete integrals of first order nonlinear PDEs
    simplify/
        structure.py         the structure theorem: ElementaryTower, canonical_form, is_zero, equal
        tests/
    ntheory/
        primetest.py         isprime (Baillie–PSW), jacobi_symbol
        generate.py          nextprime, prevprime, primerange
        factor_.py           factorint (trial division, Pollard–Brent rho), perfect_power
    polys/
        domains/             typed coefficient domains: ZZ, QQ, GF(p), algebraic number fields, polynomial rings (towers for several variables), fraction fields
        densebasic.py        dense polynomials over a typed domain: the representation
        densearith.py        dense arithmetic: multiplication (Karatsuba), division, pseudo-division
        densetools.py        derivatives, evaluation, Taylor shifts, contents, truncation
        euclidtools.py       gcd (Euclid, subresultant PRS), resultants, discriminants
        polytools.py         Poly: SymPy's polynomial class on the typed layer (ZZ, QQ, ZZ[a], QQ[a]), gcd, lcm, resultants, factor lists
        polyerrors.py        the polynomial exceptions
        heuristicgcd.py      the heuristic gcd of multivariate integer polynomials
        rationaltools.py     together: sums over common denominators
        multifactor.py       factorisation of multivariate polynomials over ZZ (Wang)
        multisqf.py          square-free decomposition of multivariate polynomials
        partfrac.py          apart: partial fraction decomposition
        polyroots.py         roots of univariate polynomials in radicals
        subresultants.py     principal subresultant coefficients on SymPy's dense representation (CAD projections)
        polyutils.py         conversions between core expressions and the typed polynomials
        sqfreetools.py       square-free decomposition (Yun; over GF(p) with p-th roots)
        galoistools.py       factorisation over GF(p): distinct-degree, Cantor–Zassenhaus, Rabin's test
        factortools.py       factorisation over ZZ and QQ (Zassenhaus with quadratic Hensel lifting, van Hoeij's knapsack recombination) and over number fields (Trager)
        lattice.py           lattice reduction: the all-integer LLL algorithm, Hermite and Smith normal forms, integer linear systems
        fglm.py              change of monomial order for zero-dimensional ideals (FGLM) on the typed rings
        zerodim.py           exact solutions of zero-dimensional systems (shape lemma, algebraic coordinates)
        galoisfactor.py      factorisation over GF(p**n) (domains/galoisfield.py)
        numberfields/        qqbar.py (exact algebraic numbers), recognize.py (LLL recognition), basis.py (integral bases, Round 2), minpoly.py
        rootisolation.py     real root isolation (Descartes with bisection), Sturm counts, refinement
        complexroots.py      complex root isolation in rectangles: exact winding numbers, certified Newton refinement
        polyclasses.py       DensePoly: the typed polynomial class over a domain, with the algorithms as methods
        rings.py             PolyRing, PolyElement: sparse multivariate polynomials with monomial orders, the division algorithm
        ideals.py            Ideal: elimination, saturation, dimension, Hilbert series, radicals
        idealdecomposition.py  radical, equidimensional parts, minimal primes, primary decomposition in any dimension
        hermite.py           real solutions of zero-dimensional systems counted by Hermite's quadratic form, sign determination
        groebnerwalk.py      Gröbner walk (order conversion for any ideal)
        f4.py                Faugère's F4 on the typed rings: Gröbner bases by its own sparse row reduction, over any field
        interop.py           conversions between SymPy's polynomial rings and the typed ones (QQ, ZZ, GF(p))
        hilbert.py           Hilbert series of monomial ideals by Bigatti's algorithm
        rur.py               the rational univariate representation of zero-dimensional ideals (Rouillier)
        stronggroebner.py    strong Gröbner bases over the integers (Kandri-Rody–Kapur, Lichtblau)
        modulargroebner.py   proven modular Gröbner bases over the rationals (Arnold)
        orderings.py         WeightOrder, BlockOrder
        virtual_substitution.py  linear and quadratic quantifier elimination by virtual substitution
        comprehensive.py     comprehensive Gröbner systems, reduction over the complex numbers
        roots.py             roots in radicals through functional decomposition
        cad/
            projection.py    projection operators (McCallum, Hong)
            samplepoints.py  exact real algebraic sample points
            lifting.py       lifting phase, cylindrical_algebraic_decomposition
            qe.py            quantifier elimination and decision
        differential/
            ring.py          DifferentialRing: derivatives and rankings
            janet.py         Janet bases of linear systems of partial differential equations
            polynomial.py    differential polynomials, Ritt's reduction
            rosenfeld_groebner.py  the Rosenfeld–Gröbner algorithm
        regularchains/
            recursive.py     polynomials in their main variable: lazy pseudo-division, subresultant chains
            operations.py    regularize, regular gcd, intersect, extend; the decompositions
            regularchain.py  RegularChain, triangularize, regular_gcd

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
