# Exact equation solving and reduction

Modules: `sympy_extras.assumptions.resolve`, `sympy_extras.assumptions.solve`,
`sympy_extras.polys.virtual_substitution`, `sympy_extras.polys.comprehensive`,
`sympy_extras.polys.roots`, `sympy_extras.solvers.integers`,
`sympy_extras.solvers.transcendental`.

Mathematica's notes on its internal implementation list the algorithms
behind `Solve`, `Reduce` and `Resolve` (and, further down, `DSolve`,
`Sum`, `Series` and `Limit`). The tables say where each of them lives for
SymPy users: in SymPy itself (nothing is duplicated here), in
sympy-extras, or nowhere yet.

| Algorithm (Mathematica) | SymPy | sympy-extras |
| --- | --- | --- |
| Linear equations: Gaussian elimination, sparse and modular methods | `linsolve`, `Matrix.rref`, `DomainMatrix` | — |
| Root objects: real roots by continued fractions (Vincent–Collins–Akritas), complex roots by Collins–Krandick, validated numerics | `CRootOf`, `Poly.intervals`, `real_roots` | radicals for the roots of decomposable polynomials (`polys.roots`) |
| Polynomial equations: explicit formulas to degree four, `Factor`, `Decompose`, cyclotomic and other special polynomials | `roots` (formulas, binomials, cyclotomic, quintics, functional decomposition), `factor` | — |
| Systems of polynomial equations: Gröbner bases | `groebner`, `solve_poly_system`, `nonlinsolve` | `Ideal` (elimination, saturation, radicals; `polys.ideals`) |
| Non-polynomial equations: change of variables and polynomial side conditions | `solveset` (`_transolve`, `_solve_trig`, `_solve_radical`) | `solvers.transcendental`: kernels with side conditions, inverse-image database, assumptions on parameters, Lambert W fallback |
| `Reduce` over the reals: cylindrical algebraic decomposition | — | `polys.cad`, `resolve` |
| `Reduce` over the complex numbers: Gröbner bases | — | `polys.comprehensive`: comprehensive Gröbner systems (Kapur–Sun–Wang), `resolve(domain=S.Complexes)` |
| Linear quantifier elimination (Loos–Weispfenning virtual substitution) | — | `polys.virtual_substitution`, used first by `resolve` |
| Linear Diophantine equations: Hermite normal form | `diophantine` (single equations), `hermite_normal_form` | `solvers.integers.linear_diophantine_system` (systems, with parameters) |
| Linear Diophantine inequalities: Contejean–Devie | — | `solvers.integers.hilbert_basis`, `minimal_nonnegative_solutions` |
| Presburger arithmetic (quantified linear integer formulas) | — | `solvers.integers.cooper`, `resolve(domain=S.Integers)` |
| Univariate polynomial equations over the integers: Cucker–Koiran–Smale | `solveset(..., S.Integers)`, `diophantine` (integer roots by factorisation) | — |
| Binary quadratic Diophantine equations: Hardy–Muskat–Williams, Gauss/Dirichlet/Lagrange (Pell) | `diophantine` (`diop_quadratic`, `diop_DN`) | — |
| Thue equations, exponential Diophantine equations | — | — (see the issue tracker) |
| Assumptions in `Simplify`/`Refine`: CAD, simplex/Loos–Weispfenning, Gröbner bases | — | `ask`, `refine`, `simplify` with statement assumptions |

## Differential equations, sums and products, series and limits

The same notes describe `DSolve`, `Sum`/`Product`, `Series` and `Limit`.
Only what SymPy lacks is implemented here.

| Algorithm (Mathematica) | SymPy | sympy-extras |
| --- | --- | --- |
| Linear ODE systems with constant coefficients: matrix exponentials | `dsolve` (`linodesolve`) | — |
| Second order linear ODEs: Kovacic's algorithm | rational Riccati solutions only (`riccati`) | `solvers.kovacic` (all three cases) |
| Higher order linear ODEs with rational coefficients: Abramov–Bronstein rational and exponential solutions, factorisation (Bronstein, van Hoeij), reduction of order | — | `solvers.linear_ode`: polynomial, rational and hyperexponential solutions (exponential parts at infinity from the Newton polygon, regular singular finite points), first order right factors, reduction of order |
| Linear ODEs solved by special functions through Mellin transforms | hints `2nd_hypergeometric`, Bessel, Airy | `solvers.special`: Bessel, Whittaker and hypergeometric equations recognised through the normal-form invariant |
| Linear ODE systems with rational coefficients: Abramov–Bronstein elimination | — | `solvers.linear_systems`: cyclic vector, `rational_system_solutions`, `dsolve_linear_system` |
| Nonlinear ODEs: Riccati, Bernoulli, Abel, Chini, Clairaut, d'Alembert, exact and integrating factors, Lie symmetries | `dsolve` hints (no Abel, Chini, d'Alembert) | Lie symmetries of any order (`solvers.ode`); Abel, Chini and d'Alembert–Lagrange equations (`solvers.first_order`) |
| PDEs: separation of variables and symmetry reduction (Göktaş), first order nonlinear complete integrals (Legendre, Euler transformations), Germundsson's trigonometric power methods | `pde_separate`, first order linear `pdsolve` | symmetry reductions (`solvers.pde`), complete integrals by Charpit's method (`solvers.charpit`) |
| Sums: rational, hypergeometric (Gosper, Zeilberger), q-rational, Adamchik's hypergeometric closed forms, polygamma series by integral representations, Dirichlet series by pattern matching | `summation` (polynomial, rational, Gosper, hypergeometric closed forms) | Karr's algorithm (`concrete.karr`), Zeilberger's algorithm and WZ certificates (`concrete.zeilberger`); q-analogues, polygamma and Dirichlet series: issue tracker |
| Products: polynomial, rational, q-rational, hypergeometric, periodic classes | `product` (polynomial, rational, hypergeometric) | — |
| Series by recursive composition of expansions | `series`, `fps`, `ring_series` | — |
| Limits from series and other methods (exp-log, Gruntz) | `limit` (Gruntz), `limit_seq` | — |
| Assumptions in limits and series through `Refine`/`Simplify` | Symbol assumptions only | `assumptions.limit`, `assumptions.series` with statement assumptions and case distinctions |

## Linear quantifier elimination

Variables which occur linearly in a formula are eliminated by virtual
substitution: `Exists(x, phi)` is the disjunction of `phi` at finitely many
test points (`-oo`, the roots of the atoms, and the roots shifted by an
infinitesimal), each substitution being replaced by the polynomial
condition describing the atom's truth there. No decomposition of the
space is needed, so parameters do not slow it down.

```python
>>> from sympy import Eq, symbols
>>> from sympy_extras.assumptions import resolve, Exists, ForAll
>>> from sympy_extras.polys.virtual_substitution import eliminate_linear
>>> a, b, x, y, z = symbols('a b x y z')
>>> eliminate_linear((x > y) & (x < 1), x)
y < 1
>>> resolve(Exists(y, Eq(a*y + b, 0) & (y > 0)))
(Eq(a, 0) & Eq(b, 0)) | ((a > 0) & (b < 0)) | ((b > 0) & (a < 0))
>>> resolve(Exists(x, (x**2 < y) & Exists(z, (z > x) & (z < 1))))
y > 0

```

`resolve` eliminates the linear variables first (innermost outwards) and
leaves the others to the cylindrical algebraic decomposition.

## Reduction over the complex numbers

Over the complex numbers only equations and inequations make sense. A
system `f = 0, g != 0` in unknowns `x` with parameters `p` has solutions
exactly on a constructible set of parameters, found with a comprehensive
Gröbner system: branches covering the parameter space on each of which a
fixed set of polynomials specialises to a Gröbner basis (algorithm of
Kapur, Sun and Wang). Inequations become equations by Rabinowitsch's
trick, a universal quantifier is `not Exists not`, and quantifier-free
formulas are simplified with reduced Gröbner bases and radical
membership.

```python
>>> from sympy import S, Ne
>>> from sympy_extras.polys.comprehensive import comprehensive_groebner_system
>>> for branch in comprehensive_groebner_system([a*x - b], [x], [a, b]):
...     print(branch)
Branch([], [a], [a*x - b])
Branch([a], [b], [1])
Branch([a, b], [], [])
>>> resolve(Exists(x, Eq(a*x, 1)), domain=S.Complexes)
Ne(a, 0)
>>> resolve(Exists(x, Eq(x**2 + a*x + b, 0) & Eq(2*x + a, 0)), domain=S.Complexes)
Eq(a**2 - 4*b, 0)
>>> resolve(ForAll(x, Exists(y, Eq(y**2, x))), domain=S.Complexes)
True
>>> resolve(Eq(x**2, 0) & Ne(x, 0), domain=S.Complexes)
False

```

## Integers: linear systems, nonnegative solutions, Presburger arithmetic

```python
>>> from sympy import Mod
>>> from sympy_extras.assumptions import solve
>>> from sympy_extras.solvers.integers import linear_diophantine_system, minimal_nonnegative_solutions, cooper
>>> linear_diophantine_system([Eq(3*x + 5*y, 7)], [x, y])
([14 - 5*t0, 3*t0 - 7], [t0])
>>> minimal_nonnegative_solutions([[3, -5]], [1])
([[2, 1]], [[5, 3]])
>>> solve(Eq(3*x + 5*y, 22), [x, y], (x >= 0) & (y >= 0), domain=S.Integers)
{(4, 2)}
>>> cooper((3*x > y) & (3*x < y + 3), x)
Eq(Mod(y + 1, 3), 0) | Eq(Mod(y + 2, 3), 0)
>>> resolve(ForAll(x, Exists(y, Eq(3*y, x) | Eq(3*y, x + 1) | Eq(3*y, x + 2))), domain=S.Integers)
True

```

The general solution of a linear system is computed from the column
Hermite normal form (with the unimodular transformation), the minimal
solutions in nonnegative integers by the completion procedure of
Contejean and Devie (a Hilbert basis of the homogeneous system), and
quantifiers over the integers are eliminated by Cooper's algorithm, with
divisibility conditions written `Eq(Mod(e, k), 0)`.

## Transcendental equations

A formula in which the unknown occurs only inside exponentials,
logarithms, trigonometric functions or roots of one argument is turned
into a polynomial formula in a kernel variable (`t = exp(u)` with
`t > 0`; `w = tan(u/2)`; `l = log(u)`; `y = u**(1/q)` with `y >= 0`) by
the functional relations of the functions, solved by the cylindrical
algebraic decomposition, and pulled back through the inverse images of
the kernel (recursively when the argument is not the unknown). With
parameters, the conditions met along the way (`v > 0` when inverting
`exp(u) = v`) are decided from the assumptions or kept in a
`ConditionSet`.

```python
>>> from sympy import exp, sin, cos, log, sqrt, cbrt
>>> solve(exp(2*x) - 3*exp(x) + 2, x, x > 0)
{log(2)}
>>> solve(Eq(sin(x) + cos(x), 1), x, (x > 0) & (x < 3))
{pi/2}
>>> solve(Eq(log(x) + log(x - 1), log(2)), x, domain=S.Reals)
{2}
>>> solve(Eq(exp(x), a), x, a > 0, domain=S.Reals)
{log(a)}
>>> solve(Eq(exp(x), a), x, domain=S.Reals)
ConditionSet(x, a > 0, {log(a)})
>>> solve(Eq(sqrt(x + 1) + cbrt(x + 1), 2), x, domain=S.Reals)
{0}

```

SymPy's `solveset` returns `{-1, 2}` for the third equation over the
reals, although the logarithms are not real at `-1`.

## Verification

- `sympy_extras/polys/tests/test_virtual_substitution.py` checks every
  elimination against the CAD (`tautology(Equivalent(...))`).
- `sympy_extras/polys/tests/test_comprehensive.py` checks the parameter
  conditions of `exists_complex` at random rational parameter values
  against Gröbner bases of the specialised systems.
- `sympy_extras/solvers/tests/test_integers.py` checks Cooper's algorithm
  against brute force on boxes of integers, and the Hilbert bases against
  enumeration.
- `sympy_extras/solvers/tests/test_transcendental.py` checks the solutions
  by membership tests and substitution.
