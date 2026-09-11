# Changelog

All notable changes to this project are documented in this file.

The project is at version 0.x: there is **no guarantee of backwards
compatibility** between releases yet, and any release may rename, move or
remove public functions. Breaking changes are listed here when they happen.

## Unreleased

### Added

- `sympy_extras.integrals`: definite integration by the Marichev–Adamchik
  method. `sympy_extras.integrals.mellin` represents Mellin transforms as
  quotients of gamma functions with their strips of convergence
  (`GammaQuotient`, a table of thirty kernels checked against Mathematica,
  SymPy and quadrature, `mellin_transform`); `slater` turns a quotient into
  a Meijer G-function (Gauss's multiplication formula for rational
  coefficients of `s`) and expands the G-function by Slater's theorem with
  the `|z| < 1` case distinction decided by the assumptions
  (`mellin_barnes`, `slater_expansion`); `marichev` integrates over
  `(0, oo)`, `(0, 1)` and `(1, oo)` a power of `x` times at most two kernels
  (Parseval's formula) with logarithms as derivatives in the exponent
  (`mellin_integrate`); `residues` does rational, Fourier-type and
  trigonometric integrals by the residue theorem; `definite`
  (`definite_integral`) cuts the range at the kinks of `Abs`, `sign`,
  `Heaviside`, `Max`, `Min` and `Piecewise` and at the singularities inside
  it, maps ranges and changes variables, and asks SymPy's `integrate` last,
  keeping its answer only when a numerical check passes. The conditions
  on the parameters are returned in a `Piecewise` or decided by the
  assumptions. `regions` (`IntegralByRanges`, `integrate_by_ranges`)
  integrates over regions described by polynomial inequalities, decomposed
  into cylindrical cells by the CAD, with parameters giving a case
  distinction.

- The other methods of definite integration: `sympy_extras.integrals.telescoping`
  (the Almkvist–Zeilberger algorithm: creative telescoping for
  hyperexponential integrands, the ODE of a parametric integral and its
  solution with initial conditions), `sympy_extras.integrals.parametric`
  (differentiation under the integral sign, with the constant fixed at a
  value of the parameter), `sympy_extras.integrals.antiderivative` (an
  antiderivative evaluated by one-sided limits at the endpoints, at the
  singularities of the integrand and at the discontinuities of the
  antiderivative), all tried by `definite_integral` before SymPy's
  `integrate`; `sympy_extras.integrals.brackets` (Ramanujan's master
  theorem from the formal power series of a factor, the method of brackets
  for products of two series) as a second source of Mellin transforms;
  `sympy_extras.integrals.recognize` (a closed form conjectured from a
  high-precision quadrature by PSLQ, `definite_integral(...,
  recognize=True)`, off by default).

- `sympy_extras.integrals.risch`: the transcendental Risch algorithm
  ported from Aaron Meurer's unmerged SymPy pull requests sympy/sympy#30180,
  #30221 (the remaining exp-log cases of Bronstein) and #30292 (the
  hypertangent cases and the coupled differential system), self-contained
  on SymPy 1.14 with the tests of the branches, under SymPy's BSD licence
  with attribution; `risch_antiderivative` and `is_nonelementary` are the
  typed entry points and the antiderivative method of `definite_integral`
  uses it first. The ported modules are strictly typed like the rest of
  the package (issue #54), with the annotations of Aaron Meurer's branch
  `risch-typing` (sympy/sympy#30282) as the starting point.

- `sympy_extras.integrals.validated`: validated numerical integration
  (composite Simpson and five-point Gauss–Legendre with the derivatives
  enclosed in interval arithmetic in centred form, a proved error bound;
  algebraic and logarithmic endpoint singularities bounded by closed
  forms or removed by power substitutions, removable singularities by the
  Taylor remainder, infinite ranges by the exponential and rational maps
  with oscillatory tails integrated by parts), `definite_integral(...,
  numeric=True)` as the answer of last resort for integrals without
  parameters; thirty digits of `exp(-x**2)` on `(0, 1)` in a second.

- `sympy_extras.integrals.reduction`: reduction-based creative telescoping
  for hyperexponential integrands (Hermite reduction with the shell of
  integer residues and the polynomial reduction of Bostan–Chen–Chyzak–Li–Xin),
  minimal-order telescopers without a certificate bound, tried before the
  ansatz of `telescoping`.

- `sympy_extras.integrals.algebraic`: algebraic integrands of genus zero
  by rationalising substitutions (Euler's substitutions, roots of Möbius
  functions, Chebyshev's binomial differentials).

- `IntegralByRanges` scales ellipses, ellipsoids and shifted discs to the
  radial case, integrates cylinder-like regions in cylindrical coordinates,
  and allows unbounded outer variables in the Fubini route.

- `sympy_extras.integrals.tables`: fifty-eight entries of Gradshteyn and
  Ryzhik with their conditions, matched first by `definite_integral`, each
  checked numerically in the tests.

- `definite_integral(..., finite_part=True)`: Hadamard's finite part of a
  divergent integral, from the antiderivative with the divergent terms of
  the excision at each singularity dropped.

- `sympy_extras.integrals.elliptic`: square roots of cubics and quartics
  reduced to Legendre's elliptic integrals (Byrd–Friedman's substitutions,
  complete and incomplete forms, `elliptic_pi` for simple poles);
  radicands with a pair of complex roots (Byrd–Friedman 240, 241, the
  bilinear substitution onto `cos(theta)`), polynomial and rational
  numerators by the recurrences 310-318 (incomplete powers, multiple
  poles), symbolic parameters with the triangle inequalities as facts;
  `1/sqrt(x**3 + 1)` over `(0, oo)` is `gamma(1/6)*gamma(1/3)/(3*sqrt(pi))`.

- `sympy_extras.integrals.series`: series expansion of a factor and termwise
  integration with the series summed in closed form (`summation`, the
  Zeilberger and polygamma-series algorithms, hypergeometric closed forms);
  Fourier series of the classical table (`log(sin(x))`, `log(cos(x))`,
  `log(tan(x))`, `log(1 - cos(x))`, `x`, `x**2`, `Abs(sin(x))`, ...,
  Gradshteyn–Ryzhik 1.441-1.444) integrated termwise, against each other
  and against harmonics by orthogonality, and against any other factor
  through the moments of the harmonics, with the Clausen sums as
  polylogarithms and the rational sums by partial fractions
  (`x*log(sin(x))` over `(0, pi/2)` is `-pi**2*log(2)/8 + 7*zeta(3)/16`).

- `IntegralByRanges(..., measure='hausdorff')`: integrals over curves and
  surfaces given by one equation among the conditions (the surface measure
  `sqrt(1 + |grad phi|**2)` on the explicit branch of the section of the
  CAD), `Eq(x**2 + y**2, 1)` has length `2*pi` and the paraboloid
  `Eq(z, x**2 + y**2)`, `z < 1` area `pi*(5*sqrt(5) - 1)/6`; cells bounded
  by monotone non-polynomial conditions (`y**2 < exp(x)`) solved for the
  bounds.

- `sympy_extras.integrals.asymptotic`: asymptotic expansions of parametric
  integrals (Watson's lemma, Laplace's method to any order, the leading
  term of the stationary phase), `asymptotic_integral`.

- `sympy_extras.integrals.dfinite`: Chyzak's algorithm (creative
  telescoping for D-finite integrands by Koutschan's ansatz on the closure
  of the factors), `chyzak`, `dfinite_ode`, `dfinite_integral`, tried by
  `definite_integral` for parametric integrands which are not
  hyperexponential.

- `sympy_extras.integrals.laplace`: the operational rules of the Laplace
  transform (division and multiplication by `t`, the shifts, periodic
  integrands, convolutions) for `g(t)*exp(-s*t)` over `(0, oo)`.

- `sympy_extras.integrals.contours`: the rectangular contour for
  `x**n*exp(k*x)*R(exp(c*x))` over the real line, the sector contour for
  `x**a/(b + c*x**n)` with symbolic exponents, and the indented contour
  for `R(x)*sin(k*x)` with simple real poles cancelled by the sine.

- `sympy_extras.integrals.transformations`: Frullani's theorem and Glasser's
  master theorem (with the Cauchy–Schlömilch transformation), applied by
  `definite_integral` before the range is cut at singularities.

- `sympy_extras.integrals.periodic`: the integral of a trigonometric
  integrand over whole periods as `2*pi*k` times the constant Laurent
  coefficient of its form in `exp(I*x)` (`exp(cos(x))*cos(sin(x))` over a
  period is `2*pi`, `exp(a*cos(x))*cos(n*x)` is `2*pi*besseli(n, a)`).

- `IntegralByRanges` integrates radial integrands over discs, annuli and
  balls in polar and spherical coordinates, and regions whose conditions
  are linear in the last variable (`y < exp(x)`) by Fubini's theorem with
  the bounds solved for it, before the cylindrical decomposition.

- `definite_integral` substitutes the equalities among the assumptions
  before integrating (a formula valid for generic parameters fails on
  `Eq(n, m)`), and contains the internal failures of SymPy 1.14 met on the
  datasets (the assertion of the LRA solver reached through `ask`, the
  cache wrapper of `meijerint`, a division by zero in `evalf`).

- `definite_integral` reads the integrand in rewritten forms (trigonometric
  products as sums, hyperbolic functions as exponentials, inverse
  hyperbolic functions as logarithms, orthogonal polynomials expanded),
  knows the Mellin transform of `log(1 - x)` on `(0, 1)`, computes Cauchy
  principal values with `principal_value=True`, and handles symbolic
  endpoints through the antiderivative (items of issue #53).

- `definite_integral(..., regularize=True)`: analytic regularisation of
  the Marichev–Adamchik method, the strips of convergence dropped and the
  gamma quotient continued analytically (Hadamard's finite part of a
  divergent Mellin-type integral, `x**(-3/2)*exp(-x)` over `(0, oo)` is
  `-2*sqrt(pi)`); products of three kernels one of which is trigonometric
  or hyperbolic, through exponentials with complex scales whose argument
  conditions are decided by their real parts (`exp(-a*x)*sin(b*x)*
  besselj(0, x)` over `(0, oo)`).

- The benchmark driver `definite_integrals` of sympy-extras-benchmarks
  gained `--extras` to run `definite_integral` on the four integration
  datasets; the results are in `benchmarks/README.md`.

### Fixed

- `sympy_extras._timeout` builds SymPy's table of Meijer G-function
  representations before setting a time limit: a limit hit during the
  build left the global table truncated, after which every later
  integration of an exponential went astray (the Mellin transform of
  `exp(-x)` came out as `uppergamma(s, 0)` on the whole plane, and the
  divergent `x**(-3/2)*exp(-x)` over `(0, oo)` got the value `-2*sqrt(pi)`
  after an interrupted integration).

- `ask(element(e, S.Reals))` no longer answers `True` for an expression in
  real variables which is not a polynomial with rational coefficients
  (`(a + I*b)**2` with `a` and `b` real).

## 0.0.1 - 2026-09-10

First release.

### Added

- [docs/comparison.md](docs/comparison.md): seventy questions answered by
  sympy-extras and not (or wrongly) by SymPy, with both answers, generated
  and checked by `benchmarks/comparison.py`; the sympy-extras side is
  doctested.

- `benchmarks/fuzz.py`: a randomised driver which cross-checks one part of
  the package at a time (`convergence`, `parametric-convergence`, `sums`,
  `isolation`, `solve`, `ask`, `limits`, `thue`, `ode`, `refine`) against
  an oracle which shares no code with it -- partial sums computed term by
  term with mpmath, brute force enumeration, sign changes on a fine grid,
  sampling of the solution set, `checkodesol`, or SymPy asked the
  parameter-free question a parametric answer specialises to. The three
  bugs listed under *Fixed* come from it.

- The remaining items of the implementation notes: `sympy_extras.solvers.thue_equation`
  (Thue equations by Baker's method: units of the order by enumeration
  and saturation, Baker–Wüstholz, de Weger's reduction with an exact
  LLL), `sympy_extras.solvers.second_order` (integrating factors
  `mu(x, y)` and `mu(y')`, Lie's linearisation test, the fibre preserving
  transformation to `u'' = 0`, rectification of commuting symmetries,
  `dsolve_second_order`), `sympy_extras.solvers.abel` (relative and
  absolute invariants, equivalence of Abel equations with the
  transformation, the AIR class through a particular solution or a
  representative; `abel_ode` uses it for non-constant invariants and no
  longer runs the separable-class search on them),
  `sympy_extras.concrete.eulersums` (Euler sums in zeta values,
  `polygamma_series`, `polygamma_integral_representation`). The Riccati
  solver tries the special functions before Kovacic's algorithm.
- More of the algorithms in Mathematica's implementation notes:
  `sympy_extras.solvers.isolation` (real roots of transcendental
  functions isolated exactly, `TranscendentalRoot` objects returned by
  `solve` when `solveset` gives a `ConditionSet`, inequalities solved
  through the roots), `sympy_extras.solvers.dae` (linear
  differential-algebraic equations with constant coefficients through
  the core-nilpotent decomposition of the pencil, `dsolve_dae`,
  `dae_index`, `dae_matrices`), `sympy_extras.concrete.convergence`
  (`sum_convergence`, `product_convergence`, `is_convergent`: the
  condition on the parameters for convergence, by the ratio, Raabe,
  Bertrand, root, power comparison, Leibniz and integral tests),
  `sympy_extras.concrete.dirichlet` (`dirichlet_series` for the
  arithmetic functions, alternating signs and logarithmic factors),
  `sympy_extras.assumptions.limit_seq` (sequence limits with assumptions
  and case distinctions, negative bases handled), `ask` deciding linear
  formulas by Loos–Weispfenning virtual substitution before the CAD,
  `refine` knowing the residues of integer polynomials
  (`(-1)**(n**2 + n)`, `Mod(n**3 - n, 6)`). The Lambert W fallback of the
  transcendental solver returns both real branches
  (`solve(exp(x) - x - 2, x, x > 0)` was empty).
- Quadratic virtual substitution (Weispfenning) in
  `sympy_extras.polys.virtual_substitution` (`eliminate_quadratic`,
  `virtual_substitution_elimination`), used by `resolve` before the CAD;
  `sympy_extras.concrete.qhyper` with q-Pochhammer symbols, q-binomial
  coefficients, the q-Gosper and q-Zeilberger algorithms;
  `sympy_extras.concrete.rational` with Abramov's decomposition of
  rational summands and `rational_sum`. `AGENTS.md` records the standing
  instructions of the maintainer.

- `sympy_extras.solvers.special` (Bessel, Whittaker and hypergeometric
  solutions of second order linear equations recognised through the
  normal-form invariant), `sympy_extras.solvers.first_order` (Abel, Chini
  and d'Alembert–Lagrange equations), `sympy_extras.solvers.linear_systems`
  (cyclic vector, rational and general solutions of `Y' = A Y`), Riccati
  equations linearised and solved through the linear solvers,
  exponential parts at infinity from the Newton polygon in
  `hyperexponential_solutions`, Lambert W fallback in the transcendental
  solver, and the Kamke driver trying these solvers after `dsolve`.

- Differential equations, sums, series and limits, after the same
  implementation notes: `sympy_extras.solvers.kovacic` (Kovacic's
  algorithm, the three cases), `sympy_extras.solvers.linear_ode`
  (polynomial, rational and hyperexponential solutions of linear ODEs
  with polynomial coefficients, first order right factors, reduction of
  order, `dsolve_linear`), `sympy_extras.solvers.charpit` (complete
  integrals of first order nonlinear PDEs), `sympy_extras.concrete.zeilberger`
  (Zeilberger's algorithm, WZ certificates, `zeilberger_sum`),
  `sympy_extras.assumptions.limit` and `series` (statement assumptions,
  `Piecewise` case distinctions).
- Exact equation solving and reduction, after the list of algorithms in
  Mathematica's implementation notes ([docs/reduce.md](docs/reduce.md)):
  `sympy_extras.polys.virtual_substitution` (linear quantifier elimination
  by virtual substitution, used first by `resolve`),
  `sympy_extras.polys.comprehensive` (comprehensive Gröbner systems of
  Kapur–Sun–Wang, existential elimination and simplification over the
  complex numbers, `resolve(..., domain=S.Complexes)`),
  `sympy_extras.solvers.integers` (Hermite normal form with transform,
  linear Diophantine systems, Hilbert bases and minimal nonnegative
  solutions by Contejean–Devie, Cooper's algorithm,
  `resolve(..., domain=S.Integers)`, integer systems in `solve`),
  `sympy_extras.solvers.transcendental` (transcendental equations and
  inequalities reduced to polynomial ones through kernels and inverse
  images, with the parameters' assumptions, used by `solve`),
  `sympy_extras.polys.roots` (roots in radicals of decomposable
  polynomials). `ask` decides the reality of roots and logarithms itself
  (SymPy's `ask(Q.real(sqrt(a - 2)), Q.positive(a))` is `True`).

- `sympy_extras.polys.cad`: cylindrical algebraic decomposition (CAD) of
  real space adapted to a set of polynomials with rational coefficients, with
  McCallum's and Hong's projection operators, exact real algebraic sample
  points and the lifting phase (`cylindrical_algebraic_decomposition`, `CAD`,
  `CADCell`, `NotWellOriented`, `projection_sets`, `mccallum_projection`,
  `hong_projection`, `squarefree_basis`, `SamplePoint`).
- `sympy_extras.polys.cad`: quantifier elimination and decision over the
  reals built on the CAD (`quantifier_elimination`, `decide`, `sample_points`,
  `solution_set`).
- `sympy_extras.polys.euclidtools`: principal subresultant coefficients of
  two polynomials (`dup_psc`, `dmp_psc`, `psc`).
- `sympy_extras.concrete`: Karr's algorithm for indefinite summation in
  ΠΣ-fields (`karr_sum`, `karr_term`, `summation`, `PiSigmaField`,
  `build_pisigma_field`): sums of harmonic numbers, nested sums,
  hypergeometric terms and their products, with a proof of non-existence
  when there is no closed form in the field. SymPy only has Gosper's
  algorithm.
- `sympy_extras.polys.ideals`: an `Ideal` class on top of SymPy's Gröbner
  bases with elimination ideals, intersections, quotients, saturations,
  radical membership, Krull dimension, Hilbert series and polynomial,
  degree, and for zero-dimensional ideals standard monomials,
  multiplication matrices, univariate polynomials, radicals and the
  radical/prime/maximal tests (all missing from SymPy's `agca` ideals).
- `sympy_extras.polys.groebnerwalk`: the Gröbner walk, converting Gröbner
  bases between orders for ideals of any dimension (SymPy's FGLM is
  restricted to zero-dimensional ideals), with lifting by normal forms so
  that SymPy's Buchberger implementation handles the initial forms.
- `sympy_extras.solvers`: Lie point symmetries of ordinary and partial
  differential equations and systems (`JetSpace`, `Symmetry`, `symmetries`,
  `check_symmetry`): prolongation, invariance condition with the leading
  derivatives eliminated, determining equations solved exactly with a
  polynomial ansatz times optional basis functions, every symmetry verified
  by substitution. Similarity reductions of PDEs in two independent
  variables and group invariant solutions (`pde_symmetries`,
  `similarity_reduction`, `pdsolve_lie`, verified with `checkpdesol`).
  ODEs of any order solved through a symmetry by reduction of order in
  canonical coordinates (`ode_symmetries`, `canonical_coordinates`,
  `reduce_order`, `dsolve_lie`, `solve_ode`, verified with `checkodesol`).
  Every SymPy step runs under a time limit (`sympy_extras._timeout`).
- `sympy_extras.assumptions.analysis`: signs of expressions of one real
  variable beyond polynomials (continuity, exact zeros with `solveset`,
  sample points between zeros, monotonicity from the derivative, limits at
  the endpoints; constants signed exactly or by certified numerical
  evaluation), used by `ask` and so by `refine`, `simplify` and `solve`:
  `refine(Abs(sin(x)), (x > 0) & (x < pi))` is `sin(x)`.
- `sympy_extras.assumptions.intervals`: rigorous interval arithmetic
  (mpmath) on boxes: branch and bound, monotonicity in every variable with
  the extreme values at the corners, ranges of inner arguments, and
  certified isolation of the zeros of a function of one variable;
  `sympy_extras.assumptions.bounds`: elementary functions replaced by
  variables constrained by polynomial bounds (MetiTarski's method) and
  decided by the CAD. Both are used by `ask`, hence by `refine`,
  `simplify` and `solve`, for expressions of several variables and for
  statements the univariate analysis cannot settle: `ask(sin(x) < x, x > 0)`,
  `refine(Abs(sin(x*y)), 0 < x < 1, 0 < y < 3)`.
- `sympy_extras.settings`: global settings `numerical_checks` (the
  certified numerical evaluation in `ask`/`solve`, on by default),
  `precision` and `timeout` (the default time limit of the solvers), with
  the context manager `configure`.
- `sympy_extras.assumptions.solve` checks the finite candidates returned
  by SymPy's solvers against the equation and drops extraneous roots
  (`solveset` returns some for radical equations).
- `benchmarks/`: random verification drivers (`verify_random.py` on the
  Kamke collection with a numerical residual check, `logic_random.py` for
  the refinement of logical expressions, `qf_nra.py` on the SMT-LIB
  Meti-Tarski problems, transcendental equations in `solve_random.py`).
- `sympy_extras.assumptions.solve`: `Solve` with assumptions and a domain;
  polynomial problems in one real unknown are solved exactly by the CAD,
  the others by `solveset`/`nonlinsolve` with the parameters carrying the
  assumptions and the solutions filtered, dropped or kept in a
  `ConditionSet` by `ask`.
- `ask` now answers `False` for an inequality whose sides are not real
  (`sqrt(a) > 0` with `a < 0`).
- `benchmarks/`: drivers running the Kamke collection of ODEs (downloaded
  from Maxima's test suite), a table of classical PDE symmetry algebras and
  random polynomial equations against an oracle.
- `sympy_extras.assumptions.refine` and `simplify` rewritten: the symbols
  and the subexpressions whose sign or nature matters are decided against
  the assumptions (SymPy's predicates plus the CAD) and replaced by symbols
  carrying the equivalent assumptions of SymPy's core while SymPy's
  `refine`/`simplify` run, so that `log(exp(x))`, `sin(n*pi)`,
  `log(x) + log(y)`, `gamma(x + 1)/gamma(x)`, `(x**y)**(1/y)` and the like
  simplify; new handlers for powers and logarithms of polynomials with
  factors of known sign (`sqrt(x**2 - 2*x + 1)`, `log((x - 1)**2)`),
  `atan2`, `arg`, `re`/`im` of compound expressions and `frac`; Boolean
  formulas polynomial in real variables are minimised by quantifier
  elimination in `simplify` (`Eq(x**2, 1)` with `x > 0` is `Eq(x, 1)`).
  SymPy's `refine` is no longer applied to relations (its `ask` on them is
  unreliable).
- Complete type annotations checked by mypy in strict mode (`python -m
  mypy`), type aliases in `sympy_extras._typing` and a `py.typed` marker;
  `Quantifier.kind` is renamed `Quantifier.quantifier` (SymPy's `Basic`
  already has a `kind`) and the internal CAD function `_truth_values`
  returns a 4-tuple. The code base has no `# type: ignore`, `cast` or
  `getattr`: `ForAll`/`Exists` no longer evaluate in the constructor
  (`ForAll(x, True)` is an instance; `simplify()`, `prenex` and the
  functions consuming quantified formulas reduce the trivial cases, and a
  quantifier without variables is a `ValueError`).
- Validation suites against results published elsewhere (Concrete
  Mathematics, A = B, the Sigma literature, Cox-Little-O'Shea, Macaulay2
  and Singular documentation examples, the Katsura and cyclic benchmark
  systems): `sympy_extras/concrete/tests/test_karr_known.py` and
  `sympy_extras/polys/tests/test_ideals_known.py`.
- `sympy_extras.polys.orderings`: hashable weight and block (elimination)
  monomial orders usable with `sympy.groebner`.
- `sympy_extras.assumptions`: assumptions written as mathematical statements
  (`x > 0`, `element(n, S.Integers)`, `ForAll`, `Exists`) instead of `Q`
  predicates, with an interface modelled on Mathematica's: `ask`, `refine`,
  `simplify`, `assuming`/`global_assumptions`, `resolve` (quantifier
  elimination), `satisfiable`, `tautology` and `find_instance`. SymPy's
  assumptions system and SAT solver are the backends together with the CAD.

### Fixed

- `sympy_extras.solvers.thue_equation` no longer leaves mpmath's global
  precision raised when it returns (`at_precision`), so the results of the
  interval arithmetic and of the root isolation no longer depend on whether
  a Thue equation was solved before; and the enumeration of the elements of
  a given norm, whose cost is proportional to the right-hand side, is
  bounded and raises `NotImplementedError` instead of running for hours.
  The module was renamed from `solvers.thue` to `solvers.thue_equation` so
  that it does not shadow the `thue` function.

- `sympy_extras.assumptions.solve` no longer passes on the answer SymPy's
  `solveset` gives for a periodic inequality, which covers a single period
  (`solveset(sin(x) > 0, x, S.Reals)` is `Interval.open(0, pi)`, so `-6`
  and `7` were reported as non-solutions). The solutions over one period
  are tiled over the region asked for: explicitly when it meets a few
  periods, and as a condition on `Mod(x, T)` when it meets infinitely
  many. Equations keep the `ImageSet` family `solveset` returns.

- `sympy_extras.assumptions.limit` no longer answers `0` for
  `limit(a**x, x, oo)` whenever `a < 1`: the sign of `log(a)` decides the
  limit only for `a > 0`, and the case split now covers a non-positive
  base, where the limit is `0` for `|a| < 1`, `nan` for `a = -1` and
  `zoo` for `a < -1`.

- `sympy_extras.concrete.sum_convergence` decides the series with a
  bounded oscillating factor by Dirichlet's test instead of falling back
  on SymPy: `sum_convergence(sin(n)/n, n)` is `True`, where
  `Sum(sin(n)/n, (n, 1, oo)).is_convergent()` is `False`.

- `sympy_extras.assumptions.limit` no longer asks the cylindrical
  algebraic decomposition for the sign of the bounds of an oscillating
  factor: `limit(sin(x)*x**a, x, oo)` raised
  `PolynomialError: polynomial with rational coefficients expected`, and
  a formula whose conditions compare an `AccumBounds` with zero is not an
  answer. The unevaluated limit is returned for the cases which stay
  undecided. `to_polynomial` no longer takes an expression without free
  symbols which is not a rational number (an `AccumBounds`, an
  unevaluated limit) for a polynomial: `Poly` turns it into a generator
  of its own.

These were found by `benchmarks/fuzz.py`, a randomised driver added in
this release which cross-checks each part of the package against an
oracle sharing no code with it, and while following up the cases it
reported; each of them has a regression test.

Twenty-six defects reported by the cross-checking audit against Wolfram
Mathematica, brute force and the test suites of Lean, Rocq, Maxima, TPTP
and REDUCE (issues #27--#52), each with a regression test that fails on the
unfixed code:

- `resolve` over the integers (#27, #45, #51): `_drop_covered` tested each
  disjunct against the original list, so two disjuncts pinning the same
  point covered each other and were both dropped; and it read an equation
  nested inside a disjunct as pinning a value. Both lost solutions, made
  the answer depend on the order of the bound variables, and in one
  direction called a false universal statement true.
- `resolve` over the integers accepts `Mod(e, k) = r` for a nonzero
  residue (#44), and decides a nonlinear universal formula true over the
  reals (or an existential one false there) by the real relaxation
  instead of refusing it (#46).
- `resolve`, `ask` and `refine` clear a parameter in a denominator
  (`x/a = 1` is `x - a = 0` with `a != 0`) instead of rejecting it (#32).
- Virtual substitution substitutes an equality with a numeric leading
  coefficient directly: a system of five linear equalities took a minute
  and now takes a chain of substitutions (#47).
- `sympy_extras._timeout.attempt` contains `RecursionError` (#49).
- `solve` no longer raises `AttributeError` on a quartic whose radical form
  `evalf` cannot close (#48), and returns the same algebraic number written
  in two radical forms once (#50).
- `solve_ode` applies its `check` on the `dsolve` branch as well, so
  truncated series, answers carrying `nan` and non-solutions are dropped
  (#40); `dsolve_linear` rejects a truncated series (#37);
  `dsolve_second_order` returns the first integral rather than an answer
  with the unknown under an integral (#29); `abel_ode` checks a relation
  against the equation before returning it (#38); `bessel_solutions`,
  `whittaker_solutions` and `riccati_ode` no longer raise `IndexError` on
  `z'' = 0` (#42); Kovacic's case 3 writes `omega` as a function of `x`
  defined by its polynomial rather than applied to it (#31).
- `zeilberger_sum` never returns a Boolean, and does not re-assert a
  rejected order-zero answer as a recurrence (#33), nor let an
  `AttributeError` from `rsolve` escape (#43); `karr_sum` adjoins the
  harmonic numbers for a rational summand automatically (#34) and repairs
  a closed form at the integer poles of its antidifference (#35).
- `sum_convergence` no longer takes `Sum.is_convergent() == False` as a
  proof of divergence, and proves `n*sin(n)` divergent by its own term
  test (#41); `product_convergence` decides an alternating factor
  exactly, `prod (1 + (-1)**n/n**a)` for `a > 1/2` (#39).
- `limit` splits on the degenerate values of a parameter, `(a x + 1)/(b x
  + 2)` at `b = 0` (#30); verifies each piece of a case split at points of
  its case and refines on the differences of the parameters, giving
  `max(a, b)` for `log(x**a + x**b)/log(x)` (#28); and returns `zoo`, not
  a directed infinity, where the modulus diverges but the argument does
  not converge (#52).

The CAD code was originally proposed to SymPy in the pull requests
[sympy/sympy#30422](https://github.com/sympy/sympy/pull/30422),
[sympy/sympy#30423](https://github.com/sympy/sympy/pull/30423),
[sympy/sympy#30424](https://github.com/sympy/sympy/pull/30424) and
[sympy/sympy#30425](https://github.com/sympy/sympy/pull/30425).
