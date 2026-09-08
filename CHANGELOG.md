# Changelog

All notable changes to this project are documented in this file.

The project is at version 0.x: there is **no guarantee of backwards
compatibility** between releases yet, and any release may rename, move or
remove public functions. Breaking changes are listed here when they happen.

## 0.0.1

First release.

### Added

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

The CAD code was originally proposed to SymPy in the pull requests
[sympy/sympy#30422](https://github.com/sympy/sympy/pull/30422),
[sympy/sympy#30423](https://github.com/sympy/sympy/pull/30423),
[sympy/sympy#30424](https://github.com/sympy/sympy/pull/30424) and
[sympy/sympy#30425](https://github.com/sympy/sympy/pull/30425).
