# Changelog

All notable changes to this project are documented in this file.

The project is at version 0.x: there is **no guarantee of backwards
compatibility** between releases yet, and any release may rename, move or
remove public functions. Breaking changes are listed here when they happen.

## Unreleased

### Added

- `sympy_extras.integrals.meijerint`: a native port of SymPy's Meijer
  G-function integration (`sympy/integrals/meijerint.py`, SymPy 1.14) on
  the typed core: the table of G-function representations
  (`lookup_table`, typed `TableEntry` records), `rewrite_single` (the
  table, the recursive Mellin transform of a product by the theorem for
  two G-functions, ported with `_rewrite_gamma` and the inverse Mellin
  transform of `transforms.py`, and the Mellin kernels of `.mellin`
  inverted by `slater.mellin_barnes`), `meijerint_indefinite` and
  `meijerint_definite` (the Mellin-convolution theorems with the
  convergence conditions of Prudnikov 2.24.1-2 and Luke 5.6, the changes
  of range), and `meijer_antiderivative`, SymPy's `integrate(f, x,
  meijerg=True, risch=False)`; values come with their conditions as pairs.
  Not ported: `meijerint_inversion`.
- `sympy_extras.integrals.deltafunctions`: a native port of SymPy's
  `deltaintegrate` and `change_mul`, with `DiracDelta.is_simple` and the
  expansion of a delta by the roots of its argument.
- `MeijerG.ap`, `bq`, `nu`, `delta`, `get_period` (ported from SymPy).

- `sympy_extras.integrals.manualintegrate`: a native port of SymPy's
  `manualintegrate` (SymPy 1.14) on the typed core, with the steps:
  `integral_steps(f, x)` returns the tree of the rules (frozen dataclasses,
  `URule`, `PartsRule`, `CyclicPartsRule`, `TrigSubstitutionRule`, ...,
  `AlternativeRule`, `DontKnowRule`), whose `eval()` is the antiderivative;
  `manualintegrate(f, x)` evaluates it. The rules are generators which yield
  the integrals they need to an `IntegralSolver` (the design of SymPy's
  development branch), which cuts the cycles and remembers the integrals
  solved completely. Every rule is ported, the special functions included
  (`Ei`, `Ci`, `Si`, `Chi`, `Shi`, `li`, `erf`, `erfi`, Fresnel, upper
  gamma, polylog, elliptic, orthogonal polynomials, `Heaviside`,
  `DiracDelta`). Against SymPy: the secant substitution says `x > c`
  (SymPy's `-c < x < c` gave a wrong antiderivative for `x < 0`).
- `sympy_extras.integrals.antiderivative.elementary_antiderivative`: the
  antiderivative of a polynomial in elementary functions of linear
  arguments, by the native `manualintegrate` (checked) and the native
  exponential and heuristic Risch methods.

### Changed

- The asymptotic series of `fresnels` and `fresnelc` at `oo` and `-oo`
  are native (`FresnelIntegral._eval_aseries`, ported from SymPy's
  `fresnels._eval_aseries` and `fresnelc._eval_aseries`): they went to
  SymPy through the bridge of the core.
- The native code sets its time limits with `native=True`: the series and
  simplifications of `solvers.kovacic`, the antiderivatives of
  `solvers.darboux` and the transfer from the reals of
  `assumptions.resolve` set limits which prepared SymPy's tables, and so
  imported SymPy; `_numeric`'s check of a hidden zero by the foreign
  minimal polynomial sets none when no foreign system is registered. A
  test of `core/tests/test_independence.py` checks every limit of the
  native modules.
- `piecewise_simplify_arguments` no longer sends a condition with an
  `Xor` to SymPy: the new `sympy_extras.logic.anf` ports SymPy's
  algebraic normal form (`to_anf`, `is_anf`, `distribute_xor_over_and`,
  `ANFform`, `anf_coeffs`, `bool_monomial` and the `Xor` constructor,
  `xor_evaluated`), and `logic.boolsimp.simplify_boolean` simplifies an
  `Xor` as SymPy's `Xor._eval_simplify` (its normal form and the patterns
  of `Xor`, no longer the univariate intervals and `simplify_logic`);
  `Piecewise._intervals` writes `Xor`, `Implies`, `Equivalent` and `ITE`
  in conjunctive normal form as SymPy does. A relation whose sides are
  not atoms is a variable of the normal form (SymPy's `simplify` and
  `to_anf` raise `AttributeError` on it).
- **Breaking**: the integrators reach SymPy no more. The method
  `'meijer'` of `sympy_extras.integrals.indefinite` is the native
  `meijerint.meijer_antiderivative`, and the method `'sympy'` (SymPy's
  `integrate`) is removed, with `indefinite.SYMPY_METHODS`; the exact
  methods of the antiderivative route of definite integration take
  `'manual'` and `'meijer'` in its place. The last method of the native definite
  integration, SymPy's `integrate` and `meijerint`, is native: the
  theorems of `meijerint_definite` and an antiderivative at the endpoints
  (cut at its discontinuities on real ranges; the delta functions by the
  native `deltaintegrate`), checked numerically as before. Removed:
  `indefinite_sympy.meijerint_of_core`, `indefinite_sympy.integrate_of_core`,
  `definite_sympy.integrate_of_core`.
- `sympy_extras.assumptions.boolsimp` is merged into `sympy_extras.logic.boolsimp` (one port of SymPy's `simplify` of Booleans, used by `refine` and, with SymPy's univariate intervals, `piecewise`; `Xor` no longer raises).
- The core's `ask` no longer asks SymPy what its facts do not decide:
  the last steps of SymPy's `ask` are ported in the new
  `assumptions.handlers` (the handlers of the predicates by class of the
  argument, `sympy/assumptions/handlers/sets.py`, `order.py`,
  `ntheory.py`, then the SAT search of `assumptions.satask`), so the
  bridge's `ask` is used only for questions about wrapped SymPy objects.
  The numbers are decided by reliable values (an imaginary part below the
  precision, as in `exp_polar(I*pi)`, decides nothing) and the integers
  among them are proved by their minimal polynomial. SymPy's rule
  `Real**Real -> Real` for a base which may be zero is not ported: SymPy
  answers `ask(Q.real(sqrt(x)), Q.real(x))` and
  `ask(Q.real(sqrt(-1 - x**2)), Q.real(x))` with `True` (here `None` and
  `False`). SymPy's relational step (`lra_satask`) is not ported, but
  for its frequent case: a sign of `e` decided by an assumed sign of `f`
  with `e/f` a rational number (`Q.positive(1 - a)` from
  `Q.negative(a - 1)`); the algebraic numbers are proved zero by their
  minimal polynomial.
- `is_meromorphic` of the incomplete gamma, Bessel, Hankel and spherical
  Bessel functions, the rewrites of `ff`, `rf` and `binomial` in terms of
  each other and of `dirichlet_eta` in terms of `zeta` are native ports
  of SymPy's methods (they went to SymPy through the bridge); an applied
  undefined function is not sent to the bridge to be rewritten (SymPy has
  no rule for it).
- `Basic.simplify` of a relation or a Boolean formula is native
  (`logic.boolsimp.simplify_boolean`, the port of SymPy's simplify of
  Booleans) instead of SymPy's `simplify` through the bridge; a formula
  with an `Xor` or with wrapped SymPy objects still goes to SymPy.
- `Expr.equals` takes the steps of SymPy's `equals` natively before the
  bridge: the difference simplified by the native `simplify` (zero:
  `True`; no sum left: `False`), its factors decided one by one, a number
  decided by its minimal polynomial; a difference with polar numbers
  (`exp_polar(0) - 1`, which SymPy does not take for zero) or wrapped
  SymPy objects, and the undecided ones, still go to SymPy.
- The point of an `Order` moved by a substitution (`O(x).subs(x,
  exp(-y))` and the asymptotic series) is solved by the native
  `solveset` (the first solution: a single element, a complement of
  finite sets, an image of the integers at `n = 0`); SymPy's `solveset`
  is asked only when the native one gives another set. The native solver
  also solves `exp(1/d) = k`, where SymPy's gives a `ConditionSet`.
- The rewrites in terms of `sin` (of `cos`, `tan`, `cot`, `sec`, `csc`,
  `sinh`, `csch`, `exp` and `E**x`) and of `sign` in terms of `Abs` are
  native ports of SymPy's `_eval_rewrite_as_sin` and
  `_eval_rewrite_as_Abs`; the series of a function without an
  `_eval_aseries` of its own in SymPy (`airyai`, ...) at an argument
  going to an infinity raises SymPy's `PoleError` natively (they went to
  SymPy through the bridge).
- `core.bridge.Bridge.is_hypergeometric` (and its SymPy implementation)
  is removed: `Expr.is_hypergeometric` is native and nothing asked it.
- The rewrites of `hyper`, `meijerg`, the pairs of `Piecewise`,
  `IndexedRoot`, `CRootOf`, `RootSum`, `Max`, `Min`, the integrals and
  the sets are decided by the table of SymPy's rules (none for a function
  target but `Max`/`Min` in terms of `Abs`, `Heaviside`, `Piecewise`):
  they are rebuilt natively instead of asking SymPy, which answered
  `None`.
- The numerical value of a `Sum` is native (`concrete._evalf`, a port of
  SymPy's `evalf_sum`, `hypsum` and `check_convergence`): the infinite
  hypergeometric sums by direct summation or Richardson extrapolation,
  SymPy's `ValueError` for the divergent ones (`Sum(1, (i, 1, oo))`), the
  finite sums of a few terms term by term; SymPy's Euler–Maclaurin
  summation is not ported (those sums still go to SymPy). Against SymPy:
  a ratio of the terms with irrational coefficients is summed numerically
  (SymPy truncates the coefficients to integers: its value of
  `Sum(pi**i/factorial(i), (i, 0, oo))` is `exp(3)`).
- The Gruntz algorithm's tractable rewrites of the special functions are
  native: SymPy's helpers `_erfs`, `_eis`, `_besseli`, `_besselk` are the
  nodes `Erfs`, `Eis` (`functions.special.error_functions`), `BesselIs`,
  `BesselKs` (`functions.special.bessel`) with their asymptotic series,
  derivatives and rewrites back (`intractable`), and the rules of `erf`,
  `erfc`, `erfi`, `Ei`, `li`, `Li`, `uppergamma`, `besseli`, `besselk`
  and `fibonacci` are ported (they went to SymPy, whose helpers came back
  wrapped). The asymptotic series (`_eval_aseries`) of `erf`, `erfc`,
  `erfi`, `Ei`, `expint`, `Si`, `Ci`, `besseli` and `besselk` are ported
  too, and `Li` rewrites in terms of `li` natively. Against SymPy: the
  series of `Ei` of a polar argument going to an infinity raises
  `PoleError` (SymPy's forgets the branch term `I*pi`: its series of
  `expint(2, 1/w)` is `-I*pi/w + O(w**3)` and its
  `limit(x*exp(x)*expint(2, x), x, oo)` is `-oo*I`; the limit is 1).
- Polar numbers are decided natively: `equals` of a difference with
  `exp_polar` or `polar_lift` takes SymPy's steps (no sum: `False`; the
  factors one by one; the values with the symbols `0` and `1` compared,
  as SymPy's `is_constant`; a derivative which is never zero) instead of
  SymPy's `equals`, and answers `None` like SymPy for `exp_polar(0) - 1`;
  `exp_polar(z)` of a number with `-pi < im(z) <= pi` has the sign of
  `exp(z)` (`exp_polar(I*pi)` is negative, `sign` of it `-1`, as in
  SymPy) and its value is that of `exp(z)` evaluated exactly;
  `periodic_argument` and `principal_branch` have numerical values (ports
  of their `_eval_evalf`).

- `piecewise_simplify_arguments` simplifies its conditions natively by
  the new `logic.boolsimp.simplify_boolean`, a typed port of SymPy's
  `simplify` of Boolean formulas (`BooleanFunction._eval_simplify`,
  `And._eval_simplify`, `Or._eval_simplify`, `simplify_univariate`, the
  two- and three-term patterns of the relations, `simplify_logic` with
  its relations simplified first), instead of SymPy's `simplify`; only a
  condition with an `Xor` still goes to SymPy (`Xor._eval_simplify`, by
  the algebraic normal form, is not ported). The endpoints of the
  intervals of `simplify_univariate` are decided on the formula, where
  SymPy decides them on the relation of the previous interval:
  SymPy's `simplify(~((x > 0) & (x < 1)))` is `(x >= 1) | (x < 0)`, false
  at `x = 0`.
- The series of `Hyper` and `MeijerG` at an argument going to an infinity
  follow SymPy's generic branch of `Function._eval_nseries` natively
  (ported): `PoleError` when the leading term goes to an infinity (these
  classes have no `_eval_aseries`, as in SymPy), the expansion around the
  logarithmic term otherwise; they went through SymPy's `_eval_nseries`.
- `simplify.ratsimp` loads no SymPy at run time: the division of the
  numerator by the denominator runs over the typed domains `QQ`, `QQ_I`
  (Gaussian coefficients) and `EX` (floats and the other numbers),
  instead of sending those to SymPy's `ratsimp`.
- `ExprWithLimits.doit` (the base class of `Sum` and `Product`, which
  have their own) evaluates its function and bounds natively, as SymPy's
  `Basic.doit`, instead of going through SymPy.
- `concrete.products` loads no SymPy at run time: the product of a
  polynomial whose coefficients our `Poly` does not take (`I`, `sqrt(2)`,
  `pi`, floats) is taken by the closed forms of its roots (the native
  `ex_roots`) instead of sending the whole product to SymPy's
  `Product.doit`; it stays unevaluated when they are not found (a cubic
  or quartic over `EX`, where SymPy has formulas, and degree five or more,
  where SymPy's recursion on the cofactor does not end).
- `concrete.eulersums.polygamma_integral_representation` returns the
  core's unevaluated `Integral` (`integrals.integral`) instead of SymPy's
  `Integral` wrapped, and loads no SymPy; it prints in SymPy's order
  (`-EulerGamma*zeta(s) + Integral(...)`).
- `concrete.delta` loads no SymPy at run time: the equation of a delta
  the linear and finite solvers do not list goes to the native
  `solve_equation`, and the systems of several deltas which are not
  linear with a unique solution to the native `solve_system` (for the
  symbols in SymPy's canonical order), instead of SymPy's `solve`;
  `NotImplementedError` when they have no method, as SymPy raises.
- `concrete.summations` loads no SymPy at run time: the residue method
  takes the poles SymPy writes as `CRootOf` by the residue of a pole of
  order `m` at an algebraic point (SymPy's series there gives
  `Sum(1/(i**4 + i + 1), (i, -oo, oo))` as `0`; the sum is about
  `2.4920`) and the coefficients our `Poly` does not take (`I`, `pi`,
  `sqrt(2)`) by the closed forms of the native `ex_roots` (SymPy raised
  `NotImplementedError` for the sorted roots over `EX`); the Hurwitz zeta
  function of `Sum.doit` is the port of SymPy's `Sum.eval_zeta_function`;
  the equations of the denominators are solved by the native
  `solve_equation` (`NotImplementedError` when it has no method), and a
  convergence condition the native solver does not decide is not taken
  for empty (it went to SymPy's `as_set`).
- `assumptions.solve` loads no SymPy at run time: the systems which the
  regular chains do not solve go to the native `solve_system` (the
  substitution method of SymPy's `nonlinsolve`, the unknowns left free
  standing for themselves) instead of SymPy's `nonlinsolve`, and the
  intersections the sets of the core leave unevaluated are evaluated
  natively (a condition set restricted to the other set, the images
  `a*k + b` of the integers in a bounded interval as a finite set, a finite
  set filtered by its decided memberships) instead of by SymPy's
  constructors; the integers in an interval stay an `Intersection` natively
  (SymPy's face still gives its `Range`). `solve_sympy.nonlinsolve_of_core`
  and `solve_sympy.evaluated_set_of_core` are removed. The points of
  `nonlinsolve` repeated in two radical forms (`sqrt(1 - y**2)` and
  `sqrt(-(y - 1)*(y + 1))`) are given once.

- The relations in one unknown which `assumptions.solve`, the
  transcendental solver and `sympy_extras._special_values` solved with
  SymPy's `solveset` at run time (`assumptions.solve_sympy.solveset_of_core`,
  `solvers.transcendental_sympy.solveset_of_core`, both removed) are solved
  by the new native port `sympy_extras.solvers.solveset` (`solveset`,
  `real_polynomial_solutions`): over the reals the polynomials with
  irrational coefficients (all their roots, the real ones told apart by
  reliable values, the signs between them for an inequality), the
  parametric linear and even binomial inequalities, the transcendental
  relations through `solve_transcendental`, the equations in exponentials
  and logarithms by their checked candidates and `g(u(x)) = c` for a sine,
  a cosine or a tangent; over the complex numbers the rational equations,
  `g(u(x)) = c` for an exponential, trigonometric or hyperbolic `g` (images
  of the integers) and the radical equations. What it does not solve is
  the `ConditionSet` of the relation, as SymPy's answer. The roots which
  are numbers are written `a + b*I`, as SymPy's `_solve_as_poly` does (the
  formulas of Ferrari put a square root on its branch cut). The extraneous
  root `-1` which SymPy's `solveset` gave for `log(x) + log(x - 1) = log(2)`
  over the reals is not returned.

- The solvers which called SymPy's `solve` at run time through the
  `solve_of_core` functions of the faces (`charpit`, `second_order`,
  `lie`, `abel`, `ode`, `pde`, `factorization`, and the Lambert W
  candidates of `transcendental`) use the native port of it, the new
  modules `sympy_extras.solvers.solvers` (`solve_equation`: one equation
  in one unknown by its roots, one generator, the radicals of one base,
  the inversion of the outer functions; `solve_system`: polynomial systems
  by a Gröbner basis and substitution; `invert`, `checksol`) and
  `sympy_extras.solvers.bivariate` (`solve_lambert`, SymPy's Lambert W
  forms), so that these steps load no SymPy. The functions
  `charpit_sympy.solve_of_core`, `factorization_sympy.solve_of_core`,
  `pde_sympy.solve_of_core` and `transcendental_sympy.solve_of_core` are
  removed. The Lambert W candidates of `solve_transcendental` include the
  branch `-1` whenever it is real (SymPy's `solve` gave only the principal
  branch of `x = exp(x) - 2`, which the transcendental solver completed
  already).
- **Breaking**: the method `'manual'` of `sympy_extras.integrals.indefinite`
  is the native `manualintegrate` (in the place SymPy's had);
  `indefinite_sympy.manualintegrate_of_core` is removed and
  `indefinite.SYMPY_METHODS` no longer lists `'manual'`. The native
  `antiderivative` integrates the polynomials in elementary functions and
  the parts of its integrations by parts by `elementary_antiderivative`
  instead of SymPy's `integrate`: `antiderivative_sympy.integrate_of_core`
  is removed, and those antiderivatives load no SymPy
  (`t**2*cos(3*t - 1)` in SymPy's form at once, `t*cos(t)*sinh(t)` in a
  real form).

- **Breaking**: the definite integration driver `sympy_extras.integrals.definite`
  (`definite_integral`, `conditional_integral`, `verify_numerically`) is
  native on the typed core: its functions take and return objects of
  `sympy_extras.core`, and importing it, and the integrals the native
  methods settle, load no SymPy (the native Marichev–Adamchik method,
  residues, tables, contours, antiderivatives, telescoping and the other
  methods of the package called directly; the native `solve`, `ask`, SAT
  solver, Gruntz limits, factorisation and simplification; the port of
  `singularities`; the quadratures evaluate the integrands by the
  numerical rules of the nodes instead of `lambdify`). An integral left
  undone is the node `sympy_extras.integrals.integral.Integral`, now with
  bounds (`Integral(f, x, a, b)`, SymPy's `Integral(f, (x, a, b))` through
  interop; the native `has_integral` of `laplace` and `linear_ode` and the
  special values treat it as unevaluated). The old API on SymPy objects
  is `sympy_extras.integrals.definite_sympy`, which keeps SymPy's
  `integrate` and `meijerint` as the last methods (`integrate_of_core`);
  the verified antiderivatives come from the native `indefinite`. The private helpers
  (`_Integrator`, `_quadrature`, `_positions`, ...) take native objects;
  the native `DefiniteIntegrator` and `definite_integrator` of
  `definite` replace those of `parametric_sympy`. The native modules call
  the native driver directly, and the routes back into it are removed:
  `laplace_sympy.definite_integral_of_core` and
  `conditional_integral_of_core` (with their re-exports in
  `summability_sympy`, `transformations_sympy` and `algebraic_sympy`),
  `telescoping_sympy.definite_integral_of_core` and
  `verify_numerically_of_core`, `residues_sympy.verify_numerically_of_core`,
  `algebraic_sympy.verify_numerically_of_core` and
  `regions_sympy.definite_integral_of_core`.
- The native ODE solvers no longer call SymPy's `dsolve`, its rational
  Riccati solver or its `linodesolve` at run time. The new
  `sympy_extras.solvers.dsolve` (`dsolve`, `explicit_solution`,
  `solution_parts`, `linear_coefficients`, `linear_basis`) ports SymPy's
  classical hints to the core: linear equations with constant
  coefficients (conjugate roots as `sin`/`cos`), Euler's equations, the
  first order linear equation, the reduction to the lowest derivative and
  rational coefficients through `linear_ode`, the particular solutions
  by variation of parameters; separable, Bernoulli, exact and
  homogeneous-coefficient first order equations and factorable ones. It
  replaces SymPy's `dsolve` in `solvers.ode` (`solve_ode`, `dsolve_lie`),
  `solvers.pde`, `solvers.second_order`, `solvers.first_order`
  (`lagrange_ode`), `solvers.abel` (`air_solution`), `solvers.linear_ode`
  (the last resort of `dsolve_linear`) and `integrals.telescoping`. The
  fallback of Kovacic's case 1 is the new `kovacic.rational_riccati`
  (the hyperexponential solutions of `z'' = r z`), and the last resort
  for a fundamental matrix of `linear_systems.dsolve_system` is
  `exp(M g(x))` for `A = g'(x) M` by the native matrix exponential. The
  routes `pde_sympy.dsolve_of_core`, `first_order_sympy.dsolve_of_core`,
  `linear_ode_sympy.dsolve_of_core`, `telescoping_sympy.dsolve_of_core`,
  `kovacic_sympy.riccati_of_core` and
  `linear_systems_sympy.linodesolve_of_core` are removed. The solutions
  are SymPy's or equivalent ones (checked by substitution); a first
  order linear solution keeps SymPy's form `(C1 + x)*exp(x)` when shorter.
  `solvers.ode.check_ode_solution` takes SymPy's `solve_for_func`
  argument (an implicit relation checked through its derivatives without
  being solved).
- Three more of the small run-time routes into SymPy are gone. The
  annihilators of `integrals.dfinite` (`annihilators`, and the equations
  of the factors of `closure` outside its table) no longer come from
  SymPy's `expr_to_holonomic` (`dfinite_sympy.holonomic_of_core` is
  removed): the annihilator of an integrand is the first linear
  dependency among its derivatives in the closure of its factors (sums,
  products, rational arguments and algebraic powers of rational
  functions), so of least order there; the Bessel, Hankel and Airy
  equations are in the table (SymPy gives a third order equation for
  `besseli` and `bessely` and none for the Airy functions), and a factor
  whose derivative is in the closure (`Ei`, `Si`, `Ci`, `atan`, `asin`)
  gets the equation of its derivative composed with `d/du`. The
  coefficients of an annihilator may differ from SymPy's by a factor (a
  scaled argument such as `J_0(t x)` is no longer substituted).
  The antiderivatives of `integrals.parametric` and `solvers.charpit` come
  from the native `indefinite.closed_antiderivative`
  (`parametric_sympy.antiderivative_of_core` and
  `charpit_sympy.antiderivative_of_core` are removed); the integral of
  `atan(p*x)/(x*(1 + x**2))` over `(0, oo)` is now
  `pi*log(p + 1)/2`. The native `assumptions.refine.simplify` of compound
  Boolean formulas no longer calls SymPy's `simplify`
  (`refine_sympy.simplify_boolean_of_core` is removed): the new
  `sympy_extras.assumptions.boolsimp` (`simplify_boolean`,
  `simplify_univariate`) ports the `_eval_simplify` of SymPy's
  `BooleanFunction`, `And` and `Or`, with the intervals of a univariate
  formula computed natively (maximal intervals, and exactly where SymPy
  drops the endpoint of `~((x > -1) & (x < 1))` or gives up on
  `(x**2 > 4) & (x > 0)`).
- The native indefinite integrator no longer tries SymPy's heuristic
  Risch integrator after its own (the method `'sympy heurisch'` and
  `indefinite_sympy.heurisch_of_core` are removed): on the integrals of
  the test suites which reached it, it answered only Piecewise integrands
  with conditions in the variable, taking the Piecewise for a constant
  (`y*Piecewise((1, y > 1), (0, True))`, which jumps at `y = 1`, is now
  `Piecewise((0, y <= 1), (y - 1, True))` by the Meijer route), and on the
  integrands of SymPy's own heurisch tests the native port missed only
  `1/(x + sin(y))`, which it now does (the transcendental constants of
  the parameters, `sin(y)`, `exp(y)`, are independent generators of the
  content of a polynomial, as in SymPy's domains).
- **Breaking**: the verified indefinite integration
  `sympy_extras.integrals.indefinite` (`indefinite_integral`,
  `verified_antiderivative`, `is_antiderivative`, `real_form`,
  `conjugate_logarithms`) is native on the typed core: its functions take
  and return objects of `sympy_extras.core`, and importing it and running
  its native methods loads no SymPy (the native rational integrator, the
  radical, exponential and trigonometric tables, the Risch package,
  Trager's algorithm on the native curves, the binomial differentials, the
  rewritings and substitutions, the heuristic Risch integrator; the checks
  by the native `cancel`, `simplify`, structure theorem and reliable
  values; the cases of the special values by the native
  `sympy_extras._special_values`). An integral which no method does is the
  new unevaluated node `sympy_extras.integrals.integral.Integral`
  (converted to SymPy's `Integral`). SymPy's routes have no native
  counterpart and are the last methods, reached through the new face
  `sympy_extras.integrals.indefinite_sympy` (`heurisch_of_core`,
  `manualintegrate_of_core`, `meijerint_of_core`, `integrate_of_core`), so
  that SymPy loads only when every native method has failed; SymPy's
  heuristic Risch integrator, which ran right after the native one, is
  now the method `'sympy heurisch'` among them. The old API on SymPy
  objects is `sympy_extras.integrals.indefinite_sympy`, which `definite`
  uses. The native `antiderivative` calls the native `indefinite`
  directly: `antiderivative_sympy.verified_antiderivative_of_core` is
  removed. The quadratures of the native ODE solvers (`kovacic`,
  `linear_ode`, `factorization`, `linear_systems`, `special`, `darboux`,
  `abel`, `ode`, `second_order`, `first_order`, `dae`) go through the
  native `indefinite` too (the new `closed_antiderivative`; SymPy's
  `integrate(conds='none')` becomes the generic antiderivative,
  `conds='piecewise'` the one with the cases of the special values), and
  an integral left undone is the native `Integral`:
  `kovacic_sympy.integrate_of_core` and `integral_of_core`,
  `first_order_sympy.integrate_of_core` and `dae_sympy.integral_of_core`
  are removed.
- The exact solutions of a rational univariate representation
  (`polys.rur.RationalUnivariateRepresentation.solutions`, `rur.exact_solutions`),
  of a regular chain (`polys.regularchains.RegularChain.solutions`, the new
  native `regularchains.solutions.exact_solutions`, a port of the face's
  module which matches the roots by their exact values and SymPy's
  `nroots` order) and the numerical poles without short radicals of the
  residue integrator (`integrals.residues.poles`) no longer load SymPy at
  run time: the roots are the native `CRootOf` of `polys.complexrootof` in
  SymPy's order (the real roots of the solutions as `IndexedRoot`). The
  routes `rur_sympy.solutions_of_core`,
  `regularchains_sympy.exact_solutions_of_core` and
  `residues_sympy.complex_roots_of_core` are removed. `IndexedRoot` prints
  its polynomial in the order of the plain variable
  (`IndexedRoot(x**3 - 2*x**2 + x - 1, x, 0)`, which printed
  `-1 + x**3 - 2*x**2 + x`).
- **Breaking**: Trager's algorithm `sympy_extras.integrals.trager`
  (`trager_antiderivative`, `trager_reduce`, `hermite_reduce_algebraic`,
  `is_nonelementary_algebraic`) is native on the typed core: its functions
  take and return objects of `sympy_extras.core`, and importing and running
  it loads no SymPy (the dense polynomials of `polys.polyclasses` over
  `QQ`, number fields of a primitive element of the constants, the fields
  of rational functions of the parameters and `EX`; the native rational
  integrator, Risch differential equation, general curve and Chebyshev
  criterion). The old API on SymPy objects is
  `sympy_extras.integrals.trager_sympy`, which `indefinite` uses; it
  reaches the general curve through `curves_sympy`, so that the residues
  of fields without radicals are still written with SymPy's `CRootOf`
  (the native module leaves those integrals undone). The private helpers
  (`_rational_parameters`, `_root_components`, `_hermite_index_n`, ...)
  take native objects.
- **Breaking**: `solvers.inequalities.solve_inequality` no longer loads
  SymPy at run time: a relational which is not rational in the variable
  is solved by the native `solve_transcendental` (absolute values by
  cases, then the kernels) in place of SymPy's
  `solve_univariate_inequality`, and raises `NotImplementedError` (as
  SymPy did) when the solutions are not a union of intervals and points.
  The module `solvers.inequalities_sympy` (only the route
  `solve_inequality_of_core`) is removed. `solve_transcendental` solves
  the formulas which the absolute values leave rational in the unknown
  (`1/Abs(c) > 1`), which gave `None`.
- The cases of the isolated values of the parameters of the native sums
  (`concrete.karr`, `creative`, `zeilberger`, `qhyper`, `rational`) no
  longer load SymPy at run time (internal change, same API): the
  summation rules of `karr_sympy.with_sum_special_values` (no case where
  the summand is undefined or has a pole in the range, the powers of `0`
  summed by their values, the numerical check against the sum term by
  term) are written on the core over the native
  `_special_values.with_special_values`; the route
  `karr_sympy.sum_special_values_of_core` is gone.
- The last resort of the convergence tests for terms without parameters
  (`concrete.convergence`) no longer loads SymPy at run time (internal
  change, same API): the integral of the integral test comes from a
  native antiderivative (the transcendental Risch algorithm, then the
  heuristic one, checked by differentiation) and the native limits, and
  SymPy's `Sum.is_convergent` is ported (its `False` still counts as "not
  shown"); the routes `convergence_sympy.improper_integral_of_core` and
  `is_convergent_of_core` are gone. The native `ask` no longer prepares
  SymPy's tables for the CAD of its MetiTarski bounds (a time limit
  without `native=True` loaded SymPy).
- The power series of Ramanujan's master theorem and the method of
  brackets (`integrals.brackets`, and `integrals.series` through
  `taylor_coefficient`) no longer load SymPy at run time (internal change,
  same API): the new `brackets.formal_series_terms` finds the coefficients
  of a single residue class as a hypergeometric term, its ratio guessed
  from the native series (`concrete.guess.guess_recurrence`) and proved by
  the differential equation it implies, checked on the function (in place
  of SymPy's `fps`; the Bessel function `besselj(0, x)`, which `fps` did
  not expand, now has its series), and the strip of convergence comes
  from the order at 0 and the behaviour at infinity (powers with
  logarithms, decays faster than every power, and oscillations
  `x**q sin(a x**p)` by Dirichlet's test) in place of the strip of SymPy's
  `mellin_transform`. The routes `brackets_sympy.fps_of_core` and
  `mellin_strip_of_core` are gone.
- The periods of the periodic formulas of `assumptions.solve` and of the
  periodic rule of `integrals.laplace` no longer load SymPy at run time
  (internal change, same API): SymPy's `periodicity` and `lcim` are
  ported to the core as the new `sympy_extras.calculus.util` (the periods
  of the trigonometric functions of linear arguments and the
  decomposition of compositions written down there), and the routes
  `solve_sympy.periodicity_of_core` and `laplace_sympy.periodicity_of_core`
  are gone.
- The limits of sequences (`assumptions.limits.limit_seq`, and the
  convergence tests through it) no longer load SymPy at run time
  (internal change, same API): SymPy's `limit_seq` is ported to the core
  as the new `sympy_extras.series.limitseq` (`limit_seq`,
  `difference_delta`, `dominant`), and the route
  `limits_sympy.limit_seq_of_core` is gone. The core gains the rules it
  needed from SymPy: the power of a negative number to an even or odd
  symbol takes the sign out (`(-2)**n` is `-2**n` for an odd `n`), the
  tractable forms of the limit algorithm for `gamma`, `binomial`, the
  rising and falling factorials and `harmonic` (and the intractable one of
  `loggamma`), and the asymptotic series of `polygamma`. Without SymPy the
  core's `limit(exp(x)/gamma(x), x, oo)` gave `oo` (the tractable form of
  `gamma` was SymPy's rule only, silently skipped); it is `0`.
- The signs of the functions which are not rational
  (`assumptions.analysis.sign_on`, and through it `ask`, `refine`, `solve`,
  `solvers.isolation`, the convergence tests) no longer load SymPy at run
  time (internal change, same API): the continuity on a set is checked
  natively on the subexpressions (the functions of SymPy's
  `continuous_domain`: powers, `exp`, `log`, `Abs`, the trigonometric and
  hyperbolic functions and their inverses, each continuous where the
  signs of its argument allow), the zeros are those of the native
  `solve_transcendental` factor by factor of the common factors, the
  periodic families listed on bounded sets, and the intersections of the
  CAD's solution sets with other intervals are made by certified
  comparisons of their ends. The routes `analysis_sympy.continuous_on_of_core`,
  `finite_zeros_of_core`, `has_discrete_zeros_of_core`,
  `intersection_of_core` and `isolation_sympy.continuous_on_of_core` are
  gone. `solve_transcendental` takes `lambert=False` to skip the Lambert W
  candidates of SymPy's `solve`, reduces `x` outside the roots of a
  linear argument (`x = sqrt(x + 2)`), and no longer rewrites a power of a
  negative base as an exponential: `(-1)**x = 1` gave `{0}` and
  `(-2)**x = 4` no solution; both give `None` now.
- **Breaking**: `polys.roots_sympy.roots_of_core` is removed: the new
  `polys.polyroots.ex_roots` gives the roots of a polynomial with
  coefficients of any kind by SymPy's formulas over `EX` (a root `-1` or
  `1` divided out, then the linear and quadratic formulas with `simplify`;
  nothing from a degree five or more, as SymPy without quintics), which
  the native `roots` now uses for the components of a decomposition and
  `_special_values` for the real roots of `a**2 - pi`; a cubic or a quartic
  over `EX` is not ported (`radical_form` then gives no radical form,
  `_special_values` no candidate).
- **Breaking**: `assumptions.refine_sympy.factor_list_of_core` is removed (the
  native factorisation raises `CoercionFailed`, never `NotImplementedError`,
  for coefficients which are not rational, so the route was not reached);
  the native `simplify` of a relation which is not polynomial in real
  variables is the new `assumptions.refine.simplify_relational`, a port of
  SymPy's `Relational._eval_simplify` (the sides simplified, solved for a
  single symbol when linear, the coefficients made coprime), the
  difference of two numbers decided by a reliable value rather than to
  two digits. `simplify_boolean_of_core` stays for compound formulas
  (SymPy's `simplify_univariate` and the patterns of `And` and `Or`).
- **Breaking**: `sympy_extras.polys.polytools_sympy` (`invert_of_core`,
  `cancel_of_core`, `factor_of_core`) is removed: the native `cancel`
  cancels over `CC` (the gcd over `QQ_I`, the cofactors rounded back, as
  SymPy's `PolyElement.cancel` of an inexact domain), over `EX` (numbers
  such as `zoo` or `oo` among the coefficients; a zero `EX` cannot detect
  leaves the fraction as it is where SymPy raises `PolynomialDivisionFailed`)
  and with generators over a composite domain (the other factors as
  elements of `EX`, the same rational function as SymPy's in another form,
  and a generator which is not a factor of the expression, `x**2` in `x**3`,
  through the same path); `factor` with generators over `ZZ_I`, `QQ_I` and
  `RR` factors in every symbol, factors `zoo`, `oo` and `nan` among the
  coefficients as generators (SymPy's `ZZ[zoo]`), and leaves a polynomial
  over `CC` or `EX` otherwise as it is (SymPy's `DomainError`); `invert` of two numbers follows
  SymPy's `Domain.invert` over `ZZ_I`, `QQ_I` and `RR` (the `TypeError` of
  SymPy over `CC`). The dense pseudo-division raises
  `PolynomialDivisionFailed` when the degree does not drop, as SymPy's.
- **Breaking**: `solvers.isolation_sympy.real_roots_of_core` is removed:
  the real roots of a polynomial with rational coefficients which are not
  in radicals are the native canonical `IndexedRoot` of their minimal
  polynomial (SymPy's `CRootOf` through `interop`), ordered exactly as
  real algebraic numbers, so that `isolate_real_roots` no longer loads
  SymPy for them.
- **Breaking**: `simplify_logic_of_core` is removed from
  `polys.virtual_substitution_sympy` and `solvers.integers_sympy`: the
  results of virtual substitution and of Cooper's algorithm, and the joined
  conditions of the cases of the ODE solvers (`solvers._special_values`),
  are simplified by the new native `sympy_extras.logic.boolalg`
  (`simplify_logic`, `SOPform`, `POSform`, ported from SymPy's
  Quine–McCluskey minimisation), without loading SymPy. The relations are
  numbered in `ordered` order instead of SymPy's hash order, so that an
  equally small form may be chosen where SymPy's depended on the hashes.
- **Breaking**: the transcendental Risch package `sympy_extras.integrals.risch`
  (`risch.py`, `rde.py`, `prde.py`, `cde.py`, `polymatrix.py`) is native on
  the typed core: its functions take and return objects of
  `sympy_extras.core` and the new `sympy_extras.integrals.risch.tower`
  (`TowerPoly`, a polynomial of the tower in one generator, over
  `RationalFunction`, an element of `C(x, t1, ..., tn)` with exact
  arithmetic over `QQ`, `QQ(I)` and the square roots of integers, in place
  of SymPy's `Poly` over `ZZ(x, t0)`, `QQ_I(x)` and `EX`); `PolyMatrix`
  becomes `FieldMatrix`; `DifferentialExtension(extension={'D': ...})`
  becomes `DifferentialExtension.manual(D, exts=..., extargs=..., Tfuncs=...)`;
  `derivation(p, DE, basic=True)` becomes `derivation_element(a, DE)`;
  `risch_integrate(..., separate_integral=True)` becomes
  `risch_integrate_separate`; `as_poly_1t` returns a `LaurentPolynomial`;
  the degree bounds are integers (`rde.OO` for `+oo`); `NonElementaryIntegral`
  is a node of the core (SymPy's, a subclass of `Integral`, is in
  `risch_sympy`, and `interop` converts both ways). Importing and running it
  loads no SymPy. The SymPy face `risch/risch_sympy.py` keeps
  `risch_antiderivative`, `is_nonelementary` and adds `risch_integrate`
  (SymPy's signature) and `rational_risch_de` (used by `trager`).
  Constants other than `I` and square roots of integers (cube roots, nested
  radicals, `CRootOf`) make the algorithm give up (`NotImplementedError`).
- `rde.weak_normalizer` raises each gcd to the power of its residue
  (Bronstein's WeakNormalizer): SymPy's took it once per root of the
  resultant, which leaves `3/(t - 1)` over `t = exp(x)` with the residue 2.

### Added

- `sympy_extras.polys.complexrootof`: a native node for the complex roots
  of polynomials with rational coefficients, `ComplexRootOf` (alias
  `CRootOf`; SymPy's `CRootOf`, ported), which never evaluates, and the
  evaluating `rootof`, `all_roots`, `real_roots`, `polynomial_roots`,
  `root_count`, `algebraic_number`. The roots are in SymPy's order (the
  real ones increasing, then the others factor by factor, each root after
  its conjugate, the pairs by SymPy's Collins–Krandick rectangles, which
  `complex_rectangles` reproduces from exact isolating regions); the
  numerical values are refined exactly to any precision (a real part on
  the imaginary axis is exactly zero). `sympy_extras.interop` converts
  SymPy's `CRootOf` to the node and back (a `CRootOf` coming from SymPy
  was a wrapped foreign node), `polys.roots.polynomial_root` and
  `radical_form` recognise it, and the native `minimal_polynomial` knows
  it and the `IndexedRoot` of the CAD.
- `sympy_extras.polys.berlekampmassey`: `berlekamp_massey` (the shortest
  linear recurrence of a sequence over any field of the typed layer),
  `extend_sequence`, `satisfies_recurrence`.
- `sympy_extras.polys.domains.galoisfield.GaloisField`: the finite fields
  `GF(p**n)` as a typed coefficient domain (Frobenius map, `p`-th roots,
  trace, enumeration), and `sympy_extras.polys.galoisfactor`:
  `gfq_factor`, `gfq_sqf_list`, `gfq_ddf`, `gfq_edf`,
  `gfq_irreducible_p`, factorisation of univariate polynomials over
  them (Cantor–Zassenhaus). No SymPy counterpart.
- `sympy_extras.polys.numberfields.basis`: `integral_basis` and
  `field_discriminant` of a number field by the Round 2 algorithm
  (p-radicals and their multiplier rings, Hermite normal forms), and
  `polynomial_discriminant`. Typed, no SymPy.
- `sympy_extras.polys.zerodim.solve_zero_dimensional`: all the complex
  solutions of a polynomial system over `QQ` with finitely many, with
  exact coordinates (`Algebraic`): radical by Seidenberg's lemma, FGLM,
  the shape lemma with a change of the last coordinate when needed;
  `univariate_in_ideal`; `Algebraic.polynomial_value` (a rational
  polynomial at an algebraic number by one resultant). Typed, no SymPy.
- `sympy_extras.polys.numberfields.qqbar.primitive_element(a, b)`: a
  primitive element `theta = a + c b` of the field of two algebraic
  numbers, with `a` and `b` as polynomials in `theta` (a gcd over
  `QQ(theta)`).
- `sympy_extras.polys.fglm`: `fglm` (change of monomial order for
  zero-dimensional ideals on the typed rings, by linear algebra on normal
  forms), `groebner_by_fglm` (F4 for `grevlex`, then the conversion) and
  `is_zero_dimensional`. Typed, no SymPy; `Ideal` still calls SymPy's
  `matrix_fglm` on its own rings.
- `sympy_extras.solvers.simplex`: `linear_program` (exact two-phase
  simplex over the rationals with Bland's rule; every result carries its
  certificate: optimal dual multipliers, a Farkas combination, or a ray)
  and `linear_feasible` (systems with strict inequalities);
  `integer_program`, branch and bound on the exact relaxations inside a
  box proved to contain an optimum (proximity theorem of Cook, Gerards,
  Schrijver and Tardos), so that it terminates on unbounded polyhedra
  too; `eliminate_variables`, Fourier–Motzkin elimination with strict
  inequalities and the redundant rows removed by linear programs
  (`LinearSystem`). Typed, no SymPy.
- `sympy_extras.polys.numberfields.recognize.recognize_algebraic`: the
  algebraic number (an exact `Algebraic`) behind a decimal approximation,
  real or complex, by LLL reduction on the powers of the approximation
  (Kannan–Lenstra–Lovász); `None` when no short relation exists.
- `sympy_extras.polys.numberfields.qqbar`: `Algebraic`, exact arithmetic
  on algebraic numbers (minimal polynomial and isolating interval or
  rectangle): `+ - * /`, integer powers, conjugates, real and imaginary
  parts, exact equality, order and signs of the real ones, enclosures of
  any precision, principal roots (`nth_root`, `sqrt`);
  `algebraic_roots(f)`, the roots of a polynomial. The
  results are identified among the roots of a resultant by interval
  arithmetic and winding numbers. Typed, no SymPy.
- `sympy_extras.polys.complexroots`: `dup_count_roots_right_of` (the roots
  on one side of a vertical line), `dup_count_roots_in_disc`,
  `dup_hurwitz_stable_p` (Routh–Hurwitz) and `dup_schur_stable_p`
  (Schur–Cohn), exact, by Cauchy indices on a line.
- `sympy_extras.solvers.exponential_diophantine`: `pillai(a, b, c)`, all
  the solutions in non-negative integers of `a**x - b**y = c` (the
  Baker–Wüstholz bound for two logarithms, reduced by the continued
  fraction of `log b/log a`, then an exact search; dependent bases
  apart; with coefficients, `pillai(a, b, c, r, s)` for `r a**x - s b**y
  = c`, three logarithms and the Baker–Davenport reduction), and
  `exponential_polynomial_roots`, all the integer zeros of
  `sum p_i(n) a_i**n`. Typed, no SymPy.
- `sympy_extras.polys.lattice`: `integer_kernel` (an LLL reduced basis of
  the integer solutions of `A x = 0`), `solve_integer_system` (all the
  integer solutions of `A x = b`: a short particular solution by Babai's
  nearest plane algorithm and the kernel; `None` when there is none) and
  `smith_normal_form` (the invariant factors).
- `sympy_extras.concrete.guess`: `guess_recurrence`,
  `guess_differential_equation`, `guess_algebraic_equation` (the equation
  with polynomial coefficients, with the fewest coefficients, which the
  first terms of a sequence or their generating function satisfy, by
  linear algebra with a margin of surplus equations; modular search,
  exact null space) and `guess_closed_form` (the guessed recurrence
  solved by `rsolve_linear` and checked on the terms). Native.
- `sympy_extras.polys.complexroots`: isolation of the complex roots of
  polynomials with rational coefficients in rectangles with rational
  corners (`dup_isolate_complex_roots_sqf`, `dup_isolate_complex_roots`
  with multiplicities, `dup_isolate_all_roots_sqf`), exact counting in a
  rectangle by winding numbers from Sturm sequences
  (`dup_count_complex_roots`; a root on the boundary is detected),
  refinement by Newton's iteration certified by Rouché's theorem
  (`dup_refine_complex_root`). Typed, integers only, no SymPy.
- `sympy_extras.polys.lattice`: `lll`, `lll_reduction` (the LLL algorithm
  in de Weger's all-integer form, with the Gram determinants and
  `short_part`), `hermite_normal_form`, `is_lll_reduced`. Exact on entries
  of any size, dependent generators allowed; `thue_equation.lll` is now
  this function (it recomputed a rational Gram–Schmidt after every step).
- `sympy_extras.polys.factortools.dup_zz_van_hoeij`: van Hoeij's knapsack
  recombination of the modular factors by lattice reduction, used by
  `dup_zz_factor` from ten modular factors on instead of the subset
  search (exponential on polynomials which split modulo every prime: the
  Swinnerton-Dyer polynomial of degree 32 in 0.04 s instead of 2 s, of
  degree 128 in 2 s).
- `sympy_extras.polys.gaussianfactor`: factorisation of multivariate
  polynomials over `ZZ_I` and `QQ_I` by Trager's algorithm over `QQ<I>`,
  ported from SymPy's `dmp_zz_i_factor`, `dmp_qq_i_factor`,
  `dmp_ext_factor`, `dmp_sqf_norm` and `_sort_factors`; `factor` of
  expressions with Gaussian coefficients in several generators (trigsimp,
  `Order`) no longer goes through SymPy, with SymPy's results (checked by
  `srepr` on 100 random cases) and about ten times faster. The univariate
  factors over `ZZ_I` are sorted as SymPy sorts them (imaginary parts
  first).
- `sympy_extras.polys.polytools.gcd_list` and `lcm_list`, native (folding
  `gcd`/`lcm`): the same results as SymPy's, structurally.
- `sympy_extras.solvers.linsolve.solve_linear_unique`: the unique solution
  of a linear system with coefficients in `QQ(other symbols)`, by the
  reduced echelon form over the native fraction field (`None` when not
  unique, not linear, or outside the field); checked against SymPy's
  `linsolve` on random systems. Building blocks for the native `creative`.

- `sympy_extras.polys.rationalsolve.solveset_rational`: SymPy's `solveset`
  over the complex numbers for the equations rational in the unknown
  (same solutions in the same order); `Order`'s change of variables
  uses it and asks the SymPy bridge only for the other equations.
- `Expr.is_hypergeometric` is native (SymPy's test: the consecutive ratio
  is a rational function), no longer through the SymPy bridge;
  `sympy_extras.concrete.gosper.consecutive_ratio`, the steps of
  `hypersimp` before its final simplification.
- `sympy_extras.polys.numberfields.minimal_polynomial` (and `minpoly`,
  `minimal_polynomial_coeffs`): minimal polynomials of algebraic numbers
  over `QQ` by resultants on the typed polynomial layer, ported from
  SymPy's `minpoly` (numbers only); `AlgebraicNumber.minpoly` no longer
  goes through the SymPy bridge.
- `sympy_extras.polys.polytools.Poly` with `degree`, `LC`, `gcd`, `lcm`,
  `resultant`, `discriminant`, `factor_list`, `sqf_list`, and
  `sympy_extras.polys.polyerrors`, ported from SymPy onto the typed
  polynomial layer (domains `ZZ`, `QQ`, `ZZ[a, ...]`, `QQ[a, ...]`);
  `SparsePolynomialRing`, `RationalPolynomialRing`,
  `IteratedPolynomialRing` domains and `dup_prs_resultant`.
- `cancel` and `cancel_fraction` on the typed polynomial layer (ported
  from SymPy, with its `signsimp` and `factor_terms(radical=True)`
  preprocessing).
- `sympy_extras.simplify.signsimp` (`signsimp`, `sub_pre`, `sub_post`) and
  `sympy_extras.polys.heuristicgcd` (`heugcd`), ported from SymPy;
  `IntegerPolynomialRing`.
- `sympy_extras.polys.rationaltools.together`, ported from SymPy.
- Multivariate factorisation over `ZZ` (`sympy_extras.polys.multifactor`,
  Wang's algorithm), `Poly.factor_list` over `ZZ`, `QQ`, `ZZ[a]`, `QQ[a]`
  in any number of generators, and the expression-level `factor`,
  `factor_list` and `factor_list_fraction`, ported from SymPy.
- `Poly.sqf_list` in several generators and over `ZZ[a]`/`QQ[a]`
  (`sympy_extras.polys.multisqf`), `sqf`, `sqf_list` and
  `sqf_list_fraction` on expressions, ported from SymPy.
- `sympy_extras.polys.polyroots.roots` (linear, quadratic, binomial,
  cyclotomic, cubic, quartic, decomposed and factorable polynomials),
  ported from SymPy; `expand` repeats its multinomial
  and mul passes until nothing changes, as SymPy's does.
- `sympy_extras.polys.partfrac.apart`, `Poly.cancel` and division over
  `ZZ[a]`/`QQ[a]` by a divisor with a numeric leading coefficient, ported
  from SymPy.

- `sympy_extras.simplify.radsimp`: `fraction`, `numer`, `denom`,
  `collect`, `collect_dict` (SymPy's `evaluate=False`) and `rcollect`,
  ported from SymPy.

- `sympy_extras.simplify.powsimp.powsimp` and `powdenest`,
  `sympy_extras.simplify.simplify.logcombine` and `posify`, ported from
  SymPy; `expand_log` (with `factor`) is native, no longer through SymPy.
- `sympy_extras.simplify.sqrtdenest.sqrtdenest`, `radsimp`,
  `rad_rationalize`, `split_surds` and `signsimp(evaluate=False)`, ported
  from SymPy; `gcd` and `lcm` of two numbers.
- `sympy_extras.simplify.simplify.nsimplify` (pslq, `mpmath.identify`
  read by a typed parser, constants, `rational`, `tolerance`), ported from
  SymPy; `radsimp` no longer goes through SymPy.
- Native `rewrite(gamma)` rules of `factorial`, `subfactorial`,
  `factorial2`, `rf`, `ff`, `binomial`, `catalan` and `beta`, with the
  `piecewise` hint (`rewrite(Gamma, piecewise=False)`); `combsimp` uses it
  in place of its own rewrite and SymPy's.
- `rewrite` is native when its targets are typed function classes
  (`sympy_extras.core.rewriting`): the rules of the trigonometric,
  hyperbolic and exponential functions and of powers in terms of `exp` and
  `cos` are a typed table; a node without a ported rule is rewritten by
  SymPy, that node only.
- `inversecombine` and `kroneckersimp` in `sympy_extras.simplify.simplify`,
  ported from SymPy (used by `simplify` in place of SymPy's).
- `sympy_extras.simplify.simplify.simplify` and `collect_abs`, ported from
  SymPy; `Expr.simplify` uses it. The steps for Integral, Sum, Product,
  Bessel functions, `hyper`, Kronecker deltas, Piecewise conditions and
  noncommutative products still go through SymPy.
- Native series machinery in the core, ported from SymPy: the evaluation
  of `Order` (`order`, `contains`, the substitution of a variable moving
  the point), the absorption of terms in sums and products with order
  terms, the `series`/`lseries` driver (`sympy_extras.core.series`),
  `nseries`, `as_leading_term`, `leadterm`, `dir`, `getn`, and the
  `_eval_nseries`/`_eval_as_leading_term` of `Add`, `Mul`, `Pow` and the
  generic function. The functions whose expansion is not ported yet, the
  limits and the asymptotic series still go through SymPy, node by node.
  `as_coeff_exponent` now collects first, as SymPy does. The series and
  leading terms of `exp` and `log` (with the branch cut of `log`) are native.
  So are those of `sin`, `cos`, `tan`, `cot`, `sec`, `csc` and of the
  hyperbolic functions (`sympy_extras.functions.elementary._series`).
  And those of `asin`, `acos`, `atan`, `acot` (branch points and cuts
  included). `expand` expands `(a + b)**(-p/q)` with `p > q` as
  `1/expand((a + b)**(p/q))`, as SymPy does.
  The series of `asec`, `acsc` and of the inverse hyperbolic functions are
  native too. Core fixes found by the comparisons: `sqrt(a*I)` is
  `sqrt(|a|/2)*(1 ± I)` for rational `a` with `|a|/2` a square (SymPy's
  `Mul._eval_power`), `log(r + i*I)` with `atan(i/r)` a rational multiple
  of pi evaluates (`log(1/2 + sqrt(3)*I/2)` is `I*pi/3`), and `cancel`
  keeps the powers of `I` as their own generators and preprocesses once.
- Native limits, ported from SymPy: `sympy_extras.core.gruntz` (Gruntz's
  algorithm: mrv sets, the rewrite in the most rapidly varying `w`,
  `limitinf`, `sign`) and `sympy_extras.core.limits` (the `Limit` node,
  `limit` with the meromorphic and leading-term shortcuts, the
  heuristics); the tractable rewrite of the functions (elementary ones
  here, special ones by SymPy's rules). `Expr.limit`, the series and
  `Order.contains` use them. Accumulation bounds are not ported (the
  limits of oscillating functions come from SymPy's leading terms).
- `Expr.is_meromorphic` is native (the rules of `Add`, `Mul`, `Pow`, the
  functions with their `singularities`, and the atoms), ported from
  SymPy; the rules of the Bessel and incomplete gamma functions and of
  `Piecewise` are SymPy's, node by node. `sympy_extras.core.fuzzy` has
  the typed three-valued `fuzzy_and`, `fuzzy_or`, `fuzzy_not`,
  `fuzzy_group`.
- `Expr.aseries` (`sympy_extras.core.series.aseries`) is native, ported
  from SymPy on the native mrv sets; the series at infinity uses it.
- Accumulation bounds, ported from SymPy: `sympy_extras.core.accumbounds`
  (`AccumBounds`, `accum_bounds`, the interval arithmetic), the bounds as
  the coefficient of sums and products, `E**AccumBounds`, and the bounds
  of `sin`, `cos`, `tan`, `cot` at infinity and of bounds, of `exp`,
  `log`, `frac(oo)`, `atan(zoo)`, `atanh(zoo)`, `asech(zoo)`; the limits
  of oscillating functions are native now.
- The Gaussian domains `ZZ_I` and `QQ_I`, ported from SymPy
  (`sympy_extras.polys.domains.gaussiandomains`: Euclidean division,
  canonical units by quadrant, gcd, gcdex, lcm); `cancel` works over them
  as SymPy's `PolyElement.cancel` does (no SymPy fallback, no generator
  standing for `I` any more).
- `factor` of univariate polynomials with Gaussian coefficients, ported
  from SymPy (`dup_zz_i_factor`, `dup_qq_i_factor`: Trager's algorithm
  over `QQ<I>`, the contents made Gaussian integers); several generators
  with `I` are still factored by SymPy.
- `cancel` over `RR` (float coefficients), ported from SymPy: the gcd of
  the floats made rational (`RealField.to_rational`) over `QQ`, the
  cofactors rounded back, a term dividing its monomial gcd with the floats
  kept. `as_coeff_mul` has SymPy's default `rational=True` (a float is a
  factor, not the coefficient), which `gcd_terms` and `factor_terms` rely
  on (`0.5*y + 0.5` is `0.5*(y + 1)`).
- `factor` over `RR`, ported from SymPy (`dmp_factor_list` of an inexact
  domain: the floats made rational, factored over `QQ`, each factor
  divided by its largest coefficient, rounded back).
- `expand` is fully native: the `power_base` hint (SymPy's
  `Pow._eval_expand_power_base`, with `force`) was missing and is ported,
  `power_exp` splits when the base is nonzero or the exponent's terms
  have one sign (not only for numerical bases), `complex` combines with
  `func`/`trig` in SymPy's order, and the derivative of `Order` is native.
- `cancel(f, *gens)` is native (the ring of `gens` over `ZZ[params]`, each
  coefficient a polynomial of its own: `x*(y + 1)`), and
  `Expr.collect(..., evaluate=False)` returns the typed dictionary of
  `collect_dict` (overloads on `evaluate`).
- `adjoint` and `transpose` are native for noncommutative expressions
  (reversed products, the rules of `exp`, `exp_polar`, `Piecewise`), and a
  function is commutative only when its arguments are (SymPy's
  `Function._eval_is_commutative`: `exp(A)` of a noncommutative `A` is
  not); `invert` with its only generator is native.
- `refine` is native, ported from SymPy (`sympy.assumptions.refine`: the
  handlers of `Abs`, `Pow`, `atan2`, `re`, `im`, `arg`, `sign` and the
  `_eval_refine` of symbols, powers, `exp` and booleans, on the native
  `ask`).
- `nc_simplify` is native, ported from SymPy (repeated subwords of
  noncommutative products grouped into powers; `simplify` uses it); the
  printer keeps the negative powers of noncommutative factors as powers
  (`(b*a)**(-1)*c*a`, not `c*a/(b*a)`), as SymPy does.
- `factor_nc` and `_mask_nc` are native, ported from SymPy; the printer
  orders the terms of noncommutative sums as SymPy's `as_terms` (the
  noncommutative factors are not generators: `x*A*(1 + B)`).
- `besselsimp` is native, ported from SymPy (the factors `I` and `-1` of
  the arguments, the half-integer and integer orders expanded, the
  recurrence of the lowest orders); `simplify` uses it.
- Dummies are ordered by creation, as in SymPy: their sort key is
  `(2, (name, dummy_index))`, and polynomial generators which print alike
  are ordered by creation (SymPy's dummies print with their number).
  Before, two dummies tied and results such as `factor_nc` depended on
  how many dummies existed (`(A - B)**2` or `(-A + B)**2`).
- hyperexpand groundwork (the port of `sympy.simplify.hyperexpand` is in
  parts): the `HyperRep_*` functions with their closed forms
  (`rewrite_nonrep`), a small immutable `ExprMatrix`
  (`sympy_extras.matrices`), the formula tables, the parameter buckets,
  the formula lookup and the shift operators with `reduce_order`, in
  `sympy_extras.simplify.hyperexpand`.
- `lowergamma` and `uppergamma` take the branch factor out of a polar
  argument as SymPy does; a rational power of a product takes out the
  polar factors as it takes out the positive ones.
- hyperexpand part 3: `devise_plan`, `try_shifted_sum`, `try_polynomial`,
  `try_lerchphi`, `build_hypergeometric_formula`, `hyperexpand_special`
  and `_hyperexpand` (the expansion of a hypergeometric function); the
  rational functions of the plans are cancelled with the coefficient
  atoms (`pi`, `gamma(a)`, ...) as generators, as SymPy's composite
  domains do. 212 of 225 expansions compared match SymPy; the others
  wait for the polar branch rules of the Bessel functions and the
  trigonometric integrals.
- `log(exp_polar(x))` is `x` unpolarified; `polylog` unpolarifies an
  argument in the unit disk (SymPy's rules).
- The polar branch rules of `besselj`, `besseli` (integer orders lose the
  polar factors, other orders take out `exp(2*n*pi*nu*I)`), of `Si`, `Ci`,
  `Shi`, `Chi` (`polar_lift(I)` and `polar_lift(-1)` taken out, the
  branch factor), of `Ei` and `expint`; 630 values on polar arguments
  match SymPy. hyperexpand matches SymPy on 215 of 225 expansions.
- `hyperexpand` is native (the last part of the port: `devise_plan_meijer`,
  `_meijergexpand` with Slater's theorem, `hyperexpand`), and `simplify`
  uses it; `residue` is native (`sympy_extras.series`).
- Native sets (`sympy_extras.sets`, ported from SymPy): `EmptySet`,
  `UniversalSet`, `Interval`, `Reals`, `FiniteSet`, `Union`,
  `Intersection`, `Complement`, `Contains`, with SymPy's union,
  intersection and subset handlers for them; interop converts them both
  ways (sets not ported stay SymPy nodes). On random combinations they
  match SymPy except where SymPy raises.
- `x < oo` and `-oo < x` are true for a finite extended real `x`, as in
  SymPy.
- `piecewise_simplify` is native (ported from SymPy with
  `Piecewise._intervals` as `piecewise_intervals`): the conditions of a
  univariate Piecewise rewritten as disjoint intervals, the equations of
  the conditions substituted, equal pieces merged; `simplify` uses it.
  `solve_inequality` (SymPy's `_solve_inequality`, linear case native,
  the nonlinear one through SymPy) in `sympy_extras.solvers.inequalities`.
  The conditions of several variables are simplified through SymPy.
- A relation is decided from the sign of the difference only between
  extended reals (or comparable numbers), as in SymPy: `x + 1 > x` stays
  for an `x` not known real.
- The relations sort by SymPy's class names (`LessThan` before
  `StrictGreaterThan`), so `And`/`Or` keep SymPy's argument order.
- `simplify` of a `Piecewise` inside an expression simplifies its pieces
  (it broke the pairs into tuples).
- `simplify` of a hypergeometric function, a Bessel function or a
  `Piecewise` node uses their native `_eval_simplify` (`hyperexpand`,
  `besselsimp`, `piecewise_simplify`); a Meijer G-function takes the
  common path, as in SymPy. Only the SymPy wrapper nodes still go to
  SymPy's `simplify`.
- Native `Sum` and `Product` (`sympy_extras.concrete.expr_with_limits`,
  ported from SymPy): unevaluated classes with `sum_`/`product_`
  building them as SymPy's constructors (nested sums merged, Piecewise
  folded), the bound variables in `free_symbols` and substitution, SymPy's
  printing, interop both ways; `doit` through SymPy. `factor_terms` takes
  the factors free of the variables out of a sum (SymPy's
  `_factor_sum_int`).
- `sum_simplify`, `sum_combine`, `sum_add`, `factor_sum`,
  `product_simplify`, `product_mul` and the `_eval_simplify` of `Sum` and
  `Product` are native (ported from SymPy); `simplify` no longer sends
  sums and products to SymPy (integrals still do). `expand` splits a sum
  of a sum into sums when the limits may be finite (SymPy's
  `AddWithLimits._eval_expand_basic`).
- `Sum.doit` and `summation` are native (`sympy_extras.concrete.summations`,
  ported from SymPy: `eval_sum`, `eval_sum_direct`, `eval_sum_symbolic`,
  `telescopic`); Gosper's algorithm, the hypergeometric and residue
  methods, the Kronecker delta sums, the equation solving of the
  telescoping and the zeta function match go through SymPy for that call.
- A pattern `c*w` (a number times a single factor) matches `e` with `w`
  matched against `e/c`, as SymPy's `Mul._matches_simple`
  (`(x + 1).match(2*z)` gives `z = x/2 + 1/2`).
- Gosper's algorithm is native (`sympy_extras.concrete.gosper`, ported
  from SymPy: `gosper_normal`, `gosper_term`, `gosper_sum`, with
  `hypersimp`): the polynomials are coefficient lists over the cancelled
  expressions (the field of the parameters), the resultant a Sylvester
  determinant; `summation` uses it.
- `eval_sum_hyper` is native (ported from SymPy with `_eval_sum_hyper`):
  the sum as a hypergeometric function expanded by the native
  `hyperexpand`, on its condition of convergence; `Hyper` has `eta` and
  `convergence_statement`. Only the residue method and the Kronecker delta
  sums of `summation` still go through SymPy.
- `eval_sum_residue` is native (ported from SymPy): the poles from the
  native `roots` where SymPy has closed forms (factors of degree at most
  2, binomials); with `CRootOf` poles the call goes through SymPy.
- `Product.doit` and `product` are native (`sympy_extras.concrete.products`,
  ported from SymPy: `_eval_product` with polynomials by their roots and
  rising factorials, products of factors and powers piece by piece, the
  exponents summed by the native `summation`).
- `deltasummation` and `deltaproduct` are native
  (`sympy_extras.concrete.delta`, ported from SymPy; the equations of the
  deltas solved through SymPy); `eval_sum` factors the nested sums and
  takes the delta method only for a delta linear in the index, as SymPy.
  A positive power of `KroneckerDelta` is the delta. `Relational.canonical`
  orders the sides as SymPy's `ordered` (fewer nodes first: `i/2 <= n` is
  `n >= i/2`).
- SymPy's generic `Function._eval_nseries` is native (`_generic_function_nseries`):
  the expansion by derivatives or by Taylor terms, for the functions whose
  SymPy class uses it and whose expansions were compared equal
  (`generic_nseries = True` on 45 classes: erf, erfi, erfc and their
  inverses, the Fresnel integrals, digamma, trigamma, harmonic, beta,
  Heaviside, re/im, the orthogonal polynomials of several parameters, ...);
  the others still use SymPy's. `Expr.taylor_term` evaluates its derivative
  at the dummy (right for an expression argument: `erfi(2*x)`).
- `Derivative` substitution follows SymPy's `Derivative._eval_subs`: a
  variable replaced by what is not a symbol, a captured symbol or a
  renaming onto a symbol of the expression gives `Subs`, otherwise the
  derivative is rebuilt unevaluated. The generic series names its
  variable as SymPy (`xi`, `xi0`, ...) and goes by derivatives for the
  classes SymPy takes as variadic (`HyperRep_*`); 97 classes expand
  natively.
- `Mul.as_coeff_Mul` takes the first argument as the coefficient, as SymPy
  (`zoo` is not one: `zoo*a + 2*zoo*a` is `3*zoo*a`); `as_independent`
  tests the symbols by the free symbols (a bound variable of `Subs` is no
  dependence); `Subs` substitution as SymPy's `Subs._eval_subs`; the
  derivative of `exp_polar` stays unevaluated, as in SymPy.
- SymPy's generic `Function._eval_as_leading_term` is native for the 107
  classes using it (the function itself, `NotImplementedError` when an
  argument vanishes); the classes with a rule of their own in SymPy still
  use SymPy's.
- Native leading terms for LambertW, factorial, binomial, gamma, polygamma,
  digamma, trigamma, zeta, erf, erfc, erfi, Ei, Si, Ci, Shi, Chi, fresnels
  and fresnelc (`sympy_extras.functions._leading_terms`, SymPy's own
  `_eval_as_leading_term` rules).
- Native leading terms for arg, floor, ceiling, frac, the Bessel functions
  J, Y, I, K, SingularityFunction, Derivative, Subs, Piecewise, hyper and
  meijerg; leading terms no longer go through SymPy except for the
  wrapped SymPy nodes.
- Native series for LambertW, Abs, sign, arg, sinc, gamma, loggamma,
  expint, Ei, Si, Ci, Shi, Chi, li, Li, SingularityFunction, elliptic_k
  and elliptic_e (`sympy_extras.functions._series_rules`, SymPy's own
  `_eval_nseries` rules). Unlike SymPy (bugs on issue #25): `sinc`
  expands in the variable, `SingularityFunction` to the order asked, and
  the series of `expint(n, x)` for integer `n > 1` has no spurious
  imaginary term.
- Native series for `hyper` (SymPy's rule) and `meijerg` (the generic
  expansion, now accepting any node); `Basic.sort_key` is computed once
  per node (`_compute_sort_key` in the classes): the nested keys were
  recomputed at every level, `hyperexpand(hyper([4], [6], 1))` went from
  59 s to 0.6 s.
- Native series for floor, ceiling, frac, polylog and the Bessel functions
  J, Y, I, K (SymPy's rules); no series rule of a ported function goes
  through SymPy any more. `Order` has no `real` fact, as in SymPy
  (`AccumBounds(0, 1) + O(x**3)` stays a sum: `frac(1/x).series(x)`
  failed).
- `meijerg` derivatives in the parameters (SymPy's `_diff_wrt_parameter`);
  `lowergamma` and `uppergamma` derivatives in the first argument; the
  Taylor terms of `airyai` and `airybi`: their series are native.
- `sympy_extras.concrete.eulersums` (`euler_sum`, `polygamma_series`,
  `polygamma_integral_representation`) runs on the native core: native
  expressions in and out (the integral is SymPy's `Integral`, wrapped);
  `integrals.series` converts at the boundary.
- `SparseFractionField` (`sympy_extras.polys.domains.sparsefractionfield`):
  typed fields of rational functions in several variables over the
  sparse polynomial rings (SymPy's `FracField`), with `cancel` on the
  rings (SymPy's `PolyElement.cancel`). Unlike SymPy, negative powers are
  normalised too (issue #25: SymPy's `(-2*x - 2*y)**-1` and
  `1/(-2*x - 2*y)` compare unequal).
- `Domain.nullspace`: a basis of the null space from the reduced row
  echelon form, the free coordinate one (SymPy's `DomainMatrix.nullspace`
  over `QQ`; over fields of rational functions SymPy's vectors are
  fraction-free multiples).
- The core does not import SymPy (`import sympy_extras.core` leaves `sympy`
  out of `sys.modules`; tested). The bridge moved out of the core:
  `sympy_extras.interop` (was `sympy_extras.core.interop`) holds the
  conversions, the wrappers of SymPy objects (`SymPyFunction`,
  `AppliedSymPyFunction`, `SymPyNode`, `SymPyExprNode`, `apply_sympy`) and
  `SymPyBridge`, which registers itself with `sympy_extras.core.bridge` when
  imported: the operations not ported (integrate, simplify of foreign
  nodes, string parsing, ...) go through it; without it they raise
  `ForeignOperationUnavailable` and the queries answer unknown.
  `sympy_extras.core` no longer re-exports `to_sympy`/`from_sympy` and the
  wrappers; `sympy_extras.concrete` and `sympy_extras.simplify` re-export
  nothing (import from the modules: `sympy_extras.concrete.karr`, ...).
  `Expr.as_poly`, `AlgebraicNumber.as_poly`/`minpoly` return the typed
  `Poly`.
- `Expr.extract_branch_factor` is native (it was the most used operation
  of the bridge); the doctests of the core which need the bridge import
  `sympy_extras.interop` explicitly.
- The tractable/intractable rewrites and `is_meromorphic` ask the bridge
  only for the classes whose SymPy counterpart has a rule of its own
  (tables checked against SymPy by tests): 2026 of 2028 and 562 of 562
  calls found no rule and were round trips to SymPy for nothing.
- Powers of order terms are native (SymPy's `Order._eval_power`); the
  unused bridge methods (`power`, `extract_branch_factor`, `facts`,
  `leading_term`) are gone.
- SymPy's `is_constant` and `equals` run through the bridge with SymPy's
  random generators seeded and restored: they evaluate at random points,
  and some points made the suite hang (only under some hash seeds, in
  `test_known_methods_on_random_expressions`).
- The rewrites of single nodes ask the bridge only when SymPy has a rule
  for that class and target (`sympy_extras.core._rewrite_rules`, generated
  from SymPy and checked by a test): 340 of 419 calls had none.
- `degree` with coefficients the typed `Poly` does not hold (rational
  functions of other symbols, functions) is native, term by term (SymPy's
  degree over `EX`); the bridge no longer has `degree`.
- `expand_func` asks the bridge only for the classes with a SymPy rule
  (a table checked against SymPy); the recurrence of the Bessel functions
  (`BesselBase._eval_expand_func`) is native.
- `sympify` parses strings natively (`sympy_extras.core.parsing`: numbers,
  names, constants, SymPy's function names, undefined functions, the
  arithmetic operators; the same trees as SymPy's `parse_expr` on the
  tested strings). Other syntax goes to the bridge, `SympifyError` without
  it; `evaluate=False` prefers SymPy's parser when it is registered (its
  literal forms). `Bridge.available()` tells whether a system is
  registered.
- `expand_func` of `loggamma` at rationals (with a `Sum`), of the
  spherical Bessel functions (`spherical_bessel_fn`, ported from
  `sympy.polys.orthopolys`) and of `lerchphi` (all its cases) is native:
  no `expand_func` of the measured tests goes through SymPy now.
- Native rewrites: the inverse trigonometric and hyperbolic functions as
  `log`, `cot` as `tan`, and (`functions/special/_rewrites.py`) the Bessel
  functions between themselves and as `jn`, `erfc`/`erfi`/`erf2` as `erf`,
  `expint` and `Ei` into each other: no rewrite of the measured tests goes
  through SymPy now.
- `equals` proves "not equal" natively: the difference evaluated at up to
  three points respecting the assumptions of its symbols (integers for an
  integer symbol, ...), a value above the noise and the same at 60 and 200
  bits. 203 of 264 bridge calls of the measured tests are decided so; some
  answers SymPy leaves undecided (`sqrt(x**2) == x`) are `False` now.
- `ask` decides by cases (the disjunctive normal form of the assumptions,
  the same answer in every consistent case) and by refutation (every case
  of `assumptions & ~prop`, or of `assumptions & prop`, contradictory)
  before the bridge; random queries agree with SymPy's `ask` (a test).
  SymPy's `ask(Q.real(x*y), Q.imaginary(y) & Q.real(x))` is wrongly
  `False` (issue #25); here it is undecided.
- `is_constant` natively: `False` by two different stable values at
  admissible points, `True` by exactly zero derivatives; the bridge only
  otherwise (none of the measured tests needs it now). The random-methods
  test seeds SymPy's generator for its oracle (one run in a few hung on a
  point drawn from the system's entropy).
- `integrate` natively for sums of terms `c*x**n` (rational `n`, `log(x)`
  for `n = -1`; definite without the logarithm), the same results as
  SymPy's; `Tuple.simplify` simplifies the elements natively.
- The generic series of a function whose argument goes to an infinity
  uses the class's own `_eval_aseries` (SymPy's asymptotic branch);
  `LogGamma._eval_aseries` (Stirling's series) is native.
- `sympy_extras.polys.polytools.invert` (univariate over `QQ`, two
  integers), ported from SymPy; `Expr.invert` uses it without generators.
- `sympy_extras.simplify.ratsimp.ratsimp`, ported from SymPy (with the
  multivariate division of `reduced` over `QQ`); `Expr.ratsimp` uses it.
- `sympy_extras.simplify.gammasimp`: `gammasimp` (rising factorials,
  reflection, duplication, Gauss multiplication) and `combsimp`, ported
  from SymPy.
- `sympy_extras.simplify.trigsimp`: `trigsimp` (methods `matching` and
  `fu`), `futrig` and `exptrigsimp`, ported from SymPy.
- `sympy_extras.simplify.fu`: `fu`, `TR0` to `TR22`, `TRmorrie`,
  `TRpower`, `L`, `sincos_to_sum`, the `greedy` strategies, `trig_split`,
  `as_f_sign_1`, `process_common_addends`, `hyper_as_trig`, and
  `ExprFactors` (SymPy's `Factors` with any exponents), ported from SymPy.

### Changed

- The native `sympy_extras.assumptions.sat` (`satisfiable`, `tautology`,
  `find_instance`) no longer loads SymPy at run time (internal change,
  same API): the propositional abstraction is solved by a typed port of
  SymPy's DPLL2 (the new `sympy_extras.logic.dpll2`:
  `propositional_models`, `is_satisfiable`, `SATSolver`) on conjunctive
  normal forms of the core's Booleans (the new `sympy_extras.logic.cnf`:
  `to_clauses`, `conjunction`, `Clauses`, a port of the conversion of
  `sympy.assumptions.cnf`); the consistency of the predicates with their
  known facts is a port of SymPy's `satask` with the class facts of
  `sathandlers` and the known number facts (the new
  `sympy_extras.assumptions.satask.predicates_consistent`); the floor and
  the comparison of rational multiples, products and integer powers of
  real roots are native (`polys.cad.samplepoints.scaled_real`, new). The
  routes `sat_sympy.propositional_models_of_core`,
  `predicates_consistent_of_core`, `floor_of_core` and
  `compare_real_of_core` are gone. The models of `all_models=True` now
  come in an order which does not depend on the hash seed (SymPy's
  numbering of the atoms followed the iteration of a set). The native
  limits under assumptions (`assumptions.limits`) set native time limits
  around their native engines, `simplify` and `resolve`, so that a
  `limit` with a parameter without assumptions (`limit(sin(a*x)/x, x,
  0)`) loads no SymPy either.
- The definite integrals from an antiderivative are native
  (**breaking**): `sympy_extras.integrals.antiderivative`
  (`antiderivative`, `antiderivative_integral`,
  `principal_value_integral`, `finite_part_integral`, `discontinuities`,
  `one_sided_limit`, `select_branch`, `quick_shape`, `by_parts_shape`)
  takes and returns objects of the core and imports no SymPy: the
  singularities by a port of SymPy's `singularities` on the native solve
  under the assumptions on the range (the new `singularities`, a list of
  points), their placement by the native ask, the native limits under
  assumptions and series. The antiderivative is a typed step
  (`AntiderivativeStep`, the new `step` argument of the integrals): by
  default the native table of quadratic radicals, then SymPy's
  `integrate` and the verified methods of `indefinite` (not ported yet)
  through the face (`antiderivative_sympy.integrate_of_core`,
  `verified_antiderivative_of_core`), which loads SymPy on that path
  only; a native antiderivative (the native `ratint`) runs the whole
  method without SymPy. `_real_logarithms` and `_elementary_polynomial`
  are public as `real_logarithms` and `elementary_polynomial`. The old
  API on SymPy objects is in `sympy_extras.integrals.antiderivative_sympy`,
  used by `definite`, `series` and the tests.
- The integrals over regions are native (**breaking**):
  `sympy_extras.integrals.regions` (`IntegralByRanges`,
  `integrate_by_ranges`) takes and returns objects of the core and
  imports no SymPy: the native cylindrical decomposition, `ask`,
  `refine`, `solve`, `simplify`, the roots of `polys.polyroots` for the
  explicit branches of the bounds, the native simplex method for the
  boundedness of a polytope and the native `axisymmetric`, which now
  calls the native `integrate_by_ranges` directly
  (`axisymmetric_sympy.integrate_by_ranges_of_core` is gone). The
  `IntegralByRanges` of the core is an unevaluating node storing its
  arguments; a numeric bound is an `IndexedRoot` (still `CRootOf` on the
  face). The one-dimensional integrals are a typed `Integrator`
  (`integrator=` of `integrate_by_ranges` and `axisymmetric_integral`):
  by default `definite_integral`, not ported yet, through the face
  (`regions_sympy.definite_integral_of_core`), so that SymPy is loaded
  on that path only; an integrator on the core runs the decomposition
  and the iterated integrals without SymPy. The old API on SymPy objects
  is in `sympy_extras.integrals.regions_sympy`, used by the tests, the
  docs and the benchmarks.
- The rationalising substitutions for algebraic integrands and the
  canonical forms and substitutions before integration are native
  (**breaking**): `sympy_extras.integrals.algebraic`
  (`algebraic_integral`, `euler_substitutions`, `euler_substitution`,
  `moebius_root`, `binomial_differential`, `binomial_exponents`,
  `chebyshev_elementary`, `binomial_hypergeometric_antiderivative`,
  `rationalised_integral`, `Rationalisation`, `BinomialDifferential`) and
  `sympy_extras.integrals.rewriting` (`rewritten_forms`,
  `power_substitutions`, `substitute_back`, `implied_assumptions`,
  `Substitution`) take and return objects of the core and import no
  SymPy: the native ask, limits under assumptions, `cancel`, `factor`,
  `powsimp`, the roots of quadratics, the rewriting rules of the core's
  functions and `tidy`. The new variables are dummies with names of their
  own (`_t3` rather than `_t`). The rational integral of `algebraic` and
  its numerical check still come from `definite` (not ported yet)
  through the face (`algebraic_sympy.conditional_integral_of_core`,
  `verify_numerically_of_core`); SymPy is loaded on that path only. The
  old API on SymPy objects is in `sympy_extras.integrals.algebraic_sympy`
  and `sympy_extras.integrals.rewriting_sympy`, used by `definite`,
  `indefinite`, `trager` and the tests.
- The integrals by series expansion and the region integrals with a
  rotational symmetry are native (**breaking**):
  `sympy_extras.integrals.series` (`series_integral`, `fourier_integral`,
  `fourier_expansion`, `expansions`, `moments`, `sum_series`,
  `Expansion`) and `sympy_extras.integrals.axisymmetric`
  (`axisymmetric_integral`, `rotation_group`, `rotation_invariant`,
  `aligned_frame` and its result, now the public `Frame`) take and return
  objects of the core and import no SymPy: the native summation,
  `hypersimp`, `hyperexpand`, roots, convergence tests, Euler sums,
  Zeilberger's algorithm, `ask`, `refine`, `tidy`, `cancel` and numerical
  comparison. The full partial fractions of the rational series are
  written over the roots of the denominator (SymPy's `apart(full=True)`
  is not ported), and the frame of `aligned_frame` comes from a port of
  SymPy's Gram–Schmidt process. The moments, the mean values and the
  computed Fourier coefficients still come from `definite` (not ported
  yet) through the faces (`laplace_sympy.definite_integral_of_core`,
  `conditional_integral_of_core`; the coefficients first from the native
  `antiderivative_integral`), the numerical check of
  an unjustified interchange from `definite.verify_numerically` through
  `residues_sympy.verify_numerically_of_core`, and the profile integral
  from `regions` through the new
  `axisymmetric_sympy.integrate_by_ranges_of_core`; SymPy is loaded on
  those paths only. Every dummy has a name of its own (`_k3`, `_Y1_0`:
  the typed polynomials map their symbols by name). The old API on SymPy
  objects is in `series_sympy` (used by `definite`, with
  `_polygamma_series_sympy` for the fuzzer) and `axisymmetric_sympy`
  (used by `regions`).
- The Laplace rules, Frullani's and Glasser's theorems, the summability
  means and differentiation under the integral sign are native
  (**breaking**): `sympy_extras.integrals.laplace` (`laplace_rules`,
  `laplace_integral`, `laplace_of_convolution`, `transform`,
  `real_arguments`, and the new `has_integral`),
  `sympy_extras.integrals.transformations` (`frullani`, `glasser`,
  `transformation_integral`), `sympy_extras.integrals.summability`
  (`summable_integral`, `oscillatory_tail`, `METHODS`) and
  `sympy_extras.integrals.parametric` (`parametric_integral`,
  `candidate_parameters`) take and return objects of the core and
  import no SymPy: the native ask, limits under assumptions, partial
  fractions, `simplify`, `tidy`, `mellin` and `conditions`. The
  integrals of the pieces still come from `definite` (not ported yet)
  through the faces (`laplace_sympy.definite_integral_of_core` and
  `conditional_integral_of_core`), as do SymPy's `periodicity` and, for
  the antiderivative in the parameter, SymPy's `integrate`; SymPy is
  loaded on those paths only. `parametric_integral` takes an
  `integrator`, a `ParametricIntegrator` (the definite integrals and the
  numerical check), by default `parametric_sympy.DefiniteIntegrator`.
  Every dummy has a name of its own (the typed polynomials map their
  symbols by name). The old API on SymPy objects is in
  `laplace_sympy`, `transformations_sympy`, `summability_sympy` and
  `parametric_sympy` (used by `definite`).
- The residue theorem integrators are native (**breaking**):
  `sympy_extras.integrals.residues` (`residue_integral`,
  `real_line_integral`, `fourier_integral`, `keyhole_integral`,
  `period_integral`, `rational_function`, `poles`) and
  `sympy_extras.integrals.contours` (`contour_integral`,
  `rectangular_integral`, `sector_integral`, `indented_integral`) take and
  return objects of the core and import no SymPy. The rational functions
  hold dense coefficients (radicals, `I`, parameters; the typed `Poly`
  has no domain `EX`), the poles are the native `roots` with the
  irrational constants as placeholders, the denominators with `I` among
  their coefficients factored over `ZZ_I` first (the double poles of
  `1/(3 + sin(x))**2` in `exp(I*x)`); a multiple pole is divided out by
  synthetic division, the branch of the logarithm of the keyhole is a
  symbol with the derivation `d/dz + (1/z) d/dL` instead of an undefined
  function, the rectangle's triangular system is solved by elimination on
  core expressions and its multiple poles by the core's `residue`; a pole
  placed at a sample of the parameters is verified by the validated
  quadrature (or mpmath's rules of `recognize`) instead of `definite`'s
  `verify_numerically`, which is reached through the face only when they
  settle nothing. The `CRootOf` roots of a numerical denominator whose
  radicals are unwieldy still come from SymPy, through
  `residues_sympy.complex_roots_of_core`. The old API on SymPy objects is
  `sympy_extras.integrals.residues_sympy` and
  `sympy_extras.integrals.contours_sympy` (used by `definite`).
- Creative telescoping for integrals is native (**breaking**):
  `sympy_extras.integrals.telescoping` (`almkvist_zeilberger`,
  `holonomic_ode`, `holonomic_integral`, `DifferentialTelescoper`, ...),
  `sympy_extras.integrals.reduction` (`HermiteReducer`, `hermite_reduce`,
  `reduction_telescoper`, `reduction_ode`, `reduction_integral`) and
  `sympy_extras.integrals.dfinite` (`chyzak`, `closure`, `annihilators`,
  `normal_form`, `dfinite_ode`, `dfinite_integral`, ...) take and return
  objects of the core and import no SymPy. The polynomials in `x` have
  their coefficients in the field `QQ_I(parameters)` of the typed layer
  (factored over `QQ`, or over `QQ_I` when `I` occurs, as SymPy's
  `QQ(t)` and `QQ_I(t)` factor), the linear systems are solved by its
  null spaces, the boundary limits are the native ones, the equation of
  the integral is solved by the native linear ODE solvers with variation
  of parameters (SymPy's `dsolve` only when they leave it unsolved), and
  the equations of the factors `exp`, `sin`, `cos`, `sinh`, `cosh`,
  `besselj`, `erf`, `erfc`, `erfi` and `log` are written down. What stays
  on SymPy goes through the `*_of_core` functions of the faces: the
  initial values and the numerical check (`definite_integral`,
  `verify_numerically`; `definite` is not ported), `expr_to_holonomic`
  for `annihilators` and the other factors. `HermiteReducer.A` and `.B`
  are expressions on the core. The old API on SymPy objects is
  `sympy_extras.integrals.telescoping_sympy`, `reduction_sympy` and
  `dfinite_sympy` (used by `definite`).

- The mean value integrals of periodic integrands are native
  (**breaking**): `sympy_extras.integrals.periodic` (`mean_value_integral`,
  `laurent_coefficient`, `to_laurent`) takes and returns objects of the
  core and imports no SymPy. It runs on the native `brackets`, `mellin`
  and `conditions` instead of their faces, the native expansion,
  summation and `simplify`; the Taylor series of `exp`, `sin`, `cos`,
  `sinh` and `cosh` are written down (in the forms of the formal power
  series), so that only the other functions reach SymPy's `fps` through
  `brackets`, and the bounds of the sums of two series are computed with
  an exact rational slope instead of a float. The old API on SymPy
  objects is `sympy_extras.integrals.periodic_sympy` (used by `definite`).
- Trager's algorithm on a general curve is native (**breaking**):
  `sympy_extras.integrals.curves` (`curve_antiderivative`, `curve_reduce`,
  `is_nonelementary_on_curve`, `integrate_on_curve`, `AlgebraicIntegrand`,
  `CurveIntegral`) takes and returns objects of the core and imports no
  SymPy. It runs on the native function fields, Puiseux expansions and
  number fields without their faces (the rationals as the number field
  `RATIONALS` of degree one, the integral basis normal at infinity
  computed over the plain rationals), the minimal polynomial of the
  primitive element is read off its powers in the monomial basis of the
  radicals (the radicals independent when it is irreducible over
  `Q(x)`: a specialisation of `x`, else the bivariate factorisation)
  instead of SymPy's `minimal_polynomial` with symbols, and the
  expressions are cancelled by the native `cancel` after the coefficients
  of the monomials of the radicals are reduced over `Q(x)` (the
  logarithms come out shorter than before, SymPy's `cancel` keeping the
  common factors of `x` and `sqrt(x)`). The field of the residues is
  named by its largest real root (an `IndexedRoot`) or in radicals;
  `CurveIntegral.elementary_part(theta)` and `remainder_expr(theta)` take
  the expression of a generator which has neither (the native
  `curve_antiderivative` returns `None` then). The old API on SymPy
  objects is `sympy_extras.integrals.curves_sympy` (used by `trager`),
  which writes such a generator as SymPy's `CRootOf`; its `CurveIntegral`
  wraps the native one (`remainder` is the native element), and
  `AlgebraicIntegrand.algebra` is the native algebra.
- The validated numerical integration is native (**breaking**):
  `sympy_extras.integrals.validated` (`validated_integral`, `enclosure`)
  takes and returns objects of the core and imports no SymPy (the value
  and the error are floats of the core). The enclosures come from the
  native interval evaluator of `assumptions.intervals` instead of its
  SymPy face (no conversion per panel and node), the derivatives,
  `together`, `cancel`, `simplify`, `powdenest` and the limits at
  infinity are the core's, the polynomials in the variable are lists of
  any coefficients (the typed `Poly` has no `EX` for `pi` or `sqrt(2)`),
  Sturm's count on a ray is `polys.rootisolation`'s, and the
  antiderivatives of the trigonometric factor of an oscillatory tail are
  written down instead of asking `integrate`. The old API on SymPy objects
  is `sympy_extras.integrals.validated_sympy` (used by `definite`); it
  returns SymPy floats as before (a zero stays `Float(0, digits)`).
- The reduction to Legendre's elliptic integrals is native
  (**breaking**): `sympy_extras.integrals.elliptic` (`elliptic_integral`,
  `radicand`, `real_roots`, `legendre_reduction`, `Reduction`) takes and
  returns objects of the core (the core's `elliptic_k`, `elliptic_e`,
  `elliptic_f`, `elliptic_pi`) and imports no SymPy: the native
  factorisations with placeholders for the irrational constants (the
  typed `Poly` has no `EX`), the partial fractions from the Taylor
  coefficients at the linear factors of the denominator, factored over
  the number field of its radicals as SymPy's `apart` does with
  `extension=True` (Trager's `dup_ext_factor`), the roots of the linear
  and quadratic factors by the native `roots` instead of `solve`, the
  real roots of numeric radicands as indexed roots (`CRootOf` through
  interop) and their complex pairs from mpmath roots and the native
  resultants, the native `ask` and reliable values, and the native
  `ratint` for the rational antiderivative of the elementary odd part of
  the two-pairs case. The old API on SymPy objects moves
  to the new face `sympy_extras.integrals.elliptic_sympy`, which
  `integrals.definite` and the tests use. The unused helper
  `_square_root` is gone.

- The rational integrator is native (**breaking**):
  `sympy_extras.integrals.risch.rationaltools` (`ratint`,
  `ratint_ratpart`, `ratint_logpart`, `log_to_real`, `log_to_atan`, and
  the new `polynomial_part`) takes and returns objects of the core and
  imports no SymPy: Hermite's reduction (the linear system of
  Horowitz–Ostrogradsky solved by elimination over the domain) and
  Lazard–Rioboo–Trager by the native subresultant PRS, square-free
  decompositions, gcds and `roots`, on typed dense polynomials over `QQ`
  or over the new domain `EX` of core expressions
  (`sympy_extras.polys.domains.expressiondomain`, ported from SymPy: the
  normal form `cancel` expanded over the products, as SymPy's `EX`);
  SymPy's domain normalisations (the `cancel` over `ZZ`, the primitive
  parts, the fraction-field forms of `Poly.as_expr`) are reproduced. The
  sums over the roots are the new native `RootSum` node of
  `sympy_extras.polys.rootoftools` (ported from SymPy: `RootSum` never
  evaluates, `root_sum` evaluates as SymPy's constructor, with
  `preprocess_roots`, the factors, the roots of linear and quadratic
  factors and the rational case by the trace in `K[t]/(q)`; `doit` by
  `roots`, `diff`, numerical values; `interop` converts both ways). The old
  API on SymPy objects (`Poly` arguments, SymPy's `RootSum`) moves to the
  new face `sympy_extras.integrals.risch.rationaltools_sympy`, which
  `risch.risch`, `trager`, `indefinite`, `trigonometric`, `radicals` and the
  tests use. `exponential.exponential_rational` calls the native `ratint`
  (`exponential_sympy.ratint_of_core` is removed), and
  `solvers.linear_systems` the new native `rational_integral`
  (`linear_systems_sympy.rational_integral_of_core` is removed). The
  package `sympy_extras.integrals.risch` exports nothing, so that the
  native module imports without SymPy: `risch_antiderivative` and
  `is_nonelementary` move to `sympy_extras.integrals.risch.risch_sympy`.
  Fixed on the way: the derivative of a `RootSum` whose function is
  constant in the variable is the degree times it (SymPy 1.14 returns the
  function itself: `RootSum(t**3 - t - 1, Lambda(t, log(t*x))).diff(x)` is
  `1/x`). Over coefficients with
  square roots of numbers the arithmetic rationalizes the denominators
  (`EXR`, which keeps the gcds from growing); the transcendental constants
  (`pi`, `tan(1)`) are parameters of the root formulas, as in SymPy's
  `ZZ(tan(1))`. Compared with the old code on the 750 distinct calls of
  the integrals and solvers suites and 78 more integrands: the same
  results but for 33 calls and 6 integrands (radical coefficients, rationalized and often
  shorter; the forms of the roots of cubics in radicals; parameters in
  the arguments of the logarithms; the sign of the arctangent pairs), every
  antiderivative checked by differentiation; two integrands which the old
  code did not finish in a minute take a second.
- The antiderivatives of exponentials and the recognition of constants are
  native (**breaking**): `sympy_extras.integrals.exponential`
  (`exponential_antiderivative`, `power_exponential`,
  `exponential_rational`, `nested_power_exponential`,
  `rational_exponential`, `composite_gaussian`,
  `reciprocal_square_exponential`) and `sympy_extras.integrals.recognize`
  (`recognize_constant`, `recognize_integral`, `quadrature`,
  `DEFAULT_CONSTANTS`) take and return objects of the core and import no
  SymPy: the native `powsimp`, `expand`, `factor`, `cancel`, `simplify`,
  `roots` (with a dummy for every coefficient outside the polynomial
  domains, `pi` or `exp(c)`, the typed `Poly` having no `EX`), the native
  `ask` and the numerical checks of `integrals.conditions`; the
  quadrature evaluates the integrand by the numerical rules of the core's
  nodes instead of `lambdify`; the two unknowns of the composite Gaussian
  are solved by a 2x2 elimination instead of SymPy's `solve` (an
  underdetermined system is refused). One step stays on SymPy, reached
  only by the substitution `u = exp(c*x)`: its `ratint`, through
  `exponential_sympy.ratint_of_core`. The old API on SymPy objects moves
  to the new faces `sympy_extras.integrals.exponential_sympy` and
  `sympy_extras.integrals.recognize_sympy`, which `integrals.indefinite`,
  `integrals.definite`, the tests and the docs use (the results are the
  same, `srepr` for `srepr`, on the calls of the tests and 70 more).
  Fixed on the way: `Integer(n).as_content_primitive()` of the core
  returned the content as a `Rational` which is not an `Integer`
  (`factor_terms(a*x**2)` had the exponent `Rational(2, 1)`), and the
  native `hyperexpand` looped forever on `1F1(a; a + 1; z)` with a
  rational `a` (the parameters of an instantiated formula were not
  evaluated, `1/3 + 1` was shifted towards `4/3`).

- The heuristic Risch integrator and the table of definite integrals are
  native (**breaking**): `sympy_extras.integrals.heurisch`
  (`components`, `heurisch_antiderivative`, `heurisch_cases`) and
  `sympy_extras.integrals.tables` (`TABLE`, `TableEntry`, `lookup`,
  `table_integral`, the patterns' `X`, `A`, `B`, `N`, `M`, `C`, `K`)
  take and return objects of the core (the native `ConditionalValue`)
  and import no SymPy. The polynomial steps of heurisch run on the typed
  `Poly`, `gcd`, `lcm`, `factor_list`, `cancel` and `roots`, the
  constants the typed `Poly` does not take as coefficients (`pi`,
  `exp(y)`, `sqrt(2)`, `I`) replaced by placeholder symbols, as SymPy's
  composite domains make them generators; the linear system of the
  candidate is solved by its reduced row echelon form (unique for the
  order of the unknowns, so the solution of SymPy's `solve_lin_sys`)
  over `QQ`, over the number field of the algebraic constants (`I` and
  radicals of rationals through a primitive element: the arctangents of
  `1/(x**4 + 1)` need `sqrt(2)**2 = 2`, which SymPy's `EX` decided), or
  over the fractions of the parameters; the tangents of the half angle,
  SymPy's `root_factors`, `denoms` and the single-equation `solve` of
  `heurisch_cases` are ported in the module; the verification by the
  native `cancel`, `simplify` and `numerically_equal`. The old API on
  SymPy objects is in the faces `sympy_extras.integrals.heurisch_sympy`
  and `sympy_extras.integrals.tables_sympy` (`TableEntry.from_core`,
  `to_core`, the table converted), which `indefinite`, `definite` and
  the tests use.
- The asymptotic expansions of integrals are native (**breaking**):
  `sympy_extras.integrals.asymptotic` (`asymptotic_integral`,
  `watson_lemma`, `laplace_method`, `stationary_phase`,
  `steepest_descent`, `uniform_expansion`, `exponential_part`,
  `oscillatory_part`) takes and returns objects of the core and imports
  no SymPy: the expansions are the core's series and `Order`, the
  stationary and saddle points the native `solve`, the signs the native
  `ask`, the limits at infinite ends the native `limit`, and the
  quadratures which confirm the complex saddle points evaluate the
  integrand with mpmath through the core (no `lambdify`); the polynomial
  coefficients in the parameter are read from the expansion (the typed
  `Poly` has no domain `EX`). The SymPy API is the new face
  `sympy_extras.integrals.asymptotic_sympy`, with the same results; it
  leaves a sum unevaluated when SymPy's `Order` cannot compare its terms
  with the order term, as before. The stationary points of
  transcendental phases and the complex saddle points still reach
  SymPy's `solveset` at run time, through the native `solve`. Import the
  face to keep the old API.
- A sum of imaginary terms is imaginary (and so nonzero) in the core only
  when the sum of the terms times `I` is known not to vanish, as SymPy's
  `Add._eval_is_imaginary` says: `I - 4*w**3` with `w = 2**(1/3)*(I -
  sqrt(3))/4`, which is zero, was imaginary, and the native `solve`
  dropped two of the three roots of `4*x**3 - I`.
- The convergence tests of series and products are native
  (**breaking**): `sympy_extras.concrete.convergence` (`sum_convergence`,
  `product_convergence`, `is_convergent`) takes and returns objects of
  the core and imports no SymPy: the limits, the decisions under
  assumptions, the boundary points, the signs, the quantifier-free
  simplification of the conditions and the simplifications of the ratios
  are the native ones of `sympy_extras.assumptions` and
  `sympy_extras.simplify`, and the polynomial coefficients of the
  majorants are read from the expansion (the typed `Poly` has no domain
  for `pi*n + 1`). Two steps of the last resort for terms without
  parameters go through the new face at run time:
  `convergence_sympy.improper_integral_of_core` (SymPy's `integrate` of
  the integral test) and `convergence_sympy.is_convergent_of_core`
  (SymPy's `Sum.is_convergent`). The old API on SymPy objects moves to
  the new face `sympy_extras.concrete.convergence_sympy`, which
  `integrals.series`, the tests, the benchmarks, the README and the docs
  use. The core's `rewrite(Factorial)` gains SymPy's rules for `gamma`,
  `binomial`, `rf` and `ff` (PORTED FROM SYMPY). In `concrete.delta` and
  `concrete.summations` the equations which `linsolve` does not solve go
  to the native `assumptions.solve` (a finite set of solutions over the
  complex numbers, those contradicting the assumptions of the unknown
  dropped as SymPy's `solve` does) before SymPy's `solve`, and SymPy's
  `as_set() == EmptySet` of a condition in one symbol is the native
  `solve` over the reals; SymPy's `solve(..., dict=True)` of the systems
  of several deltas, SymPy's `Integral` of
  `eulersums.polygamma_integral_representation`, the residue method with
  `CRootOf` poles and the zeta function match of `Sum.doit` stay on
  SymPy.
- The top-level helpers import no SymPy (**breaking**):
  `sympy_extras._special_values` (`with_special_values`,
  `special_points`, `point_condition`, `condition_point`,
  `point_assumptions`, `undefined_at`, `equality_points`,
  `defined_problem`) takes and returns objects of the core, with the
  native factorisation, roots, transcendental solver and `satisfiable`
  (SymPy's `roots` of a polynomial with irrational coefficients and its
  `solveset` of a transcendental factor the native solver does not
  reduce go through `roots_sympy`/`transcendental_sympy` at run time);
  the old API on SymPy objects is the new face
  `sympy_extras._special_values_sympy` (used by `integrals.definite`,
  `integrals.indefinite` and `concrete.karr_sympy`; not to be confused
  with `solvers._special_values_sympy`). The native
  `solvers._special_values` calls it directly, and
  `solvers._special_values_sympy.special_values_of_core` is removed.
  The SymPy aliases and narrowing helpers of `sympy_extras._typing`
  (`ExprLike`, `OrderSpec`, `QuantifierPrefix`, `QuantifierSpec`,
  `as_expr`, `as_boolean`, `as_symbol`, `as_set`, `free_symbols`,
  `sorted_symbols`) moved to the new `sympy_extras._typing_sympy`, which
  re-exports the SymPy-free aliases (`Dup`, `Dmp`, `Monomial`, `Sign`,
  `Truth`, `DomainElement`, `OrderKey`, `Weights`, `QuantifierKind`)
  that stay in `_typing`; every SymPy-side importer switched.
  `sympy_extras._timeout` already imported no SymPy at module level; its
  preparation of SymPy's integration tables is now the public
  `prepare_sympy()`, called by the limits without `native=True` and by
  the callbacks of the `_special_values_sympy` face (which run under the
  native limits of the native module). The native cases of the ODE
  solutions no longer load SymPy (the symbolic check of a solution at a
  point runs under a native limit).
- The symmetry method for ODEs and the second order solvers are native
  (**breaking**): `sympy_extras.solvers.ode` (`ode_symmetries`,
  `canonical_coordinates`, `reduce_order`, `dsolve_lie`, `solve_ode`,
  `ReducedODE`) and `sympy_extras.solvers.second_order`
  (`second_order_rhs`, `integrating_factor_xy`, `integrating_factor_p`,
  `is_linearizable`, `linearize`, `rectify_symmetries`,
  `commuting_pair`, `transformed_rhs`, `dsolve_second_order`,
  `FirstIntegral`, `Linearization`) take and return objects of the core
  and the native `lie.Symmetry`, and import no SymPy: the symmetries,
  the jets, the coefficients, the factorisations, the cancellations and
  the simplifications are the native ones, a first order equation goes
  to the native `first_order.dsolve_first_order`, the structure
  constants of the symmetries are solved by the native
  `linsolve.solve_linear_unique` (SymPy's `solve` only for a system
  without a unique solution) and their centralizers by the native null
  spaces, and the solutions are checked by the new
  `ode.check_ode_solution`, a port of SymPy's `checkodesol`
  (PORTED FROM SYMPY, `sympy/solvers/ode/subscheck.py`). SymPy's
  `dsolve` and `solve` (`pde_sympy.dsolve_of_core`,
  `pde_sympy.solve_of_core`, `first_order_sympy.dsolve_of_core`,
  `charpit_sympy.solve_of_core`, `factorization_sympy.solve_of_core`) and
  `integrate` (`kovacic_sympy.integrate_of_core`, `integral_of_core`) are
  called through the faces; no new `*_of_core` function. The old API on
  SymPy objects moves to the new faces `sympy_extras.solvers.ode_sympy`
  (also `check_ode_solution`) and `sympy_extras.solvers.second_order_sympy`,
  which the tests, the benchmarks, the README and the docs use. Two
  differences with the old modules: a quadrature which SymPy evaluates
  only in part is kept as the whole unevaluated `Integral(V, r)` (the
  same function, written `Integral(c*V, r)` rather than `c*Integral(V,
  r)`), and a relation which SymPy's `solve` splits into several
  branches is a solution only when every branch is (the old check read
  the first `(verdict, residual)` pair of `checkodesol`'s list, a
  nonempty tuple and always true). A wrapped SymPy `Integral`, `Sum` or
  `Product` substitutes its bound variable as SymPy does
  (`Integral(f(r), r).subs(r, g)` is `Integral(f(r), (r, g))`; the
  interop rebuilt invalid limits and raised `ValueError`).
- The method of brackets and the Dirichlet integrals are native
  (**breaking**): `sympy_extras.integrals.brackets` (`SeriesCoefficient`,
  `taylor_coefficient`, `ramanujan_master_theorem`, `method_of_brackets`,
  `mellin_transform_series`, the new `FormalSeriesTerms`) and
  `sympy_extras.integrals.dirichlet` (`dirichlet_integral`,
  `trigonometric_sum`, whose kinds are now the classes `Sin` and `Cos` of
  the core) take and return core expressions, the native
  `MellinTransform` and `ConditionalValue`, and import no SymPy: the
  coefficient functions, the checks of the formal series against the
  native Taylor series, the decay at infinity (native series and
  limits), the bracket sums (the native `summation`), `gammasimp`,
  `hyperexpand`, `powsimp`, `simplify`, `logcombine`, `TR8` and `ask`
  are the native ones. SymPy's formal power series (`fps`, not ported)
  and, when the decay is not found, the strip of SymPy's
  `mellin_transform` are called through the face
  (`brackets_sympy.fps_of_core`, `brackets_sympy.mellin_strip_of_core`).
  The old API on SymPy objects moves to the new faces
  `sympy_extras.integrals.brackets_sympy` and
  `sympy_extras.integrals.dirichlet_sympy` (same results), which
  `definite`, `series` and `periodic` now import. One result changes: the
  method of brackets no longer returns `E*pi/sin(pi*s)` for
  `exp(-x)/(1 + x)` (a wrong transform, from SymPy's `Sum.doit` of the
  series left by the bracket; the native summation leaves it unevaluated
  and the method gives up).
- Slater's theorem and the Marichev–Adamchik method are native
  (**breaking**): `sympy_extras.integrals.slater` (`MeijerG`,
  `mellin_barnes`, `slater_expansion`, `expand_meijerg`,
  `line_conditions`) and `sympy_extras.integrals.marichev`
  (`mellin_integrate`, `integrate_product`, `evaluate_quotient`, `tidy`,
  `checked_tidy`, `real_logarithms`, `polar_elliptic`, `polar_lowergamma`,
  `right_half_plane_powers`, `bessel_k_forms`, `exponential_form`,
  `reduce_positive_powers`) take and return core expressions, the native
  `GammaQuotient`, `Product` and `ConditionalValue`, and import no SymPy:
  the expansions by the native `hyperexpand`, the logarithmic case by the
  native limits, the simplifications by the native `simplify`,
  `gammasimp`, `factor`, `unpolarify`, `expand_func` and `refine`, the
  decisions by the native `ask`. `MeijerG.as_sympy` is `MeijerG.as_expr`
  on the core. The old API on SymPy objects is in the faces
  `sympy_extras.integrals.slater_sympy` and
  `sympy_extras.integrals.marichev_sympy` (`MeijerG` with
  `to_core`/`from_core`), which `definite`, `laplace`, `algebraic`,
  `series`, `reduction`, `telescoping`, `dfinite` and the tests use. New:
  `marichev.projected_sums`, the polar numbers in the terms of sums
  replaced by their projections, which `tidy` applies before
  `unpolarify` (the value of `exp(-x)*cos(2*x)*besselj(0, x)` over
  `(0, oo)`, where the native `hyperexpand` writes
  `1/sqrt(1 - polar_lift(w))`, was refused), and `marichev.real_constant`,
  a constant written with `I` or roots of `-1` in real terms when that
  removes them (`3 - log(2) - sqrt(3)*pi/3` for `cbrt(x)/(1 + x)` over
  `(0, 1)`); the value of `exp(-2*x)*cos(x)*besselj(0, x)` over
  `(0, oo)` in README and `docs/integrals.md` now comes from the
  Marichev–Adamchik method in another real form (SymPy's `hyperexpand` of
  its `1F0` used to time out). Fixed in
  `sympy_extras.functions.special.hyper.hyper`: the argument is
  unpolarified where the series is single valued, as SymPy's
  `hyper.eval` does. Fixed in
  `sympy_extras.simplify.hyperexpand`: the symbolic formulas were
  instantiated without evaluation (the core's `xreplace`), so that no
  origin was found for `2F1(nu/2 + 1/2, nu/2 + 1; nu + 1; z)` and the
  like, which SymPy expands.
- The linear systems `Y' = A Y + b` are native (**breaking**):
  `sympy_extras.solvers.linear_systems` (`CyclicReduction`,
  `cyclic_reduction`, `cyclic_vector`, `system_to_scalar`,
  `rational_system_solutions`, `hyperexponential_system_solutions`,
  `dsolve_system`, `LinearSystemSolution`, `dsolve_linear_system`) takes
  and returns the `ExprMatrix` of core expressions and the native
  `LinearOperator`, and imports no SymPy: the scalar equations go to the
  native `linear_ode` and `factorization` solvers, the Krylov rows of the
  cyclic vectors, their inverse and the scalar equation are computed in
  the native field of rational functions (`SparseFractionField`; the
  normal forms by the native `cancel` otherwise), Barkatou's bounds use
  the native `factor_list`, `resultant`, `roots` and null spaces (the
  limits at infinity of rational functions read from the leading
  coefficients instead of SymPy's `limit`), the determinants are the
  native Berkowitz ones, the checks and the final forms the native
  `simplify`, and a constant system whose scalar equation is not solved
  gets the columns of the native matrix exponential (Putzer,
  `dae.matrix_exponential`) before SymPy's `linodesolve`. SymPy's
  `integrate` (Liouville's formula and the quadratures of variation of
  constants, `kovacic_sympy.integrate_of_core`/`integral_of_core`), its
  rational integrator with `RootSum` (the new
  `linear_systems_sympy.rational_integral_of_core`) and `linodesolve`
  (the new `linear_systems_sympy.linodesolve_of_core`) are called through
  the face. The old API on SymPy matrices moves to the new face
  `sympy_extras.solvers.linear_systems_sympy` (`CyclicReduction` and
  `LinearSystemSolution` with SymPy matrices), which the tests, the
  benchmarks and the docs use. `linear_ode._solve_operator` imports
  `kovacic_sympy` only for first order equations, so that the scalar
  equations of higher order solved without quadratures load no SymPy.
  Fixed on the way: at a pole of order two or more whose `det N` is
  identically zero (a singular `A`, and every augmented system of an
  inhomogeneous one) the multiplicity of the zero polynomial came out 0
  and was taken for an invertible leading matrix, so that no rational
  solution was returned (`Y' = [[1, 1], [1, 1]] Y/x**2` lost `(1, -1)`)
  and the rational particular solutions at such poles were never found
  (variation of constants gave another one, with its integrals). Every
  other recorded call gives the same result as before (see the commit
  message for the comparison).
- The Darboux polynomials and the Prelle–Singer procedure are native
  (**breaking**): `sympy_extras.solvers.darboux` (`darboux_polynomials`,
  `exponential_factors`, `darboux_integrating_factor`,
  `darboux_first_integral`, `prelle_singer`, `vector_field`,
  `DarbouxPolynomial`, `ExponentialFactor`, `DarbouxIntegral`) takes and
  returns the expressions, functions and symbols of the core and imports
  no SymPy: the field is a typed sparse polynomial over `QQ`, a number
  field `QQ(alpha)` (`AlgebraicField`, of a primitive element of the
  radicals, `I` and the `extension` among the coefficients) or `QQ(a, b,
  ...)` (`SparseFractionField`, with the transcendental constants such as
  `pi` as parameters, as SymPy's `Poly` takes them); the factorisations are Wang's (the parameters
  as further variables) and Trager's, the leaf systems go to the native
  Gröbner bases and FGLM, the linear algebra to `sparse_echelon`, and the
  explicit solution to the native `solveset_rational` and `simplify`.
  The quadratures of an integrating factor still go through SymPy's
  `integrate` (`kovacic_sympy.integrate_of_core`). The old API on SymPy
  objects is in the new face `sympy_extras.solvers.darboux_sympy` (the
  same results on the tests, the doctests and 275 calls on 63 fields,
  three to four times faster); `solvers.first_order`, the fuzz driver and the
  documentation use it. Fixed on the way: Kamke 1.23 `y' = b - a y**2`
  made `darboux_polynomials` and `darboux_integrating_factor` raise
  `ValueError("0**0")` from SymPy's `PolyElement.evaluate` over `ZZ(a,
  b)` (sympy-extras#25); they give `y**2 - b/a` and `1/(y**2 - b/a)`.
  Coefficients with both parameters and algebraic numbers are refused
  with a `ValueError`, as before.
- The first order solvers and the Abel equations are native
  (**breaking**): `sympy_extras.solvers.first_order` (`riccati_ode`,
  `chini_ode`, `abel_ode`, `lagrange_ode`, `dsolve_first_order`) and
  `sympy_extras.solvers.abel` (`abel_coefficients`, `abel_invariants`,
  `constant_invariant`, `abel_equivalence`, `particular_solution`,
  `air_solution`, `abel_by_invariants`, `AbelInvariants`,
  `AbelTransformation`, `REPRESENTATIVES`) take and return the
  expressions, functions and symbols of the core and import no SymPy: the
  coefficients (read from the expanded terms, SymPy's domain `EX` not
  being ported), the cancellations, `factor`, `together`, `powdenest`,
  `real_root`, `simplify`, the gcd of the equivalence (`UPoly`), the
  linearised Riccati equations (the native `special_solutions` and
  `dsolve_linear`) and the cases of the isolated parameter values are
  native, and so is the numerical check of an implicit relation (a
  secant method in mpmath at 30 digits in place of SymPy's `nsolve`).
  SymPy's `integrate` (`first_order_sympy.integrate_of_core`,
  `kovacic_sympy.integrate_of_core`), its `dsolve` of the linear
  equation of a Lagrange equation and of an inverse Riccati equation
  (`first_order_sympy.dsolve_of_core`), its `solve` of the undetermined
  coefficients of a particular solution and of a common factor of the
  equivalence (`factorization_sympy.solve_of_core`,
  `charpit_sympy.solve_of_core`) and the Prelle–Singer procedure of
  `solvers.darboux` (`prelle_singer_of_core`) are called through the
  faces at run time. The old API on SymPy objects is in the new faces
  `sympy_extras.solvers.first_order_sympy` and
  `sympy_extras.solvers.abel_sympy` (results compared on the tests, the
  doctests and 40 further first order equations; see the commit
  message); `solvers.ode`, `solvers.darboux` (its private helpers
  `_derivative_polynomial`, `_explicit_rhs` stay on SymPy in
  `first_order_sympy`), the tests, the benchmarks and the documentation
  use the faces.

- The Lie point symmetries and the similarity reductions are native
  (**breaking**): `sympy_extras.solvers.lie` (`JetSpace`, `Symmetry`,
  `symmetries`, `check_symmetry`, `infinitesimal_condition`,
  `determining_equations`, `determining_system`, `symmetry_janet_basis`,
  and the new `leading_derivatives`) and `sympy_extras.solvers.pde`
  (`pde_symmetries`, `similarity_reduction`, `pdsolve_lie`, `Reduction`,
  and the new `check_pde_solution`) take and return the expressions,
  functions and symbols of the core and import no SymPy: the linear
  system of the ansatz is solved over `QQ` or the field of rational
  functions of the parameters (`SparseFractionField`, the constants
  outside it such as `sqrt(2)` taken as generic symbols), the
  determining system goes to the native Janet basis of
  `sympy_extras.polys.differential` (no longer through
  `polys.differential_sympy`), a quadratic leading derivative is solved
  by the native `roots_quadratic`, the reduced equations are normalised
  by the native `factor_list`, and the solutions of `pdsolve_lie` are
  checked by substitution with the native `simplify` (no longer SymPy's
  `checkpdesol`). The old API on SymPy objects is in the new faces
  `sympy_extras.solvers.lie_sympy` and `sympy_extras.solvers.pde_sympy`
  (same results on the tests, the doctests and 34 further equations);
  SymPy's `dsolve` (`dsolve_of_core`) and `solve` for the equations not
  linear in the unknown (`pde_sympy.solve_of_core`; `lie` uses
  `charpit_sympy.solve_of_core` for a leading derivative which is
  neither linear nor quadratic) are called through them. `solvers.ode`,
  `solvers.second_order`, the benchmarks and the documentation use the
  faces.

- The transcendental equations and the inequalities in one variable are
  native (**breaking**): `sympy_extras.solvers.transcendental`
  (`solve_transcendental`, `polynomialize`, `Reduction`) takes and
  returns the formulas, symbols and sets of the core and imports no
  SymPy: the kernels, the functional relations, the clearing of the
  denominators (the coefficients read from the expanded terms, SymPy's
  domain `EX` not being ported), the inverse images and the polynomial
  formulas with rational coefficients (the native CAD and radical forms)
  are native, and so are the relations linear in the unknown with real
  coefficients which SymPy's `solveset` solved when no solver is given.
  SymPy's `solveset` of the other ones (`solveset_of_core`), the Lambert
  W candidates of SymPy's `solve` (`solve_of_core`) and the reliable
  numerical values of `sympy_extras._numeric` (`reliable_form_of_core`,
  `reliable_value_of_core`) are called through the new face
  `sympy_extras.solvers.transcendental_sympy`, which keeps the old API
  on SymPy objects (`solve_transcendental`, `polynomialize`, `Reduction`
  with SymPy attributes); the tests use it. The native
  `sympy_extras.assumptions.solve` calls `solve_transcendental`
  directly, and `solve_sympy.solve_transcendental_of_core` is removed.
  `sympy_extras.solvers.inequalities.solve_inequality` imports no SymPy
  either: a relational rational in the variable with rational
  coefficients is solved by the CAD (SymPy's
  `reduce_rational_inequalities`, the same relational), the others go
  through `sympy_extras.solvers.inequalities_sympy.solve_inequality_of_core`
  at run time. Fixed on the way: a linear relational whose denominators
  vanish somewhere (`x < x*(2/x - 1)`) came back without SymPy's
  condition on them (`x < 1` for `(x < 1) & Ne(x, 0)`). Every recorded
  call gives the same result as before (see the commit message for the
  comparison).
- The linear ODE solvers are native (**breaking**):
  `sympy_extras.solvers.linear_ode` (`LinearOperator`,
  `polynomial_solutions`, `rational_solutions`,
  `hyperexponential_solutions`, `hyperexponential_search`,
  `reduce_order_linear`, `dsolve_linear`, `gcrd`, `lclm`,
  `log_derivative`), `sympy_extras.solvers.factorization`
  (`factor_operator`, `right_factor`, `left_factor`,
  `OperatorFactorization`, `variation_of_parameters`, `solve_factored`),
  `sympy_extras.solvers.irregular` (`Place`, `irregular_places`,
  `local_exponential_parts`, `finite_exponential_parts`,
  `conjugate_sum`) and `sympy_extras.solvers.special`
  (`special_solutions`, `bessel_solutions`, `whittaker_solutions`,
  `hypergeometric_solutions`, `whittaker_m`) take and return the
  expressions, functions and symbols of the core and import no SymPy.
  The polynomials in one variable are the new `linear_ode.UPoly`
  (coefficients any constant expressions, normalised by the native
  `cancel`; the native `factor_list`, `gcd`, `resultant` and `roots`
  with the coefficients the typed `Poly` refuses taken as generators),
  the linear algebra the native field of rational functions
  (`SparseFractionField`: null spaces, the inverse of Beke's cyclic
  matrix) with Gaussian elimination on expressions as the fallback, the
  determinants those of `_special_values`, the number fields of the
  irregular points `polys.domains.AlgebraicField` with Trager's
  factorisation (the root `alpha` of a place of degree more than one is
  a symbol, the generator of the field; `irregular_places` returns
  `UPoly`), the integral of an exponential part the conjugate sum of
  `-r/((m - 1) (x - alpha)**(m - 1))` instead of SymPy's
  `ratint_ratpart`, the integrals of the polynomial exponential parts
  term by term, the special functions those of
  `sympy_extras.functions`. `LinearOperator.coefficients` and
  `OperatorFactorization.factors`/`irreducible` are tuples on the core.
  SymPy's `integrate` (`kovacic_sympy.integrate_of_core`,
  `integral_of_core`), `dsolve` (the new
  `linear_ode_sympy.dsolve_of_core`) and `solve` on the
  Grassmann–Plücker relations of Beke's method (the new
  `factorization_sympy.solve_of_core`) are called through the faces. The
  old API on SymPy objects moves to the new faces
  `sympy_extras.solvers.linear_ode_sympy`, `factorization_sympy`,
  `irregular_sympy` and `special_sympy` (with the private helpers that
  `linear_systems` and the tests use: `_solve_operator`, `_nullspace`,
  `_indicial_at`, `_indicial_at_infinity`, `_quadrature`, ...), which
  `linear_systems`, `first_order`, the tests, the benchmarks, the README
  and the docs use. The wrapped SymPy expressions of the interop
  (`SymPyExprNode`) now differentiate through SymPy: the derivative of an
  unevaluated `Integral(f, x)` by `x` is `f` (it stayed
  `Derivative(Integral(f, x), x)`). Every recorded call of the tests and
  the doctests gives the same result on the faces except the form of
  some expanded `L(y)` and the order of two hyperexponential solutions,
  and so do 40 further equations (see the commit message).
- The Mellin transforms are native (**breaking**):
  `sympy_extras.integrals.mellin` (`GammaQuotient`, `MellinTransform`,
  `Kernel`, `KERNELS`, `Match`, `Product`, `mellin_kernel`,
  `mellin_transform`, `monomial`, `decompose_integrand`,
  `poles_separated`) takes and returns the expressions, functions and
  symbols of the core and imports no SymPy: the table on the core's
  `gamma`, `polygamma`, `zeta`, `dirichlet_eta`, `lerchphi`, the
  matching on the core's function classes, the scale conditions decided
  by the native `ask`. The old API on SymPy objects moves to the new face
  `sympy_extras.integrals.mellin_sympy` (the same classes as SymPy
  containers with `to_core`/`from_core`, the operations and functions
  running the native code; the entries of `KERNELS` are kept identical),
  which `slater`, `marichev`, `brackets`, `definite`, `laplace`,
  `periodic`, `contours`, `dirichlet`, `series`, `transformations`, the
  tests, the benchmarks, the README and the docs use. The products of two
  functions which are one kernel (two Bessel functions,
  `exp(-x)*besseli`, `erfc(x)*exp(x**2)`) are matched without
  placeholder expressions (`_CombinedBessel`, `_CombinedBesseli`,
  `_CombinedErfc` are gone) and come after the other kernels, as before.
  The core's derivatives of order 2 or more are not simplified as
  SymPy's are, so `mellin_transform` applies SymPy's
  `factor_terms(signsimp(...))` to the transform of `log(x)**n*f(x)`
  itself and the results keep SymPy's form.
- The radical table and the trigonometric integrator are native
  (**breaking**): `sympy_extras.integrals.radicals`
  (`quadratic_radical_antiderivative`,
  `euler_substitution_antiderivative`) and
  `sympy_extras.integrals.trigonometric` (`trigonometric_antiderivative`)
  take and return the expressions and symbols of the core and import no
  SymPy: the coefficients of the radicands are read off the expanded
  polynomial (any coefficients, as SymPy's `Poly` over `EX`), the real
  roots of an Euler substitution are the native `roots`, the products of
  sines and cosines the native `TR8`, the hyperbolic ones written in
  exponentials by hand (SymPy's `rewrite(exp)` forms), the checks the
  native `cancel`, `simplify` and `numerically_equal`, and the solution
  of a vanishing frequency the native `roots` with the roots which
  contradict the assumptions of the symbol left out (as SymPy's `solve`
  does). The rational functions which the Euler and Weierstrass
  substitutions and the integrations by parts leave go to the native
  `ratint` (`sympy_extras.integrals.risch.rationaltools`).
  The SymPy API with the same results is the new faces
  `sympy_extras.integrals.radicals_sympy` and
  `sympy_extras.integrals.trigonometric_sympy`, which `antiderivative`,
  `definite` and `indefinite` call.
- Kovacic's algorithm and the special values of the ODE solutions are
  native (**breaking**): `sympy_extras.solvers.kovacic`
  (`liouvillian_solution`, `dsolve_kovacic`, `normal_form`,
  `KovacicSolution`) and `sympy_extras.solvers._special_values`
  (`general_solution_cases`, `basis_cases`, `solution_cases`,
  `solution_jacobian`, `explicit_solution`, `as_zero`, and the new
  `ode_order`, `determinant`, `defined_equation`) take and return the
  expressions, functions and symbols of the core and import no SymPy:
  the poles are the native `roots` (the irrational numbers and the
  functions of the parameters in the coefficients taken as generators,
  as SymPy's `Poly` does), the Laurent parts of `sqrt(r)` the native
  series, the linear systems of the cases Gaussian elimination on
  expressions (`recurr._solve_linear`), the Jacobians SymPy's
  determinant formulas and Berkowitz's algorithm, the simplifications the
  native `simplify`. SymPy's `integrate` (`integrate_of_core`,
  `integral_of_core` for the unevaluated integrals) and its rational
  Riccati solver (`riccati_of_core`) are called through the new face
  `sympy_extras.solvers.kovacic_sympy`, the candidate points of
  `sympy_extras._special_values.with_special_values` through
  `sympy_extras.solvers._special_values_sympy.special_values_of_core`.
  The old API on SymPy objects moves to these faces, which
  `linear_ode`, `factorization`, `special`, `ode`, `first_order`,
  `second_order`, the tests, the benchmarks, the README and the docs
  use. The second solution of case 2 comes from the conjugate root
  `phi - omega` (Vieta) instead of SymPy's `roots` of the quadratic, so
  its form may differ (`exp(-2*x*sqrt(1/x))` for
  `exp(-2*sqrt(x**3)/x)`); every other recorded call gives the same
  result (see the commit message for the comparison).
- Charpit's method and the differential-algebraic solver are native
  (**breaking**): `sympy_extras.solvers.charpit` (`complete_integral`,
  `check_complete_integral`) takes equations of the core and imports no
  SymPy (the equations polynomial in the unknown are solved by the native
  `roots`, in the order and with the simplification of SymPy's `solve`;
  sums of powers are integrated term by term; the checks use the native
  `simplify`); `sympy_extras.solvers.dae` (`dsolve_dae`,
  `core_nilpotent_decomposition`, `dae_index`, `dae_matrices`,
  `check_dae`, `DAESolution`) works on `ExprMatrix` of the core, with
  exact Gaussian elimination by the native `cancel` and a new
  `matrix_exponential` by Putzer's algorithm (eigenvalues by the native
  `roots`) instead of SymPy's `Matrix.exp`. What stays on SymPy is called
  at run time through the faces: `solve` for equations not polynomial in
  the unknown and `integrate` for integrands which are not sums of powers
  (`charpit_sympy.solve_of_core`, `charpit_sympy.antiderivative_of_core`),
  `integrate` of the particular solution of the regular part
  (`dae_sympy.integral_of_core`, SymPy's `Integral` when it finds none).
  The old API on SymPy objects moves to the faces
  `sympy_extras.solvers.charpit_sympy` and
  `sympy_extras.solvers.dae_sympy`, with the same results on 92 recorded
  calls (the tests, the doctests and about 50 more equations, matrices
  and systems); the benchmarks, the README and the docs use the faces.
  `dae_matrices` raises `ValueError` ("... is not linear in the functions
  and their first derivatives") instead of SymPy's `NonlinearError` (a
  subclass of `ValueError`) for nonlinear equations. The constant factor
  of an integrand stays out of the antiderivative, as in SymPy's
  `integrate`: SymPy's `solve` finds no root of `a*log(u) + log(u) - 1`.

- The Thue equation solver is native (**breaking**):
  `sympy_extras.solvers.thue_equation.thue` takes a form and generators
  (symbols or other expressions, as `Poly`) of the core and imports no
  SymPy (the norms are resultants of the typed
  `euclidtools`, the irreducibility test and the integer roots of
  `small_solutions` come from `factortools.dup_zz_factor`, de Weger's
  reduction solves its rational system by Gaussian elimination on
  `Fraction`s instead of SymPy's `Matrix.LUsolve`; the numerics stay on
  mpmath). The old API on SymPy objects moves to the face
  `sympy_extras.solvers.thue_equation_sympy` (`thue`, and the integer
  level `ThueEquation`, `units_of_order`, `elements_of_norm`,
  `short_vectors`, `lll`, `at_precision` re-exported), with the same
  results on 81 recorded calls (the tests, the doctests and 41 more
  equations: cubic, quartic and quintic forms, several right-hand sides,
  reducible forms, equations without solutions, the refusals); the
  benchmarks, the README and the docs use the face.
  `ThueEquation.small_solutions` now lists the roots for each `y` in
  increasing order (the same set); the errors of the native polynomials
  come out of the face as SymPy's `PolynomialError`.
- The last polynomial modules on SymPy are native (**breaking**): every
  module of `sympy_extras.polys` but `interop` and the SymPy faces
  (`*_sympy`) imports no SymPy (a test of `core/tests/test_independence.py`
  imports them all).
  - `polys.groebnerwalk`: `groebner_walk(G, ring, target, check, lift)`
    (both liftings), `extended_groebner`, `initial_form`, and the new
    `is_groebner` and `normal_form` on the typed `rings.PolyRing` over a
    field (moved from `polys.ideals`, which re-exports `groebner_walk`,
    `normal_form` and `initial_form`). The old API on SymPy's rings and
    orders moves to the face `polys.groebnerwalk_sympy`, computed by the
    native walk over `QQ`, `GF(p)`, `QQ<a>`, `QQ_I` and `QQ(t)`/`ZZ(t)`
    (`NotImplementedError` over the others, `EX`, `RR`); over a ring which
    is not a field it walks over the field of fractions and clears the
    denominators, as SymPy's `groebner` does: the old walk over `ZZ`
    returned `[]` for `(2*x**2 - 3*y, x*y - 4)` (and the cofactor lifting
    failed), now `[8*x - 3*y**2, 3*y**3 - 32]`.
  - `polys.orderings`: the typed orders (`MonomialOrder`, `lex`, `grlex`,
    `grevlex`, `WeightOrder`, `BlockOrder`, `elimination_order`,
    `as_order`), moved from `polys.rings` (which re-exports them); the
    SymPy `MonomialOrder` subclasses of the old module (for SymPy's
    `groebner`) move to `polys.orderings_sympy`.
  - `polys.hilbert`: `hilbert_numerator` returns a `DensePoly` over `ZZ`
    in a core symbol, `hilbert_series` a core expression, and the new
    `reduced_hilbert_series` the reduced numerator and the dimension; the
    SymPy `Poly` and expression move to `polys.hilbert_sympy`
    (`ideals_sympy` uses it).
  - `polys.subresultants`: `dup_psc(f, g, K)` over a typed domain (several
    variables: over a `PolynomialRing` or `SparsePolynomialRing` tower)
    and `psc(f, g)` on `rings.PolyRing`, whose coefficients are elements of
    the same ring free of the first generator; `dup_psc`, `dmp_psc` on
    SymPy's dense representation and `psc` on SymPy's rings (the native
    `psc` over `ZZ`, `QQ`, `GF(p)`) move to `polys.subresultants_sympy`.
  - `polys.polytools` imports no SymPy: `invert` of several generators
    raises `ValueError('univariate polynomial expected')` and of two
    rationals which are not both integers `NotInvertible` natively (what
    SymPy does); the calls to SymPy which remain (`invert` of numbers in
    `ZZ_I`, `RR`, `EX`; `cancel` over `CC`, `EX`, with generators over the
    other unported domains or with a generator which is not a factor;
    `factor` over `EX` and over `ZZ_I`, `QQ_I`, `RR` with generators) are
    `invert_of_core`, `cancel_of_core` and `factor_of_core` of the new
    `polys.polytools_sympy`, imported only when needed.

  On 520 recorded calls (the tests and doctests of these modules and 40
  more) the faces return what the old modules returned, except the walk
  over `ZZ` above; the native modules, inputs and outputs converted,
  return what the faces return on 1097 checks.
- The reliable numerical values and the conditions of the integrals are
  native (**breaking**): `sympy_extras._numeric` (`reliable_form`,
  `reliable_value`, and the new `reliable_complex`) and
  `sympy_extras.integrals.conditions` (`ConditionalValue`, `decide`,
  `canonical`, `real_form`, `positive`, `plain_symbols`, `symbol_facts`,
  `with_symbol_facts`, `sample_values`, `numerically_equal`,
  `numerical_verdict`, `combine`) take and return the objects of the core
  and import no SymPy: the values are computed with mpmath by the core,
  with the same rules (a part which is a rounding error is told by
  evaluations at 15, 30 and 60 more digits, an exact `0.0` of a sum whose
  terms are not zero is a suspect too, the value returned is refined at
  doubling precisions), and a hidden zero is proved by the core's
  `is_zero` or by the native minimal polynomial (SymPy's for a number
  with a wrapped foreign object such as `CRootOf`); the decisions use the
  native `ask`, `find_instance` and structure theorem. The old API on
  SymPy objects moves to the faces `sympy_extras._numeric_sympy` and
  `sympy_extras.integrals.conditions_sympy`, which the integrators written
  on SymPy (33 modules of `sympy_extras.integrals`, `solvers._special_values`,
  `solvers.transcendental`, `concrete.karr_sympy`) now import; the face's
  `ConditionalValue` is the same container on SymPy objects, with
  `to_core`/`from_core`. The package `sympy_extras.integrals` exports
  nothing any more (as `sympy_extras.solvers`), so that importing the
  native module does not load the integrators: `from sympy_extras.integrals
  import definite_integral` becomes `from sympy_extras.integrals.definite
  import definite_integral`, and so on for every name it exported. The
  native `ask`, `limits`, `solve`, the structure theorem and
  `concrete.creative` call the native functions, and the wrappers they
  went through are removed: `ask_sympy.sample_values_of_core`,
  `ask_sympy.reliable_complex_of_core`, `structure_sympy.sample_values_of_core`,
  `structure_sympy.reliable_value_of_core`, `limits_sympy.reliable_value_of_core`,
  `solve_sympy.reliable_value_of_core`, `karr_sympy.reliable_value_of_core`.
  On the 5765 distinct calls recorded from the tests of `integrals`
  (definite, regions, indefinite, marichev, laplace), of `assumptions`,
  of the structure theorem and from the doctests, and 570 more (hidden
  zeros, cuts, conditions), the face gives the old results except two
  values which are zero (an expression in `lerchphi` and `exp_polar`:
  `0.e-167` before, `4.9e-150 - 3.3e-150*I` now); `reliable_value` takes
  26 s instead of 66 s on them, `reliable_form` 6 s instead of 2 s, the
  others the same time or less.
  Fixed on the way: mpmath converts the integer denominator of a float
  such as `1e-1737200` in about a minute, so printing such a core `Float`,
  its numerical value and its conversion to SymPy were that slow; they now
  build the mpmath number from the mantissa and the exponent
  (`core.numbers.fraction_to_mpf`). The core's `meijerg` had no numerical
  value with half-integer parameters (they went to mpmath as complex
  numbers, which its pole test compares with integers).

- The differential algebra is native (**breaking**):
  `sympy_extras.polys.differential` (`DifferentialRing`,
  `DifferentialPolynomial`, `janet_basis`, `JanetBasis`,
  `janet_multiplicative_variables`, `rosenfeld_groebner`,
  `RadicalDifferentialIdeal`, `RegularDifferentialSystem`) takes and
  returns the functions, derivatives, expressions, equalities and symbols
  of the core and imports no SymPy. The coefficients of
  `DifferentialRing.terms` are elements of the typed field `QQ(p, x)`
  (`SparseFractionField`) and those of a `DifferentialPolynomial` elements
  of the typed `ZZ[p, x]` (`IntegerPolynomialRing`), with SymPy's gcd and
  canonical fractions; the saturations are native `Ideal`s, the
  `regular_chains()` native `RegularChain`s and the factors those of the
  typed `polytools.factor_list`. The old API on SymPy objects moves to the
  face `sympy_extras.polys.differential_sympy` (SymPy's `QQ(p, x)` and
  `ZZ[p, x]` coefficients, the chains of `regularchains_sympy`), with the
  same results on 1519 recorded values of the tests, the doctests, the
  callers' tests and 40 more calls; `solvers.lie` uses the face. API
  changes: the attribute `RegularDifferentialSystem._symbols` is now
  `symbols` (a `JetSymbols`), `DifferentialRing` gains `symbols`,
  `terms_of`, `field_element`, `integer_expr` and `coefficient_expr`, and
  the face's ring the conversions `integer_in`/`integer_out`,
  `field_in`/`field_out`. Floats in the equations are refused (SymPy's
  `QQ(p, x)` converted them to rationals). The products and exact
  quotients of the large coefficients go through `ring.integer_product`
  and `ring.integer_quotient` (Python integers directly), and
  `rings.PolyElement.div` updates its remainder and quotients in place
  (the same terms in the same order): the dense Janet basis of the tests
  which took 10 s on SymPy takes 8.5 s.
- The Puiseux expansions and the function fields of algebraic curves are
  native (**breaking**): `sympy_extras.polys.puiseux` (`puiseux_expansions`,
  `expansions_in_field`, `adjoin_root`, `PuiseuxExpansion`,
  `LaurentSeries`, `Embedding`, `field_of`, `minimal_polynomial_of`,
  `in_subfield`) works on core expressions and the typed number fields
  (`AlgebraicField`; the rationals are `RATIONALS`, of degree one; the
  curves are dictionaries `{i: {k: coefficient of x**k y**i}}`;
  `minimal_polynomial_of` returns the monic coefficients; the new
  `field_from_polynomial`, `factor_over`, `curve_from_expr`,
  `PuiseuxExpansion.evaluate_expr`), and `sympy_extras.polys.functionfields`
  (`FunctionField[T]`, `Element[T]`, `function_field(F, x, y, K)`,
  `integral_basis`) is generic in its field of constants (`QQ`, a number
  field, `SparseFractionField` of the parameters); neither imports SymPy.
  The old API on SymPy objects moves to the faces
  `sympy_extras.polys.puiseux_sympy` and
  `sympy_extras.polys.functionfields_sympy` (same results on the recorded
  calls of the tests, the integrals on curves and 40 more curves, except
  the choice between `t` and `-t` in some branches, now SymPy's as before),
  which `sympy_extras.integrals.curves` calls. A new number field is named
  natively by the largest real root of its minimal polynomial
  (`IndexedRoot`, radicals for quadratic and binomial polynomials, the
  symbol `theta` for a field with no real generator and no radicals); the
  face gives SymPy's `QQ.algebraic_field(rootof(N, k, radicals=True))` as
  before. Fixed on the way: the inverse of a basis was cached by the
  identity of the list, so that a basis changed in place, or a new list at
  the address of a freed one (the loop of Trager's algorithm), got a stale
  inverse (the occasional `ValueError: 1/(6*x + 6) is not a polynomial` of
  `test_bases_are_maximal_by_puiseux_expansions` in a full run); and the
  integral basis of a curve with parameters such as `y**3 - a*x**2` no
  longer fails on SymPy's conversion of constants into `QQ(a)(x)`.

- The decompositions of ideals and the rational univariate representation
  are native (**breaking**): `sympy_extras.polys.idealdecomposition`
  (`radical`, `equidimensional_parts`, `minimal_primes`,
  `primary_decomposition`, `associated_primes`, `is_primary`,
  `independent_set`) takes and returns the native ideals of
  `sympy_extras.polys.ideals` (over `QQ`), with the native regular chains,
  F4 and the typed `polytools.Poly` factorisation, and
  `sympy_extras.polys.rur` (`rur`, `RationalUnivariateRepresentation`)
  takes native ideals or core expressions and returns `DensePoly`
  polynomials over `QQ` in `t`; neither imports SymPy. The old API on
  SymPy objects moves to the faces `sympy_extras.polys.idealdecomposition_sympy`
  and `sympy_extras.polys.rur_sympy` (same results on 704 recorded calls of
  the tests and doctests and 117 more). The native `Ideal` calls the
  decompositions directly and gains `Ideal.rur`; the run-time routes
  `radical_of_core`, `equidimensional_parts_of_core`,
  `minimal_primes_of_core`, `primary_decomposition_of_core`,
  `associated_primes_of_core` and `is_primary_of_core` of `ideals_sympy`
  are removed, and so is the quotient algebra `hermite_sympy._Algebra`
  which the old `rur` built on. One step stays on SymPy, at run time: the
  exact solutions of a representation (`CRootOf`,
  `rur_sympy.solutions_of_core`). `DensePoly.to_expr` and `from_expr` keep
  a `Dummy` generator (they wrote it as the `Symbol` of the same name).

- The modular Gröbner bases are native (**breaking**):
  `sympy_extras.polys.modulargroebner` (`modular_groebner`,
  `direct_groebner`, `groebner`, `rational_reconstruction`,
  `chinese_remainder`, `default_primes`, `ModularTrace`, and the new
  `homogeneous_order`) works on the typed `rings.PolyRing` over `QQ` and
  `ZZ` (the images over the typed `GF(p)`) and imports no SymPy;
  `direct_groebner` and `groebner` take any typed field. The old API on
  SymPy's ring elements, with the same results, is the face
  `sympy_extras.polys.modulargroebner_sympy` (SymPy's `groebner` for the
  other domains and the orders which are not global, as before): callers
  of `modulargroebner` with SymPy rings import from the face (the private
  `_HomogeneousOrder` is gone, `_image` takes variable names and one
  typed order). New `sympy_extras.polys.buchberger` (`buchberger`,
  `s_polynomial`): a typed port of SymPy's Buchberger algorithm, which
  `settings.groebner_algorithm = 'buchberger'` now selects everywhere,
  the images of the modular algorithm included (with these images the
  modular bases of Katsura-4/5, cyclic-5 and random dense systems took
  1.3 to 4.2 times less CPU time than with SymPy's; with the default F4
  images the times are unchanged). The native `polys.ideals.groebner` calls
  the native dispatcher: `modular_groebner_of_core` and
  `direct_groebner_of_core` are removed from `polys.ideals_sympy`, the
  modular algorithm now also serves the typed orders without a SymPy
  counterpart (weight orders), and with `'buchberger'` the modular
  algorithm takes over after `settings.groebner_direct_time` as it does
  with F4.
- The regular chains and the cases of systems with parameters are native
  (**breaking**): `sympy_extras.polys.regularchains` (`RegularChain`,
  `triangularize`, `regular_gcd`, and the modules `recursive` and
  `operations`) and `sympy_extras.solvers.parametric` (`parametric_cases`,
  `ParametricCase`) take and return the expressions, equalities, symbols,
  Booleans and sets of the core and import no SymPy. The polynomials are
  the native `rings.PolyElement` over `ZZ` (the gcds of
  `IntegerPolynomialRing`, Wang's `multifactor.zz_factor`); SymPy's
  `PolyElement.prem` and `PolyElement.subresultants`, behind the
  subresultant chains, and `Poly.nroots`, behind `numerical_solutions`
  (on mpmath), are ported. `RegularChain.saturated_ideal` is a native
  `ideals.Ideal` over `QQ`, `ParametricCase.solutions` a native
  `PointSet`. One step stays on SymPy and goes through it at run time: the
  exact solutions of a chain without free variables (`RegularChain.solutions`,
  SymPy's `CRootOf`), whose module `regularchains.solutions` moved to the
  face as `regularchains_sympy.solutions`. The old SymPy API is in the new
  faces `sympy_extras.polys.regularchains_sympy` (`RegularChain`,
  `triangularize`, `regular_gcd`, `exact_solutions`) and
  `sympy_extras.solvers.parametric_sympy` (`parametric_cases`,
  `ParametricCase`); **import them instead of `regularchains` and
  `solvers.parametric`** for SymPy objects. `idealdecomposition` and
  `differential.rosenfeld_groebner` use the face. The native
  `assumptions.solve` calls the native modules directly:
  `solve_sympy.triangularize_of_core` and
  `solve_sympy.parametric_cases_of_core` are removed, and so are
  `assumptions.solve.TriangularChain` and `assumptions.solve.ParametricCase`
  (the native `RegularChain` and `parametric.ParametricCase` take their
  place). The results are those of the old modules on every call made by
  the tests, the doctests and 143 more (1076 calls), in the same time (9.3 s
  for their 360 decompositions); `numerical_solutions` is three times
  faster. The heuristic gcd of `heuristicgcd` evaluates its univariate
  polynomials by Horner's scheme and its multivariate ones with the
  powers computed once (it computed each power of the evaluation point
  again: 2.4 s of the 8 s of the decomposition of a parametric quartic).
- Hermite's root counting and the comprehensive Gröbner systems are native
  (**breaking**): `sympy_extras.polys.hermite` (`count_real_solutions`,
  `count_complex_solutions`, `tarski_query`, `real_sign_counts`,
  `sign_determination`, `hermite_matrix`, `decide_zero_dimensional`) and
  `sympy_extras.polys.comprehensive` (`comprehensive_groebner_system`,
  `exists_complex`, `complex_quantifier_elimination`, `reduce_complex`,
  `Branch`) take and return the expressions, symbols and formulas of the
  core and the native ideals of `sympy_extras.polys.ideals` over `QQ`,
  compute with Python fractions and F4, and import no SymPy.
  `hermite_matrix` gives an `ExprMatrix`, a native `Branch` holds tuples,
  and a system with coefficients that are not rational raises
  `ValueError` (the face keeps `NotImplementedError`). The old SymPy API
  is in the new faces `hermite_sympy` (also the quotient algebra with
  vectors over SymPy's `QQ` on which `rur` builds) and
  `comprehensive_sympy`, with the same results on every call of the tests,
  the doctests and the assumptions; **import them instead of `hermite` and
  `comprehensive`** for SymPy objects. The native `satisfiable` (the proof
  that a zero-dimensional system has no real solution, now under a native
  time limit) and `resolve(..., domain=Complexes)` call them directly: the
  functions `decide_zero_dimensional_of_core` (`sat_sympy`),
  `complex_quantifier_elimination_of_core` and `reduce_complex_of_core`
  (`resolve_sympy`) are removed.
- The top layer of the cylindrical algebraic decomposition is native
  (**breaking**): `sympy_extras.polys.cad.qe` (`quantifier_elimination`,
  `decide`, `sample_points`, `solution_set`, `truth_tables`) and
  `sympy_extras.polys.cad.cylindrical` (`cylindrical_formula`,
  `cylindrical_set`, `cylindrical_cases`) take and return the formulas,
  expressions, symbols and sets of the core, run on the native lifting and
  import no SymPy (the prime implicants and the minimum covers of the
  solution formulas were already the package's own). Their real algebraic
  numbers are rational numbers and nodes of the new
  `sympy_extras.polys.cad.indexedroot`: `IndexedRoot(f, t, k)`, the `k`-th
  distinct real root of `f` in the bound variable `t`, is an immutable node
  of the core (it was a SymPy `Function`), with the evaluating function
  `indexed_root` (the bound variable made the dummy of its name, rational
  coefficients giving a number), `real_root` (an `Algebraic` as the node of
  its minimal polynomial, SymPy's `CRootOf`), `real_number` (the same in
  radicals when `roots.radical_form` allows) and `algebraic_of`; the root
  with rational coefficients evaluates exactly, compares, has signs, has
  `polynomial_root` (so `radical_form` and `in_radicals` take it), and
  `interop` converts it to `CRootOf` and the root functions to the SymPy
  `IndexedRoot` both ways. The old SymPy API is in the new faces
  `qe_sympy`, `cylindrical_sympy` (with `_compile` for the region
  integrals) and `indexedroot_sympy` (the SymPy `IndexedRoot`); **import
  them instead of `qe` and `cylindrical`** for SymPy objects. One result
  differs: a bound of a cylindrical description which is a root of a
  binomial whose `CRootOf` SymPy scales (`3*CRootOf(4*x**3 - 1, 0)`) is now
  written in radicals (`3*2**(1/3)/2`), as the other roots of binomials
  were. The native `sat` (`satisfiable`, `find_instance`), `solve`,
  `resolve`, `ask` and `analysis.domain_of` call the native CAD directly:
  `sat_sympy.sample_points_of_core` and `.solution_set_of_core`,
  `solve_sympy.solution_set_of_core`, `.cylindrical_set_of_core`,
  `.cylindrical_formula_of_core` and `.cylindrical_cases_of_core`,
  `resolve_sympy.quantifier_elimination_of_core`,
  `ask_sympy.truth_tables_of_core` and `analysis_sympy.solution_set_of_core`
  are gone; the floor and the comparison of the real roots of the CAD in
  `sat` are native (`floor_of_core` and `compare_real_of_core` stay for
  SymPy's `CRootOf`), and `domain_of` intersects natively, with SymPy's
  intersection (`intersection_of_core`) when the core does not compare
  the ends.
- `sympy_extras.polys.virtual_substitution` is native (**breaking**: core
  expressions, symbols and Booleans in and out, prefixes of core symbols;
  it imports no SymPy). The coefficients in the eliminated variable come
  from the native `Poly` (with the expression coefficients of its
  decomposition for the domains it does not port, `QQ(a)` and `EX`), the
  normal form of the relations from the native `assumptions.facts`; it
  holds the port of SymPy's `to_nnf` (now public, `to_nnf`) and
  `with_sides`, which `solvers.integers` imports from it (and still
  exports). SymPy's `simplify_logic` of a compound result is called at
  run time through `virtual_substitution_sympy.simplify_logic_of_core`
  (which `integers_sympy` re-exports); a result which is a truth value or
  a single relation does not go through SymPy. The new face
  `sympy_extras.polys.virtual_substitution_sympy` keeps the old API on
  SymPy objects; **import it instead of the old module**: `from
  sympy_extras.polys.virtual_substitution_sympy import eliminate_linear`.
  The native `resolve` and `ask` call the native functions directly:
  `assumptions.resolve_sympy.virtual_substitution_elimination_of_core`,
  `resolve_sympy.with_sides_of_core`, `assumptions.ask_sympy.is_linear_in_of_core`
  and `ask_sympy.linear_quantifier_elimination_of_core` are removed. The
  tests, docs and benchmarks use the face. The same results as before
  (`srepr`) through the face, and natively (inputs and outputs converted
  by `interop`), on 120 calls: the doctests, the tests of the module, the
  `resolve` calls reaching it and 65 linear and quadratic eliminations
  with parameters.
- The bottom layer of the cylindrical algebraic decomposition is native
  (**breaking**): `sympy_extras.polys.cad.projection`, `.samplepoints` and
  `.lifting` work on the typed `Poly` of `polys.polytools` in symbols of the
  core and import no SymPy. The sample points are real
  `qqbar.Algebraic` numbers (minimal polynomial and isolating interval;
  `compare_real` tells two isolations of one number equal by a Sturm count),
  their fields the native `AlgebraicField` of a primitive element, the real
  roots over `QQ(theta)` come from the norm (`factortools.dup_ext_norm`) and
  Sturm sequences over the field, Hong's principal subresultant
  coefficients from the typed `dup_inner_subresultants` over
  `IntegerPolynomialRing`. Cells give their points as `Algebraic` numbers;
  `CADCell.factor_sign`, `lifting.factor_signs`, `projection.to_polys`,
  `PROJECTIONS`, `KEPT`, `samplepoints.floor_scaled`, `real_roots_of` are the
  public names of what was private. The new faces `projection_sympy`,
  `samplepoints_sympy` and `lifting_sympy` keep the old API on SymPy objects
  (`Poly`, `Rational`, `CRootOf`; the points are named `CRootOf(m, i)` of
  the minimal polynomial in the variable of the level, so that the printed
  variable of a root may differ, as it did with SymPy's cache) and are what
  `qe`, `cylindrical`, `integrals.regions` and the assumptions use; **import
  them instead of the old modules**. The face's `SamplePoint` has no
  `field` and `coords` (its `native` point has them), `lifting._merge_roots`
  is gone. **`sympy_extras.polys.cad` exports nothing**: import from
  `polys.cad.qe` (`quantifier_elimination`, `decide`, `sample_points`,
  `solution_set`, `truth_tables`), `polys.cad.cylindrical`
  (`cylindrical_formula`, `cylindrical_set`, `cylindrical_cases`,
  `IndexedRoot`), `polys.cad.lifting_sympy` (`cylindrical_algebraic_decomposition`,
  `CAD`, `CADCell`, `NotWellOriented`), `polys.cad.projection_sympy` and
  `polys.cad.samplepoints_sympy`. Native `Poly`: `discriminant` divides by
  the leading coefficient exactly (it raised when that coefficient was a
  polynomial in the other generators), `__eq__` and `__hash__` work on the
  representation (no expression is built), new `Poly.from_dict`, `rep`,
  `is_integral`; `qqbar.Algebraic.real_interval` and `refine_real`.
  `interop`: `subs` and `xreplace` no longer go into the polynomial of a
  wrapped `CRootOf` (its variable is bound; `(x + CRootOf(x**3 - x - 1,
  0)).subs(x, 1)` raised `GeneratorsNeeded`).
- `sympy_extras.polys.ideals` is native (**breaking**: it imports no SymPy;
  its `Ideal` takes core expressions or typed ring elements, core symbols,
  a required typed coefficient field `domain=` of
  `sympy_extras.polys.domains` and a typed order; bases are lists of
  `PolyElement`, univariate and minimal polynomials `DensePoly`,
  multiplication matrices lists of rows, `reduce` and `standard_monomials`
  core expressions; no `rur` method). Gröbner bases by F4 over every field
  (algebraic fields and fields of fractions went to SymPy's Buchberger
  before), FGLM and a native Gröbner walk (`groebner_walk`,
  `initial_form`: F4 on the initial forms, lifting by normal forms) for
  `change_order`, a native `normal_form`, the Hilbert series by the new
  `sympy_extras.polys.bigatti` (`hilbert_numerator_coefficients`,
  `minimal_generators`, also used by `polys.hilbert`), the minimal
  polynomial by an incremental Krylov elimination, `separating_forms` on
  core symbols and `separating_coefficients`. `sympy_extras.polys.rings`
  gains `as_order`, `WeightOrder`, `BlockOrder` and `elimination_order`
  (named as the orders of `polys.orderings`). What is not native is called
  at run time through the new face `sympy_extras.polys.ideals_sympy` (the
  `*_of_core` functions): `modular_groebner_of_core` (over `QQ`, when F4
  has not finished within `settings.groebner_direct_time`),
  `direct_groebner_of_core` (`settings.groebner_algorithm = 'buchberger'`),
  and `radical_of_core`, `equidimensional_parts_of_core`,
  `minimal_primes_of_core`, `primary_decomposition_of_core`,
  `associated_primes_of_core`, `is_primary_of_core` (the decompositions of
  `idealdecomposition` through regular chains). The face keeps the old API
  on SymPy objects (domain inference with `extension=True`, `Poly` bases,
  SymPy rings in `ring()` and `_basis()`, matrices, `rur`, the
  decompositions; an explicit `domain=ZZ` now becomes `QQ` as an inferred
  one did); **import it instead of the old module**: `from
  sympy_extras.polys.ideals_sympy import Ideal, separating_forms,
  hilbert_numerator`. `hermite`, `rur`, `comprehensive`,
  `idealdecomposition`, `regularchains`, `differential.rosenfeld_groebner`,
  the tests, docs and the in-repository benchmarks use the face. The same
  results as before through the face on 8037 recorded calls (the doctests
  and tests of `ideals`, `hermite`, `rur`, `comprehensive`,
  `idealdecomposition`, the primary decomposition, `regularchains`,
  `rosenfeld_groebner` and `solvers.parametric`, and 50 more over `GF(7)`,
  `QQ(sqrt 2)`, `QQ_I`, `QQ(a)`, weight and block orders, Katsura-4,
  cyclic-4 and cyclic-5), and the same results natively (inputs and
  outputs converted by `interop`) on all of them but the three `rur`
  calls; replaying them took 26.6 s through the face against 31.2 s
  before (`change_order` 3.1 s to 0.6 s, saturations, quotients and
  intersections about twice as fast).
- `sympy_extras.solvers.isolation`, `sympy_extras.solvers.integers` and
  `sympy_extras.polys.roots` are native (**breaking**: core expressions,
  symbols, Booleans and sets in and out; they import no SymPy). The
  isolation runs on the native `analysis` and `intervals` (signs, limits,
  zeros of rational functions), the native `Poly`, `preprocess_roots` and
  Sturm counts (the real roots of a polynomial in radicals as SymPy's
  `real_roots` writes them); its `TranscendentalRoot` is a node of the core
  (immutable, `refined(k)` instead of `refine(k)`, `lower_fraction`,
  `upper_fraction`, `approximation(prec)`; a number for `is_number`
  through the new hook `Expr._is_number_node`). The Diophantine systems
  use the native `Poly`; Cooper's algorithm the core's Booleans with a
  port of SymPy's `to_nnf` and a native `with_sides`. The radical forms
  use the native `polyroots` and recognise SymPy's `CRootOf` wrapped in
  the core. What is not ported is called at run time through the new
  faces (`*_of_core`): `isolation_sympy.real_roots_of_core` (SymPy's
  `real_roots`, for the factors needing `CRootOf`),
  `isolation_sympy.continuous_on_of_core` (`continuous_domain` of a
  function which is not rational), `integers_sympy.simplify_logic_of_core`
  and `roots_sympy.roots_of_core` (SymPy's `roots` over `EX`). The faces
  `isolation_sympy`, `integers_sympy` and `roots_sympy` keep the old API on
  SymPy objects (`isolation_sympy.TranscendentalRoot` is the SymPy
  expression, which `interop` converts to and from the native node);
  **import them instead of the old modules**: `from
  sympy_extras.solvers.isolation_sympy import isolate_real_roots`. The
  native `solve` calls the three modules directly and the native `resolve`
  calls the native `is_presburger` and `presburger_quantifier_elimination`:
  `assumptions.solve_sympy.isolate_real_roots_of_core`,
  `point_between_of_core`, `compare_points_of_core`,
  `linear_system_of_core`, `linear_diophantine_system_of_core`,
  `minimal_nonnegative_solutions_of_core`, `in_radicals_of_core`,
  `assumptions.resolve_sympy.is_presburger_of_core` and
  `presburger_quantifier_elimination_of_core` are removed. The CAD,
  `regularchains.solutions`, `solvers.transcendental`, the tests, docs and
  benchmarks use the faces. The same results as before through the faces
  on 134 calls and on the calls of the doctests and of the tests of the
  three modules and of `solve`, and the same results natively (converted
  by `interop`) on the 134 calls; the forms of Cooper's results which
  differ come from SymPy's `simplify_logic`, which depends on the hash
  seed.
- `sympy_extras.simplify.structure` (the structure theorem: `ElementaryTower`,
  `canonical_form`, `is_zero`, `equal`) is native (**breaking**: core
  expressions and Booleans in and out; it imports no SymPy). The number
  field is the native `AlgebraicField` of a primitive element
  (`qqbar.primitive_element`), numbers are placed in it exactly
  (`qqbar.Algebraic`, a linear factor of the minimal polynomial over the
  field), the polynomials are the native sparse rings (`cancel` through
  `RationalPolynomialRing` over `QQ`), the multivariate factorisation over a
  number field is the new `polys.algebraicfactor` (Trager:
  `ext_factor_list`, `AlgebraicPolynomialRing`), the Smith normal form is
  `polys.lattice`'s, the PSLQ coordinates come from exact enclosures, and
  the signs from the native `ask`. Two steps go through the new face
  `simplify.structure_sympy` at run time: `sample_values_of_core`
  (`integrals.conditions.sample_values`) and `reliable_value_of_core`
  (`_numeric.reliable_value`). `structure_sympy` keeps the old API on SymPy
  objects; **import it instead of the old module**: `from
  sympy_extras.simplify.structure_sympy import is_zero, canonical_form,
  ElementaryTower`. Its `Element` wraps the native element (`.native`; no
  `num`/`den` on SymPy rings), its `Generator` shows `kind`, `expression`,
  `certified`, `index`, `source`, `argument`, `radicand`, `scale`,
  `eliminated` (no `symbol`), and the internals of the old tower
  (`ring`, `domain`, `_solve`, ...) are those of the native one; `as_rational`
  is gone. The native `Element.times` and `ElementaryTower.constant` take a
  `Fraction`. The native `ask` and `refine` call the native tower directly:
  `assumptions.ask_sympy.structure_zero_of_core` and
  `assumptions.refine_sympy.proved_simplification_of_core` are removed.
- `sympy_extras.assumptions.limits` is native (**breaking**: core
  expressions, symbols and Booleans in and out; the native `ask`,
  `facts`, `refine`, `sat`, `resolve`, the core's `limit`/`Limit`
  (Gruntz), `series` and `AccumBounds`, and the native `simplify`
  underneath; it imports no SymPy). What is not ported is called at run
  time through the new face `assumptions.limits_sympy` (the `*_of_core`
  functions): SymPy's `limit_seq` and `_numeric.reliable_value`.
  `limits_sympy.limit`, `limit_seq` and `series` keep the old API on
  SymPy objects; **import them instead of the old module**: `from
  sympy_extras.assumptions.limits_sympy import limit`. The integrals,
  `concrete.convergence`, the tests, README, docs and benchmarks use it.
  The same `srepr` as before through the face on 123 calls (the
  doctests, the calls of the tests of limits and 69 more with parametric
  signs, sequences and series), and natively with the inputs converted
  and the result converted back.
- The core's `expand` runs the `log` hint over the whole tree before
  distributing products, in SymPy's order: `log(a*(-1 - I))` with `a < 0`
  is `log(-a) + log(2)/2 + I*pi/4` (it became `log(-a - a*I)`), so
  `cancel` and the leading terms of sums of such logarithms simplify as
  SymPy's do. The core's Gruntz computes the mrv set of `hyper`,
  `meijerg` and the other functions which are not `Application`s (it
  raised `NotImplementedError`). Both left limits unevaluated or
  unsimplified through the native `assumptions.limits`, and broke
  `test_definite`'s binomial differentials and `test_regions`' merged
  cases.
- `functions.elementary.exponential.exp` of `oo` (or `-oo`) times a
  number evaluates as SymPy's: `zoo` when the real part of the number
  is positive and its imaginary part nonzero, `0` when it is negative,
  `nan` when it is zero (`exp(oo*sign(log(5/2) + I*pi))` stayed
  unevaluated, and the core's limit of `(-5/2)**x` at `oo` with it).
- `sympy_extras.assumptions.solve` is native (**breaking**: core
  expressions, Booleans, symbols and sets in and out, the native `ask`,
  `facts`, `refine` and `analysis` underneath; it imports no SymPy).
  Rational equations over the complex numbers with numerators of degree
  at most 3 are solved by the native `polys.rationalsolve.solveset_rational`
  (its quartics are written in another form than SymPy's), the families
  of a regular chain with the native `Poly`, `cancel` and `sqrt`, the
  candidates checked with the native `simplify`, `minimal_polynomial`
  and evaluation. What is not ported is called at run time through the
  new face `assumptions.solve_sympy` (the `*_of_core` functions): SymPy's
  `solveset`, `nonlinsolve`, `periodicity` and the intersections the
  core's sets do not evaluate, the CAD (`solution_set`,
  `cylindrical_set`, `cylindrical_formula`, `cylindrical_cases`) and
  `polys.roots.in_radicals`, `regularchains.triangularize` and
  `solvers.parametric.parametric_cases`, `solvers.transcendental`,
  `solvers.isolation`, `solvers.integers` (linear Diophantine systems)
  and `_numeric.reliable_value`. `solve_sympy.solve` keeps the old API
  on SymPy objects (also `_refuted` and `_clearly_not_real`); **import
  it instead of the old module**: `from
  sympy_extras.assumptions.solve_sympy import solve`. The integrals,
  `concrete.convergence`, the tests, README, docs and benchmarks use it.
  The same classes and `srepr` as before through the face on 189 calls
  (the doctests, the calls of the tests of solve and of the solvers, and
  68 more), and natively with the inputs converted and the result
  converted back.
- `sympy_extras.sets`: `ConditionSet`/`condition_set`
  (`sets.conditionset`), `ImageSet`/`image_set` (`sets.fancysets`; the
  function is its variables and an image which may be a tuple),
  `ProductSet`/`product_set` and `PointSet`/`point_set` (a finite set of
  tuples, SymPy's `FiniteSet` of `Tuple`), with their conversions;
  `interop.SymPySetNode` wraps the sets of SymPy not ported (`Range`,
  ...) as sets of the core (memberships, emptiness and bounds asked of
  SymPy), where `SymPyNode` did. `Union` and `Intersection` print their
  sets sorted as SymPy does.
- Fixed on the way: the order of the sets of an intersection raised
  `ValueError` for a finite set of complex numbers (their infimum; SymPy's
  key catches it); `from_sympy` gave `Reals` for SymPy's `Interval(-oo,
  oo)`, which equals it in SymPy (the cache is keyed by the class too).
- `sympy_extras.assumptions.sat` is native (**breaking**: `satisfiable`,
  `tautology` and `find_instance` take and return core Booleans,
  symbols, sets and expressions, and the module imports no SymPy). The
  propositional abstraction, the theory check of the literals, the
  search for integer points (the floor and the comparison of rational
  endpoints included) and the evaluation of the candidates run on the
  core, with the native `ask._facts`, `facts`, `quantifiers` and
  `resolve`. What is not ported is called at run time through the new
  face `assumptions.sat_sympy` (the `*_of_core` functions): SymPy's SAT
  solver on the abstraction, `facts_sympy.predicates_consistent`, the
  CAD's `sample_points` and `solution_set`, `hermite.decide_zero_dimensional`,
  and the floor and the comparison of `CRootOf` numbers. The face keeps
  the old API on SymPy objects (also `Model`, `Witness`,
  `_without_real_solution`); **import it instead of the old module**:
  `from sympy_extras.assumptions.sat_sympy import satisfiable`. `limits`,
  the integrals, `_special_values`, the tests, README, docs and
  benchmarks use it. The native `find_instance` takes `domain=None` (no
  domain) like the old one. The same printed results and classes as
  before through the face on 201 calls (the 20 doctest examples, 90
  calls of `test_sat` and 91 more), except where the old function
  already depended on the hash seed (which of two equivalent models
  SymPy's SAT solver enumerates first, 5 calls); through the native
  functions with converted inputs, the same printed values (the classes
  of the core differ: `Integer` for `0` and `1`, `Rational` for `1/2`,
  `CRootOf` wrapped as `SymPyExprNode`).
- `sympy_extras.assumptions.ask`, `sympy_extras.assumptions.refine` and
  `sympy_extras.assumptions.resolve` are native (**breaking**: core
  expressions, Booleans, sets and the native `Facts` in and out; none of
  them imports SymPy). `ask` asks the core's `ask` (which asks SymPy's,
  through the bridge, what it cannot answer), decides the reality of
  roots, logarithms and inverse functions, and calls the native
  `analysis.decide_relational` and `bounds.polynomial_abstraction`;
  `refine` and `simplify` abstract the symbols and subexpressions into
  dummies of the core and apply the core's automatic evaluation, its
  `refine` and the native `simplify`, `factor_list`, `posify`,
  `powsimp`, `powdenest`, `logcombine` and `expand_log`; `resolve` puts
  the formula in prenex form and simplifies the result under the
  assumptions natively. What is not ported is called at run time through
  the new faces: the CAD's `truth_tables` and `quantifier_elimination`,
  virtual substitution (`is_linear_in`, `linear_quantifier_elimination`,
  `virtual_substitution_elimination`, `with_sides`), Cooper's algorithm
  (`is_presburger`, `presburger_quantifier_elimination`), comprehensive
  Gröbner systems (`complex_quantifier_elimination`, `reduce_complex`),
  the structure theorem's `ElementaryTower` (the zero test of `ask`, the
  proofs of `simplify`), `integrals.conditions.sample_values` and
  `_numeric.reliable_value`, SymPy's `simplify` of a Boolean formula
  (`simplify_logic` and the rules of the relations) and SymPy's
  `factor_list` of a polynomial with coefficients which are not rational.
  The new faces `assumptions.ask_sympy`, `assumptions.refine_sympy` and
  `assumptions.resolve_sympy` keep the old API on SymPy objects (also the
  helpers `Assumptions`, `_facts`, `_evaluate`, `_evaluate_atom`,
  `_linear_ask`, `_clearly_not_zero`, `_Refiner`, `_refine_boolean`);
  **import them instead of the old modules**: `from
  sympy_extras.assumptions.ask_sympy import ask`, `refine` and `simplify`
  from `.refine_sympy`, `resolve` from `.resolve_sympy`. `sat`, `limits`,
  `solve`, the integrals, `concrete.convergence`, `simplify.structure`,
  the docs and the benchmarks use the faces. The face's `simplify` takes
  `ratio`, `rational`, `inverse` and `doit` as keyword arguments (no
  `measure` and no other keyword of SymPy's `simplify` any more). The
  same printed results and classes as before through the faces on 98
  extra calls, the 45 doctest examples and the 439 calls of the tests of
  the three modules; through the native functions with the inputs
  converted, the same printed results (the classes of the core differ:
  `Integer` for `0` and `1`, `Gt` for `StrictGreaterThan`, ...).
- Fixed in the core on the way: `sqrt(a)` of a symbol not known to be a
  number was not negative (`is_extended_negative` `False`); as in SymPy's
  `Pow._eval_is_extended_negative` that needs a complex or extended real
  base (`None` otherwise). Asked through the core's `ask`, it made
  `solve(x**2 - a, x, x > 0)` drop `-sqrt(a)`.
- `sympy_extras.assumptions.intervals` and `sympy_extras.assumptions.analysis`
  are native (**breaking**: core expressions, symbols, sets and the native
  `Facts` in and out; neither imports SymPy). The interval arithmetic,
  branch and bound, monotonicity at the corners (limits by the core's
  `limit`), the root isolation, `certified_sign` (the core's predicates,
  the minimal polynomial of an algebraic constant for a hidden zero, and
  with the numerical checks on rigorous interval arithmetic at the
  precision of the settings, or for the functions it does not cover a
  value with a margin over its precision), the bounds by the core's
  `AccumBounds`, `domain_of`, `box_of`, `signs_multivariate` and
  `decide_relational` run on the core; the zeros and the continuity of
  a rational function come from `polys.rationalsolve.solveset_rational`.
  Three steps stay on SymPy, called at run time through the new faces:
  `continuous_domain` and `solveset` for the expressions which are not
  rational (with SymPy's `ImageSet` for the periodic zeros), and the CAD's
  `solution_set` (with SymPy's intersection of its intervals, whose ends
  may be `CRootOf`) for a polynomial assumption which is not a bound.
  `range_on_box` and `monotone_range` return a `Set` (a point is a
  `FiniteSet`, as before through SymPy's `Interval`). The new faces
  `assumptions.intervals_sympy` and `assumptions.analysis_sympy` keep the
  old API on SymPy objects (also the helpers `_pieces`, `_between`,
  `_sorted_constants`, `_endpoint_value`, `_has_discrete_zeros` used by
  `solvers.isolation`); `ask`, `solve`, `solvers.isolation`,
  `concrete.convergence` and `integrals.validated` use them, and
  `facts_sympy.Facts.native` gives the native facts of the face. The
  same printed results as before on 202 compared calls through the faces
  and through the native functions, except `certified_sign` of
  `exp(pi*sqrt(163)) - 640320**3 - 744` with the numerical checks off
  (SymPy's predicates evaluate it at an adaptive precision, the core's do
  not: `None` instead of `-1`).
- Fixed in the core on the way: a sum of accumulation bounds with `nan`
  recursed without end (it is `nan`, as in SymPy); `evalf` of a product
  of numbers without a value (`-atan(AccumBounds(0, oo))`) recursed
  without end; a `CRootOf` wrapped from SymPy has SymPy's numerical value
  inside the sums of the core (`evalf`, signs).
- **Breaking:** `sympy_extras.assumptions` exports nothing any more, like
  `concrete`, `simplify` and `solvers`: import the functions from their
  modules (`from sympy_extras.assumptions.ask import ask`; `refine` and
  `simplify` from `.refine`, `resolve` from `.resolve`, `satisfiable`,
  `tautology`, `find_instance` from `.sat`, `solve` from `.solve`,
  `limit`, `limit_seq`, `series` from `.limits`, `element` and `Facts`
  from `.facts_sympy`, `ForAll`, `Exists`, `Quantifier`, `prenex` from
  `.quantifiers_sympy`, `global_assumptions`, `assuming`,
  `AssumptionsContext` from `.context_sympy`), so that importing the
  native leaves does not load the modules still written on SymPy.
- The leaves of `sympy_extras.assumptions` are native (**breaking**: core
  expressions and Booleans in and out of `assumptions.facts`,
  `assumptions.quantifiers`, `assumptions.bounds` and
  `assumptions.context`, which import no SymPy). `ForAll` and `Exists`
  are Boolean nodes of the core which never evaluate, `forall` and
  `exists` the functions which do (the formula for a constant one or
  one without the variables, the absent variables dropped); `prenex`,
  `element`, `normalize`, `conjuncts`, `to_predicates`, `to_polynomial`,
  `Facts` (immutable: its attributes are `Final`, and it asks the core's
  `ask`), `polynomial_abstraction` (MetiTarski bounds), the native
  `global_assumptions` and an `assuming` that pushes the predicates to the
  core's global assumptions. The SymPy faces `facts_sympy`,
  `quantifiers_sympy`, `bounds_sympy` and `context_sympy` take and
  return SymPy objects, converted by `sympy_extras.interop` around the
  native functions; `ForAll`/`Exists` of the face are SymPy Booleans which
  `interop` converts to and from the nodes, the face's
  `global_assumptions` is the native collection seen with SymPy objects,
  and `predicates_consistent` stays on SymPy (SymPy's SAT solver on its
  known facts of the predicates). The upper modules (`ask`, `refine`,
  `resolve`, `sat`, `limits`, `solve`, `analysis`, `intervals`), the
  integrals and the docs use the faces. The same printed results as
  before on 159 compared calls through the faces, and through the
  native functions with the inputs converted (the native `element` and
  `predicates_consistent` excepted: the first takes only the sets of
  `sympy_extras.sets`, the second is not native).
- `sympy_extras.sets.fancysets`: `Naturals`, `Naturals0`, `Integers`,
  `Rationals` and `Complexes` on the core (memberships as SymPy's,
  `Complexes` by `is_complex`), converted to and from SymPy's by
  `interop`. `interop.SymPyBooleanNode` wraps the Booleans of SymPy with
  no class here (`Contains(x, ProductSet(...))`), which were wrapped as
  non-Boolean nodes before.
- `sympy_extras.concrete.qhyper` is native (breaking: core expressions in
  and out): `QPochhammer` is a node of the core which never evaluates,
  `qpochhammer` the function which does (`QPochhammer(a, q, 2)` was the
  product, it is now `qpochhammer(a, q, 2)`); `qbinomial`, `q_ratio`,
  `normal_in`, `qgosper_term`, `qgosper_sum` and `qzeilberger` on the
  core's `cancel`, `factor`, `gcd`, `lcm_list`, `powsimp` and `expand`,
  the null space of `concrete.zeilberger` and the linear solver of
  `concrete.gosper`; `QTelescoper` is immutable. Only the isolated
  values of the parameters go through SymPy, at the call. The new
  `sympy_extras.concrete.qhyper_sympy` is the face on SymPy expressions
  (`QPochhammer` as a SymPy function, which `sympy_extras.interop`
  converts to and from the node), used by `docs/comparison.md` and
  `benchmarks/comparison.py`. The same printed results as
  before on 64 of 66 compared calls (q-Pochhammer symbols, q-binomials,
  ratios, 12 antidifferences and 21 sums with and without special
  values, 8 q-Zeilberger calls); the other two: the sum of
  `q**k/((1 - q**k)*(1 - q**(k + 1)))` contained the free unknown `_f0`
  of the q-Gosper equation, now set to 0 (a regression test), and
  `qzeilberger` on `qbinomial(n, k, q)**2*q**(k**2)` returns `None` in
  13 s where the SymPy-based version ran for more than 300 s. The gcds
  of the q-Gosper normal form are the native multivariate `gcd` of the
  numerators: `qzeilberger(qbinomial(n, k, q), n, k, q)` takes 1.2 s
  instead of 10 s.
- `sympy_extras.concrete.zeilberger` is native (breaking: core expressions
  in and out): `zeilberger`, `wz_certificate`, `wz_prove` and
  `zeilberger_sum` on the core's `cancel`, `factor`, `lcm_list`,
  `gcd_list`, `combsimp`, `gammasimp`, `simplify`, `limit`, `product` and
  `Sum`, the native `gosper_normal`, the null space over the rational
  functions of the parameters (`SparseFractionField.nullspace`), and
  `Telescoper` is immutable. The recurrence is solved by the native
  `rsolve_linear` only (SymPy's `rsolve` is gone: where it found a
  solution, `rsolve_linear` finds the same; SymPy's `rsolve` also returned
  `0` for recurrences without hypergeometric solutions whatever the
  initial values). The integer zeros of the leading coefficient come from
  its native factorisation, so a leading coefficient with parameters no
  longer raises `NotImplementedError` from SymPy's `real_roots`: the sums
  of `(-1)**k*binomial(n, k)/(k + a)` and `binomial(n, k)*binomial(n, k +
  a)` get their closed forms. The same printed results as before on the
  70 other compared calls (16 telescopers, 7 WZ certificates and proofs,
  42 sums, with and without special values); only the isolated values of
  the parameters go through SymPy, at the call. The new
  `sympy_extras.concrete.zeilberger_sympy` (`zeilberger_sum`,
  `wz_certificate`, `wz_prove` on SymPy expressions) is used by
  `sympy_extras.integrals.series`, `docs/comparison.md` and
  `benchmarks/comparison.py`. `gosper_normal` computes its resultant with
  the typed polynomials (the Sylvester determinant only as a fallback),
  four times faster on Apéry's summand.
- `sympy_extras.solvers.recurr.rsolve_linear` no longer multiplies the
  equation by a denominator of the right-hand side which is no
  polynomial in `n` (`S(n + 1) - 4*S(n) = -(2n)!/(n! (n + 1)!)` raised
  `ValueError`).
- `sympy_extras.concrete.dirichlet` is native (breaking: core expressions
  in and out): `dirichlet_series` on the core's arithmetic functions,
  `zeta`, `re`, `max_` and `Derivative`; the same printed results as
  before on 26 compared summands with and without special values.
  `karr._with_special_values` and `karr_sympy.sum_special_values_of_core`
  take `defined` (the points outside the region of convergence get no
  case). `benchmarks/comparison.py` wraps the native `dirichlet_series`,
  `polygamma_series` and `rational_sum` for its SymPy namespace.
- `sympy_extras.concrete.rational` is native (breaking: core expressions
  in and out): `abramov_decomposition` and `rational_sum` on the core's
  `Poly`, `factor_list`, `apart`, `cancel`, `factor` and `summation`;
  `RationalDecomposition` is immutable. The same printed results as
  before on the tests, the docs and 46 compared cases. Only the isolated
  values of the parameters go through SymPy at the call.
- `sympy_extras.concrete.delta` and `summations` solve natively the
  equations of their deltas and shifts when they are linear with rational
  coefficients or free of the unknown (`solvers.linsolve.solve_one`,
  `numeric_linear_solution`, same results as SymPy's `solve`); about 200
  of the 220 calls the tests made to SymPy's `solve` from there are gone.
  `telescopic` finds its shift natively for equations rational in it
  (`polys.rationalsolve.solveset_rational`); 10 calls are left.
- **Breaking:** the package `sympy_extras.solvers` exports nothing (as
  `sympy_extras.concrete`): its functions are imported from their modules
  (`from sympy_extras.solvers.ode import dsolve_lie`,
  `from sympy_extras.solvers.recurr import rsolve_linear`, ...). Importing
  the native modules (`solvers.recurr`, `solvers.linsolve`, and
  `concrete.creative` which uses them) no longer loads SymPy; a test pins
  it.
- `sympy_extras.concrete.creative` is native: `creative_telescoping` and
  `definite_sum` take and return core expressions, on the native
  `PiSigmaField` built by the builder of `karr` (the hypergeometric part
  adjoined to the immutable field as a Π-extension), `rsolve_hypergeometric`
  for the hypergeometric solution of the recurrence (instead of SymPy's
  `rsolve_hyper`), `solve_linear_unique` for the constants (instead of
  `linsolve`), the ported `combsimp`, `powsimp`, `simplify`, `hypersimp`,
  `product`, `Poly`, `cancel`, `factor_list`, `gcd_list`, `lcm_list` and
  `together`, and imports no SymPy itself (the package
  `sympy_extras.solvers`, through which `recurr` and `linsolve` are
  imported, still loads SymPy). Two steps go through SymPy for the call,
  for summands with parameters other than the limit only: the cases of
  the isolated values of the parameters and the numerical comparison of
  the closed form with the sum at sample values of the parameters
  (`sympy_extras.concrete.karr_sympy.reliable_value_of_core`). The same
  outputs as before on the tests and the documented examples.
  `sympy_extras.concrete.pisigma_sympy` (`SymPyPiSigmaField`), the SymPy
  builder of `karr_sympy` and `karr_sympy.creative_sum_of_core` are
  removed; `karr.summation` calls `definite_sum` directly.
  `PiSigmaTower.to_expr(..., positive_denominator=True)` gives the
  denominators the sign SymPy's fraction field gives them (the
  certificates print as before), and `numer_expr`/`denom_expr` are on the
  native field.
- `sympy_extras._timeout.time_limit` and `attempt` take `native=True`:
  the limit then does not prepare SymPy's tables (nor import SymPy), and
  the polynomial errors of the core are given up on as SymPy's are (a
  SymPy one only recognised once SymPy's polynomials are imported).
- `sympy_extras.solvers.recurr` is native: `rsolve_polynomial`,
  `rsolve_hypergeometric`, `rsolve_dalembertian`, `rsolve_linear`,
  `gosper_form` and `is_recurrence_solution` take and return core
  expressions, and the module imports no SymPy (the package
  `sympy_extras.solvers` still does, through its other modules). The
  coefficients are dense polynomials in `n` over `QQ`, `QQ(a, ...)` (the
  native `SparseFractionField`) or `QQ(alpha)` (`AlgebraicField`); the
  roots outside the field are adjoined (`AlgebraicField` over `QQ`, a
  simple algebraic extension otherwise, which gives `sqrt(a)**n`); the
  similarity classes of Petkovšek's terms are decided by a polynomial
  solution of a first order recurrence instead of the Gosper–Petkovšek
  form, which needs no factorisation over the extensions; the linear
  algebra is `Domain.nullspace` and `sparse_echelon`, and on
  expressions (independence, initial values) an elimination with exact
  zero tests (minimal polynomials of the algebraic numbers). The closed
  forms of the sums come from the ported summation, without a time limit.
  Same outputs as before on the tests and the documented examples;
  `gosper_form` over `QQ(a)` gives `Poly` over `QQ[a]` (and
  `CoercionFailed` when a coefficient is not a polynomial in the
  parameters, the fraction fields not being ported to `Poly`). The
  removed `RecurrenceSolution` alias was internal. `zeilberger_sum`
  converts at the call.
- `sympy_extras.concrete.karr` is native: `karr_sum`, `karr_term`,
  `summation` and `build_pisigma_field` take and return core expressions
  (`sympy_extras.core`, `sympy_extras.functions`, the native `Sum` and
  `Product`), on the native `PiSigmaField`, the ported `summation`,
  `hypersimp`, `Poly`, `cancel`, `together` and `roots`; importing the
  module loads no SymPy, and a summand without parameters is summed
  without it. Two steps go through SymPy for the call: the cases of the
  isolated values of the parameters (`sympy_extras._special_values`)
  and creative telescoping, the last resort of `summation`.
  `build_pisigma_field` returns the native field (its `extensions` a
  tuple). The SymPy face, for the modules still written on SymPy
  (`creative`, `zeilberger`, `rational`, `dirichlet`, `qhyper`), is the
  new `sympy_extras.concrete.karr_sympy`: `karr_sum`, `karr_term` and
  `summation` converting at the boundary, `with_sum_special_values`
  (moved there from `karr`), and the builder of a field on SymPy
  expressions which creative telescoping extends.
- `sympy_extras.concrete.pisigma` (ΠΣ-fields and Karr's solver) is
  native, on the typed polynomial layer, without SymPy: one
  `SparseFractionField` over `QQ` in the parameters, `k` and the
  generators (the parameters are variables, so that the gcds are those
  over `ZZ` and an element in lowest terms is canonical); `PiSigmaField`
  is generic over its constants (`RationalConstants`, `Fraction`, or
  `ParametricConstants`, `QQ(params)`) and immutable (`add_sigma` and
  `add_pi` return the new field and the generator); its elements are
  `PiSigmaElement`s with the arithmetic operators; the dispersion
  factors with the native `zz_factor`. `karr` and `creative` use it
  through `sympy_extras.concrete.pisigma_sympy.SymPyPiSigmaField`,
  which converts SymPy expressions directly to and from the elements and
  returns the constants of the solutions as SymPy expressions
  (`build_pisigma_field` returns one). Creative telescoping is about
  twice as fast (the Paule–Schneider harmonic cubes 12.5 s → 5.4 s).
- `expand(func=True)` and `expand_func` are native
  (`functions/_expand_func.py`: gamma, polygamma, loggamma, digamma,
  trigamma, beta, binomial, harmonic, erfc, erfi, erf2, expint, zeta,
  polylog, the golden ratio and the tribonacci constant); the other
  special functions (Bessel, Airy, hyper, meijerg, lerchphi, Ynm) and the
  cases needing a `Sum` are expanded by SymPy, that node only.
- `expand(trig=True)` and `expand_trig` are native
  (`functions/elementary/_expand_trig.py`: the rules of sin, cos, tan, cot,
  sinh, cosh, tanh, coth and their reciprocals).
- `expand_mul` rebuilds `n/d` after splitting a product with a numerical
  factor in its denominator, as SymPy does (`c/(2*(a + b))` is
  `c/(2*a + 2*b)`).
- `expand(complex=True)` and `expand_complex` are native: the complex
  hint first (SymPy's sorted hints), each node as `re + I*im`, the hints of
  the call passed on to `as_real_imag` (a typed `ExpandHints`) as SymPy's
  `log`, `exp`, trigonometric and hyperbolic functions use them.
- `as_real_imag` of a sum, product or power has no shortcut for a real
  node (SymPy's); the assumptions of a power follow SymPy's
  `Pow._eval_is_extended_real` and `_eval_is_extended_negative`
  (`re(y)**2` is extended nonnegative).
- The `Expr` methods `factor`, `cancel` (without generators), `apart`,
  `collect` (evaluated), `together`, `powsimp`, `radsimp`, `trigsimp`,
  `combsimp`, `gammasimp` and `nsimplify` use the typed ports instead of
  SymPy; `powsimp`'s `combine` is typed as its three literals.
- Powers of a `Piecewise` raise each piece (SymPy's
  `Piecewise._eval_power`); `(-1)**e` is 1 or -1 when `e` is known even or
  odd (`NegativeOne._eval_power`).
- `add` re-creates its terms like SymPy's `Add.flatten` (a nested
  product `(x*y*(2*y + 1))/2` is flattened one level); `Expr.normal`
  keeps a numerical denominator outside as SymPy does
  (`(2*x*y + 1)/2`); `replace` and `expand_log` rebuild a node only when
  an argument changed.
- `count_ops` follows SymPy's loop: sums in argument order with
  `_coeff_isneg` (no printing order, 30% faster `simplify`), products
  through `fraction` like SymPy (the denominator rebuilt evaluated).
- `bottom_up` rebuilds a node when its arguments are not equal (SymPy's
  test), not when they are new objects, and `expand` only when an argument
  changed: `2*(y*log(2) + 2)` is no longer distributed by unrelated hints.
- `Poly.eval`, `roots`, `_osbornei`, `hyper_as_trig` and `gammasimp`
  substitute with evaluation, as SymPy's `xreplace` does.
- `neg` (unary minus) follows SymPy's `__neg__`: a product negates its
  coefficient and anything else becomes `Mul(-1, a)` without flattening.
- `mul` rebuilds a lone power of `E` evaluated, like SymPy's
  `Mul.flatten` (`x*exp(2*log(p))` unevaluated is `p**2*x`).
- `expand_mul`, `expand_multinomial`, `expand_power_base` and
  `expand_power_exp` no longer expand logarithms; the polynomial layer
  expands with SymPy's default hints (logarithms included).
- `cancel` and `factor` fall back to SymPy for the domains not ported
  (`ZZ_I`, `RR`, `EX`) instead of raising `CoercionFailed`.
- `rewrite` accepts the typed function classes (`Cos`, `Exp`, ...) as
  targets, mapped to their SymPy classes.
- `Pow.as_numer_denom` is SymPy's: the base is split when the exponent is
  an integer or the denominator is known positive (`sqrt(sqrt(2)/4 + 1/2)`
  is `sqrt(sqrt(2) + 2)/2`), kept whole when its sign is unknown.
- `expand` distributes products as SymPy's `Mul._eval_expand_mul` does: a
  product denominator is expanded first, so `p*t/((c + 2)*t)` style terms
  keep the expanded denominator instead of cancelling `t` term by term.
- `mul` pairs rational powers of numbers through the gcd of their bases as
  SymPy does (`sqrt(2)*6**(1/4)` is `2**(3/4)*3**(1/4)`).
- `add` and `mul` of a single argument (the identities dropped) return it
  unchanged, as SymPy's `AssocOp` does (`mul(Mul(8, x + 1))` is not
  redistributed).
- `pow` follows SymPy's `Pow._eval_power` for powers of powers (`(p**2)**x`
  is `p**(2*x)` for positive `p`, `exp(2)**x` is `exp(2*x)`, the branch
  factor for half-integer and complex exponents); `Abs(x)**n` for real `x`
  as in SymPy; `mul` rebuilds a lone power whose exponent evaluates
  differently (`x**(2*(n + 1))` is `x**(2*n + 2)`), as SymPy's `_gather`.
- `subs` follows SymPy's rules for powers and exponentials
  (`x**4` with `x**2 -> y` is `y**2`, `exp(3*x + 1)` with `exp(x) -> y` is
  `E*y**3`, `x` with `1/x -> y` is `1/y`) and applies a mapping with the
  larger keys first, as SymPy does.
- `factor_terms` and `gcd_terms` are a typed port of SymPy's `exprtools`
  (`Factors`, `Term`, `decompose_power`, `_keep_coeff`), `radical=True`
  included; `Mul.as_content_primitive` and `Pow.as_content_primitive` are
  ported from SymPy (with `clear=False` they differed); `cancel` now takes
  the common factors out first, as SymPy does.
- `mul` follows SymPy's two-argument rule: a rational times `r*(sum)`
  with `r != 1` stays an undistributed product (`-(1 - x)/5`).
- `Piecewise`, `Hyper` and `MeijerG` have `is_Function = True`, as in
  SymPy.
- `is_carmichael`, `find_carmichael_numbers_in_range` and
  `find_first_n_carmichaels` in `sympy_extras.ntheory`; `sqrt` and `Id`
  exported from `sympy_extras.functions`.
- `bspline_basis`, `bspline_basis_set`, `appellf1` and `jn_zeros`, ported
  from SymPy.
- `adjoint`, `transpose` and `piecewise_fold`, ported from SymPy.
- The polar numbers `exp_polar`, `polar_lift`, `periodic_argument`,
  `unbranched_argument`, `principal_branch`, `polarify`, `unpolarify`,
  ported from SymPy; `Expr._eval_power_node` and `DefinedFunction.unbranched`.
- `hn1`, `hn2`, `marcumq`, `chebyshevt_root`, `chebyshevu_root`,
  `jacobi_normalized`, `tribonacci`, `motzkin` and `andre`, ported from
  SymPy.
- `erf2`, `erf2inv`, `betainc`, `betainc_regularized`, `lerchphi`,
  `riemann_xi`, `stieltjes` and `multigamma`, ported from SymPy.
- The number-theoretic functions `totient`, `reduced_totient`,
  `divisor_sigma`, `udivisor_sigma`, `primenu`, `primeomega`, `mobius`,
  `primepi`, `legendre_symbol`, `jacobi_symbol`, `kronecker_symbol`, and
  the aliases `ln`, `E1`, `Eijk`, `Id`, ported from SymPy.
- `subfactorial`, `factorial2`, `sinc` and `Rem`, ported from SymPy; a
  zero float result is the exact `0`, as in SymPy.
- `euler`, `bell`, `genocchi` and `partition`, ported from SymPy.
- `fibonacci`, `lucas`, `harmonic` and `bernoulli`, ported from SymPy.
- `hyper` and `meijerg` (`Hyper`, `MeijerG`), ported from SymPy; `jacobi`
  at `x = 0`; `Expr._numeric_node` and `Expr._eval_derivative_node` hooks.
- The Mathieu functions and `SingularityFunction`, ported from SymPy; the
  core's `oo**p` for a symbolic exponent.
- `Ynm`, `Znm`, `Ynm_c`, `KroneckerDelta` and `LeviCivita`, ported from
  SymPy.
- `jacobi`, `gegenbauer`, `assoc_legendre` and `hermite_prob`, ported from
  SymPy; SymPy's powers of zero, grouping of negative bases under an
  integer exponent, and sign of an integer plus a constant in the core.
- `legendre`, `chebyshevt`, `chebyshevu`, `hermite`, `laguerre` and
  `assoc_laguerre`, ported from SymPy.
- `hankel1`, `hankel2`, `jn`, `yn` and the Airy functions, ported from
  SymPy.
- `fresnels`, `fresnelc` and the elliptic integrals `elliptic_k`,
  `elliptic_f`, `elliptic_e`, `elliptic_pi`, ported from SymPy; more of
  SymPy's arithmetic with infinities in the core.
- `besselj`, `bessely`, `besseli` and `besselk`, ported from SymPy; the
  core's `pow` distributes over an infinite product as SymPy's does.
- `Si`, `Ci`, `Shi`, `Chi`, `li` and `Li`, ported from SymPy.
- `lowergamma`, `uppergamma`, `expint` and `Ei`, ported from SymPy; the
  core decides the sign of numerical function values by evaluation and
  re-flattens products whose combined power changes its base.
- `beta` (`Beta`) and the Catalan numbers `catalan` (`Catalan`), ported
  from SymPy.
- `LambertW` with `lambertw`, and `root`, `cbrt`, `real_root`, ported from
  SymPy.
- `sympy_extras.functions.special.error_functions`: `Erf`, `Erfc`, `Erfi`,
  `Erfinv`, `Erfcinv` with the evaluating `erf`, `erfc`, `erfi`, `erfinv`,
  `erfcinv`, ported from SymPy.
- `sympy_extras.functions.special.zeta_functions`: `Zeta`, `DirichletEta`,
  `PolyLog` with the evaluating `zeta`, `dirichlet_eta`, `polylog`, ported
  from SymPy; `polygamma` gains its zeta values and Gauss' digamma
  theorem; SymPy's fallback partial derivative for the ported functions.
- `sympy_extras.functions.special.delta_functions`: `DiracDelta`,
  `Heaviside` with the evaluating `dirac_delta`, `heaviside`, ported from
  SymPy; `sign`, `Min` and `Max` have SymPy's derivatives.
- `sympy_extras.functions.special.gamma_functions`: `Gamma`, `LogGamma`,
  `PolyGamma`, `DiGamma`, `TriGamma` with the evaluating `gamma`,
  `loggamma`, `polygamma`, `digamma`, `trigamma`, ported from SymPy;
  `binomial` with a non-integer `k` and the derivatives of `factorial` and
  `binomial` use them. Ported functions of floats evaluate numerically.
- `sympy_extras.functions.combinatorial.factorials`: `Factorial`, `Binomial`,
  `RisingFactorial`, `FallingFactorial` with the evaluating `factorial`,
  `binomial`, `rf`, `ff`, ported from SymPy.
- `sympy_extras.functions.elementary.piecewise`: `Piecewise` and
  `ExprCondPair` with the evaluating `piecewise`, ported from SymPy. The
  core's relations gain SymPy's `canonical` and `reversedsign`; `lt`,
  `le`, `gt`, `ge` decide from the facts of the difference and against
  the infinities.
- `sympy_extras.functions.elementary.miscellaneous`: `Max`, `Min` with the
  evaluating `max_`, `min_`, ported from SymPy; `DefinedFunction.variadic`
  for functions with any number of arguments.
- `sympy_extras.functions.elementary.integers`: `Floor`, `Ceiling`, `Frac`
  with the evaluating `floor`, `ceiling`, `frac`, ported from SymPy; the
  core's `//` no longer calls SymPy.
- `sympy_extras.functions.elementary.complexes`: `Re`, `Im`, `Abs`,
  `Sign`, `Arg`, `Conjugate`, `Atan2` with the evaluating `re`, `im`,
  `abs_`, `sign`, `arg`, `conjugate`, `atan2`, and the complex expansion
  and conjugation of every class of the core, ported from SymPy;
  `Expr.as_real_imag`, `Expr.conjugate` and `abs()` no longer call SymPy,
  and `DefinedFunction` has hooks for its own derivative, conjugate and
  complex parts. The core gains SymPy's rules that a sum with exactly one
  infinite term is infinite and that zero times an infinite factor is
  `nan`.
- The inverse trigonometric and hyperbolic functions (`Asin`, ...,
  `Acsc`, `Asinh`, ..., `Acsch` with `asin`, ..., `acsch`), ported from
  SymPy with its tables of known values and the compositions with the
  direct functions; 3744 values identical to SymPy's except the three
  `AccumBounds` answers. `log` of purely imaginary algebraic numbers
  (`log(I*(1 + sqrt(2)))`) and the infiniteness of products with an
  infinite factor and nonzero factors (`oo*I`) follow SymPy.
- The reciprocal functions `Cot`, `Sec`, `Csc`, `Coth`, `Sech`, `Csch`
  (and `cot`, ..., `csch`), ported from SymPy; with them the half-period
  shifts of `tan` and `tanh` give `-cot(x)` and `coth(x)`, and 2904
  values of the twelve functions are SymPy's character for character.
  `thue_equation` takes `primerange` from `sympy_extras.ntheory`.
- `sympy_extras.functions.elementary.trigonometric` and `hyperbolic`:
  `Sin`, `Cos`, `Tan`, `Sinh`, `Cosh`, `Tanh` with the evaluating `sin`,
  ..., `tanh`, ported from SymPy (exact values at rational multiples of
  `pi`, the imaginary-argument exchanges, peeled multiples of `pi` and
  `I*pi`, derivatives, facts); 1452 values identical to SymPy's, the
  half-period shifts of `tan`/`tanh` (SymPy's `cot`/`coth`) excepted.
- The nodes of the core and of the functions are immutable for mypy:
  their constructor attributes are `Final` (78 of them). AGENTS.md
  requires every SymPy class to be ported as a class which never
  evaluates plus an evaluating function, with `Final` attributes.
- `sympy_extras.functions`: typed functions of the core on the new base
  class `core.function.DefinedFunction` (unevaluated node, `eval` rules,
  `apply`, `fdiff` with the chain rule in `diff`, `numeric` through
  mpmath, facts). The first family is `Exp`/`Log` with `exp`/`log`,
  ported from SymPy's `exponential.py` (special values, multiples of
  `pi*I`, log factors, `log(x, b)`, derivatives, the facts including
  Lindemann–Weierstrass); the core's conversion maps SymPy's `exp` and
  `log` to them and `E**x` builds `exp(x)` natively.
- The typed polynomial layer (domains, dense and sparse polynomials,
  gcd, factorisation, root isolation, F4, strong bases) imports nothing
  from SymPy: elements convert to and from the expressions of the typed
  core (`Domain.to_expr`, `from_expr`; `polyutils.dict_from_expr` walks
  the expression tree and multiplies out with the ring arithmetic), the
  generators are core `Symbol`s, and the conversions to SymPy objects are
  in `sympy_extras.polys.interop` (`domain_to_sympy`, `dense_to_sympy`,
  `ring_to_sympy` and their inverses). The principal subresultant
  coefficients on SymPy's dense representation, used by the CAD, moved
  from `euclidtools` to `sympy_extras.polys.subresultants`. (The layer
  still loads SymPy through the typed core, whose `Expr` forwards some
  methods to SymPy.)
- `sympy_extras.ntheory` (no SymPy): Baillie–PSW primality, `nextprime`,
  `prevprime`, `primerange`, `factorint` (trial division, Pollard–Brent
  rho, perfect powers; SymPy's `limit` convention). The modular Gröbner
  bases, the regular chains, the core and the structure theorem use it,
  and the modular images are computed by the typed F4 directly.
- F4 (`sympy_extras.polys.f4`) and the strong Gröbner bases over the
  integers (`stronggroebner`) run on the typed rings of
  `sympy_extras.polys.rings` and no longer import SymPy: F4 has its own
  sparse Gauss–Jordan elimination (`Domain.sparse_reduce`,
  `sparse_echelon`; fraction-free over the integers for `QQ`), with the
  multiples of basis elements as fixed pivots and only the remaining block
  reduced. `sympy_extras.polys.interop` converts SymPy rings at the
  boundary, so `Ideal` and the modular algorithm use the typed F4 over
  `QQ` and `GF(p)` through `direct_groebner`. The typed F4 is faster than
  the one on SymPy's elimination: Katsura-6 in `grevlex` 1.3 s instead of
  8.6 s, cyclic-6 0.8 s instead of 4.5 s.
- The first layer of the standalone polynomial code
  (`sympy_extras.polys.domains`, `densebasic`, `densearith`, `densetools`,
  the typed part of `euclidtools`, `sqfreetools`): typed coefficient
  domains (`ZZ` on `int`, `QQ` on `Fraction`, `GF(p)`, polynomial rings as
  towers for several variables, fraction fields) and the dense univariate
  algorithms generic over them (arithmetic with Karatsuba, division and
  pseudo-division, derivatives and Taylor shifts, the subresultant PRS,
  gcds, resultants by the recursive identity, discriminants, Yun's
  square-free decomposition and the `GF(p)` one). Checked against SymPy
  on random inputs and against the Sylvester determinant; see
  docs/dense.md.
- `sympy_extras.polys.galoistools` and `factortools`: factorisation over
  `GF(p)` (distinct-degree and Cantor–Zassenhaus equal-degree splitting,
  Rabin's irreducibility test) and over `ZZ` and `QQ` by Zassenhaus'
  algorithm with quadratic Hensel lifting and trial-division
  recombination, checked against SymPy's `factor_list`.
- `sympy_extras.polys.rootisolation`: real root isolation by Descartes'
  rule with bisection, Sturm sequences and counts, refinement of
  isolating intervals, over the rationals; checked against SymPy's
  `intervals` and `count_roots`.
- `sympy_extras.polys.domains.AlgebraicField`: number fields `QQ(alpha)`
  with norms, characteristic and minimal polynomials of elements, and
  `dup_ext_factor`, factorisation over them by Trager's algorithm;
  checked against SymPy's `factor_list` over algebraic fields.
- The dense layer gains native kernels for `ZZ` and `GF(p)`
  (`Domain.dense_mul`, `dense_div`), a single inversion of the leading
  coefficient in the division over a field, the cyclotomic special case
  of the factorisation and Fujiwara's root bound in the real root
  isolation (tighter isolating intervals).
- `sympy_extras.polys.polyclasses.DensePoly`: the typed polynomial class
  of the standalone layer, with factorisation and real roots dispatched
  by the domain (`Domain.factor_dense`, `isolate_real_roots_dense`).
- `sympy_extras.polys.rings`: typed sparse multivariate polynomials
  (`PolyRing`, `PolyElement`) with the monomial orders and the division
  algorithm, the representation the Gröbner basis code will move to.
- `sympy_extras.solvers.irregular`: the exponential parts of the
  solutions of a linear ODE at its irregular finite singular points, from
  the local Newton polygons (the characteristic equation of each edge
  solved in the field of the point, conjugate parts summed by a trace);
  `hyperexponential_solutions` and the factorisation now find
  `exp(1/x)`, `exp(1/(x**2 - 2))` and the like, and the Fuchsian
  restriction is gone.
- `sympy_extras.polys.rur`: the rational univariate representation of a
  zero-dimensional ideal over the rationals (Rouillier's algorithm from
  the traces on the quotient algebra, Newton's identities for the
  characteristic polynomial, Hermite's rank for the separating test),
  with the zeros as `CRootOf` or radical expressions, the real zeros, the
  multiplicities and the polynomial (shape position) form; `Ideal.rur`.
- `sympy_extras.polys.hermite` builds the product table of the quotient
  algebra by FGLM-style normal forms (border monomials in order, vector
  arithmetic) instead of one polynomial division per pair of standard
  monomials: the counts of real zeros and the representation of Katsura-5
  take 2 s instead of 20 s.
- `sympy_extras.polys.hilbert`: the numerator of the Hilbert series of a
  monomial ideal by Bigatti's algorithm (pivots on the most frequent
  variable, products over disjoint variables, the pure-powers-plus-one
  formula), used by `Ideal.hilbert_series`, `dimension`, `degree` and
  `hilbert_polynomial`; 300 generators in 12 variables in 0.07 s instead
  of 28 s with the textbook recursion.
- `zeilberger_sum` solves the recurrence with
  `sympy_extras.solvers.recurr.rsolve_linear` when SymPy's `rsolve` finds
  nothing (`sum 1/k` gives `harmonic(n)`), and Zeilberger's algorithm
  finds the recurrences with certificate 0 (shifts of the term linearly
  dependent over `K(n)`, as for `n! (-1)**k/k!`), whose sums get a
  recurrence from the boundary terms alone.
- `sympy_extras.polys.stronggroebner`: reduced strong Gröbner bases over
  the integers (Buchberger's algorithm with S- and G-polynomials and
  Euclidean reduction of the coefficients), `strong_reduce` for ideal
  membership in `ZZ[x]` and `is_strong_groebner`. SymPy's `groebner` with
  `domain=ZZ` returns the basis over the rationals, which is wrong over the
  integers (`<2x, 3y>` contains `xy` and neither `x` nor `y`).

- `sympy_extras.solvers.recurr`: linear recurrences with polynomial
  coefficients solved beyond SymPy's `rsolve`: polynomial solutions with
  the degree bound from the indicial polynomials, Petkovšek's
  hypergeometric solutions (`rsolve_hypergeometric`, with the
  Gosper–Petkovšek form `gosper_form` and a basis free of similar terms),
  the d'Alembertian solutions of Abramov and Petkovšek
  (`rsolve_dalembertian`: nested sums of hypergeometric terms, found by
  reduction of order), and `rsolve_linear` for equations written with
  `y(n + i)`, with particular solutions for hypergeometric right-hand
  sides and initial values. The harmonic numbers, the derangement numbers
  and `y(n + 1) = (n + 1) y(n) + 1`, which `rsolve` misses, are solved.

- `sympy_extras.polys.f4`: Faugère's F4 algorithm for reduced Gröbner
  bases over any field, on SymPy's sparse polynomials with SymPy's sparse
  row reduction; the normal selection strategy for the graded orders and
  a selection of the pairs with the smallest lcms for the others. It is
  the default direct computation of `Ideal` and of the images of the
  modular algorithm (`settings.groebner_algorithm`, `'f4'` or
  `'buchberger'`), two to twenty times faster than SymPy's Buchberger on
  the Katsura and cyclic systems with the same results.

- `sympy_extras.core`: an experimental core proposed as a replacement of
  `sympy.core`, written from scratch. The classes of the expression tree
  (`Add`, `Mul`, `Pow`, `Eq`, `Derivative`, ...) build exactly the node
  they are asked for (`Add(x, x)` has the arguments `(x, x)`), so that
  their types are honest and the package passes mypy in strict mode
  without `Any`, `cast` or `# type: ignore`; the evaluation of SymPy's
  constructors is done by functions of the same name in lower case
  (`add(x, x)` is `2*x`; `mul`, `pow`, `sub`, `div`, `neg`, `eq`, `lt`,
  ..., `diff`), which the operators call and which `doit` applies to a
  tree. The names, methods, evaluated forms, assumptions and printed
  forms of `sympy.core` are kept (`symbols`, `S`, `sympify`, `args`,
  `func`, `subs`, `as_coeff_Mul`, the `is_*` queries, `str`, `srepr`);
  `to_sympy` and `from_sympy` convert in both directions, and
  `SymPyFunction` wraps the functions of SymPy (`sin`, `exp`, ...) as
  unevaluated nodes evaluated, differentiated and queried through SymPy.
  Checked against SymPy on a table of known forms and on random
  expressions (evaluation, derivatives, expansions, substitutions,
  numerical values, printing). See `docs/core.md` for the design and
  the differences.
- `sympy_extras.core`, the rest of `sympy.core`: pattern matching
  (`Wild`, `WildFunction`, `match`, `replace`, `find`, `count`), the
  classes `Mod` (with `mod`), `Subs`, `Order` (with `order`),
  `UnevaluatedExpr`, `AlgebraicNumber`, `Dict`, the constants
  `EulerGamma`, `Catalan`, `GoldenRatio`, `TribonacciConstant`, the
  predicates `extended_*`, `hermitian`, `antihermitian`, `algebraic`,
  `transcendental`, `composite`, `noninteger`, `polar` with their
  implications and structural rules, the facts of a node cached on the
  node, the methods of `Expr` which were missing (`as_coeff_exponent`,
  `as_coefficient`, `as_content_primitive`, `extract_additively`,
  `extract_multiplicatively`, `equals`, `as_real_imag`, `conjugate`,
  `adjoint`, `transpose`, `as_terms`, `round`, `is_polynomial`,
  `is_rational_function`, `is_algebraic_expr`, `is_constant`,
  `is_comparable`, `as_dummy`, `dummy_eq`, `has_free`, `rcall`,
  `removeO`, `getO`, `getn`, and `series`, `nseries`, `lseries`,
  `limit`, `as_leading_term`, `leadterm`, `simplify`, `factor`,
  `cancel`, `apart`, `collect`, `together`, `powsimp`, `radsimp`,
  `trigsimp`, `nsimplify`, `integrate`, `as_poly`, `rewrite`, `refine`,
  ... which go through SymPy), `expand` with SymPy's hints and the
  `expand_*` functions, `count_ops`, `nfloat`, `N`, `comp`, `seterr`,
  `arity`, `FunctionClass`, `Rel`, `evaluate`/`distribute` context
  managers and `global_parameters`, `cacheit`/`clear_cache`,
  `Transform`, the kinds, the traversals (`postorder_traversal`,
  `bottom_up`, `use`, `iterargs`, `iterfreeargs`), the integer helpers
  (`igcd`, `ilcm`, `igcdex`, `mod_inverse`, `integer_nthroot`,
  `integer_log`, `num_digits`, `trailing`), `factor_terms`,
  `gcd_terms`, `factor_nc`, `vectorize`, `var`, `greeks`, the
  exceptions, `call_highest_priority` and the `_op_priority`
  dispatch, `random_complex_number`/`verify_numerically`. `E**x` is
  `exp(x)` and `exp` is a power of `E` in products, as in SymPy. Every
  object of SymPy now converts (`Integral`, `Piecewise`, `And`, sets,
  ... become generic wrappers whose `doit` and assumptions are
  SymPy's). Checked against SymPy on tables of known results for each
  feature.
- `sympy_extras.core`, the logic, the new assumptions, SymPy's order of
  the arguments and the performance: `And`, `Or`, `Not`, `Xor`,
  `Implies`, `Equivalent`, `ITE` as unevaluating classes (`And` and `Or`
  on `LatticeOp`) with the evaluating functions `and_`, `or_`, `not_`,
  `xor`, `implies`, `equivalent`, `ite` and the operators `&`, `|`,
  `~`, `^`, `>>`, `<<` of symbols, relations and predicates; `Q`, `ask`,
  `Predicate` (with `register`), `AppliedPredicate`,
  `AssumptionsContext`, `global_assumptions`, `assuming` and `refine`
  (conjunctive assumptions are read as facts and answered by the old
  assumptions on dummies carrying them, the rest is asked to SymPy's
  solver; `refine` is SymPy's with this `Q`). Sums and products store
  their arguments in the order of SymPy's `Basic.compare` (`compare`
  and `compare_key` are the same order here), so that `args` and
  `as_terms` agree with SymPy. `factor_terms`, `gcd_terms`,
  `as_leading_term` and `leadterm` (for polynomial, rational and
  algebraic expressions) are written here; `is_polynomial`,
  `is_rational_function` and `is_algebraic_expr` give SymPy's
  three-valued answers; `Mod(n*(n + 1), 2)` is `0`; the closure of
  `nonzero=False` is SymPy's; `TupleKind(NumberKind)` prints as SymPy's.
  The arithmetic runs within a factor of three of `sympy.core` (the
  terms of a sum are not rebuilt, powers of sums expand by the
  multinomial theorem, the compare key and the SymPy form of a node are
  cached on it; a table in `docs/core.md`). Checked against SymPy on
  tables of the connectives, of `ask` and `refine`, of the order of the
  arguments on random expressions and on every class, and of the native
  algorithms.
- `sympy_extras.core`, faster and closer to SymPy: the closure of the
  facts is read from a table of the consequences of each fact, the hash
  of a number is computed once, the hot paths import nothing, so that
  the arithmetic runs within a factor of two of `sympy.core`. A fuzz
  comparing dozens of methods on random expressions with SymPy fixed
  their conventions: `srepr` in the printing order with `-1` split,
  `sort_key` with SymPy's shape, `ordered` by the number of nodes,
  `could_extract_minus_sign` of sums, `primitive` of numbers,
  `count_ops` (`E` is `EXP`, a sum of negatives is `NEG + SUB`),
  `as_ordered_factors`, `extract_multiplicatively`, `as_numer_denom`,
  `as_content_primitive` of powers, `evalf` keeping integer exponents,
  `-1`, `I` and functions of symbols, the kind of `exp`, derivatives of
  SymPy's functions through SymPy, the `log` hint of `expand`, the
  leading term over a common denominator, `is_polynomial` of powers,
  SymPy's test on a solved match, `Rational.as_base_exp`,
  `as_independent` of zero, and the facts of numerical sums, products and
  powers decided numerically (`11 - 3*E` is positive) plus several
  structural rules (a single non-real term, roots of non-real numbers,
  even products of consecutive integers). Checked by a random method
  test against SymPy.

- `sympy_extras.polys.puiseux`: Puiseux expansions of the branches of a
  plane curve `F(x, y) = 0` above a point or above infinity by the
  Newton polygon algorithm (`puiseux_expansions`), the conjugate
  branches represented once over the number field of their
  coefficients (Duval's rational Puiseux expansions, the scaling `x - a
  = lambda*t**e` avoiding the `q`-th root of the characteristic root),
  the remaining terms by Newton's iteration once a root is simple;
  `PuiseuxExpansion.evaluate` gives the Laurent series of a polynomial
  in `x`, `y` along a branch, whose valuation is the order of the
  function at the place. The foundation of the integral bases and of
  Trager's algorithm on a general curve.
- `sympy_extras.polys.functionfields`: `FunctionField(F, x, y)`, the
  function field `K(x)[y]/(F)` of a plane curve with arithmetic,
  derivatives and traces of its elements; `integral_basis` by Trager's
  algorithm (the Round 2 algorithm: the trace radical and its idealizer
  as column echelon forms of polynomial matrices over `K[x]`, no number
  field, parameters allowed), in Hermite normal form; the local basis
  at infinity (the integral basis of the curve in `1/x`), the pole
  orders at infinity, and `normal_basis`, an integral basis normal at
  infinity in Trager's sense, by reducing the leading forms. Checked
  against the bases of the literature and against an independent
  maximality test through the Puiseux expansions.
- `sympy_extras.integrals.curves`: Trager's algorithm for the
  integration of algebraic functions on an arbitrary plane curve.
  `curve_antiderivative`, `curve_reduce` and `is_nonelementary_on_curve`
  take an integrand rational in `x` and in any number of nested
  radicals (`1/(sqrt(x + 1) + sqrt(x - 1))`, `cbrt(x**2*(x + 1))`,
  `sqrt(x + sqrt(x))`), write it in the function field of a primitive
  element of the radicals (`AlgebraicIntegrand`), and run Trager's
  three steps after his change of variable `x = N + 1/z`: the Hermite
  reduction in an integral basis normal at infinity (the congruence of
  Bronstein's lazy reduction, solved modulo the squarefree factor), the
  proof of a nonelementary integral when the reduced integrand keeps a
  pole at infinity (Trager's theorem) or when the logarithmic
  derivatives found do not account for it, the residues at the places
  above the roots of the denominator from the Puiseux expansions, the
  splitting field of the residues built exactly, and for each element
  of a rational basis of the residues a function with the prescribed
  divisor by linear algebra over the rationals on the expansions
  (multiples up to Mazur's bound for genus one, a heuristic bound
  beyond). The answer is written back in the original radicals.
  `trager_antiderivative`, `trager_reduce` and
  `is_nonelementary_algebraic` fall back to it for integrands beyond one
  radical, so `indefinite_integral` reaches them. Checked against the
  worked examples of Bronstein's tutorial and of Kauers (ISSAC 2008).
- `sympy_extras.polys.puiseux.adjoin_root` and the number fields of the
  expansions are built exactly: a primitive element by a resultant, its
  embedding by a gcd, no root evaluated numerically (SymPy's
  constructions refine complex roots by bisection, which takes minutes
  at the precisions they ask).

### Fixed

- The case `a = b` of `sin(a*x)*sin(b*x)` over `(0, pi)` was missing
  under some hash seeds: the antiderivative `x/2 - Piecewise((x, Eq(b, 0)),
  ...)/2` of `sin(b*x)**2` had no branch selected for a plain `b`, and the
  methods spent 5 s on the Mellin route of its quarter periods, beyond the
  budget of the special values. The quick antiderivative route of
  `definite_integral` takes the generic branch of an antiderivative whose
  cases are isolated values of the parameters (new
  `integrals.antiderivative.generic_antiderivative`), the driver writing
  the cases of the integral itself: 0.8 s instead of 5.
- The core matcher matches a product or sum pattern with a wild bound
  already: `match(z, p*z**q, {p: 1})` gave `None` (the bound value was
  looked for among the factors). `meijerint` no longer matches its table
  factor by factor (`_match` removed).
- The core `limit` of a step function away from its jump: `Heaviside(x -
  1)` and `E*Heaviside(x - 1)` at `0+` did not finish (gruntz expanded the
  step, whose Taylor terms at `-1` are all zero, ever further; SymPy 1.14
  hangs as well). `Heaviside(u)` is replaced by 0 or 1 by the sign of the
  limit of `u`, as `sign` is. The endpoint values of the definite last
  method no longer refuse the antiderivatives with steps.
- Core nodes survive pickling: the singletons (`true`, `false`, `oo`,
  `zoo`, `nan`, `pi`, `E`, `I`, ...) are unpickled as themselves, so that
  the last pair of an unpickled `Piecewise` is again the default one
  (`cond is true` failed on it), and the hash is computed again in the
  process which unpickles a node (it kept the hash of the pickling
  process, wrong under another `PYTHONHASHSEED`). New
  `core.basic.register_singleton`.
- The native `Ei` of a positive number times `polar_lift(-1)` evaluates
  above the cut, `Ei(-x) + I*pi` (SymPy's `Ei._eval_evalf`); it gave
  `Ei(-x)`. `exp_polar` of a real number is positive (`exp_polar(0)`
  was not known positive).
- `polys.regularchains_sympy.solutions.exact_solutions` with `real=True`
  returned the complex points whose coordinate is a root times an integer
  basis (`4*CRootOf(x**2 - x + 1, 0)` for `y**3 + 64`, taken for a real
  root).

- The native `solve` (`sympy_extras.assumptions.solve`) kept the
  solutions contradicting the flags of the unknown: `pi*k = 1` gave
  `{1/pi}` for `k = Symbol('k', integer=True)`, `p**2 = 4` gave `{-2, 2}`
  for a positive `p`. As SymPy's `solve` does with `check_assumptions`,
  the finitely many solutions (points of a system included) at which the
  native `ask` finds a flag false are now dropped, the undecided ones
  kept; the workaround in `concrete.delta._finite_solutions` is gone.
- `solvers.linsolve.solve_one` gave `[1/2]` for `2*j - 1` with an integer
  `j`, where SymPy's `solve` gives `[]`: a root for which the core's
  `check_assumptions` is `False` is dropped.
- `concrete.summations` reached SymPy's `eval_sum_residue` through
  `getattr(importlib.import_module(...))`, which AGENTS.md forbids: a
  typed wrapper with an explicit import replaces it, and the unused copy
  of the helper in `concrete.products` is removed.
- `abel_ode` returned the relation `0 = C1 - 11**(2/3)*log(x)/9`, with
  no `y` in it, for the Abel equation of the second kind `y y' = y + x`:
  the separable class of the transformed equation was not checked. The
  relation is now checked on the transformed equation, and a relation
  without the unknown is never taken for a solution (`dsolve_first_order`
  goes on to the Prelle–Singer first integral).

- `rsolve_linear` with initial values did not return on a recurrence
  whose characteristic polynomial is an irreducible cubic (the
  tribonacci numbers): the zero test of the linear system for the
  constants computed the minimal polynomial of every entry, by
  resultants on Cardano's formulas. A number which is not small at 120
  digits is now known not to be zero; the minimal polynomial only proves
  the zeros.

## 0.0.3 - 2026-10-06

### Added

- `sympy_extras.solvers.darboux`: Darboux polynomials and the
  Prelle–Singer procedure for first order equations `y' = P(x, y)/Q(x,
  y)` with polynomial `P` and `Q`, which SymPy's `dsolve` does not have.
  `darboux_polynomials` finds the irreducible Darboux polynomials of the
  field `Q d/dx + P d/dy` up to a degree bound with their cofactors (the
  highest degree part from the factors of `x P_d - y Q_d`, the charts of
  the forms when that vanishes, the rest by linear propagation and a
  parametric Gaussian elimination with case splits on the pivots, the
  parameters being the smaller of the two blocks of unknowns left, the
  cofactor's or the polynomial's, whose leaves are zero-dimensional
  systems solved for their points in the coefficient field),
  `exponential_factors`
  the factors `exp(A/B)` with `B` a product of Darboux polynomials,
  `darboux_integrating_factor` a product of their powers with `R (Q dy -
  P dx)` exact, `darboux_first_integral` a first integral (a vanishing
  combination of the cofactors, else the quadrature of the integrating
  factor, checked by `D I = 0`), and `prelle_singer` the solution,
  explicit when the first integral is rational in `y` with one root
  (`Eq(y(x), C1*x + 1/(2*x))`), else implicit, `Eq(I(x, y), C1)`, which
  `dsolve_first_order` tries after the Riccati, Abel and Chini classes
  (and `solve_ode` tries `dsolve_first_order` for a first order
  equation `dsolve` does not solve, before the symmetry method). The
  degree bound is raised one by one and the search stops at the
  first degree with a first integral; the coefficients may lie in an
  algebraic extension (`extension=`, and over the rationals the search
  is repeated over the roots of an irreducible quadratic factor of `x
  P_d - y Q_d` when nothing is found) or contain parameters; rational first integrals come out as rational
  functions, elementary ones with logarithms and arctangents (the
  quadrature is done with real symbols, and every antiderivative is
  checked by differentiation: SymPy's `integrate(a/(a*y**2 - b), y)` is
  `0` for real `a`, `b`, `y`, issue #25), Liouvillian ones through `erf`
  and the like. Lotka–Volterra's `x - log(x) + y - log(y)`, the
  pencil `(x**2 + y)/(y**2 + x)` from its reducible members, `y' = (x +
  y)/(x - y)` through `x**2 + y**2`, the linear equation `y' = x y + 1`
  through the factor `exp(-x**2/2)`.

- The remaining summation functions give the isolated values of the
  parameters their own cases, as `karr_sum`, `summation` and
  `definite_sum` do: `zeilberger_sum` (`Piecewise((n + 1, Eq(y, 1)),
  ((y**(n + 1) - 1)/(y - 1), True))` for `y**k`), `rational_sum` (`a =
  b` for `1/((k + a)*(k + b))`, where the poles merge), `qgosper_sum` and
  `qgosper_term` (`q = 1` and `q = -1` for `q**(2*k)`, where the
  q-analogue is an ordinary sum, computed by `summation` and
  `karr_term`), `karr_term` and the indefinite `karr_sum` (`k` at
  `y = 1` for `y**k`), and `dirichlet_series` (`log(2)` at `s = 1` for
  the alternating series, where `(1 - 2**(1 - s))*zeta(s)` is `0*zoo`;
  points outside the region of convergence get none). All of them, and
  `karr_sum`, `summation` and `definite_sum`, take `special_values=True`;
  the inner sums of the package pass `False`. A point where the summand
  has a pole at an integer of the range (`a = 0` for `1/((k + a)*(k +
  b))` from `k = 0`) gets no case; a summand with powers of zero at the
  point (`binomial(n, k)*y**k/(k + 1)` at `y = 0`) is summed by their
  values; each case is compared, with the numerical checks of the
  settings, with the sum computed term by term (an antidifference with
  its summand) through `reliable_value`, and left out when it disagrees.
  The recurrences and certificates (`zeilberger`, `qzeilberger`,
  `creative_telescoping`, the recurrence of `zeilberger_sum`) and
  `abramov_decomposition` are identities in the parameters and get no
  cases. `QPochhammer(0, q, k)` is `1`.

- The ODE solvers give the isolated values of the parameters their own
  cases, as the integrals and the sums do (the maintainer's decision of
  2026-09-27): `solve_ode`, `dsolve_lie`, `dsolve_first_order`,
  `riccati_ode`, `dsolve_second_order`, `dsolve_linear` and
  `dsolve_kovacic` take `special_values=True`. A general solution gets
  a case, first, at the points where it is undefined
  (`solve_ode(y' - k*y - exp(x))` has `(C1 + x)*exp(x)` at `k = 1`, the
  resonance) and at those where it is defined but no longer general: the
  zeros of the Jacobian determinant of `(y, ..., y^(m-1))` with respect
  to the constants, the Wronskian of the fundamental system for a linear
  equation, which vanishes where characteristic or indicial roots
  coincide (`C1*exp(a*x) + C2*exp(b*x)` at `a = b`, `y'' + k**2*y =
  sin(x)` at `k = 0`, the Euler equation `x**2*y'' + a*x*y' + 4*y = 0` at
  `a = -3` and `a = 5`, the Riccati solution through it losing its
  constant). The equation is solved again at each point and the solution
  checked by substitution (symbolically, else at sample points through
  `reliable_value`) and for generality; points where the order drops get
  no case. `dsolve_linear` and `dsolve_kovacic` return each function as
  a `Piecewise` of the `i`-th solutions at the points. The mechanism is
  `sympy_extras.solvers._special_values`, on a new `degenerate` hook of
  `with_special_values`; the zeros of a factor under nested powers
  (`sqrt((a - b)**2)`) are now found there too.

- The binomial differentials `x**m*(a + b*x**n)**p` whose integral is
  not elementary get a closed form: `indefinite_integral(sqrt(1 + x**3), x)`
  is `x*hyper((-1/2, 1/3), (4/3,), -x**3)`, and
  `sqrt(t**7/(1 - 5*t**2))`, which it returned unevaluated, is
  `f*2*t*sqrt(1 - 5*t**2)*hyper((1/2, 9/4), (13/4,), 5*t**2)/9` with a
  real case in `1/(5*t**2)` for `t < -1/sqrt(5)`, where `f` is real on
  the cut of `2F1`. The new `binomial_exponents` recognises the
  integrand up to a locally constant factor, powers of products and
  quotients of `x` and the binomial nested in any way (the split
  `t**(7/2)*(1 - 5*t**2)**(-1/2)` differs from `f` by a sign for
  `t < -1/sqrt(5)`), and the antiderivative, written `f*H`, holds for
  every branch; `chebyshev_elementary` is Chebyshev's criterion, which
  gates the new method (tried right after Trager's algorithm, so that it
  never preempts an elementary answer) and now settles
  `is_nonelementary_algebraic` for these integrands both ways (it
  answered `None` for `sqrt(t**7/(1 - 5*t**2))`, of genus one, and for
  `sqrt(x)/(1 + x**(1/3))`). The definite integrator's antiderivative
  route tries the method too.

- The non-elementary binomial differentials with parameters in their
  coefficients integrate too: `sqrt(t**7/(a - b*t**2))`,
  `sqrt(1 + c*x**3)` and `x**(1/3)*sqrt(a + x**2)`, left unevaluated
  (`binomial_exponents` took numbers only), get the hypergeometric form
  in `z = -b*x**n/a` and the real case in `1/z` under the condition
  `z > 1` itself (`b*t**2/a > 1`), which covers every sign of the
  parameters; each case is `f` times a function real there (checked
  for all signs on every real interval, and for complex values). The
  value `a = 0`, where `z` is undefined and the integrand a monomial,
  gets its own case through the special values; at `b = 0` the generic
  form holds. The exponents must still be rational numbers, and the
  cases with parameters are not joined at `z = 1`.
- `definite_integral(sqrt(t**7/(1 - 5*t**2)), (t, -1, -1/2))` stayed
  unevaluated: the driver split the powers into their factors and
  mapped the range to `(0, 1)` before its antiderivative route, which
  then saw no binomial differential. A proven non-elementary binomial
  differential over a finite range is now integrated first, on the
  integrand as given (after the table and the elliptic forms), with
  the range cut where `z = 1` so that one case of the antiderivative
  holds on each piece: `(x*(1 - x**2)**2)**(1/4)` over `(0, 2)` and
  `(1/2, 3)`, `(x**2 - 4)**(-5/4)` over `(3, 10)`, `sqrt(x - x**3)` over
  `(-3, -1)` and `sqrt(x**3/(5*x**2 - 1))` over `(1, 3)`, all unevaluated
  before, agree with mpmath's quadrature and Mathematica's `NIntegrate`.

- Results with parameters carry their own cases for the isolated values
  of the parameters at which the generic closed form is undefined (a
  maintainer's decision: the package left them out until now). A new
  internal module, `sympy_extras._special_values`, finds the candidates
  (zeros of denominators, of the arguments of logarithms, `s = 1` for
  `zeta(s)`) described by equations `Eq(p, v)` in one parameter each,
  keeps those at which the branch taken is undefined (or is the
  unevaluated fallback between cases, when the problem there has a
  finite value: `1/x` over `(p, q)` gets no case `p = 0`, on the
  boundary of the divergence), solves the problem again at each
  point and flattens the nested cases of several parameters, the most
  specific first. `definite_integral` returns
  `Piecewise((2, Eq(y, 1)), ((y**2 - 1)/log(y), True))` for `y**x` over
  `(0, 2)` and `Piecewise((1, Eq(k, 0)), ((exp(k) - 1)/k, True))` for
  `exp(k*x)` over `(0, 1)`; the value at the point is computed, not taken
  as a limit (`k/(k**2 + x**2)` over `(0, 1)` is `0` at `k = 0`, between
  the limits `pi/2` and `-pi/2`), with the limit as a fallback when the
  quadrature confirms it; `log(Abs(x - c))` over `(0, 1)`, unevaluated at
  `c = 0` and `c = 1` between its cases, is `-1` there. `indefinite_integral`
  writes the cases as SymPy's `integrate` does (`Piecewise((exp(k*x)/k,
  Ne(k, 0)), (x, True))`), and `karr_sum`, `summation` and
  `definite_sum` get them for the parameters of the summand
  (`Piecewise((n + 1, Eq(y, 1)), ((y*y**n - 1)/(y - 1), True))`). The new
  keyword `special_values=False` of `definite_integral` and
  `indefinite_integral` turns the cases off; the region integrals turn
  them off for their inner integrals. `verify_numerically` checks the
  branches of isolated values (`Eq(y, 1)`) by substituting the point into
  the value and the integrand, which the samples of the regions of the
  branches never reach (a wrong value of `sin(a*x)*sin(b*x)` at `a = b =
  0` passed the check before). A point where the problem itself is not
  defined gets no case (`sympy_extras._special_values.defined_problem`:
  `nan` or `zoo` in the integrand there, an unevaluated `0**e`, as
  `a**(b*z)` at `a = 0`, or a `DiracDelta` of a constant), and
  `indefinite_integral` gives no case for a point whose antiderivative is
  not found (`keep_unevaluated=False`): `a**(b*z)/z` is `Ei(b*z*log(a))`,
  not `Piecewise((Ei(b*z*log(a)), Ne(a, 0)), (Integral(0**(b*z)/z, z),
  True))`. The cases have a budget of their own (a tenth of
  `settings.timeout`, at most half of what is left of an enclosing time
  limit, read by the new `sympy_extras._timeout.remaining_time`), and
  candidates which are a parameter or linear in the parameters are not
  factored: with the whole time limit of the settings for the points,
  a hundred antiderivatives of the census were lost to unevaluated
  cases and to the time limit (`e**(1/x)/x**3` came back with an
  unevaluated case at `e = 1` after twenty seconds).

- Linear systems `Y' = A Y + b` with rational function coefficients
  (sympy-extras#24), in `sympy_extras.solvers.linear_systems`.
  `dsolve_system(A, x, b=None)` returns a `LinearSystemSolution`: a
  fundamental matrix, a particular solution and a `complete` flag
  (`True` when every solution is `fundamental * C + particular`, `None`
  when fewer than `n` independent solutions were found). The system is
  reduced to a scalar equation by a cyclic vector (`cyclic_reduction`,
  `CyclicReduction`: the unit vectors, then seeded random vectors with
  constant and polynomial entries, each checked, after Katz and
  Churchill–Kovacic), whose solutions, found by `dsolve_linear`'s solvers
  and so by Beke's factorisation for the order three and more, map back
  to the system; SymPy's `linodesolve` is tried when they fall short.
  `rational_system_solutions` is now Barkatou's direct method: pole
  bounds from the negative integer eigenvalues of the residues at the
  simple poles (through a resultant at irrational poles), no rational
  solution when the leading matrix at a higher order pole or at
  infinity is invertible, the scalar equation's bounds otherwise, and a
  linear system for the numerators; the rational particular solutions
  of inhomogeneous systems come from the augmented system.
  `hyperexponential_system_solutions` maps the hyperexponential
  solutions of the scalar equation. Inhomogeneous systems are solved by
  variation of constants with Cramer's rule and Liouville's formula.
  `dsolve_linear_system` now returns the columns of `dsolve_system`.
  Verified by substitution on textbook systems, systems whose scalar
  equation has order three and four (one through the factorisation into
  three factors), and seeded random gauge transformations `Y = P Z` of
  known systems, against the elimination method for the rational
  solutions.

- Definite summation by creative telescoping in ΠΣ-fields (sympy-extras#3),
  in the new module `sympy_extras.concrete.creative`, following Schneider
  (*Symbolic summation assists combinatorics*, 2007; *A refined difference
  field theory for symbolic summation*, 2008) and Zeilberger (1991).
  `creative_telescoping(f, k, n, order=4)` writes the shifts `f(n + i, k)`
  in one ΠΣ-field of `k` over `Q(n)` (the hypergeometric part of the
  summand as a Π-extension, harmonic numbers and nested sums as
  Σ-extensions) and finds with Karr's parameterized telescoping the
  polynomials `c_i(n)` and the certificate `g` of
  `sum_i c_i(n) f(n + i, k) = g(n, k + 1) - g(n, k)`, for the least order
  (`CreativeTelescoper`, with `recurrence()` and an exact `check()`).
  `definite_sum(f, (k, a, b))` for limits `m*n + s` sums the relation over
  the range common to the shifted sums, moves its ends past the poles of
  the certificate and adds the terms left out, solves the recurrence in
  d'Alembertian terms (a hypergeometric solution from `rsolve_hyper`
  reduces the order; first order recurrences by a product and a sum,
  evaluated by Karr's algorithm or left unevaluated), splits recurrences
  in steps of `d > 1` into residue classes of `n`, fixes the constants
  from the directly computed sum and checks the closed form against it at
  further `n` (exactly or with `reliable_value`), with a `Piecewise` for
  the small `n` where it fails. `sum binomial(n, k) H_k = 2**n (H_n -
  sum_{j<=n} 1/(j 2**j))`, `sum binomial(n, k)**2 H_k = binomial(2n, n)
  (2 H_n - H_{2n})`, `sum H_k/(n + 1 - k) = H_{n+1}**2 - H_{n+1}^(2)` and
  Paule and Schneider's `sum (1 + 3 (n - 2k) H_k) binomial(n, k)**3 =
  (-1)**n` are found. `summation` tries it when Karr's algorithm fails on
  a summand depending on a limit.

- Real solutions of zero-dimensional polynomial systems counted without
  solving them (sympy-extras#16): `count_real_solutions`,
  `count_complex_solutions`, `tarski_query`, `real_sign_counts`,
  `sign_determination`, `hermite_matrix` and `decide_zero_dimensional` in
  the new module `sympy_extras.polys.hermite`. The signature of Hermite's
  quadratic form `Tr(g*p*q)` on the quotient algebra counts the distinct
  real solutions with `g > 0` minus those with `g < 0`
  (Pedersen–Roy–Szpirglas), computed exactly over the rationals by
  Lagrange's reduction; several polynomials go through the sign
  determination of Ben-Or, Kozen and Reif (Basu–Pollack–Roy, Algorithm
  10.11). `satisfiable` (and so `tautology`) now proves that a
  conjunction whose equations have finitely many complex solutions has no
  real solution this way before building a cylindrical algebraic
  decomposition (15 times faster on `x**2 + y**2 + z**2 = 3, xy + yz + zx
  = 3, xyz = 1, x > y`). Checked against the cells of the CAD on random
  systems, against SymPy's real roots in one variable and through the
  shape lemma in three.
- Factorisation of linear ordinary differential operators with rational
  function coefficients (sympy-extras#24), in the new module
  `sympy_extras.solvers.factorization`. `LinearOperator` is now an element
  of the Ore ring `Q(x)[D]`: composition (`*`), `+`, `-`, `right_divmod`
  (Euclidean division on the right), `adjoint`, `exterior_power` and
  `symmetric_power` (the associated operators, in scalar form through a
  cyclic vector), `monic`, `primitive`, and `gcrd` and `lclm` in
  `linear_ode`. `right_factor(L, k)` finds a right factor of any order
  by Beke's algorithm (Beke 1894, in the form of Schwarz 1989 and
  Bronstein 1994): the Wronskian minors of the solutions of a factor are
  a hyperexponential solution of the `k`-th associated equation, the
  coefficients of the factor are their ratios, and when several
  hyperexponential solutions differ by rational factors the
  Grassmann–Plücker relations select the decomposable combinations;
  every candidate is checked by an exact right division.
  `left_factor` takes the adjoint, and `factor_operator` returns an
  `OperatorFactorization` (unit and monic factors, whose product is the
  operator exactly) with each factor marked irreducible when every
  search was exhaustive: `hyperexponential_search` (new) says when the
  hyperexponential solutions may be incomplete (irregular finite
  singular points, irrational exponents, conjugate singular points with
  exponents of different classes), and Kovacic's algorithm decides
  second order factors. `dsolve_linear` solves equations of order three
  and more through the factors: the solutions of the right factor, then
  for each solution of the left factor a solution of the inhomogeneous
  right factor by variation of parameters (`variation_of_parameters`,
  `solve_factored`), with the Wronskian from Abel's identity and the
  integrals left unevaluated when SymPy cannot do them. On 28 linear
  equations of order 3 to 7 (products of Airy, Bessel, Whittaker,
  Euler and first order operators, the symmetric square of Airy's
  operator, and the three homogeneous ones of order three and more of
  the Postel–Zimmermann collection) the number with a full basis went
  from 9 to 25 (the other three have an irreducible third order
  factor), every solution verified by substitution
  and the Wronskians nonzero. The hyperexponential ansatz takes one
  exponent per class modulo the integers at each singular point and
  skips the combinations the exponents at infinity exclude.

- Primary decomposition of polynomial ideals over the rationals in any
  dimension (sympy-extras#11): `primary_decomposition`,
  `associated_primes` (the embedded primes included), `is_primary` and
  `independent_set` in `sympy_extras.polys.idealdecomposition`, and the
  `Ideal` methods `primary_decomposition()`, `associated_primes()` and
  `is_primary()`. The reduction of Gianni, Trager and Zacharias: a maximal
  independent set `u`, the contraction of the extension to `Q(u)[x]` as a
  saturation by the leading coefficients of a Gröbner basis for a block
  order, and the recursion on `I + (h**m)`; the contraction is split by
  separators between its minimal primes, as in Shimoyama and Yokoyama's
  algorithm, and the components are made irredundant.
  `(x**2, x*y)` is `(x) ∩ (x**2, y)`, and the square of the prime of the
  monomial curve `(t**3, t**4, t**5)` has an embedded component at the
  origin. Checked by intersection, radicals, saturations outside each
  prime, irredundancy and the minimal primes, on textbook examples and
  random intersections of ideals which are primary by construction.
- `separating_forms` in `sympy_extras.polys.ideals`: linear forms with
  small integer coefficients by increasing height, up to a height at which
  one of them separates any given number of points (Schwartz–Zippel);
  `is_maximal` and the minimal primes search them, instead of the forms
  `x1 + t*x2 + t**2*x3 + ...`, whose coefficients grow as powers.

- `definite_integral` returns the conditions under which its value holds
  when the endpoints, or the positions of the singularities and kinks of
  the integrand, depend on parameters: a singularity (a kink) whose
  position against an endpoint the assumptions do not settle splits the
  parameters into cases (below both endpoints, above both, between them
  in either order, at most two such points), each case integrated under
  its condition added to the assumptions, checked numerically at samples
  of the parameters within the case, and the cases with the same value
  (or whose value checks under the other's condition) joined; the value
  is a `Piecewise` of the cases with the unevaluated integral as the last
  branch, for the divergent and the undecided cases, and the value alone
  when the assumptions prove the condition. `1/x` over `(a, b)` is
  `-log(a) + log(b)` for `(a > 0) & (b > 0) | (a < 0) & (b < 0)` (it
  came out unconditionally, SymPy's answer checked at positive samples
  only, where it diverges for `a < 0 < b`), `1/(x - c)` over `(0, 1)` is
  `log((c - 1)/c)` for `(c < 0) | (c > 1)`, `1/(x**2 - 1)` over `(a, b)`
  with `b > a` has its three finite cases (the endpoints both below
  `-1`, both between `-1` and `1`, or both above `1`) and `Abs(x)` over
  `(a, b)` its three by the signs of the endpoints, where both were left
  unevaluated. The singularities are placed against symbolic endpoints
  under the assumptions (SymPy's `singularities` leaves `{0}` intersected
  with `(a, b)`, which was read as no singularity), also by the
  leading-term divergence verdict, SymPy's `integrate` is not asked on
  the whole range while a position is undecided, and the cases of the
  parameters get the whole time limit. `ConditionalValue.as_piecewise`
  keeps the branches of a value which is a `Piecewise` of cases,
  `ConditionalValue.cases` and `mapped` read and transform them, and
  `antiderivative_integral` takes an antiderivative already `known`.
- Solution formulas by Hong's augmented projection (sympy-extras#9):
  when the signs of the projection factors of the free variables do not
  tell the true cells from the false ones, `quantifier_elimination` adds
  the derivatives of the factors which vanish between two such cells (at
  the first level where they lie in different cells of one stack, Brown's
  choice) to the projection factor sets, projects the levels below again
  and decomposes the space of the free variables again, each cell of the
  finer decomposition taking the truth value of the cell which contains
  it, until the signs separate them (Thom's lemma ends the rounds); with
  an equational constraint the first round adds the resultants and
  discriminants which the reduced projection left out. `Exists z: z**2 =
  x and z > y` is now `((x >= 0) & (y < 0)) | (x - y**2 > 0)`, a formula
  in polynomial sign conditions, where the answer was written with the
  root function `sqrt(x)` (and `NotImplementedError` was raised before
  that). `augmented_projection_sets` is the new projection function. The
  formulas are checked against `decide` at hundreds of random rational
  points per formula, and on random formulas with two free variables.
  The sign formula is a minimum cover of the true cells by prime
  implicants (the minimal transversals of the sets of the sign
  conditions which exclude each false cell, the cover with the fewest
  conditions searched exactly for up to 18 prime implicants) instead of
  a greedy merge: the solvability of the general quadratic is
  `Eq(c, 0) | (4*a*c - b**2 < 0) | (Ne(b, 0) & (4*a*c - b**2 <= 0))`
  where it had four terms.

- Partial cylindrical algebraic decompositions (Collins–Hong) for the
  formulas: `decide`, `quantifier_elimination`, `solution_set`,
  `sample_points`, `truth_tables` (which `ask` uses; its cells are now
  the ones on which every formula is determined, of any level),
  `cylindrical_formula`, `cylindrical_set` and
  `cylindrical_cases` lift the stacks which the truth value of the formula
  depends on and no other (trial evaluation of the formula on the signs
  known at a cell, an `exists` settled by one true cell and a `forall` by
  one false one, sectors before sections), and the signs of the projection
  factors on a stack are read off the roots and their multiplicities over
  its base, so that the algebraic field of a sample point is only built to
  lift over it. The answers are those of the full decomposition, which
  `partial=False` still builds (the reference). `Lifting` (the stacks on
  demand) and `SamplePoint.specialization` are the new pieces.

- McCallum's reduced projection for an equational constraint (ISSAC
  1999): when the formula is an equation, or has one among the terms of
  its conjunction, whose polynomial depends on the last variable, and
  that variable is quantified, the projection of the last level takes the
  coefficients and discriminants of the factors of that polynomial and
  their resultants with the other polynomials only
  (`projection_sets(..., equational=f)`,
  `mccallum_projection(..., equational=E)`); the space of the free
  variables is decomposed more coarsely, and a solution formula may be
  written with fewer factors (the same set). `Exists x, y: x**2 + y**2 =
  1 and y = a*x + b` takes 646 cells and 0.2 s instead of 3638 cells and
  3.7 s.

- The liftings of the last 64 questions are kept, keyed on the projection
  factor sets, the variables and the projection operator
  (`lifting_for`): the theory checks of `satisfiable` and the handlers of
  `refine` ask about the same polynomials many times (1146 of the 1484
  liftings of the assumptions tests repeat an earlier one). A stack is
  the same whenever it is built, so the answers do not depend on the
  questions asked before.

- The real roots of a polynomial over an algebraic sample point are found
  among the roots of its norm by Sturm counts over the field, whose signs
  are decided by interval arithmetic (Collins–Loos); SymPy's `real_roots`
  over an algebraic field evaluated the polynomial at every candidate
  with growing precision (which was most of the time of a
  decomposition). The roots of the polynomials of a stack are merged by
  their isolating intervals, compared exactly only when the intervals
  overlap, and `compare_real` refines the wider interval; two roots of
  one polynomial written differently (`CRootOf(x**2 - 2, 1)` and
  `2*CRootOf(2*x**2 - 1, 1)`) are now told equal by their indices among
  the roots of their minimal polynomial, where the refinement never
  ended. Together, on 95 random and classical formulas in two and three
  variables: 951 to 53 CPU seconds, 11 to 0 formulas over a minute, and
  on the 84 answered by both 293 to 13 seconds and 29,600 to 16,800
  cells, with the same 84 answers; on 336 further random formulas every
  answer of the previous code is reproduced (229 identical, 2 proven
  equivalent at the sample points of a decomposition and on a rational
  grid, 91 which the previous code did not give in 20 s).

- Trager's algorithm integrates the logarithmic part of a root of index
  three or more, `y = P**(1/n)` with `P` squarefree over the rationals:
  a component `A_k*y**k` without a rational antiderivative goes through
  the Hermite reduction on the curve `y**n = P`, the residues at the
  places above the roots of the denominator and at the points at
  infinity, and a function of the curve with the torsion divisor of each
  orbit of places found by linear algebra over the number field of the
  residues (the general form over a rational basis of the residues when
  the orbits are not torsion by themselves), the differential minus the
  logarithmic derivatives checked to vanish exactly on the curve and the
  answer written in real form. `1/(x*(x**4 + 1)**(1/4))` is
  `log(y - 1)/4 - log(y + 1)/4 + atan(y)/2` with `y = (x**4 + 1)**(1/4)`,
  `1/(x**3 + 1)**(1/3)` and `1/(x*(x**5 + 1)**(1/5))` integrate, and
  `x/(x**3 + 1)**(1/3)` (a double pole at infinity) or
  `1/(x*(x**3 + x + 1)**(1/3))` (a divisor not found torsion) stay
  undecided. Parameters, residue norms of degree three or more and
  number fields of degree above twelve are left to the other methods.

- `Ideal` computes the Gröbner bases for `lex` and `grlex` of the ideals of
  dimension zero by converting the `grevlex` basis with SymPy's FGLM
  instead of computing them directly: Katsura-4 for `lex` takes 0.23 s
  (54 s directly, 1 s by the modular algorithm), cyclic-5 1.9 s. The
  reduced basis is unique, so the result is the same (300 random bases
  compared with the direct computation).

- `settings.time_scale`, read from the environment variable
  `SYMPY_EXTRAS_TIME_SCALE` at import, multiplies every time limit of the
  package (`configure(time_scale=...)` too). What is found within the
  limits depends on the speed of the machine; a slower machine gets the
  answers of a faster one with a larger factor.

- `sympy_extras.polys.modulargroebner`: reduced Gröbner bases over the
  rationals by the modular algorithm of E. Arnold (J. Symbolic Comput. 35,
  2003). `modular_groebner(polys, ring, primes=None, trace=None)` returns
  what `sympy.polys.groebnertools.groebner(polys, ring)` returns (monic
  over `QQ`, primitive over `ZZ`, by decreasing leading monomials; other
  domains go to SymPy), computed from the reduced bases modulo primes
  (SymPy's `groebner` over `GF(p)`), with the unlucky primes discarded by
  comparing the leading monomials (fewer of them in the first degree where
  they differ, or smaller ones, is unlucky), the Chinese remainder theorem
  (`chinese_remainder`), Wang's rational reconstruction
  (`rational_reconstruction`) and a verification over the rationals which
  proves the result: the input reduces to zero modulo the candidate and
  the candidate is a Gröbner basis (critical pairs by the criteria of
  Gebauer and Möller, fraction-free reductions). Arnold's theorem needs a
  homogeneous ideal, and is false without: `x*(z + 210*y + 1), x*(z +
  420*y + 2)` has the image `<x>` modulo 2, 3, 5 and 7, and `[x]` passes
  both tests. The input is therefore homogenised, the basis computed and
  proven for "total degree, then the given order", and the new variable
  set to 1; this also makes the images for `lex` ten to twenty-five times
  cheaper than the images of the input (Katsura-4, cyclic-5). The primes are the primes below `2**1024`
  in decreasing order (`default_primes`; the time of an image hardly
  depends on the size of the prime with the pure Python ground types, and
  the number of images does), nothing is random, and `ModularTrace` records
  the primes used, skipped and discarded. `Ideal` computes its bases
  through the dispatcher `groebner` of the module: SymPy's direct
  computation for `settings.groebner_direct_time` seconds (0.25; twenty
  times as long for `grevlex` and `grlex`), then the modular algorithm;
  `settings.modular_groebner` turns it off. Measured (CPU seconds, one run
  each on a loaded machine, see `docs/ideals.md`): Katsura-4 `lex` 54.5 s
  by SymPy and 0.98 s modular, cyclic-5 `lex` 14.7 s and 0.64 s, Katsura-5
  and cyclic-6 `lex` more than 300 s and 137 s, 122 s; for `grevlex` the
  modular algorithm is 1.7 to 2.8 times slower below a second and on a par
  on cyclic-6 (44.3 s and 38.6 s), which is why the graded orders go to it
  last.

- Radicals and prime components of polynomial ideals of any dimension,
  over the rationals (sympy-extras#11): `Ideal.radical()` and
  `Ideal.is_radical()` no longer require dimension zero, and
  `Ideal.equidimensional_parts()`, `Ideal.minimal_primes()`,
  `Ideal.height()` are new, as `Ideal.is_prime()` in positive dimension
  (`sympy_extras.polys.idealdecomposition`, new). The radical is the
  intersection of the saturated ideals of the squarefree regular chains of
  a triangular decomposition in the sense of Kalkbrener; the primes of a
  chain come from the irreducible factors of the minimal polynomial of a
  separating linear form over the free variables, by eliminations and
  saturations over the rationals. Verified on random ideals built from
  known primes (irreducible hypersurfaces, linear varieties, kernels of
  parametrizations, maximal ideals), with powers, products and embedded
  components.
- `Ideal.saturate` by one polynomial is a single elimination
  (Rabinowitsch) instead of iterated quotients.

- `solve(..., cases=True)` (`sympy_extras.assumptions`) discusses the
  values of the parameters of a polynomial system, as Mathematica's
  `Reduce` does (sympy-extras#21): `a*x = b` gives `{b/a}` for `a != 0` and
  every `x` for `a = b = 0`, as a union of `ConditionSet(unknowns,
  condition on the parameters, solutions)`; the assumptions choose among
  the cases. Over the complex numbers the cases are those of
  `sympy_extras.solvers.parametric.parametric_cases` (new): a triangular
  decomposition in the sense of Lazard with the parameters as the smallest
  variables, the polynomials of a chain in the parameters being the
  equations of the case, its initials the inequations, and those in the
  unknowns solved one after the other (degree one and two; left as
  equations beyond). Over the reals, with inequalities too, they are the
  cells of `sympy_extras.polys.cad.cylindrical_cases` (new), a cylindrical
  decomposition with the parameters first: `x**2 <= a` is
  `Interval(-sqrt(a), sqrt(a))` for `a >= 0`. Verified by substitution of
  rational values of the parameters in random systems, and by the
  quantifier elimination over the complex numbers (comprehensive Gröbner
  systems), which proves the cases equivalent to the system.
- `sympy_extras.polys.cad.cylindrical`: cylindrical descriptions of
  semialgebraic sets, the output of Mathematica's
  `CylindricalDecomposition` and `Reduce`. `cylindrical_formula(formula,
  gens, quantifiers=())` writes the set where a (quantified) formula holds
  as a disjunction of conjunctions which bound the first variable by
  numbers, the second one by root functions of the first, and so on (the
  `k`-th real root of a projection polynomial is a continuous function on
  the cell below); `cylindrical_set` gives the points with numerical
  coordinates as a `FiniteSet` and the rest as a `ConditionSet`. A root
  function is explicit for a polynomial of degree one or two on the cell
  (the signs of its coefficients there tell the degree and the branch) and
  an `IndexedRoot(f, t, k)` otherwise, the counterpart of Mathematica's
  parametric `Root`: the `k`-th distinct real root, a number once the
  other symbols have values. Consecutive cells with one description are
  joined, and a section joins the neighbour whose description at the
  section is its own (the closed disc is one piece). Verified at the
  sample points of every cell and on a rational grid for random formulas
  (degree up to four in the last variable), and by Mathematica, which
  proves 60 random descriptions equivalent to their formulas.
- `solve` (`sympy_extras.assumptions`) answers systems of polynomial
  inequalities in several unknowns over the reals, which raised
  `NotImplementedError` (sympy-extras#21), and the real solutions of
  systems with infinitely many of them, with this description
  (`nonlinsolve` gave the families `x = sqrt(1 - y**2)` for every complex
  `y`, under a condition which was not evaluated); the parameters are
  bounded before the unknowns, within the assumptions.
- `quantifier_elimination` and `resolve` with several free variables
  write their answer with root functions when the signs of the projection
  factors do not describe the solution set, where they raised
  `NotImplementedError` (sympy-extras#9): `Exists(z, z**2 = x and z > y)`
  is `(x >= 0) & (y < sqrt(x))`.

- `RegularChain.solutions(real=False)`: the points of a chain without free
  variables, exactly (`sympy_extras.polys.regularchains.solutions`).
  Rational numbers, root objects (`CRootOf`, in radicals where
  `sympy_extras.polys.roots` writes them so) and rational functions of
  them: a polynomial of degree one in its main variable gives its
  coordinate by a division (written as a root object of its own polynomial
  when it involves root objects: `FiniteSet` sorts its elements, and the
  sort key of a sum of root objects evaluates them), and the roots of a polynomial whose
  coefficients are algebraic numbers are root objects of an iterated
  resultant of the chain, told from its other roots numerically (simple
  roots, matched with the numerical roots and these with the isolating
  intervals of the root objects; the residuals are not evaluated with
  `evalf`, which takes a minute on one which is exactly zero at a complex
  root object of degree ten). `real=True` keeps the real points, a root object being real or
  not exactly. Checked against `numerical_solutions`.

- `sympy_extras.polys.regularchains`: triangular decompositions of
  polynomial systems into regular chains. `triangularize(equations,
  *symbols, inequations=..., mode=...)` writes the zeros of a system with
  rational coefficients as the union of the quasi-components of squarefree
  regular chains (`mode='lazard'`, the default), or of their closures
  (`mode='kalkbrener'`, the generic points of every component, by
  discarding the chains of height greater than the number of equations);
  it works in any dimension, separates the components, and discusses the
  values of the parameters, which are the symbols put last. `RegularChain`
  has `polys`, `main_variables`, `free_variables`, `initials`, `height`,
  `dimension`, `degree`, the membership in the saturated ideal by
  pseudo-division (`reduce`, `contains`), `saturated_ideal`, `is_regular`,
  `regularize`, `intersect` and `numerical_solutions` (all the points of
  a chain without free variables, each once); `regular_gcd` computes
  regular gcds modulo a chain from the subresultant chain. The algorithm
  is the incremental one of Chen and Moreno Maza, written recursively from
  the top of the chain; chains contained in another one are removed and
  the chains of isolated points are made disjoint. Verified against
  Gröbner bases (saturated ideals, radical membership, covering of the
  zeros) on fixed and random systems. See `docs/regularchains.md`.
- `RegularDifferentialSystem.regular_chains()`: the regular chains whose
  saturated ideals intersect in `(A) : H^oo`, with the leaders as main
  variables. `contains` of the components of `rosenfeld_groebner` now
  decides membership by pseudo-division by these chains instead of a
  Gröbner basis (both tests are compared in the unit tests).
- `sympy_extras.polys.differential`: differential elimination, the
  differential counterpart of Gröbner bases. `DifferentialRing` holds the
  derivatives of the unknown functions and a ranking (orderly, or an
  elimination ranking by blocks of functions). `janet_basis` computes the
  Janet basis of a linear system of partial differential equations with
  coefficients rational in the variables (Janet's algorithm with the
  completion to involution, free of fractions): `reduce`, `contains`,
  `is_consistent`, the `dimension` of the solution space, the
  `parametric_derivatives`, `hilbert_function`, `hilbert_series`,
  `hilbert_polynomial` (from Janet's decomposition of the complement of
  the leaders) and `series_solution`; right-hand sides, constant
  parameters and functions of fewer variables are handled, and an
  elimination ranking gives compatibility conditions (`curl A = B`
  implies `div B = 0`). `rosenfeld_groebner` writes the radical
  differential ideal of a polynomial system of ordinary or partial
  differential equations, with inequations, as an intersection of regular
  differential systems (Boulier, Lazard, Ollivier, Petitot): Ritt's
  reduction with splittings on initials, separants and factors, the
  coherence by Delta-polynomials, and the membership by Rosenfeld's lemma
  and a Gröbner basis of the saturation. It separates general from
  singular solutions (`u'^2 = 4u`, Clairaut), finds the hidden
  constraints and the equilibria of the pendulum, eliminates its
  coordinates, and reproduces the component of the example of the 1995
  paper. `DifferentialPolynomial` exposes leaders, initials, separants and
  the reductions. The Janet bases are checked against an oracle of linear
  algebra (the jets of the solutions of the prolonged system at a point)
  on fixed and random systems. See `docs/differential.md`.
- `sympy_extras.solvers.determining_system` and `symmetry_janet_basis`:
  the determining equations of the point symmetries of a differential
  equation as a linear system of partial differential equations (no
  ansatz), and its Janet basis, whose dimension is the dimension of the
  symmetry algebra: 8 for `y'' = 0`, 7 for `y''' = 0`, 2 for Blasius, 5
  for Burgers, 4 for Korteweg–de Vries, infinite for the heat equation.
- The zero test of `sympy_extras.simplify` is put to work where the
  package decided equality by evidence. `ask` proves a relation whose two
  sides are the same elementary function on the region of the
  assumptions (`Eq(4*atan(1/5) - atan(1/239), pi/4)`, `Eq(log(x**2),
  2*log(x))` under `x > 0`, `Eq(asin(x), atan(x/sqrt(1 - x**2)))` on
  `(-1, 1)`; `>=` and `<=` hold and `>`, `<`, `Ne` fail between identical
  real sides; between constants a proved difference answers too).
  `simplify` with assumptions returns zero, the canonical form, or what
  SymPy's transformations give when told to disregard the signs
  (`powsimp`, `powdenest`, `logcombine`, `expand_log` with `force=True`,
  `simplify` on positive symbols), each kept only when it is smaller and
  *proved* equal: `atan(x) + atan(1/x)` is `pi/2` under `x > 0` and
  unchanged without it, `sqrt(x - 1)*sqrt(x + 1)` is `sqrt(x**2 - 1)`
  under `x > 1`, Machin's combination is `pi/4`. `numerically_equal`
  gives agreement at its sample points a second opinion: two expressions
  which the structure theorem proves different are different
  (`exp(-10**6*x**2 - 50)` and `0` agree to every digit at the samples).

- `sympy_extras.simplify`: a canonical form and a zero test for
  elementary expressions by the Risch–Rosenlicht structure theorem
  (`canonical_form`, `is_zero`, `equal`, `ElementaryTower`). An expression
  is written as a rational function of generators proven algebraically
  independent: a new `exp(a)` is dependent exactly when `a` is a rational
  combination of the arguments of the exponentials and of the logarithms
  of the tower plus a constant, a new `log(v)` when `v'/v` is one of their
  derivatives, both linear systems over the rationals; roots `exp(z/q)`
  are the algebraic generators, certified by Kummer theory (the exponent
  vectors of the irreducible factors of the radicands, a Smith normal
  form); `pi` is `-I*log(-1)`, the logarithm of a rational number is that
  of its primes, and the logarithm of an algebraic number is looked for
  as a multiplicative relation found by PSLQ and verified exactly
  (Machin's and Gauss's arctangent formulas, `I**I = exp(-pi/2)`,
  `sqrt(5 + 2*sqrt(6)) = sqrt(2) + sqrt(3)`). A dependent logarithm is
  resolved only where the assumptions fix its branch (`log(x**2) =
  2*log(x)` for `x > 0`, `atan(x) + atan(1/x) = pi/2` for `x > 0`, by the
  constant pinned at a sample point of a convex region on which no
  argument crosses the cut), and leaves the tower uncertified otherwise:
  `True` is always a proof, `False` is proved by the independence of the
  generators which the expression involves (`ElementaryTower.certifies`;
  Schanuel's conjecture for the constants) or witnessed by a sample
  point; once a dependent generator is in the tower, the exponentials and
  logarithms adjoined after it are uncertified too, the linear system
  which would prove them independent being read over generators which
  are not. The number field grows by primitive elements and is kept below
  degree 32 (`NotElementary` beyond, and `is_zero` then answers from a
  sample point). `is_antiderivative` asks it first, within two seconds.

- `integrate_by_ranges` and `IntegralByRanges` take `parameters`, the
  symbols not integrated over, as an alternative to listing the
  `variables`: `integrate_by_ranges(1, x**2 + y**2 < a**2, parameters=[a])`
  is `pi*a**2`, where without either list the region is the unbounded
  solid in `(a, x, y)`. The cases of the parameters are returned as one
  expression when their values agree (`pi*a**2` on `a < 0` and on `a > 0`,
  `0` at `a = 0`), for the radial routes and for the decomposition alike,
  and an inner integral whose split roots (`sqrt(-a - x)*sqrt(-a + x)`
  under `a < 0`) give a complex form is retried with the product of the
  roots.

- `sympy_extras.integrals.trager`: several square roots of polynomials
  are combined into the one root of the product of the radicands when
  the integrand is rational in it (`sqrt(x + 1)*sqrt(x + 2)/x`), and the
  answer is written back in the original roots, which the algebra in
  `y**2 = P` holds for on every side (`sqrt(P)` is minus their product
  below `-2`); a perfect-square radicand `sqrt(d**2)` integrates as
  `d*sign(d)` on each component (`sqrt(x**2 + 2*x + 1)/(x + 3)` is
  `(x - 2*log(x + 3))*sign(x + 1)`, Maxima's rtestint 10 with
  `(b**2/(4*c) + b*x + c*x**2)**(-3/2)`), and `is_antiderivative` ignores
  the jump of a `sign` factor in the derivative. The logarithmic part
  runs over the field of the parameters (`1/sqrt(x**2 + a)`, FriCAS's
  tests 301 and 19), and residues algebraic over the parameters are made
  rational by reparametrizing (`a = alpha**2` for `sqrt(a)`, `b` solved
  from `beta**2 = b - 2*sqrt(a*c)`), the algorithm rerun over
  `QQ<I>(alpha, beta, ...)` and the answer written back
  (`sqrt(a + b*x**2 + c*x**4)/(a - c*x**4)`, FriCAS's test 295; the
  inverse modulo a polynomial by the extended Euclidean algorithm, as
  `Poly.invert` reports a zero divisor over that field). A square root
  of a rational function `P/Q` is handled as `y/Q` with `y**2 = P*Q`
  (`sqrt((x + 1)/x)/x`). A root `y = P**(1/n)` of index three and more of
  a squarefree polynomial integrates by components: the derivative of
  `R*y**k` stays in the component `y**k`, so each is a Risch
  differential equation over the rational functions
  (`x**5*(x**3 + 1)**(2/3)`, `x**2*(x**3 + a)**(1/3)`); components without
  a rational solution are left undecided. Trager's algorithm now runs
  before the rewriting route, which spent the budget of FriCAS's tests
  294 and 307 on substitutions.

- `sympy_extras.integrals.rewriting`: the substitutions `t**n = M(x)` for
  a radical of a Möbius function and Chebyshev's cases of the binomial
  differential reach the indefinite driver through `power_substitutions`
  (`sqrt(x**4 + 1)/x**5` through `t**2 = 1 + x**(-4)`). A polynomial
  radicand's factors with exponents beyond the index are extracted with
  their sign, one region each: `((x - 1)**2*(x + 1))**(1/3)/x**2` is
  integrated as `(x - 1)*((x + 1)/(x - 1))**(1/3)/x**2` for `x > 1` and
  as `(1 - x)*((x + 1)/(1 - x))**(1/3)/x**2` for `x < 1`, and the two
  antiderivatives assembled into a `Piecewise` (FriCAS's integration
  test 293). `Substitution` carries the `facts` of its region. A nested
  radical has its innermost root of a linear polynomial substituted
  first and the substitutions of the result composed with it
  (`sqrt(1 - sqrt(x))/(x**2 - 1)`: `x = t**2`, then `u**2 = 1 - t`).

- `sympy_extras.integrals.radicals`: an odd power of a quadratic under
  the square root, `sqrt(Q**3)`, is `Q*sqrt(Q)` for the radical table
  (Maxima's rtestint with symbolic coefficients, under the facts which
  make `Q` positive).

- `definite_integral` reports a divergence to a signed infinity: an
  infinite one-sided limit of the antiderivative at an endpoint or at a
  singularity, of one sign over the whole range, gives `oo` or `-oo`
  (`1/x` over `(0, 1)`, `1/x**2` over `(-1, 1)`, `exp(x)` over `(0, oo)`,
  `x/(x**2 + 1)` over `(0, oo)`), a claim refused when the quadrature of
  the integral converges; infinities of both signs (`1/x` over `(-1, 1)`,
  whose principal value is `principal_value=True`'s question) and an
  oscillatory divergence (`cos(x)` over `(0, oo)`, `AccumBounds` as the
  limit) stay unevaluated. The pieces of a split range add up to the
  infinity likewise; the principal value and the finite part still need
  finite limits at the endpoints. An infinite limit is checked against
  the sign of the integrand near the point (SymPy's limit of the
  antiderivative of `x*Shi(x)` at `oo` is `-oo`, where the function
  grows like `x*exp(x)/4`).

- `sympy_extras.integrals.definite`: a sum over `(0, oo)` whose terms
  diverge separately is integrated term by term with the analytic (Riesz)
  regularisation of the Mellin method, the cancellation of the
  divergences confirmed numerically (`(exp(-b*t) - exp(-a*t))*exp(-s*t)/
  (2*sqrt(pi)*t**(3/2))` is `sqrt(a + s) - sqrt(b + s)`, Maxima's `specint`
  45); the expanded form the Mellin method reads has each term's
  exponentials combined (the three-exponential Laplace transform of
  `specint` 18 answered under `a > 0` and the like instead of `a + s > 0`).

- `sympy_extras.integrals.mellin`: Kummer's confluent hypergeometric
  function of a negative argument, `1F1(a; b; -beta*x**gamma)`, as a kernel
  of the table (`Gamma(b)*Gamma(s)*Gamma(a - s)/(Gamma(a)*Gamma(b - s))` on
  `0 < Re s < Re a`), and the Laguerre polynomials of symbolic degree
  written `exp(u)*1F1(n + 1; 1; -u)` by Kummer's transformation among the
  forms of the Mellin method: the Laplace transform of `laguerre(n, t)`
  is `(s - 1)**n/s**(n + 1)` for `s > 1` (Maxima's `specint` 42, 43); the
  Hermite polynomials of a symbolic degree `2n` or `2n + 1`, `n` an integer
  under the assumptions, likewise, by `H_{2n}(u) = (-1)**n*(2n)!/n!*1F1(-n; 1/2; u**2)`
  and its odd companion (`specint` 64, 65); a degree of unknown parity is
  left alone (the census's `specint` 105 came out wrong when `hermite(n, u)`
  was read as an even degree).

- `sympy_extras.integrals.marichev`: `log(1 - x)**m` on `(0, 1)` as the
  m-th derivative of the Beta kernel `(1 - x)**(b - 1)` in `b` (the step
  function of the range being that kernel at `b = 1`), as `log(x)**n` is
  the derivative in the exponent of `x`: `t**2*(1 - t)**2*log(t)**2*log(1 - t)**2`
  over `(0, 1)` is `(12135541 - 200*pi**2*(3739 + 30*pi**2) - 3384000*zeta(3))/16200000`
  (Maxima's `rtestint` 206; checked in Mathematica).

- `sympy_extras.integrals.radicals`: the Euler substitutions
  (`euler_substitution_antiderivative`, reached through
  `quadratic_radical_antiderivative`) for a rational function of `x` and
  `sqrt(Q)` which is not of the form `x**n*Q**(m/2)`: `sqrt(Q) = t -
  sqrt(a)*x`, `sqrt(Q) = t*(x - r)` for a real root, `sqrt(Q) = x*t +
  sqrt(c)`, and `t = sqrt(Q)` for a linear `Q`; the integrand is rational
  in `t`. `1/((x + 3)*sqrt(x**2 - 1))` over `(2, 3)`, `sqrt(x**2 +
  x)/(x**2 + 1)**2` and `sqrt(x + 1)/(x**2 + 1)` over `(0, 1)` (Maxima's
  `rtest_integrate` 854, 856; FriCAS's in1186a, in143a).

- `sympy_extras.integrals.definite`: a divergence read off the leading
  term of the integrand at an end of the range (`c*(x - end)**p` with `p
  <= -1` and a real `c` of known sign), before the methods spend the
  budget: `-sin(x)*tan(x)*csc(x - 1)` over `(0, 1)` is `oo`, as `sin(x)/x**3`
  over `(-1, 1)`; opposite infinities at the two sides stay unevaluated
  (the principal value's question). A constant over an infinite range is
  its sign's infinity. The quadrature check of a claimed divergence no
  longer trusts a moderate number where the integrand grows like a pole
  at an end (a logarithmic divergence is slow).

- `sympy_extras.integrals.definite`: the substitution `u = q(x)**(1/n)`
  for the radical `q(x)**(k/n)` of the integrand, `q` a polynomial of
  degree at most two, nonnegative and monotone on the range:
  `z*sqrt(sqrt(z**2 - 1) + 1)` over `(1, sqrt(2))` is `u*sqrt(u + 1)` over
  `(0, 1)`, `4*(1 + sqrt(2))/15` (FriCAS's in295ba); `log(1 -
  z)*atanh(sqrt(z))` over `(0, 1)` is `log(4) - 3` (in1314a); the ends of
  the range simplified through their square roots (`(sqrt(5) - 2)**(1/3)`
  is `(sqrt(5) - 1)/2`). And `u = x + c` for a radical of `(x + c)**2 - d`.

- `sympy_extras.integrals.definite`: `t = tan(k*x)` on a range inside
  `(-pi/2, pi/2)/k` and the Weierstrass substitution `t = tan(k*x/2)` on
  one inside `(-pi, pi)/k`, when the integrand becomes algebraic in `t`:
  `sqrt(tan(x))` over `(0, pi/2)` is `sqrt(2)*pi/2`, `sqrt(tan(x) +
  sec(x))*sec(x)` over `(0, pi/4)` is `2*sqrt(1 + sqrt(2)) - 2`, the
  FriCAS integrals of `sin(z)**2*sqrt(tan(z))` and the like over `(0, 1)`
  have their elementary values.

- `sympy_extras.integrals.residues`: a question about numbers (the
  half-plane of a root of `x**4 + x**2 + x + 1`, which SymPy writes as
  nested cube roots of complex numbers) is decided by evaluation, and
  such roots are taken as `ComplexRootOf`; the residue route is tried
  before the range is cut at 0, and the simplifications of a value run
  under an eighth of the time limit. `1/(x**4 + x**2 + x + 1)` over the
  real line has its exact value (Wester's problem 22).

- `sympy_extras.integrals.antiderivative`: the limits are taken of a
  real antiderivative, `log(u)` written `log(Abs(u))` for a real argument
  (`log(sin(x)/tan(1) - cos(x))` on `(0, 1)`, whose complex values made
  the infinite limit at 1 `zoo`); the finite numbers of a sum with an
  infinity are absorbed (`-Si(1)/2 + oo`); a polynomial in `x` and
  exponentials times `log(x)` is integrated by parts (`u**3*exp(-u)*log(u)`
  in two seconds, `integrate` took six). `t*exp(-sqrt(t))*log(t)` over
  `(x, oo)` (Maxima's `rtestint` 201) has its value in exponential
  integrals.

- `sympy_extras.integrals.definite`: the exponential substitution follows
  the inner exponential of `exp(-a*exp(-u))` and writes `exp(v*log(t))`
  as `t**v`: `exp(-a*exp(-u))*exp(-u*v)` over `(0, oo)` is
  `lowergamma(v, a)/a**v` (Maxima's `laplace` 42). The simplification of
  an answer of `integrate` is checked numerically too (it turned the
  `log(-exp_polar(I*pi))` of `u**3*exp(-u)*log(u)` over `(1, oo)` into
  `2*I*pi`). The exponentials free of the variable are not combined into
  those depending on it.

- `sympy_extras.integrals.tables`: an entry's range may start at a
  parameter; the Laplace transforms over `(k, oo)` of `1/sqrt(t**2 -
  k**2)` (`besselk(0, k*s)`, GR 3.364.3) and of the Bessel functions of
  `sqrt(t**2 - k**2)` (Abramowitz and Stegun 29.3.91-96, checked in
  Mathematica), reached through the Heaviside function of the census's
  entries (Maxima's `specint` 109-115, 137); the transforms of
  `erfc(k/(2*sqrt(t)))` and of `exp(a**2*t)*erfc(a*sqrt(t) + k/(2*sqrt(t)))`
  (AS 29.3.83, 29.3.89; `specint` 106-108); `exp(-I*c*x**n)` and
  `x**(a - 1)*exp(-I*c*x**n)` over `(0, oo)` (the rotated Gamma integral,
  Fresnel's integrals for `n = 2`); Ahmed's integral, `5*pi**2/96`.
  The radicals of products split by the driver (`sqrt(x - k)*sqrt(x + k)`)
  are recombined for the lookup.

- `sympy_extras.integrals.mellin`: `mellin_transform` honours the powers
  of `log(x)` of the integrand (the derivatives of the transform), which
  it dropped.

- `sympy_extras.integrals.definite`: the Euler-constant integrals. A sum
  over `(0, oo)` whose terms' Mellin transforms have poles at `s = 1`
  (`exp(-u)/u` is `gamma(s - 1)`, `1/(exp(u) - 1)` is `gamma(s)*zeta(s)`)
  is the limit at 1 of the transforms summed, with `zeta` expanded about
  1 through the Stieltjes constants (SymPy has no series for it there):
  `1/(exp(x) - 1) - exp(-x)/x` over `(0, oo)` is `EulerGamma`. An
  integrand over `(0, 1)` with `log(x)` inside a function or a
  denominator goes through `x = exp(-u)`: Wester's problem 30,
  `-log(log(1/t)) + 1/log(t) + 1/(1 - t)` over `(0, 1)`, is `2*EulerGamma`.
  `atan(u) + atan(1/u)` is `pi/2` where `u` is positive on the range
  (Maxima's `rtestint` 75). The hyperbolic functions and exponentials of
  a parameter of a rational function are made a positive symbol for the
  residues: `1/(x**4 + 2*x**2*cosh(2*a) + 1)` over `(0, oo)` is
  `pi/(4*cosh(a))` for a real `a` (HOL-Py's partialFraction 1).

- `sympy_extras.integrals.exponential`: three routes of integration in
  terms of the incomplete gamma function, the exponential integral and
  the error function. A nested power `(x**r)**p` in the exponent or the
  prefactor (`exp(a*(x**r)**p)`, `exp(a*x)/sqrt(x**3)`, `exp(a*sqrt(x**2))`)
  is integrated with the nested power kept, `-x*V*u**(-s)*uppergamma(s, u)/d`
  for `u = -a*W`, an identity everywhere (the derivative of a nested
  power of formal degree `d` is `d*W/x`), where `x**(r*p)` would hold on
  the positive axis only; a rational function times `exp(c*x)` whose
  denominator has simple roots, symbolic ones too, through
  `exp(c*r)*Ei(c*(x - r))` (`exp(c*z)/(a*z**2 + b)`); and
  `exp(alpha*x**2 + beta/x**2)` through the pair of error functions
  `erf(A*x +- B/x)`. Maxima's `rtest_integrate` 12-22, 65, 125, 126, 145,
  147 and their kin; `d**(...)` reaches them through the rewriting route,
  and `exp(c*(z**r)**(1/r))**v` no longer needs `z > 0` for an
  antiderivative (the nested power is kept in it). Compared with
  Mathematica on the classes still open: the repeated roots of the `Ei`
  route are now integrated too (the Laurent coefficients at a repeated
  symbolic root, the by-parts reduction of `exp(c*x)/(x - r)**k`), the
  error function of a composite argument `k*g'*exp(a*g**2 + c)` with
  `g` read off the exponent by factoring (`(1 - 1/x**2)*exp(-(x + 1/x)**2)`
  is `sqrt(pi)*erf(x + 1/x)/2`, FriCAS's integ 121, 132; `exp((x + 1/x)**2)/x**2`
  a pair of imaginary error functions of `x +- 1/x`, integ 122), a sum with a
  plain monomial in `power_exponential`, and the logarithmic
  substitution `x = exp(t)` for a function of `log(x)` with exponentials
  and powers of `x` (`exp(-log(x)**2 - 1)/x**3` is `sqrt(pi)*erf(log(x) + 1)/2`);
  Mathematica has no closed form for `z**n*exp(a*sqrt(z) + b*z)` with a
  symbolic `n` either.

- `sympy_extras.integrals.indefinite`: the rewriting route (the methods
  on the canonical forms and the substitutions) before the heuristics:
  `x = t**3` makes `(3*x**2 + 4*x + (3*x + 1)*log(x) + 3)*exp(x)/x**(2/3)`
  a tower the Risch algorithm settles in a tenth of a second, where the
  heuristic integrator took thirteen (FriCAS's integ 34, 36, 39).

- `sympy_extras.integrals.marichev`: the derivative in the order of a
  modified Bessel function at 0, which the limit of the logarithmic case
  of Slater's theorem leaves, is written `-besselk(0, z)` (DLMF 10.38.6):
  `exp(-k**2/(4*t))*exp(-s*t)/(2*t)` over `(0, oo)` is `besselk(0, k*sqrt(s))`
  (Maxima's `specint` 138).

- `sympy_extras.integrals.antiderivative`: a polynomial in elementary
  functions of linear arguments times `erf(k*x)` by parts, the Gaussian
  remainder integrated term by term with the exponentials combined
  (`integrate` spent a minute on `cosh(u - w)*exp(-w**2)` and gave up):
  `sinh(u - w)*erf(w)` over `(0, u)` (Maxima's `laplace` 57).

- `sympy_extras.integrals.definite`: the antiderivative route early for
  the shapes it answers at once (a polynomial in elementary functions of
  linear arguments over a finite range, a polynomial in exponentials
  times `log(x)`); the substitutions of the driver on the integral as
  given only, under half the time limit, and not on sums; the early
  radical route under a quarter of it. The nested powers of the tangent
  substitution are flattened only for a base positive on the range with
  rational exponents (`((-sin(x))**a)**(1/a)` is not `-sin(x)`: a census
  run gave 0 for its integral over `(0, pi)`). The singularities SymPy
  leaves as `Intersection({0}, Interval(sqrt(x), oo))` for a parameter
  not declared positive are placed under the assumptions.

- `sympy_extras.integrals.slater`: the logarithmic case of Slater's
  theorem takes SymPy's expansion of the G-function when the limit of the
  perturbed series does not come within an eighth of the time limit (it
  took a minute on the Parseval integrand of `expint(1, a*t)*exp(-(s - a)*t)`):
  `(log(a) + expint(1, a*t))*exp((a - s)*t)` over `(0, oo)` is
  `log(s)/(s - a)` in seconds (Maxima's `specint` 118).

- `sympy_extras.integrals.tables`: the coefficients of Kummer's Fourier
  series of `log(gamma(x))` on `(0, 1)`, `1/(4*n)` against `cos(2*pi*n*x)`
  and `(EulerGamma + log(2*pi*n))/(2*pi*n)` against `sin(2*pi*n*x)` (GR
  6.443; Wester's `log(gamma(x))*cos(6*pi*x)` is `1/12`).

- `sympy_extras.integrals.dirichlet`: trigonometric sums over powers of
  `x` on the half-lines and the real line, the Dirichlet, Frullani and
  Borwein integrals. `g(x)/x**n` with `g` a sum (or a product, written as
  a sum) of sines and cosines which vanishes to order `n` at 0 is brought
  by `n - 1` integrations by parts to `g^(n-1)(x)/x`, whose integral is
  `-sum(A_k*log(w_k)) + pi/2*sum(B_k)` (Dirichlet's integral for the sines,
  Frullani's theorem for the cosines): `sin(x)**3/x**3` is `3*pi/8`,
  `(cos(x) - cos(2*x))/x` is `log(2)`, the products of the sinc functions
  of `x/(2*k + 1)` give `pi/2` up to `k = 6` and
  `467807924713440738696537864469*pi/935615849440640907310521750000` at
  `k = 7` (Borwein and Borwein); a sum which does not vanish to the order,
  or has a constant term over `x`, is refused as divergent (Maxima answers
  `pi/2` for `exp(I*x)*sin(x)/x` over the real line, whose imaginary part
  `sin(x)**2/x` diverges).

- `sympy_extras.integrals.definite`: the powers and logarithms of a
  product are split over the factors whose sign on the piece is known
  (`(u*v)**r = u**r*v**r` and `log(u*v) = log(u) + log(v)` for `u > 0`),
  the range cut at the zeros of the factors first: the square roots of
  squares in general (`sqrt(x**2 + 2*x + 1)/x` over `(1, E)`,
  `sqrt(x - 2 + 1/x)` over `(0, 2)`, `sqrt(1 + u**2/(c - u**2))`), the
  trigonometric squares after `trigsimp` and the half-angle formulas
  (`(1 - cos(x))**(3/2)`, `sqrt(tan(x)**2 + 1)*sin(x)`, the arc length of the
  cardioid `sqrt(a**2*(1 - cos(t))**2 + a**2*sin(t)**2)`), the logarithm
  of a product (`log(sin(x)/x)` over `(0, pi/2)`, `log(x**2)/sqrt(1 - x**2)`
  over `(-1, 1)`) and `(x - x**2)**k`. The squares are rewritten before the
  other methods, the rest after those which read a radicand whole; the
  Beta substitution of the trigonometric route takes the signed form
  (`(-cos(t))**(2/3)` on `(0, pi/2)`). Eleven of the definite census's
  unevaluated integrals (Maxima's `rtest_integrate`, Wester, HOL-Py).

- `sympy_extras.integrals.definite`: a quotient of sums of sines and
  cosines is written with the sums as products and cancelled
  (`(sin(19*x) + sin(20*x))/(cos(19*x) + cos(20*x))` is `tan(39*x/2)`,
  HOL-Py's MIT 2019); the factors of a radicand or of the argument of a
  logarithm keep rational exponents, so that `log(sqrt(1 - u))` on
  `(0, 1)` is `log(1 - u)/2`, and the Beta substitution of the
  trigonometric route takes the signed form after the substitution too
  and hands the substituted integral to the other methods when the
  Mellin table has no entry (`log(1/cos(x))*cos(x)/sin(x)` over
  `(0, pi/2)` is `pi**2/24`, by the dilogarithm antiderivative of
  `-log(1 - u)/(4*u)`). SymPy's `integrate`, the last resort, runs under
  a quarter of the limit for each of its two forms (it ran under the
  whole limit twice: 75 s under a limit of 30).

- `sympy_extras.integrals.slater`: the logarithmic case of Slater's
  theorem on one side of `|z| = 1` only, with the argument undecided,
  takes SymPy's expansion of the G-function as the one formula for both
  sides (the Parseval integrand of `exp(-s*t)*erf(sqrt(t))`, whose `b`'s
  differ by an integer): the Laplace transforms of `erf(sqrt(a*t))` come
  out, `1/(s*sqrt(s + 1))` and `sqrt(a/s)/(s - a)` for `exp(a*t)*erf(sqrt(a*t))`
  under `s > a > 0` (Maxima's `specint` 48, 50, 54, 225; checked in
  Mathematica).

- `sympy_extras.integrals.definite`: the zeros of `sin(w*x + c)` and
  `cos(w*x + c)` inside a numeric range are enumerated instead of solved
  (seconds per call, and a `ConditionSet` under an assumption on another
  symbol); the assumptions handed to the solver for the zeros of a factor
  are those on its own symbols. A finite range is also mapped reflected
  onto `(0, 1)` when the direct map fails (`log(-x)/sqrt(1 - x**2)` over
  `(-1, 0)`). The numerical check uses mpmath's oscillatory quadrature
  when the plain rules disagree on a half-line (`sin(x)/x` over `(2, oo)`
  is `pi/2 - Si(2)` now: the value was found and dropped as unconfirmed).

- `sympy_extras.integrals.antiderivative`: a polynomial in `x` and in
  exponentials, sines, cosines and hyperbolic functions of linear
  arguments goes to `integrate` before anything else (the Risch port
  spent seven seconds on the gcds over `pi` in `sin(pi*t/4 + pi/4)**3`,
  on every quarter period of a trigonometric integral), and the
  antiderivatives of the definite route come from the exact verified
  methods of `sympy_extras.integrals.indefinite` (the tables, the
  trigonometric integrator, the Risch port, Trager's algorithm, SymPy's
  `integrate` last; the heuristics spend their budget on every piece of
  every mapped range) under the assumptions, in an eighth of the time
  limit: `1/(cosh(n*t)**2 + 1)` over `(0, 1)` for
  `n > 0`, where the Risch port gives up and `integrate` answers a
  `Piecewise` in `tanh`. The route runs before the slow methods
  (differentiation under the integral sign spent the whole budget
  first), a singularity of the antiderivative off the real line under
  the assumptions is dropped (`(log(3 - 2*sqrt(2)) + I*pi)/(2*n)` for
  `n > 0`, which SymPy leaves intersected with the range), and the
  arguments of a logarithm known positive on the range are not solved
  for zeros. A second pass after the slow methods tries the heuristics
  (the Risch–Norman method, the substitutions, SymPy's manual and Meijer
  routes) under a quarter of the limit: `t*exp(-sqrt(t))*log(t)` over
  `(x, oo)` has an antiderivative in exponential integrals from SymPy.

- `sympy_extras.integrals.definite`: the common factors of a sum come out
  with the constants (`log(t + 1)/(a**2*t**2 + a**2)` is
  `log(t + 1)/(t**2 + 1)` over `a**2`, and `log(a + t)/(a**2 + t**2)` over
  `(0, a)` is `pi*log(2*a**2)/(8*a)`); the periodic rule of the Laplace
  transform takes a period `2*pi/Abs(k)` under the sign of `k`, and a
  condition `Abs(arg(e)) < pi/2` is decided by the real part of `e` with
  the parameters the assumptions make real (`exp(-s*t)*Abs(sin(k*t))`
  over `(0, oo)` is `k*coth(pi*s/(2*k))/(k**2 + s**2)`).

- `sympy_extras.integrals.exponential`: `(a*x + b)**w*exp(c*x + d)` by the
  shift `t = a*x + b` to the incomplete gamma family (`exp(c*x)/(a*x + b)**k`
  is the exponential integral `E_k`, DLMF 8.19.3); the exponent of
  `x**(v - 1)*exp(a*x**n)` may be a symbol `n`, and a combined exponent
  such as `b*x**2*log(a) + c*x**2*log(h)` is read as one monomial.

- `sympy_extras.integrals.heurisch`: the special functions the structure of
  the differential tower allows, after Cherry's theorems on integration in
  finite terms with error functions and logarithmic integrals, as
  candidates of the Risch–Norman method with their derivatives registered
  in terms of the components: `Ei(k*(theta + c))` for a factor
  `alpha*theta + beta` of the denominator under `exp(theta)` or `theta =
  log(h)` (`exp(x)/(x + 1)**2`, `x/(log(x) + 1)`, the logarithmic integral
  `Ei(log(exp(2*x) + exp(x)))`), `erf(u)` and `erfi(u)` when `-theta` or
  `theta` is `u**2 + c` with `exp(c)` in the field, from the total exponent
  of a term or of one factor (`erf(x + exp(x))`, `erf(exp(x))`,
  `erf(x - 1)`, `erfi(x + 1/2)`), and `polylog(2, -exp(theta))` where
  `exp(theta) + 1` divides the denominator (`x/(exp(x) + 1)`). Two
  weaknesses of the method itself, SymPy's too, removed on the way: the
  components are substituted from the largest to the smallest whatever the
  mapping tried (substituting `exp(x)` before `exp(exp(2*x))` turned the
  inner argument into a power of the symbol and lost the outer component;
  `log(log(x))/x` and the nested exponentials were never integrated), and
  every family of rationally related exponentials is represented by one
  base of which the others are powers, exponentials of sums written as
  products of exponentials of their terms (`exp(2*x)` is `exp(x)**2` to
  the method now). `exp(x**2)*exp(x)` and `log(log(x))/x`, pinned as
  refusals, integrate.

- `sympy_extras.integrals.risch`: parameters in the constant field. The
  structure theorem's equation with a symbolic coefficient in its solution
  (the `v` of `exp(v*log(x))`) has no rational solution for a generic
  value: the monomial is new, and `x**(v - 1)*a**(b*x)` is proved
  non-elementary instead of raising `NotImplementedError`; the rational
  roots of a polynomial with parameters in its coefficients (`ZZ[v]`,
  where `real_roots` is not available) come from `roots`, in the weak
  normalizer, the residues and the recognition of logarithmic derivatives
  of radicals. Nine of the FriCAS suite's parametric cases are decided.

- `sympy_extras.integrals.exponential`: a plain linear factor is a shift too
  (`(x + 1)*exp(-x**3 - 3*x**2 - 3*x)` is `t*exp(-t**3)`), and the
  incomplete gamma form `-uppergamma(s, -a*x**n)`, real where `-a*x**n > 0`
  only, is used for an even `n` or under `x > 0`; the confluent form,
  real on the whole line, otherwise (the dispatcher's check had refused
  the gamma form at `x < -1`).

- `sympy_extras.integrals.radicals`: negative powers of `x` in the table,
  `Q**(m/2)/x**k`, by the recurrence of the moments solved for the lowest
  power and the two forms of `1/(x*sqrt(Q))` (GR 2.266, a logarithm for a
  positive constant term of `Q`, an arcsine for a negative one, checked
  in Mathematica on both sides of zero); the signs the table needs are
  decided under the assumptions, which `indefinite_integral` now hands
  to the typed methods (the exponential and radical tables) and not only
  to the check; integer powers of the radicand merge into the radical
  (`Q**(-2)/sqrt(Q)` is `Q**(-5/2)`).

- `sympy_extras.integrals.rewriting`: the substitution `s = x**2` for an
  odd integrand with a radical (`1/(r*sqrt(-a**2 + 2*h*r**2 - 2*k*r**4))`
  goes to the radical table in `s`), `u = exp(c*x)` for algebraic as well
  as rational functions of exponentials (`exp(c*z)/(a + b*exp(2*c*z))**(5/2)`,
  the multiples `c*z`, `2*c*z` of one symbolic coefficient and constants
  in the arguments accepted), and the substitutions of the first
  canonical form tried too (`sqrt(a + b*c**(d*z))`, whose exponential
  appears once `c**(d*z)` is written as one); `implied_assumptions`
  lists the facts the forms assume (`c > 0` for `c**(d*z)`), under which
  the dispatcher verifies what it finds through them. 84 of
  the 100 unevaluated integrals of the census with a recorded
  antiderivative integrate now.

- `sympy_extras.integrals.rewriting`: `a**g(x)` is written `exp(g(x)*log(a))`
  for a parameter `a` not known non-positive (a real `a` is positive
  wherever such a power is real on an interval), `exp(X)**v` and
  `(exp(X)*exp(Y)*k)**v` as one exponential for a `v` not known non-real,
  `(x**r)**p` as `x**(r*p)`, and `acosh(u)`, `asech(u)` as the logarithms of
  their real branches (`log(u + sqrt(u**2 - 1))`, SymPy's
  `log(u + sqrt(u - 1)*sqrt(u + 1))` being real for `u < -1` too, where it
  is another branch). The parameters of Maxima's test suite carry no
  assumptions, and its symbolic exponential families were stuck on these
  gates.

### Fixed

- A native `Dummy` converted to SymPy and back is the same dummy (the
  reverse mapping was not registered), so dummies survive the steps
  which go through SymPy.
- A time limit shorter than its own entry (the `1e-6` s budget of the
  direct Gröbner computation before the modular one takes over)
  raised its expiry before its owner held the `Deadline`, so that
  `attempt` let `TimeLimitExceeded` through instead of returning
  `None`; the expiry of the limit itself now happens in its body.

- `zeilberger_sum(y**k, (k, 0, n))` took the parameter `y` for the
  variable of the sum (and raised `ValueError`, `y**k` not being
  hypergeometric in `y`) instead of the upper limit `n`.

- `qgosper_sum` returned `nan` as the sum of `qbinomial(k, 2, q)*q**k`
  from `k = 0`, its antidifference being `0*zoo` there; it returns
  `None`.

- `sum_convergence` spent seconds on series with oscillating factors such
  as `Abs((-1)**n - 1)/(4*n**3)`, where the ratio and root tests find no
  limit, and still left them undecided. The termwise Fourier integral of
  `x*log(sin(x))` over `(0, pi/2)` asks exactly this, and the README
  example came back unevaluated under load. The factors `(-1)**e`,
  `cos(e)` and `sin(e)` are now bounded by `1`, and a rational majorant
  decaying like `1/n**2` proves absolute convergence at once; the
  integral takes 4 s instead of 10.

- A time limit which expired during a time-limited step inside it was
  taken by that step for its own: the internal `attempt` returned `None`
  ("this method failed"), the caller went on without the step, and every
  later `attempt` failed at once the same way, so that the computation
  finished with a fallback instead of stopping.
  `indefinite_integral(sqrt((a + b*x + c*x**2)**3)/x**2, x, facts)` under
  an enclosing limit of 0.3–1.7 s returned the integral unevaluated (a
  verdict of "cannot" for an integral it solves in twenty seconds), and
  a census integral got a case valued as an unevaluated `Integral`. The
  limits running now form a stack in `sympy_extras._timeout`; the
  exception carries the limit which expired (the outermost one when
  several have), each owner catches only its own expiry
  (`TimeLimitExceeded.expired`, with the `Deadline` which `time_limit`
  yields), `attempt(f, None)` lets an enclosing expiry through, and a
  limit entered after an enclosing one has expired (its exception
  swallowed on the way) raises that expiry at once. The signal handler
  raises nothing while a limit is being entered or left (it defers
  itself out of the machinery of the limits): an expiry raised there
  escaped the cleanup and left the timer armed, or the limit's handler
  installed, after the limit was gone. The residue
  integrator's own limit around `ask` let an enclosing expiry through
  too, instead of marking its questions slow.
- `definite_integral(x**n, (x, 0, 1))` returned `1/(n + 1)` with no
  condition, finite for `n < -1` where the integral diverges: SymPy's
  `integrate(..., meijerg=True)` gives `gamma(n + 1)/gamma(n + 2)` without
  its condition, which the numerical check confirmed at positive samples
  only, and an unconditional answer of SymPy replaced the conditional
  value of the package's methods (`n > -1`). Such an answer is now also
  checked at samples where that condition fails, and dropped when the
  quadrature disagrees there or has no finite value; the value is
  `Piecewise((1/(n + 1), n > -1), (Integral(x**n, (x, 0, 1)), True))`.
- `definite_integral(y**x, (x, 0, 2))` returned `Piecewise((2*log(y), y
  <= 0), ((y**2 - 1)/log(y), True))`, wrong for every `y <= 0` (at `y =
  -2` it gave `1.386 + 6.283*I` for the principal value `0.2009 -
  0.9106*I`, at `y = -1` `2*pi*I` for 0); the same for `c*y**x`,
  `y**(2*x)`, `x*y**x` and other ranges. Differentiation under the
  integral sign took SymPy's value of `x*y**x` over the range, whose
  branch for `y <= 0` is wrong (SymPy writes the condition `Ne(log(y), 0)`
  of the antiderivative as `(y > 0) & Ne(y, 1)` and gives the value at `y
  = 1` elsewhere), and the numerical check had sampled `y > 0` only.
  `verify_numerically` now also checks each branch of a `Piecewise` value
  at a sample of its own region, and `sample_values` moves a witness on
  the upper boundary of the region (`y = 0` for `y <= 0`) inside. The
  value is now `(y**2 - 1)/log(y)`, right for every `y` but the removable
  point `y = 1`; `x*y**x` and `y**x` over `(-1, 2)` are left unevaluated.
- Integrals whose numerical check had been passing vacuously were left
  unevaluated once `nan` and infinite quadratures stopped counting as
  agreement. `2**log(z)` and `3**log(z)` from `-I` to `I` pass through the
  branch point `0` at the midpoint, which is a node of the rules. The
  quadrature on a complex segment now subdivides it where the arguments
  of its functions cross an axis. For `sqrt(z)*acoth((1 - z)/(z + 1))`
  over `(0, 1)`, the argument of `acoth` rounds to its singularity at the
  nodes next to an end. A non-finite value at such a node now counts as
  0, while one elsewhere still makes the quadrature untrusted. Their
  values (`2/9 - I*pi/3` for the second) are now checked and returned.
- The late antiderivative tried the heuristic Risch integrator on the
  rewritten logarithmic forms of `acoth(sqrt(x))` and `acosh(x)` before
  trying it on the integrand itself or the substitution `x = t**2`. That
  took up to 15 s and failed. `log(1 - sqrt(z)) - acoth(sqrt(z))` and
  `cos(sqrt(z) - 1) + acoth(sqrt(z))` over `(0, 1)` ran out of their
  budget, and `acosh(x)` over `(-3, -2)` did too when the machine was
  loaded. The rewriting route of `indefinite` now tries the fast typed
  methods on the rewritten forms, then the substitutions. The heuristic
  on the rewritten forms is now a method of its own, `rewritten
  heurisch`, tried after `heurisch`.
- That reordering left `sinh(a + b*log(c*x**n))**2` (FriCAS integ 10)
  unevaluated under a 20 s limit: the heuristic spent both of its
  budgets on the `sinh` and failed before it came to the exponentials,
  on which it answers in a few seconds. For integrands with hyperbolic
  functions and no inverse ones, whose rewritten forms are exponentials,
  the rewriting route tries the heuristic on the forms again, after the
  substitutions.
- The integral over the real line added its two halves under the
  conjunction of their conditions, even when the halves hold on
  disjoint regions. For `exp(-(x - mu)**2/(2*sigma**2))`, one half holds
  for `mu < 0` and the other for `mu > 0`. The sum is undefined at
  `mu = 0` and cancelled to `0`, so the result was `Piecewise((0, ...),
  (Integral(...), True))`. Such a sum is now dropped when its condition
  is false at every random real sample of the parameters, and SymPy's
  `sqrt(2)*sqrt(pi)*sigma` is returned instead.
- `solve` dropped real solutions of a system when `nonlinsolve` gave
  several families of solutions whose conditions could not be decided.
  The condition set around them required the coordinates of the last
  family, a complex branch of the cube root, to be in the domain, and
  applied this to every family.
  So `solve([x**2 - y**3 > 0, Eq(x**3 + y**3 + y**2 - 1, 0)], [x, y],
  domain=S.Reals)` did not contain `(1, 0)`, and `solve(Eq(x**3, y),
  [x, y], domain=Interval(-10, 10))` did not contain `(1, 1)`. The
  condition now requires the unknowns to be in the domain, and the
  condition set checks this for the coordinates of each family.

- `definite_sum` could return an unevaluated `Sum` whose limits are
  reversed for small `n` (found by an audit against Mathematica 12.2).
  `sum k*binomial(n, k)*harmonic(k)` had `Sum(1/(2**j*j*(j + 1)), (j, 2,
  n - 1))` for every `n >= 1`, so the sum ran from 2 to 0 at `n = 1`. The
  result was right only under SymPy's reversed-limit (Karr) convention,
  not under the empty-sum reading. The lower limit of such a sum is now
  lowered, with the terms taken out added back, so that the limits are in
  order from the first `n` possible. The `n` at which a sum is still
  reversed become cases of the `Piecewise`.

- `dsolve_linear` raised `IndexError` on `y'''' + y'''/(x + 2) + q*y'' = 0`
  with `q = (3*x**3 + 16*x**2 + 28*x + 17)/((x + 1)**2*(x + 2)**2)` (found
  by an audit against Mathematica 12.2). The equation in `y'` left after
  the reduction by the solution 1 went to SymPy's `dsolve`, which reduces
  it to the equation in `y''` again, and fails there: its power series
  comes out as `Eq(g(x), O(1))`, and `dsolve(Eq(f(x).diff(x), O(1)), f(x))`
  raises `IndexError`. An equation without `y` that was reduced by a
  constant solution is no longer given to `dsolve`, whose reduction would
  repeat it. An `IndexError` from `dsolve` leaves the solutions found. The
  result is `[1, x]`, a partial basis: the equation in `y''` has no
  Liouvillian solution.

- `definite_integral` with singularities or kinks whose position depends on
  the parameters (found by an audit against Mathematica 12.2):
  `1/(x**2 - c)` over `(0, 1)` with `c > 0` had a case under
  `sqrt(c) < 0`, which never holds. Its value was on the wrong branch of
  the logarithm. The SAT solver leaves the relations of a radical
  undecided, so the conjuncts are now also asked one by one. The
  integrals of `1/((x - a)*(x - b))` with `a < b` (SymPy lists the
  singularities as a `Union`), of `1/(x**3 - c**3)` with `c > 0` (the
  point `c` written `Abs(c**3)**(1/3)*sign(c**3)`) and of `log(Abs(x - c))`
  (the antiderivative sought again on every piece, the absolute value
  kept in the cases without a kink inside, the cases out of time) over
  `(0, 1)` were left unevaluated, or had only some of their convergent
  cases. A pole of a rational integrand inside the range now makes a case
  divergent at once. A closed bound of a case is made strict where its
  value is undefined (`c*log(-c)` at `c = 0`). The quadrature of the
  numerical checks splits at the points inside the range and no longer
  returns an infinite value.

- `dsolve_linear` (and the scalar solver behind the systems) reduces the
  order of a second order equation by a known rational or
  hyperexponential solution before running Kovacic's algorithm, which
  could take the whole time limit (30 s) on an equation whose second
  solution is a quadrature, such as `x*log(x) + ...`.

- `PiSigmaField.solve` stopped the recursion over the coefficients of a
  Π-extension at the first exponent without a solution, so the
  homogeneous equation `sigma(g) = a g`, solved by `c t**j` for a single
  `j`, had no solution: `binomial(n + 1, k)` was not recognised as a
  multiple of `binomial(n, k)` and became a second extension.
- The bases of the successive steps of `PiSigmaField.solve` are
  normalised: composed without it, the constants of the parameterized
  telescoping of `binomial(n + i, k)**2 H_k` had degree 530 in `n`
  (instead of 3) and took two minutes.
- The dispersion of the universal denominator in `PiSigmaField` is
  found from the irreducible factors (a factor `q(k + h)` equal to a
  factor `p(k)` fixes `h` by the second coefficients) instead of a
  resultant in `h`, which took two thirds of the four minutes of the
  creative telescoping of `(1 + 3 (n - 2k) H_k) binomial(n, k)**3`.
- The indicial polynomial at the roots of a factor `q` of the leading
  coefficient of a linear operator missed the factor `q'(c)**k` of each
  coefficient divisible by `q**k`, so the local exponents at the roots
  of a factor which is not monic and linear (`2*x + 1`, `x**2 + 2`) were
  wrong: `(2x + 1)**2 y'' - 8 y = 0` got the exponents `(1 +- sqrt(33))/2`
  instead of `2, -1`, and `rational_solutions` and
  `hyperexponential_solutions` missed `1/(2x + 1)`.
- The exponential parts at infinity of `hyperexponential_solutions`
  followed terms of any degree at every level of the recursion (each
  level transforming the operator again): an operator of order four did
  not finish in five minutes. The terms after the first now have lower
  degrees, which is also what makes the search exhaustive.
- `dsolve_linear` compared solutions with unevaluated integrals through
  `is_constant`, which simplified the integrals and could run SymPy's
  heuristic integrator for minutes; such ratios are now taken as not
  constant.
- The cylindrical algebraic decomposition could hang in SymPy's real root
  isolation: the real roots of a polynomial with several factors over a
  rational sample point came from `Poly.real_roots`, which refines the
  isolating intervals of all the factors against one another
  (`ComplexRootOf._reals_sorted`), and with intervals SymPy had cached
  from earlier questions that refinement did not end (the whole test
  suite hung twice in a decomposition which takes seven seconds alone).
  The roots of each irreducible factor are now isolated alone and merged
  by the exact comparison of the package
  (`sympy_extras.polys.cad.samplepoints.rational_real_roots`), in the
  stacks, the sample points over algebraic fields and the indexed roots
  of the cylindrical descriptions.

- `definite_integral` checks its last step too: the verified value went
  through `tidy` (`unpolarify`, `simplify`, `refine`, the rewriting of
  logarithms and powers) and the tidied form was returned unchecked, the
  only step after the quadrature on the way to the caller (the same
  simplification once turned `log(-exp_polar(I*pi))` into `2*I*pi` in the
  method which asks SymPy's `integrate`, which guards it since). The new
  `sympy_extras.integrals.marichev.checked_tidy` compares the tidied value
  with the untidied one at values of the parameters satisfying the
  assumptions and the condition of the value, under a quarter of the time
  limit, and keeps the untidied one when they differ; it needs no new
  quadrature. The comparison is the new
  `sympy_extras.integrals.conditions.numerical_verdict`, which tells
  "different" (`False`) from "nothing could be compared" (`None`: no
  sample, a pole, a value a rounding error decides), where
  `numerically_equal` answers `False` to both; a condition the solver has
  no instance of (`re(a) > 0`) is sampled by rejection. Only a difference
  which was seen rejects the tidied form, and nothing is compared when the
  numerical checks are off. The guard did not fire in the tests of
  `sympy_extras/integrals` (160 comparisons) nor in the census of 1141
  definite integrals (232 comparisons): no value changes.

- The continuous integration had failed on every push since 18 September:
  `definite_integral(log(sin(x)/x), (x, 0, pi/2))` takes 9 s of its limit
  of 15 s on the machine where the limits were chosen, and came out
  unevaluated on GitHub's machines (reproduced with `time_scale=0.4`). Its
  test failed, and the examples of the documentation, a later step, were
  skipped, which is how four stale examples went unnoticed. The workflow
  runs with `SYMPY_EXTRAS_TIME_SCALE=3`, and runs the documentation also
  when a test has failed.

- `definite_integral(1/x**2, (x, -1, 1), finite_part=True)` came out `oo`
  instead of `-2` since the divergences to a signed infinity are reported:
  the infinite value of the pieces was returned before the finite part (or
  the principal value) which was asked for was computed. The example was
  in `docs/integrals.md`, whose doctests had not been run in full: three
  other outputs of the documentation which had changed form, all of them
  right (`pi*c**2` for the area of a disc of radius `c` without its case
  distinction, `oo` for a divergent integral, an arctangent written
  otherwise), are brought up to date.

- `Ideal.is_maximal()` raised `NotImplementedError` ("no separating linear
  form was found") when none of five fixed linear forms separated the
  zeros, as for the five points `(0, 0), (1, -1), (2, -1), (3, -1), (2,
  1)`: the forms `x1 + t*x2 + t**2*x3 + ...` are now tried for as many `t`
  as it takes, and one of the first `(n - 1) d (d - 1)/2 + 1` separates `d`
  points. It returns `False` for an ideal of positive dimension, which
  raised too.

- The numerical checks of the package no longer rest on a value which a
  rounding error decides (`sympy_extras._numeric`: `reliable_form`,
  `reliable_value`). A constant sum which is exactly zero evaluates to a
  rounding error of either sign, which under a root next to a branch cut
  chooses the branch (`log(-2*sqrt(2) - 2*sqrt(z))` with `z = -6 + (-2 +
  sqrt(2))**2 + 4*sqrt(2)`, which is 0, is `log(2*sqrt(2))` plus `I*pi` at
  15 and 50 digits, minus `I*pi` at 20, 30 and 100), and `evalf` gives the
  part of an argument which is exactly zero as a small number with all its
  digits "significant" (`5.7e-29` at twenty digits, `-1.3e-48` at forty).
  The two guards written for sympy-extras#65 (`definite_integral`) and for
  the zero test (`is_zero`) are one routine now: the sums under a function
  or a fractional power which have no significant digit are written 0 when
  SymPy proves them zero (their real or imaginary part alone too), the
  parts of the arguments of logarithms, fractional powers and inverse
  functions which differ at a higher precision are rounding errors, and a
  value which still depends on one is refused. It is applied before the
  values of the symbols are substituted as well (SymPy's `log` of such a
  sum gives `zoo` or raises `RecursionError` in some runs). Found wrong in
  some runs before it: `numerically_equal` (the right closed form found
  different, `RecursionError` in other runs), the refutation of an
  identity at a point in `ask`, the solutions of `solve` "clearly not
  real" or "not satisfying their equation", the real roots of
  `solve_transcendental` (a root refused: SymPy's `is_real` of such a
  number is random too). Also through it: the antiderivative check of
  `integrate`, the forms of the roots in `integrate_by_ranges`, the signs
  decided numerically in `sympy_extras.integrals.residues`, `elliptic`,
  `slater` and `antiderivative`, the direction of an infinite limit.

- `is_zero` (`sympy_extras.simplify`) answered `False` for differences which
  are identically zero, `atan(x)**(1/3)` minus its form in logarithms and
  `sin(x)**(1/6)` minus its form in exponentials: the witness was the real
  sample point `x = -3`, where the base of the second power is a negative
  number whose imaginary part, exactly zero, evaluates to `1e-41` of either
  sign, so that the root took the other branch. A sample point at which
  the argument of a logarithm, of a fractional power or of an inverse
  function has a real or imaginary part without a significant digit is no
  witness any more (the answer is `None` there). Found by evaluating the
  verdicts of a random batch with Mathematica.

- `solve` returned `{(x, -sqrt(2)), (x, sqrt(2))}` for `[x**5 - x - 1 - y,
  y**2 - 2]`, the unknown `x` left free: `nonlinsolve` leaves an unknown
  it cannot solve for as it is (and gives `x`, `y` as polynomials in a free
  `z` for three equations in three unknowns with finitely many solutions,
  after 37 s for one with twelve). A polynomial system with rational
  coefficients, no parameter and finitely many solutions is now solved by
  its triangular decomposition first (`RegularChain.solutions`, the real
  points only over the reals: all the solutions by construction, in 2 s
  for the system above). For the other systems the points of `nonlinsolve`
  are substituted back in the equations, numerically at random values of
  the symbols they keep; a system they do not satisfy goes through the
  chains too (a chain with free variables whose polynomials have degree
  one in their main variables gives the family it parametrizes), and is
  returned as a `ConditionSet` of its equations when that is not possible,
  instead of the wrong points.

- `definite_integral` returned, in some runs, the right real value minus
  `2*I*pi` (or `zoo`) for the integral of `x/2 + sqrt(x**2 + 8)/2 -
  sqrt(x**2 + 4*x + 2)` from a root of the second radicand, and
  `integrate_by_ranges` a complex area for the region whose cell it is
  (issue #65). Two causes. The answer of SymPy's `integrate`, exactly
  wrong by `-2*I*pi`, passed the numerical check: it has the logarithm of
  `-2*sqrt(2) - 2*sqrt(z)` with `z` a sum which is exactly zero, to which
  `evalf` gives a rounding error whose sign chooses the branch (SymPy
  decides the sign of such a number at random, issue #25); `verify_numerically`
  now writes the sums which are exactly zero as 0 before it evaluates a
  value, and checks nothing when such a sum is not decided. And the
  integral reached SymPy only when the time limits had stopped the
  methods of the package, which is what changed from run to run:
  `quadratic_radical_antiderivative` allowed one radicand for a whole
  sum, and now takes a sum of radicals of different quadratics, so that
  the integral is answered at once by the table.

- `ask` answers `None`, not `True`, for `exp(-I*x) > 0` on `-1 < x < 1`:
  the sign analysis took the value `1` at `0` and the absence of zeros
  for a sign, where a complex-valued expression has none. Found by the
  fuzz of `sympy_extras.simplify`, which had split `log(u*exp(-I*x))` as
  if the exponential were positive.

- `sympy_extras._timeout.attempt` contains `ArithmeticError` too:
  `PrecisionExhausted` (SymPy's `evalf` unable to decide a sign inside
  `integrate`), a division by zero or an overflow come back as "not
  computed" instead of crashing `definite_integral` and
  `integrate_by_ranges` (#55).

- `integrate_by_ranges` no longer takes a non-radial integrand as
  radial: the comparison with the value on the axis is sampled in
  every orthant and under finite rotations, not on the positive orthant
  alone, where `Heaviside(x)`, `sign(x)` and `Abs(x)` agree with their
  radial candidates (`Heaviside(x)` over the unit disc is `pi/2`, not
  `pi`; #56). The axisymmetric route checks the invariance under finite
  rotations as well as under the infinitesimal ones, which a function
  constant off a set of measure zero passes (`Piecewise((1, x*y > 0), (0,
  True))` over the disc and `Heaviside(x)` on the circle with the
  Hausdorff measure gave `0` and `2*pi`; #57).

- `integrate_by_ranges` with `measure='hausdorff'` takes the dimension
  of the variety from the decomposition instead of from the number of
  equations, so that dependent equations (`Eq(y, x) & Eq(x - y, 0)`,
  `Eq(y, x) & Eq(y**2, x**2)`, the cylinder through the equator of the
  sphere) measure their curve instead of `0` (#58).

- `integrate_by_ranges` restricts the outer variables to where a solved
  bound is real: `y < log(x)` on `-1 < x < 1` means `0 < x < 1` (the
  integral of `exp(y)` there is `1/2`, not `0`; areas came out complex
  and the cube root `y > x**(1/3)` was taken real for `x < 0`; #59).

- `ask` answers `None`, not `True`, for `log(x) < 0` and `x**(1/3) < 2`
  under `-1 < x < 1`: the polynomial bounds of a logarithm or a root
  hold where it is real, and the facts must imply that (`u > 0` for
  `log(u)`, `u >= 0` for a root) before the abstraction is trusted, and
  the sign analysis declines a difference with a logarithm or a root not
  real on the whole domain (#61).

- The cylindrical route of `integrate_by_ranges` starts the radial range
  at the origin: `solve` gave `(-sqrt(a), sqrt(a))` for `rho**2 < a` with
  `a` symbolic, and the integrand, odd in `rho`, integrated to `0` (the
  volume under the paraboloid `z < a` with `a > 0` is `pi*a**2/2`; #60).

- The `Piecewise` of `integrate_by_ranges` labels the cells of the
  parameters by their roots in explicit form (`a > sqrt(2)`) instead of
  by the signs of the projection polynomials, which do not tell `a <
  -sqrt(2)` from `a > sqrt(2)`: the value of one cell was attached to the
  other whenever a projection polynomial with two real roots did not
  factor over the rationals (`(x**2 < 2) & (x < a)` at `a = 2` gave `0`
  instead of `2*sqrt(2)`, the disc of radius `sqrt(2)` cut by `x < a`
  likewise; #62).

- `integrate_by_ranges` drops the equations and non-equations in the
  integration variables under the Lebesgue measure (`Ne(p, 0)` is true,
  `Eq(p, 0)` false, up to a set of measure zero): a box cut along the
  curve of a `Ne` was decomposed along it, with algebraic bounds whose
  logarithms ran out of memory (#63). `real_logarithms` decides the sign
  of a numeric argument numerically instead of expanding it.

- `IntegralByRanges` with `measure='hausdorff'` or a `dimension` can be
  rebuilt from its arguments, so `subs`, `xreplace`, `copy.deepcopy` and
  pickling work; the integration variables are bound (`free_symbols`
  excludes them, `subs` leaves them alone; #64).

- `sympy_extras._timeout`: the special-function patterns of SymPy's
  `manualintegrate` are built before a limit is set, like the Meijer G
  table. SymPy builds them on first use, the wildcards first and the
  patterns after, and a limit hit in between left the wildcards in
  place and the patterns empty: the next call doubled the wildcards
  and every special-function rule (`exp(exp(x))` to `Ei(exp(x))`)
  failed for the rest of the process. The recursion marks which a limit
  or an error leaves in the cache of `integral_steps` (the integrand
  it was working on, `DontKnowRule` for it ever after) are cleared
  before a limit is set as well.

- `is_antiderivative` trusted a candidate in complex form (`I`, a polar
  number the real-form rewriting could not clear) on points of one sign
  only: with a non-integer sample of the `r` of `exp(a*x**r)` every
  `x < 0` is complex and skipped, and SymPy's polar incomplete gamma, an
  antiderivative for `x > 0` and the negative of one for `x < 0`, passed.
  Such a candidate now needs tested points of both signs, the parameters
  sampled again to find them. The numerical check also took a point where
  the integrand is `-2e-48 + 4e-49*I` as real (the imaginary part below
  an absolute threshold while a fifth of the value) and its vanishing
  difference as a test: the imaginary part must be negligible next to
  the value as well, and a negligible value is no test.

- `numerically_equal` compared the values as Python complex numbers, and
  `abs()` overflowed for the 1e300 of `exp(A*x**r)` at a sampled point,
  refusing a right antiderivative; it compares SymPy floats now. The
  check also draws points inside the real domain of the integrand
  (the radicands and logarithm arguments positive, from the sampler of
  the assumptions) when its fixed points miss a parameter-dependent
  interval, and samples the parameters again when a sample leaves it
  undecided.

- `sympy_extras.integrals.rewriting` combined the nested power
  `(z**r)**p` into `z**(r*p)` without `z > 0`, and wrote `acosh(u)` as
  `log(u + sqrt(u**2 - 1))`, which is `acosh(-u)` for `u < -1` where the
  function is `acosh(-u) + I*pi`: the census's `exp(c*(z**r)**(1/r))**v`
  came out as `exp(c*v*z)/(c*v)`, wrong at `z < 0` for an even `r`, and
  `exp(acosh(z))` with the wrong sign of the radical for `z < -1`. The
  identity needs the assumption now, and the inverse hyperbolic functions
  take SymPy's logarithm forms, the functions' branches on the whole real
  line. (Both passed the check because the integrand is complex at the
  negative test points for a generic exponent, and SymPy's assumptions
  cannot tell `exp(acosh(z))` real for `z < -1`.)

- `taylor_coefficient` took the formula of `fps` as valid from the first
  term: for `(exp(4*x) - exp(-4*x))**2` the formula holds from `k = 3`
  and gives 2 at `k = 0`, where the series has no term, so the series
  route made the integral over `(0, 1)` `sinh(8)/4`, 2 too much
  (`sinh(8)/4 - 2` is right), and for `1/(x**2 + x + 1)` `fps` answers
  `(-1)**k`, the series of `1/(x + 1)`, which made the integral of
  `(1 - x)/(x**2 + x + 1)` over `(0, 1)` `log(4) - 1` (HOL-Py's
  `euler_log_sin06`; `sqrt(3)*pi/6 - log(3)/2` is right). The formula is
  now checked term by term against the Taylor expansion from `series`
  and refused on a mismatch; `fps` failing inside its hypergeometric
  algorithm (a `KeyError` on `2**(-x)`) is a refusal too, not a crash.

- `definite_integral(exp(-a*t)*exp(-s*t), (t, 0, oo), a + s > 0)` went
  through Parseval's formula with two exponential kernels and answered
  under the condition `a > 0` (`(a > 0) & (s/a > 0)` in a `Piecewise`)
  instead of `1/(a + s)`: the exponential factors of a product are
  combined first, and the expanded form the Mellin method reads keeps
  them whole (`expand` wrote `exp(-(a + s)*t)` as the two factors again:
  `(1 - 2*a*t)*exp(-(a + s)*t)/sqrt(t)` is `s/(a + s)**(3/2)` under
  `a + s > 0`, with no condition on `a`).

- `sympy_extras._timeout.TimeLimitExceeded` is a `BaseException`: SymPy's
  routines catch `Exception` in places (the heuristics of `integrate`,
  the simplifiers), and the limit, once swallowed there, was gone for the
  rest of the computation (integrals ran for hundreds of seconds under a
  limit of twenty).

- `sympy_extras.integrals.conditions.decide` settles an alternative which
  is a conjunction: the Mellin conditions on two scales come as
  `(Abs(arg(a)) <= pi/2 & Abs(arg(v)) < pi/2) | ...`, left undecided under
  `a > 0`, `v > a` (`exp(-u*v)*besselk(0, a*u)` over `(0, oo)` stayed a
  `Piecewise`).

- `tidy` writes `besseli(-n, z) - besseli(n, z)` as `2*sin(pi*n)*besselk(n, z)/pi`
  (the definition of `K_n` for a non-integer order), so that the value
  of `t**(n - 1)*exp(-a/t - s*t)` over `(0, oo)` is
  `2*a**(n/2)*besselk(n, 2*sqrt(a*s))/s**(n/2)` (Maxima's `specint` 174)
  and not a quotient by `sin(pi*n)`, `0/0` at every integer order.

- `real_logarithms` crashed on a logarithm with a complex argument (the
  relation `argument < 0` cannot be formed: Wester's
  `x*exp(-p*x**2 + 2*x*(atan(43*sqrt(771)/2313)/3 + pi/6))`); such
  arguments are left alone.

## 0.0.2 - 2026-09-12

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
  `1/sqrt(x**3 + 1)` over `(0, oo)` is `gamma(1/6)*gamma(1/3)/(3*sqrt(pi))`;
  quartics with no real roots (Byrd–Friedman 267, the bilinear map and
  `t = lambda*tan(theta)`; `1/sqrt((x**2 + 1)*(x**2 + 4))` over the real
  line is `elliptic_k(3/4)`) and the irrational quadratic factors of
  numeric quartics from their `CRootOf` roots.

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

- `sympy_extras.integrals.exponential`: `x**(v - 1)*exp(a*x**n + b)` with
  a symbolic `v` by the incomplete gamma function in real forms (no polar
  numbers: `uppergamma` for `a < 0`, the confluent series
  `x**v*hyper((s,), (s + 1,), a*x**n)/v` otherwise), reduced to the
  elementary, `Ei` and error-function cases where they apply
  (`x**2*exp(x**2)` is `x*exp(x**2)/2 - sqrt(pi)*erfi(x)/4`); polynomials
  times `exp(a*x**2 + b*x + c)` by the completed square; rational
  functions and binomials of an exponential by `u = exp(c*x)` through the
  typed rational integrator; every result checked by differentiation and
  fourteen against Mathematica. Tried by `indefinite_integral` after the
  radical table (the census's largest open family, 501 integrands with
  symbolic exponents).

- `sympy_extras.integrals.rewriting`: canonical forms of an integrand
  (powers of positive bases and of exponentials as one exponential,
  hyperbolic functions as exponentials, inverse hyperbolic functions as
  real-branch logarithms, radicals of products combined where the
  radicands are nonnegative, `Abs` by a sign the assumptions fix) and the
  substitutions `x = t**k`, `u = exp(c*x)`, `x = exp(t)` with their back
  substitution, each checked against the integrand; `indefinite_integral`
  runs the typed methods on them before SymPy's routes
  (`1/(x**(1/3) + sqrt(x))`, `sqrt(x)/(1 + x)`, `2**x*cosh(x)`).

- `sympy_extras.integrals.heurisch`: a strictly typed port of SymPy's
  heuristic Risch integrator (Bronstein's poor man's integrator), every
  result verified by differentiation, floating-point coefficients
  refused, the logarithms of conjugates written as arctangents, the
  candidate table always on and extended to the error functions of any
  completed square, `Ei`, `Si` and `Ci`, the degree bound raised once
  before giving up, the component order deterministic; sixty-six
  integrands of Gradshteyn–Ryzhik's chapter 2 families verify in twelve
  seconds, among them `1/sqrt(2*x - x**2)`, `1/sqrt(x**2 + 2*x + 5)` and
  `x**2*sqrt(x**2 + 1)`, which SymPy's `heurisch` leaves. Tried by
  `indefinite_integral` before SymPy's.

- `sympy_extras.integrals.trigonometric`: a typed integrator for products
  of powers of trigonometric and hyperbolic functions (the raising and
  lowering formulas, `tan`, `sec`, `csc`, `cot` and their hyperbolic
  counterparts), rational functions of `sin` and `cos` by `t = tan(x/2)`
  and of `sinh`, `cosh` by `u = exp(x)` through the typed rational
  integrator, products with polynomials and exponentials, and
  polynomials times inverse trigonometric and hyperbolic functions by
  parts, every result checked by differentiation (SymPy's `trigintegrate`
  as the starting point); `tan(x)/(1 + sin(x))` and
  `cos(x)**2/(1 + sin(x)**2)`, unevaluated or timed out in SymPy, take a
  second. Tried by `indefinite_integral` before the Risch algorithm.

- `sympy_extras.integrals.indefinite`: `indefinite_integral(f, x)`, the
  antiderivative by the typed methods of the package in turn (rational
  functions, radicals of a quadratic, the Risch algorithm, the heuristic
  Risch integrator, Trager's algorithm) and SymPy's routes last, every
  candidate checked by differentiation (`is_antiderivative`) and
  rewritten to a real form when it carries `I` for a real integrand;
  the `Integral` is returned unevaluated rather than a wrong
  antiderivative.

- `sympy_extras.integrals.radicals`: real antiderivatives of
  `x**n * Q**(m/2)` with `Q` quadratic, by the reduction formulas of
  Gradshteyn–Ryzhik 2.26, tried before the Risch port and SymPy and early
  in the driver; with them, and three fixes on the way (the leak guard of
  the one-sided limits refused the symbol of a symbolic bound, the
  singularity search could not form an interval with symbolic ends, and
  the sampler of the numerical check had no point under an irrational
  bound such as `x > -sqrt(2)/2`), the slices of a region between
  algebraic bounds evaluate: `Integral(2*sqrt(1 - y**2), (y,
  -sqrt(1 - x**2), x))` is `asin(x) + asin(sqrt(1 - x**2))` for
  `-sqrt(2)/2 < x < 0`, and the three intersecting cylinders, with no
  axis of symmetry, have volume `8*(2 - sqrt(2))` through the
  decomposition in half a minute.

- `definite_integral` integrates a sum term by term when the whole
  defeats every method, keeping the value only when every term is finite
  and the total passes the numerical check: the difference of two arcs
  `sqrt(x)*sqrt(2 - x) - sqrt(1 - x)*sqrt(x + 1)` on `(1/2, 1)`, whose
  antiderivative is a complex `Piecewise`, is `sqrt(3)/4 - pi/12`, and the
  union and symmetric difference of two overlapping discs in
  `IntegralByRanges` now evaluate (`4*pi/3 + sqrt(3)/2` and
  `2*pi/3 + sqrt(3)`).

- `sympy_extras.integrals.trager`: Trager's algorithm for integrands
  rational in `x` and one square root of a polynomial (the Hermite
  reduction with the integral basis, the Rothstein–Trager resultant of
  the residues, logarithms of prescribed divisor by Newton lifting,
  torsion up to Mazur's bound on curves of genus one),
  `trager_antiderivative`, `trager_reduce` and
  `is_nonelementary_algebraic`; `definite_integral` takes the
  antiderivative from it (`(x**2 - 1)/((x**2 + 1)*sqrt(x**4 + 1))` over
  `(0, 1)` is `-sqrt(2)*pi/8`).

- `sympy_extras.integrals.summability`: Abel, Cesàro `(C, k)` and Gaussian
  means of divergent oscillatory integrals, `definite_integral(...,
  summability='abel')` (`sin(x)` over `(0, oo)` is 1, `x*sin(x)` is 0 by
  `(C, 2)`, `sin(x)**2` has no mean).

- `sympy_extras.integrals.mellin`: kernels for the Airy function `Ai`, the
  polylogarithms `Li_n(-x)`, the Fresnel integrals `S` and `C`,
  `erfc(x)*exp(x**2)` and the products of two Bessel functions of one
  argument (`J_mu*J_nu`, `K_mu*K_nu`, `I_mu*K_nu`, `J_nu*Y_nu`,
  `J_nu*K_nu`, so that `exp(-a*x)*besselj(0, x)**2` over `(0, oo)` is
  `2*elliptic_k(-4/a**2)/(pi*a)` and `x**2*besselk(nu, x)**2` is
  `pi**2*(1 - 4*nu**2)/(32*cos(pi*nu))`), checked against Mathematica and
  quadrature; the
  Laplace transform of `Ai` comes out as real confluent series (the
  `lowergamma` of a polar argument `hyperexpand` writes is rewritten).

- `sympy_extras.integrals.asymptotic`: Laplace's method with several
  maxima and maxima of any order, the uniform Airy and error-function
  expansions of Chester–Friedman–Ursell for coalescing stationary points
  (`asymptotic_integral(..., uniform=True)`), and steepest descent through
  complex saddle points (`steepest_descent`).

- `sympy_extras.integrals.axisymmetric`: region integrals with a
  rotational symmetry reduced to their profile in the half-plane of the
  radius (the group of variables entering through the sum of their
  squares only, decided by the infinitesimal rotations, becomes `rho > 0`
  with the weight of the unit sphere `S**(k-1)` times `rho**(k-1)`), for
  the Lebesgue and the Hausdorff measure: two balls a distance 1 apart
  meet in `5*pi/12`, the ball cut by the paraboloid `z = x**2 + y**2` has
  volume `5*pi*(3 - sqrt(5))/12`, the cap `z > 1/2` of the unit sphere
  area `pi`, two unit spheres meet in a circle of length `sqrt(3)*pi`,
  and two unit 4-balls a distance 1 apart in `pi*(8*pi - 9*sqrt(3))/24`,
  each in a second or two where the decomposition took twenty seconds or
  timed out; a ball cut by planes with a common normal goes through the
  frame along the normal, scaled so that the offsets stay rational (the
  slab `-1 < x + y + z < 1` of the unit ball has volume
  `16*sqrt(3)*pi/27`, the unit sphere meets the plane `x + y + z = 1` in a
  circle of length `2*sqrt(6)*pi/3`). Numeric cell bounds of degree at most four are written in
  radicals rather than as `CRootOf` (the bug: `definite_integral` left an
  integral up to `CRootOf(4*x**2 - 3, 1)` unevaluated).

- `IntegralByRanges`: polynomials over bounded polytopes integrated
  exactly by vertex enumeration, a pulling triangulation and Dirichlet's
  formula on the standard simplex (ten times faster than the
  decomposition on a pentagon), and `dimension=n` with a symbolic `n` for
  radial integrands (the unit ball has volume `pi**(n/2)/gamma(n/2 + 1)`,
  the Gaussian over `R**n` is `pi**(n/2)`).

- `sympy_extras.integrals.series`: Fourier series computed for factors
  outside the table (the coefficients as integrals with a symbolic index
  in the orthogonal systems of the range) and the parametric entries
  `log(P + Q*cos(x))`, `1/(P + Q*cos(x))` (Gradshteyn–Ryzhik 1.447-1.448):
  `x**2/(1 - 2*a*cos(x) + a**2)` over `(0, pi)` is
  `pi*(pi**2/3 + 4*polylog(2, -a))/(1 - a**2)` for `a < 1`,
  `x*log(1 - 2*a*cos(x) + a**2)` is `2*polylog(3, a) - 2*polylog(3, -a)`.

- `IntegralByRanges`: curves and surfaces given by several equations with
  `measure='hausdorff'` (the Gram determinant of the graph
  parametrisation; Viviani's curve has length `4*sqrt(2)*elliptic_e(1/2)`),
  cubic cell boundaries in trigonometric or hyperbolic form, and the
  reordering of the variables when a bound has no explicit form
  (`x**3 + y**3 < 1` in the first quadrant has area
  `2**(1/3)*gamma(1/6)*gamma(1/3)/(12*sqrt(pi))`).

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

- `sympy_extras.integrals.elliptic` evaluates the elementary primitive of
  the odd part exactly at the roots of the radicand: the value at `s = 0`
  was written with `sqrt(W(0))` as a nested radical SymPy does not see is
  zero, and the later rationalisation, whose factorisation draws random
  evaluation points, turned it into `log(1)`, `log(-1)` or `zoo` from run
  to run (the integral of `x/sqrt((x - 3)*(x - 2)*(x**2 + 1))` over
  `(3, 5)` came out with a spurious `pi*I` or as `nan` in about one run
  in three).

- Slater's expansion on the unit circle `|z| = 1` is used only when every
  hypergeometric series converges there: `Integral(airyai(x)**2, (x, 0,
  oo))` came out as `zoo` from Gauss's summation of divergent series, and
  the driver now refuses a value `zoo` or `nan` from any method.

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
