# Changelog

All notable changes to this project are documented in this file.

The project is at version 0.x: there is **no guarantee of backwards
compatibility** between releases yet, and any release may rename, move or
remove public functions. Breaking changes are listed here when they happen.

## Unreleased

### Added

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
