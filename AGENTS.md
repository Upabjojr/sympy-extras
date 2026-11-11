# Guidance for AI agents and contributors

This file describes how to work on `sympy-extras`. It is meant for AI coding
agents as well as for people.

## What this project is

`sympy-extras` is an **extension to [SymPy](https://www.sympy.org)**: a
collection of algorithms built on top of SymPy which are not (yet) part of
SymPy itself. It depends on released SymPy versions and adds functionality;
it never monkeypatches or replaces anything inside SymPy.

The project has a **lax policy on AI-generated algorithms**: code written
with AI assistance, or entirely by AI models, is welcome. Much of the code
in this repository was produced that way. The bar is the same as for any
other code, and it is about correctness and verifiability rather than
authorship:

- the algorithm is described in the module docstring with references to
  the literature;
- every public function has a docstring with doctested examples;
- there are tests which check the results against independent sources
  (hand computations, known results, other computer algebra systems such as
  QEPCAD or Redlog), not only against the code's own output. Results
  published in the literature or in the documentation of other systems
  may be used as test data (they are facts), but their code must not be
  copied: the licenses do not allow it. Such tests live in the
  `test_*_known.py` files;
- **every bug which is found gets a unit test which fails before the fix
  and passes after it**, in the `tests/` directory next to the code (see
  *Regression tests* below);
- the provenance of the code (which model, which pull request it was ported
  from) is stated in the commit message.

The project is at version 0.x and **does not guarantee support against
breaking changes** for now. Public functions may be renamed, moved or
removed in any release. Every user-visible change goes into `CHANGELOG.md`.

## Standing instructions from the maintainer

These were given while the project was being built and apply to every
change:

- **Never copy code** from Mathematica, Maxima, Maple or any other system
  whose license does not allow it. Published results, test data and
  algorithm descriptions are facts and may be used; implementations are
  written from the literature and stated in the module docstring with
  references.
- **Code from SymPy may be ported** (decision of the maintainer,
  2026-11-07, to make sympy-extras independent of SymPy). SymPy is under
  the BSD 3-clause license, which allows it provided the copyright notice
  is kept. Every file containing code ported from SymPy, wholly or in
  part, must:
  1. carry this notice at the top of the file (before or right after the
     module docstring), filled in:

     ```python
     # ---------------------------------------------------------------------
     # PORTED FROM SYMPY: <sympy/module/path.py>[, <other paths>] (SymPy <version>)
     # Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
     # see sympy_extras/LICENSE-SymPy.
     # Changes: <strict typing, unevaluating classes, ... in one or two lines>
     # ---------------------------------------------------------------------
     ```

  2. say so in the module docstring (one sentence naming the SymPy
     module);
  3. mark each ported function, class or method in a file which mixes
     ported and original code with a comment on the line above it:
     `# PORTED FROM SYMPY: sympy/module/path.py:<name>`;
  4. be typed to the rules below (no `Any`, no `cast`, no `getattr`, no
     `# type: ignore`) and have its own tests: a port is not a copy of
     SymPy's tests, which are BSD-licensed too and may be ported under
     the same notice;
  5. split every SymPy class which evaluates in its constructor into a
     **class which never evaluates** and a **lower case function which
     does**, as the core does for `Add`/`add` and `Mul`/`mul`: `Sin(0)`
     is the node `sin(0)` and stays so, `sin(0)` (the function) is `0`;
     `Exp`/`exp`, `Log`/`log`, `Sin`/`sin`, ... The class constructor only
     checks and stores its arguments; the evaluation rules live in the
     classmethod `eval` (`DefinedFunction.apply` runs them), and `doit`
     rebuilds a tree with the functions. Where SymPy's name is the class
     (`sympy.sin`), our function keeps that name and the class takes the
     capitalised one;
  6. keep nodes **immutable after their creation**: every attribute set
     in a constructor is annotated `Final[...]`, so that mypy rejects any
     later assignment (a runtime guard in `__setattr__` cost 30-80 % of
     the speed of the core and was rejected); only the lazily filled
     caches (`_compare_key_cache`, `_sympy_cache`, `_facts_cache`) are
     not `Final`, and global settings (`global_parameters`) are not nodes.

  `grep -rn "PORTED FROM SYMPY" sympy_extras` lists every port; the
  commit message names the SymPy modules and version. Files with
  original code only carry no notice. `sympy_extras/LICENSE-SymPy` is
  the license text and must not be removed while any port remains.
- **Algorithms that need no change stay in SymPy.** SymPy is a dependency:
  call it (`solveset`, `dsolve`, `groebner`, `roots`, `gosper_sum`,
  `limit`, ...) and only add what is missing or must be rewritten
  (assumptions as statements, the CAD, decision procedures). The tables
  in `docs/reduce.md` say for each algorithm of Mathematica's
  implementation notes whether it lives in SymPy, here, or nowhere yet.
- **Assumptions are mathematical statements** (`x > 0`, `element(x,
  S.Integers)`), not `Q` predicates or `Symbol` flags; every function
  that takes assumptions accepts them in this form.
- **Verify against independent sources at scale**: datasets and
  benchmark drivers live in `benchmarks/` (Kamke's ODEs from Maxima's
  test suite, SMT-LIB QF_NRA, random equations against numerical
  oracles); random samples are pulled from them and wrong results are
  fixed, not worked around. Results go into `benchmarks/README.md`.
- **Numerical checks are optional**: `sympy_extras.settings` has a global
  switch (`numerical_checks`), the precision and the time limit, with
  the `configure` context manager; exact results must not depend on
  them, they only prune candidates and verify.
- **Report SymPy bugs and limitations in this repository's issue
  tracker** (issue #25 collects them with snippets and the workaround
  used here), so that they can be reported upstream; work around them
  in this package rather than waiting.
- **Time limits everywhere**: SymPy routines that may not terminate run
  under `sympy_extras._timeout.attempt` with the limit of the settings.
- **Isolated values of the parameters get their own cases**: a closed
  form in parameters which is undefined at isolated values where the
  problem is defined (`(y**2 - 1)/log(y)` for the integral of `y**x`
  over `(0, 2)`, at `y = 1`) comes with `Piecewise` cases for them,
  `Eq(y, 1)` first, the value computed at the point rather than taken as
  a limit (`sympy_extras._special_values`, used by `definite_integral`,
  `indefinite_integral`, the summation functions and the ODE solvers,
  for which a general solution that is defined but no longer general,
  its Wronskian zero at a repeated characteristic root, counts as
  undefined); SymPy's own cases
  (`x**n` at `n = -1`) are kept, and the numerical checks check them.
  No case where the problem is not defined (`a**(b*z)` at `a = 0`,
  `defined_problem`), no unevaluated case of an antiderivative, and the
  cases stay cheap: a budget of their own, a tenth of the time limit.
- **Merge into `master`**: this is a private experimental repository, the
  development branch is fast-forwarded into `master` after every
  verified change (tests, doctests, mypy, pyflakes); the changelog and
  the docs are updated in the same commit.
- **Track the remaining work in issues**: what is not implemented or
  only partially (issues #23 and #24 for the Mathematica lists) is
  written down there with the reasons, and updated when progress is
  made.

## Layout

The package mirrors SymPy's layout: code extending `sympy.polys` goes in
`sympy_extras/polys`, code extending `sympy.solvers` would go in
`sympy_extras/solvers`, and so on. This makes it straightforward to move a
module into SymPy later.

```
sympy_extras/
    __init__.py              __version__
    assumptions/             assumptions written as mathematical statements (exports nothing: import the modules)
        facts.py             element(), normalize(), Facts: translation to predicates and polynomial relations (native)
        facts_sympy.py       the SymPy face of facts, and predicates_consistent (SymPy's SAT on its known facts)
        quantifiers.py       ForAll, Exists (nodes), forall, exists, prenex() (native)
        quantifiers_sympy.py the SymPy face: ForAll, Exists as SymPy Booleans (interop converts), prenex
        context.py           assuming(), global_assumptions (native)
        context_sympy.py     the SymPy face: the same global_assumptions, assuming() pushing to SymPy's
        bounds.py            polynomial_abstraction: MetiTarski bounds of elementary functions (native)
        bounds_sympy.py      the SymPy face of bounds
        intervals.py         rigorous interval arithmetic: evaluate, signs_on_box, corner_signs, isolate_signs (native)
        intervals_sympy.py   the SymPy face of intervals
        analysis.py          signs beyond polynomials: certified_sign, sign_on, domain_of, decide_relational (native;
                             the native CAD; continuous_domain/solveset of non-rational expressions through analysis_sympy)
        analysis_sympy.py    the SymPy face of analysis, and the steps which stay on SymPy (the *_of_core functions)
        ask.py               ask(): the core's ask first, then the CAD (native, with the native CAD and virtual
                             substitution; the structure theorem and the sample points through ask_sympy)
        ask_sympy.py         the SymPy face of ask, and the steps which stay on SymPy (the *_of_core functions)
        refine.py            refine(), simplify(): the core's refine and simplify plus CAD-decided handlers (native)
        refine_sympy.py      the SymPy face of refine (_Refiner too), SymPy's simplify of Booleans, the proofs
        resolve.py           resolve(): quantifier elimination over R, Z, C (native, with the native CAD, virtual
                             substitution, Cooper and comprehensive Groebner systems)
        resolve_sympy.py     the SymPy face of resolve
        sat.py               satisfiable(), tautology(), find_instance(): SAT + theory check (native, with the native CAD
                             and Hermite's count; SymPy's SAT solver and predicates_consistent through sat_sympy)
        sat_sympy.py         the SymPy face of sat, and the steps which stay on SymPy (the *_of_core functions)
        solve.py             solve(): Solve under assumptions (native, with the native CAD, regular chains and parametric cases;
                             solveset, nonlinsolve, transcendental equations through solve_sympy)
        solve_sympy.py       the SymPy face of solve, and the steps which stay on SymPy (the *_of_core functions)
        limits.py            limit(), limit_seq(), series() under assumptions, with case distinctions (native; the core's
                             Gruntz and series; SymPy's limit_seq and the reliable numerical values through limits_sympy)
        limits_sympy.py      the SymPy face of limits, and the steps which stay on SymPy (the *_of_core functions)
        tests/
    functions/               functions of the typed core on core.function.DefinedFunction (ports of sympy.functions)
        elementary/
            exponential.py   Exp, Log, LambertW (nodes) and exp, log, lambertw (evaluating), PORTED FROM SYMPY
            complexes.py     Re, Im, Abs, Sign, Arg, Conjugate, Atan2 / re, im, abs_, ...; as_real_imag, conjugate_of, PORTED FROM SYMPY
            integers.py      Floor, Ceiling, Frac / floor, ceiling, frac, PORTED FROM SYMPY
            miscellaneous.py Max, Min / max_, min_ (variadic DefinedFunction); root, cbrt, real_root, PORTED FROM SYMPY
            piecewise.py     Piecewise, ExprCondPair / piecewise, PORTED FROM SYMPY
            trigonometric.py Sin, ..., Csc and the inverses Asin, ..., Acsc / sin, ..., acsc; pi_coeff, peeloff_pi, PORTED FROM SYMPY
            hyperbolic.py    Sinh, ..., Csch and the inverses Asinh, ..., Acsch / sinh, ..., acsch; peeloff_ipi, PORTED FROM SYMPY
        combinatorial/
            factorials.py    Factorial, Binomial, RisingFactorial, FallingFactorial / factorial, binomial, rf, ff, PORTED FROM SYMPY
            numbers.py       Catalan, Fibonacci, Lucas, Harmonic, Bernoulli, Euler, Bell, Genocchi, Partition / catalan, ..., PORTED FROM SYMPY
        special/
            gamma_functions.py Gamma, LogGamma, PolyGamma, DiGamma, TriGamma, LowerGamma, UpperGamma / gamma, ..., PORTED FROM SYMPY
            delta_functions.py DiracDelta, Heaviside / dirac_delta, heaviside, PORTED FROM SYMPY
            zeta_functions.py Zeta, DirichletEta, PolyLog / zeta, dirichlet_eta, polylog; Bernoulli numbers, PORTED FROM SYMPY
            error_functions.py Erf, ..., Expint, EiNode, SiNode, CiNode, ShiNode, ChiNode, LiNode, OffsetLiNode / erf, ..., Ei, Si, Ci, Shi, Chi, li, Li, PORTED FROM SYMPY
            beta_functions.py Beta / beta, PORTED FROM SYMPY
            bessel.py        BesselJ, ..., Hankel1, Hankel2, SphericalJ, SphericalY, AiryAi, ... / besselj, ..., jn, yn, airyai, ..., PORTED FROM SYMPY
            polynomials.py   Legendre, ChebyshevT, ChebyshevU, Hermite, Laguerre, AssocLaguerre, Jacobi, Gegenbauer, AssocLegendre, HermiteProb / legendre, ..., PORTED FROM SYMPY
            spherical_harmonics.py YnmNode, ZnmNode / Ynm, Znm, Ynm_c, PORTED FROM SYMPY
            tensor_functions.py KroneckerDeltaNode, LeviCivitaNode / KroneckerDelta, LeviCivita, PORTED FROM SYMPY
            mathieu_functions.py MathieuS, MathieuC, MathieuSPrime, MathieuCPrime / mathieus, ..., PORTED FROM SYMPY
            singularity_functions.py SingularityFunctionNode / SingularityFunction, PORTED FROM SYMPY
            hyper.py         Hyper, MeijerG (Expr nodes with parameter lists), AppellF1 / hyper, meijerg, appellf1, PORTED FROM SYMPY
            bsplines.py      bspline_basis, bspline_basis_set, PORTED FROM SYMPY
            elliptic_integrals.py EllipticK, EllipticF, EllipticE, EllipticPi / elliptic_k, ..., PORTED FROM SYMPY (Fresnel in error_functions.py)
    core/                    an experimental core: unevaluating classes (Add, Mul, Pow, Mod, Eq, And, Or, Derivative,
                             Subs, Order, ...) and evaluating functions (add, mul, pow, mod, eq, and_, or_, diff,
                             order, ...), typed; Wild/match/replace, the full set of is_* predicates, Q/ask/refine,
                             the Expr methods of SymPy (integrate, aseries, ... through the conversion),
                             native Order, series driver and limits (order.py, series.py, gruntz.py, limits.py; per-class _eval_nseries and
                             _eval_as_leading_term, SymPy's per node for the functions not ported),
                             factor_terms, a native rewrite (rewriting.py), SymPy's order of the arguments, to_sympy/from_sympy with generic wrappers
                             of any SymPy object; see docs/core.md
    concrete/                summation algorithms extending sympy.concrete
        pisigma.py           PiSigmaField: the tower, sigma, the parameterized first order solver (typed, no SymPy)
        karr.py              summand analysis (build_pisigma_field), karr_sum, karr_term, summation (native)
        karr_sympy.py        the SymPy face of karr: karr_sum, karr_term, summation, with_sum_special_values, reliable_value_of_core
        creative.py          creative telescoping in PiSigma-fields: creative_telescoping, definite_sum (native)
        guess.py             guess_recurrence, guess_differential_equation, guess_algebraic_equation, guess_closed_form: equations from the first terms (native)
        zeilberger.py        Zeilberger's algorithm, WZ certificates, zeilberger_sum (native)
        zeilberger_sympy.py  the SymPy face of zeilberger: zeilberger_sum, wz_certificate, wz_prove
        qhyper.py            q-Pochhammer symbols (QPochhammer node, qpochhammer), q-Gosper, q-Zeilberger (native)
        qhyper_sympy.py      the SymPy face of qhyper: QPochhammer as a SymPy function, qgosper_sum, qzeilberger, ...
        tests/
    integrals/               definite integration extending sympy.integrals
        mellin.py            Mellin transforms as gamma quotients (GammaQuotient), the table of kernels
        slater.py            Mellin–Barnes integrals as Meijer G-functions, Slater's theorem
        marichev.py          the Marichev–Adamchik method over (0, oo), (0, 1), (1, oo)
        residues.py          rational, Fourier and trigonometric integrals by the residue theorem
        definite.py          definite_integral: splitting, range mappings, the verified SymPy fallback
        regions.py           IntegralByRanges: integrals over semialgebraic regions through the CAD
        conditions.py        ConditionalValue: values with the conditions under which they hold
        telescoping.py       Almkvist–Zeilberger creative telescoping, holonomic_ode, holonomic_integral
        parametric.py        differentiation under the integral sign (Feynman's trick)
        antiderivative.py    antiderivatives evaluated by one-sided limits at every discontinuity
        brackets.py          Ramanujan's master theorem and the method of brackets (Mellin transforms from series)
        recognize.py         PSLQ recognition of constants and integrals (an oracle, off by default)
        risch/               the transcendental Risch algorithm ported from Aaron Meurer's SymPy pull requests
        trager.py            Trager's algorithm for integrands rational in x and one square root of a polynomial
        curves.py            Trager's algorithm on an arbitrary curve: several or nested radicals through a primitive element, the change of variable x = N + 1/z, Hermite reduction in a normal integral basis, residues by Puiseux expansions, the logarithmic part by linear algebra
        summability.py       Abel, Cesàro and Gaussian means of divergent oscillatory integrals
        axisymmetric.py      region integrals with a rotational symmetry (bodies and surfaces of revolution) reduced to their profile
        radicals.py          real antiderivatives of x**n * Q**(m/2), Q quadratic (Gradshteyn-Ryzhik 2.26), for the slices of regions
        indefinite.py        indefinite_integral: the typed methods in turn, then SymPy's, each antiderivative checked by differentiation
        trigonometric.py     powers, products and rational functions of trigonometric and hyperbolic functions, inverse functions by parts
        heurisch.py          typed port of SymPy's heuristic Risch integrator (BSD, see risch/LICENSE-SymPy), every result verified
        rewriting.py         canonical forms (exponentials, logarithms of inverse hyperbolic functions) and substitutions x = t**k, u = exp(c x), x = exp(t)
        exponential.py       x**(v-1) exp(a x**n + b) by the incomplete gamma function in real forms, completed squares, rational functions of exp(c x)
        tests/
    solvers/                 solvers extending sympy.solvers (exports nothing: import the modules)
        isolation.py         TranscendentalRoot (node), isolate_real_roots, point_between, compare_points: exact real roots of
                             exp-log functions (native; SymPy's real_roots with CRootOf and continuous_domain through isolation_sympy)
        isolation_sympy.py   the SymPy face of isolation (TranscendentalRoot as a SymPy Expr, interop converts), the *_of_core functions
        integers.py          linear Diophantine systems (Hermite), Contejean-Devie, Cooper's algorithm (native;
                             SymPy's simplify_logic through integers_sympy)
        integers_sympy.py    the SymPy face of integers, and simplify_logic_of_core (from virtual_substitution_sympy)
        parametric.py        parametric_cases, ParametricCase: the cases of the parameters of a polynomial system, from
                             the regular chains (native)
        parametric_sympy.py  the SymPy face of parametric
        tests/
    simplify/
        structure.py         the structure theorem: ElementaryTower, canonical_form, is_zero, equal (native; the sample
                             points and the reliable values through structure_sympy)
        structure_sympy.py   the SymPy face of structure, and the steps which stay on SymPy (the *_of_core functions)
        signsimp.py          signsimp, sub_pre, sub_post (ported from SymPy)
        radsimp.py           fraction, numer, denom, collect, collect_dict, rcollect, radsimp, rad_rationalize, split_surds (ported from SymPy)
        sqrtdenest.py        sqrtdenest, sqrt_depth, is_algebraic (ported from SymPy)
        gammasimp.py         gammasimp, combsimp (ported from SymPy)
        ratsimp.py           ratsimp (ported from SymPy)
        trigsimp.py          trigsimp (matching, fu), futrig, exptrigsimp (ported from SymPy)
        fu.py                fu and its TR rules (TR0-TR22, TRmorrie, TRpower), greedy strategies, hyper_as_trig (ported from SymPy)
        powsimp.py           powsimp, powdenest, multiplicity, _denest_pow (ported from SymPy)
        simplify.py          simplify, logcombine, posify, nsimplify, collect_abs, nc_simplify, besselsimp (ported from SymPy)
        hyperexpand.py       hyperexpand: formula tables, shift operators, plans, Slater's theorem (ported from SymPy)
        tests/
    matrices/                ExprMatrix: a small immutable matrix of expressions (what hyperexpand needs)
    series/                  residue (ported from SymPy)
    sets/                    Interval, FiniteSet, Union, Intersection, Complement, Reals, Contains, ProductSet, PointSet (finite sets of tuples);
                             Naturals, Naturals0, Integers, Rationals, Complexes, ImageSet (fancysets.py); ConditionSet (conditionset.py) (ported from SymPy)
    ntheory/                 number theory without SymPy: isprime (BPSW), nextprime/prevprime, factorint (Pollard-Brent)
    polys/
        domains/             the typed coefficient domains of the standalone layer (Domain[T]: ZZ, QQ, GF, AlgebraicField, PolynomialRing, FractionField, SparsePolynomialRing with SymPy's PolyElement gcd, RationalPolynomialRing, IteratedPolynomialRing for dmp_rr_prs_gcd); see docs/dense.md
        densebasic.py        dense univariate polynomials over a Domain[T] (list[T], highest degree first)
        densearith.py        dup_add/mul/pow/div/pdiv/exquo over a Domain[T]
        densetools.py        dup_diff/eval/shift/compose/content/primitive/monic/trunc/clear_denoms
        euclidtools.py       dup_gcd (Euclid, subresultant PRS), dup_resultant, dup_prs_resultant (ported), dup_discriminant, dup_gcdex (typed, no SymPy)
        polytools.py         Poly over ZZ, QQ, ZZ[a,..], QQ[a,..] (ported from SymPy, printed as SymPy): gcd, lcm, resultant, factor_list (multivariate, ZZ[a]), sqf_list, div; degree, LC, gcd, lcm, resultant, factor, factor_list, sqf, sqf_list, cancel, invert (no SymPy; EX, CC and the other domains not ported through polytools_sympy)
        polytools_sympy.py   the steps of polytools which stay on SymPy: invert_of_core, cancel_of_core, factor_of_core
        gaussianfactor.py    multivariate factorisation over ZZ_I and QQ_I (Trager over QQ<I>, ported from SymPy)
        polyerrors.py        SymPy's polynomial exceptions (ported)
        algebraicfactor.py   ext_factor_list, AlgebraicPolynomialRing: multivariate factorisation and gcd over a number field QQ(alpha) (Trager)
        heuristicgcd.py      heugcd over ZZ (ported from SymPy), deflate/inflate; IntegerPolynomialRing uses it
        rationaltools.py     together (ported from SymPy)
        multifactor.py       zz_factor, zz_wang: multivariate factorisation over ZZ by Wang's EEZ on sparse PolyElement (ported from SymPy)
        multisqf.py          sqf_list over ZZ, QQ, ZZ[a], QQ[a] in several variables (SqfDomain; ported from SymPy)
        partfrac.py          apart by undetermined coefficients (ported from SymPy; no full=True, no ZZ(a))
        polyroots.py         roots: linear, quadratic, binomial, cyclotomic, cubic, quartic, decomposition, factoring (ported from SymPy; no EX/simplify)
        roots.py             radical_form, in_radicals: SymPy's CRootOf (wrapped foreign nodes) in radicals (native; SymPy's roots over EX through roots_sympy)
        roots_sympy.py       the SymPy face of roots, and roots_of_core
        subresultants.py     dup_psc (over a Domain[T], towers for several variables), psc (on rings.PolyRing): principal
                             subresultant coefficients (native)
        subresultants_sympy.py the SymPy face of subresultants: dup_psc, dmp_psc on SymPy's dense representation, psc on SymPy's rings
        polyutils.py         dict_from_expr, expr_from_dict: core expressions <-> typed polynomials (no SymPy)
        sqfreetools.py       dup_sqf_list/part/p (Yun, and GF(p))
        galoisfactor.py      gfq_factor, gfq_sqf_list, gfq_ddf, gfq_edf over domains.galoisfield.GaloisField (GF(p**n))
        galoistools.py       gf_factor, gf_ddf, gf_edf, gf_irreducible_p over FiniteField
        factortools.py       dup_zz_factor, dup_qq_factor (Zassenhaus; van Hoeij's knapsack recombination from 10 modular factors on), dup_zz_hensel_lift, dup_ext_factor (Trager)
        lattice.py           lll, lll_reduction (integral LLL, Gram determinants), hermite_normal_form, integer_kernel, solve_integer_system, smith_normal_form (typed, no SymPy)
        rootisolation.py     dup_isolate_real_roots(_sqf), dup_count_real_roots (Sturm), dup_refine_real_root, over QQ
        numberfields/basis.py  integral_basis, field_discriminant (Round 2), polynomial_discriminant
        numberfields/qqbar.py  Algebraic: exact arithmetic on algebraic numbers (minimal polynomial + isolating region), algebraic_roots
        complexroots.py      Rectangle, dup_count_complex_roots (winding numbers by Sturm), dup_isolate_complex_roots(_sqf), dup_refine_complex_root, dup_count_roots_right_of, dup_count_roots_in_disc, Hurwitz and Schur stability, over QQ
        polyclasses.py       DensePoly[T]: rep, domain, gen; the dense algorithms as methods, factorisation and roots through the domain
        rings.py             PolyRing[T], PolyElement[T] (dict[Monomial, T]), the orders of orderings.py, div by a list
        ideals.py            Ideal: operations and invariants of ideals over the typed fields, groebner (F4, the modular
                             fallback of modulargroebner), normal_form, the Gröbner walk, separating_forms, the decompositions
                             and rur (native)
        ideals_sympy.py      the SymPy face of ideals (the old API on SymPy objects), and the steps which stay on SymPy
                             (the *_of_core functions)
        idealdecomposition.py radical, equidimensional_parts, minimal_primes, primary_decomposition (GTZ + SY),
                             associated_primes, is_primary, independent_set through regular chains (native)
        idealdecomposition_sympy.py the SymPy face of idealdecomposition (the ideals of ideals_sympy)
        bigatti.py           hilbert_numerator_coefficients, minimal_generators: Bigatti's algorithm on exponent tuples (no SymPy)
        hermite.py           real solutions of zero-dimensional systems: Hermite's quadratic form, sign determination,
                             decide_zero_dimensional (native)
        hermite_sympy.py     the SymPy face of hermite
        comprehensive.py     comprehensive Gröbner systems (Kapur-Sun-Wang), exists_complex,
                             complex_quantifier_elimination, reduce_complex: reduction over C (native)
        comprehensive_sympy.py the SymPy face of comprehensive
        groebnerwalk.py      groebner_walk (liftings by normal forms and by cofactors), extended_groebner (Buchberger with
                             representations), is_groebner, normal_form, initial_form on rings.PolyRing (native; ideals re-exports)
        groebnerwalk_sympy.py the SymPy face of groebnerwalk (SymPy's rings and orders; ZZ through QQ, as SymPy's groebner)
        f4.py                f4 on rings.PolyRing (typed, no SymPy): reduced Gröbner bases by Faugère's F4 (the default direct algorithm); row_echelon
        zerodim.py           solve_zero_dimensional: exact solutions (Algebraic coordinates) of zero-dimensional systems over QQ by the shape lemma
        berlekampmassey.py   berlekamp_massey: the shortest linear recurrence of a sequence over a field
        fglm.py              fglm, groebner_by_fglm on rings.PolyRing (typed, no SymPy): change of order for zero-dimensional ideals
        interop.py           qq/zz/gf_from_sympy and _to_sympy, typed_order: the boundary with SymPy's rings
        hilbert.py           hilbert_numerator (DensePoly over ZZ), hilbert_series, reduced_hilbert_series, minimal_generators:
                             Bigatti's algorithm for monomial ideals (native; bigatti.py computes)
        hilbert_sympy.py     the SymPy face of hilbert (SymPy Poly and expressions)
        rur.py               rur, RationalUnivariateRepresentation: Rouillier's representation of zero-dimensional ideals from
                             traces (native; the exact solutions, SymPy's CRootOf, through rur_sympy)
        rur_sympy.py         the SymPy face of rur (Poly and SymPy expressions), and solutions_of_core
        modulargroebner.py   modular_groebner: reduced Gröbner bases over QQ and ZZ modulo primes, proven (Arnold), on rings.PolyRing;
                             direct_groebner (F4 or Buchberger), groebner (the dispatcher Ideal calls) (native)
        modulargroebner_sympy.py the SymPy face of modulargroebner (SymPy's ring elements; SymPy's groebner over the other domains)
        buchberger.py        buchberger, s_polynomial on rings.PolyRing over a typed field (settings.groebner_algorithm 'buchberger'), PORTED FROM SYMPY
        stronggroebner.py    strong_groebner, strong_reduce on rings.PolyRing over ZZ (typed, no SymPy): S- and G-polynomials
        orderings.py         MonomialOrder, lex, grlex, grevlex, WeightOrder, BlockOrder, elimination_order, as_order: the
                             orders of rings.PolyRing (native; rings re-exports them)
        orderings_sympy.py   the SymPy face of orderings: WeightOrder, BlockOrder, elimination_order as SymPy MonomialOrder
        virtual_substitution.py  linear and quadratic quantifier elimination by virtual substitution (Loos-Weispfenning,
                             Weispfenning), with_sides, to_nnf (native; SymPy's simplify_logic of the results through
                             virtual_substitution_sympy)
        virtual_substitution_sympy.py  the SymPy face of virtual_substitution, and simplify_logic_of_core
        puiseux.py           Newton–Puiseux expansions of the branches of a plane curve at a point and at infinity, conjugate branches over
                             exact number fields (Duval; AlgebraicField, the rationals as RATIONALS) (native)
        puiseux_sympy.py     the SymPy face of puiseux (Poly, SymPy's number fields both ways), and field_of (CRootOf points)
        functionfields.py    FunctionField[T]: arithmetic in K(x)[y]/(F), integral bases (Trager's Round 2 over K[x]), local bases at
                             infinity, bases normal at infinity, over QQ, number fields, parameters (native)
        functionfields_sympy.py the SymPy face of functionfields (coordinates in SymPy's K(x)), used by integrals/curves.py
        tests/
        cad/                 cylindrical algebraic decomposition (exports nothing: import the modules)
            projection.py    projection operators (McCallum, Hong), projection_sets, on the typed Poly (native)
            projection_sympy.py the SymPy face of projection (SymPy's Poly in and out)
            samplepoints.py  exact real algebraic sample points (SamplePoint) on qqbar.Algebraic and AlgebraicField (native)
            samplepoints_sympy.py the SymPy face of samplepoints (Rational and CRootOf in and out)
            lifting.py       lifting phase, cylindrical_algebraic_decomposition, CAD, CADCell, Lifting (native)
            lifting_sympy.py the SymPy face of lifting (cells with SymPy points), used by qe, cylindrical, regions
            qe.py            quantifier_elimination, decide, sample_points, solution_set, truth_tables (native)
            qe_sympy.py      the SymPy face of qe (SymPy formulas and sets, CRootOf), and _compile for the regions
            cylindrical.py   cylindrical_formula, cylindrical_set, cylindrical_cases (native)
            cylindrical_sympy.py the SymPy face of cylindrical
            indexedroot.py   IndexedRoot (node), indexed_root, real_root, real_number, algebraic_of: root functions and
                             real algebraic numbers of the answers (native)
            indexedroot_sympy.py IndexedRoot as a SymPy Function; interop converts (a root with rational coefficients is CRootOf)
            tests/
        differential/        differential algebra on the core (native; coefficients in ZZ[p, x] of the typed layer)
            ring.py          DifferentialRing: derivatives, rankings (orderly, elimination by blocks), conversion of
                             core expressions, integer_product/integer_quotient (fast ZZ[p, x] arithmetic)
            janet.py         janet_basis, JanetBasis: linear systems of PDEs, Hilbert series, series solutions
            polynomial.py    DifferentialPolynomial: leader, initial, separant, Ritt's reduction
            rosenfeld_groebner.py  rosenfeld_groebner: regular differential systems of polynomial ODEs and PDEs
                             (the native Ideal, regular chains and factor_list)
            tests/           test_differential_native.py on the core; the other tests on the face
        differential_sympy.py the SymPy face of differential (the old API on SymPy objects, SymPy's QQ(p, x) and
                             ZZ[p, x] coefficients, the chains of regularchains_sympy)
        regularchains/       triangular decompositions of polynomial systems (native; the exact solutions of a chain,
                             SymPy's CRootOf, through regularchains_sympy)
            recursive.py     polynomials of rings.PolyRing over ZZ in their main variable: initial, lazy pseudo-division,
                             subresultant chains (prem, subresultants ported from SymPy)
            operations.py    Chain, Decomposer: regularize, regular_gcd, intersect, extend, triangularize (Lazard, Kalkbrener)
            regularchain.py  RegularChain, triangularize, regular_gcd (the public interface on core expressions)
            tests/           oracle.py checks the decompositions of the face with Gröbner bases
        regularchains_sympy/ the SymPy face of regularchains (RegularChain, triangularize, regular_gcd on SymPy objects),
                             and exact_solutions_of_core
            solutions.py     exact_solutions: the points of a chain without free variables (CRootOf, radicals)
docs/                        longer documentation per module
.github/workflows/           tests.yml (CI), release.yml (PyPI publishing on tags)
conftest.py                  pytest configuration (SymPy-style doctest display)
pyproject.toml               package metadata; version is read from sympy_extras/__init__.py
```

Tests live next to the code in `tests/` subdirectories, as in SymPy, and
are shipped with the package.

## Conventions

- Follow SymPy's coding conventions: the low level polynomial routines use
  the dense `dup_*`/`dmp_*` representation, the high level ones take `Expr`
  or `Poly`; use `Poly`, `CRootOf`, the domains in `sympy.polys.domains`
  and SymPy's own containers rather than reinventing them.
- Only use public or well-established SymPy APIs and check that the code
  runs on the minimum supported SymPy version declared in `pyproject.toml`
  (SymPy master and released versions differ; for instance
  `sympy.polys.densebasic.dmp_zero` takes one argument in releases).
- Docstrings use the SymPy style: a summary, `Parameters`, `Returns`,
  `Examples` (with `>>>` doctests) and `References` sections with `====`
  underlines.
- Doctests are run with `pytest --doctest-modules` and follow SymPy's
  conventions: results are displayed with `str`, so `QQ(2, 5)` prints as
  `2/5`. `conftest.py` installs the display hook.
- Tests use plain `assert` statements and `sympy.testing.pytest.raises`
  for exceptions, like SymPy's tests.
- Keep the code pure Python with SymPy (and mpmath, through SymPy) as the
  only runtime dependency.
- Do not reimplement what SymPy already does: call SymPy's algorithms
  (`ask`, `refine`, `simplify`, `satisfiable`, the polynomial routines) and
  add what is missing on top. Algorithms that need no change stay in SymPy.

## Types: strict type stability (required)

Type stability is a hard requirement of this project, not a style
preference: the code must be translatable to a statically typed language
(C++ or Rust) one day, so every value must have one explicit, statically
known type. mypy in strict mode enforces it (`python -m mypy`, configured
in `pyproject.toml`) and the CI job fails on any error; the package ships a
`py.typed` marker.

Rules, all of them mandatory for every change:

1. **Every function, method, lambda-free callable and test has explicit
   parameter and return annotations.** No untyped or partially typed
   definitions; tests return `None` and their helpers are annotated too.
2. **No implicit `Any`.** `Any` is allowed only for elements of SymPy
   coefficient domains (`DomainElement`) and inside the dense polynomial
   aliases `Dup`/`Dmp`, both defined in `sympy_extras/_typing.py`. Do not
   introduce new uses of `Any`, `object` as a catch-all, or untyped
   containers (`list`, `dict`, `tuple` without parameters).
3. **Narrow at the SymPy boundary.** SymPy 1.14 ships no `py.typed` and
   few annotations, so its results are `Basic` or `Any`. The moment a
   SymPy value enters our code, convert it with the helpers of
   `sympy_extras/_typing.py` (`as_expr`, `as_boolean`, `as_symbol`,
   `as_set`, `free_symbols`, `sorted_symbols`), which check the class at
   run time and raise `TypeError`. Never use `typing.cast` to silence a
   SymPy type; never rely on duck typing across a union.
4. **Unions are narrowed with `isinstance` before use**, `Optional` values
   are checked before use, and a function returns one type of value (or a
   documented `Optional`), never a value whose type depends on an
   argument, except through `@overload` on a literal argument (as
   `satisfiable` does with `all_models`).
5. **Name the structures.** Recurring shapes get an alias in
   `sympy_extras/_typing.py` (`Monomial`, `Sign`, `Truth`, `Weights`,
   `OrderSpec`, `QuantifierPrefix`, ...) or in the module that owns them
   (`Solution`, `Rhs` in `pisigma.py`, `CellTruth` in `qe.py`, `Model`,
   `Witness` in `sat.py`). Use them instead of repeating tuple types.
6. **Attributes are declared with their type**, in `__init__` or as class
   level annotations, before being assigned in other methods
   (`PiSigmaField` declares `field`, `gens`, `zero`, ... in `__init__`).
   Attributes that SymPy sets dynamically (so that mypy cannot see them)
   are not read at all: narrow the object with `isinstance` to the class
   that declares the attribute (`_unit` narrows to `AlgebraicField`), or
   rebuild the value from a typed API (`_root_poly` builds the polynomial
   of a `ComplexRootOf` from its expression; `PiSigmaField` gets its zero
   and one from `field(0)` and `field(1)`). `getattr` is forbidden.
7. **`# type: ignore`, `typing.cast` and `getattr` are forbidden.** The
   code base has none, and a change that adds one is not accepted:
   restructure the code so that the types are honest instead. Two
   consequences of this rule shape the API:
   - Constructors do not evaluate. A SymPy-style `__new__` that returns a
     different object in trivial cases cannot be typed, so `ForAll(x, True)`
     is a `ForAll` instance and the trivial cases are reduced by
     `simplify()`, `prenex` and the functions consuming quantified
     formulas. Follow the same pattern for new classes.
   - Tests which check that a wrong argument type raises `TypeError` call
     the function through `sympy_extras._testing.untyped`, whose type
     `Callable[..., object]` makes the mismatch explicit instead of
     suppressing it. A compiled formula in `qe.py` is a small hierarchy of
     node classes with typed fields, not a tagged union.
8. **The mypy configuration is not to be weakened.** `strict = true` stays;
   the only relaxations are `disallow_untyped_calls` and
   `warn_return_any` (both forced off by SymPy's untyped API) and
   `strict_equality` for the tests (they compare SymPy numbers with Python
   literals). Do not add `ignore_errors`, per-module `disallow_*` = false,
   or new `ignore_missing_imports` entries beyond `mpmath`.
9. **Run mypy before every commit.** `python -m mypy` must report
   `Success: no issues found`; a change with type errors is not done, and
   the CI job (`typecheck` in `.github/workflows/tests.yml`) rejects it.

When SymPy gains annotations, tighten the configuration (enable
`warn_return_any`, then `disallow_untyped_calls`) rather than relying on
the relaxations. Issue #15 tracks what cannot be checked yet and the
planned typed core layer independent of SymPy objects.

## Regression tests for every bug

A bug which is fixed without a test comes back. Whenever a wrong result, a
crash, a hang or a state leak is found in this package — by a user, by a
benchmark driver, by a doctest of the documentation, or while working on
something else — the fix comes with a test:

1. **Write the test first**, as the smallest input which shows the wrong
   behaviour, and check that it fails on the unfixed code. A test which
   passes before the fix is testing something else.
2. **Put it next to the code it covers**, in the `tests/` directory of the
   subpackage, in the test function which covers that feature or in a new
   one named after the symptom (`test_precision_is_restored`,
   `test_extraneous_roots_are_dropped`).
3. **Say what the bug was**, in one comment line above the assertion: what
   was returned before, and why it was wrong. The comment is what makes
   the test readable in a year.
4. **Check the property, not the printed form**, whenever the two differ:
   a solution is checked by substitution, a sum against its partial sums,
   a sign against a sample point. A test which pins a printed expression
   fails on every harmless change of form and gets weakened or deleted.
5. **A bug found in SymPy** (not in this package) gets an entry in the
   issue tracker and, if the package works around it, a test of the
   workaround with the SymPy behaviour quoted in the comment, so that the
   workaround can be removed when SymPy fixes it.

The randomised drivers in `benchmarks/` (`fuzz.py`, `solve_random.py`,
`logic_random.py`, `verify_random.py`, `qf_nra.py`) exist to find these
cases: they cross-check the results against independent oracles
(numerical evaluation, brute force search, substitution). When one of them
reports a failure, the fix and its unit test follow; the driver itself is
not a substitute for the test.

## Running the tests

```
pip install -e ".[test]"
python -m pytest
```

This runs the unit tests and the doctests of every module under
`sympy_extras/`. The examples in `README.md` and `docs/` are doctests too:

```
python -m pytest --doctest-glob='*.md' README.md docs
python -m mypy
python -m pyflakes sympy_extras conftest.py
```

Run everything before committing; a change is not done until the tests,
mypy and pyflakes pass. Run the documentation examples as a whole, not
page by page: stale examples hid a regression for a week. After a push,
look at the result of the continuous integration (`gh run list`): its
machines are slower, it runs with `SYMPY_EXTRAS_TIME_SCALE=3`
(`settings.time_scale` multiplies every time limit), and a test which only
fails there is a computation too close to its time limit; on a loaded
machine a test can fail for the same reason.

## Adding a new algorithm

1. Put it in the subpackage matching the SymPy module it extends, with
   tests in the neighbouring `tests/` directory.
2. Export the public names from the subpackage `__init__.py` and list them
   in `__all__`.
3. Add a page in `docs/` if the module needs more explanation than the
   docstrings, and a short section with an example to `README.md`.
4. Add an entry to `CHANGELOG.md` under the upcoming version.
5. State the provenance in the commit message (AI model used, original
   pull request if the code was ported from a SymPy pull request).

## Releasing

Releases are cut from `master` by pushing a tag `vX.Y.Z` that matches
`__version__` in `sympy_extras/__init__.py`. The `Release` workflow checks
the version, runs the tests, builds the sdist and wheel, publishes them to
PyPI with trusted publishing and creates a GitHub release. Bump the version
and the changelog in a commit before tagging; never publish a version that
was already published. See the `Releasing` section of `README.md`.

## Things not to do

- Do not call a SymPy routine which may not terminate (`integrate`,
  `dsolve`, `solve`, `simplify` on unbounded input, `checkodesol`) without
  a time limit: wrap it with `sympy_extras._timeout.attempt`, as the
  solvers do, so that every public function returns.
- Do not decide anything from a plain `evalf` of a closed form (a value
  against a quadrature, a difference at a sample point, a residual, a
  sign): evaluate it with `sympy_extras._numeric.reliable_value`, which
  writes the sums which are exactly zero 0 and returns `None` for a value
  whose branch a rounding error chooses (sympy-extras#65), and treat
  `None` as "nothing was checked".
- Benchmarks against external collections live in `benchmarks/`, download
  their data on first use into `benchmarks/.cache/` (never committed) and
  are not part of the test suite; the unit tests must stay fast.
- Do not change SymPy's behaviour from this package (no monkeypatching).
- Do not add dependencies beyond SymPy without a discussion.
- Do not delete or weaken a test to make the suite pass.
- Do not fix a bug without adding the test which fails on the unfixed code
  (see *Regression tests for every bug*).
- Do not silently drop the provenance of ported code, nor the
  `PORTED FROM SYMPY` notice of a file (keep it when the code is
  rewritten further; remove it only when nothing of the port remains).
- Do not add an unannotated function, an untyped container, a new `Any`,
  a `cast`, a `getattr` or a `# type: ignore`, and do not weaken the mypy
  configuration.
