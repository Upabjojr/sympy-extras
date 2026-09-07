# Changelog

All notable changes to this project are documented in this file.

The project is at version 0.x: there is **no guarantee of backwards
compatibility** between releases yet, and any release may rename, move or
remove public functions. Breaking changes are listed here when they happen.

## 0.0.1

First release.

### Added

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
  restricted to zero-dimensional ideals).
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
