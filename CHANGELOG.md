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

The CAD code was originally proposed to SymPy in the pull requests
[sympy/sympy#30422](https://github.com/sympy/sympy/pull/30422),
[sympy/sympy#30423](https://github.com/sympy/sympy/pull/30423),
[sympy/sympy#30424](https://github.com/sympy/sympy/pull/30424) and
[sympy/sympy#30425](https://github.com/sympy/sympy/pull/30425).
