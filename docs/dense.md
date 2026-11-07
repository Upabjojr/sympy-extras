# The standalone polynomial layer: typed domains and dense arithmetic

`sympy_extras.polys.domains`, `densebasic`, `densearith`, `densetools`,
`euclidtools` and `sqfreetools` are a reimplementation of the bottom of
`sympy.polys` written for static typing: the algorithms are generic over a
type variable `T` and a `Domain[T]` object carrying the arithmetic, so each
function corresponds to one generic function over a ring trait in a
statically typed language. Nothing is coerced silently and no element is
`Any`: `ZZ` works on `int`, `QQ` on `fractions.Fraction`, `GF(p)` on `int`
reduced by every operation, a polynomial ring on lists of coefficients, a
fraction field on reduced pairs. This is the first layer of the plan to
make `sympy-extras` independent of SymPy; the algorithms above it (Gröbner
bases, the CAD, the solvers) still run on SymPy's `Poly` until the next
layers (factorisation, root isolation, a `Poly` wrapper) exist.

## Domains

| domain | elements | notes |
|---|---|---|
| `ZZ` | `int` | truncated `quo`/`rem`, `gcdex` by the extended Euclidean algorithm |
| `QQ` | `Fraction` | `numer`, `denom` |
| `GF(p)` | `int` in `0..p-1` | inverses by `pow(a, -1, p)` |
| `PolynomialRing(K, 'x')` | `list[T]` | `K[x]`; `PolynomialRing(PolynomialRing(ZZ, 'y'), 'x')` is `ZZ[y][x]` |
| `FractionField(PolynomialRing(K, 'x'))` | `FracElement` | reduced pairs, canonical denominator |
| `AlgebraicField(minpoly, alpha)` | `list[Fraction]` | `QQ(alpha)`, reduced modulo the minimal polynomial; `norm`, `minimal_polynomial` of an element |

Every domain has `zero`, `one`, `add`, `sub`, `neg`, `mul`, `pow`,
`exquo` (exact division, `ValueError` otherwise), `quo`, `rem`, `invert`,
`gcd`, `lcm`, `gcdex`, `canonical_unit` (the unit making an element
canonical: positive over `ZZ`, monic over a field), `from_int`,
`convert(a, K)`, `to_expr`, `from_expr` (expressions of the typed core
`sympy_extras.core`); ordered domains add
`is_negative`, `abs`, `sign`.

```python
>>> from fractions import Fraction
>>> from sympy_extras.polys.domains import ZZ, QQ, GF
>>> from sympy_extras.polys.domains.polynomialring import PolynomialRing
>>> ZZ.gcdex(12, 18), QQ.invert(Fraction(-2, 3)), GF(7).invert(3)
((-1, 1, 6), Fraction(-3, 2), 5)
>>> R = PolynomialRing(ZZ, 'y')
>>> R.gcd([2, 0, -2], [4, 4]), R.to_expr([1, 0, -1])
([2, 2], y**2 - 1)

```

## Dense polynomials

A polynomial is a list of coefficients, highest degree first, without
leading zeros (`[]` is zero); a polynomial in several variables is a
polynomial over a polynomial ring, and the generic algorithms are then
the multivariate ones through the tower (the gcd of the coefficients is
the recursive gcd).

- `densebasic`: `dup_strip`, `dup_degree`, `dup_LC`, `dup_TC`,
  `dup_from_dict`, `dup_to_dict`, `dup_from_list`, `dup_convert`, ...
- `densearith`: `dup_add`, `dup_sub`, `dup_mul` (Karatsuba above degree
  32), `dup_sqr`, `dup_pow`, `dup_div`/`dup_quo`/`dup_rem` over a field,
  `dup_pdiv`/`dup_prem` (pseudo-division over a ring), `dup_exquo`, the
  ground operations and the norms.
- `densetools`: `dup_diff`, `dup_eval` (Horner), `dup_shift` (Taylor
  shift), `dup_compose`, `dup_content`, `dup_primitive`, `dup_monic`,
  `dup_clear_denoms`, `dup_trunc` (symmetric residues), `dup_integrate`,
  `dup_mirror`, `dup_scale`, `dup_sign_variations`.
- `euclidtools`: `dup_gcd` (Euclid over a field, the subresultant PRS of
  Brown over an integral domain with the gcd of the contents),
  `dup_qq_gcd` (through `ZZ`), `dup_lcm`, `dup_gcdex`, `dup_invert`,
  `dup_subresultants`, `dup_resultant` (the recursive identity on
  pseudo-remainders, with exact divisions in the domain),
  `dup_prs_resultant` (SymPy's subresultant PRS resultant, no gcds: the fast
  route over polynomial rings; SymPy's sign error for `deg f < deg g`
  corrected), `dup_discriminant`.
- `polytools.Poly` (ported from SymPy): a polynomial in generators over
  `ZZ`, `QQ`, `ZZ[a, ...]` or `QQ[a, ...]`, built from a core expression
  as SymPy builds it (generator order, domain), printed as SymPy prints
  it, with SymPy's normalisations: the gcd over a polynomial domain is
  `dmp_rr_prs_gcd` over `SparsePolynomialRing` (SymPy's `PolyElement` gcd,
  which is not associative over `QQ[a]`, so contents are folded in SymPy's
  order), factor lists sorted by degree, multiplicity, coefficients. Not
  ported yet: the domains `EX`, `RR`, fraction fields, `ZZ_I`, algebraic
  extensions; multivariate division and factorisation.
- `sqfreetools`: `dup_sqf_list` (Yun's algorithm in characteristic zero;
  over `GF(p)` with the `p`-th roots of the factors of multiplicity
  divisible by `p`), `dup_sqf_part`, `dup_sqf_p`.

```python
>>> from sympy_extras.polys.densearith import dup_mul, dup_pdiv
>>> from sympy_extras.polys.euclidtools import dup_gcd, dup_resultant
>>> from sympy_extras.polys.sqfreetools import dup_sqf_list
>>> dup_mul([1, 1], [1, -1], ZZ), dup_pdiv([1, 0, -1], [2, 1], ZZ)
([1, 0, -1], ([2, -1], [-3]))
>>> dup_gcd([2, 0, -2], [4, -8, 4], ZZ), dup_resultant([1, 0, -2], [1, 0, 0, -3], ZZ)
([2, -2], 1)
>>> dup_sqf_list([2, 0, -4, 0, 2], ZZ)
(2, [([1, 0, -1], 2)])
>>> S = PolynomialRing(R, 'x')
>>> dup_gcd(S.mul([[1], [1, 0]], [[1], [-1, 0]]), S.mul([[1], [1, 0]], [[1], [1, 0]]), R)
[[1], [1, 0]]

```

The last line is the gcd of `(x + y)(x - y)` and `(x + y)**2` in `ZZ[y][x]`.

## Factorisation

- `galoistools` (over `GF(p)`): powers modulo a polynomial, the Frobenius
  map through the precomputed powers of `x**p`, the distinct-degree
  factorisation (`gf_ddf`), the equal-degree splitting of Cantor and
  Zassenhaus (`gf_edf`; the trace map for `p = 2`), `gf_factor_sqf`,
  `gf_factor` (with the square-free decomposition first), Rabin's
  irreducibility test (`gf_irreducible_p`) and random irreducible
  polynomials.
- `factortools` (over `ZZ` and `QQ`): Zassenhaus' algorithm. A prime not
  dividing the leading coefficient modulo which the polynomial stays
  square-free is chosen among the first few (the one with the fewest
  modular factors), the modular factorisation is lifted by quadratic
  Hensel steps (`dup_zz_hensel_step`, the list of factors split in halves
  recursively in `dup_zz_hensel_lift`) above twice the Mignotte bound, and
  the factors are recombined by trial division of the primitive parts of
  `lc * prod f_i` over the subsets of increasing size. `dup_zz_factor`
  and `dup_qq_factor` return the content and the primitive factors with
  their multiplicities.
- Over a number field `QQ(alpha)`, Trager's algorithm (`dup_ext_factor`):
  a shift `x - s alpha` making the norm `N(f) = Res_alpha(m, f)`
  square-free, the norm factored over `QQ`, and each factor `N_i` giving
  the factor `gcd(f, N_i)` over the field; `AlgebraicField.minimal_polynomial`
  is the square-free part of the characteristic polynomial, itself a
  resultant.

```python
>>> from sympy_extras.polys.domains import GF
>>> from sympy_extras.polys.galoistools import gf_factor
>>> from sympy_extras.polys.factortools import dup_zz_factor
>>> gf_factor([1, 0, 0, 0, 4], GF(5))
(1, [([1, 1], 1), ([1, 2], 1), ([1, 3], 1), ([1, 4], 1)])
>>> dup_zz_factor([1, 0, 0, 0, -1], ZZ)
(1, [([1, -1], 1), ([1, 1], 1), ([1, 0, 1], 1)])

```

## Real roots

`rootisolation`: Descartes' rule with bisection (Collins–Akritas,
Rouillier–Zimmermann) isolates the real roots of a square-free polynomial
with rational coefficients (`dup_isolate_real_roots_sqf`; with
multiplicities through the square-free decomposition in
`dup_isolate_real_roots`): the positive roots are brought into `(0, 1)`
by a root bound (the smaller of Cauchy's and Fujiwara's), an interval is
kept when the sign variations of
`(x + 1)**n p(1/(x + 1))` are one, discarded when zero, bisected
otherwise, with the midpoint checked for an exact root. Each interval is
open with rational endpoints and holds one root, or is a point for a
rational root. Sturm sequences count the roots in an interval
(`dup_count_real_roots`) and `dup_refine_real_root` bisects an isolating
interval below a width.

```python
>>> from fractions import Fraction
>>> from sympy_extras.polys.rootisolation import dup_isolate_real_roots, dup_count_real_roots
>>> dup_isolate_real_roots([Fraction(1), Fraction(0), Fraction(-2)])
[((Fraction(-2, 1), Fraction(0, 1)), 1), ((Fraction(0, 1), Fraction(2, 1)), 1)]
>>> dup_isolate_real_roots([Fraction(1), Fraction(0), Fraction(-2)], Fraction(1, 100))
[((Fraction(-91, 64), Fraction(-181, 128)), 1), ((Fraction(181, 128), Fraction(91, 64)), 1)]
>>> dup_count_real_roots([Fraction(1), Fraction(0), Fraction(-2)], Fraction(0), Fraction(3))
1

```

## Verification

`polys/tests/test_factor.py` compares the factorisations over `GF(p)`
(`p = 2, 3, 5, 7, 11, 101`), `ZZ` and `QQ` with SymPy's `factor_list` on
random products, checks that every factor is irreducible and that the
products rebuild the input, and the Hensel lifting on hand-made cases.
`polys/tests/test_rootisolation.py` compares the isolating intervals,
multiplicities and counts with SymPy's `intervals` and `count_roots` on
random polynomials (with rational roots at dyadic points thrown in),
checks that every interval holds exactly one root by sign changes and
Sturm counts, and the Chebyshev and Wilkinson polynomials.
`polys/tests/test_dense.py` compares every operation with SymPy's on
random polynomials over `ZZ`, `QQ` and `GF(7)`, checks the defining
identities (pseudo-division, Bézout coefficients, inverses modulo a
polynomial, the product of the square-free factors) and the multivariate
gcd and resultant through the tower against `sympy.gcd` and
`sympy.resultant`. The resultant is also checked against the determinant
of the Sylvester matrix, which found a sign convention problem in SymPy:
`resultant(f, g)` returns the resultant of the higher-degree polynomial
first whatever the order of the arguments, so it is off by the sign
`(-1)**(deg f deg g)` when `deg f < deg g` and both degrees are odd
(`resultant(-7x - 3, -2x**3 + 6x**2 + x - 6)` is `1773` by the Sylvester
determinant, SymPy gives `-1773`).

## The polynomial class

`polyclasses.DensePoly(rep, domain, gen)` wraps a dense polynomial with
its domain and variable and exposes the algorithms as methods:
arithmetic operators, `div`/`quo`/`rem`/`exquo`/`pdiv`, `diff`,
`integrate`, `eval`, `shift`, `scale`, `compose`, `content`,
`primitive`, `monic`, `gcd`, `lcm`, `gcdex`, `invert`, `resultant`,
`discriminant`, `subresultants`, `sqf_list`, `factor_list`,
`is_irreducible`, `real_root_intervals`, `count_real_roots`, and the
conversions `from_expr`/`to_expr`/`convert`. Factorisation and real
roots are dispatched by the domain (`Domain.factor_dense`,
`isolate_real_roots_dense`), so a polynomial over a domain without them
raises `NotImplementedError` rather than silently converting. A
polynomial in several variables is a `DensePoly` over a polynomial ring
domain, and its conversions handle all the variables.

```python
>>> from sympy_extras.core import symbols
>>> x, y = symbols('x y')
>>> from sympy_extras.polys.polyclasses import DensePoly
>>> f = DensePoly.from_expr(x**4 - 1, x, ZZ)
>>> [g.to_expr() for g, _ in f.factor_list()[1]]
[x - 1, x + 1, x**2 + 1]
>>> g = DensePoly.from_expr(x**2*y - y**3, x, PolynomialRing(ZZ, 'y'))
>>> g.gcd(DensePoly.from_expr(x*y + y**2, x, PolynomialRing(ZZ, 'y'))).to_expr()
x*y + y**2

```

## Sparse multivariate polynomials

`rings.PolyRing(domain, names, order)` and `PolyElement` are the sparse
representation (exponent tuples to coefficients) with the monomial
orders `lex`, `grlex`, `grevlex`: leading monomials, terms and
coefficients under the order, arithmetic, `div` (the division algorithm
by a list of polynomials, the multivariate reduction), `rem`, `exquo`,
`monic`, `primitive`, derivatives, evaluation of one variable, and the
conversions to and from the expressions of the typed core. This is the
representation of F4 and of the strong Gröbner bases.

```python
>>> from sympy_extras.polys.rings import PolyRing, grevlex
>>> R = PolyRing(QQ, ['x', 'y'], grevlex)
>>> X, Y = R.gens
>>> f = X**2*Y - R.one
>>> str(f), f.LM, (X**2*Y + X).div([X*Y - R.one])[1].to_expr()
('x**2*y - 1', (2, 1), 2*x)

```

## Independence from SymPy

The domains, the dense and sparse polynomials and every algorithm above
import nothing from SymPy: elements convert to and from the expressions
of the typed core (`to_expr`, `from_expr`; `polyutils.dict_from_expr`
walks `Add`, `Mul` and integer `Pow` nodes and multiplies out with the
ring arithmetic, so no expansion of the expression is needed). The
conversions to SymPy objects are in `sympy_extras.polys.interop`
(`domain_to_sympy`, `dense_to_sympy`, `ring_to_sympy` and their
inverses, and the conversions of SymPy's own rings), the one module of
the layer which imports SymPy.

## Performance

The generic functions call the domain's methods for every coefficient
operation, which is what a statically typed translation makes free but
costs a Python call here. `ZZ` and `GF(p)` therefore provide *dense
kernels* (`Domain.has_dense_kernels`, `dense_mul`, `dense_div`) running
the inner loops on machine integers, the division over a field inverts
the leading coefficient once, and `x**n ± 1` is factored into cyclotomic
polynomials directly. With these the factorisation of random products of
degree 30 to 40 runs within two to three times SymPy's pure Python
implementation; `x**100 - 1` is immediate.

## Not done

Factorisation of multivariate polynomials (Wang), the recombination by lattice reduction (van Hoeij;
the subset search is exponential in the number of modular factors),
complex root isolation, the conversion of the algorithms above to this
layer (they still use SymPy's `Poly`); moving the Gröbner basis code onto
`rings.PolyElement` (it uses SymPy's); a fast
gcd over `ZZ` (the heuristic gcd) and over `GF(p)`.

## References

- K. O. Geddes, S. R. Czapor, G. Labahn, *Algorithms for Computer
  Algebra*, Kluwer, 1992, Chapters 2, 7 and 8.
- D. E. Knuth, *The Art of Computer Programming*, vol. 2, 3rd ed.,
  Addison-Wesley, 1998, Section 4.6.
- W. S. Brown, *The subresultant PRS algorithm*, ACM Transactions on
  Mathematical Software 4 (1978), 237-249.
- D. Y. Y. Yun, *On square-free decomposition algorithms*, SYMSAC '76.
- J. von zur Gathen, J. Gerhard, *Modern Computer Algebra*, 3rd ed.,
  Cambridge University Press, 2013, Chapters 14 and 15.
- D. G. Cantor, H. Zassenhaus, *A new algorithm for factoring polynomials
  over finite fields*, Mathematics of Computation 36 (1981), 587-592.
