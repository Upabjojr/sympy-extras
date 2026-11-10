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
- From ten modular factors on, the recombination is van Hoeij's
  (`dup_zz_van_hoeij`): the subset search costs `2**(r - 1)` trial
  divisions on a polynomial with `r` modular factors and few rational
  ones (the Swinnerton-Dyer polynomials, the resolvents and minimal
  polynomials of sums of roots), the knapsack lattice a few LLL
  reductions in dimension `r + 2`. The scaled power sums of the roots
  `S_k(g) = lc**k sum alpha**k` of the lifted factors are additive, and
  integers of bounded size on the rational factors; their middle `p`-adic
  digits make a lattice in which the 0-1 vectors of the rational factors
  are short, and the LLL-reduced basis vectors whose Gram–Schmidt length
  passes the bound are discarded until the rest is a partition of the
  modular factors, proved by trial division (rank one: irreducible).
  The Swinnerton-Dyer polynomial of degree 32 takes 0.04 s instead of
  2 s, those of degree 64 and 128 (out of reach of the subset search)
  0.2 s and 2 s.
- `lattice`: the LLL algorithm in de Weger's all-integer form (`lll`,
  `lll_reduction` with the Gram determinants, from which the
  Gram–Schmidt lengths and `short_part`, the vectors spanning everything
  below a length), and the Hermite normal form (`hermite_normal_form`,
  which also extracts a basis from dependent generators). No fractions
  and no floating point: entries of any size (SymPy's
  `DomainMatrix.lll` fails on entries of a few hundred bits). The Thue
  solver uses it.
  The same row operations carried along on an identity matrix give
  `integer_kernel` (a reduced basis of the integer solutions of
  `A x = 0`), `solve_integer_system` (all the integer solutions of
  `A x = b`, the particular one shortened by Babai's nearest plane
  algorithm) and `smith_normal_form` (the invariant factors).
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
>>> from sympy_extras.polys.lattice import lll
>>> lll([[1, 0, 0, 1000000], [0, 1, 0, 1414214], [0, 0, 1, 2828427]])[0]
[0, -2, 1, -1]

```

`domains.galoisfield.GaloisField(p, n)` is the field with `p**n` elements,
`GF(p)[a]/(m)` for the first monic irreducible `m` of degree `n` (or a
given one); `galoisfactor.gfq_factor` factors over it: the square-free
decomposition in characteristic `p` (the `p`-th roots of the coefficients
are their powers `q/p`), distinct degrees by `gcd(f, x**(q**d) - x)`,
equal degrees by Cantor and Zassenhaus (the trace map in characteristic
2).

```python
>>> from sympy_extras.polys.domains.galoisfield import GaloisField
>>> from sympy_extras.polys.galoisfactor import gfq_factor
>>> GF4 = GaloisField(2, 2)
>>> gfq_factor([[1], [1], [1]], GF4)             # x**2 + x + 1 = (x + a)(x + a + 1) over GF(4)
([1], [([[1], [1, 0]], 1), ([[1], [1, 1]], 1)])

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

## Complex roots

`complexroots`: the roots which are not real are isolated in rectangles
with rational corners by exact winding numbers. On each edge `z = z0 + t
w` of a rectangle, `c f(z) = u(t) + i v(t)` (`c` a constant making `u`
non-zero at the corners), and the winding number of `f` along the
boundary is minus half the sum of the Cauchy indices of `v/u` on
`[0, 1]`, which the sign variations of the signed remainder sequence of
`(u, v)` give at the two ends (Sturm); the last polynomial of the
sequence is `gcd(u, v)`, whose roots in `[0, 1]` are the roots of `f` on
the edge, so that a root on the boundary is detected, not miscounted
(`dup_count_complex_roots` raises `ValueError`). Integer arithmetic
throughout.

`dup_isolate_complex_roots_sqf` returns the rectangles of the roots above
the real axis of a square-free polynomial (the conjugate rectangles
isolate the others): the number of pairs is known from the real root
count, a strip around the real axis free of them is found by halving its
height, and the rectangle above it is bisected along its longer side, the
cut being moved when it meets a root. With `eps`, each rectangle is
refined by Newton's iteration in fixed point arithmetic from its centre,
and the small square around the last iterate is accepted when Rouché's
theorem on the Taylor expansion at its centre (or else its winding
number) proves that it contains the root; bisection is the fallback.
`dup_isolate_complex_roots` handles multiplicities (the rectangles of
different square-free factors are refined until disjoint),
`dup_isolate_all_roots_sqf` returns the real intervals and the
rectangles, `dup_refine_complex_root` refines one rectangle.

```python
>>> from sympy_extras.polys.complexroots import Rectangle, dup_count_complex_roots, dup_isolate_complex_roots_sqf
>>> f = [Fraction(1), Fraction(0), Fraction(0), Fraction(0), Fraction(4)]       # roots 1 + i, 1 - i, -1 + i, -1 - i
>>> dup_isolate_complex_roots_sqf(f)
[Rectangle(-5, 0, 5/8, 5), Rectangle(0, 5, 5/8, 5)]
>>> dup_isolate_complex_roots_sqf(f, Fraction(1, 1000))[1].contains(Fraction(1), Fraction(1))
True
>>> dup_count_complex_roots(f, Rectangle(Fraction(0), Fraction(2), Fraction(-2), Fraction(2)))
2

```

The Cauchy index on a whole line counts the roots on one side of it:
`dup_count_roots_right_of(f, a)` (real part larger than `a`),
`dup_hurwitz_stable_p` (the criterion of Routh and Hurwitz), and through
the map `w -> c + rho (w - 1)/(w + 1)` of a half-plane onto a disc,
`dup_count_roots_in_disc(f, center, radius)` and `dup_schur_stable_p`
(Schur–Cohn).

```python
>>> from sympy_extras.polys.complexroots import dup_count_roots_right_of, dup_count_roots_in_disc, dup_hurwitz_stable_p
>>> dup_count_roots_right_of(f), dup_count_roots_in_disc(f, (Fraction(1), Fraction(1)), Fraction(1, 2))
(2, 1)
>>> dup_hurwitz_stable_p([Fraction(1), Fraction(3), Fraction(3), Fraction(1)])
True

```

On random polynomials of degree 10, 20, 30 with coefficients below 100
the isolation takes 0.01, 0.1 and 1 s (SymPy's
`dup_isolate_complex_roots_sqf`: 0.07, 0.25, 1.7 s), and the refinement
to `10**-50` 0.01, 0.17 and 1.7 s in all (SymPy's refinement to
`10**-10`: 1.8 s and 21 s at degrees 10 and 20).

## Algebraic numbers

`numberfields.qqbar`: exact arithmetic in the field of all algebraic
numbers. An `Algebraic` is its minimal polynomial over `QQ` together with
a region isolating it among the roots (a rational point, an interval, a
rectangle). For a sum or a product, a resultant of the two minimal
polynomials vanishes at the result; it is factored over `ZZ`, and the
factor and the root are identified by enclosures: the operands are
refined, interval arithmetic gives a box for the result, and the roots
of each factor in the box are counted by winding numbers until one root
of one factor is left (real exactly when the box made symmetric about
the real axis still has one root). Equality, the order of the real
numbers and signs are decided exactly; `algebraic_roots(f)` lists the
roots of a polynomial, real ones first in increasing order.

```python
>>> from sympy_extras.polys.numberfields.qqbar import Algebraic, algebraic_roots
>>> sqrt2 = algebraic_roots([1, 0, -2])[1]
>>> sqrt3 = algebraic_roots([1, 0, -3])[1]
>>> sqrt2 + sqrt3
Algebraic([1, 0, -10, 0, 1], 3.14626437)
>>> 1/(sqrt3 - sqrt2) == sqrt2 + sqrt3, (1 + sqrt2)**2 == 3 + 2*sqrt2
(True, True)
>>> (5 + 2*(sqrt2*sqrt3)).sqrt() == sqrt2 + sqrt3          # a denesting, decided
True
>>> Algebraic.from_rational(-8).nth_root(3)                # the principal cube root
Algebraic([1, -2, 4], 1 + 1.732050808*I)
>>> a, b, c = algebraic_roots([1, 0, 0, -2])
>>> a + b + c == 0, b.conjugate() == c, (b*c).is_real
(True, True, True)

```

`numberfields.recognize.recognize_algebraic(re, im, digits, max_degree)`
goes from a decimal approximation to the exact number: a short vector of
the lattice of the powers of the approximation (LLL) is an integer
polynomial which nearly vanishes there, accepted when it is much shorter
than a number without relation would give, and its root nearest to the
approximation is returned.

```python
>>> from fractions import Fraction
>>> from sympy_extras.polys.numberfields.recognize import recognize_algebraic
>>> recognize_algebraic(Fraction(3146264369941972, 10**15), digits=15)
Algebraic([1, 0, -10, 0, 1], 3.14626437)
>>> recognize_algebraic(Fraction(3141592653589793, 10**15), digits=15) is None
True

```

`primitive_element(a, b)` returns `theta = a + c b` generating the field
of the two numbers, with `a` and `b` as polynomials in `theta` (the gcd
of `m_b(x)` and `m_a(theta - c x)` over `QQ(theta)` is `x - b`): the
entry to computing in one fixed field.

```python
>>> from sympy_extras.polys.numberfields.qqbar import primitive_element
>>> theta, p, q = primitive_element(sqrt2, sqrt3)
>>> theta.minpoly, p == [Fraction(1, 2), 0, Fraction(-9, 2), 0], theta.polynomial_value(q) == sqrt3
([1, 0, -10, 0, 1], True, True)

```

The resultant of a sum or product has the product of the degrees (a
power keeps the degree): this representation is convenient up to
degrees of a few dozen, not a replacement for computing in one fixed
number field (`AlgebraicField`).

`numberfields.basis`: `integral_basis(T)` and `field_discriminant(T)` for
the field `QQ(theta)`, `T(theta) = 0`, by the Round 2 algorithm: at each
prime whose square divides the discriminant of `T`, the order is replaced
by the multiplier ring of its `p`-radical (the kernel of a power of the
Frobenius map on `O/pO`) until it no longer grows (Pohst–Zassenhaus);
linear algebra modulo `p` and Hermite normal forms.

```python
>>> from sympy_extras.polys.numberfields.basis import integral_basis, field_discriminant
>>> integral_basis([1, -1, -2, -8])[2] == [Fraction(1, 2), Fraction(1, 2), 0]      # Dedekind's cubic: (theta**2 + theta)/2
True
>>> field_discriminant([1, -1, -2, -8]), field_discriminant([1, 0, -10, 0, 1])
(-503, 2304)

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

`fglm` changes the monomial order of the reduced Gröbner basis of a
zero-dimensional ideal on these rings (Faugère, Gianni, Lazard, Mora):
the monomials are visited in increasing order for the new order, the
normal form of each is computed from that of the monomial it is a
multiple of, and a linear dependence between the normal forms kept is an
element of the new basis. `groebner_by_fglm` computes the basis for
`grevlex` with F4 and converts it: Katsura-4 for `lex` in 0.14 s instead
of 1.1 s by F4 directly, the conversion of Katsura-5 in 1.5 s (SymPy's
`matrix_fglm`: 1.9 s).

```python
>>> from sympy_extras.polys.rings import grevlex, lex
>>> from sympy_extras.polys.f4 import f4
>>> from sympy_extras.polys.fglm import fglm
>>> graded = PolyRing(QQ, ['x', 'y'], grevlex)
>>> gx, gy = graded.gens
>>> basis = f4([gx*gx + gy*gy - graded.from_int(5), gx*gy - graded.from_int(2)], graded)
>>> [str(g) for g in fglm(basis, graded, lex)]
['x + 1/2*y**3 - 5/2*y', 'y**4 - 5*y**2 + 4']

```

`zerodim.solve_zero_dimensional(polys, ring)` puts these together into
exact solutions of systems with finitely many: the ideal is made radical
(the square-free parts of the univariate polynomials of the ideal, found
by linear algebra on normal forms), the lexicographic basis by FGLM is
brought to the shape `x_i - g_i(x_n), h(x_n)` by a change of the last
coordinate when needed, and each solution is the list of the algebraic
numbers `g_i(t)` for a root `t` of `h` (`Algebraic.polynomial_value`,
one resultant each).

```python
>>> from sympy_extras.polys.zerodim import solve_zero_dimensional
>>> plane = PolyRing(QQ, ['x', 'y'], lex)
>>> px, py = plane.gens
>>> points = solve_zero_dimensional([px*px + py*py - plane.one, py - px*px], plane)     # a circle and a parabola
>>> len(points), [p for p in points if p[0].is_real and p[0] > 0]
(4, [[Algebraic([1, 0, 1, 0, -1], 0.7861513778), Algebraic([1, 1, -1], 0.6180339887)]])

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
- H. S. Wilf, *A global bisection algorithm for computing the zeros of
  polynomials in the complex plane*, Journal of the ACM 25 (1978), 415-420.
- G. E. Collins, W. Krandick, *An efficient algorithm for infallible
  polynomial complex root isolation*, ISSAC '92, 189-194.
- M. van Hoeij, *Factoring polynomials and the knapsack problem*, Journal
  of Number Theory 95 (2002), 167-189.
- A. K. Lenstra, H. W. Lenstra, L. Lovász, *Factoring polynomials with
  rational coefficients*, Mathematische Annalen 261 (1982), 515-534.
- H. Cohen, *A Course in Computational Algebraic Number Theory*, Springer,
  1993, Sections 2.4 and 2.6.
