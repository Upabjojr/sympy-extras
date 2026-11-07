# Polynomial ideals and Gröbner bases

Modules: `sympy_extras.polys.ideals`, `sympy_extras.polys.groebnerwalk`,
`sympy_extras.polys.modulargroebner`, `sympy_extras.polys.orderings`,
`sympy_extras.polys.hermite`

## What SymPy has and what is added here

SymPy computes reduced Gröbner bases with the Buchberger and F5B algorithms
(`sympy.groebner`), reduces polynomials modulo a basis, tests membership,
decides whether an ideal is zero-dimensional and converts bases between
orders with FGLM, for zero-dimensional ideals only. Its ideal class
(`sympy.polys.agca.ideals`) implements sums, products, intersections and
quotients and raises `NotImplementedError` for saturation, radicals,
primality and maximality tests, height and depth. Product orders exist in
`sympy.polys.orderings` but cannot be used with `groebner` (they are not
hashable).

This package adds, on top of SymPy's Gröbner basis engine:

- hashable **block orders** and **weight orders** usable with
  `sympy.groebner` and with the ring level functions;
- an `Ideal` class with **elimination ideals**, intersections, quotients,
  **saturations** and **radical membership**;
- the **Krull dimension**, the **Hilbert series**, the **Hilbert polynomial**
  and the **degree**, from the leading term ideal, the numerator of the
  series by **Bigatti's algorithm** (`sympy_extras.polys.hilbert`);
- for zero-dimensional ideals: the standard monomials, the dimension of the
  quotient algebra, multiplication matrices, the univariate polynomial in
  each variable, the **radical** (Seidenberg's lemma), the test for
  **maximal** ideals and the **rational univariate representation**
  (Rouillier, `sympy_extras.polys.rur`);
- for zero-dimensional ideals over the rationals: the number of **real
  solutions** and the **signs** of other polynomials at them, counted
  without solving, by Hermite's quadratic form and sign determination;
- in any dimension, over the rationals: the **radical**, its
  **equidimensional parts**, the **minimal primes**, the **height** and
  the tests for **radical** and **prime** ideals, through triangular
  decompositions into squarefree regular chains
  ([regularchains.md](regularchains.md)), and the **primary
  decomposition**, the **associated primes** (embedded ones included) and
  the test for **primary** ideals (Gianni–Trager–Zacharias);
- the **Gröbner walk**, converting a Gröbner basis between orders for
  ideals of any dimension (`change_order` uses FGLM when the ideal is
  zero-dimensional and the walk otherwise);
- **Faugère's F4**, the Gröbner basis algorithm which reduces many
  S-polynomials at once by one row reduction of a sparse matrix, over any
  field; `Ideal` and the modular algorithm compute their bases with it;
- **modular Gröbner bases** over the rationals, proven, which `Ideal` uses
  for the bases the direct computation does not finish in a quarter of a
  second;
- **strong Gröbner bases over the integers** (Kandri-Rody–Kapur,
  Becker–Weispfenning, Lichtblau), which decide membership in ideals of
  `ZZ[x]`; SymPy's `groebner` with `domain=ZZ` returns the basis over the
  rationals.

Not done here: the depth,
decompositions over other fields than the rationals (algebraic numbers,
finite fields), F5, and `Ideal` over the integers (it converts the
coefficients to the rationals).

## Examples

```python
>>> from sympy.abc import x, y, z, t, d
>>> from sympy_extras.polys.ideals import Ideal
>>> I = Ideal([x*z - y**2, x**2 - y*z], x, y, z)
>>> I.groebner_basis()
[Poly(x**2 - y*z, x, y, z, domain='QQ'), Poly(-x*z + y**2, x, y, z, domain='QQ')]
>>> I.contains(x**3 - z**3), I.reduce(x**3)
(False, x*y*z)
>>> I.dimension(), I.degree()
(1, 4)
>>> I.hilbert_series(t)
(t**2 + 2*t + 1)/(1 - t)
>>> I.eliminate([x])
Ideal([y**4 - y*z**3], y, z)

```

The variety of `I` is the union of three lines through the origin (on
which `x**3 = y**3`) and the `z` axis (where `x = y = 0`), four lines in
all, hence degree 4. Saturating by `y` removes the axis:

```python
>>> lines = I.saturate(Ideal([y], x, y, z))
>>> lines
Ideal([x**2 - y*z, x*y - z**2, -x*z + y**2], x, y, z)
>>> lines.degree(), lines.hilbert_polynomial(d)
(3, 3)
>>> I.quotient(lines)
Ideal([x, y], x, y, z)
>>> Ideal([x], x, y).intersect(Ideal([y], x, y))
Ideal([x*y], x, y)
>>> Ideal([x**2, y**3], x, y).radical_contains(x*y)
True

```

Zero-dimensional ideals:

```python
>>> J = Ideal([x**2 + y**2 - 1, x - y**2], x, y)
>>> J.vector_space_dimension(), J.standard_monomials()
(4, [1, y, x, x*y])
>>> J.univariate(y)
Poly(y**4 + y**2 - 1, y, domain='QQ')
>>> J.minimal_polynomial(x + y, t)
Poly(t**4 + 2*t**3 - 6*t - 1, t, domain='QQ')
>>> J.multiplication_matrix(y)
Matrix([[0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 0]])
>>> J.is_radical(), J.is_maximal()
(True, True)
>>> Ideal([x**2, y**2 - 2*y + 1], x, y).radical()
Ideal([x, y - 1], x, y)

```

## Counting real solutions

For a zero-dimensional ideal `I` over the rationals and a polynomial `g`,
the quadratic form `(p, q) -> Tr(g*p*q)` on the quotient algebra
`Q[x]/I`, `Tr` the trace of the multiplication matrix, is *Hermite's
quadratic form*. Its signature is the number of distinct real solutions
at which `g > 0` minus the number at which `g < 0` (the *Tarski query* of
`g`), and its rank the number of distinct complex solutions at which
`g != 0` (Pedersen, Roy and Szpirglas; Basu, Pollack and Roy, Theorem
4.102). The multiplicities do not count. With `g = 1` it counts the real
solutions; the queries of `1`, `g` and `g**2` give the numbers of real
solutions at which `g` is zero, positive and negative.

```python
>>> from sympy.abc import x, y
>>> from sympy_extras.polys.hermite import (count_real_solutions,
...     count_complex_solutions, real_sign_counts, tarski_query)
>>> count_real_solutions([x**2 + y**2 - 1, x - y], x, y)
2
>>> count_real_solutions([x**2 + y**2 - 1, x + y - 2], x, y), count_complex_solutions([x**2 + y**2 - 1, x + y - 2], x, y)
(0, 2)
>>> count_real_solutions([(x - 1)**3*(x + 2), y**2], x, y)
2
>>> real_sign_counts([x**3 - x, y - x**2], x - y, x, y)
SignCounts(zero=2, positive=0, negative=1)
>>> tarski_query([x**3 - x, y - x**2], 2*y - 1, x, y)
1

```

For several polynomials, `sign_determination` gives every realizable sign
condition with its number of real solutions, by the scheme of Ben-Or,
Kozen and Reif (Basu–Pollack–Roy, Algorithm 10.11): the polynomials are
added one at a time, the Tarski queries of the products of their powers
are computed for at most three times as many products as there are
realizable conditions so far, and the conditions of count zero are
dropped.

```python
>>> from sympy_extras.polys.hermite import sign_determination
>>> sign_determination([x**2 - 1, y**2 - 1], [x + y, x], x, y)
{(-1, -1): 1, (0, -1): 1, (0, 1): 1, (1, 1): 1}

```

The matrices are exact over the rationals and the signature is found by
Lagrange's reduction of the symmetric matrix to a diagonal one by
congruence; no floating point number is involved. `decide_zero_dimensional`
decides whether a conjunction of equations with finitely many solutions
and of inequalities and inequations has a real solution, and
`satisfiable` uses it (for two or more variables known to be real, within
two seconds) to answer `False` before building a cylindrical algebraic
decomposition, which must visit every cell to prove that there is none.

```python
>>> from sympy import Eq
>>> from sympy_extras.polys.hermite import decide_zero_dimensional
>>> decide_zero_dimensional(Eq(x**2 + y**2, 1) & Eq(x*y, 1), [x, y])
False

```

## The rational univariate representation

For a zero-dimensional ideal over the rationals, Rouillier's rational
univariate representation writes every zero as a rational function of one
algebraic number: a linear form `u` which separates the zeros, the
squarefree polynomial `f(t)` of its values, and the coordinates
`x_i = g_i(t)/g_1(t)` at each root. Everything comes from traces on the
quotient algebra: the power sums `Tr(u^j)` give the characteristic
polynomial of `u` by Newton's identities (its squarefree part is `f`), the
traces `Tr(x_i u^j)` give the numerators through the Horner polynomials of
`f`, and `u` is separating exactly when the degree of `f` equals the number
of distinct zeros, the rank of Hermite's quadratic form. The candidates
are the first forms of `separating_forms`, then random forms, each tested
modulo a large prime (the squarefree degree can only drop modulo `p`, so
equality with the number of zeros is a proof); the exact computation is
done once, for the form found. Cyclic-5 (70 zeros, whose separating form
is the 124th in the enumeration) takes 4 s. SymPy's `solve_poly_system`
needs a lexicographic basis in shape position and nests `CRootOf` objects;
the representation is one algebraic number per zero, the real zeros are
the real roots of `f`, and the multiplicities are those of the roots of
the characteristic polynomial.

```python
>>> from sympy.abc import x, y, t
>>> from sympy_extras.polys.rur import rur
>>> R = rur([x**2 + y**2 - 1, x*y], x, y, t=t)
>>> R.form, R.polynomial
(x + 2*y, Poly(t**4 - 5*t**2 + 4, t, domain='QQ'))
>>> R.denominator, R.numerators
(Poly(4*t**3 - 10*t, t, domain='QQ'), [Poly(2*t**2 - 8, t, domain='QQ'), Poly(4*t**2 - 4, t, domain='QQ')])
>>> R.polynomials()
[Poly(-1/3*t**3 + 4/3*t, t, domain='QQ'), Poly(1/6*t**3 - 1/6*t, t, domain='QQ')]
>>> R.solutions()
[{x: 0, y: -1}, {x: -1, y: 0}, {x: 1, y: 0}, {x: 0, y: 1}]
>>> Ideal([x**2 - 2, y - x**3], x, y).rur(t).solutions()
[{x: -sqrt(2), y: -2*sqrt(2)}, {x: sqrt(2), y: 2*sqrt(2)}]

```

`polynomials()` gives the coordinates as polynomials in `t` (the
numerators times the inverse of the denominator modulo `f`, the form of a
lexicographic basis in shape position), `solutions(real=False)` the zeros
with `CRootOf` roots or radicals, `multiplicity(root)` the multiplicity of
a zero. The representation is checked on random systems by substituting
the coordinates into the generators modulo `f`, against Hermite's counts
of complex and real zeros, and against `solve_poly_system`.

## Radicals and prime components in any dimension

The zeros of an ideal are the union of the closures of the quasi-components
of regular chains (a triangular decomposition in the sense of Kalkbrener).
The chains of this package are squarefree, their saturated ideals are then
radical and unmixed, and the radical of the ideal is their intersection;
the parts of each dimension, cleared of what lies in a part of greater
dimension, are the equidimensional parts. No Gröbner basis of the ideal
itself is needed.

```python
>>> K = Ideal([x**2*z - y**2*z, x**3*y - x*y**3, z**2*(x - y)], x, y, z)
>>> K.radical()
Ideal([x**3*y - x*y**3, x*z - y*z], x, y, z)
>>> K.equidimensional_parts()
[Ideal([x - y], x, y, z), Ideal([x**2*y + x*y**2, z], x, y, z)]
>>> K.minimal_primes()
[Ideal([x - y], x, y, z), Ideal([x + y, z], x, y, z), Ideal([x, z], x, y, z), Ideal([y, z], x, y, z)]
>>> K.is_radical(), K.is_prime(), K.height()
(False, False, 1)

```

The prime components of the saturated ideal of a chain with free variables
`u` are found in dimension zero over `Q(u)`, without a Gröbner basis over
that field: a linear form in the other variables which separates the zeros
is found among `x1 + c2*x2 + c3*x3 + ...` with small integer coefficients
(`separating_forms`, by increasing height, up to a height at which one of
them is sure to separate the zeros), its minimal polynomial over
`Q[u]` is the generator of an elimination ideal, and to each irreducible
factor answers the prime which is the saturation by the other factors.
The primes are those over the rationals: `x**2 - 2*y**2` is one.

```python
>>> Ideal([x**2 - z, y**2 - z], x, y, z).minimal_primes()
[Ideal([y**2 - z, x + y], x, y, z), Ideal([y**2 - z, x - y], x, y, z)]
>>> Ideal([x**2 - 2*y**2], x, y).is_prime()
True

```

## Primary decomposition

`primary_decomposition()` returns an irredundant primary decomposition as
pairs `(Q, P)` of a primary ideal and its prime, by decreasing dimension;
`associated_primes()` the primes, the embedded ones included, and
`is_primary()` tests an ideal. The line `x = 0` of the plane with an
embedded point at the origin, and a plane with an embedded line and an
embedded point on it:

```python
>>> Ideal([x**2, x*y], x, y).primary_decomposition()
[(Ideal([x], x, y), Ideal([x], x, y)), (Ideal([x**2, y], x, y), Ideal([x, y], x, y))]
>>> L = Ideal([x], x, y, z).intersect(Ideal([x**2, y], x, y, z)).intersect(Ideal([x**3, y**3, z], x, y, z))
>>> L.associated_primes()
[Ideal([x], x, y, z), Ideal([x, y], x, y, z), Ideal([x, y, z], x, y, z)]
>>> L.minimal_primes()
[Ideal([x], x, y, z)]
>>> L.is_primary(), Ideal([x**2, y**3], x, y).is_primary()
(False, True)

```

The algorithm is the reduction of Gianni, Trager and Zacharias to
dimension zero. For a maximal independent set `u` of variables (as many as
the dimension, with `I ∩ K[u] = 0`, read off the leading monomials), the
extension of `I` to `K(u)[x]` has dimension zero; its contraction back to
`K[x]` is the saturation `I : h**oo` by the leading coefficients `h` in
`K[u]` of a Gröbner basis for a block order with `x > u`, and it is the
intersection of the primary components whose primes do not meet `K[u]`,
all of them minimal. With `I : h**m = I : h**oo`, the ideal is `(I :
h**oo) ∩ (I + (h**m))`, and the decomposition goes on with the larger
ideal `I + (h**m)`, where the embedded components are found. The
contraction is split into its primary components by separators, as in
Shimoyama and Yokoyama's algorithm rather than by a decomposition over
`K(u)`: its minimal primes `P_i` are known (from the regular chains), and
saturating by a polynomial which lies in the other primes and not in
`P_i` leaves the `P_i`-primary component. The components are merged by
prime, and those which contain the intersection of the others are
dropped.

The prime of the monomial curve `(t**3, t**4, t**5)` has a square with an
embedded component at the origin; its primary component is the symbolic
square, which contains a quintic not in `P**2`:

```python
>>> P = Ideal([x - t**3, y - t**4, z - t**5], t, x, y, z).eliminate([t])
>>> (Q, prime), (E, M) = (P**2).primary_decomposition()
>>> prime == P, M
(True, Ideal([x, y, z], x, y, z))
>>> f = x**5 - 3*x**2*y*z + x*y**3 + z**3
>>> Q.contains(f), (P**2).contains(f)
(True, False)

```

The decompositions are checked, in the tests, by computations which do
not use the algorithm: the intersection of the components is the ideal,
each component has the prime as its radical and is not changed by
saturations by polynomials outside it, no component can be left out, and
the minimal ones among the primes are the minimal primes of the regular
chains; on textbook examples (thirteen with embedded primes) and on random
intersections and products of ideals which are primary by construction
(powers of ideals of linear forms, complete intersections with a prime
radical, over rational and irrational points).

The twisted cubic in `P^3` has the classical invariants (Macaulay2's
documentation examples):

```python
>>> from sympy.abc import w
>>> C = Ideal([x*z - y**2, y*w - z**2, x*w - y*z], x, y, z, w)
>>> C.dimension(), C.degree(), C.hilbert_series(t), C.hilbert_polynomial(d)
(2, 3, (2*t + 1)/(1 - t)**2, 3*d + 1)

```

Changing the order (the walk for the positive-dimensional `I`, FGLM for
`J`):

```python
>>> I.change_order('lex')
[Poly(x**2 - y*z, x, y, z, domain='QQ'), Poly(x*y**2 - y*z**2, x, y, z, domain='QQ'), Poly(x*z - y**2, x, y, z, domain='QQ'), Poly(y**4 - y*z**3, x, y, z, domain='QQ')]
>>> J.change_order('lex')
[Poly(x - y**2, x, y, domain='QQ'), Poly(y**4 + y**2 - 1, x, y, domain='QQ')]

```

Orders with SymPy's own `groebner`:

```python
>>> from sympy import groebner
>>> from sympy_extras.polys.orderings import BlockOrder, WeightOrder
>>> list(groebner([t - x**2, t - y], t, x, y, order=BlockOrder([('grevlex', 1), ('grevlex', 2)])))
[t - y, x**2 - y]
>>> list(groebner([x**2 - y, x*y - 1], x, y, order=WeightOrder((1, 3), 'lex')))
[x**3 - 1, -x**2 + y]

```

## The Gröbner walk

`groebner_walk(G, ring, target)` converts the reduced Gröbner basis `G` of
a SymPy `PolyRing` (whose order is the source order) to the target order.
The source and target orders are represented by weight vectors; along the
segment between them, at every point where the ordering of the monomials
of the basis changes, the initial forms of the basis elements are computed,
a Gröbner basis of the ideal they generate is computed by a Buchberger
algorithm which keeps the representations (`extended_groebner`), and those
representations lift it to a Gröbner basis of the ideal for the new weight
order. The initial forms are small, so every step is cheap compared to a
direct computation for the target order.

## F4

`f4(polys, ring)` in `sympy_extras.polys.f4` returns the reduced Gröbner
basis of polynomials of a typed `sympy_extras.polys.rings.PolyRing` over a
field (`QQ`, `GF(p)`, an algebraic number field of
`sympy_extras.polys.domains.algebraicfield`) for the order of the ring,
sorted by decreasing leading monomial. It does not use SymPy: the matrix
reduction is a sparse Gauss–Jordan elimination of its own, fraction-free
over the integers for `QQ`. `Ideal` and the modular algorithm still build
SymPy rings and reach it through `direct_groebner`, which converts at the
boundary (`sympy_extras.polys.interop`); the results equal SymPy's
`groebner`, the reduced basis being unique.

```python
>>> from sympy_extras.polys.domains import QQ as TQQ, GF as TGF
>>> from sympy_extras.polys.rings import PolyRing, grevlex as tgrevlex, lex as tlex
>>> from sympy_extras.polys.f4 import f4
>>> T = PolyRing(TQQ, ['x', 'y', 'z'], tgrevlex)
>>> from sympy_extras.polys.interop import ring_from_sympy
>>> F = [ring_from_sympy(T, e) for e in (x**2 + y**2 + z**2 - 1, x + y + z, x*y*z - 1)]
>>> [str(g) for g in f4(F, T)]
['z**3 - 1/2*z - 1', 'y**2 + y*z + z**2 - 1/2', 'x + y + z']
>>> P = PolyRing(TGF(7), ['x', 'y'], tlex)
>>> [str(g) for g in f4([ring_from_sympy(P, x**2 - y), ring_from_sympy(P, 3*y**2 - x)], P)]
['x + 4*y**2', 'y**4 + 3*y']

```

Buchberger's algorithm, SymPy's, reduces the S-polynomials one at a time,
and every reduction multiplies the same basis elements by the same
monomials again. Faugère's F4 selects a set of critical pairs, collects
the two halves of each S-polynomial and, by *symbolic preprocessing*, a
reducer for every monomial which appears (the basis element with the
smallest leading monomial dividing it, multiplied up), and writes them as
a sparse matrix, one row per polynomial and one column per monomial in
decreasing order, whose reduced row echelon form
(`sympy.polys.matrices.sdm.sdm_irref`) is computed once. The rows whose
leading monomial is not among the leading monomials of the rows written
down are the new basis elements; the critical pairs are updated with the
criteria of Gebauer and Möller, and the basis is interreduced at the end.

The selection of the pairs decides the running time. For the graded
orders (`grevlex`, `grlex`) it is Faugère's normal strategy, all the pairs
whose lcm has the lowest total degree, which builds the Macaulay matrix of
each degree; for the other orders (`lex`, the elimination and weight
orders) the lowest total degree says little about the position in the
order, and the normal strategy built, on Katsura-4 for `lex`, matrices of
hundreds of columns which yielded three new rows each and had not finished
after 40 s where SymPy takes 2.5 s; selecting the 16 pairs with the
smallest lcms in the order instead takes 0.3 s (4 or 32 at a time: 6.4 s
and 0.5 s; the same 16 restricted to the lowest degree: not finished).
Whether an order is graded is decided by comparing the squares of the
variables with the variables.

Timings in seconds of `f4` and of SymPy's `groebner` on the same ring
elements (SymPy 1.14, pure Python ground types, Python 3.12, one run per
entry on a shared machine, a limit of 120 s):

| system    | field       | order     | typed `f4` | `f4` on SymPy's elimination | SymPy |
|-----------|-------------|-----------|-----------:|-------:|------:|
| Katsura-4 | `QQ`        | `grevlex` |       0.06 |   0.03 |  0.06 |
| Katsura-4 | `GF(32003)` | `lex`     |       0.15 |   0.30 |  1.65 |
| Katsura-4 | `QQ`        | `lex`     |       1.82 |   2.38 | 28.31 |
| Katsura-5 | `QQ`        | `grevlex` |       0.12 |   0.26 |  0.58 |
| Katsura-5 | `QQ`        | `lex`     |      > 120 |  > 120 | > 120 |
| cyclic-5  | `QQ`        | `grevlex` |       0.16 |   0.09 |  0.26 |
| cyclic-5  | `QQ`        | `lex`     |       1.25 |   2.33 | 35.74 |
| cyclic-6  | `QQ`        | `grevlex` |       1.27 |   4.49 | 102.6 |
| Katsura-6 | `QQ`        | `grevlex` |       1.01 |   8.61 | 17.26 |
| Katsura-7 | `GF(32003)` | `grevlex` |       3.09 |   56.5 | > 120 |
| cyclic-7  | `GF(32003)` | `grevlex` |       88.1 |  > 120 | > 120 |

The typed version (2026-11-07) has its own elimination: the multiples of
basis elements are the fixed pivots of their columns, the S-polynomial
halves are reduced by them only, and the small remaining block by
Gauss–Jordan, fraction-free over the integers for `QQ` (each step divided
by the gcd of the two multipliers, the content removed once per row).
Pivoting on whatever row comes first instead let rows with large
coefficients become pivots (Katsura-4 in `lex`: over 80 s for one matrix),
and the forward elimination alone, without the back substitution, let the
coefficients of the new elements grow to thousands of digits.

On 400 random ideals in two to four variables (one to four polynomials of
degree up to three, over `QQ`, `GF(7)` and `GF(32003)`, the three orders)
the two bases were equal every time. `settings.groebner_algorithm`
(`'f4'`, or `'buchberger'` for SymPy's) chooses the algorithm of every
direct computation of the package, those of the images of the modular
algorithm included: the modular basis of cyclic-6 for `grevlex` took 10 s
with F4 images and 56 s with Buchberger images.

## Hilbert series by Bigatti's algorithm

The Hilbert series of `K[x]/M` for a monomial ideal `M` is `N(t)/(1-t)^n`,
and the numerator of any homogeneous ideal is that of its leading term
ideal for a graded order. The textbook recursion
`N(M' + (m)) = N(M') - t^deg(m) N(M' : m)` applied to one generator at a time
is exponential in the number of generators: 300 generators in 12 variables
took 28 s, 200 in 10 variables 12 s. Bigatti's algorithm applies the same
recursion to a *pivot* `p = x_j^e` (`x_j` the variable occurring in the
most generators, `e` the median of its exponents), so that both `M + (p)` and
`M : p` lose many generators at once, and stops at the ideals with a
formula: generators in disjoint sets of variables multiply the numerators
(a complete intersection of pure powers `x_i^{a_i}` gives
`prod (1 - t^{a_i})`), and pure powers plus one monomial `m` give
`prod (1 - t^{a_i}) - t^deg(m) prod (1 - t^{a_i - m_i})`. The same ideals
take 0.07 s and 0.04 s; 800 generators in 15 variables 0.7 s.
`hilbert_numerator`, `hilbert_series` (reduced, with the Krull dimension as
the exponent of `1 - t`) and `minimal_generators` are in
`sympy_extras.polys.hilbert`; `Ideal.hilbert_series`, `dimension`,
`degree` and `hilbert_polynomial` use them.

```python
>>> from sympy.abc import t
>>> from sympy_extras.polys.hilbert import hilbert_numerator, hilbert_series
>>> hilbert_numerator([(2, 0, 0), (0, 3, 0), (0, 0, 5)], t)
Poly(-t**10 + t**8 + t**7 - t**3 - t**2 + 1, t, domain='ZZ')
>>> hilbert_series([(1, 0, 1, 0), (0, 1, 0, 1)], 4, t)
(t**2 + 2*t + 1)/(1 - t)**2

```

The second example is the Stanley–Reisner ring of the boundary of a
square (the non-faces are the diagonals), whose `h`-vector `(1, 2, 1)` is the
numerator. The algorithm is checked against the textbook recursion on 150
random monomial ideals and against the count of standard monomials degree
by degree.

## Strong Gröbner bases over the integers

A Gröbner basis over a field decides ideal membership because every
leading term in the ideal is a multiple of the leading term of a basis
element. Over `ZZ` the coefficients count: `<2x, 3y>` contains
`xy = 3 xy - 2 xy` but neither `x` nor `y`, and SymPy's `groebner` with
`domain=ZZ` returns `[x, y]`, the basis over the rationals, which answers
membership questions over the integers wrongly. A *strong* Gröbner basis
`G` of an ideal of `ZZ[x]` (Kandri-Rody and Kapur; the D-Gröbner bases of
Becker and Weispfenning, chapter 10; Lichtblau) has, for every nonzero
`f` of the ideal, an element whose leading term divides that of `f`,
coefficient included; `f` is in the ideal exactly when it reduces to zero
by `G` (`strong_reduce`, with Euclidean division of the coefficients).
`strong_groebner(polys, ring)` in `sympy_extras.polys.stronggroebner`
computes the reduced strong Gröbner basis (unique for the order and the
remainder convention: positive leading coefficients, remainders in
`[0, |lc|)`) by Buchberger's algorithm with the S-polynomials (leading
terms brought to their lcm, coefficients included) and the G-polynomials
(a Bézout combination whose leading coefficient is the gcd of the two
leading coefficients); `is_strong_groebner` checks the definition.

```python
>>> from sympy_extras.polys.domains import ZZ as TZZ
>>> from sympy_extras.polys.stronggroebner import strong_groebner, strong_reduce, is_strong_groebner
>>> Z = PolyRing(TZZ, ['x', 'y'], tlex)
>>> [str(g) for g in strong_groebner([ring_from_sympy(Z, 2*x), ring_from_sympy(Z, 3*y)], Z)]
['x*y', '2*x', '3*y']
>>> [str(g) for g in strong_groebner([4, ring_from_sympy(Z, 2*x + 1)], Z)]   # 2x - 1, then 2, then 1 are in the ideal
['1']
>>> G = strong_groebner([ring_from_sympy(Z, 2*x**2 + y), ring_from_sympy(Z, 3*x*y + 1)], Z)
>>> [str(g) for g in G]
['x*y + 3*y**3 + 1', '2*x - 3*y**2', '9*y**3 + 2']
>>> str(strong_reduce(ring_from_sympy(Z, 2*x - 3*y**2), G)), str(strong_reduce(ring_from_sympy(Z, x), G)), is_strong_groebner(G)
('0', 'x', True)

```

The strong bases run on the typed rings over `ZZ`, without SymPy.

## Modular Gröbner bases

SymPy's Buchberger algorithm over `QQ` spends its time on the intermediate
coefficients: the lexicographic basis of Katsura-4 has coefficients of 450
bits and the polynomials met on the way are far larger.
`modular_groebner(polys, ring)` returns the same reduced Gröbner basis as
`sympy.polys.groebnertools.groebner(polys, ring)` (monic over `QQ`,
primitive over `ZZ`, sorted by decreasing leading monomials), computed from
its images modulo primes after E. Arnold:

```python
>>> from sympy import QQ, ring, lex
>>> from sympy.polys.groebnertools import groebner
>>> from sympy_extras.polys.modulargroebner import modular_groebner, ModularTrace
>>> R, u, v, w = ring("u,v,w", QQ, lex)
>>> F = [u**2 + v**2 + w**2 - 1, u*v - w/3, u + v - 2*w]
>>> modular_groebner(F, R)
[u + v - 2*w, v**2 - 2*v*w + 1/3*w, w**2 - 2/15*w - 1/5]
>>> _ == groebner(F, R)
True

```

The steps (the module docstring has the proofs):

1. the denominators are cleared and the input is **homogenised** with a
   new variable; the order becomes "total degree, then the given order on
   the old variables", for which setting the new variable to 1 in a Gröbner
   basis gives a Gröbner basis of the input for the given order;
2. the reduced Gröbner basis modulo a prime is computed by SymPy's own
   `groebner` over `GF(p)`, for primes which divide no leading coefficient
   of the input;
3. **unlucky primes** are detected by comparing the leading monomials of
   the images, as Arnold does: degree by degree, the image with fewer
   leading monomials (a larger Hilbert function) is unlucky, and with
   equally many the image whose leading monomials are smaller is; only
   the images with the best leading monomials are kept;
4. the coefficients are combined by the **Chinese remainder theorem** and
   recovered by **rational reconstruction** (Wang's algorithm,
   `rational_reconstruction`), after every prime; the candidate is
   verified once the next image agrees with it, or at once if all its
   coefficients are below the square root of the modulus by 16 bits more
   than needed;
5. the **verification** is what makes the result proven: every input
   polynomial reduces to zero modulo the candidate, and the candidate is a
   Gröbner basis (the S-polynomials of the critical pairs left by the
   criteria of Gebauer and Möller reduce to zero; all this without
   fractions, on integer multiples). By Arnold's theorem 7.1 a homogeneous
   candidate with the leading monomials of one image which passes both
   tests is the Gröbner basis of the ideal: the Hilbert function of the
   ideal is at most the one of its image, which is the one of the
   candidate, which generates a larger ideal. If a test fails, more primes
   are used;
6. the new variable is set to 1 and the basis is reduced.

Why homogenise. **Without homogeneity the two tests prove less**: that the
candidate is a Gröbner basis of an ideal which *contains* the input. For
`x*(z + 210*y + 1), x*(z + 420*y + 2)` with `x > z > y`, whose leading
coefficients are 1, the ideal modulo 2, 3, 5 and 7 is `<x>`, and `[x]`
passes both tests and has the leading monomials of all four images, while
the ideal is `<x*z, x*(210*y + 1)>`: the component `210*y + 1 = 0` is at
infinity modulo these primes. On the homogenisation the same primes give a
candidate which fails the first test, and 11 discards them:

```python
>>> from sympy import primerange
>>> R, x, z, y = ring("x,z,y", QQ, lex)
>>> trace = ModularTrace()
>>> modular_groebner([x*(z + 210*y + 1), x*(z + 420*y + 2)], R, primes=primerange(2, 100), trace=trace)
[x*z, x*y + 1/210*x]
>>> trace.unlucky, trace.failed_verifications
([2, 3, 5, 7], 1)

```

Homogenising is also what makes the images fast where SymPy is slow: the
Buchberger algorithm then proceeds degree by degree, and the image of
Katsura-4 for `lex` takes 0.8 s homogenised against 7.5 s directly (for
`grevlex` homogenising costs up to a factor 2.5 instead).

The primes. With SymPy's pure Python ground types the time of a basis over
`GF(p)` hardly depends on the size of `p` (the homogenised Katsura-4 for
`lex`, one run each: 0.66 s below `2**31`, 0.84 s below `2**256`, 1.04 s
below `2**1024`, 1.7 s below `2**2048`, where finding the prime takes 5 s
as well), while the number of images needed is inversely proportional to
it: the primes are the primes below `2**1024` in decreasing order
(`default_primes`), a fixed sequence starting at `2**1024 - 105`. The `primes`
argument takes another sequence; `ModularTrace` records the primes used,
skipped and discarded.

Timings, in seconds of CPU time, of `sympy.polys.groebnertools.groebner`
("SymPy") and of `modular_groebner` on the same ring elements, SymPy 1.14
with the pure Python ground types, Python 3.12, a limit of 300 s of
wall-clock time per entry:

| system    | `grevlex` SymPy | `grevlex` modular | `lex` SymPy | `lex` modular |
|-----------|----------------:|------------------:|------------:|--------------:|
| Katsura-3 |            0.01 |              0.02 |        0.03 |          0.04 |
| Katsura-4 |            0.07 |              0.14 |        54.5 |          0.98 |
| Katsura-5 |            0.67 |              1.12 |       > 300 |         136.8 |
| cyclic-4  |            0.00 |              0.02 |        0.00 |          0.02 |
| cyclic-5  |            0.28 |              0.79 |        14.7 |          0.64 |
| cyclic-6  |            44.3 |              38.6 |       > 300 |         121.5 |

How they were measured: one run per entry, on a shared machine with other
jobs running, so that they are good to some tens of percent and no better.
Every entry was run twice, in two passes; the table is the second pass, the
one which recorded CPU time next to the wall-clock time (they agreed within
1 %). The first pass, wall-clock only and under a heavier load, was 4 to
34 % slower on every entry above a tenth of a second, with one exception
which is not explained: its
modular cyclic-6 `grevlex` entry showed 1777 s of wall-clock time, and the
limit of 300 s did not end it. Rerun alone, the same computation with the
same prime took 43.3 s (CPU and wall-clock), and 38.6 s in the second
pass; nothing of that run was recorded besides its wall-clock time, so
whether the process was not scheduled or something else happened is not
known. All the modular bases of the table used one image, except
Katsura-5 for `lex` (three; its coefficients have 2738 bits), and the
images are 79 to 98 % of the time, the verification most of the rest
(28 s for Katsura-5 `lex`).

What the table says: for `lex` the modular algorithm is 23 to 56 times
faster where SymPy needs more than a few seconds, and finishes two of the
entries which SymPy does not; for `grevlex` it is 1.7 to 2.8 times slower
on the entries below a second and on a par on cyclic-6. Two measurements
go with that: the `grevlex` bases of the table have coefficients of 3 to
113 bits and the `lex` bases of up to 2738, so that there is less growth
to avoid; and the image of the homogenised input costs more than the image
of the input for `grevlex` (cyclic-5: 0.77 s against 0.31 s, a basis of 38
elements against 20). On 40 random dense systems in three and
four variables the picture was the same, on few cases: of the bases which
took SymPy more than a second, the two for `grevlex` were 1.3 and 1.8
times slower by the modular algorithm, the two for `lex` 4.2 and 4.5 times
faster, three more (`lex` and elimination) which SymPy did not finish in
30 s took 6, 7 and 17 s, and one `lex` basis was finished by neither (30 s
and 60 s).

The `lex` column compares with SymPy's direct computation, which is not
the fastest way to these bases when the ideal is zero-dimensional, as all
of the table are: SymPy's `grevlex` basis converted by SymPy's FGLM
(`Ideal.change_order('lex')`) took 0.6 s + 1.8 s for Katsura-5 and 49.8 s
+ 25.2 s for cyclic-6, less than the modular algorithm for `lex` (these
conversions are also what the two large modular `lex` bases were checked
against, and found equal). `Ideal` takes that way by itself: a basis for
`lex` or `grlex` of an ideal of dimension zero (a power of every variable
among the leading monomials of the `grevlex` basis) is converted from the
`grevlex` basis by FGLM, Katsura-4 in a quarter of a second. The modular
algorithm remains what computes the `lex` and elimination bases of the
ideals of positive dimension, where FGLM does not apply.

Hence the rule of `Ideal`, which computes its bases through
`sympy_extras.polys.modulargroebner.groebner`: the direct computation
(`direct_groebner`, F4 by default, see above; the table above this
paragraph was measured with SymPy's Buchberger) is given
`settings.groebner_direct_time` seconds (0.25), twenty times as long for
`grevlex` and `grlex` (5 s), and the modular algorithm runs if it has not
finished. On the inputs which SymPy handles in milliseconds the
modular algorithm costs about twice as much (3.9 ms against 1.7 ms on
three quadrics in three variables), which is why it is not run first, and
nothing cheap was found which tells in advance the inputs where it wins;
`settings.modular_groebner = False` (or `configure(modular_groebner=False)`)
turns it off, and time limits need the main thread: elsewhere SymPy's
computation runs to its end. Other coefficient
domains (finite fields, algebraic fields, fields of fractions) always go to
SymPy.

## Reference

`Ideal(gens, *symbols, domain=None, order='grevlex')` with

- bases: `groebner_basis(order)`, `leading_monomials(order)`,
  `change_order(order)`, `reduced()`, `ring(order)`;
- membership: `contains(f)`, `f in I`, `reduce(f, order)`, `is_zero()`,
  `is_whole_ring()`, `subset(J)`, `==`;
- operations: `I + J`, `I*J`, `f*I`, `I**n`, `eliminate(symbols)`,
  `intersect(J)`, `quotient(J)`, `saturate(J)`, `radical_contains(f)`;
- invariants: `dimension()`, `is_zero_dimensional()`, `degree()`,
  `hilbert_series(t)`, `hilbert_polynomial(d)`;
- zero-dimensional ideals: `standard_monomials(order)`,
  `vector_space_dimension()`, `multiplication_matrix(f, order)`,
  `univariate(x)`, `is_maximal()`;
- any dimension: `radical()`, `is_radical()`, `equidimensional_parts()`,
  `minimal_primes()`, `is_prime()`, `height()` (the functions `radical`,
  `equidimensional_parts` and `minimal_primes` of
  `sympy_extras.polys.idealdecomposition` compute them from the chains, in
  dimension zero too), `primary_decomposition()`, `associated_primes()`,
  `is_primary()` (and the functions of the same names, with
  `independent_set`, in `sympy_extras.polys.idealdecomposition`).

`count_real_solutions(system, *gens)`, `count_complex_solutions(system,
*gens)`, `tarski_query(system, g, *gens)`, `real_sign_counts(system, g,
*gens)` (a `SignCounts(zero, positive, negative)`), `sign_determination(system,
polys, *gens)`, `hermite_matrix(system, g, *gens)` and
`decide_zero_dimensional(formula, gens, max_dimension=64)` are in
`sympy_extras.polys.hermite`; `system` is an `Ideal` or a list of
polynomials with rational coefficients.

`separating_forms(variables, points)` enumerates the linear forms with
small integer coefficients among which one separates any `points` points.

`f4(polys, ring)`, `reduced_basis(basis, ring)`, `is_graded(ring)` and
`row_echelon(rows, K)` are in `sympy_extras.polys.f4` (typed rings);
the conversions from and to SymPy rings (`qq_from_sympy`, `gf_from_sympy`,
`zz_from_sympy`, the `_to_sympy` inverses, `typed_order`) in
`sympy_extras.polys.interop`; `strong_groebner(polys, ring)`,
`strong_reduce(f, basis)`, `is_strong_groebner(basis)`,
`s_polynomial_zz(f, g)` and `g_polynomial_zz(f, g)` in
`sympy_extras.polys.stronggroebner`;
`modular_groebner(polys, ring, primes=None, trace=None)`, `direct_groebner(polys, ring)`, `groebner(polys,
ring)` (the dispatcher used by `Ideal`), `rational_reconstruction(a, m,
bound=None)`, `chinese_remainder(a, m, b, p)`, `default_primes()` and
`ModularTrace` are in `sympy_extras.polys.modulargroebner`.

`rur(system, *gens, t=None)` and `RationalUnivariateRepresentation`
(attributes `form`, `polynomial`, `characteristic`, `denominator`,
`numerators`; methods `polynomials`, `solutions`, `coordinates`,
`multiplicity`) are in `sympy_extras.polys.rur`; `Ideal.rur(t)` calls it.
`hilbert_numerator(monomials, t)`, `hilbert_series(monomials, variables,
t)` and `minimal_generators(monomials)` in `sympy_extras.polys.hilbert`
compute the Hilbert series of a monomial ideal by Bigatti's algorithm. `groebner_walk`, `extended_groebner` and
`initial_form` are in `sympy_extras.polys.groebnerwalk`; `WeightOrder`,
`BlockOrder` and `elimination_order` in `sympy_extras.polys.orderings`.

## References

- A. M. Bigatti, *Computation of Hilbert-Poincaré series*, Journal of Pure
  and Applied Algebra 119 (1997), 237-253.
- F. Rouillier, *Solving zero-dimensional systems through the rational
  univariate representation*, Applicable Algebra in Engineering,
  Communication and Computing 9 (1999), 433-461.

- P. Pedersen, M.-F. Roy, A. Szpirglas, *Counting real zeros in the
  multivariate case*, in Computational Algebraic Geometry, Progress in
  Mathematics 109, Birkhäuser, 1993, 203-224.
- S. Basu, R. Pollack, M.-F. Roy, *Algorithms in Real Algebraic Geometry*,
  2nd ed., Springer, 2006 (Hermite's quadratic form, Theorem 4.102; sign
  determination, Section 10.3).
- M. Ben-Or, D. Kozen, J. Reif, *The complexity of elementary algebra and
  geometry*, J. Computer and System Sciences 32 (1986) 251-264.

- E. A. Arnold, *Modular algorithms for computing Gröbner bases*,
  J. Symbolic Computation 35 (2003) 403-419.
- N. Idrees, G. Pfister, S. Steidel, *Parallelization of modular
  algorithms*, J. Symbolic Computation 46 (2011) 672-684.
- F. Pauer, *On lucky ideals for Gröbner basis computations*, J. Symbolic
  Computation 14 (1992) 471-482.
- P. S. Wang, M. J. T. Guy, J. H. Davenport, *P-adic reconstruction of
  rational numbers*, ACM SIGSAM Bulletin 16 (1982) 2-3.
- R. Gebauer, H. M. Möller, *On an installation of Buchberger's
  algorithm*, J. Symbolic Computation 6 (1988) 275-286.
- S. Collart, M. Kalkbrener, D. Mall, *Converting bases with the Gröbner
  walk*, J. Symbolic Computation 24 (1997) 465-469.
- D. Cox, J. Little, D. O'Shea, *Ideals, Varieties, and Algorithms*, 4th
  ed., Springer, 2015 (elimination, Hilbert functions, Seidenberg's lemma).
- E. Hubert, *Notes on triangular sets and triangulation-decomposition
  algorithms I: polynomial systems*, LNCS 2630 (2003) (the saturated ideal
  of a squarefree regular chain is radical and unmixed).
- T. Becker, V. Weispfenning, *Gröbner bases*, Springer, 1993, chapter 8
  (separating forms, primality in dimension zero, extension and
  contraction).
- P. Gianni, B. Trager, G. Zacharias, *Gröbner bases and primary
  decomposition of polynomial ideals*, J. Symbolic Computation 6 (1988)
  149-167.
- T. Shimoyama, K. Yokoyama, *Localization and primary decomposition of
  polynomial ideals*, J. Symbolic Computation 22 (1996) 247-277.
- H. Kredel, V. Weispfenning, *Computing dimension and independent sets
  for polynomial ideals*, J. Symbolic Computation 6 (1988) 231-247.
- W. Vasconcelos, *Computational Methods in Commutative Algebra and
  Algebraic Geometry*, Springer, 1998.
