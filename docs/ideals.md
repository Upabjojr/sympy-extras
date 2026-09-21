# Polynomial ideals and Gröbner bases

Modules: `sympy_extras.polys.ideals`, `sympy_extras.polys.groebnerwalk`,
`sympy_extras.polys.modulargroebner`, `sympy_extras.polys.orderings`

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
  and the **degree**, from the leading term ideal;
- for zero-dimensional ideals: the standard monomials, the dimension of the
  quotient algebra, multiplication matrices, the univariate polynomial in
  each variable, the **radical** (Seidenberg's lemma) and the test for
  **maximal** ideals;
- in any dimension, over the rationals: the **radical**, its
  **equidimensional parts**, the **minimal primes**, the **height** and
  the tests for **radical** and **prime** ideals, through triangular
  decompositions into squarefree regular chains
  ([regularchains.md](regularchains.md));
- the **Gröbner walk**, converting a Gröbner basis between orders for
  ideals of any dimension (`change_order` uses FGLM when the ideal is
  zero-dimensional and the walk otherwise);
- **modular Gröbner bases** over the rationals, proven, which `Ideal` uses
  for the bases SymPy does not compute in a quarter of a second.

Not done here: primary decomposition (the primary components and the
embedded primes; only the minimal primes are computed), the depth,
decompositions over other fields than the rationals (algebraic numbers,
finite fields), F4 and strong Gröbner bases over the integers.

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
is found among `x1 + t*x2 + t**2*x3 + ...`, its minimal polynomial over
`Q[u]` is the generator of an elimination ideal, and to each irreducible
factor answers the prime which is the saturation by the other factors.
The primes are those over the rationals: `x**2 - 2*y**2` is one.

```python
>>> Ideal([x**2 - z, y**2 - z], x, y, z).minimal_primes()
[Ideal([y**2 - z, x + y], x, y, z), Ideal([y**2 - z, x - y], x, y, z)]
>>> Ideal([x**2 - 2*y**2], x, y).is_prime()
True

```

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
against, and found equal). `Ideal.groebner_basis('lex')` does not take
that way by itself.

Hence the rule of `Ideal`, which computes its bases through
`sympy_extras.polys.modulargroebner.groebner`: SymPy's direct computation
is given `settings.groebner_direct_time` seconds (0.25), twenty times as
long for `grevlex` and `grlex` (5 s), and the modular algorithm runs if it
has not finished. On the inputs which SymPy handles in milliseconds the
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
  dimension zero too).

`modular_groebner(polys, ring, primes=None, trace=None)`, `groebner(polys,
ring)` (the dispatcher used by `Ideal`), `rational_reconstruction(a, m,
bound=None)`, `chinese_remainder(a, m, b, p)`, `default_primes()` and
`ModularTrace` are in `sympy_extras.polys.modulargroebner`.

`hilbert_numerator(monomials, t)` computes the numerator of the Hilbert
series of a monomial ideal. `groebner_walk`, `extended_groebner` and
`initial_form` are in `sympy_extras.polys.groebnerwalk`; `WeightOrder`,
`BlockOrder` and `elimination_order` in `sympy_extras.polys.orderings`.

## References

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
- W. Vasconcelos, *Computational Methods in Commutative Algebra and
  Algebraic Geometry*, Springer, 1998.
