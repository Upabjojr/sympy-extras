# Polynomial ideals and Gröbner bases

Modules: `sympy_extras.polys.ideals`, `sympy_extras.polys.groebnerwalk`,
`sympy_extras.polys.orderings`

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
  each variable, the **radical** (Seidenberg's lemma) and tests for
  **radical**, **prime** and **maximal** ideals;
- the **Gröbner walk**, converting a Gröbner basis between orders for
  ideals of any dimension (`change_order` uses FGLM when the ideal is
  zero-dimensional and the walk otherwise).

Not done here: general (positive-dimensional) radicals and primary
decomposition, F4, modular Gröbner bases and strong Gröbner bases over the
integers.

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

The variety of `I` is the union of a twisted cubic and the `z` axis (where
`x = y = 0`). Saturating by `y` removes the axis:

```python
>>> cubic = I.saturate(Ideal([y], x, y, z))
>>> cubic
Ideal([x**2 - y*z, x*y - z**2, -x*z + y**2], x, y, z)
>>> cubic.degree(), cubic.hilbert_polynomial(d)
(3, 3)
>>> I.quotient(cubic)
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
>>> J.multiplication_matrix(y)
Matrix([[0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 0]])
>>> J.is_radical(), J.is_maximal()
(True, True)
>>> Ideal([x**2, y**2 - 2*y + 1], x, y).radical()
Ideal([x, y - 1], x, y)

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
  `univariate(x)`, `radical()`, `is_radical()`, `is_prime()`,
  `is_maximal()`.

`hilbert_numerator(monomials, t)` computes the numerator of the Hilbert
series of a monomial ideal. `groebner_walk`, `extended_groebner` and
`initial_form` are in `sympy_extras.polys.groebnerwalk`; `WeightOrder`,
`BlockOrder` and `elimination_order` in `sympy_extras.polys.orderings`.

## References

- S. Collart, M. Kalkbrener, D. Mall, *Converting bases with the Gröbner
  walk*, J. Symbolic Computation 24 (1997) 465-469.
- D. Cox, J. Little, D. O'Shea, *Ideals, Varieties, and Algorithms*, 4th
  ed., Springer, 2015 (elimination, Hilbert functions, Seidenberg's lemma).
- W. Vasconcelos, *Computational Methods in Commutative Algebra and
  Algebraic Geometry*, Springer, 1998.
