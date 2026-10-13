# Algebraic curves: Puiseux expansions and integral bases

`sympy_extras.polys.puiseux` and `sympy_extras.polys.functionfields` are
the foundation of the integration of algebraic functions on a general
curve (`sympy_extras.integrals.curves`): the branches of a plane curve
`F(x, y) = 0` above a point, and the ring of the functions on the curve
which are finite above every finite point.

## What SymPy has and what is added

SymPy computes with algebraic *numbers* (`AlgebraicField`, `minpoly`,
factorization over number fields) and with polynomials in several
variables, but it has no notion of the function field of a curve: no
Puiseux expansion (its `series` expands a root `y(x)` only when it is
given explicitly, and in integer powers), no integral closure of `Q[x]`
in `Q(x)[y]/(F)`. Both are needed by Trager's integration algorithm and
by anything which reasons about the places of a curve.

## Puiseux expansions (`sympy_extras.polys.puiseux`)

`puiseux_expansions(F, x, y, point, order)` returns the branches of
`F(x, y) = 0` above `x = point` (an algebraic number, or `oo`), one per
class of conjugate branches: a `PuiseuxExpansion` with the ramification
index `e`, the number field of its coefficients, the embedding of the
field of the curve into it, the scaling `lambda` of Duval's rational
expansions (`x - a = lambda*t**e`, so that the `q`-th root of the
characteristic root is never adjoined) and the series `y = sum c_k t**k`
known below the order asked. The Newton polygon algorithm follows the
edges of the lower hull of the points `(i, ord a_i)`; the characteristic
polynomial of an edge is factored over the current field, a root of each
irreducible factor is adjoined (its conjugates give the conjugate
branches) and the recursion continues with the roots sharing the first
term until one root is left, which Newton's iteration completes.

```python
>>> from sympy import symbols, oo
>>> from sympy_extras.polys.puiseux import puiseux_expansions
>>> x, y = symbols('x y')
>>> for b in puiseux_expansions(y**4 + (x**2 + x)*y - x**2, x, y, 0, order=1):
...     print(b.ramification, b.conjugates, b.as_expr())
1 1 x
3 1 -x**(1/3)
>>> [b.as_expr() for b in puiseux_expansions(y**2 - x**3 - x**2, x, y, 0, order=4)]
[x**3/8 - x**2/2 - x, -x**3/8 + x**2/2 + x]
>>> [b.as_expr() for b in puiseux_expansions(y**2 - x**3 - 1, x, y, oo, order=3)]
[x**(3/2) + 1/(2*x**(3/2))]

```

`PuiseuxExpansion.evaluate(p)` gives the Laurent series of a polynomial
in `x` and `y` along the branch, whose valuation is the order of the
function at the place: the tool with which integrality, residues and
vanishing orders are read off.

The number fields are built exactly (`adjoin_root`): a root `u` of an
irreducible `psi` over `K = Q(theta)` is adjoined through the primitive
element `Theta = theta + k*u`, whose minimal polynomial is the resultant
of `k**d psi((T - theta)/k)` with the minimal polynomial of `theta`, and
the image of `theta` in `Q(Theta)` is the common root of its minimal
polynomial and of `psi((Theta - t)/k; t)`, a gcd. No root is evaluated
numerically: SymPy's own `primitive_element` and `to_number_field`
identify roots by refining complex isolating intervals, which takes
minutes at the precisions they ask for degrees beyond four.

## Function fields and integral bases (`sympy_extras.polys.functionfields`)

`FunctionField(F, x, y)` is `K(x)[y]/(F)`, `K` the rationals or the
field of the parameters of `F`; when the leading coefficient `a_n(x)` of
`F` in `y` is not constant the generator is `a_n*y`, which is integral.
Its `Element`s have the arithmetic of the field (products modulo `F`,
inverses by the extended Euclidean algorithm, derivatives through `y' =
-F_x/F_y`, traces by Newton's identities) and `integral_basis()` is a
basis of the integral closure of `K[x]` by Trager's algorithm, the
Round 2 algorithm of Zassenhaus in the function field case: the
discriminant of the order, the `Q`-trace radical for `Q` the product of
the square factors of the discriminant (the kernel over `K[x]` of `[T |
-Q I]`, `T` the trace matrix), its idealizer (a column echelon form over
`K[x]`), the order enlarged until the idealizer is the order itself,
which is Trager's criterion of maximality. Everything is polynomial
linear algebra over `K[x]`, so parameters are allowed. The basis is in
Hermite normal form: the `i`-th element has degree `i` in `y`.

```python
>>> from sympy import symbols
>>> from sympy_extras.polys.functionfields import integral_basis, FunctionField
>>> x, y, a = symbols('x y a')
>>> integral_basis(y**3 - x**2*(x + 1), x, y)
[1, y, y**2/x]
>>> integral_basis(y**4 + (x**2 + x)*y - x**2, x, y)
[1, y, y**2, y**3/x]
>>> [w.as_expr() for w in FunctionField(y**2 - (x - a)**2*(x - 1), x, y).integral_basis()]
[1, y/(-a + x)]

```

`local_basis_at_infinity()` is the basis of the elements with no pole
above `x = oo` (the integral basis of the curve in `z = 1/x`),
`pole_order_at_infinity` the least `k` with `x**(-k)*w` integral there,
and `normal_basis()` an integral basis normal at infinity in Trager's
sense: `(x**(-k_i) w_i)` is a local basis at infinity, so that a
combination `sum a_i w_i` with polynomial `a_i` has pole order `max(deg
a_i + k_i)` there, which bounds the degrees in the search of a function
with a prescribed divisor and decides the nonelementary integrals.

The tests check the bases of the literature (Bronstein's tutorial,
Kauers' examples, the formula for radical extensions), the integrality
of every element by its characteristic polynomial, and the maximality
of every basis by an independent criterion through the Puiseux
expansions (van Hoeij's: no element `sum c_i w_i/(x - a)` with constant
`c_i` is integral at a root `a` of the discriminant).

## What is not done

- Puiseux expansions over fields with parameters (the coefficient field
  must be the rationals or a number field).
- Van Hoeij's local algorithm for the integral basis (by Puiseux
  expansions), which can be faster than Trager's for curves with few
  singularities of high multiplicity; the Böhm–Decker–Laplagne–Pfister
  normalization through Hensel lifting.
- Riemann–Roch spaces and the arithmetic of divisors as objects; the
  functions with a prescribed divisor are found inside
  `sympy_extras.integrals.curves` for the divisors it meets.
