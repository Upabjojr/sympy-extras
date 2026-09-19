# A canonical form and a zero test for elementary expressions

`sympy_extras.simplify` decides whether an elementary expression vanishes,
and writes it in a canonical form, by the Risch–Rosenlicht structure
theorem.

## What SymPy has and what is added

SymPy's `simplify`, `cancel`, `trigsimp`, `powsimp`, `logcombine` are
rewriting heuristics: they often reduce a zero to `0`, never prove that
something is *not* zero, and do not look at branches (`logcombine` and
`powsimp` with `force=True` apply `log(x*y) = log(x) + log(y)` and
`sqrt(x)*sqrt(y) = sqrt(x*y)` whatever the signs). `equals` falls back on
random sampling. Every zero test of this package did the same.

Here an expression is written as a rational function of generators which
are *proven* algebraically independent, so that it vanishes identically
exactly when its numerator does:

- `canonical_form(expr, assumptions)`: the quotient of two polynomials in
  the generators, the denominator monic;
- `is_zero(expr, assumptions)`: `True` is a proof; `False` means "does not
  vanish identically on the region", proved by the independence of the
  generators or witnessed by a sample point; `None` that neither was
  established;
- `equal(a, b, assumptions)`;
- `ElementaryTower(assumptions)`: the tower itself, with `element`,
  `to_expr`, `generators`, `certified`, and `certifies(a)`: whether the
  generators which the element `a` involves are independent (a dependent
  generator which `a` does not involve does not matter).

```python
>>> from sympy import symbols, exp, log, sqrt, sin, cos, atan, asin, acos, pi, Rational, I
>>> from sympy_extras.simplify import is_zero, canonical_form, ElementaryTower
>>> x, y = symbols('x y')
>>> is_zero(sin(x + y) - sin(x)*cos(y) - cos(x)*sin(y))
True
>>> is_zero(exp(x) - x - 1)
False
>>> canonical_form((sqrt(x) + 1)**2 - x - 1)
2*sqrt(x)

```

## Branches are part of the question

An identity between logarithms or roots holds on a region, and the
assumptions say which:

```python
>>> is_zero(log(x**2) - 2*log(x)), is_zero(log(x**2) - 2*log(x), x > 0)
(False, True)
>>> is_zero(sqrt(x)*sqrt(y) - sqrt(x*y)), is_zero(sqrt(x)*sqrt(y) - sqrt(x*y), [x > 0, y > 0])
(False, True)
>>> is_zero(log(x*y) - log(x) - log(y), x > 0)
True
>>> is_zero(atan(x) + atan(1/x) - pi/2, x > 0), is_zero(atan(x) + atan(1/x) + pi/2, x < 0)
(True, True)
>>> is_zero(asin(x) - atan(x/sqrt(1 - x**2)), [x > -1, x < 1])
True

```

Without assumptions a symbol is a complex number, as in SymPy: `log(x**2)
- 2*log(x)` is `-2*I*pi` at `x = -1`. One positive factor is enough for
`log(u*w) = log(u) + log(w)`, which is why the third example needs `x > 0`
only.

## How it works

The field is `K(g_1, ..., g_n)`, `K` a number field which grows as
algebraic numbers are met. A generator is a variable, an exponential
`exp(a)`, a logarithm `log(v)`, or a root `exp(z/q)` of an earlier
exponential or logarithm (`v**(1/q)` is `exp(log(v)/q)`, which is how
SymPy defines the principal root). Trigonometric and hyperbolic functions
are written with exponentials, their inverses with logarithms, `a**b` is
`exp(b*log(a))`, and `pi` is `-I*log(-1)`.

Every generator carries a pair `(z, e)` with `e = exp(z)`. The structure
theorem says that a new `exp(a)` is algebraic over the field exactly when
`a = c + sum(r_i*z_i)` with rational `r_i` and a constant `c`, and a new
`log(v)` exactly when `v'/v = sum(r_i*z_i')`: linear systems over the
rationals, solved exactly.

- A dependent `exp(a)` *is* `exp(c)*prod(e_i**r_i)`: the exponential is a
  homomorphism, there is no branch to choose. A fractional `r_i` brings a
  root generator with its relation `rho**q = e_i`.
- A dependent `log(v)` differs from `sum(r_i*z_i)` by a *locally* constant
  function, which jumps across the branch cuts. It is pinned when the
  assumptions prove the arguments positive (everything is real), when `v`
  is negative (`log(v) = I*pi + log(-v)`), or, for complex arguments, when
  the variables are real, the region is convex and no argument crosses
  the negative axis on it: then the constant is a logarithm of a number,
  told from the others by its value at one point. Otherwise `log(v)`
  becomes a generator which is not independent, and the tower is
  uncertified: `True` answers remain proofs, `False` needs a witness.
- The roots are certified by Kummer theory: `rho_i**q_i = e_i` generate an
  extension of full degree exactly when the `e_i**(N/q_i)` generate a
  subgroup of order `prod(q_i)` modulo `N`-th powers, which the exponent
  vectors of the irreducible factors and a Smith normal form decide.
- Constants have no derivation. `exp(c)` is dependent when `c` is exactly
  a rational combination of the constant `z_j`; the logarithm of a
  positive rational is the combination of the logarithms of its primes;
  the logarithm of an algebraic number is looked for as a multiplicative
  relation with the numbers already there, *found* by PSLQ on the moduli
  and arguments together and *verified* exactly in the number field. That
  the constants left as generators are independent is Schanuel's
  conjecture; its hypothesis, the linear independence of the `z_j` over
  the rationals, holds by construction.

```python
>>> is_zero(pi/4 - 4*atan(Rational(1, 5)) + atan(Rational(1, 239)))     # Machin
True
>>> is_zero(pi/4 - 12*atan(Rational(1, 18)) - 8*atan(Rational(1, 57)) + 5*atan(Rational(1, 239)))   # Gauss
True
>>> is_zero(I**I - exp(-pi/2)), is_zero(sqrt(5 + 2*sqrt(6)) - sqrt(2) - sqrt(3))
(True, True)
>>> is_zero(exp(pi) - pi**exp(1))
False
>>> tower = ElementaryTower()
>>> u = tower.element(exp(2*x) + exp(x) + log(2*x) + sqrt(x))
>>> [g.expression for g in tower.generators]
[x, log(x), sqrt(x), exp(x), log(2)]
>>> [g.kind for g in tower.generators], tower.certified
(['variable', 'log', 'root', 'exp', 'log'], True)

```

`sqrt(x)` is `exp(log(x)/2)`, so `log(x)` is a generator before it;
`exp(2*x)` is the square of `exp(x)`; and `log(2*x)` is `log(2) + log(x)`
for every `x`, the factor `2` being positive.

## Limitations

- The elementary functions only: `gamma`, `erf`, `Abs`, `Piecewise`,
  floating point numbers raise `NotElementary` (`is_zero` then answers
  from a sample point, or `None`).
- A root of a radicand which involves another root is adjoined with its
  relation but not certified, and denominators are not rationalised, so
  the canonical form of an expression with roots in a denominator is
  canonical up to that.
- The independence of logarithms of several algebraic numbers is not
  proved (a relation not found by PSLQ within its bounds is not a proof
  that there is none): such towers are uncertified.
- `False` answers for constants rest on Schanuel's conjecture when no
  sample point is involved; `certified` says which.
- The number field grows by primitive elements, whose cost explodes with
  the degree: it is kept below degree 32, and an expression with more
  unrelated radicals and roots of unity raises `NotElementary`.

## References

- R. H. Risch, Algebraic properties of the elementary functions of
  analysis, American Journal of Mathematics 101 (1979), 743-759.
- M. Rosenlicht, On Liouville's theory of elementary functions, Pacific
  Journal of Mathematics 65 (1976), 485-492.
- M. Bronstein, Symbolic Integration I, second edition, Springer, 2005,
  chapter 9.
- D. Richardson, How to recognize zero, Journal of Symbolic Computation 24
  (1997), 627-645.
