# Definite integration

Module: `sympy_extras.integrals`

`definite_integral(f, (x, a, b), assumptions)` computes a definite integral
without computing an antiderivative: the value of an integral over an
infinite range, or with a singular integrand, or with parameters, is read
off the structure of the integrand, and the conditions on the parameters
under which it converges come out of the computation. This is how
Mathematica's `Integrate` and REDUCE's DEFINT package work, and what
SymPy's `integrate` lacks: `integrate` evaluates an antiderivative at the
endpoints, misses singularities in between (`integrate(1/(x*sqrt((x + 1)**2)),
(x, -oo, -2))` is left unevaluated, `integrate(1/x, (x, -1, 2))` is
`log(2) + I*pi`) and reports a condition from one endpoint only
(`integrate(x**k/(x + 3), (x, 0, oo))` is `-3**k*pi/sin(pi*k)` for
`k > -1`, though the integral diverges for `k >= 0`).

## The Marichev–Adamchik method

The Mellin transform of a function on `(0, oo)`,
`F(s) = Integral(x**(s - 1)*f(x), (x, 0, oo))`, converges in a vertical
strip of the complex plane and is, for the functions of mathematical
physics, a *gamma quotient*: a constant times powers `rho**(-s)` times a
quotient of products of `gamma(a + b*s)`. Such a quotient is the integrand
of the Mellin–Barnes integral defining a Meijer G-function, which is
Marichev's observation, made into an algorithm by Adamchik and Marichev
for REDUCE (and later Mathematica): the integral of a product of two such
functions is, by Parseval's formula for the Mellin transform, a G-function
evaluated at the ratio of the scales, and Slater's theorem writes the
G-function as a sum of hypergeometric series which `hyperexpand` reduces
to elementary and special functions.

The package implements the four pieces:

- `sympy_extras.integrals.mellin`: the representation (`GammaQuotient`,
  with the strip and the conditions on the parameters), the table of
  transforms of the elementary and special functions (`exp`, powers of
  `1 + x`, `(1 - x)**b` on `(0, 1)`, `sin`, `cos`, `atan`, `log(1 + x)`,
  `erf`, `erfc`, `E1`, `E_n`, `Si`, `Ci`, the Bessel functions `J`, `Y`,
  `K` and `exp(-x)*I`, the step functions, `(-log(x))**k` on `(0, 1)`,
  `1/(exp(x) - 1)`, `1/(exp(x) + 1)`, `1/sinh`, `1/cosh`, and the
  differences `exp(-x) - 1`, `cos(x) - 1`, `sin(x) - x`, `atan(x) - pi/2`
  whose transforms continue the strips), and the matching of an integrand
  against it (`mellin_transform`, `mellin_kernel`);
- `sympy_extras.integrals.slater`: the Mellin–Barnes integral of a quotient
  as a G-function (`mellin_barnes`, through Gauss's multiplication formula
  when the coefficients of `s` are not `±1`) and Slater's theorem
  (`slater_expansion`, `expand_meijerg`), with the case distinction
  `|z| < 1` or `|z| > 1` for `p = q` decided by the assumptions;
- `sympy_extras.integrals.marichev`: the integrals over `(0, oo)`,
  `(0, 1)` and `(1, oo)` (`mellin_integrate`): one factor is the transform
  at `s = alpha + 1`, two factors go through Parseval's formula and the
  G-function, and logarithms are derivatives with respect to the exponent;
- `sympy_extras.integrals.definite`: the driver, which cuts the range at
  the kinks (`Abs`, `sign`, `Heaviside`, `Max`, `Min`, `Piecewise`) and at
  the singularities of the integrand, maps every range onto the three
  canonical ones, changes variables (`x = log(u)`, `x = 1/u`,
  `x = asin(sqrt(u))` for trigonometric integrands over `(0, pi/2)`), and
  falls back to SymPy's `integrate` under the time limit, accepting its
  answer only when it passes a numerical check.

The strips of the transforms are the convergence conditions:

```python
>>> from sympy import symbols, exp, sin, log, sqrt, Abs, oo, pi, cos, S
>>> from sympy_extras.integrals import definite_integral, mellin_transform
>>> x, k, s = symbols('x k s')
>>> a, b = symbols('a b', positive=True)
>>> mellin_transform(1/(1 + x), x, s)
MellinTransform(gamma(s)*gamma(1 - s), (0, 1))
>>> definite_integral(x**k/(x + 3), (x, 0, oo))
Piecewise((-3**k*pi/sin(pi*k), (k > -1) & (k < 0)), (Integral(x**k/(x + 3), (x, 0, oo)), True))
>>> definite_integral(x**k/(x + 3), (x, 0, oo), (k > -1) & (k < 0))
-3**k*pi/sin(pi*k)

```

Products of two functions of the table, over the canonical ranges or
ranges mapped onto them:

```python
>>> definite_integral(exp(-a*x)*sin(b*x)/x, (x, 0, oo))
atan(b/a)
>>> definite_integral(exp(-x)*log(x), (x, 0, oo))
-EulerGamma
>>> definite_integral(x**k*(1 - x)**k, (x, 0, 1), k > -1)
gamma(k + 1)**2/gamma(2*k + 2)
>>> definite_integral(log(1 + 7/x**2), (x, 1, oo))
-log(8) + 2*sqrt(7)*atan(sqrt(7))
>>> definite_integral(x*exp(x)*exp(k*x)/(exp(x) + 3), (x, -oo, oo), (k > -1) & (k < 0))
3**k*pi*(polygamma(0, -k) - polygamma(0, k + 1) - log(3))/sin(pi*k)

```

Singular integrands and kinks:

```python
>>> definite_integral(Abs(x - 1)/sqrt(x), (x, 0, 2))
2*(4 - sqrt(2))/3
>>> definite_integral(1/(x*sqrt((x + 1)**2)), (x, -oo, -2))
-log(2)
>>> definite_integral(1/x, (x, -1, 2))
Integral(1/x, (x, -1, 2))
>>> definite_integral(sqrt(sin(x)), (x, 0, pi/2))
2*sqrt(pi)*gamma(3/4)/gamma(1/4)
>>> definite_integral(sqrt(1 - cos(x)), (x, 0, 2*pi))
4*sqrt(2)

```

## Integrals over regions: `IntegralByRanges`

`IntegralByRanges(f, condition)` is the integral of `f` over the region of
the integration variables described by `condition`, a Boolean combination
of polynomial inequalities and equations, the counterpart of Mathematica's
`Integrate[f, {x, y} ∈ ImplicitRegion[...]]` or `Integrate[Boole[cond] f,
...]`. It is evaluated through the cylindrical algebraic decomposition of
`sympy_extras.polys.cad`: the cells of the decomposition adapted to the
polynomials of the condition are stacks of intervals whose endpoints are
algebraic functions of the earlier variables, so the integral over a
region is the sum over the full-dimensional cells where the condition
holds of iterated integrals with explicit bounds, computed innermost
first by `definite_integral`. Parameters of the condition come first in
the variable order and give a case distinction on their cells.

```python
>>> from sympy import symbols, exp
>>> from sympy_extras.integrals import IntegralByRanges, integrate_by_ranges
>>> x, y, z, r = symbols('x y z r')
>>> IntegralByRanges(1, x**2 + y**2 < 1).doit()
pi
>>> integrate_by_ranges(x*y, (x > 0) & (y > 0) & (x + y < 1))
1/24
>>> integrate_by_ranges(1, (0 < x) & (x < y) & (y < x**2) & (x < 2))
5/6
>>> integrate_by_ranges(1, x**2 + y**2 + z**2 < 1)
4*pi/3
>>> integrate_by_ranges(1, x**2 + y**2 < r**2, [x, y], r > 0)
pi*r**2

```

## What is not done

Tracked in issue #53 of the repository, with the alternatives worth adding.

- Products of three or more functions of the table (a double
  Mellin–Barnes integral, not a G-function), and functions outside the
  table (`sinh`, `cosh` and `besseli` without the exponential which makes
  them decay, inverse hyperbolic functions, `LambertW`, ...): the driver
  falls back to SymPy.
- The logarithmic cases of Slater's theorem (parameters of the G-function
  differing by integers) are computed as limits of the general case, which
  SymPy's `limit` does not always manage.
- Principal values of divergent integrals.
- Complex parameters: the conditions are written for real parameters
  (an inequality on a parameter states that it is real); a parameter which
  may be complex gets a condition on its argument, `Abs(arg(a)) < pi/2`.
- Region integrals whose cell boundaries have no explicit form (roots of
  polynomials of degree five or more in the last variable), and regions
  described by non-polynomial conditions.

## References

- O. I. Marichev, *Handbook of integral transforms of higher
  transcendental functions: theory and algorithmic tables*, Ellis Horwood,
  1983.
- V. S. Adamchik, O. I. Marichev, *The algorithm for calculating
  integrals of hypergeometric type functions and its realization in
  REDUCE system*, ISSAC 1990, pp. 212–224.
- L. J. Slater, *Generalized hypergeometric functions*, Cambridge
  University Press, 1966, section 5.2.
- A. Erdélyi et al., *Tables of integral transforms*, vol. I,
  McGraw-Hill, 1954, chapter VI (Mellin transforms).
- A. P. Prudnikov, Yu. A. Brychkov, O. I. Marichev, *Integrals and
  series*, vol. 3, Gordon and Breach, 1990, chapters 2.24 and 8.4.
- NIST Digital Library of Mathematical Functions, chapter 16.17,
  https://dlmf.nist.gov/16.17.
- G. E. Collins, *Quantifier elimination for real closed fields by
  cylindrical algebraic decomposition*, 1975 (the CAD behind
  `IntegralByRanges`).
