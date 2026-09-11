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

The integrand is also read in rewritten forms: products and powers of
`sin` and `cos` as sums, hyperbolic functions as exponentials, inverse
hyperbolic functions as logarithms, orthogonal polynomials expanded;
`log(1 - x)` on `(0, 1)` is a kernel of its own. With
`principal_value=True` a divergent integral with a simple pole inside the
range gets its Cauchy principal value, from the antiderivative with the
symmetric limits at the pole taken as one limit; symbolic endpoints go
through the antiderivative with limits under the assumptions.

```python
>>> from sympy import sinh, atanh
>>> definite_integral(exp(-a*x)*sinh(b*x), (x, 0, oo), a > b)
b/((a - b)*(a + b))
>>> definite_integral(x*atanh(x), (x, 0, 1))
1/2
>>> definite_integral(1/(x - 1), (x, 0, 3), principal_value=True)
log(2)
>>> definite_integral(exp(-x), (x, a, b))
-exp(-b) + exp(-a)

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

## The other methods

The driver tries, after the Mellin method and the residues:

- **Creative telescoping** (`sympy_extras.integrals.telescoping`): for a
  hyperexponential integrand `F(x, t)` (both logarithmic derivatives
  rational), the Almkvist–Zeilberger algorithm finds `a_0(t), ..., a_J(t)`
  and a rational certificate `R` with `sum(a_j * d^j F/dt^j) = d(R F)/dx`,
  so that the integral `I(t)` satisfies the linear ODE
  `sum(a_j I^(j)) = [R F]` between the bounds. `holonomic_ode` returns the
  equation, `holonomic_integral` solves it with `dsolve` and fixes the
  constants at a value of the parameter where the integral is computed
  directly. This is the continuous analogue of Zeilberger's algorithm of
  `sympy_extras.concrete.zeilberger`, and the way to integrals of
  products of special functions which are not Meijer G-functions (the
  general D-finite case, Chyzak's algorithm, is not implemented).
- **The rules of the Laplace transform** (`sympy_extras.integrals.laplace`):
  for `g(t)*exp(-s*t)` over `(0, oo)`, division by `t` (the transform
  integrated from `s` to `oo`), multiplication by `t**n` (derivatives of the
  transform), the shifts `Heaviside(t - a)` and `exp(c*t)`, periodic
  integrands (`Abs(sin(t))` gives `coth(pi*s/2)/(s**2 + 1)`) and
  convolutions `Integral(p(u)*q(t - u), (u, 0, t))` (the product of the
  transforms), each piece transformed by the other methods.
- **More contours** (`sympy_extras.integrals.contours`): the rectangle of
  height `2*pi/c` for `x**n*exp(k*x)*R(exp(c*x))` over the real line
  (`x**2/cosh(x)` is `pi**3/4`, `exp(a*x)/(1 + exp(x))` is `pi/sin(pi*a)`),
  the sector of angle `2*pi/n` for `x**a/(b + c*x**n)` over `(0, oo)` with
  symbolic `a` and `n`, and the indented contour for `R(x)*sin(k*x)` with
  simple real poles cancelled by the sine.
- **Frullani's theorem and Glasser's master theorem**
  (`sympy_extras.integrals.transformations`): `(f(a x) - f(b x))/x` over
  `(0, oo)` is `(f(0) - f(oo)) log(b/a)`, and `F(x - sum(a_i/(x - b_i)))`
  over the real line (or over `(0, oo)` for an even `F`, the
  Cauchy–Schlömilch transformation) integrates like `F` itself.
- **The mean value of a periodic integrand** (`sympy_extras.integrals.periodic`):
  `Integral(f, (x, c, c + 2*pi*k))` is `2*pi*k` times the constant
  Laurent coefficient of `f` written in `z = exp(I*x)`, read off or summed
  from the Taylor coefficients of the factors (`exp(cos(x))*cos(sin(x))`
  gives `2*pi`, `exp(a*cos(x))*cos(n*x)` gives `2*pi*besseli(n, a)`);
  rational functions of `sin` and `cos` go to the residues instead.
- **Elliptic integrals** (`sympy_extras.integrals.elliptic`): square roots
  of cubics and quartics with real roots, `R(x)/sqrt(P(x))` and
  `R(x)*sqrt(P(x))` over a cell between the roots, reduced by the
  substitutions of Byrd–Friedman to Legendre's `elliptic_k`, `elliptic_e`,
  `elliptic_pi` and the incomplete `elliptic_f` (SymPy's parameter
  `m = k**2`); `Integral(1/sqrt(1 - x**4), (x, 0, 1))` gives
  `elliptic_k(1/2)`, the lemniscate constant.
- **Series expansion and termwise integration**
  (`sympy_extras.integrals.series`): one factor expanded in its formal
  power series (or a geometric series of exponentials), the moments of the
  rest integrated in closed form in the index, and the series summed by
  `summation`, the Zeilberger and polygamma-series algorithms of
  `sympy_extras.concrete`, or a hypergeometric closed form; the interchange
  is justified by absolute convergence or checked numerically.
  `Integral(log(x)*log(1 - x), (x, 0, 1))` gives `2 - pi**2/6`.
- **Chyzak's algorithm** (`sympy_extras.integrals.dfinite`): creative
  telescoping for D-finite integrands beyond the hyperexponential ones
  (Bessel functions, orthogonal polynomials, their products), by
  Koutschan's ansatz on the closure of the factors; `dfinite_ode` gives
  the equation of the parametric integral and `dfinite_integral` solves
  it (`exp(-t*x)*besselj(0, x)` over `(0, oo)` gives `1/sqrt(1 + t**2)`).
- **Differentiation under the integral sign**
  (`sympy_extras.integrals.parametric`): `I'(p)` is a simpler integral
  (a factor `1/x` disappears against `exp(-p x)`, a logarithm against
  `x**p`), integrated by the other methods, integrated back in `p`, with
  the constant fixed at `p = 0`, `1` or at infinity; the interchange is
  checked numerically when `settings.numerical_checks` is on.
- **An antiderivative evaluated by one-sided limits**
  (`sympy_extras.integrals.antiderivative`): the range is cut at the
  singularities of the integrand *and* at the discontinuities of the
  antiderivative (the jumps of `atan` and `log` across branch cuts, which
  Rioboo's and Jeffrey–Rich's constructions repair for rational and
  trigonometric integrands), and the pieces are summed with one-sided
  limits under the assumptions; an infinite limit exposes a divergence.
  The antiderivative comes from the Risch port
  (`sympy_extras.integrals.risch`, Aaron Meurer's unmerged SymPy pull
  requests, see below) or from SymPy's `integrate`.

- **Ramanujan's master theorem and the method of brackets**
  (`sympy_extras.integrals.brackets`): for a factor outside the Mellin
  table, the coefficients of its Taylor series as a function of the index
  (from SymPy's formal power series) give the Mellin transform directly,
  `Integral(x**(s-1) f(x), (x, 0, oo)) = gamma(s) phi(-s)` for
  `f = sum(phi(k) (-x)**k/k!)`; for a product of two series the bracket
  rule of Gonzalez and Moll eliminates one index and the remaining series
  is summed. The theorem is checked against the table on the kernels both
  know; the method of brackets is a heuristic and is used only where its
  series converge.
- **Asymptotic expansions** (`sympy_extras.integrals.asymptotic`,
  `asymptotic_integral`): for a parametric integral without a closed form,
  the first terms as the parameter grows, by Watson's lemma
  (`exp(-t*x)*phi(x)`), Laplace's method (`phi*exp(t*h)` with a maximum
  inside or at an endpoint, to any order) and the stationary phase
  (`phi*exp(I*t*h)`, leading term), with the order term.
- **Symbolic-numeric recognition** (`sympy_extras.integrals.recognize`,
  `definite_integral(..., recognize=True)`): a high-precision quadrature
  and an integer relation (PSLQ) with a basis of constants propose a
  closed form, re-checked at forty-five digits. A conjecture, not a proof:
  off by default.

```python
>>> from sympy import symbols, exp, sin, log, atan, cos, oo, pi
>>> from sympy_extras.integrals import definite_integral, holonomic_ode, parametric_integral
>>> x = symbols('x')
>>> p, t = symbols('p t', positive=True)
>>> holonomic_ode(exp(-x**2)*cos(2*t*x), x, 0, oo, t)
Eq(2*t*I(t) + Derivative(I(t), t), 0)
>>> definite_integral(exp(-x**2)*cos(2*t*x), (x, 0, oo))
sqrt(pi)*exp(-t**2)/2
>>> from sympy import besselj
>>> definite_integral(exp(-t*x)*besselj(0, x), (x, 0, oo))
1/sqrt(t**2 + 1)
>>> definite_integral(1/sqrt(1 - x**4), (x, 0, 1))
sqrt(pi)*gamma(1/4)/(4*gamma(3/4))
>>> definite_integral(log(x)*log(1 - x), (x, 0, 1))
2 - pi**2/6
>>> definite_integral(atan(p*x)/(x*(1 + x**2)), (x, 0, oo))
pi*log(p + 1)/2
>>> definite_integral((x**p - 1)/log(x), (x, 0, 1))
log(p + 1)
>>> definite_integral(1/(2 + cos(x)), (x, 0, 2*pi))
2*sqrt(3)*pi/3
>>> definite_integral(exp(cos(x))*cos(sin(x)), (x, 0, 2*pi))
2*pi
>>> definite_integral(exp(-(x - 1/x)**2), (x, 0, oo))
sqrt(pi)/2
>>> from sympy import cosh, Abs
>>> definite_integral(x**2/cosh(x), (x, -oo, oo))
pi**3/4
>>> s = symbols('s', positive=True)
>>> definite_integral(Abs(sin(x))*exp(-s*x), (x, 0, oo))
1/((s**2 + 1)*tanh(pi*s/2))
>>> n = symbols('n', positive=True)
>>> definite_integral(1/(1 + x**n), (x, 0, oo), n > 1)
pi/(n*sin(pi/n))
>>> from sympy_extras.integrals import ramanujan_master_theorem, recognize_integral
>>> s = symbols('s')
>>> ramanujan_master_theorem(exp(-x**2), x, s)
MellinTransform(gamma(s/2)/2, (0, oo))
>>> recognize_integral(1/(x**3 + 1), (x, 0, 1))
log(2)/3 + sqrt(3)*pi/9
>>> from sympy_extras.integrals import asymptotic_integral
>>> asymptotic_integral(exp(-t*x)/(1 + x), x, 0, oo, t)
2/t**3 - 1/t**2 + 1/t + O(t**(-4), (t, oo))

```

## The Risch algorithm (`sympy_extras.integrals.risch`)

The transcendental Risch algorithm of SymPy, extended by Aaron Meurer's
unmerged pull requests: sympy/sympy#30180 and #30221 (the remaining
exp-log cases of Bronstein's book: the cancellation cases of the Risch
differential equation and the parametric problems), #30292 (the
hypertangent cases, with the coupled differential system of `cde.py`),
ported into `sympy_extras/integrals/risch/` with SymPy's BSD licence and
the attribution in every file; #30239 (radicals through the
transcendental algorithm, experimental) and #30282 (type annotations)
are documented there and not ported. The port is self-contained on
SymPy 1.14 and keeps the tests of the branches. `risch_antiderivative`
returns an antiderivative or `None` (nonelementary, or a case still
unimplemented); `is_nonelementary` is the decision. The ported modules
are not strictly typed: they are excluded from mypy's checks by a
per-module override in `pyproject.toml`, an exception to the typing
rule of `AGENTS.md` made for ported code.

```python
>>> from sympy import symbols, tan, exp, log
>>> from sympy_extras.integrals import risch_antiderivative, is_nonelementary
>>> x = symbols('x')
>>> risch_antiderivative(tan(x)**5, x)
log(tan(x)**2 + 1)/2 + tan(x)**4/4 - tan(x)**2/2
>>> risch_antiderivative(exp(x)/((exp(x) + 1)**2 + 1), x)
atan(exp(x) + 1)
>>> risch_antiderivative(1/(x*(log(x)**2 + 1)), x)
atan(log(x))
>>> is_nonelementary(exp(-x**2), x)
True

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
the variable order and give a case distinction on their cells. Two
shortcuts come before the decomposition: a disc, annulus or ball with an
integrand depending on the distance from the origin only is integrated in
polar or spherical coordinates, and a condition whose relations are
linear in the last variable (`y < exp(x)`, `y*exp(x) < 1`, which the CAD
cannot take) is integrated by Fubini's theorem with the bounds solved
for that variable and the crossings of the bounds found by `solve`.

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
>>> integrate_by_ranges(exp(-x**2 - y**2), x**2 + y**2 < 1)
-pi*exp(-1) + pi
>>> integrate_by_ranges(1, (x > 0) & (x < 1) & (y > 0) & (y < exp(x)))
-1 + E

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
