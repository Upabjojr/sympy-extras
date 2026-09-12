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
  `1/(exp(x) - 1)`, `1/(exp(x) + 1)`, `1/sinh`, `1/cosh`, the Airy
  function `Ai`, the polylogarithms `Li_n(-x)`, the Fresnel integrals `S`
  and `C`, `erfc(x)*exp(x**2)`, the products of two Bessel functions of
  one argument `J_mu*J_nu`, `K_mu*K_nu`, `I_mu*K_nu`, `J_nu*Y_nu`,
  `J_nu*K_nu` as one kernel each (so `exp(-a*x)*besselj(0, x)**2` is a
  product of two, `2*elliptic_k(-4/a**2)/(pi*a)`), and the differences
  `exp(-x) - 1`,
  `cos(x) - 1`, `sin(x) - x`, `atan(x) - pi/2` whose transforms continue
  the strips), and the matching of an integrand against it
  (`mellin_transform`, `mellin_kernel`);
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

A trigonometric or hyperbolic factor next to two kernels is written as
exponentials with complex scales, `sin(b x)` as `(exp(I b x) -
exp(-I b x))/(2 I)`, and each term is a product of two kernels; the
condition on the argument of the complex scale, `Abs(arg(a + I b)) <
pi/2`, is decided by its real part. With `regularize=True` the strips of
convergence are dropped and the gamma quotient is continued analytically:
Hadamard's finite part of a divergent Mellin-type integral (Marichev's
regularisation).

```python
>>> from sympy import besselj, I
>>> definite_integral(exp(-a*x)*sin(b*x)*besselj(0, x), (x, 0, oo))
I/(2*sqrt((a + I*b)**2 + 1)) - I/(2*sqrt((a - I*b)**2 + 1))
>>> definite_integral(exp(-2*x)*cos(x)*besselj(0, x), (x, 0, oo))
2**(3/4)*sqrt(sqrt(2) + 2)/8
>>> definite_integral(x**(-S(3)/2)*exp(-x), (x, 0, oo))
Integral(exp(-x)/x**(3/2), (x, 0, oo))
>>> definite_integral(x**(-S(3)/2)*exp(-x), (x, 0, oo), regularize=True)
-2*sqrt(pi)

```

The integrand is also read in rewritten forms: products and powers of
`sin` and `cos` as sums, hyperbolic functions as exponentials, inverse
hyperbolic functions as logarithms, orthogonal polynomials expanded;
`log(1 - x)` on `(0, 1)` is a kernel of its own. With
`principal_value=True` a divergent integral with a simple pole inside the
range gets its Cauchy principal value, from the antiderivative with the
symmetric limits at the pole taken as one limit, and with
`finite_part=True` Hadamard's finite part (the divergent terms of the
excision dropped, so a double pole is regularised); symbolic endpoints go
through the antiderivative with limits under the assumptions.

```python
>>> from sympy import sinh, atanh
>>> definite_integral(exp(-a*x)*sinh(b*x), (x, 0, oo), a > b)
b/((a - b)*(a + b))
>>> definite_integral(x*atanh(x), (x, 0, 1))
1/2
>>> definite_integral(1/(x - 1), (x, 0, 3), principal_value=True)
log(2)
>>> definite_integral(1/x**2, (x, -1, 1), finite_part=True)
-2
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
  `sympy_extras.concrete.zeilberger`. `sympy_extras.integrals.reduction`
  finds the same telescopers by Hermite reduction (Bostan, Chen, Chyzak, Li
  and Xin): the reduced forms are canonical, so the order is minimal and no
  bound on the certificate is needed; it is tried first, the ansatz second.
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
- **A table** (`sympy_extras.integrals.tables`): fifty-eight entries of
  Gradshteyn and Ryzhik which no algorithm here reproduces (the Poisson
  and Fejér kernels, `sin(a x)**3/x**3`, `x*log(sin(x))`, `log(gamma(x))`,
  products of Bessel functions, ...), matched with their conditions on the
  parameters and checked numerically at sample values in the tests; tried
  first because it is cheap.
- **Elliptic integrals** (`sympy_extras.integrals.elliptic`): square roots
  of cubics and quartics with real roots, `R(x)/sqrt(P(x))` and
  `R(x)*sqrt(P(x))` over a cell between the roots, reduced by the
  substitutions of Byrd–Friedman to Legendre's `elliptic_k`, `elliptic_e`,
  `elliptic_pi` and the incomplete `elliptic_f` (SymPy's parameter
  `m = k**2`); `Integral(1/sqrt(1 - x**4), (x, 0, 1))` gives
  `elliptic_k(1/2)`, the lemniscate constant. Radicands with a pair of
  complex roots (Byrd–Friedman 240, 241) are mapped onto `cos(theta)` by
  the bilinear substitution, any polynomial or rational numerator is
  reduced by the recurrences 310-318 (incomplete powers, multiple poles),
  the organising principle being Carlson's symmetric `R_F` (DLMF 19.25,
  19.29): `Integral(1/sqrt(x**3 + 1), (x, 0, oo))` is
  `gamma(1/6)*gamma(1/3)/(3*sqrt(pi))`. Quartics with no real roots
  (Byrd–Friedman 267) go through the bilinear map sending both quadratic
  factors to `(A*t**2 + B)/(1 + t)**2` and `t = lambda*tan(theta)`:
  `Integral(1/sqrt((x**2 + 1)*(x**2 + 4)), (x, -oo, oo))` is
  `elliptic_k(3/4)`; the irrational quadratic factors of a numeric
  quartic (`x**4 + x - 1`) are built from its `CRootOf` roots, the
  algebraic numbers travelling as dummies decided numerically.
- **Algebraic integrands of genus zero** (`sympy_extras.integrals.algebraic`):
  Euler's substitutions for `R(x, sqrt(a*x**2 + b*x + c))`, `t**n = M(x)`
  for roots of a Möbius function, and Chebyshev's three integrable cases
  of the binomial differential `x**m*(a + b*x**n)**p`; the rational
  integral in the new variable goes back to the driver (Trager's
  algorithm for the general algebraic case is not implemented).
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
  (`exp(-t*x)*phi(x)`), Laplace's method (`phi*exp(t*h)` with the maxima
  inside or at the endpoints, of any order, several maxima of equal
  height added, to any number of terms), the stationary phase, the
  uniform Airy and error-function expansions of Chester–Friedman–Ursell
  for stationary points which coalesce (`asymptotic_integral(...,
  uniform=True)`; the Airy integral `cos(t*(x**3/3 - a*x))` gives
  `2*pi*t**(-1/3)*airyai(-a*t**(2/3))` exactly) and steepest descent
  through the complex saddle points of `exp(t*h)` when the deformation
  of the contour is justified (`steepest_descent`);
  (`phi*exp(I*t*h)`, leading term), with the order term.
- **Validated numerical integration** (`sympy_extras.integrals.validated`,
  `definite_integral(..., numeric=True)`): when no closed form is found and
  the integral has no parameters, a value with a proved error bound from
  composite Simpson's rule or the five-point Gauss–Legendre rule, whichever
  bound is smaller on each panel, the derivatives enclosed in interval
  arithmetic in centred form (the Taylor shift to the midpoint of the
  panel), panels refined where the bound is largest; algebraic and
  logarithmic endpoint singularities (`x**alpha` with `alpha > -1`,
  `log(x)`) bounded by their closed forms or transformed away by a power
  substitution, removable singularities (`sin(x)/x`) by the Taylor
  remainder, infinite ranges by the exponential or rational map, with
  oscillatory tails integrated by parts (Bonnet's bound). Thirty digits of
  the Gaussian on `(0, 1)`, twenty of `1/(1 + x**2)` on `(0, oo)`.
- **Summability of divergent oscillatory integrals**
  (`sympy_extras.integrals.summability`, `definite_integral(...,
  summability='abel')`, `'cesaro'`, `'gaussian'`): the Abel mean
  `lim Integral(f*exp(-eps*x))`, the Cesàro means `(C, k)` with the
  kernel `(1 - x/R)**k` and the Gaussian mean, each through the inner
  integral under `eps > 0` and the limit under the assumptions; a value
  is kept only when the limit is finite and free of the regulator, so
  `sin(x)**2` over `(0, oo)` has none, and a convergent integral keeps its
  value (Hardy, *Divergent series*, ch. 4-5).
- **Term by term**: a sum which defeats every method is integrated term
  by term, the value kept only when every term is finite (terms which
  diverge separately may cancel in the sum) and the total passes the
  numerical check; the difference of two circular arcs,
  `sqrt(x)*sqrt(2 - x) - sqrt(1 - x)*sqrt(x + 1)` on `(1/2, 1)`, whose
  antiderivative is a complex `Piecewise`, gives `sqrt(3)/4 - pi/12`, and
  with it the union of two overlapping discs in `IntegralByRanges`.
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
>>> from sympy import besselj, gamma
>>> definite_integral(exp(-t*x)*besselj(0, x), (x, 0, oo))
1/sqrt(t**2 + 1)
>>> definite_integral(log(gamma(x)), (x, 0, 1))
log(2*pi)/2
>>> definite_integral(1/sqrt(1 - x**4), (x, 0, 1))
sqrt(pi)*gamma(1/4)/(4*gamma(3/4))
>>> definite_integral(log(x)*log(1 - x), (x, 0, 1))
2 - pi**2/6
>>> definite_integral(sqrt(x)/(1 + x)**2, (x, 0, oo))
pi/2
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
>>> definite_integral(x**x, (x, 0, 1), numeric=True, digits=6)
0.783430391

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
unimplemented); `is_nonelementary` is the decision. The ported modules are strictly
typed like the rest of the package, with the annotations of Aaron
Meurer's branch `risch-typing` (sympy/sympy#30282) as the starting point.

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

## Trager's algorithm (`sympy_extras.integrals.trager`)

`trager_antiderivative(f, x)` integrates a function rational in `x` and
in one square root `y = sqrt(P(x))`: the integrand is written
`A(x) + B(x)*y`, the rational part goes to the rational integration, and
the algebraic part through Trager's Hermite reduction with the integral
basis `{1, y/d}` (`d` the product of the repeated factors of `P`), which
leaves a remainder with simple poles only; the logarithmic part is found
from the Rothstein–Trager resultant of the residues, with a
logarithm `log(u + v*y)` of prescribed divisor built by Newton lifting
and a linear system, the torsion orders on a curve of genus one tried
up to Mazur's bound. `trager_reduce` returns the elementary part and the
remainder, `is_nonelementary_algebraic` decides the cases it can prove
(a nonzero remainder without residues is a differential of the first
kind). `definite_integral` uses the antiderivative through the
one-sided limits of the antiderivative route.

```python
>>> from sympy import symbols, sqrt, log
>>> from sympy_extras.integrals import trager_antiderivative, is_nonelementary_algebraic, definite_integral
>>> x = symbols('x')
>>> trager_antiderivative(x/sqrt(x**4 + 1), x)
log(x**2 + sqrt(x**4 + 1))/2
>>> trager_antiderivative(1/(x*sqrt(x**2 + 1)), x)
-log((sqrt(x**2 + 1) + 1)/x)
>>> is_nonelementary_algebraic(1/sqrt(x**3 + 1), x)
True
>>> definite_integral((x**2 - 1)/((x**2 + 1)*sqrt(x**4 + 1)), (x, 0, 1))
-sqrt(2)*pi/8

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
the variable order and give a case distinction on their cells. Shortcuts come before the decomposition: a disc, annulus or ball (or an
ellipse, ellipsoid or shifted disc, scaled to one) with an integrand
depending on the distance from the centre only is integrated in polar or
spherical coordinates, a cylinder-like region (a disc condition in `x`, `y`
and bounds on `z` linear in `z`) in cylindrical coordinates, and a
condition whose relations are linear in the last variable (`y < exp(x)`,
`y*exp(x) < 1`, which the CAD cannot take) is integrated by Fubini's
theorem with the bounds solved for that variable, the crossings of the
bounds found by `solve`, and unbounded outer variables allowed.

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
>>> integrate_by_ranges(1, x**2/4 + y**2/9 < 1)
6*pi
>>> integrate_by_ranges(1, (x**2 + y**2 < 1) & (z > x**2 + y**2) & (z < 1))
pi/2

```

With `measure='hausdorff'` the integral is taken with respect to the
`(n-1)`-dimensional Hausdorff measure on the hypersurface given by one
equation among the conditions (a curve in the plane, a surface in space):
the section of the decomposition is the explicit branch of the root, the
integrand is multiplied by the surface element `sqrt(1 + |grad phi|**2)`
and the remaining variables are integrated as before. Bounds which are
not polynomial but monotone in the last variable (`y**2 < exp(x)`) are
solved for it.

```python
>>> from sympy import Eq, log
>>> integrate_by_ranges(1, Eq(x**2 + y**2, 1), measure='hausdorff')
2*pi
>>> integrate_by_ranges(y, Eq(x**2 + y**2, 1) & (y > 0), measure='hausdorff')
2
>>> integrate_by_ranges(1, Eq(z, x**2 + y**2) & (z < 1), measure='hausdorff')
pi*(-1 + 5*sqrt(5))/6
>>> integrate_by_ranges(1, (x > 0) & (x < 1) & (y > 0) & (y**2 < exp(x)))
-2 + 2*exp(1/2)

```

A bounded polytope (linear inequalities with rational coefficients,
parameters allowed in the constant terms) with a polynomial integrand is
integrated exactly without the decomposition: the vertices are
enumerated from the facets, the polytope is triangulated by pulling, and
each simplex is mapped onto the standard simplex where the monomials
integrate by Dirichlet's formula. With `dimension=n`, `n` an integer or
a symbol, a radial integrand of the distance `r` integrates in `n`
dimensions through the area of the unit sphere:

```python
>>> integrate_by_ranges(x**2 + y**3, (x > 0) & (y > 0) & (x + y < 1) & (x + 2*y < S(3)/2) & (y - x < S(1)/2))
2521/25920
>>> n = symbols('n', positive=True, integer=True)
>>> integrate_by_ranges(1, r < 1, dimension=n)
pi**(n/2)/gamma(n/2 + 1)
>>> integrate_by_ranges(exp(-r**2), True, dimension=n)
pi**(n/2)

```

Several equations give curves (the Gram determinant of the graph
parametrisation is the line or surface element), and a cell bounded by
the root of a cubic gets the trigonometric or hyperbolic form of the root
when it is real, or the variables are reordered so that the bound is
explicit in another variable:

```python
>>> integrate_by_ranges(1, Eq(x**2 + y**2 + z**2, 1) & Eq(z, 0), measure='hausdorff')
2*pi
>>> integrate_by_ranges(1, Eq(y, x**2) & Eq(z, x) & (x > 0) & (x < 1), measure='hausdorff')
(asinh(sqrt(2)) + sqrt(6))/2
>>> integrate_by_ranges(1, (x > 0) & (y > 0) & (x**3 + y**3 < 1))
2**(1/3)*gamma(1/6)*gamma(1/3)/(12*sqrt(pi))

```

The Fourier series of the classical table (`log(sin(x))`, `log(cos(x))`,
`log(tan(x))`, `log(1 - cos(x))`, `x`, `x**2`, `Abs(sin(x))`, ...,
Gradshteyn–Ryzhik 1.441-1.444), the parametric entries `log(P + Q*cos(x))`
and `1/(P + Q*cos(x))` (the Poisson kernel, GR 1.447-1.448) and the
series computed for other factors (the coefficients as integrals with a
symbolic index, in the half-range or full orthogonal systems of `(a, b)`)
are integrated termwise over their ranges of validity, against each
other and against harmonics by orthogonality, and against another factor
through the moments of the harmonics (Parseval's theorem justifies the
interchange for two square-integrable factors):

```python
>>> from sympy import log
>>> definite_integral(log(sin(x)), (x, 0, pi))
-pi*log(2)
>>> definite_integral(x*log(sin(x)), (x, 0, pi/2))
-pi**2*log(2)/8 + 7*zeta(3)/16
>>> definite_integral(log(sin(x))*log(cos(x)), (x, 0, pi/2))
pi*(-pi**2 + 24*log(2)**2)/48
>>> from sympy_extras.integrals.series import fourier_integral
>>> a = symbols('a', positive=True)
>>> fourier_integral(x**2/(1 - 2*a*cos(x) + a**2), x, 0, pi, [a < 1]).value
pi*(-12*polylog(2, -a) - pi**2)/(3*(a**2 - 1))

```

## What is not done

Tracked in issue #53 of the repository, with the alternatives worth adding.

- Products of three or more functions of the table (a double
  Mellin–Barnes integral, not a G-function) unless one of them is
  trigonometric or hyperbolic, and functions outside the
  table (`sinh`, `cosh` and `besseli` without the exponential which makes
  them decay, inverse hyperbolic functions, `LambertW`, the Struve
  functions, which SymPy lacks, ...): the driver falls back to SymPy.
- A G-function on the unit circle `|z| = 1` (products of two kernels with
  equal scales, `airyai(x)**2`) whose Slater series diverge there: the
  analytic continuation of the sum is not attempted.
- The logarithmic cases of Slater's theorem (parameters of the G-function
  differing by integers) are computed as limits of the general case, which
  SymPy's `limit` does not always manage.
- Principal values of divergent integrals.
- Complex parameters: the conditions are written for real parameters
  (an inequality on a parameter states that it is real); a parameter which
  may be complex gets a condition on its argument, `Abs(arg(a)) < pi/2`.
- Region integrals whose cell boundaries have no explicit form in any
  order of the variables (roots of polynomials of degree five or more
  with parametric coefficients), and regions described by non-polynomial
  conditions which are not monotone in the last variable.

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
