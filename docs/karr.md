# Karr's algorithm for symbolic summation

Module: `sympy_extras.concrete`

SymPy evaluates sums with Gosper's algorithm (`sympy.concrete.gosper`), which
handles *hypergeometric* summands (ratios `f(k+1)/f(k)` rational in `k`:
factorials, binomials, powers), with a few heuristics on top. It only
mentions Karr in the *summation convention* it follows for the limits: the
algorithm of Karr (1981) itself, which extends Gosper's algorithm to sums
whose summands contain *sums* such as harmonic numbers, nested sums, and
products of all of them, is not implemented in SymPy. This module
implements it.

```python
>>> from sympy import harmonic, factorial, binomial, Sum
>>> from sympy.abc import j, k, n
>>> from sympy_extras.concrete import karr_sum, karr_term, summation
>>> karr_sum(harmonic(k), (k, 1, n))
n*harmonic(n) - n + harmonic(n)
>>> karr_sum(harmonic(k)**2, (k, 1, n))
n*harmonic(n)**2 - 2*n*harmonic(n) + 2*n + harmonic(n)**2 - harmonic(n)
>>> karr_sum(harmonic(k)/k, (k, 1, n))
(harmonic(n)**2 + harmonic(n, 2))/2
>>> karr_sum(Sum(1/j**2, (j, 1, k)), (k, 1, n))
n*harmonic(n, 2) - harmonic(n) + harmonic(n, 2)
>>> karr_sum(k*factorial(k), (k, 1, n))
n*factorial(n) + factorial(n) - 1
>>> karr_sum(binomial(2*k, k)/4**k, (k, 0, n))
(2*n + 1)*binomial(2*n, n)/4**n
>>> karr_term(harmonic(k)/(k + 1), k)
(harmonic(k)**2 - harmonic(k, 2))/2

```

`summation` calls SymPy's `summation` first and Karr's algorithm on the
sums SymPy leaves unevaluated.

## ΠΣ-fields

Karr's algorithm works in a *ΠΣ-field*: a tower of fields
$\mathbb{C} \subset \mathbb{C}(k) \subset \mathbb{C}(k)(t_1) \subset \cdots$
with the shift $\sigma$ ($k \to k + 1$) acting as an automorphism, where each
generator is either a *Π-extension*, $\sigma(t) = \alpha t$ (products like
$k!$, $2^k$, $\binom{2k}{k}$: $\alpha$ is the ratio $t(k+1)/t(k)$), or a
*Σ-extension*, $\sigma(t) = t + \beta$ (sums like $H_k$ with
$\beta = 1/(k+1)$, or nested sums). The constants $\mathbb{C}$ are the
rationals with the parameters (symbols other than the index) adjoined.

The summand is analysed by `build_pisigma_field`: every product or sum in it
becomes a generator, unless it is already expressible in the field built so
far, which is decided by the algorithm itself (`factorial(k + 1)` is
recognised as `(k + 1)*factorial(k)`, `harmonic(k + 1)` as
`harmonic(k) + 1/(k + 1)`, a `Sum(1/j, (j, 1, k))` as `harmonic(k)`).
Harmonic numbers with arguments `a*k + b` are Σ-extensions too:

```python
>>> karr_sum(harmonic(2*k), (k, 1, n))
n*harmonic(2*n) - n + harmonic(n)/4 + harmonic(2*n)/2

```

```python
>>> from sympy_extras.concrete import build_pisigma_field
>>> F, f = build_pisigma_field(harmonic(k + 1)*factorial(k), k)
>>> F.extensions
[Extension(pi, factorial(k), k + 1), Extension(sigma, harmonic(k), 1/(k + 1))]
>>> F.to_expr(f)
(k*harmonic(k) + harmonic(k) + 1)*factorial(k)/(k + 1)

```

Then the telescoping equation $\sigma(g) - g = f$ is solved by recursion
over the tower. In the top generator $t$ the unknown $g$ is a polynomial
(Σ-case) or a Laurent polynomial (Π-case) in $t$ with a degree bound, and
its coefficients are found from the top down by solving equations
$\sigma(g) - a\,g = c_1 f_1 + \cdots + c_r f_r$ of the same shape one level
lower, for $g$ *and* the constants $c_i$ (the *parameterized* first order
equation, which is what makes the recursion close). At the bottom, in
$\mathbb{C}(k)$, the rational solutions are found with Abramov's universal
denominator and a degree bound for the numerator, and the linear systems
are solved over the constants. The `PiSigmaField` class exposes this
machinery (`solve`, `telescope`, `sigma`, `add_sigma`, `add_pi`).

The decomposition of $\sum_{k=a}^{b} f(k) = g(b+1) - g(a)$ follows Karr's
summation convention, the one used by SymPy's `Sum`.

## Deciding that no closed form exists

Karr's algorithm is a decision procedure: when it returns no solution there
is no telescoper in the field, so no closed form in terms of the sequences
of the field.

```python
>>> karr_sum(2**k/k, (k, 1, n)) is None
True
>>> karr_sum(factorial(k)/k, (k, 1, n)) is None
True
>>> karr_sum(harmonic(k)*2**k, (k, 1, n)) is None
True

```

One restriction applies: in a Σ-extension only summands polynomial in the
generator are handled (so not `1/harmonic(k)`), and in a Π-extension only
Laurent polynomials (not `1/(2**k + 1)`). For Σ-extensions a theorem of
Karr shows that the solution is then polynomial too, so this does not
weaken the decision.

A closed form may need generators which do not occur in the summand:
$\sum_{k \le n} H_k/k = (H_n^2 + H_n^{(2)})/2$ needs $H^{(2)}$. Karr's
algorithm does not invent extensions, so when the telescoping fails for a
summand with harmonic numbers or nested sums, `karr_sum` retries with the
harmonic numbers of orders up to the highest order plus the degree of the
summand adjoined (`auto=True`); further sequences can be given with
`extensions=[...]`. What cannot be found this way is reported as `None`:
$\sum H_k^2/k$, for instance, needs the nested sum $\sum_j H_j^{(2)}/j$,
which is not a harmonic number.

## Reference

- `karr_sum(f, (k, a, b), extensions=(), auto=True)`: the definite sum, or
  `None`; with `k` a symbol, the indefinite sum.
- `karr_term(f, k, extensions=(), auto=True)`: `g` with
  `g(k + 1) - g(k) == f(k)`, or `None`.
- `summation(f, *limits, extensions=(), auto=True)`: SymPy's summation
  followed by Karr's algorithm.
- `build_pisigma_field(f, k, extensions=())`: the field and the element
  representing `f`.
- `PiSigmaField(k, params)`: the tower, with `add_sigma(beta, expr)`,
  `add_pi(alpha, expr)`, `sigma(f, power=1)`, `solve(a, fs)`,
  `telescope(f)`, `from_expr`, `to_expr`.

## References

- M. Karr, *Summation in finite terms*, J. ACM 28 (1981) 305-350.
- M. Karr, *Theory of summation in finite terms*, J. Symbolic Computation 1
  (1985) 303-315.
- C. Schneider, *Symbolic summation assists combinatorics*, Séminaire
  Lotharingien de Combinatoire 56 (2007), B56b.
- S. A. Abramov, *Rational solutions of linear difference and q-difference
  equations with polynomial coefficients*, ISSAC 1995.
