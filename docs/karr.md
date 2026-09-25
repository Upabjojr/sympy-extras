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

## Definite sums: Zeilberger's algorithm and WZ certificates

`sympy_extras.concrete.zeilberger` adds creative telescoping for
definite sums of hypergeometric terms `F(n, k)`: a recurrence
`sum_j a_j(n) S(n + j) = 0` for `S(n) = sum_k F(n, k)` with its rational
certificate (`zeilberger`), Wilf–Zeilberger certificates proving
identities `sum_k F(n, k) = f(n)` (`wz_certificate`, `wz_prove`), and
closed forms of definite sums through the recurrence and SymPy's
`rsolve` (`zeilberger_sum`). SymPy has Gosper's algorithm for indefinite
sums and closed forms of hypergeometric functions for some definite
ones, but no creative telescoping: Dixon's sum and Apéry's numbers get no
recurrence from `summation`.

```python
>>> from sympy import binomial, symbols
>>> from sympy_extras.concrete import zeilberger, zeilberger_sum, wz_prove
>>> n, k = symbols('n k', integer=True)
>>> zeilberger(binomial(n, k)**2, n, k)
Telescoper([-2*(2*n + 1), n + 1], k**2*(2*k - 3*n - 3)/(k - n - 1)**2)
>>> zeilberger_sum((-1)**k*binomial(2*n, k)**3, (k, 0, 2*n))
(-1)**n*factorial(3*n)/factorial(n)**3
>>> zeilberger_sum(binomial(n, k)**2*binomial(n + k, k)**2, (k, 0, n))
Eq((n + 1)**3*S(n) + (n + 2)**3*S(n + 2) - (2*n + 3)*(17*n**2 + 51*n + 39)*S(n + 1), 0)
>>> wz_prove(binomial(n, k)**2, binomial(2*n, n), n, k)
True

```

## Definite sums in ΠΣ-fields: creative telescoping

`sympy_extras.concrete.creative` extends creative telescoping from
hypergeometric terms to the summands of Karr's algorithm, following
Schneider (*Symbolic summation assists combinatorics*, 2007). The shifts
`f(n + i, k)` of the summand are written in one ΠΣ-field of `k` whose
constants are the rational functions of `n`: the hypergeometric part of
the summand is a Π-extension (`binomial(n + 1, k)` is
`binomial(n, k)*(n + 1)/(n + 1 - k)`), harmonic numbers and nested sums
are Σ-extensions. Karr's parameterized telescoping then gives the
recurrence `sum_i c_i(n) f(n + i, k) = g(n, k + 1) - g(n, k)` with its
certificate `g`, for the least order (`creative_telescoping`).

```python
>>> from sympy import binomial, harmonic, symbols
>>> from sympy_extras.concrete import creative_telescoping, definite_sum
>>> n, k = symbols('n k')
>>> Z = creative_telescoping(binomial(n, k)*harmonic(k), k, n)
>>> Z.coefficients
[4*n + 4, -4*n - 6, n + 2]
>>> Z.check()
True
>>> definite_sum(binomial(n, k)*harmonic(k), (k, 0, n))
2**n*(harmonic(n) - Sum(1/(2**j*j), (j, 1, n)))
>>> definite_sum(binomial(n, k)**2*harmonic(k), (k, 0, n))
(2*harmonic(n) - harmonic(2*n))*binomial(2*n, n)
>>> definite_sum(harmonic(k)/(n + 1 - k), (k, 1, n))
harmonic(n + 1)**2 - harmonic(n + 1, 2)
>>> definite_sum((1 + 3*(n - 2*k)*harmonic(k))*binomial(n, k)**3, (k, 0, n))
(-1)**n

```

`definite_sum(f, (k, a, b))` takes limits `m*n + s`. The relation is
summed over the range common to all the shifted sums, whose ends are moved
inwards past the poles of the certificate's representation (the
certificate of `binomial(n, k)` has the factor `1/(k - n - 1)`), and the
terms left out are added explicitly. The recurrence is solved in
d'Alembertian terms: a hypergeometric solution of the homogeneous
recurrence (SymPy's `rsolve_hyper`) reduces the order, a first order
recurrence is a product and a sum, and the sums are evaluated with Karr's
algorithm or left unevaluated when they have no closed form in their field
(the `Sum` above). A recurrence in steps of two gives a result for each
residue class of `n`. The constants come from the sum computed directly,
and the closed form is compared with the sum at further values of `n`;
where it fails for small `n` the result is a `Piecewise`:

```python
>>> definite_sum((-1)**k*binomial(n, k)*harmonic(k), (k, 0, n))
Piecewise((0, Eq(n, 0)), (-1/n, True))
>>> definite_sum((-1)**k*binomial(n, k)**2, (k, 0, n))
Piecewise((I**n*binomial(n, n/2), Eq(Mod(n, 2), 0)), (0, True))

```

`None` means that no certified closed form was found: there is no
recurrence of order at most `order` (4) in the field, its solutions are
not d'Alembertian (Apéry's numbers), a pole of the certificate lies inside
the summation range (`sum binomial(n, 2*k)` up to `n`, whose summand
vanishes from `n/2` on), or the summand is outside the field
(`harmonic(n + k)`: `n` may only occur in the hypergeometric and rational
parts). `summation` tries `definite_sum` when Karr's algorithm fails on a
summand depending on a limit.

## Rational sums: Abramov's decomposition

`abramov_decomposition(f, k)` writes a rational function as
`f = g(k+1) - g(k) + h` with `h` of minimal denominator (the irreducible
factors of the denominator grouped into shift classes and moved to one
representative each); `f` has a rational indefinite sum exactly when
`h = 0`, and `rational_sum` uses `g` for the rational part and SymPy's
polygamma functions for the rest.

```python
>>> from sympy_extras.concrete import abramov_decomposition, rational_sum
>>> abramov_decomposition(1/(k*(k + 1)), k)
RationalDecomposition(-1/k, 0)
>>> abramov_decomposition(1/k, k)
RationalDecomposition(0, 1/k)
>>> rational_sum(1/(k*(k + 2)), (k, 1, n))
n*(3*n + 5)/(4*(n + 1)*(n + 2))

```

## q-analogues: q-Gosper and q-Zeilberger

`sympy_extras.concrete.qhyper` adds q-Pochhammer symbols
(`QPochhammer(a, q, k)`, `qbinomial`), the q-Gosper algorithm for
indefinite sums of q-hypergeometric terms (terms whose ratio
`t(k+1)/t(k)` is rational in `q**k`) and the q-Zeilberger algorithm for
their definite sums. SymPy has no q-summation at all.

```python
>>> from sympy import factor
>>> from sympy.abc import q, k, n, x
>>> from sympy_extras.concrete import qgosper_sum, qzeilberger, qbinomial
>>> factor(qgosper_sum(q**k, (k, 0, n), q))
(q**(n + 1) - 1)/(q - 1)
>>> qzeilberger(qbinomial(n, k, q)*q**(k*(k - 1)/2)*x**k, n, k, q).coefficients
[-q**n*x - 1, 1]

```

The second example is the q-binomial theorem: the sum `S(n)` satisfies
`S(n + 1) = (1 + x q**n) S(n)`.

## Reference

- `karr_sum(f, (k, a, b), extensions=(), auto=True)`: the definite sum, or
  `None`; with `k` a symbol, the indefinite sum.
- `karr_term(f, k, extensions=(), auto=True)`: `g` with
  `g(k + 1) - g(k) == f(k)`, or `None`.
- `summation(f, *limits, extensions=(), auto=True)`: SymPy's summation
  followed by Karr's algorithm.
- `build_pisigma_field(f, k, extensions=())`: the field and the element
  representing `f`.
- `creative_telescoping(f, k, n, order=4, extensions=())`: a
  `CreativeTelescoper` with the `coefficients` `c_i(n)`, the `certificate`
  `g(n, k)`, `recurrence()` and `check()`, or `None`.
- `definite_sum(f, (k, a, b), n=None, order=4, extensions=())`: the
  definite sum by creative telescoping, or `None`.
- `PiSigmaField(k, params)`: the tower, with `add_sigma(beta, expr)`,
  `add_pi(alpha, expr)`, `sigma(f, power=1)`, `solve(a, fs)`,
  `telescope(f)`, `from_expr`, `to_expr`.

## References

- M. Karr, *Summation in finite terms*, J. ACM 28 (1981) 305-350.
- M. Karr, *Theory of summation in finite terms*, J. Symbolic Computation 1
  (1985) 303-315.
- C. Schneider, *Symbolic summation assists combinatorics*, Séminaire
  Lotharingien de Combinatoire 56 (2007), B56b.
- C. Schneider, *A refined difference field theory for symbolic
  summation*, J. Symbolic Computation 43 (2008) 611-644.
- D. Zeilberger, *The method of creative telescoping*, J. Symbolic
  Computation 11 (1991) 195-204.
- S. A. Abramov, M. Petkovšek, *D'Alembertian solutions of linear
  differential and difference equations*, ISSAC 1994.
- P. Paule, C. Schneider, *Computer proofs of a new family of harmonic
  number identities*, Adv. Appl. Math. 31 (2003) 359-378.
- S. A. Abramov, *Rational solutions of linear difference and q-difference
  equations with polynomial coefficients*, ISSAC 1995.
