# What sympy-extras answers and SymPy does not

Seventy-nine questions, each put to SymPy alone and to `sympy-extras` on
top of it. The results below were produced by
[`benchmarks/comparison.py`](../benchmarks/comparison.py), which asks both
libraries and prints what they return, against **SymPy 1.14.0** and
sympy-extras 0.0.1. Regenerate the raw material with

```
python benchmarks/comparison.py                 # a table
python benchmarks/comparison.py --markdown      # this document's examples
```

The examples of sympy-extras are doctests and are run by the test suite
(`python -m pytest --doctest-glob='*.md' docs`), so the left hand column
cannot drift. The SymPy answers are quoted from the same run and are
labelled by what SymPy does with the question:

| | | count |
| --- | --- | --- |
| **no answer** | SymPy raises (`NotImplementedError`, `ValueError`, `KeyError`, `TypeError`) or has no counterpart | 39 |
| **undecided** | it returns the question unevaluated: `Sum(...)`, `ConditionSet`, `None`, `nan`, an unchanged expression | 26 |
| **partial** | it answers, but less completely (a truncated series instead of a solution, one branch of two, a value under conditions it cannot resolve) | 10 |
| **wrong** | it returns something which is not correct | 4 |

By area:

| Area | Examples | What is added |
| --- | --- | --- |
| [Assumptions and real algebra](#assumptions-and-real-algebra) | 13 | cylindrical algebraic decomposition, virtual substitution, interval arithmetic, polynomial bounds |
| [Solving equations](#solving-equations) | 14 | exact real solution sets, transcendental root objects, Diophantine systems, Thue equations |
| [Differential equations](#differential-equations) | 16 | Kovacic, Abramov–Bronstein–Petkovšek, Lie symmetries, integrating factors, Abel invariants, DAEs |
| [Sums, products and series](#sums-products-and-series) | 14 | Karr, Zeilberger, q-analogues, convergence with parameters, Euler and Dirichlet series |
| [Limits and series expansions](#limits-and-series-expansions) | 5 | limits with statement assumptions and case distinctions |
| [Polynomials and ideals](#polynomials-and-ideals) | 8 | ideal operations, the Gröbner walk, subresultant coefficients |
| [Definite integration](#definite-integration) | 9 | Mellin transforms as gamma quotients, Parseval's formula and Slater's theorem (the Marichev–Adamchik method), splitting at kinks and singularities, region integrals through the CAD |

Nothing here is a criticism of SymPy's implementations: most of these
questions need algorithms which are simply not in SymPy, and four are
bugs worth reporting upstream (they are collected at the end, with two
more found by `benchmarks/fuzz.py`).

The setup used throughout:

```python
>>> from sympy import *
>>> from sympy.abc import a, b, c, k, n, p, q, s, t, x, y, z
>>> from sympy_extras.assumptions import (ask, refine, simplify, resolve, solve, element,
...     ForAll, Exists, satisfiable, find_instance, limit, limit_seq, series)
>>> from sympy_extras.concrete import (karr_sum, zeilberger_sum, wz_prove, sum_convergence,
...     product_convergence, dirichlet_series, polygamma_series, qgosper_sum, qzeilberger)
>>> from sympy_extras.concrete.qhyper import qbinomial
>>> from sympy_extras.solvers import (dsolve_kovacic, dsolve_linear, special_solutions, dsolve_lie,
...     dsolve_first_order, dsolve_second_order, dsolve_dae, dsolve_linear_system, complete_integral,
...     isolate_real_roots, thue, abel_by_invariants, linear_diophantine_system, is_linearizable,
...     minimal_nonnegative_solutions, symmetries, pde_symmetries, pdsolve_lie)
>>> from sympy_extras.polys.ideals import Ideal
>>> from sympy_extras.polys.euclidtools import psc
>>> from sympy_extras.polys.groebnerwalk import groebner_walk
>>> from sympy_extras.polys.virtual_substitution import eliminate_linear
>>> from sympy_extras.integrals import definite_integral, IntegralByRanges
>>> from sympy_extras.integrals import mellin_transform as xmellin
>>> f = Function('y')(x)
>>> g = Function('u')(x, y)

```

## Assumptions and real algebra

### Quantifier elimination over the reals

```python
>>> resolve(ForAll(x, x**2 + b*x + c > 0))
b**2 < 4*c
>>> resolve(Exists(x, Eq(a*x**2 + b*x + c, 0)))
Eq(c, 0) | (4*a*c - b**2 < 0) | ((a > 0) & Eq(4*a*c - b**2, 0)) | ((a < 0) & (4*a*c - b**2 <= 0))

```

SymPy has no quantifier elimination. The cylindrical algebraic
decomposition is the only complete decision procedure for the elementary
theory of the reals, and it is what `resolve` runs (with virtual
substitution first for the variables which occur linearly).

### A polynomial inequality under polynomial assumptions

```python
>>> ask(x**2 + y**2 >= 2*x*y, (x > 0) & (y > 0))
True

```

SymPy: `ask(Q.nonnegative(x**2 + y**2 - 2*x*y), Q.positive(x) & Q.positive(y))` → `None`.

### An inequality between elementary functions

```python
>>> ask(sin(x) < x, x > 0)
True
>>> ask(exp(x) >= 1 + x, element(x, S.Reals))
True

```

SymPy: `ask(Q.negative(sin(x) - x), Q.positive(x))` → `None`, and
`ask(Q.nonnegative(exp(x) - 1 - x), Q.real(x))` → `None`. Here the
elementary functions are replaced by variables constrained by polynomial
bounds (MetiTarski's method) and the result is decided by the CAD.

### Reality of a square root (a SymPy bug)

```python
>>> ask(element(sqrt(a - 2), S.Reals), a > 0)

```

Nothing is printed: the answer is `None`, correctly, because `sqrt(a - 2)`
is real for `a = 3` and not for `a = 1`. SymPy:
`ask(Q.real(sqrt(a - 2)), Q.positive(a))` → `True`, which is wrong; its
`Q.real` handler for powers does not look at the sign of the base. At
`a = 1` the root is `I`.

### Refining expressions whose sign matters

```python
>>> refine(Abs(x - 1) + sqrt(x**2), x > 2)
2*x - 1
>>> refine(sqrt(x**2 - 2*x + 1), x > 1)
x - 1

```

SymPy: `refine(Abs(x - 1) + sqrt(x**2), Q.positive(x - 2))` →
`sqrt(x**2) + Abs(x - 1)`, and
`refine(sqrt(x**2 - 2*x + 1), Q.positive(x - 1))` → unchanged. SymPy's
handlers look at the syntactic form of the argument; here the sign of
each polynomial factor is decided by the CAD first.

### Parity and residues of integer polynomials

```python
>>> refine((-1)**(n**2 + n), element(n, S.Integers))
1
>>> refine(Mod(n**3 - n, 6), element(n, S.Integers))
0

```

SymPy: both unchanged (`(-1)**(n**2 + n)` and `Mod(n**3 - n, 6)`). The
residues of an integer polynomial are the same at every integer point
exactly when they agree on the residues modulo the modulus, which is a
finite check.

### Simplifying a relation under assumptions

```python
>>> simplify(Eq(x**2, 1), x > 0)
Eq(x, 1)

```

SymPy: `refine(Eq(x**2, 1), Q.positive(x))` → `Eq(x**2, 1)`.

### Satisfiability of a nonlinear real system (a SymPy bug)

```python
>>> satisfiable((x**2 + y**2 < 1) & (x*y > Rational(1, 2)))
False
>>> find_instance((x**2 + y**2 < 1) & (x*y > Rational(1, 5)), [x, y])
[{x: -1/2, y: -1/2}]

```

SymPy: `satisfiable((x**2 + y**2 < 1) & (x*y > Rational(1, 2)))` →
`{Q.gt(x*y, 1/2): True, Q.lt(x**2 + y**2, 1): True}`. Its SAT solver is
propositional and treats each inequality as an opaque literal, so it
reports a model of a system which has no real solution at all
(`x**2 + y**2 >= 2*Abs(x*y) > 1`). There is no counterpart of
`find_instance`.

## Solving equations

### A logarithmic equation over the reals (a SymPy bug)

```python
>>> solve(Eq(log(x) + log(x - 1), log(2)), x, domain=S.Reals)
{2}

```

SymPy: `solveset(Eq(log(x) + log(x - 1), log(2)), x, S.Reals)` →
`{-1, 2}`. At `x = -1` the left hand side is `log(2) + 2*I*pi`, so `-1`
is not a solution, and neither logarithm is real there.

### A radical equation (a SymPy bug)

```python
>>> solve(2*x**2 + 3*sqrt(x + 6) - 1, x, domain=S.Reals)
EmptySet

```

SymPy: `solveset(2*x**2 + 3*sqrt(x + 6) - 1, x, S.Reals)` returns an
unevaluated `Intersection` of four candidates with the reals. Squaring
without checking is what produces them: the residual of the equation at
the first two is `17.17` and `12.17`, and the other two are not real. The
real solution set is empty.

### Equations with no closed form solution

```python
>>> dottie = solve(Eq(x, cos(x)), x, domain=S.Reals)
>>> dottie
{TranscendentalRoot(x - cos(x), x, 5/8, 3/4)}
>>> N(list(dottie)[0], 25)
0.7390851332151606416553121
>>> isolate_real_roots(exp(x) - x - 2, x)
[TranscendentalRoot(-x + exp(x) - 2, x, -61/32, -29/16), TranscendentalRoot(-x + exp(x) - 2, x, 17/16, 37/32)]

```

SymPy: `ConditionSet(x, Eq(x - cos(x), 0), Reals)` in both cases. A
`TranscendentalRoot` is a real number isolated in an interval which is
proved to contain exactly one root; it evaluates to any precision and
compares with other numbers.

### An inequality with transcendental endpoints

```python
>>> solve(cos(x) - x**2/4 <= 0, x, domain=S.Reals)
Union(Interval(-oo, TranscendentalRoot(x**2/4 - cos(x), x, -5/4, -9/8)), Interval(TranscendentalRoot(x**2/4 - cos(x), x, 9/8, 5/4), oo))

```

SymPy: `ConditionSet(x, -x**2/4 + cos(x) <= 0, Reals)`.

### Both real branches of a Lambert equation

```python
>>> solve(exp(x) - x - 2, x, domain=S.Reals)
{-2 - LambertW(-exp(-2)), -2 - LambertW(-exp(-2), -1)}

```

SymPy: `solve(exp(x) - x - 2, x)` → `[-2 - LambertW(-exp(-2))]`. Only the
principal branch, so the equation looks as if it had one real root
instead of two.

### Assumptions and conjunctions in the solution set

```python
>>> solve(x**2 - a, x, (x > 0) & (a > 0))
{sqrt(a)}
>>> solve((x**3 - 2*x > 0) & (x < 3), x, domain=S.Reals)
Union(Interval.open(-sqrt(2), 0), Interval.open(sqrt(2), 3))

```

SymPy: `solveset(x**2 - a, x, S.Reals)` → `Intersection({-sqrt(a), sqrt(a)}, Reals)`,
and `solveset` on the conjunction raises
`ValueError: (x < 3) & (x**3 - 2*x > 0) is not a valid SymPy expression`.

### Systems over the integers

```python
>>> linear_diophantine_system([Eq(2*x + 3*y + 5*z, 7), Eq(x + y + z, 2)], [x, y, z])
([3 - 2*t0, 3*t0 - 3, 2 - t0], [t0])
>>> minimal_nonnegative_solutions([[3, -5]], [1])
([[2, 1]], [[5, 3]])
>>> resolve(ForAll(x, Exists(y, Eq(3*y, x) | Eq(3*y, x + 1) | Eq(3*y, x + 2))), domain=S.Integers)
True

```

SymPy's `diophantine` takes one equation at a time (there is no solver
for systems), and has no counterpart of the Contejean–Devie minimal
nonnegative solutions or of Presburger arithmetic.

### Thue equations

```python
>>> thue(x**3 + x**2*y - 2*x*y**2 - y**3, 1, x, y)
[(-9, 5), (-1, -1), (-1, 1), (-1, 2), (0, -1), (1, 0), (2, -1), (4, -9), (5, 4)]
>>> thue(x**3 - 2*y**3, 1, x, y)
[(-1, -1), (1, 0)]

```

SymPy: `diophantine(...)` raises
`NotImplementedError: No solver has been written for cubic_thue.` The
nine solutions of Thomas's cubic are complete, and proved complete:
Baker–Wüstholz bounds the exponents of the unit part by about `10^18`,
de Weger's lattice reduction brings the bound down to a few dozens, and
what is left is enumerated.

### Reduction over the complex numbers

```python
>>> resolve(Exists(x, Eq(a*x**2 + b*x + c, 0) & Ne(x, 0)), domain=S.Complexes)
(Ne(a, 0) & Ne(c, 0)) | (Eq(a, 0) & Eq(b, 0) & Eq(c, 0)) | (Eq(a, 0) & Ne(b, 0) & Ne(c, 0)) | (Eq(c, 0) & Ne(a, 0) & Ne(b, 0))

```

No counterpart in SymPy. The parameter space is covered by a
comprehensive Gröbner system (Kapur–Sun–Wang).

## Differential equations

### Linear equations: closed forms where SymPy returns a power series

```python
>>> dsolve_kovacic(f.diff(x, 2) - (x**2 + 3)*f, f)
[x*exp(x**2/2), (-sqrt(pi)*x*exp(x**2)*erf(x) - 1)*exp(-x**2/2)]
>>> dsolve_kovacic(x*f.diff(x, 2) + (1 - x)*f.diff(x) - f, f)
[exp(x), exp(x)*Integral(exp(-x)/x, x)]
>>> dsolve_linear(x*f.diff(x, 2) - (x + 2)*f.diff(x) + 2*f, f)
[x**2 + 2*x + 2, exp(x)]
>>> special_solutions(f.diff(x, 2) + (-Rational(1, 4) + 1/x + (Rational(1, 4) - n**2)/x**2)*f, f)
[x**(n + 1/2)*exp(-x/2)*hyper((n - 1/2,), (2*n + 1,), x), x**(1/2 - n)*exp(-x/2)*hyper((-n - 1/2,), (1 - 2*n,), x)]

```

SymPy answers each of these with six terms of a power series, for
instance
`Eq(y(x), C2*(x**2/2 + x + 1) + C1*x**3*(x**2/20 + x/4 + 1) + O(x**6))`
for the third one, whose exact solutions are `x**2 + 2*x + 2` and
`exp(x)`. The four algorithms are Kovacic's (twice), the
Abramov–Bronstein–Petkovšek polynomial and hyperexponential solutions,
and the recognition of a Whittaker equation through its normal form
invariant.

### Nonlinear second order equations

```python
>>> bocharov = f.diff(x, 2) + 3*f*f.diff(x) + f**3
>>> found = dsolve_second_order(bocharov, f)
>>> found
[Eq(y(x), 2*(C2 + x)/(2*C1 + 2*C2*x + x**2))]
>>> checkodesol(bocharov, found[0])
(True, 0)
>>> is_linearizable(-3*y*p - y**3, x, y, p)
True

```

SymPy: `dsolve` raises `NotImplementedError: The given ODE ... cannot be
solved by the factorable group method`. This is Bocharov's example: it
is linearisable, but not by a fibre preserving transformation, so it is
solved by rectifying two commuting point symmetries, after which the
equation is `u'' = 1`. SymPy has no linearisation test.

### First order equations: Abel, Chini, Riccati

```python
>>> abel_by_invariants(f.diff(x) + f**3 + 2*(x - 1)*f**2, f) is not None
True
>>> dsolve_first_order(f.diff(x) - f**3 - x**Rational(-3, 2), f) is not None
True
>>> sorted(str(u) for u in dsolve_first_order(f.diff(x) - f**2 - x, f).rhs.atoms(besselj))[0]
besselj(-1/3, 2*x**(3/2)/3)

```

SymPy: `dsolve` gives no answer within 60 seconds on the first two, and
crashes with `TypeError: bad operand type for unary -: 'list'` on the
Riccati equation `y' = y**2 + x`, whose general solution is the one in
Bessel functions above (SymPy's Riccati hints need a rational particular
solution). The first equation is recognised by its invariants as
equivalent to the Airy class and the second is Chini's separable class;
both implicit solutions are long, and implicit differentiation along the
equation gives residuals below `10^-13` at sample points.

### Symmetry methods

```python
>>> [X.generator() for X in symmetries(f.diff(x, 2) + 2*f.diff(x)/x + f**5, f)]
['x*d/dx - y/2*d/dy']
>>> dsolve_lie(f.diff(x, 2) - f.diff(x)**2/f - f.diff(x)/x, f)
[Eq(y(x), exp(C1*x**2/2 + C2))]

```

SymPy: `classify_ode` offers only `('factorable',)` for the first
equation, and `dsolve` raises `NotImplementedError` on the second. Its
`lie_group` hint is for first order equations only.

### Systems and differential-algebraic equations

```python
>>> dsolve_dae(Matrix([[1, 0], [0, 0]]), Matrix([[0, -1], [1, 0]]), Matrix([0, sin(x)]), x).solution.T
Matrix([[sin(x), cos(x)]])
>>> [Y.T for Y in dsolve_linear_system(Matrix([[0, 1], [2/x**2, 0]]), x)]
[Matrix([[1/x, -1/x**2]]), Matrix([[x**2, 2*x]])]

```

SymPy: the first system (`y1' = y2`, `y1 = sin(x)`) makes `dsolve` raise
`KeyError: y2(x)`, the second (`y1' = y2`, `y2' = 2 y1/x**2`) raises
`ValueError: The function cannot be automatically detected for nan.`
The first is a differential-algebraic equation of index 2: its solution
is unique, with no free constant, and is found from the core-nilpotent
decomposition of the pencil.

### Partial differential equations

```python
>>> complete_integral(g.diff(x)*g.diff(y) - 1, g)
Eq(u(x, y), a*x + b + y/a)
>>> [X.generator() for X in pde_symmetries(g.diff(y) - g.diff(x, 2), g, degree=2)]
['u*d/du', 'd/dy', 'd/dx', 'x*d/dx + 2*y*d/dy', 'y*d/dx - u*x/2*d/du']
>>> pdsolve_lie(g.diff(y) - g.diff(x, 2), g)
[Eq(u(x, y), C1 + C2*x), Eq(u(x, y), C1), Eq(u(x, y), C1 + C2*erf(x/(2*sqrt(y)))), Eq(u(x, y), C1*exp(-x**2/(4*y))/sqrt(y))]

```

SymPy: `pdsolve` raises `NotImplementedError: psolve: Cannot solve ...`
for both equations; it handles first order linear equations and
separation of variables, and has no symmetry analysis. The complete
integral comes from Charpit's method, and every group invariant solution
of the heat equation is verified with `checkpdesol` before it is
returned.

## Sums, products and series

### Indefinite summation beyond Gosper

```python
>>> karr_sum(harmonic(k), (k, 1, n))
n*harmonic(n) - n + harmonic(n)
>>> karr_sum(harmonic(k)**2, (k, 1, n))
n*harmonic(n)**2 - 2*n*harmonic(n) + 2*n + harmonic(n)**2 - harmonic(n)
>>> karr_sum(harmonic(k)/(k + 1), (k, 1, n))
(n*harmonic(n)**2 - n*harmonic(n, 2) + harmonic(n)**2 + 2*harmonic(n) - harmonic(n, 2))/(2*(n + 1))

```

SymPy: `summation` returns each of these unevaluated. Karr's algorithm
decides, in the ΠΣ-field built from the summand, whether a closed form
exists there, and returns it or a proof that there is none.

### Definite summation by creative telescoping

```python
>>> zeilberger_sum((-1)**k*binomial(2*n, k)**3, (k, 0, 2*n))
(-1)**n*gamma(3*n + 1)/gamma(n + 1)**3
>>> wz_prove(binomial(n, k)**2, binomial(2*n, n), n, k)
True

```

SymPy: `summation((-1)**k*binomial(2*n, k)**3, (k, 0, 2*n))` →
`Piecewise((hyper((-2*n, -2*n, -2*n), (1, 1), 1), re(n) > -1/3), ...)`,
an unevaluated hypergeometric function rather than the closed form of
Dixon's identity. There is no Wilf–Zeilberger certificate machinery.

### q-analogues

```python
>>> factor(qgosper_sum(q**k, (k, 0, n), q))
(q**(n + 1) - 1)/(q - 1)
>>> qzeilberger(qbinomial(n, k, q)*q**(k*(k - 1)/2)*x**k, n, k, q).coefficients
[-q**n*x - 1, 1]

```

No counterpart in SymPy: the recurrence found for the q-binomial theorem
is `S(n + 1) = (1 + q**n x) S(n)`.

### Convergence with parameters

```python
>>> sum_convergence(x**n/n, n)
(x >= -1) & (x < 1)
>>> sum_convergence(1/n**p, n)
p > 1
>>> sum_convergence((-1)**n*n**p, n)
p < 0
>>> product_convergence(1 + x/n**2, n)
True

```

SymPy: each of `Sum(...).is_convergent()` and
`Product(...).is_convergent()` raises `NotImplementedError: convergence
checking for more than one symbol containing series is not handled`. The
conditions above come from the ratio test with Raabe's and Bertrand's
tests on its boundary, the root test, comparison with the p-series,
Leibniz's test and the integral test, with the boundary points of a
parametric ratio examined one by one — the first answer includes the
convergent boundary point `x = -1`.

### Euler, polygamma and Dirichlet series

```python
>>> polygamma_series(harmonic(n)/n**2, n)
2*zeta(3)
>>> polygamma_series(polygamma(1, n)/n**2, n)
7*pi**4/360
>>> dirichlet_series(totient(n)/n**s, n)
(zeta(s - 1)/zeta(s), re(s) > 2)
>>> dirichlet_series(mobius(n)/n**s, n)
(1/zeta(s), re(s) > 1)

```

SymPy: all four sums come back unevaluated from `.doit()`. The Dirichlet
series are returned with their half plane of convergence.

## Limits and series expansions

### Limits which depend on a parameter

```python
>>> limit(exp(a*x), x, oo)
Piecewise((oo, a > 0), (1, Eq(a, 0)), (0, a < 0))
>>> limit(x**a*log(x), x, 0, assumptions=a > 0)
0

```

SymPy: `limit(exp(a*x), x, oo)` raises `NotImplementedError: Result
depends on the sign of sign(a)` and `limit(x**a*log(x), x, 0)` raises
`NotImplementedError: Not sure of sign of a`. The cases are the
counterpart of Mathematica's `GenerateConditions`.

### Sequence limits

```python
>>> limit_seq(a**n, n, assumptions=a > 0)
Piecewise((oo, a > 1), (1, Eq(a, 1)), (0, a < 1))
>>> limit_seq(a**n, n, assumptions=(a > -1) & (a < 0))
0

```

SymPy: `limit_seq(a**n, n)` → `exp(oo*sign(log(a)))`, which is not an
answer. A negative base makes the sequence oscillate, and it tends to
zero exactly when its absolute value does.

### Series expansion under an assumption

```python
>>> series(sqrt(a**2 + x), x, 0, 2, assumptions=a < 0)
-x/(2*a) - a + O(x**2)

```

SymPy: `sqrt(a**2 + x).series(x, 0, 2)` →
`x*sqrt(a**2)/(2*a**2) + sqrt(a**2) + O(x**2)`, which cannot resolve
`sqrt(a**2)`.

## Polynomials and ideals

### Ideal operations

```python
>>> Ideal([x - t**2, y - t**3], t, x, y).eliminate([t])
Ideal([x**3 - y**2], x, y)
>>> Ideal([x*z - y**2, x**2 - y*z], x, y, z).saturate(Ideal([y], x, y, z))
Ideal([x**2 - y*z, x*y - z**2, -x*z + y**2], x, y, z)
>>> Ideal([x**2 - 2*x*y + y**2, x**3], x, y).radical()
Ideal([x, y], x, y)
>>> Ideal([x**2, x*y], x, y).hilbert_series()
(-t**2 + t + 1)/(1 - t)
>>> I = Ideal([x*z - y**2, x**2 - y*z], x, y, z)
>>> I.dimension(), I.degree()
(1, 4)

```

SymPy's `agca` ideals have no elimination, intersection, quotient,
saturation, radical, dimension, degree or Hilbert series: they support
membership and arithmetic only.

### The Gröbner walk

```python
>>> from sympy.polys import ring, QQ, grevlex, lex
>>> from sympy.polys.groebnertools import groebner as ring_groebner
>>> R, X, Y, Z = ring("x,y,z", QQ, grevlex)
>>> groebner_walk(ring_groebner([X*Z - Y**2, X**2 - Y*Z], R), R, lex)
[x**2 - y*z, x*y**2 - y*z**2, x*z - y**2, y**4 - y*z**3]

```

SymPy: `groebner([x*z - y**2, x**2 - y*z], x, y, z, order='grevlex').fglm('lex')`
raises `NotImplementedError: Cannot convert Groebner bases of ideals with
positive dimension.` The walk works in any dimension.

### Subresultant coefficients and linear elimination

```python
>>> from sympy.polys import ZZ
>>> R, X, A, B = ring("x,a,b", ZZ)
>>> psc(X**3 + A*X + B, 3*X**2 + A)
[4*a**3 + 27*b**2, 6*a, 3]
>>> eliminate_linear((x > y) & (x < 1), x)
y < 1

```

SymPy has `subresultants` (the polynomial remainder sequence) but not
its principal coefficients, which are what the CAD projection operators
need; the first one here is the discriminant of the cubic. There is no
virtual substitution either.

## Definite integration

### A singularity inside the range

```python
>>> definite_integral(1/x, (x, -1, 2))
Integral(1/x, (x, -1, 2))

```

SymPy: `integrate(1/x, (x, -1, 2))` -> `nan`. The integral diverges;
`definite_integral` cuts the range at the singularities it finds inside
it and claims nothing when a piece diverges, so the unevaluated integral
comes back instead of a number.

### Powers of trigonometric functions: Beta integrals

```python
>>> definite_integral(sqrt(sin(x)), (x, 0, pi/2))
2*sqrt(pi)*gamma(3/4)/gamma(1/4)

```

SymPy: `integrate(sqrt(sin(x)), (x, 0, pi/2))` -> `Integral(sqrt(sin(x)), (x, 0, pi/2))`.
The substitution `x = asin(sqrt(u))` turns powers of `sin` and `cos` over
a quarter period into a Beta integral, one kernel of the Mellin table.

### A periodic integrand with kinks

```python
>>> definite_integral(sqrt(1 - cos(x)), (x, 0, 2*pi))
4*sqrt(2)

```

SymPy: `integrate(sqrt(1 - cos(x)), (x, 0, 2*pi))` -> `Integral(sqrt(1 - cos(x)), (x, 0, 2*pi))`.
`1 - cos(x)` is `2*sin(x/2)**2` and the square root is
`sqrt(2)*Abs(sin(x/2))`, whose kinks are found before integrating.

### Logarithms and powers on the unit interval

```python
>>> definite_integral(x**Rational(1, 3)/sqrt(-log(x)), (x, 0, 1))
sqrt(3)*sqrt(pi)/2

```

SymPy: `integrate(x**Rational(1, 3)/sqrt(-log(x)), (x, 0, 1))` -> `Integral(x**(1/3)/sqrt(-log(x)), (x, 0, 1))`.
`(-log(x))**k` on `(0, 1)` has the Mellin transform `gamma(k + 1)/s**(k + 1)`.

### Integrands in exp(x) over the real line

```python
>>> definite_integral(x*exp(x)*exp(k*x)/(exp(x) + 3), (x, -oo, oo), (k > -1) & (k < 0))
3**k*pi*(polygamma(0, -k) - polygamma(0, k + 1) - log(3))/sin(pi*k)

```

SymPy: `integrate(x*exp(x)*exp(k*x)/(exp(x) + 3), (x, -oo, oo))` -> `Integral(x*exp(x)*exp(k*x)/(exp(x) + 3), (x, -oo, oo))`.
The substitution `u = exp(x)` gives `log(u)*u**k/(u + 3)` over `(0, oo)`:
a power of `u` times a kernel of the table, and the logarithm is a
derivative with respect to the exponent. The condition on `k` is the
strip of the Mellin transform of `1/(1 + u)`.

### A Laplace transform with a product of two kernels

```python
>>> definite_integral(exp(-s*x)*sin(a*x)/x, (x, 0, oo), (s > 0) & (a > 0))
atan(a/s)

```

SymPy: `integrate(exp(-s*x)*sin(a*x)/x, (x, 0, oo))` -> `Piecewise((a*atan(sqrt(a**2/s**2))/(s*sqrt(a**2/s**2)), (Eq(Abs(arg(a)), 0) & (Abs(arg(s)) < pi/2)) | ...), (Integral(...), True))`.
Parseval's formula for the Mellin transform turns the product into a
Meijer G-function, which Slater's theorem expands; the assumptions decide
the conditions, where SymPy leaves conditions on `arg(a)` and `arg(s)`
and the value written with `sqrt(a**2/s**2)`.

### The Mellin transform of atan

```python
>>> xmellin(atan(x), x, s)
MellinTransform(gamma(-s)*gamma(1/2 - s/2)*gamma(s/2 + 1/2)/(2*gamma(1 - s)), (-1, 0))

```

SymPy: `mellin_transform(atan(x), x, s)` -> `MellinTransform(atan(x), x, s)`.
The table of transforms comes with the strips of convergence, which are
the convergence conditions of the integrals; SymPy has no entry for
`atan`.

### An integral over a region described by inequalities

```python
>>> IntegralByRanges(x*y, (x > 0) & (y > 0) & (x + y < 1)).doit()
1/24

```

No counterpart in SymPy. The region is decomposed into stacks of
intervals by the cylindrical algebraic decomposition and the iterated
integrals are computed innermost first, the way Mathematica's
`Integrate[f, {x, y} ∈ region]` works.

### The area of a disc of parametric radius

```python
>>> IntegralByRanges(1, x**2 + y**2 < c**2, [x, y]).doit()
Piecewise((pi*c**2, (c > 0) | (c < 0)), (0, True))

```

No counterpart in SymPy. The parameter is the first variable of the
decomposition and gives the case distinction.

## The six wrong answers, for the record

These are worth reporting to SymPy (issue #25 of this repository collects
these and the other limitations found while building the package):

| Question | SymPy | correct |
| --- | --- | --- |
| `ask(Q.real(sqrt(a - 2)), Q.positive(a))` | `True` | undecidable: `I` at `a = 1` |
| `satisfiable((x**2 + y**2 < 1) & (x*y > Rational(1, 2)))` | a model | unsatisfiable over the reals |
| `solveset(Eq(log(x) + log(x - 1), log(2)), x, S.Reals)` | `{-1, 2}` | `{2}` |
| `solveset(2*x**2 + 3*sqrt(x + 6) - 1, x, S.Reals)` | four candidates | empty |
| `solveset(sin(x) > 0, x, S.Reals)` | `Interval.open(0, pi)` | one interval per period |
| `Sum(sin(n)/n, (n, 1, oo)).is_convergent()` | `False` | convergent (Dirichlet's test) |

The last two were found by `benchmarks/fuzz.py` and used to be passed on
by this package; both are fixed here:

```python
>>> from sympy import sin, pi, Interval, Union, S, Sum, oo
>>> from sympy.abc import n, x
>>> solve(sin(x) > 0, x, (x > 0) & (x < 10))
Union(Interval.open(0, pi), Interval.open(2*pi, 3*pi))
>>> solve(sin(x) > 0, x, domain=S.Reals)
ConditionSet(x, Contains(Mod(x, 2*pi), Interval.open(0, pi)), Reals)
>>> sum_convergence(sin(n)/n, n)
True
>>> Sum(sin(n)/n, (n, 1, oo)).is_convergent()
False

```

Two crashes on well posed input are worth reporting as well:
`dsolve(y' - y**2 - x, y)` raises `TypeError: bad operand type for unary
-: 'list'`, and `dsolve([Eq(y1', y2), Eq(y1, sin(x))], [y1, y2])` raises
`KeyError: y2(x)`.
