# Assumptions as mathematical statements

Module: `sympy_extras.assumptions`

SymPy's assumptions system expresses facts with predicates: `Q.positive(x)`,
`Q.integer(n)`, combined with `&` and `|`. This module is an alternative
front end where assumptions are ordinary mathematical statements, in the
spirit of Mathematica's user interface but with Python names:

| Mathematica                        | `sympy_extras.assumptions`                    |
|------------------------------------|-----------------------------------------------|
| `x > 0`                            | `x > 0`                                       |
| `Element[n, Integers]`             | `element(n, S.Integers)` (a `Contains`)       |
| `Element[x, Reals]`                | `element(x, S.Reals)`                         |
| `0 <= x <= 1`                      | `element(x, Interval(0, 1))` or `(x >= 0) & (x <= 1)` |
| `a && b`, `a \|\| b`, `!a`          | `a & b`, `a \| b`, `~a` (or `And`, `Or`, `Not`) |
| `ForAll[x, expr]`, `Exists[x, expr]` | `ForAll(x, expr)`, `Exists(x, expr)`        |
| `Refine[expr, assum]`              | `refine(expr, assum)`                         |
| `Simplify[expr, assum]`            | `simplify(expr, assum)`                       |
| `Assuming[assum, expr]`            | `with assuming(assum): ...`                   |
| `$Assumptions`                     | `global_assumptions`                          |
| `Resolve[expr, Reals]`, `Reduce`   | `resolve(expr)`                               |
| `SatisfiableQ[expr]`               | `satisfiable(expr)`                           |
| `TautologyQ[expr]`                 | `tautology(expr)`                             |
| `FindInstance[expr, vars, dom]`    | `find_instance(expr, vars, dom)`              |
| (no direct counterpart)            | `ask(query, assum)`                           |

Predicates of `sympy.assumptions` such as `Q.prime(n)` are still accepted
wherever an assumption is expected, for the facts which have no
mathematical notation.

## Conventions

- **An inequality is a statement about real numbers.** `x > y` says that
  `x` and `y` are real and `x - y` is positive, as in Mathematica.
  Equations `Eq(x, y)` and `Ne(x, y)` do not imply anything about the
  domain.
- **Variables are complex unless something says otherwise.** `ask(x**2 >= 0)`
  is `None` because `x` may be imaginary; `ask(x**2 >= 0, domain=S.Reals)` is
  `True`. Every function takes a `domain` argument, a named SymPy set that
  all the variables belong to, like the domain argument of Mathematica's
  `Reduce` and `Resolve`.
- **Three truth values.** `ask`, `tautology` and `satisfiable` answer
  `True`, `False` or `None` when the answer cannot be established with the
  available methods. `refine` leaves alone what it cannot decide.
- **Symbols with old style assumptions** (`Symbol('x', positive=True)`) are
  taken into account through SymPy's own `ask`.

## Backends

Nothing in SymPy is reimplemented: the module translates the assumptions
and dispatches to two engines, and the CAD engine is the one of this
package.

1. The assumptions are translated to the equivalent predicates of
   `sympy.assumptions` (`x > 2` becomes `Q.positive(x - 2)`, and its
   consequences `Q.positive(x)` and `Q.real(x)` are added). SymPy's `ask`,
   `refine` and its SAT solver answer everything they can: signs of
   products and sums, integrality, powers, `re`, `im`, and so on.
2. What remains undecided and is a Boolean combination of polynomial
   relations with rational coefficients between real variables is decided
   exactly by the cylindrical algebraic decomposition of
   `sympy_extras.polys.cad`: `ask(x**2 - 2*x + 1 >= 0, x > 0)` or
   `ask(x > 1, x > 2)` are beyond SymPy's predicates but are decided by a
   decomposition. Quantified statements are handled by the quantifier
   elimination of the same package.

The decomposition is exact but its cost grows quickly with the number of
variables and the degrees: the module is meant for statements with a few
variables.

3. **Beyond polynomials** (`sympy_extras.assumptions.analysis`). A relation
   between expressions of *one* real variable which are not polynomials
   (`sin`, `exp`, `log`, `atan`, roots, ...) is decided by calculus on the
   set where the variable is assumed to lie: the expression must be
   continuous there, its zeros are found exactly with `solveset`, the sign
   is constant between consecutive zeros and is read at a rational sample
   point of each piece, and when the zeros cannot be found the sign of the
   derivative (found the same way) gives monotonicity and the sign follows
   from the limits at the endpoints. Constant expressions such as
   `sin(3) - 1/7` are signed by SymPy's own knowledge (exact, or its
   numerical evaluation of constants) or, when
   `sympy_extras.settings.settings.numerical_checks` is on (the default),
   by high precision numerical evaluation with a margin over the error
   bound. With the checks off, only what SymPy knows by itself is used,
   and `solve` no longer discards the candidates of `solveset` by their
   numerical residual. This is what
   makes `ask(sin(x) > 0, (x > 0) & (x < pi))` `True`,
   `refine(Abs(sin(x)), (x > 0) & (x < pi))` `sin(x)`, and
   `ask(x*exp(x) > 1, x > 1)` `True`.

```python
>>> from sympy import S, sin, exp, log, Abs, Max, pi
>>> from sympy.abc import x
>>> from sympy_extras.assumptions import ask, refine
>>> ask(sin(x) > 0, (x > 0) & (x < pi))
True
>>> refine(Abs(exp(x) - 2) + Abs(log(x)), x > 1)
exp(x) + log(x) - 2
>>> ask(exp(x) >= x + 1, domain=S.Reals)
True

```

4. **Several variables and slack** (`sympy_extras.assumptions.intervals`,
   `sympy_extras.assumptions.bounds`). For expressions of several real
   variables the box of their assumptions is analysed with rigorous
   interval arithmetic (mpmath's, which rounds outwards): branch and bound
   subdivides the box until every piece has a definite sign, monotonicity
   in every variable (each partial derivative of definite sign) puts the
   extreme values at two corners, and an expression depending on the
   variables through a single inner argument is treated as a univariate
   problem on the range of that argument. In one variable, the zeros are
   isolated by bisection with interval arithmetic when `solveset` cannot
   list them. Finally, elementary functions of polynomial arguments are
   replaced by variables constrained by classical polynomial bounds
   (`t >= 1 + u` for `t = exp(u)`, `t**2 = u, t >= 0` for `sqrt(u)`,
   Taylor bounds of `sin`, `cos`, `log`, `atan`, ...; the method of
   MetiTarski) and the cylindrical algebraic decomposition decides whether
   the relation holds for every allowed value of the new variables. All
   of this is sound; statements which are tight at a point (such as
   `exp(x) >= 1 + x` at `x = 0`) are only reached through the exact bounds.

```python
>>> from sympy.abc import y
>>> ask(exp(x) + y > 1, (x > 0) & (y > 0))
True
>>> refine(Abs(sin(x*y)), (x > 0) & (x < 1) & (y > 0) & (y < 3))
sin(x*y)
>>> ask(sin(x) < x, x > 0)
True
>>> refine(Max(exp(x), 1 + x), domain=S.Reals)
exp(x)

```

The global settings (`sympy_extras.settings`) switch the numerical checks
off for speed or reproducibility, set their precision, and set the default
time limit of the solvers:

```python
>>> from sympy_extras.settings import settings, configure
>>> with configure(numerical_checks=False, timeout=5.0):
...     settings
Settings(numerical_checks=False, precision=30, timeout=5.0)
>>> settings
Settings(numerical_checks=True, precision=30, timeout=30.0)

```

## Examples

### Asking

```python
>>> from sympy import S, Eq, Q, Abs, sqrt, sign, Max, floor, Piecewise
>>> from sympy.abc import a, b, c, x, y, n
>>> from sympy_extras.assumptions import ask, element
>>> ask(x**3 > 0, x > 0)
True
>>> ask(x > 1, x > 2)
True
>>> ask(x > 3, x > 2) is None
True
>>> ask(x*y > 0, (x > 0) & (y < 0))
False
>>> ask(element(n**2 + n, S.Integers), element(n, S.Integers))
True
>>> ask(x**2 + y**2 >= 2*x*y, domain=S.Reals)
True
>>> ask(Q.prime(n), Eq(n, 7)) is None
True

```

### Refining and simplifying

`refine` and `simplify` are the counterparts of `Refine` and `Simplify`
with assumptions. They work in two steps. First every symbol, and every
subexpression whose sign or nature matters (the base of a fractional
power, the argument of `Abs`, `log`, `floor`, `atan2`, ...), is decided
against the assumptions and temporarily replaced by a symbol carrying the
equivalent assumptions of SymPy's core (`positive`, `integer`, `real`,
...): SymPy's automatic evaluation, `sympy.refine` and `sympy.simplify`
then apply unchanged, which is how `log(exp(x))`, `sin(n*pi)`,
`gamma(x + 1)/gamma(x)` or `log(x) + log(y)` get simplified. Second, what
SymPy cannot decide is rewritten here with the CAD: relations, `Max` and
`Min`, `atan2`, the conditions of `Piecewise`, and powers and logarithms of
polynomials whose factors have a known sign (`sqrt(x**2 - 2*x + 1)` is
`x - 1` when `x > 1`). A result is kept only when it is not larger than the
input.

```python
>>> from sympy import log, exp, atan2, gamma
>>> from sympy_extras.assumptions import refine, simplify
>>> refine(Abs(x - 1) + sqrt(x**2), x > 2)
2*x - 1
>>> refine(sqrt(x**2 - 2*x + 1) + log(x**2), x > 1)
x + 2*log(x) - 1
>>> refine(atan2(y, x - 1), x > 1)
atan(y/(x - 1))
>>> simplify(log(x) + log(y) + log(exp(y)), (x > 0) & (y > 0))
y + log(x*y)
>>> simplify(gamma(x**2 + 1)/gamma(x**2), x > 0)
x**2
>>> refine(sign(x*y), (x > 0) & (y < 0))
-1
>>> refine(Max(x, x**2), (x > 0) & (x < 1))
x
>>> refine(floor(n + 1), element(n, S.Integers))
n + 1
>>> refine(Piecewise((Abs(x), x > 0), (2, x**2 + y**2 < 1), (3, True)), (x > 1) & (y > 0))
x
>>> refine(Piecewise((Abs(x), x > 0), (Abs(x) + 1, True)), domain=S.Reals)
Piecewise((x, x > 0), (1 - x, True))
>>> simplify((x**2 - 1)/(x - 1), x > 1)
x + 1

```

Boolean expressions are refined too: the decided parts disappear. When a
formula is polynomial in real variables, `simplify` minimises it by
quantifier elimination.

```python
>>> refine((x > 0) & (y > 0), x > 1)
y > 0
>>> refine(x*y > 0, (x > 0) & (y > 0))
True
>>> simplify(Eq(x**2, 1), x > 0)
Eq(x, 1)
>>> simplify((x > 1) | (x**2 > 1), domain=S.Reals)
(x > 1) | (x < -1)

```

### Assuming

```python
>>> from sympy_extras.assumptions import assuming, global_assumptions
>>> with assuming(x > 1):
...     refine(Abs(x - 1))
x - 1
>>> global_assumptions.add(n > 0, element(n, S.Integers))
>>> ask(element(n, S.Naturals))
True
>>> global_assumptions.clear()

```

Inside an `assuming` block SymPy's own `ask` and `refine` also see the
predicates implied by the assumptions.

### Quantifiers and quantifier elimination

```python
>>> from sympy_extras.assumptions import ForAll, Exists, resolve
>>> resolve(ForAll(x, x**2 + b*x + c > 0))
b**2 - 4*c < 0
>>> resolve(Exists(x, Eq(x**2 + a*x + b, 0)))
a**2 - 4*b >= 0
>>> resolve(Exists(y, Eq(x**2 + y**2, 1) & (y > x)))
(x >= -1) & (x < CRootOf(2*x**2 - 1, 1))
>>> resolve(ForAll(x, Exists(y, y > x)))
True
>>> resolve(~ForAll(x, x**2 > 0))
True
>>> ask(ForAll(x, x**2 + 2*x*y + y**2 >= 0), domain=S.Reals)
True

```

Quantifiers may appear anywhere in the formula: it is put in prenex form
(bound variables are renamed when they clash) before the elimination.
Variables occurring linearly are eliminated first by virtual substitution
(Loos–Weispfenning), without any decomposition. With `domain=S.Integers`
the formula must be one of Presburger arithmetic (linear relations and
divisibilities `Eq(Mod(e, k), 0)`) and Cooper's algorithm is used; with
`domain=S.Complexes` it is a combination of equations and inequations and
comprehensive Gröbner systems are used. See
[docs/reduce.md](reduce.md).

```python
>>> from sympy import Mod
>>> resolve(Exists(x, Eq(2*x, y)), domain=S.Integers)
Eq(Mod(y, 2), 0)
>>> resolve(Exists(x, Eq(a*x, 1)), domain=S.Complexes)
Ne(a, 0)

```

### Limits and series

`limit` and `series` pass the assumptions on the parameters to SymPy
and, when a limit still depends on the sign of some expression, compute
each case compatible with the assumptions (a `Piecewise`, like
Mathematica's `GenerateConditions`).

```python
>>> from sympy import exp, oo, sqrt
>>> from sympy_extras.assumptions import limit, series
>>> limit(exp(a*x), x, oo, assumptions=a < 0)
0
>>> limit(exp(a*x), x, oo)
Piecewise((oo, a > 0), (1, Eq(a, 0)), (0, a < 0))
>>> limit(a**x, x, oo, assumptions=(a > 0) & (a < 1))
0
>>> series(sqrt(a**2 + x), x, 0, 2, assumptions=a < 0)
-x/(2*a) - a + O(x**2)

```

### Satisfiability and instances

`satisfiable` combines SymPy's SAT solver, for the Boolean structure, with
the decomposition, for the polynomial relations, in the lazy way of SMT
solvers: each propositional model is checked against the theory, and a
witness is returned.

```python
>>> from sympy import symbols
>>> from sympy_extras.assumptions import satisfiable, tautology, find_instance
>>> p, q = symbols('p q')
>>> satisfiable(p & ~q)
{p: True, q: False}
>>> satisfiable((p | (x > 1)) & (~p | (x < 0)) & (x > 0))
{p: False, x: 2}
>>> satisfiable((x**2 + y**2 < 1) & (x + y > 2))
False
>>> satisfiable((x > 1) & (x < 3), domain=S.Integers)
{x: 2}
>>> tautology((x > 0) | (x <= 0))
True
>>> find_instance((x**2 + y**2 < 1) & (x > y), [x, y], count=3)
[{x: 0, y: -1/2}, {x: CRootOf(2*x**2 - 1, 1), y: 0}, {x: 3/4, y: 0}]
>>> find_instance((x**2 > 5) & (x < 0), [x], S.Integers)
[{x: -3}]

```

With a single integer variable the answer is exact (the solution set is
projected on that variable and searched for an integer); with several
integer variables only the neighbourhood of the real sample points is
searched and `None` is returned when nothing is found there.

## Reference

Everything is importable from `sympy_extras.assumptions`.

- `element(x, domain)`: membership in a SymPy set (`Contains`).
- `ForAll(variables, formula)`, `Exists(variables, formula)`: quantified
  formulas; `prenex(formula)` gives the quantifier prefix and the matrix.
  The constructors do not evaluate: `ForAll(x, True)` is a `ForAll` and
  `simplify()` reduces it to `True` (and drops the variables which do not
  occur in the formula), as do `prenex`, `resolve`, `ask` and the others.
- `ask(query, assumptions=None, domain=None)`: `True`, `False` or `None`.
- `refine(expr, assumptions=None, domain=None)` and
  `simplify(expr, assumptions=None, domain=None, **kwargs)`.
- `assuming(*assumptions)`: context manager; `global_assumptions`: the
  assumptions in force everywhere (`add`, `remove`, `clear`).
- `resolve(formula, domain=S.Reals, assumptions=None, method=None)`:
  quantifier elimination over the reals.
- `satisfiable(formula, assumptions=None, domain=None, all_models=False)`,
  `tautology(formula, assumptions=None, domain=None)`,
  `find_instance(formula, variables, domain=S.Reals, assumptions=None, count=1)`.
- `Facts(assumptions, domain, symbols)`: the translation of assumptions to
  predicates and polynomial relations, for those who want to build on it.
