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
and dispatches to two engines.

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

```python
>>> from sympy_extras.assumptions import refine, simplify
>>> refine(Abs(x - 1) + sqrt(x**2), x > 2)
2*x - 1
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

Boolean expressions are refined too: the decided parts disappear.

```python
>>> refine((x > 0) & (y > 0), x > 1)
y > 0
>>> refine(x*y > 0, (x > 0) & (y > 0))
True

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
