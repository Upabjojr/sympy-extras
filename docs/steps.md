# Step-by-step solutions

`sympy_extras.steps` produces derivations, not only results: the answer
to SymPy issue 11088, for equations in one unknown and for derivatives. It is separate from the solvers of the package,
whose methods (Gröbner bases, cylindrical decompositions, the Risch
algorithm) have intermediate states which explain nothing; the steps
come from small rules of their own, each one a rule of the textbooks.

## The design

The engine (`sympy_extras.steps.core`) follows
`sympy.integrals.manualintegrate` in SymPy's development branch:

* a **step** is a frozen dataclass. The class is the rule, the fields are
  what the rule needs and the steps of its subproblems: a derivation is a
  tree of dataclasses. A step holds no sentence, only its identifier (the
  name of the class) and its data; wording is the job of a renderer.
* a **rule** is a generator. It never calls the solver: it `yield`s each
  subproblem it needs and receives the solved step, as many times as it
  wants, and `return`s the step it builds (or `None` if it does not
  apply).
* the **solver** (`StepSolver`) drives the generators and owns the state
  of a run: the problems on the current path (a rule leading back to one
  of them is cut), the steps already found (a subproblem met twice is
  solved once) and the limit on the depth.
* **alternatives**: by default the first rule which solves a problem is
  kept; with `branch=True` every rule which works is kept, in an
  alternative step, at every level of the tree.
* **collapsible detail**: `format_steps(step, depth=...)` prints the tree
  down to a depth; steps marked minor (a rewriting) are hidden unless
  `minor=True`.

## Equations

`sympy_extras.steps.equations.solve_steps(lhs, rhs, x)` solves an
equation in one unknown over the real numbers the way it is done by
hand. A rule turns the equation into a simpler one, or into several, and
yields them as subproblems; `solutions()` assembles the solution set
from the substeps.

```python
>>> from sympy_extras.core import Symbol
>>> from sympy_extras.functions import sqrt
>>> from sympy_extras.steps.core import format_steps
>>> from sympy_extras.steps.equations import solve_steps, manual_solve
>>> x = Symbol('x')
>>> print(format_steps(solve_steps(2*x + 3, 7, x)))
AddToBothSidesRule: 2*x + 3 = 7 -> {2}
  DivideBothSidesRule: 2*x = 4 -> {2}
    SolvedRule: x = 2 -> {2}
>>> print(format_steps(solve_steps(x**2 - 5*x + 6, 0, x)))
ZeroProductRule: x**2 - 5*x + 6 = 0 -> {2, 3}
  AddToBothSidesRule: x - 3 = 0 -> {3}
    SolvedRule: x = 3 -> {3}
  AddToBothSidesRule: x - 2 = 0 -> {2}
    SolvedRule: x = 2 -> {2}

```

**What a step does to the solutions** is part of the data. Each rule has
a `kind`: an `'equivalence'` keeps the solutions, `'cases'` takes the
union of the substeps (a zero product, the two square roots), and an
`'implication'` (multiplying by a denominator, squaring) may add
solutions, so that the step checks those of the new equation in the
original one and records the ones it rejects:

```python
>>> radical = solve_steps(sqrt(x + 2), x, x)
>>> print(format_steps(radical, depth=1))
SquareBothSidesRule: sqrt(x + 2) = x -> {2}
  AddToBothSidesRule: x + 2 = x**2 -> {-1, 2}
>>> radical.kind, radical.rejected
('implication', (-1,))
>>> rational = solve_steps(1/(x - 1), 2/(x**2 - 1), x)
>>> rational.solutions(), rational.rejected
(EmptySet, (1,))

```

**Alternatives.** A quadratic equation is solved by factoring (when it
factors over the rationals), by the formula and by completing the
square; the first which applies by default, all of them with
`branch=True`:

```python
>>> print(format_steps(solve_steps(x**2 - 2*x - 1, 0, x, branch=True)))
AlternativeRule: x**2 - 2*x - 1 = 0 -> {1 - sqrt(2), 1 + sqrt(2)}
  [1] QuadraticFormulaRule: x**2 - 2*x - 1 = 0 -> {1 - sqrt(2), 1 + sqrt(2)}
  [2] CompleteSquareRule: x**2 - 2*x - 1 = 0 -> {1 - sqrt(2), 1 + sqrt(2)}
    RootRule: (x - 1)**2 = 2 -> {1 - sqrt(2), 1 + sqrt(2)}
      AddToBothSidesRule: x - 1 = sqrt(2) -> {1 + sqrt(2)}
        SolvedRule: x = 1 + sqrt(2) -> {1 + sqrt(2)}
      AddToBothSidesRule: x - 1 = -sqrt(2) -> {1 - sqrt(2)}
        SolvedRule: x = 1 - sqrt(2) -> {1 - sqrt(2)}
>>> [way.rule for way in solve_steps(x**2 - 5*x + 6, 0, x, branch=True).substeps()]
['ZeroProductRule', 'QuadraticFormulaRule', 'CompleteSquareRule']

```

**The domain.** `solve_steps(lhs, rhs, x, domain)` and `manual_solve`
solve over `Complexes` (the default) or `Reals`:

```python
>>> from sympy_extras.sets import Reals
>>> from sympy_extras.functions import exp, log, sin, abs_
>>> print(format_steps(solve_steps(x**4 + 2*x**2 + 1, 0, x), depth=1))
ZeroProductRule: x**4 + 2*x**2 + 1 = 0 -> {-I, I}
  QuadraticFormulaRule: x**2 + 1 = 0 -> {-I, I}
>>> manual_solve(x**4 + 2*x**2 + 1, 0, x, Reals), manual_solve(x**3, 8, x), manual_solve(x**3, 8, x, Reals)
(EmptySet, {2, -1 - sqrt(3)*I, -1 + sqrt(3)*I}, {2})
>>> manual_solve(exp(x), 2, x), manual_solve(exp(x), 2, x, Reals)
(ImageSet(Lambda(n, 2*I*pi*n + log(2)), Integers), {log(2)})
>>> manual_solve(x + 1, x + 1, x)
Complexes

```

Over the complex numbers a quadratic has its two roots whatever the
discriminant, `u**n = q` its `n` roots, a cubic its three roots (Cardano),
and `exp(f) = c` a solution for every integer `n`.

A real solution is a solution over the complex numbers too, so the
rules of the reals apply there as well. Where such a rule does not give
every solution, the step (and every step built on it) answers `False`
to `complete()` and is printed with `(not all)`: the two cases of
`|A| = B` miss the other points of a circle, and the isolated real roots
of a polynomial which does not factor miss the roots which are not real
(`missing` counts them).

```python
>>> quartic = solve_steps(x**4 - x - 1, 0, x)
>>> print(format_steps(quartic))
RealRootsRule: x**4 - x - 1 = 0 -> {IndexedRoot(x**4 - x - 1, x, 0), IndexedRoot(x**4 - x - 1, x, 1)} (not all)
>>> quartic.complete(), quartic.missing, solve_steps(x**4 - x - 1, 0, x, Reals).complete()
(False, 2, True)
>>> absolute = solve_steps(abs_(x - 3), 5, x)
>>> absolute.solutions(), absolute.complete(), solve_steps(abs_(x - 3), 5, x, Reals).complete()
({-2, 8}, False, True)

```

Radicals, absolute values, exponentials, logarithms and trigonometric
functions have their rules; an equation which is a polynomial in one of
them (`exp(x)`, `sin(x)`, `log(x)`) is solved by substitution:

```python
>>> print(format_steps(solve_steps(exp(2*x) - 3*exp(x) + 2, 0, x, Reals), depth=1))
SubstitutionRule: exp(2*x) - 3*exp(x) + 2 = 0 -> {0, log(2)}
  ZeroProductRule: u**2 - 3*u + 2 = 0 -> {1, 2}
  TakeLogarithmRule: exp(x) = 1 -> {0}
  TakeLogarithmRule: exp(x) = 2 -> {log(2)}
>>> manual_solve(sqrt(x + 5) + sqrt(x), 5, x), manual_solve(abs_(2*x + 1), x + 4, x, Reals), manual_solve(2**x, 8, x, Reals)
({4}, {-5/3, 3}, {3})
>>> manual_solve(log(x) + log(x - 3), log(10), x), solve_steps(abs_(x - 2), 2*x + 5, x, Reals).rejected
({5}, (-7,))
>>> manual_solve(2*sin(x), 1, x)
Union(ImageSet(Lambda(n, 2*pi*n + pi/6), Integers), ImageSet(Lambda(n, 2*pi*n + 5*pi/6), Integers))
>>> manual_solve(x**3 + x + 1, 0, x, Reals), manual_solve(x**3 - 3*x + 1, 0, x)
({-(1/2 + sqrt(93)/18)**(1/3) + (-1/2 + sqrt(93)/18)**(1/3)}, {IndexedRoot(x**3 - 3*x + 1, x, 0), IndexedRoot(x**3 - 3*x + 1, x, 1), IndexedRoot(x**3 - 3*x + 1, x, 2)})

```

Over the reals a solution must make both sides real: `log(x - 5) =
log(1 - x)` has the candidate 3, where the two logarithms are equal
complex numbers; it is the solution over the complex numbers and there
is none over the reals. An equation the rules do not solve gives a tree which
`contains_dont_know()`, and `manual_solve` returns `None`. Two rules
mark the unsolved places: `DontKnowRule` where no rule applies, and
`InProgressRule` where a derivation came back to a problem it was still
working on (a loop). The tests
compare the solution set of every node of every tree, the alternatives
included, with the solution set SymPy finds for that node's own
equation over the same domain, and check the trigonometric families by substitution and by
counting the roots on an interval.

## Inequalities

`sympy_extras.steps.inequalities.inequality_steps` solves an inequality
in one unknown. Linear inequalities are solved by the operations on both
sides (a division by a negative number reverses the inequality: the
step is `flipped`); polynomial and rational ones by a **sign chart**:
the function is factored, its zeros and poles are ordered exactly, and
the sign of each interval follows from the parity of the exponents,
from the right, without evaluating a test point. The chart is data
(`points`, `poles`, `signs`) and the derivations of the roots are its
substeps. An absolute value is split into the cases of its sign.

```python
>>> from sympy_extras.steps.inequalities import inequality_steps, manual_inequality
>>> print(format_steps(inequality_steps(3 - 2*x < 7, x)))
AddToBothSidesRule: 3 - 2*x < 7 -> Interval.open(-2, oo)
  DivideBothSidesRule: -2*x < 4 -> Interval.open(-2, oo)
    SolvedRule: x > -2 -> Interval.open(-2, oo)
>>> chart = inequality_steps((x - 1)/(x + 2) <= 0, x)
>>> chart.rule, chart.points, chart.poles, chart.signs, chart.solutions()
('SignChartRule', (-2, 1), (-2,), (1, -1, 1), Interval.Lopen(-2, 1))
>>> manual_inequality(x**3 - x >= 0, x), manual_inequality(abs_(x - 1) < 3, x)
(Union(Interval(-1, 0), Interval(1, oo)), Interval.open(-2, 4))
>>> manual_inequality((x**2 - 1)/(x - 1) > 0, x)              # the common factor is not cancelled: 1 is excluded
Union(Interval.open(-1, 1), Interval.open(1, oo))

```

The tests compare the solution set of every node with the inequality
evaluated directly at several hundred rational points.

## Systems

`sympy_extras.steps.systems.system_steps` solves a system in several
unknowns by **substitution**: an equation is solved for an unknown (that
derivation is the first substep), the result is substituted in the
others, and the smaller system is solved; an equation with two values
for the unknown gives two cases. Every solution is checked in the whole
system (`rejected`). With `branch=True` every choice of equation and
unknown is kept, and for linear unknowns the **elimination** by a
combination of two equations. The `domain` argument is `Complexes` (the
default) or `Reals`, as for one equation.

```python
>>> from sympy_extras.steps.systems import system_steps, manual_system
>>> y = Symbol('y')
>>> print(format_steps(system_steps([(x + y, 3), (x - y, 1)], [x, y])))
SubstitutionRule: x + y = 3, x - y = 1 -> {(2, 1)}
  AddToBothSidesRule: x + y = 3 -> {3 - y}
    SolvedRule: x = 3 - y -> {3 - y}
  SingleEquationRule: 3 - 2*y = 1 -> {(1,)}
    AddToBothSidesRule: 3 - 2*y = 1 -> {1}
      DivideBothSidesRule: -2*y = -2 -> {1}
        SolvedRule: y = 1 -> {1}
>>> manual_system([(x**2 + y**2, 5), (x + y, 1)], [x, y]), manual_system([(x + y, 1), (2*x + 2*y, 3)], [x, y])
(((2, -1), (-1, 2)), ())
>>> ways = system_steps([(2*x + 3*y, 7), (4*x - y, 7)], [x, y], branch=True)
>>> sorted({way.rule for way in ways.alternatives}), ways.solutions()
(['EliminationRule', 'SubstitutionRule'], ((2, 1),))

```

Systems with infinitely many solutions are not solved (`manual_system`
returns `None`). The tests substitute every solution of every node in
that node's system and check that none of the real solutions SymPy
finds is missing.

## Derivatives

```python
>>> from sympy_extras.core import Symbol
>>> from sympy_extras.functions import sin, exp
>>> from sympy_extras.steps.core import format_steps
>>> from sympy_extras.steps.derivatives import derivative_steps, manual_derivative
>>> x = Symbol('x')
>>> steps = derivative_steps(x**2*sin(3*x), x)
>>> print(format_steps(steps))
ProductRule: d/dx(x**2*sin(3*x)) = 3*x**2*cos(3*x) + 2*x*sin(3*x)
  PowerRule: d/dx(x**2) = 2*x
  ChainRule: d/dx(sin(3*x)) = 3*cos(3*x)
    FunctionRule: d/du(sin(u)) = cos(u)
    ConstantMultipleRule: d/dx(3*x) = 3
      VariableRule: d/dx(x) = 1
>>> print(format_steps(steps, depth=1))
ProductRule: d/dx(x**2*sin(3*x)) = 3*x**2*cos(3*x) + 2*x*sin(3*x)
  PowerRule: d/dx(x**2) = 2*x
  ChainRule: d/dx(sin(3*x)) = 3*cos(3*x)

```

The tree itself is data:

```python
>>> derivative_steps(5*x**3, x)
ConstantMultipleRule(expression=5*x**3, variable=x, constant=5, rest=x**3, substep=PowerRule(expression=x**3, variable=x, exponent=3))
>>> manual_derivative(exp(x**2), x)
2*x*exp(x**2)

```

With `branch=True` a quotient is differentiated by the quotient rule and
as a product, and both derivations are kept:

```python
>>> print(format_steps(derivative_steps(sin(x)/x, x, branch=True)))
AlternativeRule: d/dx(sin(x)/x) = (x*cos(x) - sin(x))/x**2
  [1] QuotientRule: d/dx(sin(x)/x) = (x*cos(x) - sin(x))/x**2
    FunctionRule: d/dx(sin(x)) = cos(x)
    VariableRule: d/dx(x) = 1
  [2] ProductRule: d/dx(sin(x)/x) = cos(x)/x - sin(x)/x**2
    FunctionRule: d/dx(sin(x)) = cos(x)
    PowerRule: d/dx(1/x) = -1/x**2

```

A power with a variable exponent is rewritten first; the rewriting is a
minor step, shown on request:

```python
>>> print(format_steps(derivative_steps(x**x, x), depth=0, minor=True))
RewriteRule: d/dx(x**x) = (log(x) + 1)*exp(x*log(x))

```

## Writing a rule

A rule is a generator function of the problem and of the context of the
run. This is the product rule:

```python
def _product(problem, context):
    first, second = _split_product(problem.expression)
    first_step = yield DerivativeProblem(first, problem.variable)
    second_step = yield DerivativeProblem(second, problem.variable)
    return ProductRule(problem.expression, problem.variable, first, second, first_step, second_step)
```

A domain gives the solver three things: the candidate rules of a problem
in order of preference, the step which leaves a problem undone, and the
step which holds alternatives.

## Not done

* equations over the complex numbers: the solutions of `|A| = B` which
  are not real, the roots which are not real of a polynomial of degree
  four or more which does not factor (no quartic formula, no complex
  root object; both are reported by `complete()`), `log(f) = c` with `c`
  not real;
* equations: exponentials of unrelated bases (`4**x - 3**x = 1`), sums
  of trigonometric functions of different arguments; systems with
  infinitely many solutions (parametrised solution sets);
  inequalities with radicals, exponentials or parameters; case
  distinctions on parameters (a coefficient by which
  a side is divided is taken to be nonzero);
* integrals are deliberately left out (maintainer's decision of
  2026-11-11): SymPy's `manualintegrate` is the model of the engine, not
  a domain to redo here;
* a renderer with sentences or LaTeX (the steps carry identifiers only);
* the step-by-step elimination of `sympy_extras.matrices.steps` predates
  this engine and is a list of row operations, not a rule tree.

## References

* SymPy issue 11088, *Step by Step solutions*.
* SymPy Development Team, `sympy/integrals/manualintegrate.py`, SymPy
  master, 2026.
