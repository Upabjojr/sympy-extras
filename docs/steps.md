# Step-by-step solutions

`sympy_extras.steps` produces derivations, not only results: the answer
to SymPy issue 11088. It is separate from the solvers of the package,
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

* integrals and equations on this engine (the rules of `manualintegrate`;
  equations with the bookkeeping of equivalent, implied and split
  cases); limits, series, inequalities;
* a renderer with sentences or LaTeX (the steps carry identifiers only);
* the step-by-step elimination of `sympy_extras.matrices.steps` predates
  this engine and is a list of row operations, not a rule tree.

## References

* SymPy issue 11088, *Step by Step solutions*.
* SymPy Development Team, `sympy/integrals/manualintegrate.py`, SymPy
  master, 2026.
