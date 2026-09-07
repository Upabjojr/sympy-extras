# Differential equations: Lie point symmetries, similarity reductions, and solving with assumptions

Modules: `sympy_extras.solvers` and `sympy_extras.assumptions.solve`.

## What SymPy has and what is added

SymPy's `pdsolve` handles first order linear partial differential equations
and separation of variables (`pde_separate`); it has no symmetry analysis.
SymPy's `dsolve` has a `lie_group` hint for **first order** ordinary
differential equations, with heuristics for the infinitesimals, and no
symmetry method for higher order.

`sympy_extras.solvers` adds:

- **Lie point symmetries of any system of differential equations**
  (`symmetries`, `check_symmetry`): the prolongation of a vector field to
  the jet space, the invariance condition with the leading derivatives
  eliminated, and the determining equations solved with a polynomial
  ansatz for the infinitesimals (times user-given functions). Every
  symmetry returned is verified by substitution; symmetries outside the
  ansatz are not found.
- **Similarity reductions of PDEs in two independent variables**
  (`pde_symmetries`, `similarity_reduction`, `pdsolve_lie`): the invariant
  `z(x, t)` of a symmetry, the form `u = phi(x, F(z))` of the invariant
  solutions, the reduced ordinary differential equation for `F`, and the
  solutions `dsolve` finds for it, verified with `checkpdesol`.
- **Solving ODEs through their symmetries** (`ode_symmetries`,
  `canonical_coordinates`, `reduce_order`, `dsolve_lie`, `solve_ode`): in
  canonical coordinates `(r, s)` the symmetry is `d/ds`, the equation loses
  `s`, an equation of order `n` becomes one of order `n - 1` for
  `v = ds/dr` (a quadrature for `n = 1`), solved by `dsolve`; the
  solutions are verified with `checkodesol`. `solve_ode` runs `dsolve`
  first and the symmetry method when it fails.

Every step handed to SymPy (integration, `dsolve`, `solve`, the checks)
runs under a time limit (`timeout` argument, 30 s by default) so that the
solvers always return.

```python
>>> from sympy import Function, symbols
>>> from sympy_extras.solvers import pde_symmetries, pdsolve_lie, dsolve_lie
>>> x, t = symbols('x t')
>>> u = Function('u')(x, t)
>>> heat = u.diff(t) - u.diff(x, 2)
>>> for X in pde_symmetries(heat, u, degree=3):
...     print(X.generator())
u*d/du
d/dt
d/dx
x*d/dx + 2*t*d/dt
t*d/dx - u*x/2*d/du
t*x*d/dx + t**2*d/dt + (-t*u/2 - u*x**2/4)*d/du
>>> for s in pdsolve_lie(heat, u):
...     print(s)
Eq(u(x, t), C1 + C2*x)
Eq(u(x, t), C1)
Eq(u(x, t), C1 + C2*erf(x/(2*sqrt(t))))
Eq(u(x, t), C1*exp(-x**2/(4*t))/sqrt(t))
>>> y = Function('y')(x)
>>> dsolve_lie(y.diff(x, 2) - y.diff(x)**2/y - y.diff(x)/x, y)
[Eq(y(x), exp(C1*x**2/2 + C2))]

```

## Solving with assumptions

`sympy_extras.assumptions.solve(equations, symbols, assumptions, domain)`
is `Solve` with `Assumptions` and a domain. One unknown with polynomial
equations and inequalities over the reals goes to the cylindrical
algebraic decomposition (exact intervals and points with algebraic
endpoints); otherwise `solveset` or `nonlinsolve` solve the equations,
with the parameters carrying the assumptions, and each solution is kept,
dropped or put in a `ConditionSet` according to whether the assumptions
hold at it, fail, or cannot be decided.

```python
>>> from sympy import S
>>> from sympy.abc import a, y
>>> from sympy_extras.assumptions import solve
>>> solve(x**2 - 2, x, x > 0)
{sqrt(2)}
>>> solve((x**2 - 2 > 0) & (x < 3), x, domain=S.Reals)
Union(Interval.open(-oo, -sqrt(2)), Interval.open(sqrt(2), 3))
>>> solve(x**2 - a, x, (x > 0) & (a > 0))
{sqrt(a)}
>>> solve([x**2 + y**2 - 1, x - y], [x, y], x > 0)
{(sqrt(2)/2, sqrt(2)/2)}

```

## Limitations

- The ansatz is polynomial (plus the functions given in `basis`). Linear
  equations whose symmetries involve their own solutions (Bessel, Airy,
  Mathieu, ... equations) get no useful symmetry; this is the nature of
  the method, not a bug: Lie symmetries help with nonlinear equations.
- Parameters in the equations are generic: the exceptional values with
  more symmetries (such as `m = -4/3` for `u_t = (u^m u_x)_x`) must be
  substituted explicitly.
- Similarity reductions need two independent variables and infinitesimals
  of the independent variables free of `u`; the reduced equation is only
  as solvable as `dsolve` makes it.
- Quadratures which SymPy cannot evaluate are returned as `Integral`s.

## Verification

- `benchmarks/pde_symmetries.py`: twelve classical equations (heat,
  Burgers, KdV, wave, nonlinear diffusion with generic and exceptional
  exponent, Fisher, Boussinesq, Liouville, sine-Gordon, potential Burgers,
  a Black-Scholes-type equation) against the dimensions of the symmetry
  algebras in Olver, Bluman-Kumei and Ibragimov's handbook.
- `benchmarks/kamke_odes.py`: the Kamke collection of about 700 first and
  second order equations, downloaded from Maxima's test suite, solved with
  `dsolve` and with `dsolve_lie` and verified with `checkodesol`.
- `benchmarks/solve_random.py`: random polynomial equations with sign
  assumptions against real root isolation and high-precision evaluation.
- The unit tests verify every symmetry by substitution and every solution
  with `checkodesol`/`checkpdesol`.
