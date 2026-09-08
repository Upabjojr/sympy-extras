# Differential equations: Lie point symmetries, linear equations, first order PDEs, and solving with assumptions

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

## Transcendental equations and integer systems

`solve` reduces equations and inequalities in which the unknown occurs
only inside exponentials, logarithms, trigonometric functions or roots
to polynomial problems in a kernel variable and inverts the kernels with
the assumptions (`sympy_extras.solvers.transcendental`); systems of
linear equations over the integers are solved through the Hermite normal
form and, when the unknowns are nonnegative and the solutions finitely
many, by the completion procedure of Contejean and Devie
(`sympy_extras.solvers.integers`). See [docs/reduce.md](reduce.md).

```python
>>> from sympy import exp, sin, cos, Eq, S
>>> solve(exp(2*x) - 3*exp(x) + 2, x, x > 0)
{log(2)}
>>> solve(Eq(sin(x) + cos(x), 1), x, (x > 0) & (x < 3))
{pi/2}
>>> solve(Eq(3*x + 5*y, 22), [x, y], (x >= 0) & (y >= 0), domain=S.Integers)
{(4, 2)}

```

## Linear ODEs with rational coefficients

`sympy_extras.solvers.linear_ode` and `sympy_extras.solvers.kovacic` add
the algorithms behind Mathematica's `DSolve` for linear equations that
SymPy lacks:

- **Kovacic's algorithm** (`liouvillian_solution`, `dsolve_kovacic`): a
  decision procedure for the Liouvillian solutions of second order
  equations `y'' + p y' + q y = 0` with rational `p, q`, in its three
  cases (rational, quadratic and finite-group `omega = z'/z`), with the
  second solution by reduction of order. SymPy's rational Riccati solver
  covers only the first case (and misses equations without finite
  poles).
- **Polynomial, rational and hyperexponential solutions** of equations
  of any order with polynomial coefficients (`polynomial_solutions`,
  `rational_solutions`, `hyperexponential_solutions`): degree bounds
  from the indicial polynomial at infinity, denominator bounds from the
  indicial polynomials at the singular points (Abramov, Bronstein,
  Petkovšek; Singer), and for Fuchsian equations the ansatz
  `prod (x - c)**e_c * P(x)` with the local exponents (Beke's first
  order factors).
- **Reduction of order** (`reduce_order_linear`) by a known solution,
  and `dsolve_linear` which chains all of the above with SymPy's
  `dsolve` for the reduced equations.
- **Exponential parts at infinity** from the Newton polygon of the
  operator (`hyperexponential_solutions` finds `exp(P(x))` times the
  Fuchsian ansatz, so `y'' - 2x y' + 4y = 0` gives `x**2 - 1/2` and
  `exp(x**2)` times a polynomial is found where it exists).
- **Special functions** (`sympy_extras.solvers.special`): equations
  equivalent to the Bessel, Whittaker (confluent hypergeometric) or
  Gauss hypergeometric equations are recognised through the invariant
  of the normal form `z'' = r z` under `t = a (x - c)**k` and Möbius
  transformations, and solved with `besselj`/`bessely`
  (`besseli`/`besselk`), Whittaker's `M` written with `hyper`, and
  `2F1`. Airy, parabolic cylinder, Hermite, Kummer, Legendre and
  Chebyshev equations with symbolic parameters are all covered. This is
  the "solutions through Mellin transforms" of Mathematica's notes.
- **Systems** `Y' = A(x) Y` (`sympy_extras.solvers.linear_systems`):
  a cyclic vector turns the system into a scalar equation solved by the
  same machinery; `rational_system_solutions` is the elimination method
  for rational solutions of systems (Abramov–Bronstein).

```python
>>> from sympy import Matrix, Function
>>> from sympy.abc import n
>>> from sympy_extras.solvers import special_solutions, dsolve_linear_system
>>> y = Function('y')(x)
>>> special_solutions(y.diff(x, 2) + x*y, y)
[sqrt(x)*besselj(1/3, 2*x**(3/2)/3), sqrt(x)*besselj(-1/3, 2*x**(3/2)/3)]
>>> special_solutions(x**2*y.diff(x, 2) + x*y.diff(x) + (x**2 - n**2)*y, y)
[besselj(n, x), bessely(n, x)]
>>> dsolve_linear_system(Matrix([[1/x, 1], [0, 1/x]]), x)
[Matrix([
[x],
[0]]), Matrix([
[x**2],
[   x]])]

```

## First order equations of Riccati, Abel, Chini and d'Alembert–Lagrange type

`sympy_extras.solvers.first_order` linearises Riccati equations
`y' = a y**2 + b y + c` (`y = -u'/(a u)`) and solves the linear equation
with Kovacic's algorithm or in special functions, so that Riccati
equations without rational particular solutions (which SymPy's hints
need) get Bessel or hypergeometric general solutions; it solves Chini's
equation `y' = f(x) y**n + g(x) y + h(x)` and Abel's equations of the
first and second kind in the constant-invariant cases (implicit
solutions), and d'Alembert–Lagrange equations `y = x F(y') + G(y')`
parametrically. SymPy has hints for Bernoulli, Riccati (rational
particular solutions) and Clairaut equations only.

```python
>>> from sympy_extras.solvers import lagrange_ode, abel_ode, riccati_ode
>>> riccati_ode(y.diff(x) + y**2 - 2/x**2, y)
Eq(y(x), (2*C1*x**3 - 1)/(C1*x**4 + x))
>>> lagrange_ode(Eq(y, 2*x*y.diff(x) + y.diff(x)**2), y)
[Eq(x, C1/p**2 - 2*p/3), Eq(y(x), (6*C1 - p**3)/(3*p))]
>>> abel_ode(y.diff(x) - y**3 - 3*y**2 - 3*y, y)     # doctest: +ELLIPSIS
Eq(..., C1 + x)

```

```python
>>> from sympy import Function, Rational
>>> from sympy.abc import n
>>> from sympy_extras.solvers import dsolve_linear, dsolve_kovacic, liouvillian_solution
>>> y = Function('y')(x)
>>> dsolve_linear(x*y.diff(x, 2) - (x + 2)*y.diff(x) + 2*y, y)
[x**2 + 2*x + 2, exp(x)]
>>> dsolve_kovacic(y.diff(x, 2) + y.diff(x)/x + (1 - 1/(4*x**2))*y, y)
[exp(I*x)/sqrt(x), exp(-I*x)/sqrt(x)]
>>> liouvillian_solution(1/x - Rational(3, 16)/x**2, x).case
2
>>> liouvillian_solution(x, x) is None
True

```

## First order nonlinear PDEs

`sympy_extras.solvers.charpit.complete_integral` finds complete
integrals (solutions with two arbitrary constants) of
`F(x, y, u, u_x, u_y) = 0` by Charpit's method: the standard first
integrals of the characteristic system (`F(p, q) = 0`, `F(u, p, q) = 0`,
separable and Clairaut equations, `p = a` or `q = a` when they are first
integrals) and the integration of `du = p dx + q dy`. SymPy's `pdsolve`
handles first order linear equations only.

```python
>>> from sympy import symbols
>>> from sympy_extras.solvers import complete_integral
>>> x, y = symbols('x y')
>>> u = Function('u')(x, y)
>>> complete_integral(u.diff(x)*u.diff(y) - 1, u)
Eq(u(x, y), a*x + b + y/a)
>>> complete_integral(Eq(u, u.diff(x)*x + u.diff(y)*y + u.diff(x)*u.diff(y)), u)
Eq(u(x, y), a*b + a*x + b*y)

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

## Differential-algebraic equations

`sympy_extras.solvers.dae` solves linear differential-algebraic systems
`A y' + B y = f(x)` with constant coefficients and a singular `A`
(SymPy's `dsolve` rejects them). The pencil must be regular; then
`(lambda A + B)^{-1} A` has a core-nilpotent decomposition whose
invertible part carries the free constants (an ordinary system solved
by the matrix exponential) and whose nilpotent part isolates the
algebraic constraints, solved by differentiating the right-hand side up
to the index of the equation.

```python
>>> from sympy import Function, Matrix, Eq, sin, symbols
>>> from sympy_extras.solvers import dae_matrices, dsolve_dae, dae_index
>>> x = symbols('x')
>>> y1, y2 = Function('y1')(x), Function('y2')(x)
>>> A, B, f, _ = dae_matrices([Eq(y1.diff(x), y2), Eq(y1, sin(x))], [y1, y2])
>>> dae_index(A, B)
2
>>> dsolve_dae(A, B, f, x).solution.T
Matrix([[sin(x), cos(x)]])

```
