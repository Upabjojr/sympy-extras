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

Over the reals, a system of polynomial inequalities in several unknowns
(with or without equations), and a system of equations with infinitely
many solutions, are described cylindrically, as Mathematica's `Reduce`
does: the points with algebraic coordinates as a finite set, and the rest
as a `ConditionSet` whose condition bounds the first unknown by numbers,
the second by functions of the first, and so on
(`sympy_extras.polys.cad.cylindrical_set`, see [docs/cad.md](cad.md)). The
parameters are bounded before the unknowns, within the assumptions.

```python
>>> solve([x**2 + y**2 <= 1, x + y >= 1], [x, y], domain=S.Reals)
ConditionSet((x, y), (x >= 0) & (x <= 1) & (y >= 1 - x) & (y <= sqrt(1 - x**2)), ProductSet(Reals, Reals))
>>> solve([x**2 + y**2 < a], [x, y], domain=S.Reals)
ConditionSet((x, y), (a > 0) & (x < sqrt(a)) & (x > -sqrt(a)) & (y < sqrt(a - x**2)) & (y > -sqrt(a - x**2)), ProductSet(Reals, Reals))
>>> solve([x**2 + y**2 <= 0], [x, y], domain=S.Reals)
{(0, 0)}

```

### The cases of the parameters

`solve` gives the generic solutions, like `Solve`: `{b/a}` for `a*x = b`.
With `cases=True` the values of the parameters are discussed, like
`Reduce`, for polynomial relations with rational coefficients, and the
result is a union of sets `ConditionSet(unknowns, condition on the
parameters, solutions)`; the assumptions choose among the cases.

```python
>>> from sympy.abc import b
>>> solve(a*x - b, x, cases=True)
Union(ConditionSet(x, Eq(a, 0) & Eq(b, 0), Complexes), ConditionSet(x, Ne(a, 0), {b/a}))
>>> solve(a*x - b, x, a > 0, cases=True)
{b/a}
>>> solve([a*x + y - 1, x + a*y - 1], [x, y], cases=True)
Union(ConditionSet((x, y), Eq(a, 1), {(1 - y, y)}), ConditionSet((x, y), Ne(a + 1, 0), {(1/(a + 1), 1/(a + 1))}))
>>> solve(x**2 <= a, x, domain=S.Reals, cases=True)
ConditionSet(x, a >= 0, Interval(-sqrt(a), sqrt(a)))

```

Over the complex numbers (equations and inequations) the cases are read
from a triangular decomposition in the sense of Lazard with the parameters
as the smallest variables (`sympy_extras.solvers.parametric.parametric_cases`,
on `sympy_extras.polys.regularchains`): the zeros of the system are the
union of the quasi-components of the regular chains, and in a chain the
polynomials in the parameters alone are the equations of the case, the
initials its inequations, and the polynomials in the unknowns give them
one after the other (a division for degree one, the quadratic formula for
degree two; a chain with a polynomial of higher degree in an unknown is
left as its equations, and an unknown which the chain does not constrain
stands for itself, as in `nonlinsolve`). An inequation in the unknowns
which the rest of its case implies is not written; one in the parameters
always is, since the written solution `1/(a + 1)` needs it even when the
equations imply it. The cases may overlap, and their number depends on the
order of the variables.

Over the reals (`domain=S.Reals`; the parameters are real too), with
inequalities as well, they are the cells of a cylindrical decomposition
with the parameters first (`sympy_extras.polys.cad.cylindrical_cases`):
the conditions bound the first parameter by numbers and the next ones by
functions of those before, the solutions of one unknown are intervals and
points with endpoints in the parameters, and the cases with the same
solutions are joined.

Verified by substitution (at rational values of the parameters the points
of the cases which hold are the solutions of the system with the values
put in, for random systems of one and two equations with one and two
parameters) and by the quantifier elimination over the complex numbers of
`resolve`, which works with comprehensive Gröbner systems and proves the
cases equivalent to the system.

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
- **Factorisation of operators** (`sympy_extras.solvers.factorization`):
  the arithmetic of the Ore ring `Q(x)[D]` on `LinearOperator`
  (composition, right division, `gcrd`, `lclm`, adjoint, exterior and
  symmetric powers), right factors of any order by Beke's algorithm,
  left factors through the adjoint, `factor_operator` for a complete
  factorisation with proofs of irreducibility, and `dsolve_linear`
  solving equations of order three and more through the factors (see
  below).
- **Special functions** (`sympy_extras.solvers.special`): equations
  equivalent to the Bessel, Whittaker (confluent hypergeometric) or
  Gauss hypergeometric equations are recognised through the invariant
  of the normal form `z'' = r z` under `t = a (x - c)**k` and Möbius
  transformations, and solved with `besselj`/`bessely`
  (`besseli`/`besselk`), Whittaker's `M` written with `hyper`, and
  `2F1`. Airy, parabolic cylinder, Hermite, Kummer, Legendre and
  Chebyshev equations with symbolic parameters are all covered. This is
  the "solutions through Mellin transforms" of Mathematica's notes.
- **Systems** `Y' = A(x) Y + b(x)` (`sympy_extras.solvers.linear_systems`):
  a cyclic vector turns the system into a scalar equation solved by the
  same machinery, rational solutions by Barkatou's method, variation of
  constants; see [Linear systems](#linear-systems-with-rational-coefficients).

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

### Factorisation of linear differential operators

`LinearOperator` is an element of the Ore ring `K[D]`, `K = Q(x)` (extended
by the constants of the coefficients), where `D a = a D + a'`: `*` is the
composition, `right_divmod` the Euclidean division on the right, and
`gcrd`, `lclm` the greatest common right divisor (the common solutions)
and the least common left multiple (the sums of solutions).

A right factor `R` of order `k` of `L` is determined by the Wronskian
matrix of `k` solutions: `R(y) = Wr(y_1, ..., y_k, y)/Wr(y_1, ..., y_k)`,
whose coefficients are ratios of `k x k` minors of the Wronskian matrix
of `L`. The minors satisfy a linear system (the `k`-th exterior power,
`L.exterior_power(k)` in scalar form), and `R` has rational coefficients
exactly when the minors are hyperexponential; the vector must also be
decomposable (the Grassmann–Plücker relations), which is a system of
quadratic equations when several hyperexponential solutions differ by
rational factors. This is **Beke's algorithm** (1894), in the form of
Schwarz (1989) and Bronstein (1994). `right_factor(L, k)` finds a right
factor, `left_factor(L, k)` a left one (the adjoint of a right factor of
the adjoint), and `factor_operator(L)` a factorisation into factors which
are proven irreducible when every search was exhaustive
(`hyperexponential_search` says when it is not: irregular finite
singular points, irrational exponents); for second order factors
Kovacic's algorithm decides it.

`dsolve_linear` uses the factorisation for equations of order three and
more: the solutions of the right factor, then for each solution `z` of
the left factor a solution of `R(y) = z` by variation of parameters,
with the integrals left unevaluated when SymPy cannot evaluate them.

```python
>>> from sympy import Function
>>> from sympy_extras.solvers import (LinearOperator, factor_operator, right_factor,
...     left_factor, gcrd, lclm, dsolve_linear)
>>> D = LinearOperator([0, 1], x)
>>> A = LinearOperator([-x, 0, 1], x)          # Airy: y'' - x y
>>> B = LinearOperator([1, 1/x, 1], x)         # Bessel of order 0
>>> F = factor_operator(B*A)
>>> F.factors, F.irreducible
([LinearOperator([1, 1/x, 1], x), LinearOperator([-x, 0, 1], x)], [True, True])
>>> F.expand() == B*A
True
>>> right_factor((D - LinearOperator([1/x], x))*A, 2) == A
True
>>> left_factor(A*(D - LinearOperator([1/x], x)), 2) == A
True
>>> gcrd(A*D, B*D)
LinearOperator([0, 1], x)
>>> lclm(D - LinearOperator([1], x), D).coefficients
[0, -1, 1]
>>> y = Function('y')(x)
>>> equation = ((D - LinearOperator([1], x))*A)(y)
>>> equation
x*y(x) - x*Derivative(y(x), x) - y(x) - Derivative(y(x), (x, 2)) + Derivative(y(x), (x, 3))
>>> len(dsolve_linear(equation, y))
3

```

The third solution of the last equation solves `y'' - x y = exp(x)`, written
with integrals of Airy functions (as modified Bessel functions). Before the
factorisation, `dsolve_linear` found no solution of this equation: it has
no hyperexponential solution to reduce the order with.

Not implemented: van Hoeij's local method (factors from the generalised
exponents at one singular point, much faster than the associated
equations for high orders), the eigenring, and hyperexponential solutions
at irregular finite singular points (so right factors whose solutions
have essential singularities at finite points are only found for second
order operators, through Kovacic's algorithm).

## Linear systems with rational coefficients

`sympy_extras.solvers.linear_systems` solves `Y' = A(x) Y + b(x)` with `A`
an `n x n` matrix of rational functions. SymPy's `dsolve` solves systems
with constant coefficients, systems whose matrix commutes with its
antiderivative, and a few special types of two and three equations.

- **Cyclic vectors** (`cyclic_reduction`, `cyclic_vector`,
  `system_to_scalar`): for a row vector `c`, `u = c Y` has the derivatives
  `u^(k) = c_k Y`, `c_{k+1} = c_k A + c_k'`; when `c_0, ..., c_{n-1}` are
  independent (`c` is *cyclic*) `u` satisfies a scalar equation of order
  `n` and `Y = M^-1 (u, u', ..., u^(n-1))` maps its solutions to those of
  the system. The unit vectors are tried first, then seeded random
  vectors with constant and then polynomial entries (a generic vector is
  cyclic: Katz, Churchill–Kovacic); every choice is checked.
- **`dsolve_system(A, x, b=None)`** returns a `LinearSystemSolution`: the
  `fundamental` matrix (its columns are independent solutions), a
  `particular` solution, and `complete`, `True` when every solution is
  `fundamental * C + particular` and `None` when the solvers found fewer
  than `n` columns. The scalar equation goes through `dsolve_linear`'s
  solvers, so equations of order three and more are factored (Beke), and
  SymPy's `linodesolve` is tried when they fall short.
- **Rational solutions** (`rational_system_solutions`, Barkatou 1999): the
  pole order of a rational solution at a simple pole of `A` is bounded by
  the negative integer eigenvalues of the residue (found over Q through a
  resultant for irrational poles), and its degree by the integer
  eigenvalues of `lim x A` at infinity; at a higher order pole with an
  invertible leading matrix there is no rational solution at all.
  Otherwise the bound comes from the scalar equation of a cyclic vector
  (Moser's reduction is not implemented). The numerators then solve a
  linear system. The inhomogeneous system has a rational particular
  solution when the augmented system `(Y, 1)` has one.
- **Hyperexponential solutions** (`hyperexponential_system_solutions`):
  those of the scalar equation, mapped back.
- **Inhomogeneous systems**: a rational particular solution first, then
  variation of constants `Y = Phi Integral(Phi^-1 b)` by Cramer's rule, with
  `det Phi` from Liouville's formula.

```python
>>> from sympy import Matrix, symbols, simplify, exp
>>> from sympy_extras.solvers import dsolve_system, rational_system_solutions
>>> A = Matrix([[0, 1], [2/x**2, 0]])
>>> S = dsolve_system(A, x)
>>> S.fundamental
Matrix([[1/x, x**2], [-1/x**2, 2*x]])
>>> S.complete
True
>>> rational_system_solutions(Matrix([[-1/x, 0], [1, 0]]), x)
[Matrix([
[0],
[1]])]
>>> b = Matrix([2*exp(-x), 3*x])
>>> S = dsolve_system(Matrix([[-2, 1], [1, -2]]), x, b)
>>> residual = S.particular.diff(x) - Matrix([[-2, 1], [1, -2]])*S.particular - b
>>> residual.applyfunc(simplify).T
Matrix([[0, 0]])

```

Not implemented: Moser's and Barkatou's reductions (super-irreducible and
simple forms), so the local bounds at irregular singular points come from
the scalar equation, whose cyclic vector may add apparent singularities
and large coefficients; Barkatou's companion block diagonal form for
systems that decompose; exponential solutions computed on the system
itself (the generalised exponents of a system); and the local data at
irregular points (formal solutions, Stokes phenomena).

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

## Second order equations: integrating factors and linearisation

`sympy_extras.solvers.second_order` finds integrating factors `mu(x, y)`
and `mu(y')` of `y'' = Phi(x, y, y')` (the exactness conditions of
Cheb-Terrab and Roche, with a linear equation for the `x`-dependence),
tests Lie's linearisation conditions, constructs the fibre preserving
transformation to `u'' = 0` when it exists, and otherwise rectifies two
commuting point symmetries into translations, after which the equation
is `u'' = F(u')`, solvable by quadratures.

```python
>>> from sympy_extras.solvers import dsolve_second_order
>>> f = Function('y')(x)
>>> dsolve_second_order(f.diff(x, 2) + 3*f*f.diff(x) + f**3, f)
[Eq(y(x), 2*(C2 + x)/(2*C1 + 2*C2*x + x**2))]

```

## Abel equations: invariants and equivalence classes

`sympy_extras.solvers.abel` computes the relative invariants `s3`, `s5`
and the absolute invariants `I1`, `I2` of an Abel equation of the first
kind, decides the equivalence of two equations under `y = P(x) u + Q(x)`,
`x = xi(t)` (with the transformation), and solves the equations of the
AIR class (equivalent to the inverse of a Riccati equation with linear
coefficients) either directly, through a particular solution, or by
equivalence with the representatives of the database. `abel_ode` uses
it when the invariant is not constant.

```python
>>> from sympy_extras.solvers.abel import abel_invariants, abel_equivalence
>>> abel_invariants((-1, -2*x, 0, 0), x).I1
11664*x**6*(8*x**3 - 15)**3/(8*x**3 - 9)**5
>>> from sympy import Symbol
>>> t = Symbol('t')
>>> abel_equivalence((-1, -2*(x - 1), 0, 0), x, (-1, -2*t, 0, 0), t)
AbelTransformation(xi=x - 1, P=1, Q=0)

```
