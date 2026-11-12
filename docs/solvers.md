# Differential equations: Lie point symmetries, linear equations, first order PDEs, and solving with assumptions

Modules: `sympy_extras.solvers` and `sympy_extras.assumptions.solve` (native,
on the core) with its SymPy face `sympy_extras.assumptions.solve_sympy`.

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
  first, then, for a first order equation, the solvers of
  `dsolve_first_order` (Riccati, Abel, Chini, Prelle–Singer), and the
  symmetry method when they fail.

Every step handed to SymPy (integration, `dsolve`, `solve`, the checks)
runs under a time limit (`timeout` argument, 30 s by default) so that the
solvers always return.

```python
>>> from sympy import Function, symbols
>>> from sympy_extras.solvers.pde_sympy import pde_symmetries, pdsolve_lie
>>> from sympy_extras.solvers.ode_sympy import dsolve_lie
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

`sympy_extras.assumptions.solve_sympy.solve(equations, symbols, assumptions, domain)`
(the native `sympy_extras.assumptions.solve.solve` takes and returns objects
of the core) is `Solve` with `Assumptions` and a domain. One unknown with polynomial
equations and inequalities over the reals goes to the cylindrical
algebraic decomposition (exact intervals and points with algebraic
endpoints); otherwise `solveset` or `nonlinsolve` solve the equations,
with the parameters carrying the assumptions, and each solution is kept,
dropped or put in a `ConditionSet` according to whether the assumptions
hold at it, fail, or cannot be decided.

```python
>>> from sympy import S
>>> from sympy.abc import a, y
>>> from sympy_extras.assumptions.solve_sympy import solve
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
(`sympy_extras.polys.cad.cylindrical_sympy.cylindrical_set`, see [docs/cad.md](cad.md)). The
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
on `sympy_extras.polys.regularchains`, both native; their SymPy faces are
`solvers.parametric_sympy` and `polys.regularchains_sympy`): the zeros of the system are the
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
with the parameters first (`sympy_extras.polys.cad.cylindrical_sympy.cylindrical_cases`):
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
  poles). `sympy_extras.solvers.kovacic` is native (core expressions,
  the native polynomials, roots, series and linear systems); its SymPy
  face is `sympy_extras.solvers.kovacic_sympy`; the antiderivatives
  come from the native verified indefinite integration
  (`sympy_extras.integrals.indefinite.closed_antiderivative`, SymPy's
  routes its last methods), and the rational solutions of the Riccati
  equation, the fallback of the first case, are the native
  `kovacic.rational_riccati` (the hyperexponential solutions of
  `z'' = r z`).
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
  `exp(x**2)` times a polynomial is found where it exists), and
  **exponential parts at the irregular finite singular points** from the
  local Newton polygons (`sympy_extras.solvers.irregular`): the terms
  `r/(x - c)**m`, `m >= 2`, of `y'/y` come from the edges of slope `m` of
  the polygon of the orders of the coefficients at `c` and the roots `r`
  of the characteristic equation of the edge, term by term; at the
  conjugate roots of an irreducible factor the characteristic equation
  is solved in the field of the point and the conjugate parts are summed
  by a trace, so everything stays over the rationals. `x**4 y'' + 2
  x**3 y' - y = 0` gives `exp(1/x)` and `exp(-1/x)`, `exp(1/(x**2 - 2))` is
  found from its equation. Ramified parts (slopes which are not
  integers) are not hyperexponential and are dropped; a leading
  coefficient outside the field of the point makes the search
  incomplete.
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

`linear_ode`, `factorization`, `irregular` and `special` are native
(core expressions, the native polynomials, roots, number fields, linear
algebra and simplification); their SymPy faces are `linear_ode_sympy`,
`factorization_sympy`, `irregular_sympy` and `special_sympy`, which the
examples below use. The quadratures come from the native verified
indefinite integration (`sympy_extras.integrals.indefinite.closed_antiderivative`,
an integral it cannot do left as the unevaluated
`sympy_extras.integrals.integral.Integral`); the last resort on the
reduced equations is the native classical hints of
`sympy_extras.solvers.dsolve` (constant coefficients, Euler's equations,
the reduction to the lowest derivative), and the Grassmann–Plücker
relations of Beke's method are solved by the native port of SymPy's
`solve` (`sympy_extras.solvers.solvers.solve_system`).

```python
>>> from sympy import Matrix, Function
>>> from sympy.abc import n
>>> from sympy_extras.solvers.special_sympy import special_solutions
>>> from sympy_extras.solvers.linear_systems_sympy import dsolve_linear_system
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
(`hyperexponential_search` says when it is not: irrational exponents,
exponential parts with leading coefficients outside the field of the
point); for second order factors Kovacic's algorithm decides it.

`dsolve_linear` uses the factorisation for equations of order three and
more: the solutions of the right factor, then for each solution `z` of
the left factor a solution of `R(y) = z` by variation of parameters,
with the integrals left unevaluated when SymPy cannot evaluate them.

```python
>>> from sympy import Function
>>> from sympy_extras.solvers.linear_ode_sympy import LinearOperator, gcrd, lclm, dsolve_linear
>>> from sympy_extras.solvers.factorization_sympy import factor_operator, right_factor, left_factor
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
equations for high orders) and the eigenring.

## Linear systems with rational coefficients

`sympy_extras.solvers.linear_systems` solves `Y' = A(x) Y + b(x)` with `A`
an `n x n` matrix of rational functions. SymPy's `dsolve` solves systems
with constant coefficients, systems whose matrix commutes with its
antiderivative, and a few special types of two and three equations.

The module is native: it takes and returns the `ExprMatrix` of core
expressions, with the native scalar solvers, polynomials, linear algebra
(over the field of rational functions) and simplification; its SymPy face
`linear_systems_sympy`, which the examples below use, keeps the API on
SymPy matrices. The antiderivatives of rational functions are native
(`rational_integral`: `ratint_ratpart` and `ratint_logpart` of
`integrals.risch.rationaltools`, with the native `RootSum` of
`polys.rootoftools`). SymPy's `integrate` (Liouville's formula, variation
of constants) is native too, and so is the last resort for a
fundamental matrix, SymPy's `linodesolve` of a matrix `A = g'(x) M` with
`M` constant: `exp(M g(x))` by the native matrix exponential.

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
  the matrix exponential of a constant system (Putzer) and SymPy's
  `linodesolve` are tried when they fall short.
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
>>> from sympy_extras.solvers.linear_systems_sympy import dsolve_system, rational_system_solutions
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
>>> from sympy_extras.solvers.first_order_sympy import lagrange_ode, abel_ode, riccati_ode
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
>>> from sympy_extras.solvers.linear_ode_sympy import dsolve_linear
>>> from sympy_extras.solvers.kovacic_sympy import dsolve_kovacic, liouvillian_solution
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

## Linear recurrences: hypergeometric and d'Alembertian solutions

`sympy_extras.solvers.recurr` solves linear recurrences with polynomial
(or rational) coefficients, `p_d(n) y(n + d) + ... + p_0(n) y(n) = f(n)`.
SymPy's `rsolve` finds polynomial, rational and hypergeometric solutions
(Petkovšek's Hyper) and nothing else: for `y(n + 1) = (n + 1) y(n) + 1` it
returns `None`, and for the recurrence of the harmonic numbers,
`(n + 2) y(n + 2) = (2n + 3) y(n + 1) - (n + 1) y(n)`, it returns the
constants and misses `H_n`. The module adds the **d'Alembertian
solutions** of Abramov and Petkovšek, the nested indefinite sums of
hypergeometric terms, which are exactly the solutions of the operators
that factor into first order factors over the rational functions:

- `rsolve_polynomial(coefficients, n)`: the polynomial solutions, with the
  degree bound from the first nonvanishing polynomial `b_k(e)` of
  `L[n^e] = sum b_k(e) n^(D + e - k)`;
- `rsolve_hypergeometric(coefficients, n)`: Petkovšek's algorithm, written
  from the paper: `a | p_0`, `b | p_d(n - d + 1)` with
  `gcd(a(n), b(n + h)) = 1` for `h >= 0` (decided by comparing the
  irreducible factors of `a` and `b` under integer shifts, which is far
  cheaper than the resultant), `z` from the leading coefficients, `c` a
  polynomial solution of the auxiliary recurrence; the terms are grouped
  into similarity classes (the quotient of their ratios is `q(n + 1)/q(n)`
  for a polynomial `q`, a polynomial solution of a first order
  recurrence), so that the result is a basis; `gosper_form` gives the
  Gosper–Petkovšek form of a ratio;
- `rsolve_dalembertian(coefficients, n)`: a hypergeometric solution `h`
  gives `L = L' (N - r)`, the substitution `y = h sum z` reduces the order
  by one, and the solutions are `h` and the `h sum_{k < n} z(k)` for the
  d'Alembertian solutions `z` of the reduced equation; the hypergeometric
  solutions are listed first and a maximal independent subset is returned
  (independence decided from the values at `d` consecutive points past
  the roots of the leading coefficient, which determine a solution);
- `rsolve_linear(equation, y(n), init=None)`: the equation written with
  `y(n + i)` (negative shifts and rational coefficients allowed), the
  general solution with constants `C0, C1, ...`, a particular solution
  when the right-hand side is a sum of hypergeometric terms (each term
  `f_j` is annihilated by `N - f_j(n + 1)/f_j(n)`; a d'Alembertian solution
  `u` of the operator of order `d + 1` has `L u = kappa f_j` with a
  constant `kappa`, read off at one point), the constants from the initial
  values; `None` when the d'Alembertian solutions do not span the solution
  space.

The solutions are written with `factorial`, `RisingFactorial`, `Product`
(for the irreducible factors of degree two and more), `Sum` and `z**n`;
the sums the summation (SymPy's, ported) evaluates without special
functions are evaluated (`harmonic(n)` for the harmonic numbers,
`2**n - n - 1` for `y(n + 1) = 2 y(n) + n`;
`Sum(1/factorial(k + 1), (k, 0, n - 1))` is kept because the summation
writes it with the incomplete gamma function). A product
whose factor vanishes at a nonnegative integer starts after it, and the
solution holds from there.
Every solution is checked by substitution at six consecutive integers
before it is returned (`is_recurrence_solution`).

The module is native: it takes and returns expressions of the core and
does not import SymPy (`zeilberger_sum`, native too, calls it directly). The
coefficients are dense polynomials in `n` over `QQ`, over the rational
functions `QQ(a, b, ...)` of the parameters, or over a number field
`QQ(alpha)` for one radical `alpha`; a root `z` outside the field is
adjoined (an `AlgebraicField` over `QQ`, a simple extension of `QQ(a, ...)`
otherwise: `sqrt(a)**n` for `y(n + 2) = a y(n)`). The d'Alembertian
reduction needs a factorisation over the field of the hypergeometric
solution it reduces with, which exists over `QQ`, `QQ(a, ...)` and the
number fields over `QQ`; the roots of degree five and more that the
ported `roots` does not find (SymPy's `CRootOf`) are left out, and the
closed forms of the sums are computed without a time limit.

```python
>>> from sympy_extras.core import Eq, Function, Symbol
>>> from sympy_extras.solvers.recurr import rsolve_linear, rsolve_dalembertian, rsolve_hypergeometric
>>> n = Symbol('n')
>>> u = Function('u')
>>> rsolve_linear(u(n + 1) - (n + 1)*u(n) - 1, u(n))
C0*factorial(n) + factorial(n)*Sum(1/factorial(k + 1), (k, 0, n - 1))
>>> rsolve_linear(Eq((n + 2)*u(n + 2), (2*n + 3)*u(n + 1) - (n + 1)*u(n)), u(n), {u(0): 0, u(1): 1})
harmonic(n)
>>> rsolve_linear(u(n + 1) - n*(u(n) + u(n - 1)), u(n), {u(0): 1, u(1): 0})   # derangements
-factorial(n)*Sum((-1)**k/factorial(k + 1), (k, 0, n - 1)) + factorial(n)
>>> rsolve_linear(u(n + 2) - u(n + 1) - u(n), u(n), {u(0): 0, u(1): 1})
-sqrt(5)*(1/2 - sqrt(5)/2)**n/5 + sqrt(5)*(1/2 + sqrt(5)/2)**n/5
>>> rsolve_hypergeometric([n**2 + 1, -(n + 1)], n)
[Product(k**2 + 1, (k, 0, n - 1))/factorial(n)]
>>> rsolve_dalembertian([n + 1, -(n + 3), 1], n)
[factorial(n), factorial(n)*Sum(1/factorial(k + 1), (k, 0, n - 1))]
>>> rsolve_dalembertian([-(n + 1), -1, 1], n)    # the involution numbers: no d'Alembertian solution
[]

```

On random operators `(N - r_3)(N - r_2)(N - r_1)` with ratios made of
small linear factors, the solver returns the full basis of `d`
independent solutions (the oracle used in the tests, each solution also
verified by substitution).

## Guessing recurrences and equations from the first terms

`sympy_extras.concrete.guess` goes the other way: from the first terms of
a sequence to an equation, which the solvers above then solve. Three
kinds of equations with polynomial coefficients are searched, each a
linear problem in the unknown coefficients (Salvy–Zimmermann's gfun,
Kauers' Guess):

- `guess_recurrence(terms)`: `sum_i p_i(n) a(n + i) = 0`;
- `guess_differential_equation(terms)`: `sum_k q_k(x) f^(k)(x) = 0`
  (possibly with a polynomial right-hand side) for the generating
  function `f = sum a(n) x**n`;
- `guess_algebraic_equation(terms)`: `sum_k q_k(x) f**k = 0` (degree one:
  a rational generating function).

An equation is accepted only when the linear system has `margin` (three
by default) more equations than unknowns, so that it has passed that many
independent checks; among the orders and degrees which the number of
terms allows, the one with the fewest coefficients is returned. The
search runs modulo the prime `2**61 - 1` (for each order the largest
degree, then the smallest by bisection) and the exact null space is
computed once. `guess_closed_form(terms, n)` solves the recurrence with
`rsolve_linear`, the initial values taken after the last zero of the
leading coefficient, and checks the expression on every term.

```python
>>> from math import comb
>>> from sympy_extras.core import Function, Symbol
>>> from sympy_extras.concrete.guess import (guess_recurrence, guess_differential_equation,
...     guess_algebraic_equation, guess_closed_form)
>>> n, x = Symbol('n'), Symbol('x')
>>> apery = [sum(comb(m, k)**2*comb(m + k, k)**2 for k in range(m + 1)) for m in range(20)]
>>> apery[:5]
[1, 5, 73, 1445, 33001]
>>> guess_recurrence(apery).as_expr(Function('a'), n)
(-34*n**3 - 153*n**2 - 231*n - 117)*a(n + 1) + (n**3 + 3*n**2 + 3*n + 1)*a(n) + (n**3 + 6*n**2 + 12*n + 8)*a(n + 2)
>>> motzkin = [1, 1, 2, 4, 9, 21, 51, 127, 323, 835, 2188, 5798, 15511, 41835]
>>> guess_algebraic_equation(motzkin).as_expr(x, Symbol('f'))
f**2*x**2 + f*(x - 1) + 1
>>> guess_closed_form([1, 0, 1, 2, 9, 44, 265, 1854, 14833, 133496, 1334961, 14684570], n)
-factorial(n)*Sum((-1)**k/factorial(k + 1), (k, 0, n - 1)) + factorial(n)

```

The first is Apéry's recurrence for `zeta(3)`; with 40 terms
`guess_differential_equation` finds the differential equation of order
three of their generating function. A guess is a conjecture which holds
on the terms given: a few exceptions to a pattern are absorbed by a
coefficient vanishing at them, and more terms settle such a case.

## Exponential Diophantine equations

`sympy_extras.solvers.exponential_diophantine` solves two families
completely, with a proof that nothing is missed.

`pillai(a, b, c)` returns all the non-negative integers `x, y` with
`a**x - b**y = c` (`a, b >= 2`, `c != 0`). For multiplicatively
dependent bases the equation is `g**u - g**v = c` and the smaller power
divides `c`. Otherwise `Lambda = x log a - y log b = log(1 + c/b**y)` is
a non-zero linear form in two logarithms with `|Lambda| <= 2 |c|/b**y`,
and the theorem of Baker and Wüstholz bounds `y` (by about `10**11`);
the continued fraction of `log b/log a` reduces the bound: a solution
beyond a few dozen would make `x/y` a convergent approximating too well
for the partial quotients met below the Baker bound. The rest is
searched with exact integers.

`exponential_polynomial_roots(terms)` returns all the integers `n` with
`sum p_i(n) a_i**n = 0` (rational polynomials, positive rational bases):
beyond a bound which is logarithmic in the coefficients the term of the
largest base dominates wherever its polynomial does not vanish, and the
zeros beyond it are the common zeros of that polynomial and of the sum
of the other terms, found recursively; negative `n` by the mirrored sum.

```python
>>> from fractions import Fraction
>>> from sympy_extras.solvers.exponential_diophantine import pillai, exponential_polynomial_roots
>>> pillai(3, 2, 1)
[(1, 1), (2, 3)]
>>> pillai(2, 3, 13)
[(4, 1), (8, 5)]
>>> pillai(13, 3, 10)
[(1, 1), (3, 7)]
>>> one = [Fraction(1)]
>>> exponential_polynomial_roots([(one, Fraction(3)), (one, Fraction(4)), ([Fraction(-1)], Fraction(5))])    # 3**n + 4**n = 5**n
[2]
>>> exponential_polynomial_roots([(one, Fraction(2)), ([Fraction(-1), Fraction(0), Fraction(0)], Fraction(1))])   # 2**n = n**2
[2, 4]

```

With coefficients, `pillai(a, b, c, r, s)` solves `r a**x - s b**y = c`:
the linear form has the third logarithm `log(r/s)`, the Baker–Wüstholz
bound for three logarithms is reduced by the lemma of Baker and Davenport
(Dujella–Pethő: a convergent `p/q` of `log a/log b` with `q log(r/s)/log
b` far from an integer), and a coefficient made of the primes of the
bases, `(r/s)**k = a**i b**j`, brings back two logarithms and continued
fractions.

```python
>>> pillai(5, 2, -3, s=2)          # 5**x - 2*2**y = -3: 125 - 128
[(0, 1), (1, 2), (3, 6)]

```

Not covered: more than two exponentials (S-unit equations, which need
the LLL reduction of `polys.lattice`), unknown bases.

## Exact linear programming with certificates

`sympy_extras.solvers.simplex.linear_program(c, A, b)` maximises `c . x`
on `A x <= b` over the rationals (unknowns of either sign) by the
two-phase simplex method in exact arithmetic with Bland's rule, and
returns a certificate with every answer, checked by a few products
whatever the method did:

- *optimal*: the point and multipliers `y >= 0` with `y^T A = c` and
  `y . b = c . x` (strong duality);
- *infeasible*: `y >= 0` with `y^T A = 0` and `y . b < 0` (Farkas' lemma);
- *unbounded*: a feasible point and a ray `d` with `A d <= 0`, `c . d > 0`.

`linear_feasible(A, b, strict)` decides systems with strict inequalities
on chosen rows (a common slack is maximised) and returns a point: the
linear fragment of real arithmetic, without a cylindrical decomposition.

```python
>>> from sympy_extras.solvers.simplex import linear_program, linear_feasible
>>> r = linear_program([5, 4, 3], [[2, 3, 1], [4, 1, 2], [3, 4, 2], [-1, 0, 0], [0, -1, 0], [0, 0, -1]], [5, 11, 8, 0, 0, 0])
>>> r.value == 13, r.point == [2, 0, 1], r.dual == [1, 0, 1, 0, 3, 0]
(True, True, True)
>>> linear_program([0, 0], [[1, 1], [-1, 0], [0, -1]], [-1, 0, 0]).dual == [1, 1, 1]      # x + y <= -1, x, y >= 0
True
>>> linear_feasible([[1], [-1]], [1, -1], [True, False]) is None                      # x < 1 and x >= 1
True

```

`integer_program(c, A, b)` maximises over the integer points by branch
and bound on the exact relaxations. It always terminates, also on an
unbounded polyhedron without integer points: an integer optimum, if
there is one, lies within `n Delta(A)` of the optimum of the relaxation
(Cook, Gerards, Schrijver and Tardos), and a polyhedron with integer
points has one of size `(n + 1) Delta([A b])`; `Delta`, the largest
subdeterminant, is bounded by Hadamard's inequality, and the search
stays in these boxes.

```python
>>> from sympy_extras.solvers.simplex import integer_program
>>> integer_program([5, 4], [[6, 4], [1, 2], [-1, 0], [0, -1]], [24, 6, 0, 0])      # the relaxation: 21 at (3, 3/2)
IntegerProgramResult('optimal', value=20, point=[4, 0])
>>> integer_program([0, 0], [[3, -3], [-3, 3]], [2, -1]).status                     # 1/3 <= x - y <= 2/3
infeasible

```

`eliminate_variables(A, b, variables, strict)` projects a system onto
the other unknowns by Fourier–Motzkin elimination (quantifier elimination
for existential linear formulas): the upper and the lower bounds of the
eliminated unknown are combined, a combination being strict when one of
the two is, and the inequalities a linear program proves redundant are
removed after each unknown.

```python
>>> from sympy_extras.solvers.simplex import eliminate_variables
>>> shadow = eliminate_variables([[-1, 0], [0, -1], [1, 1]], [0, 0, 1], [1])     # the triangle, without y
>>> sorted(zip(shadow.A, shadow.b)) == [([-1], 0), ([1], 1)]
True

```

None of this is called yet by `ask`, `satisfiable` or `resolve`, which
still send linear formulas to virtual substitution and Cooper's method.

## Darboux polynomials and the Prelle–Singer procedure

`sympy_extras.solvers.darboux` solves `y' = P(x, y)/Q(x, y)` with
polynomial `P` and `Q` by Darboux's method: the *Darboux polynomials*
of the field `D = Q d/dx + P d/dy` (`D f = g f` with a polynomial
cofactor `g`, their zero sets are solution curves) and the
*exponential factors* `exp(A/B)` (`D(A/B)` a polynomial) are searched up
to a degree bound, and a linear system for the exponents gives either a
first integral `prod f_i**c_i exp(...)` at once (a vanishing combination
of the cofactors) or a Darboux integrating factor `R` (`sum c_i g_i = -div
D`), whose quadrature is the first integral. Prelle and Singer proved
that an equation with an elementary first integral has such an
integrating factor with rational exponents, Singer that a Liouvillian
first integral comes with exponential factors: within the degree bound
the procedure is complete. The Darboux polynomials of degree `N` are
found from their highest degree part (a product of factors of `x P_d -
y Q_d`, or any form when that vanishes), the rest by linear propagation
and a parametric elimination in the few cofactor coefficients left.
SymPy's `dsolve` has nothing of the kind; `dsolve_first_order` tries it
after the Riccati, Abel and Chini classes. The module works on the core
(the polynomials over `QQ`, number fields and fields of parameters are the
typed ones, only the quadratures go through SymPy's `integrate`);
`sympy_extras.solvers.darboux_sympy` is its face on SymPy objects, used here.

```python
>>> from sympy_extras.solvers.darboux_sympy import (darboux_polynomials, darboux_integrating_factor,
...     darboux_first_integral, prelle_singer)
>>> from sympy.abc import x
>>> from sympy import Symbol
>>> Y = Symbol('y')
>>> darboux_polynomials(Y*(x - 1), x*(1 - Y), x, Y)             # Lotka–Volterra
[DarbouxPolynomial(y, x - 1), DarbouxPolynomial(x, 1 - y)]
>>> darboux_integrating_factor(Y*(x - 1), x*(1 - Y), x, Y)
1/(x*y)
>>> darboux_first_integral(Y*(x - 1), x*(1 - Y), x, Y)
-x - y + log(x) + log(y)
>>> darboux_first_integral(x + Y, x - Y, x, Y)                  # x**2 + y**2 is Darboux
-log(x**2 + y**2)/2 - atan(x/y)
>>> darboux_first_integral(x*Y + 1, 1, x, Y)                    # the factor exp(-x**2/2)
y*exp(-x**2/2) - sqrt(2)*sqrt(pi)*erf(sqrt(2)*x/2)/2
>>> prelle_singer(y.diff(x) - (x + y)/(x - y), y)
Eq(-log(x**2 + y(x)**2)/2 - atan(x/y(x)), C1)
>>> prelle_singer(x**2*y.diff(x) - x*y + 1, y)                  # explicit when rational in y
Eq(y(x), C1*x + 1/(2*x))
>>> prelle_singer(y.diff(x) - y**2 - x, y) is None              # Airy's Riccati equation
True

```

The bound (`degree`, 4 by default) is raised one by one and the search
stops at the first degree which gives a first integral; the coefficients
may lie in an algebraic extension (`extension=I` finds `x + I*y` and
`x - I*y` rather than their product; over the rationals the search is
repeated over the roots of an irreducible quadratic factor of
`x P_d - y Q_d` when nothing is found) or contain parameters. The
solution is explicit when the first integral is rational in `y` with
one root, implicit otherwise (a transcendental relation is not cut to
one branch). Every step
runs under the time limit of the settings; a field with a rational first
integral of high degree, whose Darboux polynomials come in families,
is the expensive case.

## First order nonlinear PDEs

`sympy_extras.solvers.charpit.complete_integral` finds complete
integrals (solutions with two arbitrary constants) of
`F(x, y, u, u_x, u_y) = 0` by Charpit's method: the standard first
integrals of the characteristic system (`F(p, q) = 0`, `F(u, p, q) = 0`,
separable and Clairaut equations, `p = a` or `q = a` when they are first
integrals) and the integration of `du = p dx + q dy`. SymPy's `pdsolve`
handles first order linear equations only.

```python
>>> from sympy import Eq, Function, symbols
>>> from sympy_extras.solvers.charpit_sympy import complete_integral
>>> x, y = symbols('x y')
>>> u = Function('u')(x, y)
>>> complete_integral(u.diff(x)*u.diff(y) - 1, u)
Eq(u(x, y), a*x + b + y/a)
>>> complete_integral(Eq(u, u.diff(x)*x + u.diff(y)*y + u.diff(x)*u.diff(y)), u)
Eq(u(x, y), a*b + a*x + b*y)

```

## Isolated values of the parameters

A general solution of an ODE with parameters, found for generic values
of them, may fail at isolated values in two ways: it is **undefined**
there (the resonance of `y' - k*y = exp(x)` at `k = 1`, where
`exp(x)/(1 - k)` divides by zero), or it is defined but **no longer
general** (`C1*exp(a*x) + C2*exp(b*x)` for `y'' - (a + b)*y' + a*b*y = 0`
at `a = b`, a repeated characteristic root, where both exponentials are
the same function). `solve_ode`, `dsolve_lie`, `dsolve_first_order`,
`riccati_ode`, `dsolve_second_order`, `dsolve_linear` and
`dsolve_kovacic` give such values cases of their own, first, as the
integrals and the sums do (`sympy_extras._special_values`):

- the candidates are the zeros of the denominators (and of the
  arguments of logarithms) of the solution, and the zeros of the
  Jacobian determinant of `(y, y', ..., y^(m-1))` with respect to the
  `m` constants: for a solution linear in the constants it is the
  Wronskian of the fundamental system, which vanishes where the roots
  of the characteristic or indicial polynomial coincide (the square
  root of its discriminant is a factor), and for a nonlinear family,
  such as the Riccati solution `-(u1' + C1*u2')/(u1 + C1*u2)`, it
  vanishes where a constant disappears;
- the equation is solved again at each point, by the same solver, and
  the solution there is checked: it satisfies the equation (the residual
  simplified to zero, or zero at sample points to 30 digits by
  `reliable_value`), carries the same constants and is general;
- a point where the equation is undefined or where its order drops
  (`k*y'' + y' = 0` at `k = 0`, a singular perturbation) gets no case;
  the work has a budget of a tenth of the time limit.

`dsolve_linear` and `dsolve_kovacic` return a list of functions: the
`i`-th function becomes a `Piecewise` whose case at a point is the
`i`-th solution of the equation there. `special_values=False` turns
the cases off.

```python
>>> from sympy import Function, symbols, exp, sin
>>> from sympy_extras.solvers.ode_sympy import solve_ode
>>> from sympy_extras.solvers.linear_ode_sympy import dsolve_linear
>>> x, k, a = symbols('x k a')
>>> y = Function('y')(x)
>>> solve_ode(y.diff(x) - k*y - exp(x), y)
[Eq(y(x), Piecewise(((C1 + x)*exp(x), Eq(k, 1)), (C1*exp(k*x) - exp(x)/(k - 1), True)))]
>>> solve_ode(y.diff(x, 2) + k**2*y - sin(x), y)
[Eq(y(x), Piecewise((C1*sin(x) + C2*cos(x) - x*cos(x)/2, Eq(k, -1) | Eq(k, 1)), (C1 + C2*x - sin(x), Eq(k, 0)), (C1*exp(-I*k*x) + C2*exp(I*k*x) + sin(x)/(k**2 - 1), True)))]
>>> dsolve_linear(x**2*y.diff(x, 2) + a*x*y.diff(x) + 4*y, y)[1]
Piecewise((x**2*log(x), Eq(a, -3)), (log(x)/x**2, Eq(a, 5)), (exp(-(a + sqrt(a**2 - 2*a - 15) - 1)*log(x)/2), True))

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
>>> from sympy_extras.solvers.dae_sympy import dae_matrices, dsolve_dae, dae_index
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
>>> from sympy_extras.solvers.second_order_sympy import dsolve_second_order
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
>>> from sympy_extras.solvers.abel_sympy import abel_invariants, abel_equivalence
>>> abel_invariants((-1, -2*x, 0, 0), x).I1
11664*x**6*(8*x**3 - 15)**3/(8*x**3 - 9)**5
>>> from sympy import Symbol
>>> t = Symbol('t')
>>> abel_equivalence((-1, -2*(x - 1), 0, 0), x, (-1, -2*t, 0, 0), t)
AbelTransformation(xi=x - 1, P=1, Q=0)

```
