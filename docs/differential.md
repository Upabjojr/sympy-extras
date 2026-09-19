# Differential elimination: Janet bases and the Rosenfeld–Gröbner algorithm

`sympy_extras.polys.differential` is the differential counterpart of
Gröbner bases: canonical forms of systems of differential equations which
contain all their integrability conditions, without solving anything.

## What SymPy has and what is added

SymPy solves differential equations (`dsolve`, `pdsolve`) and has Gröbner
bases for polynomial ideals, but nothing which treats a *system* of
differential equations algebraically. It cannot say whether an
overdetermined system of partial differential equations is consistent,
how many solutions it has, whether an equation follows from others, or
what a system implies for some of its unknowns alone.

Here:

- `DifferentialRing(functions, variables, ranking)`: the derivatives of the
  unknown functions with a *ranking* (an orderly one by default; blocks of
  functions give the elimination rankings);
- `janet_basis(equations, functions)`: the Janet basis of a **linear**
  system with coefficients rational in the variables, with `reduce`,
  `contains`, `is_consistent`, `dimension`, `parametric_derivatives`,
  `hilbert_function`, `hilbert_series`, `hilbert_polynomial` and
  `series_solution`;
- `rosenfeld_groebner(equations, functions, inequations=...)`: for a
  **polynomial** system, ordinary or partial, the radical differential
  ideal as an intersection of regular differential systems, with
  `contains` and `is_consistent`;
- `DifferentialPolynomial`: leaders, initials, separants, Ritt's reduction;
- `sympy_extras.solvers.symmetry_janet_basis(equation, function)`: the size
  of the Lie algebra of point symmetries of a differential equation, from
  the Janet basis of its determining equations.

## Janet bases

Janet's example: two innocent equations of the second order hide
integrability conditions of the orders three and four, and the solution
space has the dimension 12.

```python
>>> from sympy import Function, symbols, oo
>>> from sympy_extras.polys.differential import janet_basis
>>> x, y, z, t = symbols('x y z t')
>>> u = Function('u')(x, y, z)
>>> J = janet_basis([u.diff(z, 2) + y*u.diff(x, 2), u.diff(y, 2)], [u])
>>> len(J.equations), J.dimension
(7, 12)
>>> J.contains(u.diff(x, 2, y)), J.contains(u.diff(z, 4)), J.contains(u.diff(z, 3))
(True, True, False)
>>> J.hilbert_series(t)
t**4 + 3*t**3 + 4*t**2 + 3*t + 1

```

The coefficient of `t**q` is the number of Taylor coefficients of order
`q` of a solution which can be chosen freely (the *parametric*
derivatives). When there are infinitely many, the Hilbert polynomial says
how many: the Cauchy–Riemann equations leave two per order, like one
holomorphic function.

```python
>>> f, g = Function('f')(x, y), Function('g')(x, y)
>>> C = janet_basis([f.diff(x) - g.diff(y), f.diff(y) + g.diff(x)], [f, g])
>>> C.dimension, C.hilbert_polynomial(t)
(oo, 2)

```

A system is inconsistent exactly when its basis is `[1]`, and the normal
form of an expression is unique:

```python
>>> v = Function('v')(x, y)
>>> janet_basis([v.diff(x) - y, v.diff(y)], [v]).is_consistent
False
>>> E = janet_basis([v.diff(x, 2) - y**2*v, v.diff(y) - x*v], [v])
>>> E.equations
[-x*v(x, y) + Derivative(v(x, y), y), -y*v(x, y) + Derivative(v(x, y), x)]
>>> E.reduce(v.diff(x, y))
(x*y + 1)*v(x, y)
>>> E.series_solution(2)[v]
C0*x*y + C0

```

(the solutions are the multiples of `exp(x*y)`).

With an elimination ranking the basis contains the consequences of the
system for the lower functions: the conditions under which
`w_x = f, w_y = g` can be solved for `w`.

```python
>>> w = Function('w')(x, y)
>>> K = janet_basis([w.diff(x) - f, w.diff(y) - g], [w, f, g], ranking=[[w], [f, g]])
>>> K.equations[0]
Derivative(f(x, y), y) - Derivative(g(x, y), x)

```

The symbols which are not variables are generic constants. Every equation
is divided by the coefficient of its leader: the basis describes the
solutions at the generic points, and `series_solution` refuses the points
where such a coefficient vanishes.

### The size of a symmetry algebra

The determining equations of the point symmetries of a differential
equation are a linear system for the coefficients `xi(x, u)`, `eta(x, u)`
of the generator. `symmetries` solves them with a polynomial ansatz;
the Janet basis says how many symmetries there are, so that one knows
whether the ansatz found them all.

```python
>>> from sympy_extras.solvers import symmetry_janet_basis, symmetries
>>> s = Function('s')(x)
>>> symmetry_janet_basis(s.diff(x, 2), s).dimension
8
>>> q = Function('q')(x, t)
>>> burgers = q.diff(t) + q*q.diff(x) - q.diff(x, 2)
>>> symmetry_janet_basis(burgers, q).dimension, len(symmetries(burgers, q))
(5, 5)

```

## The Rosenfeld–Gröbner algorithm

For polynomial systems the solutions split into components. Each
component is a *regular differential system*: equations `A = 0` which are
autoreduced and coherent, and inequations `H != 0` which contain the
initials and separants of `A`. By Rosenfeld's lemma the membership in the
differential ideal `[A] : H^oo` is decided in a polynomial ring with
finitely many derivatives, where a Gröbner basis is computed; by Lazard's
lemma that ideal is radical.

The general and the singular solutions of `u'^2 = 4 u` (the parabolas
`(x + c)^2` and their envelope `0`):

```python
>>> from sympy_extras.polys.differential import rosenfeld_groebner
>>> p = Function('p')(x)
>>> I = rosenfeld_groebner([p.diff(x)**2 - 4*p], [p])
>>> [(c.equations, c.inequations) for c in I.components]
[([-4*p(x) + Derivative(p(x), x)**2], [Derivative(p(x), x)]), ([p(x)], [])]
>>> I.components[0].contains(p.diff(x, 2) - 2), I.contains(p.diff(x, 2) - 2)
(True, False)
>>> I.contains(p.diff(x)*(p.diff(x, 2) - 2))
True

```

The pendulum in Cartesian coordinates is a differential-algebraic system
of index three. The algorithm finds its hidden constraints (here the
derivative of the multiplier), and the equilibria as separate components:

```python
>>> g0 = symbols('g')
>>> X, Y, L = Function('X')(t), Function('Y')(t), Function('L')(t)
>>> pendulum = [X.diff(t, 2) + L*X, Y.diff(t, 2) + L*Y + g0, X**2 + Y**2 - 1]
>>> P = rosenfeld_groebner(pendulum, [L, X, Y])
>>> len(P.components)
3
>>> P.components[0].equations[-1]
3*g*Derivative(Y(t), t) + Derivative(L(t), t)
>>> P.components[0].contains(L - X.diff(t)**2 - Y.diff(t)**2 + g0*Y)
True
>>> [c.equations for c in P.components[1:]]
[[Y(t) + 1, X(t), -g + L(t)], [Y(t) - 1, X(t), g + L(t)]]

```

With the ranking `[[X, Y], [L]]` the coordinates are eliminated and the
general component contains one equation of the second order for the
multiplier alone.

An equation follows from a system when every component contains it, and a
system without solutions has no component:

```python
>>> rosenfeld_groebner([p.diff(x) - p, p.diff(x, 2) - p - 1], [p]).is_consistent
False

```

## Limits

- The coefficients are rational functions of the variables and of
  constant parameters; the parameters are generic (no case distinction).
- The rankings are orderly inside blocks of functions.
- Janet bases of dense systems with polynomial coefficients can have very
  large intermediate coefficients.
- The components of `rosenfeld_groebner` may be redundant, and they are
  regular differential systems, not characteristic sets of prime
  components: equations are split along their factors over the
  rationals, but no decomposition into regular chains is made, so that
  `reduce`-like normal forms exist for Janet bases only.

## References

- M. Janet, *Leçons sur les systèmes d'équations aux dérivées partielles*, 1929.
- D. Robertz, *Formal Algorithmic Elimination for PDEs*, LNM 2121, Springer 2014.
- V. P. Gerdt, Yu. A. Blinkov, Involutive bases of polynomial ideals, Math. Comput. Simulation 45 (1998).
- F. Boulier, D. Lazard, F. Ollivier, M. Petitot, Representation for the
  radical of a finitely generated differential ideal, ISSAC 1995; Computing
  representations for radicals of finitely generated differential ideals,
  AAECC 20 (2009).
- E. Hubert, Notes on triangular sets and triangulation-decomposition
  algorithms II: differential systems, LNCS 2630 (2003).
- E. R. Kolchin, *Differential Algebra and Algebraic Groups*, 1973.
