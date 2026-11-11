# Curvature of a metric, curves and surfaces, vector calculus

`sympy_extras.diffgeom.Metric` computes, from the components $g_{ij}$ of
a (pseudo-)Riemannian metric in coordinates, the objects of Riemannian
geometry and of general relativity: the Christoffel symbols, the
Riemann, Ricci, Einstein and Weyl tensors, the scalar curvature, the
Kretschmann scalar, geodesics, covariant derivatives and the Killing
equations. The conventions are those of Misner, Thorne and Wheeler:

$$\Gamma^i_{jk} = \tfrac12 g^{il} (\partial_j g_{lk} + \partial_k g_{lj} - \partial_l g_{jk}), \qquad
R^i{}_{jkl} = \partial_k \Gamma^i_{lj} - \partial_l \Gamma^i_{kj} + \Gamma^i_{km} \Gamma^m_{lj} - \Gamma^i_{lm} \Gamma^m_{kj},$$

$R_{jl} = R^i{}_{jil}$, $R = g^{jl} R_{jl}$, so that a sphere has
positive scalar curvature. Indices count from zero, in the order of the
coordinates.

```python
>>> from sympy_extras.core import Function, Symbol
>>> from sympy_extras.functions import sin, cos
>>> from sympy_extras.diffgeom import Metric
>>> t, r, theta, phi, a, m = (Symbol(s) for s in ['t', 'r', 'theta', 'phi', 'a', 'm'])
>>> sphere = Metric([[a**2, 0], [0, a**2*sin(theta)**2]], [theta, phi])
>>> sphere.christoffel_symbols()
{(0, 1, 1): -sin(theta)*cos(theta), (1, 0, 1): cos(theta)/sin(theta)}
>>> sphere.riemann(0, 1, 0, 1), sphere.ricci_scalar()
(sin(theta)**2, 2/a**2)

```

## Spacetimes

The Schwarzschild metric is a vacuum solution, and its Kretschmann
scalar shows that $r = 2m$ is not a singularity of the curvature:

```python
>>> f = 1 - 2*m/r
>>> schwarzschild = Metric([[-f, 0, 0, 0], [0, 1/f, 0, 0], [0, 0, r**2, 0], [0, 0, 0, r**2*sin(theta)**2]],
...                        [t, r, theta, phi])
>>> schwarzschild.ricci()
ExprMatrix([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
>>> schwarzschild.kretschmann(), schwarzschild.weyl(0, 1, 0, 1)
(48*m**2/r**6, -2*m/r**3)
>>> schwarzschild.is_killing([1, 0, 0, 0]), schwarzschild.is_killing([0, 1, 0, 0])
(True, False)

```

The entries may contain undefined functions of the coordinates; the
Friedmann equations come from the Einstein tensor of the
Robertson–Walker metric:

```python
>>> A, k = Function('a')(t), Symbol('k')
>>> frw = Metric([[-1, 0, 0, 0], [0, A**2/(1 - k*r**2), 0, 0], [0, 0, A**2*r**2, 0],
...               [0, 0, 0, A**2*r**2*sin(theta)**2]], [t, r, theta, phi])
>>> frw.einstein()[0, 0]
(3*k + 3*Derivative(a(t), t)**2)/a(t)**2
>>> frw.ricci_scalar()
(6*k + 6*a(t)*Derivative(a(t), (t, 2)) + 6*Derivative(a(t), t)**2)/a(t)**2

```

A metric can be given by its line element, and the geodesic equations
are written for functions of a parameter:

```python
>>> dr, dphi, s = Symbol('dr'), Symbol('dphi'), Symbol('s')
>>> polar = Metric.from_line_element(dr**2 + r**2*dphi**2, [r, phi], [dr, dphi])
>>> polar.is_flat(), polar.geodesic_equations(s)[0]
(True, -r(s)*Derivative(phi(s), s)**2 + Derivative(r(s), (s, 2)))
>>> polar.laplacian(Function('u')(r, phi))
(r**2*Derivative(u(r, phi), (r, 2)) + r*Derivative(u(r, phi), r) + Derivative(u(r, phi), (phi, 2)))/r**2

```

## Curves and surfaces

`sympy_extras.diffgeom.classical` has the classical objects: for a
parametrised `Curve` the speed, the curvature, the torsion and the Frenet
frame; for a parametrised `Surface` the two fundamental forms, the unit
normal, the Gaussian, mean and principal curvatures. The symbols are
real; square roots of squares are taken out, as absolute values unless
the factor is known to be positive.

```python
>>> from sympy_extras.diffgeom import Curve, Surface
>>> from sympy_extras.functions import cosh
>>> u, v = Symbol('u'), Symbol('v')
>>> p, q = Symbol('p', positive=True), Symbol('q', positive=True)
>>> helix = Curve([p*cos(t), p*sin(t), q*t], t)
>>> helix.curvature(), helix.torsion()
(p/(p**2 + q**2), q/(p**2 + q**2))
>>> Curve([t, t**2], t).curvature()
2/(4*t**2 + 1)**(3/2)
>>> torus = Surface([(2 + cos(u))*cos(v), (2 + cos(u))*sin(v), sin(u)], u, v)
>>> torus.first_fundamental_form(), torus.gaussian_curvature()
((1, 0, cos(u)**2 + 4*cos(u) + 4), cos(u)/(cos(u) + 2))
>>> Surface([cosh(u)*cos(v), cosh(u)*sin(v), u], u, v).mean_curvature()          # the catenoid
0

```

The Gaussian curvature is computed from the determinants $[\mathbf{r}_{uu},
\mathbf{r}_u, \mathbf{r}_v]$, without the square root of the normal. By
the Theorema Egregium it equals half the scalar curvature of the first
fundamental form, which `Metric` computes independently:

```python
>>> from sympy_extras.simplify.normalform import normal_form
>>> E, F, G = torus.first_fundamental_form()
>>> normal_form(Metric([[E, F], [F, G]], [u, v]).ricci_scalar()/2 - torus.gaussian_curvature())
0

```

## Vector calculus in orthogonal coordinates

`sympy_extras.vector.OrthogonalCoordinates` holds orthogonal coordinates
with their scale factors $h_i$ and computes the gradient, the
divergence, the curl and the Laplacian of fields given by their physical
components (along the unit tangent vectors), as in the tables of the
textbooks. `cartesian`, `cylindrical` and `spherical` are predefined;
`from_transformation` computes the scale factors from the Cartesian
coordinates as functions of the new ones and checks the orthogonality.

```python
>>> from sympy_extras.vector import OrthogonalCoordinates, cylindrical, spherical
>>> S = spherical(r, theta, phi)
>>> S.gradient(1/r), S.laplacian(1/r), S.divergence([r, 0, 0])
([-1/r**2, 0, 0], 0, 3)
>>> S.curl([0, 0, r*sin(theta)])
[2*cos(theta), -2*sin(theta), 0]
>>> rho, z, u, v = (Symbol(s) for s in ['rho', 'z', 'u', 'v'])
>>> cylindrical(rho, phi, z).laplacian(Function('f')(rho))
(rho*Derivative(f(rho), (rho, 2)) + Derivative(f(rho), rho))/rho
>>> parabolic = OrthogonalCoordinates.from_transformation([u, v], [(u**2 - v**2)/2, u*v])
>>> parabolic.scale_factors, parabolic.laplacian(u**3 - 3*u*v**2)
((sqrt(u**2 + v**2), sqrt(u**2 + v**2)), 0)

```

## Simplification

Whether a tensor vanishes depends on recognising zero. Every component
is brought to the normal form of
`sympy_extras.simplify.normalform.normal_form`: a quotient of
polynomials in the kernels, in lowest terms, reduced modulo
$\sin^2 + \cos^2 = 1$ (and $\cosh^2 - \sinh^2 = 1$), the relations
$\rho^q = b$ of the fractional powers, the powers $e^{r w} = (e^{g
w})^{r/g}$ of exponentials and $i^2 = -1$. These relations
have coprime leading monomials, so they are a Gröbner basis and the
remainder is canonical: a component is returned as `0` exactly when it
vanishes by them. The reduction is applied after every operation, which
keeps the denominators in one form, so that sums are taken over least
common denominators: the Kerr metric in Boyer–Lindquist coordinates is
found Ricci-flat in about 3 s, and its Kretschmann scalar in about 10 s.

```python
>>> from sympy_extras.simplify.normalform import normal_form
>>> x = Symbol('x')
>>> normal_form(sin(x)**4 - cos(x)**4 + cos(x)**2 - sin(x)**2), normal_form(1 + sin(x)**2/cos(x)**2)
(0, cos(x)**(-2))

```

Another simplifier can be passed (`Metric(g, coordinates, simplifier)`),
for instance `sympy_extras.simplify.simplify.simplify`.

## Not done

* tensors as objects with index positions (only the components of the
  listed tensors), tetrads and the Newman–Penrose formalism;
* relations between kernels beyond the ones above (`sin(2*x)` against
  `sin(x)`, logarithms): such metrics are
  computed, but a vanishing component may not be reduced to zero;
* solving the Killing equations or the geodesic equations (they are
  returned as expressions).

## References

* C. W. Misner, K. S. Thorne, J. A. Wheeler, *Gravitation*, Freeman, 1973.
* S. M. Carroll, *Spacetime and Geometry*, Addison-Wesley, 2004.
* D. Cox, J. Little, D. O'Shea, *Ideals, Varieties, and Algorithms*, 4th
  ed., Springer, 2015, Section 2.9.
