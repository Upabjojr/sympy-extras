"""Triangular decompositions against published results.

The decomposition of the first system is the one shown in the
documentation of ``Triangularize`` of Maple's RegularChains library and in
[ChenMorenoMaza]_; the numbers of solutions of the cyclic and Katsura
systems and the discriminants are classical.

.. [ChenMorenoMaza] C. Chen, M. Moreno Maza, Algorithms for computing
   triangular decomposition of polynomial systems, Journal of Symbolic
   Computation 47 (2012), 610-642.
"""
from __future__ import annotations

from sympy import discriminant, expand, symbols

from sympy_extras.polys.ideals import Ideal
from sympy_extras.polys.regularchains import RegularChain, triangularize

from .oracle import same_radical

x, y, z, w, a, b, c, p, q = symbols('x y z w a b c p q')


def test_the_example_of_the_regular_chains_library() -> None:
    chains = triangularize([x**2 + y + z - 1, x + y**2 + z - 1, x + y + z**2 - 1], x, y, z)
    assert set(chains) == {RegularChain([x - z, y - z, z**2 + 2*z - 1], x, y, z),
        RegularChain([x, y, z - 1], x, y, z), RegularChain([x, y - 1, z], x, y, z),
        RegularChain([x - 1, y, z], x, y, z)}


def test_cyclic_roots() -> None:
    # the six cyclic 3-roots
    chains = triangularize([x + y + z, x*y + y*z + z*x, x*y*z - 1], x, y, z)
    assert sum(chain.degree for chain in chains) == 6 and all(chain.dimension == 0 for chain in chains)
    # the cyclic 4-roots are two curves
    cyclic4 = [x + y + z + w, x*y + y*z + z*w + w*x, x*y*z + y*z*w + z*w*x + w*x*y, x*y*z*w - 1]
    chains = triangularize(cyclic4, x, y, z, w)
    assert [chain.dimension for chain in chains] == [1, 1]
    assert {str(chain.polys) for chain in chains} == {'[w*z - 1, w + y, x*y - 1]', '[w*z + 1, w + y, x*y + 1]'}
    assert same_radical(chains, cyclic4, [x, y, z, w])
    assert triangularize(cyclic4, x, y, z, w, mode='kalkbrener') == chains


def test_katsura() -> None:
    # 2**n solutions, all simple
    katsura2 = [x + 2*y + 2*z - 1, x**2 + 2*y**2 + 2*z**2 - x, 2*x*y + 2*y*z - y]
    chains = triangularize(katsura2, x, y, z)
    assert sum(chain.degree for chain in chains) == 4
    assert Ideal(katsura2, x, y, z).vector_space_dimension() == 4
    # SymPy's factorization of the resultants of this system did not finish
    # (Wang's algorithm looks for a prime above an astronomical bound, issue
    # #25): large polynomials are only divided by the factors already known
    katsura3 = [x + 2*y + 2*z + 2*w - 1, x**2 + 2*y**2 + 2*z**2 + 2*w**2 - x,
        2*x*y + 2*y*z + 2*z*w - y, y**2 + 2*x*z + 2*y*w - z]
    chains = triangularize(katsura3, x, y, z, w)
    assert sum(chain.degree for chain in chains) == 8 and all(chain.dimension == 0 for chain in chains)
    assert all(chain.contains(f) for chain in chains for f in katsura3)


def test_discriminants() -> None:
    # the double roots of a polynomial lie over its discriminant, and are
    # rational functions of the coefficients
    cubic = x**3 + p*x + q
    [chain] = triangularize([cubic, cubic.diff(x)], x, p, q, mode='kalkbrener')
    assert chain.polys == [4*p**3 + 27*q**2, 2*p*x + 3*q]
    assert expand(chain.polys[0] - discriminant(cubic, x)*(-1)) == 0
    quartic = x**4 + a*x**2 + b*x + c
    [chain] = triangularize([quartic, quartic.diff(x)], x, a, b, c, mode='kalkbrener')
    assert expand(chain.polys[0] - discriminant(quartic, x)) == 0
    # the double root of the depressed quartic
    assert chain.polys[1] == expand((2*a**3 - 8*a*c + 9*b**2)*x + a**2*b + 12*b*c)


def test_lagrange_multipliers() -> None:
    # the extrema of x*y on the ellipse x**2 + 4*y**2 = 4: four points with x*y = +-1
    system = [y - 2*w*x, x - 8*w*y, x**2 + 4*y**2 - 4]
    chains = triangularize(system, w, x, y)
    assert sum(chain.degree for chain in chains) == 4
    assert all(chain.contains(x**2 - 2) and chain.contains(2*y**2 - 1) and chain.contains(16*w**2 - 1)
        for chain in chains)


def test_the_singular_points_of_a_surface() -> None:
    # the Whitney umbrella x**2 = y**2*z is singular along the z axis
    f = x**2 - y**2*z
    chains = triangularize([f, f.diff(x), f.diff(y), f.diff(z)], x, y, z)
    assert chains == [RegularChain([x, y], x, y, z)]
