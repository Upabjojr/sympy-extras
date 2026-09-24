"""Primary decompositions known from the literature, as results: Cox,
Little, O'Shea, *Ideals, Varieties, and Algorithms*, chapter 4 (CLO);
Atiyah, Macdonald, *Introduction to Commutative Algebra*, chapter 4 (AM);
ideals built as intersections of primary ideals known to be primary (ideals
of variables and powers of them, principal ideals); and the classical
monomial curve `(t**3, t**4, t**5)`, whose prime
has a square with an embedded component at the origin (the symbolic
square contains the quintic found by Hochster). Every result is also
checked by :func:`primary_oracle.check_decomposition`, which does not
trust the algorithm."""
from __future__ import annotations

from sympy.abc import t, w, x, y, z

from sympy_extras.polys.ideals import Ideal
from sympy_extras.polys.idealdecomposition import primary_decomposition

from .primary_oracle import check_decomposition, intersection


def _decomposed(ideal: Ideal, expected: list[tuple[Ideal, Ideal]], embedded: list[Ideal]) -> None:
    """The components of the minimal primes are unique: they are compared;
    the embedded primes are compared, their components (not unique) only
    checked."""
    found = primary_decomposition(ideal)
    check_decomposition(ideal, found)
    primes = [p for _, p in found]
    assert len(primes) == len(expected) + len(embedded)
    for q, p in expected:
        assert any(q == q2 and p == p2 for q2, p2 in found)
    for p in embedded:
        assert any(p == p2 for p2 in primes)


def test_embedded_points_and_lines() -> None:
    X, XY, M2 = Ideal([x], x, y), Ideal([x, y], x, y), Ideal([x**2, y], x, y)
    # CLO 4: (x**2, x*y) = (x) ∩ (x**2, y), the y axis with an embedded origin
    _decomposed(Ideal([x**2, x*y], x, y), [(X, X)], [XY])
    assert Ideal([x**2, x*y], x, y) == X.intersect(M2)
    # CLO 4: (x*y, x*z, y*z), the three axes, no embedded component
    axes = [Ideal([a, b], x, y, z) for a, b in ((x, y), (x, z), (y, z))]
    _decomposed(Ideal([x*y, x*z, y*z], x, y, z), [(a, a) for a in axes], [])
    # (x**2*y, x*y**2) = (x) ∩ (y) ∩ (x, y)**3, the embedded part not unique
    Y = Ideal([y], x, y)
    _decomposed(Ideal([x**2*y, x*y**2], x, y), [(X, X), (Y, Y)], [XY])
    # (x**3, x**2*y) = (x**2) ∩ (x**3, y): a double line, an embedded point
    _decomposed(Ideal([x**3, x**2*y], x, y), [(Ideal([x**2], x, y), X)], [XY])
    # the plane x = 0 with an embedded point, and with an embedded line
    P, M = Ideal([x], x, y, z), Ideal([x, y, z], x, y, z)
    _decomposed(Ideal([x**2, x*y, x*z], x, y, z), [(P, P)], [M])
    _decomposed(Ideal([x**2, x*y], x, y, z), [(P, P)], [Ideal([x, y], x, y, z)])
    Z = Ideal([z], x, y, z)
    _decomposed(Ideal([x*z, y*z, z**2], x, y, z), [(Z, Z)], [M])


def test_atiyah_macdonald_square_of_a_prime() -> None:
    # AM 4: p = (x, z) in k[x, y, z]/(x*y - z**2) has a square which is not
    # primary: (x**2, x*y, x*z, z**2) = (x, z**2) ∩ (x, y, z)**2
    I = Ideal([x**2, x*y, x*z, z**2], x, y, z)
    M = Ideal([x, y, z], x, y, z)
    _decomposed(I, [(Ideal([x, z**2], x, y, z), Ideal([x, z], x, y, z))], [M])
    assert I == Ideal([x, z**2], x, y, z).intersect(M**2)


def test_irrational_points_and_curves() -> None:
    # the embedded prime (x**2 - 2, y) has no rational point
    f = x**2 - 2
    _decomposed(Ideal([f**2, f*y], x, y), [(Ideal([f], x, y), Ideal([f], x, y))], [Ideal([f, y], x, y)])
    # a parabola with an embedded point: (y - x**2)*(x, y)
    parabola = Ideal([y - x**2], x, y)
    _decomposed(Ideal([(y - x**2)*x, (y - x**2)*y], x, y), [(parabola, parabola)], [Ideal([x, y], x, y)])
    # dimension zero: a double conjugate pair and a rational point
    pair, point = Ideal([f, y], x, y), Ideal([x - 1, y - 1], x, y)
    I = (pair**2).intersect(point)
    _decomposed(I, [(pair**2, pair), (point, point)], [])


def test_monomial_curve_and_twisted_cubic() -> None:
    # the curve (t**3, t**4, t**5): its prime P has a square with an
    # embedded component at the origin, and the P-primary component (the
    # symbolic square) contains Hochster's quintic, which P**2 does not
    P = Ideal([x - t**3, y - t**4, z - t**5], t, x, y, z).eliminate([t])
    quintic = x**5 - 3*x**2*y*z + x*y**3 + z**3
    found = primary_decomposition(P**2)
    check_decomposition(P**2, found)
    assert [p for _, p in found] == [P, Ideal([x, y, z], x, y, z)]
    assert found[0][0].contains(quintic) and not (P**2).contains(quintic)
    # the square of the twisted cubic in 4 variables is primary: a
    # saturation by a variable does not change it
    C = Ideal([x*z - y**2, y*w - z**2, x*w - y*z], x, y, z, w)
    C2 = C**2
    assert C2.saturate(x) == C2 and C2.is_primary()
    found = primary_decomposition(C2)
    assert len(found) == 1 and found[0][0] == C2 and found[0][1] == C


def test_redundant_embedded_component() -> None:
    # (x, y, z)**3 ∩ (x - y - z)**2 ∩ (x - y, x - z)**2: the (x, y, z)-primary
    # part contains the intersection of the others, so that there are two
    # components
    M = Ideal([x, y, z], x, y, z)
    plane, line = Ideal([x - y - z], x, y, z), Ideal([x - y, x - z], x, y, z)
    I = intersection([M**3, plane**2, line**2])
    assert I == (plane**2).intersect(line**2)
    _decomposed(I, [(plane**2, plane), (line**2, line)], [])


def test_chains_of_embedded_primes() -> None:
    # a plane, a line in it and a point on the line, all three associated
    V = (x, y, z)
    X, XY, M = Ideal([x], *V), Ideal([x, y], *V), Ideal([x, y, z], *V)
    I = intersection([X, Ideal([x**2, y], *V), Ideal([x**3, y**3, z], *V)])
    _decomposed(I, [(X, X)], [XY, M])
    I = intersection([X, Ideal([x**2, y**2], *V), Ideal([x**3, y**3, z**2], *V)])
    _decomposed(I, [(X, X)], [XY, M])
    # the twisted cubic times the maximal ideal of the origin, and times a
    # line which meets it at (1, 1, 1, 1), where the product is not reduced
    V4 = (x, y, z, w)
    C = Ideal([x*z - y**2, y*w - z**2, x*w - y*z], *V4)
    _decomposed(C*Ideal(list(V4), *V4), [(C, C)], [Ideal(list(V4), *V4)])
    L = Ideal([x - w, y - z], *V4)
    _decomposed(C*L, [(C, C), (L, L)], [Ideal([x - w, y - w, z - w], *V4)])
