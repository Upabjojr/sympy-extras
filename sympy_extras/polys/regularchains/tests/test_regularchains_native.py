"""Tests of the native regular chains (on objects of the core), checked
against the native Gröbner bases of :mod:`sympy_extras.polys.ideals` and
by substitution of the numerical solutions."""
from __future__ import annotations

import random

from sympy.testing.pytest import raises

from sympy_extras._testing import untyped
from sympy_extras._typing_sympy import as_expr, as_symbol
from sympy_extras.core import Eq, Integer, Rational, Symbol, expand
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr
from sympy_extras.core.mul import mul
from sympy_extras.core.numbers import Float
from sympy_extras.polys.domains import QQ
from sympy_extras.polys.ideals import Ideal
from sympy_extras.polys.regularchains import RegularChain, regular_gcd, triangularize

x, y, z, t, a, b, c = [Symbol(name) for name in 'xyztabc']


def _check(equations: list[Expr], variables: list[Symbol]) -> list[RegularChain]:
    """The decomposition in the sense of Lazard: every equation in every
    saturated ideal, and (in dimension zero) the intersection of the
    saturated ideals is the radical of the ideal of the system (by
    Seidenberg's lemma, natively)."""
    chains = triangularize(equations, *variables)
    for chain in chains:
        assert all(chain.contains(e) for e in equations)
        ideal = chain.saturated_ideal()
        assert all(ideal.contains(e) for e in equations)
    system = Ideal(equations, *variables, domain=QQ)
    if system.is_zero_dimensional():
        assert sum(chain.degree for chain in chains) == system.radical().vector_space_dimension()
        intersection = chains[0].saturated_ideal()
        for chain in chains[1:]:
            intersection = intersection.intersect(chain.saturated_ideal())
        assert intersection == system.radical()
    return chains


def test_triangular_decompositions() -> None:
    chains = _check([x**2 + y + z - 1, x + y**2 + z - 1, x + y + z**2 - 1], [x, y, z])
    assert [str(chain.polys) for chain in chains] == [
        '[z**2 + 2*z - 1, y - z, x - z]', '[z, y, x - 1]', '[z, y - 1, x]', '[z - 1, y, x]']
    chains = _check([x**2 + y**2 - 1, x*y - 1], [x, y])
    assert [str(chain.polys) for chain in chains] == ['[y**4 - y**2 + 1, x*y - 1]']
    # a double solution counts once: the chains are squarefree
    chains = _check([(x - y)**2, y**2*(y - 1)], [x, y])
    assert [str(chain.polys) for chain in chains] == ['[y, x]', '[y - 1, x - 1]']
    # no solution: three circles without a common point
    assert _check([x**2 + y**2 - 1, (x - 1)**2 + y**2 - 1, (2*x - 1)**2 + (2*y - 1)**2 - 4], [x, y]) == []
    # a plane and a line, which a lexicographic Gröbner basis does not separate
    chains = _check([x*y, x*z], [x, y, z])
    assert [str(chain.polys) for chain in chains] == ['[x]', '[z, y]'] and [c.dimension for c in chains] == [2, 1]
    assert chains[0].free_variables == [y, z] and chains[1].free_variables == [x]


def test_parameters_and_modes() -> None:
    chains = triangularize([a*x**2 + b*x + c], x, a, b, c)
    assert [(str(chain.polys), str(chain.initials)) for chain in chains] == [
        ('[a*x**2 + b*x + c]', '[a]'), ('[a, b*x + c]', '[1, b]'), ('[c, b, a]', '[1, 1, 1]')]
    p, q = Symbol('p'), Symbol('q')
    f = x**3 + p*x + q
    generic = triangularize([f, f.diff(x)], x, p, q, mode='kalkbrener')
    assert str(generic) == '[RegularChain([4*p**3 + 27*q**2, 2*p*x + 3*q], x, p, q)]'
    lazard = triangularize([f, f.diff(x)], x, p, q)
    assert len(lazard) == 2 and str(lazard[1].polys) == '[q, p, x]'
    # the parameters are the other symbols, after the variables
    [chain] = triangularize([t*x - 1], x)
    assert chain.symbols == (x, t) and chain.initials == [t]
    raises(ValueError, lambda: triangularize([x], x, mode='other'))


def test_inequations_and_equalities() -> None:
    chains = triangularize([x**2 - y**2, x*(y - 1)], x, y, inequations=[x])
    assert [str(chain.polys) for chain in chains] == ['[y - 1, x + 1]', '[y - 1, x - 1]']
    assert triangularize([x*y - 1], x, y, inequations=[Integer(0)]) == []
    assert triangularize([Eq(x**2, y), Eq(y, 4)], x, y) == triangularize([x**2 - y, y - 4], x, y)
    # rational coefficients are cleared
    [chain] = triangularize([x**2/2 - Rational(1, 3)*y, y - 1], x, y)
    assert str(chain.polys) == '[y - 1, 3*x**2 - 2]'


def test_chains_and_their_operations() -> None:
    T = RegularChain([z**2 - 2, y**2 - z, (y + 1)*x - 1], x, y, z)
    assert T.main_variables == [z, y, x] and T.dimension == 0 and T.degree == 4
    assert T.contains(y**4 - 2) and T.contains(x*z - x - y + 1) and not T.contains(x - 1)
    assert str(T.reduce(y**3 + x)) == 'y*z + 3' and T.is_regular(y - z)
    S = RegularChain([y**2 - 1, x**2 - y], x, y)
    zero, regular = S.regularize(y - 1)
    assert [str(chain.polys) for chain in zero] == ['[y - 1, x**2 - y]']
    assert [str(chain.polys) for chain in regular] == ['[y + 1, x**2 - y]']
    assert str(RegularChain([x**2 + y**2 - 1], x, y).intersect(x - y)) == '[RegularChain([2*y**2 - 1, x - y], x, y)]'
    U = RegularChain([y**2 - 2, x - y], x, y).with_symbols(x, t, y)
    assert U.free_variables == [t] and U.contains(t*x**2 - 2*t)
    found = regular_gcd(x**2 - y, x**2 - 3*x + 2*y, x, RegularChain([y**2 - 1], x, y))
    assert [(str(g), str(chain.polys)) for g, chain in found] == [('1', '[y + 1]'), ('x - 1', '[y - 1]')]
    assert str(RegularChain([z*y - 1, y*x - z], x, y, z).saturated_ideal().reduced().exprs) == \
        '[x*y - z, y*z - 1, -x + z**2]'


def test_what_is_not_a_regular_chain() -> None:
    raises(ValueError, lambda: RegularChain([x**2 - y, x*y - 1], x, y))
    raises(ValueError, lambda: RegularChain([Integer(3)], x))
    raises(ValueError, lambda: RegularChain([y**2 - z*y, (y - z)*x - 1], x, y, z))
    raises(ValueError, lambda: RegularChain([(x - y)**2], x, y))
    raises(ValueError, lambda: RegularChain([x - 1], x, x))
    raises(ValueError, lambda: RegularChain([x - 1/y], x, y))
    raises(ValueError, lambda: RegularChain([x - mul(Integer(2)**Rational(1, 2), y)], x, y))
    raises(ValueError, lambda: RegularChain([x - y], x, y).contains(z))
    raises(TypeError, lambda: untyped(triangularize)([x - 1], 3))
    raises(ValueError, lambda: regular_gcd(x**2 - y, y - 1, x, RegularChain([y**2 - 1], x, y)))


def test_numerical_solutions() -> None:
    [chain] = triangularize([x**2 + y**2 - 5, x*y - 1], x, y)
    assert [str(s) for s in chain.numerical_solutions(5)] == [
        '{y: -2.1889, x: -0.45685}', '{y: -0.45685, x: -2.1889}', '{y: 0.45685, x: 2.1889}',
        '{y: 2.1889, x: 0.45685}']
    system = [x**3 + y*z - 2, y**3 + x*z - 3, z**2 + x*y - 1]
    solutions = [s for chain in triangularize(system, x, y, z) for s in chain.numerical_solutions(12)]
    # Bezout's number, all the solutions being simple and at finite distance
    assert len(solutions) == 18
    for s in solutions:
        rule: dict[Basic, Basic] = {v: value for v, value in s.items()}
        for f in system:
            residual = f._xreplace_evaluated(rule)
            assert isinstance(residual, Expr) and abs(complex(residual)) < 1e-9
    # real solutions come out real, as Floats
    [circles] = triangularize([x**2 + y**2 - 1, (x - 1)**2 + y**2 - 1], x, y)
    points = circles.numerical_solutions(10)
    assert all(isinstance(s[y], Float) and isinstance(s[x], Float) for s in points)
    assert str(points[1][y]) == '0.8660254038'
    raises(ValueError, lambda: RegularChain([x*y - 1], x, y).numerical_solutions())


def test_exact_solutions_through_the_face() -> None:
    [chain] = triangularize([y**2 - 2, x*y - 1], x, y)
    assert str(chain.solutions()) == '[{y: -sqrt(2), x: -sqrt(2)/2}, {y: sqrt(2), x: sqrt(2)/2}]'
    [chain] = triangularize([x**2 + 1, y - x], y, x)
    assert chain.solutions(real=True) == [] and len(chain.solutions()) == 2
    raises(ValueError, lambda: RegularChain([x*y - 1], x, y).solutions())


def test_the_face_gives_the_native_results() -> None:
    # the face converts and calls the native module: the chains of random
    # systems are the same through both
    from sympy_extras.interop import from_sympy, to_sympy
    from sympy_extras.polys.regularchains_sympy import triangularize as sympy_triangularize
    generator = random.Random(2)
    for _ in range(6):
        monomials = [x*y, y*z, x, y, z, x**2, Integer(1)]
        system = [expand(sum((generator.randint(-3, 3)*m for m in monomials), Integer(0))) for _ in range(2)]
        native = triangularize(system, x, y, z)
        face = sympy_triangularize([as_expr(to_sympy(e)) for e in system], *[as_symbol(to_sympy(v)) for v in (x, y, z)])
        assert [[from_sympy(p) for p in chain.polys] for chain in face] == [chain.polys for chain in native]
