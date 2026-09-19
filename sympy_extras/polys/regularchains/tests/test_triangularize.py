"""Tests of the triangular decompositions against Gröbner bases."""
from __future__ import annotations

import random
from typing import Sequence

from sympy import Eq, Integer, Rational, expand, sqrt, symbols
from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.testing.pytest import raises

from sympy_extras._testing import untyped
from sympy_extras.polys.ideals import Ideal
from sympy_extras.polys.regularchains import RegularChain, triangularize

from .oracle import covers, is_squarefree_regular_chain, lies_in, same_radical

x, y, z, w, a, b, c = symbols('x y z w a b c')


def _verified(equations: Sequence[Expr], variables: Sequence[Symbol]) -> list[RegularChain]:
    """The decomposition in the sense of Lazard, checked: squarefree regular
    chains, in the zeros of the system, which cover them, and whose
    saturated ideals intersect in the radical; the one in the sense of
    Kalkbrener has the same closure."""
    chains = triangularize(equations, *variables)
    assert all(is_squarefree_regular_chain(chain) for chain in chains)
    assert lies_in(chains, equations)
    assert covers(chains, equations, variables)
    assert same_radical(chains, equations, variables)
    generic = triangularize(equations, *variables, mode='kalkbrener')
    assert all(is_squarefree_regular_chain(chain) for chain in generic)
    assert same_radical(generic, equations, variables)
    assert all(chain.height <= len(equations) for chain in generic)
    return chains


def test_zero_dimensional_systems() -> None:
    chains = _verified([x**2 + y**2 - 1, x*y - 1], [x, y])
    assert [chain.polys for chain in chains] == [[y**4 - y**2 + 1, x*y - 1]]
    chains = _verified([x*y - 1, x**2 - y], [x, y])
    assert sum(chain.degree for chain in chains) == 3
    # the three circles of radius one centred at the vertices of a triangle
    assert _verified([x**2 + y**2 - 1, (x - 1)**2 + y**2 - 1, (2*x - 1)**2 + (2*y - 1)**2 - 4], [x, y]) == []
    chains = _verified([x**2 + y**2 + z**2 - 4, x*y*z - 1, x + y + z], [x, y, z])
    assert sum(chain.degree for chain in chains) == Ideal([x**2 + y**2 + z**2 - 4, x*y*z - 1, x + y + z],
        x, y, z).radical().vector_space_dimension() == 6
    # a double solution is counted once: the chains are squarefree
    chains = _verified([(x - y)**2, y**2*(y - 1)], [x, y])
    assert [chain.polys for chain in chains] == [[y, x], [y - 1, x - 1]]


def test_positive_dimension() -> None:
    chains = _verified([x*y, x*z, y*z], [x, y, z])
    assert sorted(chain.dimension for chain in chains) == [1, 1, 1]
    chains = _verified([x*y, x*z], [x, y, z])
    assert [chain.polys for chain in chains] == [[x], [z, y]]
    assert [chain.dimension for chain in chains] == [2, 1]
    assert chains[0].free_variables == [y, z] and chains[1].free_variables == [x]
    # the twisted cubic: the curve, and its point where the initial vanishes
    chains = _verified([y - x**2, z - x**3, x*z - y**2], [x, y, z])
    assert [chain.dimension for chain in chains] == [1, 0]
    assert chains[0].initials == [1, z] and chains[1].polys == [z, y, x]
    chains = _verified([x*(y**2 - z), (x - 1)*(y - z**2), x*y*z - z], [x, y, z])
    assert [chain.dimension for chain in chains] == [1, 0, 0]
    _verified([(x - 1)*(y - 2), (x - 1)*(z - 3), x*y - z], [x, y, z])
    _verified([x**2 - y**2, x*z - y*z + x - y], [x, y, z])
    # a hypersurface: its squarefree factors, and the line where the initial
    # of x*y - z vanishes
    chains = _verified([(x*y - z)**2*(y + 1)], [x, y, z])
    assert [chain.polys for chain in chains] == [[x*y - z], [y + 1], [z, y]]


def test_the_order_of_the_variables() -> None:
    equations = [x**2 + y + z - 1, x + y**2 + z - 1, x + y + z**2 - 1]
    for variables in ([x, y, z], [z, y, x], [y, x, z]):
        chains = _verified(equations, variables)
        assert sum(chain.degree for chain in chains) == 5
        assert all(chain.symbols == tuple(variables) and chain.dimension == 0 for chain in chains)
    # all the symbols in their sorted order when none is given
    assert triangularize(equations) == triangularize(equations, x, y, z)


def test_parameters_are_discussed() -> None:
    chains = _verified([a*x**2 + b*x + c], [x, a, b, c])
    assert [chain.polys for chain in chains] == [[a*x**2 + b*x + c], [a, b*x + c], [c, b, a]]
    # the symbols which are not given are the smallest variables
    assert triangularize([a*x**2 + b*x + c], x) == chains
    assert chains[0].symbols == (x, a, b, c)
    # a linear system with a parameter: the rank drops at a = 1 and a = -2
    system = [a*x + y + z - 1, x + a*y + z - 1, x + y + a*z - 1]
    chains = _verified(system, [x, y, z, a])
    assert [chain.polys for chain in chains] == [[a - 1, x + y + z - 1],
        [a*z + 2*z - 1, a*y + 2*y - 1, a*x + 2*x - 1]]
    assert [chain.dimension for chain in chains] == [2, 1]
    # no solution at a = -2
    assert not any(chain.contains(a + 2) for chain in chains)


def test_kalkbrener_keeps_the_generic_points() -> None:
    f = x**3 + a*x + b
    generic = triangularize([f, f.diff(x)], x, a, b, mode='kalkbrener')
    # the discriminant of the cubic, and the double root
    assert [chain.polys for chain in generic] == [[4*a**3 + 27*b**2, 2*a*x + 3*b]]
    assert generic[0].main_variables == [a, x] and generic[0].dimension == 1
    # in the sense of Lazard the point where the initial a vanishes is there too
    assert [chain.polys for chain in triangularize([f, f.diff(x)], x, a, b)] == \
        [[4*a**3 + 27*b**2, 2*a*x + 3*b], [b, a, x]]
    # with four equations in three unknowns nothing is discarded
    assert triangularize([x*y, x*z, y*z, x + y + z], x, y, z, mode='kalkbrener') == \
        [RegularChain([z, y, x], x, y, z)]


def test_inequations() -> None:
    system = [x**2 - y**2, x*(y - 1)]
    chains = triangularize(system, x, y, inequations=[x])
    assert sorted(str(chain.polys) for chain in chains) == ['[y - 1, x + 1]', '[y - 1, x - 1]']
    assert covers(chains, system, [x, y], [x])
    assert all(chain.is_regular(x) for chain in chains)
    # an inequation which holds nowhere
    assert triangularize(system, x, y, inequations=[x, y - 1]) == []
    assert triangularize(system, x, y, inequations=[Integer(0)]) == []
    # a component is removed, another one is kept although the inequation
    # vanishes at one of its points
    system = [x*y, x*z]
    chains = triangularize(system, x, y, z, inequations=[x])
    assert [chain.polys for chain in chains] == [[z, y]]
    chains = triangularize(system, x, y, z, inequations=[y])
    assert [chain.polys for chain in chains] == [[x]] and covers(chains, system, [x, y, z], [y])


def test_input_forms() -> None:
    expected = triangularize([x**2 - 2*y, 3*y - 1], x, y)
    assert triangularize([Eq(x**2, 2*y), Eq(y, Rational(1, 3))], x, y) == expected
    assert triangularize([x**2/2 - y, y - Rational(1, 3)], x, y) == expected
    assert triangularize([], x, y) == [RegularChain([], x, y)]
    assert triangularize([Integer(0), x - 1], x) == [RegularChain([x - 1], x)]
    assert triangularize([Integer(1)], x) == []
    assert triangularize([x, x - 1], x) == []
    raises(ValueError, lambda: triangularize([x**2 - y], x, y, mode='wu'))
    raises(ValueError, lambda: triangularize([x**2 - sqrt(2)], x))
    raises(ValueError, lambda: triangularize([1/x - y], x, y))
    raises(ValueError, lambda: triangularize([x - 1], x, x))
    raises(ValueError, lambda: triangularize([Integer(1)]))
    raises(TypeError, lambda: untyped(triangularize)([x - 1], 3))


def _monomial(generator: random.Random, variables: Sequence[Symbol], total: int) -> Expr:
    result: Expr = Integer(1)
    for _ in range(generator.randint(0, total)):
        result = result*generator.choice(list(variables))
    return result


def _polynomial(generator: random.Random, variables: Sequence[Symbol], terms: int, total: int) -> Expr:
    result: Expr = Integer(0)
    for _ in range(terms):
        result = result + generator.randint(-3, 3)*_monomial(generator, variables, total)
    return result


def test_random_systems() -> None:
    generator = random.Random(2)
    done = 0
    while done < 14:
        variables = [x, y, z] if generator.random() < 0.6 else [x, y]
        equations: list[Expr] = []
        for _ in range(generator.randint(1, len(variables))):
            if generator.random() < 0.4:
                # products give components of different dimensions
                p = expand(_polynomial(generator, variables, 2, 1)*_polynomial(generator, variables, 3, 2))
            else:
                p = expand(_polynomial(generator, variables, generator.randint(2, 4), generator.randint(1, 3)))
            if p.free_symbols:
                equations.append(p)
        if equations:
            _verified(equations, variables)
            done += 1


def test_isolated_solutions_are_in_one_chain_only() -> None:
    # the lines x = y and x = -y meet at the origin, which is a solution:
    # it is in the chains of both factors, and removed from the second one,
    # so that the degrees of the chains add up to the number of solutions
    for system in ([x**2 - y**2, x**2 - x], [(x - y)*(x + y - 2), (x - 1)*(y - 1)*(x + y), z**2 - x*y]):
        variables = [x, y, z] if z in set().union(*[f.free_symbols for f in system]) else [x, y]
        chains = _verified(system, variables)
        assert all(chain.dimension == 0 for chain in chains)
        assert sum(chain.degree for chain in chains) == Ideal(system, *variables).radical().vector_space_dimension()
