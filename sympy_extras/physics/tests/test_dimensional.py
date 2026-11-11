"""Dimensional analysis: the classical groups and laws, the count of the
Pi theorem against the rank computed by SymPy, and the kernel property
of every group."""
from __future__ import annotations

import random
from fractions import Fraction

import pytest
import sympy

from sympy_extras.core import Symbol
from sympy_extras.functions import sqrt
from sympy_extras.physics.dimensional import (SI, Dimension, dimension_matrix, dimension_of, is_dimensionless, pi_exponents, pi_groups,
                                              solve_for)


def _p(name: str) -> Symbol:
    return Symbol(name, positive=True)


def test_classical_groups() -> None:
    # the Reynolds number, and the Mach number for the speed of sound c
    assert pi_groups({'rho': SI['density'], 'v': SI['velocity'], 'L': SI['length'], 'mu': SI['dynamic viscosity']}) == [
        _p('L')*_p('rho')*_p('v')/_p('mu')]
    assert pi_groups({'v': SI['velocity'], 'c': SI['velocity']}) == [_p('v')/_p('c')]
    # Taylor's analysis of an explosion: E t**2/(rho R**5) is the only group
    blast = {'R': SI['length'], 't': SI['time'], 'E': SI['energy'], 'rho': SI['density']}
    assert pi_exponents(blast) == [[5, -2, -1, 1]]
    assert pi_groups({'m': SI['mass'], 'l': SI['length'], 't': SI['time']}) == []
    # the fine structure constant needs the permittivity: without it e, hbar, c have no group
    assert pi_groups({'e': SI['charge'], 'hbar': SI['action'], 'c': SI['velocity']}) == []


def test_laws() -> None:
    # the pendulum, Kepler's third law, the Planck length, the capillary length
    pendulum = {'T': SI['time'], 'l': SI['length'], 'g': SI['acceleration'], 'm': SI['mass']}
    assert solve_for('T', pendulum) == sqrt(_p('l'))/sqrt(_p('g'))
    kepler = {'T': SI['time'], 'a': SI['length'], 'G': SI['gravitational constant'], 'M': SI['mass']}
    assert solve_for('T', kepler) == _p('a')**Fraction(3, 2)/(sqrt(_p('G'))*sqrt(_p('M')))
    planck = {'l': SI['length'], 'G': SI['gravitational constant'], 'hbar': SI['action'], 'c': SI['velocity']}
    assert solve_for('l', planck) == sqrt(_p('G'))*sqrt(_p('hbar'))/_p('c')**Fraction(3, 2)
    assert solve_for('E', {'E': SI['energy'], 'm': SI['mass'], 'c': SI['velocity']}) == _p('c')**2*_p('m')
    # Taylor: the radius of the blast wave R = C (E t**2/rho)**(1/5)
    blast = {'R': SI['length'], 't': SI['time'], 'E': SI['energy'], 'rho': SI['density']}
    assert solve_for('R', blast) == _p('E')**Fraction(1, 5)*_p('t')**Fraction(2, 5)/_p('rho')**Fraction(1, 5)
    # not determined: the drag depends on a function of the Reynolds number; impossible: a mass from lengths
    with pytest.raises(ValueError):
        solve_for('F', {'F': SI['force'], 'rho': SI['density'], 'v': SI['velocity'], 'L': SI['length'],
                        'mu': SI['dynamic viscosity']})
    with pytest.raises(ValueError):
        solve_for('m', {'m': SI['mass'], 'l': SI['length'], 'h': SI['length']})
    with pytest.raises(ValueError):
        solve_for('x', {'m': SI['mass']})


def test_pi_theorem_count_and_kernel() -> None:
    rnd = random.Random(161)
    names = list(SI)
    for _ in range(40):
        chosen = rnd.sample(names, rnd.randint(1, 7))
        quantities = {('q%d' % i): SI[name] for i, name in enumerate(chosen)}
        bases, rows = dimension_matrix(quantities)
        rank = sympy.Matrix(rows).rank() if rows else 0
        vectors = pi_exponents(quantities)
        assert len(vectors) == len(quantities) - rank
        for vector in vectors:
            assert is_dimensionless(dict(zip(quantities, vector)), quantities)
            assert next(e for e in vector if e) > 0
        if vectors:
            # independent, and generating all the integer solutions: the gcd of the maximal minors is one
            M = sympy.Matrix(vectors)
            assert M.rank() == len(vectors)
            minors = [abs(M.extract(list(range(M.rows)), list(columns)).det())
                      for columns in sympy.utilities.iterables.subsets(range(M.cols), M.rows)]
            assert sympy.gcd(minors) == 1
        assert len(pi_groups(quantities)) == len(vectors)


def test_fractional_dimensions_and_helpers() -> None:
    # the Gaussian charge has the dimension M**(1/2) L**(3/2) T**-1
    charge: Dimension = {'M': Fraction(1, 2), 'L': Fraction(3, 2), 'T': -1}
    gaussian: dict[str, Dimension] = {'q': charge, 'F': SI['force'], 'r': SI['length']}
    assert pi_exponents(gaussian) == [[2, -1, -2]]                       # Coulomb's law: F r**2/q**2
    assert dimension_of({'F': 1, 'r': 1}, gaussian) == SI['energy']
    assert not is_dimensionless({'F': 1}, gaussian)
    with pytest.raises(ValueError):
        dimension_of({'z': 1}, gaussian)
    assert dimension_matrix({}) == ([], [])
