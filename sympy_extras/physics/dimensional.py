r"""Dimensional analysis: the dimensionless groups of the Buckingham Pi
theorem, and the form a physical law must take.

Each quantity $q_j$ of a problem has a dimension $\prod_i B_i^{d_{ij}}$
in base dimensions $B_i$ (mass, length, time, ...). A product
$\prod_j q_j^{e_j}$ is dimensionless exactly when $D e = 0$ for the
dimension matrix $D = (d_{ij})$. By the Pi theorem [1]_ a relation
between $n$ quantities whose dimension matrix has the rank $r$ is a
relation between $n - r$ independent dimensionless products.

The exponent vectors are the integer points of the kernel of $D$. A
basis of that lattice is computed, generating *all* the integer
solutions, and reduced by the LLL algorithm
(:func:`sympy_extras.polys.lattice.integer_kernel`), so that the groups
have small exponents: the Reynolds number comes out as $\rho v L / \mu$
and not as a power or a product of groups [2]_ (for the drag on a body
the second group is $F/(\mu v L)$, the one of Stokes' law, rather than
the drag coefficient $F/(\rho v^2 L^2)$, their quotient being the
Reynolds number). The signs are chosen so
that the first quantity which occurs has a positive exponent.

:func:`solve_for` expresses one quantity through the others when the
dimensions leave a single possibility (up to a dimensionless constant):
the period of a pendulum, $T \propto \sqrt{l/g}$.

Dimensions are dictionaries from the names of the base dimensions to
rational exponents; :data:`SI` has those of the usual quantities.

Examples
========

>>> from sympy_extras.physics.dimensional import pi_groups, solve_for, SI
>>> pi_groups({'rho': SI['density'], 'v': SI['velocity'], 'L': SI['length'], 'mu': SI['dynamic viscosity']})
[L*rho*v/mu]
>>> pi_groups({'F': SI['force'], 'rho': SI['density'], 'v': SI['velocity'], 'L': SI['length'], 'mu': SI['dynamic viscosity']})
[F/(L*mu*v), L*rho*v/mu]
>>> solve_for('T', {'T': SI['time'], 'l': SI['length'], 'g': SI['acceleration'], 'm': SI['mass']})
sqrt(l)/sqrt(g)

References
==========

.. [1] E. Buckingham, *On physically similar systems; illustrations of
       the use of dimensional equations*, Physical Review 4 (1914),
       345-376.
.. [2] G. I. Barenblatt, *Scaling, Self-similarity, and Intermediate
       Asymptotics*, Cambridge University Press, 1996, Chapter 1.
"""
from __future__ import annotations

from fractions import Fraction
from math import gcd
from typing import Final, Mapping, Sequence, Union

from sympy_extras.core.expr import Expr
from sympy_extras.core.mul import mul
from sympy_extras.core.numbers import Integer, number_from_fraction
from sympy_extras.core.power import pow
from sympy_extras.core.symbol import Symbol
from sympy_extras.polys.lattice import integer_kernel

__all__ = ['Dimension', 'SI', 'dimension_matrix', 'pi_groups', 'pi_exponents', 'solve_for', 'is_dimensionless',
           'dimension_of']

#: the exponents of the base dimensions
Dimension = Mapping[str, Union[int, Fraction]]


def _d(**exponents: int) -> dict[str, Union[int, Fraction]]:
    return dict(exponents)


#: the dimensions of the usual quantities in mass M, length L, time T, current I, temperature K
SI: Final[dict[str, dict[str, Union[int, Fraction]]]] = {
    'dimensionless': _d(), 'mass': _d(M=1), 'length': _d(L=1), 'time': _d(T=1), 'current': _d(I=1),
    'temperature': _d(K=1), 'area': _d(L=2), 'volume': _d(L=3), 'velocity': _d(L=1, T=-1),
    'acceleration': _d(L=1, T=-2), 'frequency': _d(T=-1), 'density': _d(M=1, L=-3), 'force': _d(M=1, L=1, T=-2),
    'pressure': _d(M=1, L=-1, T=-2), 'energy': _d(M=1, L=2, T=-2), 'power': _d(M=1, L=2, T=-3),
    'momentum': _d(M=1, L=1, T=-1), 'angular momentum': _d(M=1, L=2, T=-1), 'dynamic viscosity': _d(M=1, L=-1, T=-1),
    'kinematic viscosity': _d(L=2, T=-1), 'surface tension': _d(M=1, T=-2), 'charge': _d(I=1, T=1),
    'voltage': _d(M=1, L=2, T=-3, I=-1), 'resistance': _d(M=1, L=2, T=-3, I=-2),
    'capacitance': _d(M=-1, L=-2, T=4, I=2), 'inductance': _d(M=1, L=2, T=-2, I=-2),
    'thermal conductivity': _d(M=1, L=1, T=-3, K=-1), 'specific heat': _d(L=2, T=-2, K=-1),
    'gravitational constant': _d(M=-1, L=3, T=-2), 'action': _d(M=1, L=2, T=-1),
}


def dimension_matrix(quantities: Mapping[str, Dimension]) -> tuple[list[str], list[list[int]]]:
    """``(bases, D)``: the base dimensions which occur (sorted) and the
    integer matrix with one row for each of them and one column for each
    quantity (the rows are scaled when the exponents are fractions, which
    does not change the kernel).

    >>> from sympy_extras.physics.dimensional import dimension_matrix, SI
    >>> dimension_matrix({'v': SI['velocity'], 'g': SI['acceleration'], 'h': SI['length']})
    (['L', 'T'], [[1, 1, 1], [-1, -2, 0]])
    """
    bases = sorted({base for dimension in quantities.values() for base, exponent in dimension.items() if exponent})
    rows: list[list[int]] = []
    for base in bases:
        line = [Fraction(dimension.get(base, 0)) for dimension in quantities.values()]
        scale = 1
        for entry in line:
            scale = scale*entry.denominator//gcd(scale, entry.denominator)
        rows.append([int(entry*scale) for entry in line])
    return bases, rows


def pi_exponents(quantities: Mapping[str, Dimension]) -> list[list[int]]:
    """The exponent vectors (in the order of ``quantities``) of a basis
    of the dimensionless products: an LLL reduced basis of the lattice of
    all the integer solutions, the first nonzero exponent of each
    positive."""
    _, rows = dimension_matrix(quantities)
    basis = integer_kernel(rows, len(quantities))
    oriented = [[-e for e in vector] if next(e for e in vector if e) < 0 else list(vector) for vector in basis]
    return sorted(oriented, key=lambda v: (next(i for i, e in enumerate(v) if e), v))


def _product(names: Sequence[str], exponents: Sequence[Union[int, Fraction]]) -> Expr:
    factors: list[Expr] = []
    for name, exponent in zip(names, exponents):
        if exponent:
            factors.append(pow(Symbol(name, positive=True), number_from_fraction(Fraction(exponent))))
    return mul(*factors) if factors else Integer(1)


def pi_groups(quantities: Mapping[str, Dimension]) -> list[Expr]:
    """A complete set of independent dimensionless products of the
    quantities (positive symbols with their names), with small integer
    exponents; empty when there is none.

    >>> from sympy_extras.physics.dimensional import pi_groups, SI
    >>> pi_groups({'v': SI['velocity'], 'g': SI['acceleration'], 'L': SI['length']})             # the Froude number, squared
    [v**2/(L*g)]
    >>> pi_groups({'m': SI['mass'], 'l': SI['length']})
    []
    """
    names = list(quantities)
    return [_product(names, vector) for vector in pi_exponents(quantities)]


def dimension_of(exponents: Mapping[str, Union[int, Fraction]], quantities: Mapping[str, Dimension]) -> dict[str, Fraction]:
    """The dimension of the product of the quantities raised to the
    given exponents (the base dimensions with nonzero exponents).

    >>> from sympy_extras.physics.dimensional import dimension_of, SI
    >>> dimension_of({'m': 1, 'v': 2}, {'m': SI['mass'], 'v': SI['velocity']}) == {'M': 1, 'L': 2, 'T': -2}
    True
    """
    total: dict[str, Fraction] = {}
    for name, exponent in exponents.items():
        if name not in quantities:
            raise ValueError('no quantity %r' % (name,))
        for base, power in quantities[name].items():
            total[base] = total.get(base, Fraction(0)) + Fraction(power)*Fraction(exponent)
    return {base: power for base, power in total.items() if power}


def is_dimensionless(exponents: Mapping[str, Union[int, Fraction]], quantities: Mapping[str, Dimension]) -> bool:
    """Whether the product of the quantities with these exponents has
    no dimension."""
    return not dimension_of(exponents, quantities)


def solve_for(target: str, quantities: Mapping[str, Dimension]) -> Expr:
    """The product of powers of the other quantities with the dimension
    of ``target``, when the dimensions determine it: the law ``target =
    constant * product``. Quantities which cannot enter (the mass of a
    pendulum) get the exponent zero. ``ValueError`` when there is no such
    product, or when it is not unique (a dimensionless group of the
    other quantities exists: the law then involves an unknown function
    of it).

    >>> from sympy_extras.physics.dimensional import solve_for, SI
    >>> solve_for('E', {'E': SI['energy'], 'm': SI['mass'], 'c': SI['velocity']})
    c**2*m
    >>> solve_for('v', {'v': SI['velocity'], 'p': SI['pressure'], 'rho': SI['density']})          # the speed of sound
    sqrt(p)/sqrt(rho)
    """
    if target not in quantities:
        raise ValueError('no quantity %r' % (target,))
    others = {name: dimension for name, dimension in quantities.items() if name != target}
    if pi_exponents(others):
        raise ValueError('the other quantities have a dimensionless group: %s is not determined by the dimensions'
                         % (target,))
    ordered = dict(others)
    ordered[target] = quantities[target]
    vectors = pi_exponents(ordered)
    if not vectors:
        raise ValueError('no product of the other quantities has the dimension of %s' % (target,))
    [vector] = vectors
    own = vector[-1]
    names = list(others)
    return _product(names, [Fraction(-e, own) for e in vector[:-1]])
