"""The Rust port of :mod:`sympy_extras.solvers.lie` against the Python
module: the symmetries found (in order, as strings), the invariance
conditions, the determining systems and the dimensions of their Janet
bases."""
from __future__ import annotations

from typing import Callable

import pytest

from sympy_extras.core import Eq, Function, Integer, Rational, Symbol, Tuple
from sympy_extras.core.basic import Basic
from sympy_extras.functions import exp, sin
from sympy_extras.solvers.lie import (JetSpace, Symmetry, check_symmetry, determining_system,
    infinitesimal_condition, leading_derivatives, symmetries, symmetry_janet_basis)
from ..conftest import load

rc = load("sympy_extras_rs.core")

x, t, m = Symbol('x'), Symbol('t'), Symbol('m')
u = Function('u')(x, t)
v = Function('v')(x, t)
y = Function('y')(x)
heat = u.diff(t) - u.diff(x, 2)
burgers = u.diff(t) + u*u.diff(x) - u.diff(x, 2)
kdv = u.diff(t) + u*u.diff(x) + u.diff(x, 3)

CASES: list[tuple[list[Basic], list[Basic], int, list[Basic], bool]] = [
    ([heat], [u], 3, [], False),
    ([burgers], [u], 2, [], True),
    ([kdv], [u], 2, [], True),
    ([u.diff(t, 2) - u.diff(x, 2)], [u], 2, [], False),
    ([u.diff(t) - (u**m*u.diff(x)).diff(x)], [u], 2, [], True),
    ([y.diff(x, 2) + 2*y.diff(x)/x + y**5], [y], 2, [], True),
    ([y.diff(x, 2) + y**3], [y], 2, [], True),
    ([y.diff(x, 2)], [y], 2, [], True),
    ([y.diff(x) + 2*x*y - x*exp(-x**2)], [y], 2, [], True),
    ([y.diff(x) + 2*x*y - x*exp(-x**2)], [y], 2, [exp(-x**2)], True),
    ([Eq(u.diff(t), v.diff(x)), Eq(v.diff(t), u.diff(x))], [u, v], 1, [], True),
    ([y.diff(x, 2) + Rational(2)**Rational(1, 2)*y.diff(x)], [y], 2, [], True),
    ([y.diff(x, 2) - y.diff(x)**2/y], [y], 1, [], True),
    ([u.diff(t) - u.diff(x, 2) - u**2], [u], 2, [], True),
    # issue #72: equations not linear in the leading derivative, all their
    # branches (a denominator cleared, polynomials in exp(y'), in sqrt(y'))
    ([(y*y.diff(x) + x)*(y.diff(x) - 1)/y.diff(x)], [y], 2, [], True),
    ([(exp(y.diff(x)) - 1)*(exp(y.diff(x)) - y)], [y], 2, [], True),
    ([exp(2*y.diff(x)) - (1 + y)*exp(y.diff(x)) + y], [y], 2, [], True),
    ([(y.diff(x)**Rational(1, 2) - 1)*(y.diff(x)**Rational(1, 2) - y)], [y], 2, [], True),
    ([y.diff(x)**Rational(1, 2) + y], [y], 2, [], True),
    ([exp(y.diff(x)) - x], [y], 2, [], True),
    ([sin(y.diff(x)) - x], [y], 1, [], True),
]


def _outcome(f: Callable[[], object]) -> object:
    try:
        return f()
    except NotImplementedError:
        return 'SKIP'


def _python(eqs: list[Basic], funcs: list[Basic], degree: int, basis: list[Basic], sup: bool) -> object:
    found = symmetries(eqs, funcs, degree=degree, basis=basis, superposition=sup)
    return [(str(s.xi), str(s.eta), s.generator()) for s in found]


def _rust(eqs: list[Basic], funcs: list[Basic], degree: int, basis: list[Basic], sup: bool) -> object:
    r = rc.logic_call('lie', [Symbol('symmetries'), Tuple(*eqs), Tuple(*funcs), Integer(degree), Tuple(*basis),
                              Symbol(str(sup))])
    out = []
    for s in r.args:
        xi, eta = tuple(s.args[0].args), tuple(s.args[1].args)
        out.append((str(xi), str(eta), str(s.args[2])))
    return out


@pytest.mark.parametrize('case', range(len(CASES)))
def test_symmetries(case: int) -> None:
    eqs, funcs, degree, basis, sup = CASES[case]
    assert _outcome(lambda: _rust(eqs, funcs, degree, basis, sup)) == \
        _outcome(lambda: _python(eqs, funcs, degree, basis, sup))


@pytest.mark.parametrize('case', range(len(CASES)))
def test_check_symmetries(case: int) -> None:
    eqs, funcs, degree, basis, sup = CASES[case]
    found = _outcome(lambda: symmetries(eqs, funcs, degree=degree, basis=basis, superposition=sup))
    if found == 'SKIP':
        return
    assert isinstance(found, list)
    for s in found:
        r = rc.logic_call('lie', [Symbol('check'), Tuple(*eqs), Tuple(*funcs), Tuple(*s.xi, *s.eta)])
        assert bool(r) == check_symmetry(eqs, funcs, s) is True


def test_conditions_through_a_modulus() -> None:
    # the conditions of an equation in exp(y'), the generator written back
    jets = JetSpace([y])
    for eq, xi, eta in [(exp(y.diff(x)) - jets.u[0], [Integer(0)], [Integer(1)]),
                        (y.diff(x)**Rational(1, 2) + y, [x**2], [-2*x*jets.u[0] - 1]),
                        (sin(y.diff(x)) - x, [Integer(0)], [Integer(1)])]:
        p = _outcome(lambda: [str(c) for c in infinitesimal_condition(eq, y, Symmetry(jets, xi, eta))])
        r = _outcome(lambda: [str(c) for c in rc.logic_call('lie', [Symbol('condition'), Tuple(eq), Tuple(y),
                                                                   Tuple(*xi, *eta)]).args])
        assert r == p
    system, _, _ = determining_system(exp(y.diff(x)) - y, y)
    r = rc.logic_call('lie', [Symbol('system'), Tuple(exp(y.diff(x)) - y), Tuple(y)])
    assert [str(e) for e in r.args[0].args] == [str(e) for e in system]


def test_conditions_and_symmetry_operations() -> None:
    jets = JetSpace([u])
    usym = jets.u[0]
    vector_fields = [([2*t, Integer(0)], [-x*usym]), ([t, Integer(0)], [Integer(0)]),
                     ([x, 2*t], [usym]), ([Integer(1), Integer(0)], [Integer(0)])]
    for eq in [heat, burgers, kdv]:
        for xi, eta in vector_fields:
            p = infinitesimal_condition(eq, u, Symmetry(jets, xi, eta))
            r = rc.logic_call('lie', [Symbol('condition'), Tuple(eq), Tuple(u), Tuple(*xi, *eta)])
            assert [str(c) for c in r.args] == [str(c) for c in p]
            r = rc.logic_call('lie', [Symbol('check'), Tuple(eq), Tuple(u), Tuple(*xi, *eta)])
            assert bool(r) == check_symmetry(eq, u, Symmetry(jets, xi, eta))
    for (xi, eta), (xi2, eta2) in zip(vector_fields, vector_fields[1:]):
        s, o = Symmetry(jets, xi, eta), Symmetry(jets, xi2, eta2)
        args = [Symbol('symmetry'), Tuple(), Tuple(u), Tuple(*xi, *eta)]
        assert str(tuple(rc.logic_call('lie', args + [Symbol('characteristic')]).args)) == str(s.characteristic())
        assert str(rc.logic_call('lie', args + [Symbol('generator')])) == s.generator()
        c = rc.logic_call('lie', args + [Symbol('commutator'), Tuple(*xi2, *eta2)])
        expected = s.commutator(o)
        assert (str(tuple(c.args[0].args)), str(tuple(c.args[1].args))) == (str(expected.xi), str(expected.eta))
        applied = rc.logic_call('lie', args + [Symbol('apply'), burgers])
        assert str(applied) == str(s.apply(jets.to_jet(burgers)))


def test_leading_derivatives() -> None:
    # an equation not linear in its leading derivative raises: solving it
    # for one root would keep one branch only
    jets = JetSpace([u])
    for eq in [heat, burgers, kdv, u.diff(x, 2)**2 - u.diff(t)]:
        try:
            p = leading_derivatives(jets, [jets.to_jet(eq)])
        except NotImplementedError:
            with pytest.raises(NotImplementedError):
                rc.logic_call('lie', [Symbol('leading'), Tuple(eq), Tuple(u)])
            continue
        r = rc.logic_call('lie', [Symbol('leading'), Tuple(eq), Tuple(u)])
        assert [(str(a.args[0]), str(a.args[1])) for a in r.args] == [(str(k), str(w)) for k, w in p.items()]


@pytest.mark.parametrize('eq, funcs', [(y.diff(x, 2), [y]), (y.diff(x, 3) + y*y.diff(x, 2), [y]),
                                       (kdv, [u]), (heat, [u]), (y.diff(x, 2) + y**3, [y])])
def test_determining_system_and_janet(eq: Basic, funcs: list[Basic]) -> None:
    system, unknowns, base = determining_system(eq, funcs)
    r = rc.logic_call('lie', [Symbol('system'), Tuple(eq), Tuple(*funcs)])
    assert [str(e) for e in r.args[0].args] == [str(e) for e in system]
    assert [str(e) for e in r.args[1].args] == [str(e) for e in unknowns]
    assert [str(e) for e in r.args[2].args] == [str(e) for e in base]
    d = rc.logic_call('lie', [Symbol('janet'), Tuple(eq), Tuple(*funcs)])
    assert str(d) == str(symmetry_janet_basis(eq, funcs).dimension)
