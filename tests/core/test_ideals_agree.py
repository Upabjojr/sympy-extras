"""``sympy_extras.polys.ideals`` against its Rust port: the reduced
Gröbner bases for the three orders (``lex`` and ``grlex`` of
zero-dimensional ideals by FGLM), the change of order (FGLM and the
Gröbner walk), normal forms and membership, sums, products, powers,
intersections, quotients, saturations, elimination, the radical
membership, the Hilbert series and polynomial, dimension, degree,
height, standard monomials, multiplication matrices, minimal and
univariate polynomials, the radical in dimension zero and the maximality
test, over ``QQ`` and prime fields. The decompositions of positive
dimension (``idealdecomposition``) are not ported and not compared."""

from __future__ import annotations

from typing import Any, Callable

import pytest

from ..conftest import load
from .test_integers_agree import outcome


def _ideals() -> list[tuple[list[Any], int, str]]:
    from sympy_extras.core import Rational, symbols
    x, y, z = symbols('x y z')
    return [
        ([x**2 + y**2 - 1, x - y], 0, 'grevlex'),
        ([x**2 + y**2 - 1, x*y - 2], 0, 'grevlex'),
        ([x**3 - y**2, x*y - z], 0, 'grevlex'),
        ([x**2 - y*z, x*z - y**2], 0, 'lex'),
        ([x**2, x*y, y**3], 0, 'grevlex'),
        ([x**2 - 2, y**2 - 3], 0, 'grlex'),
        ([x*y - 1, x**2 + y - Rational(1, 2)], 0, 'grevlex'),
        ([x**2 + 1, y - x], 5, 'grevlex'),
        ([x**3 + y + 1, y**2 - x], 7, 'grevlex'),
        ([x**2 + y**2 + z**2 - 1, x - y, y - z], 0, 'grevlex'),
        ([x**2 - 1, (x - 1)*y, y**2 - y], 0, 'grevlex'),
        ([x*y, x*z, y*z], 0, 'grevlex'),
        ([x**4 - 1, y**2 - x], 0, 'grevlex'),
        ([x**2*y - z, y**2 - x*z + 1], 0, 'grevlex'),
    ]


def _python(gens: list[Any], p: int, order: str) -> Any:
    from sympy_extras.core import symbols
    from sympy_extras.polys.domains import GF, QQ
    from sympy_extras.polys.ideals import Ideal
    x, y, z = symbols('x y z')
    variables = [s for s in (x, y, z) if any(g.has(s) for g in gens)]
    return Ideal(gens, *variables, domain=QQ if p == 0 else GF(p), order=order), variables


def _strs(values: Any) -> Any:
    return [str(v) for v in values]


@pytest.mark.parametrize("index", range(14))
def test_ideal(rust: Any, index: int) -> None:
    from sympy_extras.core import Integer, Symbol, Tuple, symbols
    rc = load("sympy_extras_rs.core")
    gens, p, order = _ideals()[index]
    I, variables = _python(gens, p, order)
    x, y, z = symbols('x y z')
    others = [[x**2 - y], [x*y - 1], [x - 1, y - 1]]
    others = [[g for g in o if all(s in variables for s in g.free_symbols)] or [variables[0]] for o in others]

    def rust(op: str, *extra: Any) -> Callable[[], Any]:
        def call() -> Any:
            return rc.logic_call('ideal', [Symbol(op), Tuple(*gens), Tuple(*variables), Symbol(order), Integer(p),
                                           *extra])
        return call

    def show(r: Any) -> Any:
        return [str(v) for v in r.args] if isinstance(r, Tuple) else str(r)

    def check(op: str, expected: Callable[[], Any], *extra: Any, convert: Callable[[Any], Any] = show) -> None:
        got = outcome(lambda: convert(rust(op, *extra)()))
        assert got == outcome(expected), (op, gens, p, order, extra)

    def ideal_strs(J: Any) -> list[str]:
        return [str(J.to_expr(g)) for g in J.groebner_basis()]
    for o in ('grevlex', 'lex', 'grlex'):
        check('groebner_basis', lambda o=o: [str(I.to_expr(g)) for g in I.groebner_basis(o)], Symbol(o))
    for o in ('lex', 'grevlex'):
        J, _ = _python(gens, p, order)
        check('change_order', lambda o=o: [str(J.to_expr(g)) for g in J.change_order(o)], Symbol(o))
    for f in (x**3, variables[0]*variables[-1] + 1, variables[-1]**2 - 1):
        check('reduce', lambda f=f: str(I.reduce(f)), f)
        check('contains', lambda f=f: str(I.contains(f)), f)
        check('radical_contains', lambda f=f: str(I.radical_contains(f)), f)
    for o in others:
        J, _ = _python(o, p, order)
        J = type(J)(o, *variables, domain=J.domain, order=order)
        for op, fn in [('intersect', lambda J=J: ideal_strs(I.intersect(J))),
                       ('quotient', lambda J=J: ideal_strs(I.quotient(J))),
                       ('saturate', lambda J=J: ideal_strs(I.saturate(J))),
                       ('sum', lambda J=J: ideal_strs(I + J)), ('product', lambda J=J: ideal_strs(I*J)),
                       ('equals', lambda J=J: str(I == J)), ('subset', lambda J=J: str(I.subset(J)))]:
            check(op, fn, Tuple(*o))
    check('power', lambda: ideal_strs(I**2), Integer(2))
    if len(variables) > 1:
        E = I.eliminate([variables[0]])
        check('eliminate', lambda: [ideal_strs(E), _strs(E.symbols)], Tuple(variables[0]),
              convert=lambda r: [show(r.args[0]), show(r.args[1])])
    t, d = Symbol('t'), Symbol('d')
    check('hilbert_series', lambda: str(I.hilbert_series(t)), t)
    check('hilbert_polynomial', lambda: str(I.hilbert_polynomial(d)), d)
    for op in ('dimension', 'degree', 'height', 'is_zero_dimensional', 'is_whole_ring'):
        check(op, lambda op=op: str(getattr(I, op)()))
    zero_dimensional = I.is_zero_dimensional() or I.is_whole_ring()
    if zero_dimensional:
        check('standard_monomials', lambda: _strs(I.standard_monomials()))
        check('vector_space_dimension', lambda: str(I.vector_space_dimension()))
        for f in (variables[0], variables[0] + 2*variables[-1]):
            check('multiplication_matrix', lambda f=f: [_strs(I.domain.to_expr(c) for c in row)
                                                        for row in I.multiplication_matrix(f)], f,
                  convert=lambda r: [show(row) for row in r.args])
            check('minimal_polynomial', lambda f=f: str(I.minimal_polynomial(f, Symbol('w')).to_expr()), f,
                  Symbol('w'))
        for v in variables:
            check('univariate', lambda v=v: str(I.univariate(v).to_expr()), v)
        check('radical', lambda: ideal_strs(I.radical()))
        for op in ('is_radical', 'is_maximal', 'is_prime'):
            check(op, lambda op=op: str(getattr(I, op)()))
