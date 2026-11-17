"""The evaluation of ``ite`` and the normal forms ``to_nnf``, ``to_cnf``
and ``to_dnf`` of :mod:`sympy_extras.logic.boolalg`, checked against
SymPy 1.14 through the interop (the same printed result), and against
the truth tables of the formulas (an independent oracle). Some cases port
the tests of ``sympy/logic/tests/test_boolalg.py`` (BSD 3-clause license,
see sympy_extras/LICENSE-SymPy)."""
from __future__ import annotations

from itertools import product

import sympy as s
from sympy.logic.boolalg import ITE as SymPyITE

from sympy_extras.core import Eq, Ne, Symbol
from sympy_extras.core.basic import Basic
from sympy_extras.core.logic import ITE, Boolean, Not, false, ite, to_boolean, true
from sympy_extras.interop import from_sympy, to_sympy
from sympy_extras.logic.boolalg import (distribute_and_over_or, distribute_or_over_and, eliminate_implications,
    is_cnf, is_dnf, is_nnf, simplify_logic, to_cnf, to_dnf, to_nnf)

a, b, c, x, y, z = (Symbol(n) for n in 'abcxyz')
sa, sb, sx, sy, sz = s.symbols('a b x y z')


def _native(e: object) -> Boolean:
    return to_boolean(from_sympy(s.sympify(e)))


def _same_function(f: Boolean, g: Boolean, atoms: list[Boolean]) -> bool:
    for values in product((true, false), repeat=len(atoms)):
        rule: dict[Basic, Basic] = dict(zip(atoms, values))
        if to_boolean(f._xreplace_evaluated(rule)) != to_boolean(g._xreplace_evaluated(rule)):
            return False
    return True


def test_ite_evaluation_matches_sympy() -> None:
    cases = [(True, sy, sz), (False, sy, sz), (sx, sy, sy), (sx, True, False), (sx, False, True), (sx > 0, sy, sy),
             (s.Eq(sy, 2), sy, sz), (s.Ne(sy, 2), sy, sz), (s.Eq(sx, sy), sy, sz), (s.Eq(sy, sz), sy, sz),
             (s.Eq(sx, 2), sy, sz), (sx, SymPyITE(sy, True, False), sz), (sx > 0, sx < 1, sx > -1),
             (sx | ~sx, sy, sz), (sx, sy & sz, sy | sz), (sx > 0, True, True), (sx > 0, False, True),
             (s.Or(True, False), s.And(True, True), s.Xor(True, True)), (True, False, True)]
    for case in cases:
        theirs = SymPyITE(*case)
        ours = ite(*[_native(t) for t in case])
        assert str(to_sympy(ours)) == str(theirs), (case, theirs, ours)


def test_ite_class_does_not_evaluate() -> None:
    # the node keeps its arguments; doit and subs evaluate
    e = ITE(x, y, y)
    assert e.args == (x, y, y) and e.doit() == y
    assert ITE(x, y, z).subs(x, True) == y and ITE(x, true, false).doit() == x
    assert ite(Eq(y, 2), y, z) == z and ite(Ne(y, 2), y, z) == y and ite(x > 0, false, true) == (x <= 0)


def test_ite_interop_round_trip() -> None:
    for e in [SymPyITE(sx, sy, sz), SymPyITE(sx > 0, SymPyITE(sy, sa, sb), sz)]:
        native = from_sympy(e)
        assert isinstance(native, ITE)
        assert to_sympy(native) == e


def test_normal_forms_match_sympy() -> None:
    formulas = [SymPyITE(sx, sy, sz), SymPyITE(sx, sy, sz) & sx, ~SymPyITE(sx, sy, sz),
                SymPyITE(sx, SymPyITE(sy, sa, sb), sz), SymPyITE(sx, sy, sz) | SymPyITE(sy, sz, sx),
                s.Xor(sx, sy, sz), s.Equivalent(sx, sy, sz), s.Implies(SymPyITE(sx, sy, sz), sa),
                SymPyITE(sx > 0, sy, sz) & (sx <= 0), ~(sx | sy) | sz, s.Equivalent(sx >> sy, sy >> sx),
                ~s.Xor(sx, sy), ~s.Equivalent(sx, sy), ~s.Implies(sx, sy), (sa & sb) | (sx & sy) | sz,
                (sx | ~sx) & (sy | sz)]
    for e in formulas:
        n = _native(e)
        for theirs, ours in [(s.to_nnf(e), to_nnf(n)), (s.to_nnf(e, False), to_nnf(n, False)),
                             (s.to_cnf(e), to_cnf(n)), (s.to_dnf(e), to_dnf(n)),
                             (s.logic.boolalg.eliminate_implications(e), eliminate_implications(n))]:
            assert str(theirs) == str(to_sympy(ours)), (e, theirs, ours)
        for form in (None, 'cnf', 'dnf'):
            assert str(s.simplify_logic(e, form, deep=False)) == str(to_sympy(simplify_logic(n, form))), (e, form)
        assert str(s.to_cnf(e, True)) == str(to_sympy(to_cnf(n, True))), e
        assert str(s.to_dnf(e, True)) == str(to_sympy(to_dnf(n, True))), e
        assert s.logic.boolalg.is_nnf(e) == is_nnf(n) and s.logic.boolalg.is_nnf(e, False) == is_nnf(n, False)


def test_normal_forms_are_equivalent_and_in_form() -> None:
    # the truth tables: an oracle independent of SymPy
    atoms: list[Boolean] = [a, b, x, y, z]
    formulas: list[Boolean] = [ITE(x, y, z), ITE(x, ITE(y, a, b), z), Not(ITE(x, y, z)) | a,
                               ITE(x, y, z) & ITE(y, z, x), ITE(a, x >> y, Not(ITE(b, z, x)))]
    for f in formulas:
        for g in (to_nnf(f), to_nnf(f, False), to_cnf(f), to_dnf(f), to_cnf(f, True), to_dnf(f, True)):
            assert _same_function(f, g, atoms), (f, g)
        assert is_nnf(to_nnf(f)) and is_cnf(to_cnf(f)) and is_dnf(to_dnf(f))
        assert is_cnf(to_cnf(f, True)) and is_dnf(to_dnf(f, True))


def test_normal_forms_examples_of_sympy() -> None:
    # PORTED FROM SYMPY: sympy/logic/tests/test_boolalg.py (test_to_nnf, test_to_cnf, test_to_dnf,
    # test_is_nnf, test_distribute)
    assert to_nnf(true) == true and to_nnf(false) == false and to_nnf(a) == a and to_nnf(Not(a)) == Not(a)
    assert str(to_nnf(a | Not(a) | b)) == 'True' and str(to_nnf(a & Not(a) & b)) == 'False'
    assert str(to_nnf(ITE(a, b, c))) == '(a | c) & (b | ~a)'
    assert str(to_nnf(Not(ITE(a, b, c)))) == '(a | ~c) & (~a | ~b)'
    assert str(to_cnf(~(b | c))) == '~b & ~c' and str(to_cnf((a & b) | c)) == '(a | c) & (b | c)'
    assert str(to_cnf(a >> b)) == 'b | ~a' and str(to_cnf(a >> (b & c))) == '(b | ~a) & (c | ~a)'
    assert str(to_dnf(a & (b | c))) == '(a & b) | (a & c)' and str(to_dnf(a >> b)) == 'b | ~a'
    assert is_nnf(a & b | ~c) and not is_nnf((a | ~a) & (b | c)) and is_nnf((a | ~a) & (b | c), False)
    assert not is_nnf(Not(a & b) | c) and not is_nnf((a >> b) & (b >> a))
    assert str(distribute_and_over_or(((a & b) | c) & d_)) == 'd & (a | c) & (b | c)'
    assert str(distribute_or_over_and(((a | b) & c) | d_)) == 'd | (a & c) | (b & c)'
    try:
        to_cnf(and_many(), True)
    except ValueError:
        pass
    else:
        raise AssertionError('more than 8 variables need force')
    assert is_cnf(to_cnf(and_many(), True, force=True))


d_ = Symbol('d')


def and_many() -> Boolean:
    vs = [Symbol('v%d' % i) for i in range(9)]
    return ITE(vs[0], vs[1] | vs[2], vs[3] & vs[4]) | (vs[5] & vs[6]) | (vs[7] & vs[8])
