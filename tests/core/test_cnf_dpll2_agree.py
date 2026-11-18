"""``sympy_extras.logic.cnf`` and ``sympy_extras.logic.dpll2`` against
their Rust ports: the clauses of random formulas (atoms, numbering,
clauses), conjunctions of them, every model in the solver's order and
satisfiability. The formulas are kept small: the distribution of the
conjunctive normal form is exponential."""

from __future__ import annotations

import random
from typing import Any

import pytest

from ..conftest import load, stable_seed
from .test_boolalg_agree import formula


def encoded(c: Any) -> str:
    return str((tuple(c.atoms), tuple(tuple(cl) for cl in c.clauses)))


def rust_encoded(t: Any) -> str:
    atoms, clauses = t.args
    return str((tuple(atoms.args), tuple(tuple(int(v) for v in cl.args) for cl in clauses.args)))


@pytest.mark.parametrize("seed", range(60))
def test_cnf_dpll2(rust: Any, seed: int) -> None:
    from sympy_extras.logic import cnf, dpll2
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("cnf dpll2", seed))
    for _ in range(5):
        f = formula(rng, rng.randint(1, 2))
        g = formula(rng, 1)
        c = cnf.to_clauses(f)
        assert rust_encoded(rc.logic_call('to_clauses', [f])) == encoded(c), f
        both = cnf.conjunction([c, cnf.to_clauses(g)])
        assert rust_encoded(rc.logic_call('clauses_conjunction', [f, g])) == encoded(both), (f, g)
        for clauses, args in ((c, [f]), (both, None)):
            if args is None:
                continue
            models = [[(str(k), v) for k, v in m.items()] for m in dpll2.propositional_models(clauses)]
            got = [[(str(p.args[0]), bool(p.args[1])) for p in m.args]
                   for m in rc.logic_call('propositional_models', args).args]
            assert got == models, f
            assert bool(rc.logic_call('is_satisfiable', args)) == dpll2.is_satisfiable(clauses), f
        # contradictions (no model) and their clauses
        from sympy_extras.core.logic import Not, and_
        h = and_(f, Not(f)) if not f.is_Atom else and_(f, Not(f), g)
        assert rust_encoded(rc.logic_call('to_clauses', [h])) == encoded(cnf.to_clauses(h)), h
        assert bool(rc.logic_call('is_satisfiable', [h])) == dpll2.is_satisfiable(cnf.to_clauses(h)), h
