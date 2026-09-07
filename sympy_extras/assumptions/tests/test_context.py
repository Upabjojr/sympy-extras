from __future__ import annotations

from sympy import Abs, Q, ask as sympy_ask, refine as sympy_refine, sqrt
from sympy.assumptions import global_assumptions as sympy_global_assumptions
from sympy.abc import x, y

from sympy_extras.assumptions import (assuming, global_assumptions, ask, refine,
    AssumptionsContext)


def test_assumptions_context() -> None:
    c = AssumptionsContext([x > 0, y > 0])
    assert len(c) == 2 and (x > 0) in c and list(c) == [x > 0, y > 0]
    c.add(x > 0)
    assert len(c) == 2
    c.add(x < 1, y < 1)
    assert len(c) == 4
    c.remove(y < 1, x > 5)
    assert list(c) == [x > 0, y > 0, x < 1]
    assert c.as_boolean() == ((x > 0) & (y > 0) & (x < 1))
    assert bool(c)
    c.clear()
    assert len(c) == 0 and not c and c.as_boolean() is True or c.as_boolean() == True
    assert repr(AssumptionsContext([x > 0])) == "AssumptionsContext([x > 0])"


def test_global_assumptions() -> None:
    assert len(global_assumptions) == 0
    assert ask(x > 0) is None
    global_assumptions.add(x > 0)
    try:
        assert ask(x > 0) is True
        assert ask(x**3 + x > 0) is True
        assert refine(Abs(x)) == x
        # explicit assumptions are combined with the global ones
        assert ask(x*y > 0, y > 0) is True
    finally:
        global_assumptions.clear()
    assert ask(x > 0) is None


def test_assuming() -> None:
    assert refine(Abs(x - 1)) == Abs(x - 1)
    with assuming(x > 1):
        assert refine(Abs(x - 1) + sqrt(x**2)) == 2*x - 1
        assert ask(x > 0) is True
        # sympy's own functions see the predicates
        assert sympy_ask(Q.positive(x)) is True
        assert sympy_refine(Abs(x)) == x
        assert (x > 1) in global_assumptions
        with assuming(y < 0):
            assert ask(x*y < 0) is True
            assert (y < 0) in global_assumptions
        assert (y < 0) not in global_assumptions
        assert ask(x*y < 0) is None
    assert refine(Abs(x - 1)) == Abs(x - 1)
    assert sympy_ask(Q.positive(x)) is None
    assert len(global_assumptions) == 0
    assert len(sympy_global_assumptions) == 0
    # an assumption already in the global context is not removed
    global_assumptions.add(x > 0)
    try:
        with assuming(x > 0, y > 0):
            assert ask(x + y > 0) is True
        assert (x > 0) in global_assumptions
        assert (y > 0) not in global_assumptions
    finally:
        global_assumptions.clear()
    # an exception in the block restores the context
    try:
        with assuming(x > 0):
            raise RuntimeError
    except RuntimeError:
        pass
    assert len(global_assumptions) == 0 and len(sympy_global_assumptions) == 0
