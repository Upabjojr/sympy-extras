"""``sympy_extras.polys.regularchains.solutions.exact_solutions`` against
its Rust port: chains solved by root objects, divisions, and the
numerical matching of the roots of an eliminant (mpmath's ``polyroots``,
ported); the points compared as strings, in order."""

from __future__ import annotations

from typing import Any

import pytest

from ..conftest import load


def _symbols() -> Any:
    from sympy_extras.core import Symbol
    return Symbol('x'), Symbol('y'), Symbol('z')


def _chains() -> list[tuple[list[Any], list[Any]]]:
    x, y, z = _symbols()
    chains = [([y**2 - 2, x*y - 1], [y, x]), ([y**3 - y - 1, x**2 - y], [y, x]), ([y**3 - y - 1, x - y**2], [y, x]),
              ([y**2 + 1, x - y], [y, x]), ([y**3 + 64, x - y], [y, x]), ([y**3 - 2, x**3 - 2], [y, x]),
              ([y**2 - 3, x**2 - y], [y, x]), ([y**2 - y - 1, x**2 + y*x + 1], [y, x]),
              ([y**4 - 10*y**2 + 1, x**2 - 2*y], [y, x]), ([z**2 - 2, y**2 - z, x - y*z], [z, y, x]),
              ([y**5 - y + 1, x*y**2 - 1], [y, x]), ([y**3 - 3*y + 1, x**3 - y], [y, x])]
    from sympy_extras.polys.regularchains import triangularize
    for system in ([x**2 + y**2 - 5, x*y - 1, z - x - y], [x**2 - 2, y**2 - x*y - 1]):
        gens = (x, y, z) if len(system) == 3 else (x, y)
        for chain in triangularize(system, *gens):
            chains.append((list(chain.polys), list(chain.main_variables)))
    return chains


def _python(chain: Any, real: bool) -> Any:
    from sympy_extras.polys.regularchains.solutions import exact_solutions
    try:
        return ('ok', [[(str(k), str(v)) for k, v in point.items()] for point in exact_solutions(*chain, real=real)])
    except NotImplementedError:
        return ('SKIP', None)
    except Exception as exc:
        return ('ERROR ' + type(exc).__name__, None)


def _rust(chain: Any, real: bool) -> Any:
    from sympy_extras.core import Tuple
    from sympy_extras.core.logic import false, true
    rc = load("sympy_extras_rs.core")
    try:
        r = rc.logic_call('rc_solutions', [Tuple(*chain[0]), Tuple(*chain[1]), true if real else false])
        return ('ok', [[(str(k), str(v)) for k, v in (pair.args for pair in point.args)] for point in r.args])
    except NotImplementedError:
        return ('SKIP', None)
    except Exception as exc:
        return ('ERROR ' + type(exc).__name__, None)


@pytest.mark.parametrize("index", range(len(_chains())))
@pytest.mark.parametrize("real", [False, True])
def test_exact_solutions(rust: Any, index: int, real: bool) -> None:
    chain = _chains()[index]
    assert _rust(chain, real) == _python(chain, real), chain
