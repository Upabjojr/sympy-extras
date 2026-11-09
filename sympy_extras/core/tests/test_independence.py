"""The core does not import SymPy: importing it, the elementary and special
functions, and running the usual operations leave ``sympy`` out of
``sys.modules`` (SymPy is lent to the core only by ``sympy_extras.interop``,
through :mod:`sympy_extras.core.bridge`)."""
from __future__ import annotations

import subprocess
import sys

_PROGRAM = r'''
import sys
from sympy_extras.core import symbols, Rational, sqrt, expand, diff, Eq, pi, factor_terms, I, Integer
from sympy_extras.core.limits import limit
from sympy_extras.core.bridge import ForeignOperationUnavailable
from sympy_extras.functions import sin, exp, log, gamma
x, y = symbols('x y')
results = [expand((x + y)**3/2), diff(sin(x)*exp(x), x), ((x + y)**2).subs(x, 2), factor_terms(2*x + 2*y),
           (sin(x)/x).series(x, 0, 6), limit(sin(x)/x, x, Integer(0)), gamma(Rational(1, 2)), log(exp(2)),
           pi.evalf(30), (x*I)**2, sqrt(8), Eq(x, x)]
assert str(results[4]) == '1 - x**2/6 + x**4/120 + O(x**6)', results[4]
try:
    from sympy_extras.core import sympify
    sympify('x + 1')
except ForeignOperationUnavailable:
    pass
else:
    raise AssertionError('parsing should need the SymPy bridge')
print('SYMPY' if 'sympy' in sys.modules else 'FREE')
'''


def test_core_does_not_load_sympy() -> None:
    out = subprocess.run([sys.executable, '-c', _PROGRAM], capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == 'FREE', out.stdout


def test_the_bridge_is_registered_by_interop() -> None:
    import sympy_extras.interop as interop
    from sympy_extras.core.bridge import bridge
    from sympy_extras.core import sympify
    assert isinstance(bridge(), interop.SymPyBridge)
    assert str(sympify('x + 1')) == 'x + 1'
