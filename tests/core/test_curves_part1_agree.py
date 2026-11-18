"""``sympy_extras.integrals.curves`` (part 1) against its Rust port: the
radicals of the integrand, the primitive element and its minimal
polynomial (the curve), the integrand as an element of the function field
and back in the radicals, and the curve and integrand after `x = N + 1/z`;
dependent radicals (the random weights of a primitive element) and the
errors (no radical, a parameter, dependent radicals after the random
weights of Python's ``random.Random(str(f))``) included."""

from __future__ import annotations

from typing import Any

import pytest

from ..conftest import load
from .test_integers_agree import outcome


def integrands() -> list[Any]:
    from sympy_extras.core import Rational, symbols
    from sympy_extras.functions import sqrt
    x, a = symbols('x a')
    return [1/(sqrt(x + 1) + sqrt(x - 1)), sqrt(x), x/sqrt(x**2 + 1), (x + 1)**Rational(1, 3), sqrt(x) + sqrt(x + 1),
            1/(x*sqrt(x**3 + 1)), sqrt(1 + sqrt(x)), x**Rational(1, 3)*sqrt(x + 2), sqrt(x)*sqrt(4*x + 4) + 1,
            sqrt(x)*sqrt(x + 1)/(x + 2), x**2 + 1, sqrt(x + a), (x**2 - 1)**Rational(2, 3)/x,
            sqrt(x) + sqrt(-x), sqrt(x) + sqrt(x + 1) + sqrt(x**2 + x)]


@pytest.mark.parametrize("index", range(15))
def test_integrand(rust: Any, index: int) -> None:
    from sympy_extras.core import Integer, Symbol
    from sympy_extras.integrals.curves import AlgebraicIntegrand, _Transformed
    rc = load("sympy_extras_rs.core")
    x, y, z = Symbol('x'), Symbol('y'), Symbol('z')
    f = integrands()[index]

    def expected() -> Any:
        A = AlgebraicIntegrand(f, x)
        element = A.element_expr()
        return (str(A.curve.xreplace({A.y: y})), str(element.xreplace({A.y: y})), str(A.polynomial_to_expr(element)),
                str(A.theta), list(A.weights))

    def got() -> Any:
        r = rc.logic_call('curve_integrand', [f, x])
        rule = {r.args[5]: y}
        return (str(r.args[0].xreplace(rule)), str(r.args[1].xreplace(rule)), str(r.args[2]), str(r.args[3]),
                [int(w) for w in r.args[4].args])
    assert outcome(got) == outcome(expected), f
    for N in (2, -1):
        def transformed_py() -> Any:
            T = _Transformed(AlgebraicIntegrand(f, x), N)
            return (str(T.field.original.xreplace({T.z: z, T.integrand.y: y})),
                    str(T.to_x_expr(T.integrand_element, Integer(0))))

        def transformed_rs() -> Any:
            r = rc.logic_call('curve_transformed', [f, x, Integer(N)])
            return (str(r.args[0].xreplace({r.args[2]: z, r.args[3]: y})), str(r.args[1]))
        assert outcome(transformed_rs) == outcome(transformed_py), (f, N)
