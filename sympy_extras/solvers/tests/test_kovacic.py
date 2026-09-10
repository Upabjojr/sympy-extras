from __future__ import annotations

from sympy import Function, exp, simplify, Rational, I, Symbol, Poly, Integral, S, N
from sympy.abc import x
from sympy.testing.pytest import raises

from sympy_extras.solvers.kovacic import liouvillian_solution, dsolve_kovacic, normal_form, KovacicSolution
from sympy_extras.settings import configure
from sympy_extras._typing import as_expr

y = Function('y')(x)


def _riccati(omega: object, r: object) -> bool:
    w = as_expr(omega)
    return simplify(w.diff(x) + w**2 - as_expr(r)) == 0


def test_normal_form() -> None:
    assert normal_form(1/x, 1 - 1/(4*x**2), x) == -1
    assert normal_form(S.Zero, -x, x) == x


def test_case1() -> None:
    z = liouvillian_solution(1 + 2/x**2, x)
    assert isinstance(z, KovacicSolution) and z.case == 1
    assert z.omega is not None and _riccati(z.omega, 1 + 2/x**2)
    assert simplify(z.solution - (x - 1)*exp(x)/x) == 0
    z = liouvillian_solution(S(-1), x)
    assert z is not None and z.case == 1 and z.solution in (exp(I*x), exp(-I*x))
    z = liouvillian_solution(2/x**2, x)
    assert z is not None and z.case == 1 and z.solution in (1/x, x**2)
    assert liouvillian_solution(S.Zero, x) is not None


def test_case2() -> None:
    r = 1/x - Rational(3, 16)/x**2
    z = liouvillian_solution(r, x)
    assert z is not None and z.case == 2 and z.omega is not None
    assert _riccati(z.omega, r)
    # the two roots of the quadratic are both solutions of the Riccati equation
    w = Symbol('w')
    assert Poly(z.omega_polynomial, w).degree() == 2


def test_case3() -> None:
    # Kovacic's tetrahedral example
    r = -Rational(3, 16)/x**2 - Rational(2, 9)/(x - 1)**2 + Rational(3, 16)/(x*(x - 1))
    with configure(timeout=60.0):
        z = liouvillian_solution(r, x)
    assert z is not None and z.case == 3
    w = Symbol('w')
    p = Poly(z.omega_polynomial, w)
    assert p.degree() == 4
    # the root (when written in radicals) satisfies its polynomial: checked numerically
    if z.omega is not None:
        value = p.as_expr().subs(w, z.omega).subs(x, Rational(7, 3))
        assert abs(N(value, 30)) < 1e-20
    assert isinstance(z.solution, exp) and z.solution.args[0].has(Integral)


def test_no_liouvillian_solution() -> None:
    assert liouvillian_solution(x, x) is None            # Airy
    assert liouvillian_solution(x**2 + 2, x) is None     # Weber's equation with a non-integer parameter
    z = liouvillian_solution(x**2 + 3, x)                # ... and with an integer one
    assert z is not None and z.case == 1 and simplify(z.solution/(x*exp(x**2/2))).is_constant(x)
    raises(ValueError, lambda: liouvillian_solution(exp(x), x))


def test_dsolve_kovacic() -> None:
    found = dsolve_kovacic(y.diff(x, 2) + y.diff(x)/x + (1 - 1/(4*x**2))*y, y)
    assert found is not None and len(found) == 2
    for s in found:
        assert simplify(s.diff(x, 2) + s.diff(x)/x + (1 - 1/(4*x**2))*s) == 0
    found = dsolve_kovacic(y.diff(x, 2) - 2*x*y.diff(x) + 4*y, y)   # Hermite with n = 2
    assert found is not None and any(simplify(s/(2*x**2 - 1)).is_constant(x) for s in found)
    found = dsolve_kovacic(x*y.diff(x, 2) + (1 - x)*y.diff(x) - y, y)  # Laguerre-type: exp(x) solution
    assert found is not None and len(found) >= 1 and all(
        simplify(x*s.diff(x, 2) + (1 - x)*s.diff(x) - s) == 0 for s in found)
    assert dsolve_kovacic(y.diff(x, 2) - x*y, y) is None
    assert dsolve_kovacic(y.diff(x, 2) - y**2, y) is None
    assert dsolve_kovacic(y.diff(x, 2) - exp(x)*y, y) is None


def test_case_three_solution_has_no_auxiliary_symbol() -> None:
    # sympy-extras#31: the placeholder was omega *applied to the
    # polynomial*, so the solution read as a function of a polynomial in w
    # and w was a free symbol of it; omega is a function of x, defined by
    # the KovacicSolution's omega_polynomial.
    from sympy import Rational, Symbol
    from sympy.abc import x
    from sympy_extras.solvers import liouvillian_solution
    r = -Rational(3, 16)/x**2 - Rational(2, 9)/(x - 1)**2 + Rational(3, 16)/(x*(x - 1))
    sol = liouvillian_solution(r, x)
    assert sol is not None and sol.case == 3
    assert sol.solution.free_symbols == {x}
    assert Symbol('w') in sol.omega_polynomial.free_symbols
