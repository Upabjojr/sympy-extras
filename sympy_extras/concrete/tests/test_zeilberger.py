from __future__ import annotations

import sympy
from sympy.testing.pytest import raises

from sympy_extras._testing import untyped
from sympy_extras._typing_sympy import as_expr, as_symbol
from sympy_extras.core import Basic, Eq, Expr, Function, Integer, Symbol, add, oo
from sympy_extras.core.evalf import numeric_value
from sympy_extras.core.function import AppliedUndef
from sympy_extras.concrete.expr_with_limits import Sum
from sympy_extras.concrete.zeilberger import zeilberger, wz_certificate, wz_prove, zeilberger_sum, Telescoper
from sympy_extras.functions import Piecewise, binomial, factorial, harmonic
from sympy_extras.interop import to_sympy
from sympy_extras.simplify.gammasimp import combsimp
from sympy_extras.simplify.simplify import simplify

n, k = Symbol('n', integer=True), Symbol('k', integer=True)
x = Symbol('x')
ZERO = Integer(0)


def _ex(b: object) -> Expr:
    assert isinstance(b, Expr), b
    return b


def _sympy(e: Expr) -> sympy.Expr:
    return as_expr(to_sympy(e))


def _same(a: object, b: object) -> bool:
    difference = _ex(a) - _ex(b)
    if simplify(combsimp(difference)) == ZERO:
        return True
    # numerically at a few points (the symbolic simplifiers miss some gamma identities)
    for i in range(0, 6):
        value = numeric_value(difference.subs(n, Integer(i)), 100)
        if value is None or abs(value) > 1e-20:
            return False
    return True


def test_zeilberger() -> None:
    Z = zeilberger(binomial(n, k), n, k)
    assert isinstance(Z, Telescoper) and Z.order == 1
    assert Z.coefficients == [Integer(-2), Integer(1)] and Z.check()
    Z = zeilberger(binomial(n, k)**2, n, k)
    assert Z is not None and Z.check()
    assert _same(Z.coefficients[0]/Z.coefficients[1], -2*(2*n + 1)/(n + 1))
    Z = zeilberger((-1)**k*binomial(2*n, k)**3, n, k)
    assert Z is not None and Z.check()
    Z = zeilberger(binomial(n, k)**2*binomial(n + k, k)**2, n, k)
    assert Z is not None and Z.order == 2 and Z.check()
    # a term with a free parameter
    Z = zeilberger(binomial(n, k)*x**k, n, k)
    assert Z is not None and Z.check()
    assert _same(Z.coefficients[0]/Z.coefficients[1], -(x + 1))
    assert zeilberger(binomial(n, k), n, k, max_order=0) is None
    raises(ValueError, lambda: zeilberger(binomial(n, k)**k, n, k))
    raises(TypeError, lambda: untyped(zeilberger)(n > k, n, k))


def test_wz() -> None:
    R = wz_certificate(binomial(n, k), 2**n, n, k)
    assert R is not None and _same(R, k/(2*(k - n - 1)))
    assert wz_prove(binomial(n, k), 2**n, n, k) is True
    assert wz_prove(binomial(n, k)**2, binomial(2*n, n), n, k) is True
    assert wz_prove(binomial(n, k)*binomial(x, k), binomial(n + x, n), n, k) is True
    assert wz_prove((-1)**k*binomial(2*n, k)**3, (-1)**n*factorial(3*n)/factorial(n)**3, n, k) is True
    # a wrong closed form has no certificate
    assert wz_certificate(binomial(n, k), 3**n, n, k) is None
    assert wz_prove(binomial(n, k), 2**n + 1, n, k) is None


def test_zeilberger_sum() -> None:
    assert zeilberger_sum(binomial(n, k), (k, 0, n)) == 2**n
    assert zeilberger_sum(binomial(n, k)**2, (k, 0, n)) == binomial(2*n, n)
    assert _same(zeilberger_sum((-1)**k*binomial(2*n, k)**3, (k, 0, 2*n)), (-1)**n*factorial(3*n)/factorial(n)**3)
    assert _same(zeilberger_sum(k*binomial(n, k), (k, 0, n)), n*2**(n - 1))
    assert _same(zeilberger_sum(binomial(n, k)*x**k, (k, 0, n)), (x + 1)**n)
    # sums with no hypergeometric closed form: the recurrence
    result = zeilberger_sum(binomial(n, k)**2*binomial(n + k, k)**2, (k, 0, n))
    assert isinstance(result, Eq) and result.has(Function('S')(n + 2))
    # infinite upper limit with vanishing terms
    assert zeilberger_sum(binomial(n, k), (k, 0, oo)) == 2**n
    assert zeilberger_sum(1/(k*(k + 1)), (k, 1, n)) is not None


def _is_closed_form(result: Basic) -> bool:
    return isinstance(result, Expr) and not result.has(Sum) and not result.atoms(AppliedUndef)


def test_polynomial_times_factorial_does_not_crash() -> None:
    # sympy-extras#43: SymPy's rsolve raised AttributeError on the
    # recurrence for this summand, and it escaped zeilberger_sum; the
    # sum has no hypergeometric closed form, so None (or a recurrence)
    # is the acceptable answer, a crash is not.
    result = zeilberger_sum((2*k**2 + k)*factorial(k), (k, 0, n), n)
    # a recurrence in S(n) is a legitimate answer; a closed form must be right
    if result is not None and _is_closed_form(result):
        for value in range(0, 6):
            direct = add(*[(2*j**2 + j)*factorial(Integer(j)) for j in range(0, value + 1)])
            assert _ex(result).subs(n, Integer(value)) == direct


def test_zeilberger_sum_never_returns_a_boolean() -> None:
    # sympy-extras#33: for this summand zeilberger_sum returned the
    # Boolean False -- a rejected check re-asserted as an answer -- where
    # the sum is a well defined integer for every n.
    from sympy_extras.core.logic import BooleanFalse, BooleanTrue
    from sympy_extras.core.numbers import NaN
    term = (-1)**k*binomial(n, k)*binomial(2*n - 2*k, n)
    result = zeilberger_sum(term, (k, 0, n), n)
    assert not isinstance(result, (BooleanTrue, BooleanFalse))
    assert result is None or not result.has(NaN)
    # a recurrence in S(n) is a legitimate answer; a closed form must be right
    if result is not None and _is_closed_form(result):
        for value in range(1, 6):
            direct = Sum(term.subs(n, Integer(value)), (k, Integer(0), Integer(value))).doit()
            assert _ex(result).subs(n, Integer(value)) == direct, value


def test_isolated_values_of_the_parameters() -> None:
    from sympy_extras.concrete.tests.cases import check_sum_cases
    y = Symbol('y')
    m = Symbol('m')
    ks, ns, ms = as_symbol(to_sympy(k)), as_symbol(to_sympy(n)), as_symbol(to_sympy(m))
    # the bug: the upper limit m was not taken for the variable of the
    # sum when it is not in the summand; y was, and y**k was "not
    # hypergeometric in y"
    value = zeilberger_sum(y**k, (k, 0, m))
    # the closed form (y**(m + 1) - 1)/(y - 1) is 0/0 at y = 1, where the
    # sum is m + 1
    assert check_sum_cases(_sympy(_ex(value)), _sympy(y**k), ks, 0, ms, ms) == 1
    value = zeilberger_sum(k*y**k, (k, 0, n), n)
    assert check_sum_cases(_sympy(_ex(value)), _sympy(k*y**k), ks, 0, ns, ns) == 1
    assert _ex(value).subs(y, Integer(1)) == n*(n + 1)/2
    assert not isinstance(zeilberger_sum(k*y**k, (k, 0, n), n, special_values=False), Piecewise)
    # a closed form defined for every value of the parameter gets no case
    assert zeilberger_sum(binomial(n, k)*x**k, (k, 0, n)) == (x + 1)**n
    # nor does a recurrence, an identity in the parameters
    recurrence = zeilberger_sum(binomial(n, k)**2*binomial(n + k, k)**2, (k, 0, n))
    assert isinstance(recurrence, Eq)


def test_recurrences_sympy_rsolve_misses() -> None:
    # the recurrence S(n+1) - S(n) = 1/(n+1) is solved by the d'Alembertian
    # solver of sympy_extras.solvers.recurr (SymPy's rsolve returns None)
    assert zeilberger_sum(1/k, (k, 1, n)) == harmonic(n)
    # the shifts in n of n! (-1)**k/k! are linearly dependent: Zeilberger's
    # recurrence has certificate 0 and the sum's recurrence comes from the
    # boundary term alone (it used to give no recurrence at all)
    Z = zeilberger((-1)**k*factorial(n)/factorial(k), n, k)
    assert Z is not None and Z.certificate == ZERO
    assert [simplify(c) for c in Z.coefficients] == [-n - 1, Integer(1)]
    S = Function('S')
    result = zeilberger_sum((-1)**k*factorial(n)/factorial(k), (k, 0, n))
    assert isinstance(result, Eq)
    assert simplify(_ex(result.lhs) - (S(n + 1) - (n + 1)*S(n) - (-1)**(n + 1))) == ZERO
    # a solution which is itself a sum is no closed form: the recurrence
    result = zeilberger_sum(factorial(k), (k, 0, n))
    assert isinstance(result, Eq) and result.has(factorial(n + 1))


def test_hypergeometric_right_hand_side_with_factorials() -> None:
    # the recurrence S(n + 1) - 4 S(n) = -(2n)!/(n! (n + 1)!) of the sum
    # of binomial(2n, k) up to n: rsolve_linear multiplied the equation
    # by the factorials of the right-hand side and refused it (SymPy's
    # rsolve solved it)
    assert str(zeilberger_sum(binomial(2*n, k), (k, 0, n))) == '4**n/2 + factorial(2*n)/(2*factorial(n)**2)'


def test_parametric_leading_coefficient() -> None:
    # SymPy's real_roots, which found the integer zeros of the leading
    # coefficient of the recurrence, raised NotImplementedError over
    # ZZ[a]; the sums are Beta(a, n + 1) and binomial(2n, n + a)
    from sympy_extras.core import Rational
    a = Symbol('a')
    for term, expected in [((-1)**k*binomial(n, k)/(k + a), 'gamma(a)*gamma(n + 1)/gamma(a + n + 1)'),
                           (binomial(n, k)*binomial(n, k + a), 'gamma(2*n + 1)/(gamma(-a + n + 1)*gamma(a + n + 1))')]:
        value = zeilberger_sum(term, (k, 0, n), n)
        assert str(value) == expected
        for v in range(4):
            at = {n: Integer(v), a: Rational(5, 3)}
            direct = add(*[term.subs(at).subs(k, Integer(i)) for i in range(v + 1)])
            difference = numeric_value(_ex(value).subs(at) - direct, 100)
            assert difference is not None and abs(difference) < 1e-20


def test_sympy_face() -> None:
    from sympy_extras.concrete.zeilberger_sympy import zeilberger_sum as sympy_sum, wz_certificate as sympy_wz
    ns, ks = sympy.symbols('n k', integer=True)
    assert sympy_sum(sympy.binomial(ns, ks)**2, (ks, 0, ns)) == sympy.binomial(2*ns, ns)
    assert isinstance(sympy_sum(sympy.factorial(ks), (ks, 0, ns)), sympy.Eq)
    assert str(sympy_wz(sympy.binomial(ns, ks), 2**ns, ns, ks)) == 'k/(2*(k - n - 1))'
