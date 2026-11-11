"""Oracle tests of the native limits (Gruntz's algorithm) against SymPy."""
import sympy

from sympy_extras.interop import expr_from_sympy, to_sympy
from sympy_extras.core.limits import Limit, limit
from sympy_extras.core.symbol import Symbol

s = sympy
x = s.Symbol('x')
X = Symbol('x')

CASES = [
    ('sin(x)/x', '0'), ('(1-cos(x))/x**2', '0'), ('(1+1/x)**x', 'oo'), ('x*exp(-x)', 'oo'), ('log(x)/x', 'oo'),
    ('x**x', '0'), ('1/x', '0'), ('exp(1/x)', '0'), ('exp(-1/x)', '0'), ('(exp(x)-1)/x', '0'), ('tan(x)/x', '0'),
    ('atan(x)', 'oo'), ('atan(x)', '-oo'), ('exp(x)/x**100', 'oo'), ('(x**2+x)/(2*x**2-1)', 'oo'),
    ('sqrt(x**2+x)-x', 'oo'), ('x*sin(1/x)', 'oo'), ('(1+x)**(1/x)', '0'), ('x*log(x)', '0'),
    ('exp(x)*(exp(1/x - exp(-x)) - exp(1/x))', 'oo'), ('exp(exp(x))/exp(exp(x - exp(-x)))', 'oo'),
    ('gamma(1/x)*x', 'oo'), ('erf(x)', 'oo'), ('x**(1/x)', 'oo'), ('log(log(x))/log(x)', 'oo'),
    ('sinh(x)/exp(x)', 'oo'), ('tanh(x)', 'oo'), ('asin(x)/x', '0'), ('(3**x+5**x)**(1/x)', 'oo'),
    ('x**2*(exp(1/x)-1-1/x)', 'oo'), ('factorial(x)/x**x', 'oo'), ('Abs(x)/x', '0'), ('sign(x)', '0'),
    ('1/(x-1)', '1'), ('(sin(x)-x)/x**3', '0'), ('exp(-x)*sin(x)', 'oo'), ('loggamma(x)/x', 'oo'),
    ('Ei(x)*exp(-x)*x', 'oo'), ('acot(x)', '0'), ('sqrt(1-cos(x))/x', '0'), ('(1/x)**(1/log(1/x))', '0'),
]


def test_limits_match_sympy() -> None:
    for text, point in CASES:
        e = s.sympify(text)
        z0 = s.sympify(point)
        for d in (('+',) if point in ('oo', '-oo') else ('+', '-')):
            want = str(s.limit(e, x, z0, d))
            got = str(to_sympy(limit(expr_from_sympy(e), X, expr_from_sympy(z0), d), evaluate=False))
            assert got == want, (text, point, d)


def test_limit_node() -> None:
    from sympy_extras.functions.elementary.trigonometric import sin
    node = Limit(sin(X) / X, X, 0)
    assert str(node) == "Limit(sin(x)/x, x, 0, dir='+')"
    assert node.doit() == 1
    assert to_sympy(node, evaluate=False) == s.Limit(s.sin(x) / x, x, 0)
    assert expr_from_sympy(s.Limit(s.sin(x) / x, x, 0)) == node


def test_limit_through_a_hypergeometric_function() -> None:
    # mrv raised NotImplementedError ("Don't know how to calculate the mrv
    # of 'hyper(...)'") because Hyper is not an Application of the core,
    # so the limit stayed unevaluated where SymPy's gruntz treats hyper as
    # a Function (test_definite's binomial differentials lost their value)
    from sympy_extras.core.numbers import Rational
    from sympy_extras.core.power import sqrt
    from sympy_extras.functions.special.hyper import hyper
    t = Symbol('t')
    h = hyper([Rational(-7, 4), Rational(1, 2)], [Rational(-3, 4)], 1/(5*t**2))
    e = 2*sqrt(5)*t*sqrt(t**7/(1 - 5*t**2))*sqrt(5 - 1/t**2)*h/35
    value = limit(e, t, Rational(-1, 2), '-')
    expected = -sqrt(10)*hyper([Rational(-7, 4), Rational(1, 2)], [Rational(-3, 4)], Rational(4, 5))/280
    assert value == expected
    T = s.Symbol('t')
    assert to_sympy(value) == s.limit(to_sympy(e), T, s.Rational(-1, 2), '-')


def test_logarithms_of_products_cancel_in_the_leading_term() -> None:
    # expand split log(a*(-1 - I)) only after distributing the product into
    # -a - a*I, so the leading term of I*log(a*(-1 + I))/2 - I*log(a*(-1 - I))/2
    # (a < 0) stayed a sum of logarithms instead of SymPy's pi/4
    from sympy_extras.core.numbers import I
    from sympy_extras.core.symbol import Dummy
    from sympy_extras.functions import log
    a = Dummy('a', negative=True)
    t = Dummy('t', positive=True)
    from sympy_extras.core.power import sqrt
    r = sqrt(a**2 - t**2*a**2)/(t + 1)
    e = -a*(-I*log(-I*a + r)/2 + I*log(I*a + r)/2)
    from sympy_extras.core.numbers import pi
    assert limit(e, t, 0, '+') == -pi*a/4
