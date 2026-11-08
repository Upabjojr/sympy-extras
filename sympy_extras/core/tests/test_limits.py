"""Oracle tests of the native limits (Gruntz's algorithm) against SymPy."""
import sympy

from sympy_extras.core.interop import expr_from_sympy, to_sympy
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
