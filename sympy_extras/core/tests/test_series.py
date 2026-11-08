"""Oracle tests of the native Order and series machinery against SymPy."""
import random

import sympy

from sympy_extras.core.basic import Basic
from sympy_extras.core.containers import Tuple
from sympy_extras.core.interop import expr_from_sympy, to_sympy
from sympy_extras.core.order import Order, order
from sympy_extras.core.symbol import Symbol

s = sympy
x, y = s.symbols('x y')
X, Y = Symbol('x'), Symbol('y')
SPOINTS = {'0': s.Integer(0), '1': s.Integer(1), 'oo': s.oo}


def _ours(e: Basic) -> str:
    return str(to_sympy(e, evaluate=False))


SERIES_CASES = ['cos(x)', 'exp(x)', '1/(1-x)', '1/x', 'sin(x)/x', 'sqrt(1+x)', 'x**x', 'exp(x)/x', '1/(x**2+x)',
                '(1+x)**y', 'log(1+x)', 'tan(x)', 'x/(exp(x)-1)', '1/sin(x)', '(x+y)**3', 'sqrt(x+x**2)',
                'log(x)*x', '(exp(x)-1)/x', 'exp(1/x)*x', 'x**(1/3)+x', 'log(cos(x))', 'sqrt(1-x)/x**2',
                'log(sin(x))', 'exp(exp(x))', 'log(x**2+x)', 'exp(x*log(x)+x)', 'log(-x+x**2)', 'exp(-1/x)',
                'log(exp(x)-1)', 'x**y*log(x)', 'log(1-x)/x', 'exp(sqrt(x))', 'log(sqrt(x)+x)',
                'sec(x)', 'csc(x)', 'cot(x)', 'tan(x+1)', 'cosh(x)', 'tanh(x)', 'coth(x)', 'sech(x)', 'csch(x)',
                'tan(pi/2+x)', 'cot(pi+x)', 'sin(pi+x)', 'cos(pi/2+x)', 'sec(pi/2+x)', 'csc(pi+x)', 'tanh(sin(x))',
                'cos(log(x))*x', 'sin(sqrt(x))', 'asin(x)', 'acos(x)', 'atan(x)', 'acot(x)', 'atan(1/x)',
                'acot(x+x**2)', 'acos(2*x)', 'atan(x**2)', 'asin(x+1)', 'acos(x-1)', 'atan(x+I)',
                'asec(x)', 'acsc(x)', 'asinh(x)', 'acosh(x)', 'atanh(x)', 'acoth(x)', 'asech(x)', 'acsch(x)',
                'asinh(x+I)', 'acsch(x+I)', 'asech(x+2)', 'asec(x+1)', 'acosh(x+1)']


def _sympy_expr(text: str) -> sympy.Expr:
    e = s.sympify(text)
    assert isinstance(e, s.Expr)
    return e


def test_series_matches_sympy() -> None:
    for text in SERIES_CASES:
        e = _sympy_expr(text)
        for point, n in (('0', 6), ('1', 3)):
            try:
                want = str(e.series(x, SPOINTS[point], n))
            except Exception as error:
                want = type(error).__name__
            try:
                got = _ours(expr_from_sympy(e).series(X, expr_from_sympy(SPOINTS[point]), n))
            except Exception as error:
                got = type(error).__name__
            assert got == want, (text, point)


def test_as_leading_term_matches_sympy() -> None:
    for text in SERIES_CASES:
        e = _sympy_expr(text)
        try:
            want = str(e.as_leading_term(x))
        except Exception as error:
            want = type(error).__name__
        try:
            got = _ours(expr_from_sympy(e).as_leading_term(X))
        except Exception as error:
            got = type(error).__name__
        assert got == want, text


def test_order_matches_sympy() -> None:
    rng = random.Random(1)
    atoms: list[sympy.Expr] = [x, y, x**2, 1/x, s.log(x), s.exp(x), s.sin(x), s.sqrt(x), s.Integer(2),
                               s.Rational(1, 3), x*y]

    def rnd(d: int) -> sympy.Expr:
        if d == 0:
            return rng.choice(atoms)
        op = rng.choice('+*/^')
        a, b = rnd(d - 1), rnd(d - 1)
        if op == '+':
            return a + b
        if op == '*':
            return a * b
        if op == '/':
            return a / b if b != 0 else a
        return a ** rng.choice([s.Integer(2), s.Integer(-1), s.Rational(1, 2), s.Integer(3)])

    specs: list[list[tuple[str, str]]] = [[('x', '0')], [('x', 'oo')], [('x', '1')], [('x', '0'), ('y', '0')]]
    symbols = {'x': (x, X), 'y': (y, Y)}
    for _ in range(80):
        e = rnd(rng.randint(1, 3))
        spec = rng.choice(specs)
        try:
            want = str(s.Order(e, *[(symbols[v][0], SPOINTS[p]) for v, p in spec]))
        except Exception as error:
            want = type(error).__name__
        try:
            got = _ours(order(expr_from_sympy(e), *[Tuple(symbols[v][1], expr_from_sympy(SPOINTS[p]))
                                                    for v, p in spec]))
        except Exception as error:
            got = type(error).__name__
        assert got == want, (e, spec)


def test_order_arithmetic() -> None:
    o = order(X**2)
    assert _ours(X + o + X**2) == 'x + O(x**2)'
    assert _ours(1 + order(1, X)) == str(1 + s.Order(1, x))
    assert _ours(X * order(X)) == 'O(x**2)'
    assert _ours(-o) == 'O(x**2)'
    assert _ours(order(X).subs(X, X - 1)) == 'O(x - 1, (x, 1))'
    ox, ox2 = order(X), order(X**2)
    assert isinstance(ox, Order) and isinstance(ox2, Order)
    assert ox.contains(X**2) is True and ox2.contains(X) is False


def test_is_meromorphic_matches_sympy() -> None:
    rng = random.Random(3)
    atoms: list[sympy.Expr] = [x, x**2, 1/x, s.Integer(2), s.Rational(1, 2), s.pi, x - 1, s.zoo, s.oo]
    funcs = [s.sin, s.cos, s.exp, s.log, s.sqrt, s.atan, s.tan, s.sinh, s.asin, s.Abs, s.gamma, s.erf,
             lambda a: a**2, lambda a: 1/a, lambda a: a**s.Rational(1, 3), lambda a: 2**a, lambda a: a**x,
             s.LambertW, s.acot, s.re]

    def rnd(d: int) -> sympy.Expr:
        if d == 0:
            return rng.choice(atoms)
        k = rng.random()
        if k < 0.3:
            return rnd(d - 1) + rnd(d - 1)
        if k < 0.6:
            return rnd(d - 1) * rnd(d - 1)
        result = rng.choice(funcs)(rnd(d - 1))
        assert isinstance(result, s.Expr)
        return result

    points = [s.Integer(0), s.Integer(1), s.Integer(-1), s.oo, s.Integer(2), s.I, -1/s.E]
    for _ in range(200):
        e = rnd(rng.randint(1, 3))
        a = rng.choice(points)
        want = e.is_meromorphic(x, a)
        got = expr_from_sympy(e).is_meromorphic(X, expr_from_sympy(a))
        # our assumptions may know more (gamma(tan(exp(-2))) is finite): an
        # unknown answer of SymPy may be decided here, a decided one agrees
        assert want is None or got == want, (e, a)


def test_generic_function_series_matches_sympy() -> None:
    # SymPy's generic Function._eval_nseries, native for the functions it
    # expands (their Taylor terms or derivatives)
    import sympy
    from sympy_extras.core import to_sympy
    from sympy_extras.core.interop import expr_from_sympy
    sx, sa = sympy.Symbol('x'), sympy.Symbol('a')
    cases = [f(ar) for f in (sympy.erf, sympy.erfi, sympy.erfc, sympy.fresnels, sympy.fresnelc)
             for ar in (sx, 2*sx, sx + 1, 3*sx**2, sx/2 - sx**2, 2*sx + 1)]
    cases += [sympy.erfinv(sx), sympy.erfinv(2*sx), sympy.erfinv(sx/2 - sx**2)]   # erfinv(1) is a pole
    cases += [sympy.digamma(sx + 1), sympy.harmonic(sx + 1), sympy.beta(sx + 1, 2), sympy.trigamma(sx + 2),
              sympy.Heaviside(sx + 1), sympy.re(sx), sympy.jacobi(2, sa, 1, sx), sympy.gegenbauer(3, sa, sx),
              sympy.assoc_laguerre(2, sa, sx), sympy.elliptic_f(sx, 2), sympy.multigamma(sx + 2, 2),
              sympy.airyaiprime(sx), sympy.airybiprime(2*sx)]
    from sympy_extras.core.interop import from_sympy
    from sympy_extras.core.symbol import Symbol as OurSymbol
    ox = from_sympy(sx)
    assert isinstance(ox, OurSymbol)
    for e in cases:
        got = expr_from_sympy(e).series(ox, 0, 5)
        assert str(to_sympy(got, evaluate=False)) == str(e.series(sx, 0, 5)), e


def test_generic_leading_terms_match_sympy() -> None:
    # SymPy's generic Function._eval_as_leading_term: the function itself, or
    # NotImplementedError when an argument vanishes (valid arities only)
    import sympy
    from sympy.core.function import Function as SympyFunction
    from sympy_extras.core import to_sympy
    from sympy_extras.core.interop import _native_to_sympy, expr_from_sympy, from_sympy
    from sympy_extras.core.symbol import Symbol as OurSymbol
    sx = sympy.Symbol('x')
    ox = from_sympy(sx)
    assert isinstance(ox, OurSymbol)
    shapes = {1: [[sx], [2*sx + 1], [1/sx]], 2: [[1, sx], [sx, 2], [sx + 1, sx]], 3: [[2, sx, 1]]}
    checked = 0
    for native, theirs in _native_to_sympy().items():
        if theirs._eval_as_leading_term is not SympyFunction._eval_as_leading_term:
            continue
        for k in sorted(native.nargs):
            for args in shapes.get(k, []):
                try:
                    e = theirs(*args)
                    want = str(e.as_leading_term(sx))
                except (NotImplementedError, ValueError, TypeError, ZeroDivisionError) as exc:
                    want = 'ERR ' + type(exc).__name__
                    e = None
                if e is None:
                    continue
                try:
                    got = str(to_sympy(expr_from_sympy(e).as_leading_term(ox), evaluate=False))
                except NotImplementedError:
                    got = 'ERR NotImplementedError'
                assert got == want, e
                checked += 1
    assert checked > 100


def test_special_leading_terms_match_sympy() -> None:
    # the class-specific _eval_as_leading_term rules of sympy_extras.functions._leading_terms
    import sympy
    from sympy.core.cache import clear_cache
    from sympy.core.function import PoleError as SympyPoleError
    from sympy_extras.core.coreerrors import PoleError
    from sympy_extras.core import to_sympy
    from sympy_extras.core.interop import expr_from_sympy, from_sympy
    from sympy_extras.core.symbol import Symbol as OurSymbol
    sx = sympy.Symbol('x')
    ox = from_sympy(sx)
    assert isinstance(ox, OurSymbol)
    cases = [sympy.LambertW(sx), sympy.LambertW(2*sx + 1), sympy.LambertW(sx, 2), sympy.factorial(sx),
             sympy.factorial(sx + 3), sympy.factorial(1/sx), sympy.binomial(sx, 2), sympy.gamma(sx), sympy.gamma(sx - 2),
             sympy.gamma(sx + 3), sympy.gamma(1/sx), sympy.polygamma(0, 1/sx), sympy.polygamma(0, -1/sx),
             sympy.polygamma(2, sx + 1), sympy.digamma(sx), sympy.trigamma(sx + 1), sympy.zeta(sx), sympy.zeta(3, sx),
             sympy.zeta(2, 1/sx), sympy.erf(sx), sympy.erf(1/sx), sympy.erfc(2*sx), sympy.erfi(sx), sympy.erfi(sx + 1),
             sympy.Ei(sx), sympy.Ei(-sx), sympy.Ei(sx + 1), sympy.Si(sx), sympy.Si(1/sx), sympy.Shi(2*sx),
             sympy.Ci(sx), sympy.Ci(sympy.sqrt(sx)), sympy.Chi(sx**2 + sx), sympy.Chi(sx + 3),
             sympy.fresnels(sx), sympy.fresnels(1/sx), sympy.fresnelc(sx), sympy.fresnelc(-1/sx), sympy.fresnelc(sx + 1)]
    for e in cases:
        clear_cache()
        try:
            want = str(e.as_leading_term(sx))
        except (NotImplementedError, ValueError, SympyPoleError) as exc:
            want = 'ERR ' + type(exc).__name__
        try:
            got = str(to_sympy(expr_from_sympy(e).as_leading_term(ox), evaluate=False))
        except (NotImplementedError, ValueError, PoleError) as exc:
            got = 'ERR ' + type(exc).__name__
        assert got == want, e


def test_more_leading_terms_match_sympy() -> None:
    # arg, floor, ceiling, frac, the Bessel functions, SingularityFunction,
    # and the nodes Derivative, Subs, Piecewise, hyper, meijerg
    import sympy
    from sympy.core.cache import clear_cache
    from sympy.core.function import PoleError as SympyPoleError
    from sympy_extras.core import to_sympy
    from sympy_extras.core.coreerrors import PoleError
    from sympy_extras.core.interop import expr_from_sympy, from_sympy
    from sympy_extras.core.symbol import Symbol as OurSymbol
    sx, sy = sympy.symbols('x y')
    ox = from_sympy(sx)
    assert isinstance(ox, OurSymbol)
    g = sympy.Function('f')
    half = sympy.Rational(1, 2)
    cases = [sympy.arg(sx), sympy.arg(-sx), sympy.arg(sx - 1), sympy.arg(sympy.I*sx),
             sympy.floor(sx), sympy.floor(-sx), sympy.floor(sx + half), sympy.floor(1/sx), sympy.floor(sympy.sin(1/sx)),
             sympy.ceiling(sx), sympy.ceiling(2 - sx), sympy.ceiling(-sx**2), sympy.frac(sx), sympy.frac(-sx),
             sympy.frac(sx + 1), sympy.frac(1/sx),
             sympy.besselj(0, sx), sympy.besselj(half, 1/sx), sympy.besselj(2, sx + 1), sympy.bessely(0, sx),
             sympy.bessely(1, 2*sx), sympy.bessely(2, 1/sx), sympy.besseli(1, sx**2), sympy.besseli(0, 1/sx),
             sympy.besselk(0, sx), sympy.besselk(2, sx), sympy.besselk(1, 1/sx), sympy.besselk(half, sx + 1),
             sympy.SingularityFunction(sx, 0, 0), sympy.SingularityFunction(sx, -1, 2), sympy.SingularityFunction(sx, 1, 1),
             sympy.SingularityFunction(sx, 0, -1),
             sympy.Derivative(sympy.sin(sx)*sx, sx), sympy.Derivative(sympy.exp(sx)/sx, sx),
             sympy.Subs(g(sy), sy, sx), sympy.Subs(g(sx), sx, 0), sympy.Subs(sy*sx + sx**2, sy, 1),
             sympy.Piecewise((sx, sx > 1), (sx**2 + sx, True)), sympy.Piecewise((sx, sy > 0)),
             sympy.hyper([1], [2], sx), sympy.hyper([1, 2], [3], 2*sx + 3),
             sympy.meijerg([[], []], [[0], []], sx + 1), sympy.meijerg([[], []], [[0], [half]], sx**2)]
    for e in cases:
        clear_cache()
        try:
            want = str(e.as_leading_term(sx))
        except (NotImplementedError, ValueError, SympyPoleError) as exc:
            want = 'ERR ' + type(exc).__name__
        clear_cache()
        try:
            got = str(to_sympy(expr_from_sympy(e).as_leading_term(ox), evaluate=False))
        except (NotImplementedError, ValueError, PoleError) as exc:
            got = 'ERR ' + type(exc).__name__
        assert got == want, e
