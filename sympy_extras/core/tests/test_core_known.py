"""The experimental core against SymPy: the evaluated forms, the
derivatives, the expansions, the numerical values and the printed forms
of expressions computed by both, and the round trip of the conversions.

SymPy's results are the facts here: what ``Add``, ``Mul`` and ``Pow``
evaluate to is documented by SymPy's behaviour, and ``add``, ``mul`` and
``pow`` are expected to agree with it up to the order of the arguments
(and up to two quirks of SymPy noted in ``test_known_differences``).
"""
from __future__ import annotations

import importlib
import random
from itertools import islice
from typing import Callable

import sympy

from sympy_extras import core as C
from sympy_extras._testing import untyped
from sympy_extras.core import from_sympy, to_sympy
from sympy_extras.core.assumptions import ask_structural

sx, sy, sz = sympy.Symbol('x'), sympy.Symbol('y'), sympy.Symbol('z')
sp = sympy.Symbol('p', positive=True)
sn = sympy.Symbol('n', integer=True)
cx, cy, cz = C.Symbol('x'), C.Symbol('y'), C.Symbol('z')
cp = C.Symbol('p', positive=True)
cn = C.Symbol('n', integer=True)
LOCALS = {'x': sx, 'y': sy, 'z': sz, 'p': sp, 'n': sn}

#: a SymPy expression and the expression of this core
Pair = tuple[sympy.Expr, C.Expr]


def parse(text: str) -> sympy.Expr:
    """A SymPy expression from its text, with the symbols of this module."""
    from sympy.parsing.sympy_parser import parse_expr
    e = parse_expr(text, local_dict=dict(LOCALS, f=sympy.Function('f'), g=sympy.Function('g')))
    if not isinstance(e, sympy.Expr):
        raise TypeError("an expression is expected, got %s" % (e,))
    return e


def sympy_expr(e: sympy.Basic) -> sympy.Expr:
    if not isinstance(e, sympy.Expr):
        raise TypeError("an expression is expected, got %s" % (e,))
    return e


def same_structure(a: sympy.Basic, b: sympy.Basic) -> bool:
    """Structural equality of two SymPy expressions up to the order of
    the arguments of sums and products."""
    if a.is_Atom or b.is_Atom:
        return bool(a == b)
    if a.func is not b.func or len(a.args) != len(b.args):
        return False
    if isinstance(a, (sympy.Add, sympy.Mul)):
        rest = list(b.args)
        for u in a.args:
            for i, v in enumerate(rest):
                if same_structure(u, v):
                    del rest[i]
                    break
            else:
                return False
        return True
    return all(same_structure(u, v) for u, v in zip(a.args, b.args))


def agrees(c: C.Basic, s: sympy.Basic) -> bool:
    """Whether the core's result ``c`` is SymPy's ``s``: the same tree, or
    the same expression once SymPy's quirks are evaluated by SymPy itself
    (``9*(2/3 + I)/13`` is left undistributed by SymPy's ``Add._eval_power``)."""
    cs = to_sympy(c, evaluate=False)
    if same_structure(cs, s):
        return True
    difference = sympy.expand(sympy_expr(to_sympy(c)) - sympy_expr(s))
    return bool(difference == 0)


def test_known_evaluations() -> None:
    """Evaluated forms documented by SymPy's behaviour (SymPy 1.14)."""
    half, third = C.Rational(1, 2), C.Rational(1, 3)
    cases: list[tuple[C.Expr, str]] = [
        (C.sqrt(8), '2*sqrt(2)'), (C.sqrt(12), '2*sqrt(3)'), (C.Integer(8)**C.Rational(2, 3), '4'),
        (C.Integer(2)**C.Rational(7, 6), '2*2**(1/6)'), (C.Integer(4)**third, '2**(2/3)'),
        (C.Integer(18)**third, '18**(1/3)'), (C.Integer(72)**third, '2*3**(2/3)'),
        (third**half, 'sqrt(3)/3'), (C.Rational(2, 3)**half, 'sqrt(6)/3'), (third**third, '3**(2/3)/3'),
        (C.Rational(8, 27)**third, '2/3'), (C.Integer(8)**(-half), 'sqrt(2)/4'),
        (C.Integer(-8)**third, '2*(-1)**(1/3)'), (C.Integer(-4)**half, '2*I'), (C.Integer(-2)**half, 'sqrt(2)*I'),
        (C.Integer(-1)**C.Rational(4, 3), '-(-1)**(1/3)'), (C.Integer(-1)**C.Rational(-1, 3), '-(-1)**(2/3)'),
        (C.Integer(-1)**C.Rational(3, 2), '-I'), (C.Integer(-1)**C.Rational(5, 2), 'I'),
        (C.Integer(-3)**third, '(-3)**(1/3)'), (C.Integer(-4)**third, '(-1)**(1/3)*2**(2/3)'),
        (C.Integer(-16)**third, '2*(-2)**(1/3)'), (C.Integer(-12)**C.Rational(2, 3), '2*(-1)**(2/3)*18**(1/3)'),
        (C.Integer(-3)**C.Rational(4, 3), '-3*(-3)**(1/3)'), (C.Integer(-3)**(-third), '-(-3)**(2/3)/3'),
        (C.Rational(-8, 27)**third, '2*(-1)**(1/3)/3'), (C.Rational(-2, 3)**third, '(-2)**(1/3)*3**(2/3)/3'),
        (C.Integer(-9)**C.Rational(1, 4), '(-1)**(1/4)*sqrt(3)'),
        (C.sqrt(2)*C.sqrt(6), '2*sqrt(3)'), (C.sqrt(6)/C.sqrt(2), 'sqrt(3)'), (C.sqrt(2)*C.sqrt(2), '2'),
        (C.Integer(2)**half*C.Integer(2)**third, '2**(5/6)'), (C.Integer(2)**cx*C.Integer(3)**cx, '6**x'),
        (C.Integer(2)**cx*C.Integer(2)**cy, '2**x*2**y'), (C.Integer(2)**cx/2, '2**x/2'),
        (C.Integer(-5)**C.Rational(2, 3)*C.Integer(2)**C.Rational(2, 3), '(-10)**(2/3)'),
        (C.Integer(-2)**third*C.Integer(-3)**third, '(-1)**(2/3)*6**(1/3)'),
        (C.I*C.I, '-1'), (C.I**C.Rational(5, 3), 'I**(5/3)'), (C.I**C.Rational(-1, 3), '-I**(5/3)'),
        (C.I**C.Rational(7, 3), '-I**(1/3)'), (C.I**half, 'sqrt(I)'), (C.I*C.I**C.Rational(5, 3), 'I*I**(5/3)'),
        (1/(2 + C.I), '(2 - I)/5'), (1/(1 - C.I), '(1 + I)/2'), ((2 + C.I)**-2, '(2 + I)**(-2)'),
        ((2*cx)**2, '4*x**2'), ((2*cx)**half, 'sqrt(2)*sqrt(x)'), ((4*cx)**half, '2*sqrt(x)'),
        ((-4*cx)**half, '2*sqrt(-x)'), ((cx*cy)**half, 'sqrt(x*y)'), ((cp*cx)**half, 'sqrt(p)*sqrt(x)'),
        ((cp**2*cx)**half, 'p*sqrt(x)'), ((2*cx)**cy, '(2*x)**y'), ((cp*cx)**cy, '(p*x)**y'),
        ((-C.pi)**cz, '(-pi)**z'), ((C.Rational(-13, 6)/cp)**(-third), '-(-13)**(2/3)*6**(1/3)*p**(1/3)/13'),
        ((cx**2)**3, 'x**6'), ((cx**2)**cn, 'x**(2*n)'), ((cx**half)**third, 'x**(1/6)'),
        ((cx**2)**half, 'sqrt(x**2)'), ((cx**-1)**half, 'sqrt(1/x)'), ((cp**2)**half, 'p'),
        ((cx**C.Rational(2, 3))**C.Rational(3, 2), 'x'), ((C.pi**C.Rational(2, 3))**C.I, 'pi**(2*I/3)'),
        ((third**cx)**-1, '3**x'), ((C.Rational(2, 3)**cx)**-1, '(2/3)**(-x)'), (third**(-cx), '(1/3)**(-x)'),
        (-half**cp, '-1/2**p'), (half**cp, '(1/2)**p'),
        (2*(cx + cy), '2*x + 2*y'), (-(cx + cy), '-x - y'), (cx*(cx + cy), 'x*(x + y)'),
        (2*cx*(cx + cy), '2*x*(x + y)'), (C.oo*(cx + cy), 'oo*(x + y)'), (C.Float(2)*(cx + cy), '2.0*x + 2.0*y'),
        (cx**cy*cx**cz, 'x**y*x**z'), (cx**2*cx**cy, 'x**2*x**y'), (cx**cy*cx**cy, 'x**(2*y)'),
        (cx**cy*cx**(2*cy), 'x**(3*y)'), (cx**2*cx**-2, '1'), (cx**cn*cx**(cn + 1), 'x**n*x**(n + 1)'),
        ((cx + 1)**cn*(cx + 1)**-1, '(x + 1)**n/(x + 1)'),
        (C.oo + cp, 'oo'), (C.oo + cx, 'x + oo'), (C.zoo + C.pi, 'zoo'), (C.zoo + 3*C.I, 'zoo'),
        (C.zoo + 2*C.I*cp, 'zoo'), (C.oo*C.pi, 'oo'), (C.oo*(2*cp + 2), 'oo'), (C.zoo*(2*cp + 2), 'zoo'),
        (C.oo*(-cp), '-oo'), (C.oo*(cx + 1), 'oo*(x + 1)'), (C.oo*C.I*C.pi, 'oo*I'), (C.zoo*(1 + C.I), 'zoo'),
        (C.zoo*(cx + C.I), 'zoo*(x + I)'), (C.oo*cx + 2*cx, 'oo*x'), (C.oo*cx - C.oo*cx, 'nan'),
        (3*C.oo - C.oo, 'nan'), (C.Integer(0)**C.zoo, 'nan'), (C.Integer(2)**-C.oo, '0'), (C.Integer(-2)**C.oo, 'zoo'),
        (C.Integer(0)**C.I, 'nan'),
        (C.Float(2)**3, '8.00000000000000'), (C.Float(-4)**half, '2.0*I'), (C.Float(0)**-1, 'zoo'),
        (C.Float(0)*cx, '0'), (C.Float(1)*cx, '1.0*x'), (C.Float(2)*cx + C.Float(3)*cx, '5.0*x'),
        (cx**C.Float(2)*cx**3, 'x**5.0'),
    ]
    for result, expected in cases:
        assert agrees(result, parse(expected)), (result, expected)


def test_known_differences() -> None:
    """Where SymPy leaves a product of a number and a sum undistributed
    (``Add._eval_power``) or nests a product in a product, the core gives
    the form SymPy's own ``Mul`` constructor gives."""
    assert C.Integer(9)/13*(C.Rational(2, 3) + C.I) == C.Rational(6, 13) + C.Rational(9, 13)*C.I
    assert sympy.Rational(9, 13)*(sympy.Rational(2, 3) + sympy.I) == sympy.Rational(6, 13) + sympy.Rational(9, 13)*sympy.I


def _leaf(rng: random.Random) -> Pair:
    r = rng.random()
    if r < 0.5:
        return rng.choice([(sx, cx), (sy, cy), (sz, cz), (sp, cp), (sn, cn), (sympy.pi, C.pi), (sympy.I, C.I)])
    if r < 0.8:
        k = rng.randint(-4, 4)
        return (sympy.Integer(k), C.Integer(k))
    a, b = rng.randint(-6, 6), rng.randint(1, 6)
    return (sympy.Rational(a, b), C.Rational(a, b))


def _build(rng: random.Random, depth: int) -> Pair:
    """A random expression built in SymPy and with the evaluating functions."""
    if depth == 0 or rng.random() < 0.25:
        return _leaf(rng)
    op = rng.choice(['add', 'add', 'mul', 'mul', 'pow', 'neg', 'sub', 'div'])
    if op in ('add', 'mul'):
        parts = [_build(rng, depth - 1) for _ in range(rng.randint(2, 3))]
        if op == 'add':
            return (sympy_expr(sympy.Add(*[a for a, _ in parts])), C.add(*[b for _, b in parts]))
        return (sympy_expr(sympy.Mul(*[a for a, _ in parts])), C.mul(*[b for _, b in parts]))
    if op == 'pow':
        base = _build(rng, depth - 1)
        r = rng.random()
        if r < 0.5:
            k = rng.randint(-2, 3)
            exponent: Pair = (sympy.Integer(k), C.Integer(k))
        elif r < 0.8:
            q, m = rng.choice([2, 3]), rng.choice([1, -1, 2])
            exponent = (sympy.Rational(m, q), C.Rational(m, q))
        else:
            exponent = _leaf(rng)
        return (sympy_expr(sympy.Pow(base[0], exponent[0])), C.pow(base[1], exponent[1]))
    if op == 'neg':
        a = _build(rng, depth - 1)
        return (-a[0], C.neg(a[1]))
    a, b = _build(rng, depth - 1), _build(rng, depth - 1)
    if op == 'sub':
        return (a[0] - b[0], C.sub(a[1], b[1]))
    return (a[0]/b[0], C.div(a[1], b[1]))


def _samples(seed: int, count: int) -> list[Pair]:
    rng = random.Random(seed)
    samples: list[Pair] = []
    while len(samples) < count:
        try:
            samples.append(_build(rng, 3))
        except (ZeroDivisionError, ValueError, TypeError):
            continue  # SymPy refused the expression (0**0 in a float, ...)
    return samples


def test_random_expressions_evaluate_as_in_sympy() -> None:
    for s, c in _samples(2026, 150):
        if not _finite(s):
            continue  # the infinities are compared in test_known_evaluations
        assert agrees(c, s), (s, c)


def _finite(e: sympy.Basic) -> bool:
    return not e.has(sympy.zoo, sympy.nan, sympy.oo)


def test_random_derivatives_expansions_and_substitutions_agree() -> None:
    operations: list[tuple[Callable[[sympy.Expr], sympy.Expr], Callable[[C.Expr], C.Expr]]] = [
        (lambda s: sympy.diff(s, sx), lambda c: C.diff(c, cx)),
        (lambda s: sympy.expand(s), lambda c: C.expand(c)),
        (lambda s: s.subs(sx, sy + 1), lambda c: c.subs(cx, cy + 1)),
    ]
    for s, c in _samples(7, 60):
        if not _finite(s) or not isinstance(s, sympy.Expr) or not isinstance(c, C.Expr):
            continue
        for sympy_op, core_op in operations:
            rs, rc = sympy_op(s), core_op(c)
            difference = sympy.expand(rs - to_sympy(rc))
            assert difference == 0, (s, rs, rc)


def test_random_numerical_values_agree() -> None:
    values: dict[sympy.Basic | complex, sympy.Expr | complex] = {sx: 2, sy: 3, sz: 5, sp: 7, sn: 11}
    core_values: dict[C.Basic, int] = {cx: 2, cy: 3, cz: 5, cp: 7, cn: 11}
    for s, c in _samples(11, 80):
        if not isinstance(s, sympy.Expr) or not isinstance(c, C.Expr):
            continue
        vs = s.subs(values).evalf(15)
        if not vs.is_number or not _finite(vs):
            continue
        vc = c.subs(core_values).evalf(15)
        expected, got = complex(vs), complex(sympy_expr(to_sympy(vc)))
        assert abs(expected - got) <= 1e-9*(1 + abs(expected)), (s, vs, vc)


def test_printing_agrees_with_sympy() -> None:
    """The printed forms SymPy gives for the usual shapes of expressions."""
    for text in ['x**2 + 2*x + 1', 'x*y + x + y', 'x + y**2', '-x*y/(2*z)', '(x + 1)/(x + 2)',
                 '1/(x*y)', 'x**(-2)', '1/sqrt(x)', 'x - 1/x', 'sqrt(x) + x', '2*x/3', '-x/2',
                 '1 - x', 'x - 1', '-x + y + 1', '1 - x**2', '(x**2)**y', 'x**(y + 1)', '2**x*3**y',
                 '(1/2)**x', '(-1)**(1/3)', '2*sqrt(2)*I', 'x + 1 + pi', 'x + I', 'x**x', '(x + 1)**2',
                 '2**(-x)', 'x**n*x**(n + 1)', '(x + 1)**n/(x + 1)', 'sqrt(2)*sqrt(x)', '2*sqrt(-x)',
                 'sqrt(p)*sqrt(x)', '-(-1)**(1/3)', '(-3)**(1/3)', '(2 - I)/5', 'oo*x', 'x + oo',
                 'zoo*(x + I)', '2.5*x', 'x**5.0', 'n**2 + 1', '(x**2 + 1)*(y + 2)', 'x*(x + y)',
                 '(x + 1)*(x + 2)*(x + 3)', '-2*x + 3', '3 - 2*x', 'x/y', 'x*y/z', '1/(2*x)', 'x/2 + 1/3',
                 'sqrt(x)/2', '2/sqrt(x)', 'x**(3/2)', 'x**(-3/2)', 'x**(2/3)', '(x + 1)**(-2)',
                 'exp(x) + sin(x)', 'x*sin(x) + cos(x)', 'log(x)**2', 'f(x) + g(x, y)', 'Derivative(f(x), x)']:
        s = parse(text)
        assert str(from_sympy(s)) == str(s), (text, str(s), str(from_sympy(s)))


def test_round_trip_with_sympy() -> None:
    for s, c in _samples(3, 60):
        assert to_sympy(from_sympy(s)) == s, s
        assert from_sympy(to_sympy(c)) == C.add(c, 0) if isinstance(c, C.Expr) else True
    for text in ['sin(x)**2 + cos(x)**2', 'exp(x) + log(x) + gamma(x)', 'f(x, y) + Derivative(f(x), x, y)',
                 'Lambda(x, x**2)', 'Eq(x, 1)', 'x < y', 'Tuple(x, 1)', 'Float("2.5", 30) + 1',
                 'Symbol("q", positive=True, integer=True)', 'Symbol("A", commutative=False)*x']:
        from sympy.parsing.sympy_parser import parse_expr
        parsed = parse_expr(text, local_dict=dict(LOCALS, f=sympy.Function('f')))
        if not isinstance(parsed, sympy.Basic):
            raise TypeError("a SymPy object is expected, got %r" % (parsed,))
        tree = from_sympy(parsed)
        assert to_sympy(tree) == parsed, (parsed, tree)
        assert from_sympy(to_sympy(tree)) == tree, (parsed, tree)
    # SymPy's functions are evaluated by SymPy on doit, differentiated by the chain rule
    sin = C.SymPyFunction(sympy.sin)
    assert sin(0) == C.AppliedSymPyFunction(sympy.sin, 0) and sin(0).doit() == 0
    assert to_sympy(C.diff(sin(cx**2), cx)) == 2*sx*sympy.cos(sx**2)
    assert C.as_expr(C.sympify('sin(x)')).is_real is None and C.SymPyFunction(sympy.exp)(cp).is_positive is True


def test_symbols_with_assumptions_round_trip() -> None:
    for assumptions in [{'positive': True}, {'integer': True}, {'real': True}, {'even': True},
                        {'negative': True, 'integer': True}, {'nonzero': True}, {'commutative': False},
                        {'zero': True}, {'prime': True}, {'rational': True}, {'imaginary': True}]:
        s = sympy.Symbol('q', **assumptions)
        c = C.Symbol('q', **assumptions)
        assert from_sympy(s) == c and to_sympy(c) == s, assumptions
        for name in ['positive', 'negative', 'zero', 'nonzero', 'integer', 'rational', 'real', 'even', 'odd',
                     'prime', 'finite', 'commutative']:
            assert c._ask(name) == getattr_(s, name), (assumptions, name)


def getattr_(s: sympy.Basic, name: str) -> bool | None:
    """SymPy's ``is_<name>`` of an expression, read without ``getattr``."""
    queries: dict[str, Callable[[sympy.Basic], bool | None]] = {
        'positive': lambda e: e.is_positive, 'negative': lambda e: e.is_negative, 'zero': lambda e: e.is_zero,
        'nonzero': lambda e: e.is_nonzero, 'integer': lambda e: e.is_integer, 'rational': lambda e: e.is_rational,
        'real': lambda e: e.is_real, 'even': lambda e: e.is_even, 'odd': lambda e: e.is_odd,
        'prime': lambda e: e.is_prime, 'finite': lambda e: e.is_finite, 'commutative': lambda e: e.is_commutative,
        'irrational': lambda e: e.is_irrational, 'imaginary': lambda e: e.is_imaginary, 'complex': lambda e: e.is_complex,
        'infinite': lambda e: e.is_infinite, 'nonnegative': lambda e: e.is_nonnegative,
        'nonpositive': lambda e: e.is_nonpositive, 'extended_real': lambda e: e.is_extended_real,
        'extended_positive': lambda e: e.is_extended_positive, 'extended_negative': lambda e: e.is_extended_negative,
        'extended_nonnegative': lambda e: e.is_extended_nonnegative,
        'extended_nonpositive': lambda e: e.is_extended_nonpositive, 'extended_nonzero': lambda e: e.is_extended_nonzero,
        'hermitian': lambda e: e.is_hermitian, 'antihermitian': lambda e: e.is_antihermitian,
        'algebraic': lambda e: e.is_algebraic, 'transcendental': lambda e: e.is_transcendental,
        'composite': lambda e: e.is_composite, 'noninteger': lambda e: e.is_noninteger, 'polar': lambda e: e.is_polar,
    }
    return queries[name](s)


# ----------------------------------------------------------------------
# the rest of sympy.core, against SymPy

_sa = importlib.import_module('sympy.core.assumptions')


def _cv(text: str) -> C.Expr:
    """The expression of this core parsed by SymPy from ``text``."""
    return C.as_expr(from_sympy(parse(text)))


def _match_pairs(got: dict[C.Basic, C.Basic] | None, expected: dict[sympy.Basic, sympy.Basic] | None) -> bool:
    if got is None or expected is None:
        return got is None and expected is None
    by_name = {str(k): v for k, v in expected.items()}
    if {str(k) for k in got} != set(by_name):
        return False
    return all(sympy.simplify(sympy_expr(to_sympy(v)) - sympy_expr(by_name[str(k)])) == 0 for k, v in got.items())


def test_known_matches() -> None:
    """``match`` on the cases of SymPy's behaviour (the ambiguous cases
    where SymPy's search order gives an odd binding are left out)."""
    a, b, c = sympy.Wild('a'), sympy.Wild('b'), sympy.Wild('c')
    cases = ['x|a', 'x|a*x', '2*x|a*x', 'x*y*z|a*x', 'x + y + 1|a + x', 'x|a + x', '2*x + 3|a*x + b', 'x + 1|a*x + b',
             'x**2|x**a', 'x|x**a', 'x|a**b', 'x**2|a*x**b', '2|a*x', 'sin(x)|sin(a)', 'sin(x)|a', 'sin(x)|cos(a)',
             '2*sin(x)|a*sin(b)', 'x*y|a*b', 'x + y|a + b', 'x**y|a**b', '3|a', '3*x|a*x', 'x/2|a*x', '2/x|a/x', '2/x|a*x',
             'x**3|a*x**2', 'x**3*y|x**a*y**b', 'sqrt(x)|x**a', 'exp(2*x)|exp(a*x)', 'exp(x)|exp(a*x)', 'x*y + 1|a*x + b',
             'x*y + 1|a*x*y + b', '-x|a*x', '-x|a', 'x - 1|a*x + b', '2*x*y|a*x*y', 'x*y*z|a*b*z', 'f(x)|f(a)',
             'f(x, y)|f(a, b)', 'f(x)|g(a)', 'x + 2|x + a', 'x**2|a*x', '5|a*x + b', 'x|a*x + b', '2*x|a*x + b',
             'x*(x + 1)|a*(x + b)', '(x + 1)**2|(x + a)**b', 'x + y + z|a + b', 'x*y*z|a*b', 'x|a + b', '1|a + b',
             '2*x + 3*y|a*x + b*y', '2*x + 3*y|a*x + b', 'I*x|a*x', 'pi*x|a*x', 'x*exp(x)|a*exp(b*x)', 'x*y|x*a',
             'x**2*y|a*x**b', 'x*y + 2*x|a + x', 'y|a + x', 'y|a*x', 'sqrt(x)|a*x', 'x**y|a*x', '2*x**3|a*x**2',
             'x**3*y|a*x**2', '2*x|a*x**2', 'x**-3|a*x**-2', 'x**-3|a/x', 'x**2|a*x**-1', '2*x*y|a*x**2', 'pi|a*x',
             'x + 1|a*x', 'sin(x)|a*x', 'x*y + 2*x|a*x', '0|a + x', '2|a + x', 'x*y|a + x', 'x**2|a + x', 'x**2 + x|a*x',
             'x|a*b', 'x**2|a*b', 'x|a*x*b', 'x*y|a*x*y*b', 'x*y|a**2*x', '4*x**2|a*b**2', 'x + 1|a + 1', 'sqrt(x)|sqrt(a)',
             'x*y|a*y', 'x/y|a/y', 'x/y|a/b', 'x**-2|a**-2', '2*x**2|a*x**2', '0|a', '0|a*x', 'x + y|a - b', 'x - y|a - b',
             '-x|-a', 'x|-a', '2*x|-a', 'x**2|a**2', 'x**4|a**2', '4*x**2|a**2', 'x**2*y**2|a**2', '4|a**2', 'x**2|x**(2*a)',
             'x**4|x**(2*a)', 'x**2*y|x**(2*a)']
    for case in cases:
        subject, pattern = case.split('|')
        s = parse(subject)
        from sympy.parsing.sympy_parser import parse_expr
        p = parse_expr(pattern, local_dict=dict(LOCALS, a=a, b=b, c=c, f=sympy.Function('f'), g=sympy.Function('g')))
        if not isinstance(p, sympy.Basic):
            raise TypeError("a pattern is expected, got %r" % (p,))
        assert _match_pairs(from_sympy(s).match(from_sympy(p)), s.match(p)), case
    sx_excluded = [sympy.Wild('a', exclude=[sx]), sympy.Wild('b', exclude=[sx])]
    for subject in ['2*x + 3', 'x**2 + 3', 'x*y', 'x**2', 'x', '3']:
        p = sx_excluded[0]*sx + sx_excluded[1]
        assert _match_pairs(from_sympy(parse(subject)).match(from_sympy(p)), parse(subject).match(p)), subject
    assert _match_pairs(_cv('x*y + 2*x').replace(from_sympy(a*sx), from_sympy(a)).match(from_sympy(a)),
                        (sx*sy + 2*sx).replace(a*sx, a).match(a))
    assert {to_sympy(e) for e in _cv('x*y + 2*x').find(from_sympy(a*sx))} == (sx*sy + 2*sx).find(a*sx)
    assert _cv('x*y + 2*x').count(from_sympy(a*sx)) == (sx*sy + 2*sx).count(a*sx)


def test_known_mod() -> None:
    """``mod`` on the cases of SymPy's ``Mod`` (``Mod(n*(n + 1), 2)`` is left out)."""
    cases = ['Mod(7, 3)', 'Mod(-7, 3)', 'Mod(7, -3)', 'Mod(Rational(7, 2), 2)', 'Mod(x, 1)', 'Mod(n, 1)', 'Mod(x, x)',
             'Mod(2*n, 2)', 'Mod(4*n, 2)', 'Mod(n**2, n)', 'Mod(0, x)', 'Mod(Mod(x, 6), 3)', 'Mod(Mod(x, 3), 6)',
             'Mod(x + 3, 3)', 'Mod(2*x + 6, 2)', 'Mod(-x, 3)', 'Mod(x, -3)', 'Mod(p, p)', 'Mod(6, 4)', 'Mod(n, n)',
             'Mod(3*n, 3)', 'Mod(x*3, 3)', 'Mod(6*x, 3)', 'Mod(-n, 3)', 'Mod(I, 2)', 'Mod(1 + 2*n, 2)', 'Mod(n + 1, n)',
             'Mod(x**2, x)', 'Mod(Rational(-7, 2), 2)', 'Mod(7, Rational(3, 2))', 'Mod(x*y, y)', 'Mod(x*y, x)',
             'Mod(2*x*y, x)', 'Mod(x**2*y, x)', 'Mod(n**2 + n, n)', 'Mod(2*n + 2, 2)', 'Mod(n**3, n)', 'Mod(3*n**2, n)',
             'Mod(Mod(n, 4), 2)', 'Mod(Mod(n, 2), 4)', 'Mod(Mod(n, 3), 2)', 'Mod(3*x, 3)', 'Mod(x, 2*x)', 'Mod(-x, x)',
             'Mod(3*x, x)', 'Mod(x + y + 3, 3)', 'Mod(2*x + 3, 2)', 'Mod(x - 3, 3)', 'Mod(x + Rational(7, 2), Rational(3, 2))',
             'Mod(x + 3*y, 3)', 'Mod(n + 4, 2)', 'Mod(2*n + 4, 4)', 'Mod(4*n + 6, 4)', 'Mod(-4, 3)', 'Mod(4, -3)',
             'Mod(-4, -3)', 'Mod(3, 5)', 'Mod(0, 3)', 'Mod(3, 3)', 'Mod(x, 3) + Mod(x, 3)', '2*Mod(x, 3)', 'Mod(x, 3)**2',
             '-Mod(x, 3)', 'Mod(x, 3) + 1']
    for text in cases:
        s = parse(text)
        unevaluated = from_sympy(s)
        core = _cv(text).doit() if text.startswith('Mod(') else _cv(text)
        assert agrees(core, s) and str(core) == str(s), (text, core, s)
        assert to_sympy(unevaluated) == s
    for text, name in [('Mod(x, 3)', 'integer'), ('Mod(n, 3)', 'integer'), ('Mod(x, 3)', 'nonnegative'),
                       ('Mod(x, -3)', 'nonpositive'), ('Mod(x, 3)', 'real'), ('Mod(n, 3)', 'finite')]:
        assert _cv(text)._ask(name) == getattr_(parse(text), name), (text, name)


def test_known_count_ops() -> None:
    for text in ['x**2 + 2*x + 1', 'sin(x)*2', 'x', '2', 'x/y', 'x - y', '-x', 'x**-1', 'sqrt(x)', 'Eq(x, 1)',
                 'sin(x) + cos(x)', 'x + y + z', 'x**2 + x**3', 'x*y*z', '2*x', 'x/2', 'x**-2', 'x < y', '2*x*y + 3*x + 1',
                 'x*(x + 1)', 'x**Rational(3, 2)', '2/x', 'x**y', '-2*x', '-x*y', 'Rational(-1, 2)*x', 'Integer(-2)',
                 'Rational(1, 2)', 'pi + I', 'Float(2.5)', 'x - 1', '1 - x', 'Mod(x, 3)', 'Subs(f(x), x, 2)', 'Order(x)',
                 'f(x)', 'Derivative(f(x), x)', 'exp(x)*sin(x)', 'x**2*y**2', '1/(x*y)', 'x/(2*y)', '-x/2 + 1', 'x**(-x)',
                 'x**(x - 1)']:
        s = parse_expr_any(text)
        assert (str(C.count_ops(from_sympy(s))), str(C.count_ops(from_sympy(s), visual=True))) == \
            (str(sympy.count_ops(s)), str(sympy.count_ops(s, visual=True))), text


_NAMES = ['extended_positive', 'extended_negative', 'extended_nonnegative', 'extended_nonpositive', 'extended_nonzero',
          'hermitian', 'antihermitian', 'algebraic', 'transcendental', 'composite', 'noninteger', 'polar', 'positive',
          'negative', 'zero', 'nonzero', 'integer', 'rational', 'irrational', 'real', 'imaginary', 'complex', 'even',
          'odd', 'prime', 'finite', 'infinite', 'nonnegative', 'nonpositive', 'extended_real', 'commutative']


def test_known_closures_of_every_predicate() -> None:
    """The closure of each predicate, given true or false, is SymPy's,
    and the symbols round trip."""
    for name in _NAMES:
        for value in (True, False):
            s = sympy.Symbol('q', **{name: value})
            c = C.Symbol('q', **{name: value})
            got = {k: v for k, v in c.assumptions0.items() if v is not None}
            expected = {str(k): v for k, v in s.assumptions0.items() if v is not None}
            assert got == expected, (name, value)
            assert from_sympy(s) == c and to_sympy(c) == s, (name, value)
            for query in _NAMES:
                if query == 'odd' and name in ('nonzero', 'extended_nonzero') and not value:
                    continue
                assert c._ask(query) == getattr_(s, query), (name, value, query)


def test_known_facts_of_numbers_and_compounds() -> None:
    """Every fact this core states is SymPy's; what it leaves unknown is
    listed when SymPy knows it (a sound subset)."""
    exprs = ['0', '1', '2', '4', '-3', 'Rational(1, 2)', 'Float(2.5)', 'pi', 'E', 'I', 'oo', '-oo', 'zoo', 'nan',
             'GoldenRatio', 'EulerGamma', 'Catalan', 'TribonacciConstant', 'sqrt(2)', '2 + I', 'n + 1', '2*n', 'p + 1',
             'x + Rational(1, 2)', 'n + Rational(1, 2)', 'sqrt(2) + 1', 'sqrt(2)*pi', 'x + 1', 'pi**2', 'pi + 1', '2*pi',
             'sqrt(pi)', 'pi*E', 'n*pi', 'x*pi', 'x + pi', '2**Rational(1, 3) + pi', 'p + oo', 'oo*p', 'x**2',
             '2*Symbol("r", real=True)', 'I*Symbol("r", real=True)', 'Symbol("h", hermitian=True) + 1',
             'Symbol("q", algebraic=True)**2', 'Symbol("q", algebraic=True)**x', 'Symbol("q", algebraic=True)**Rational(1, 3)',
             'Symbol("q", algebraic=True, nonzero=True)*pi', 'Symbol("q", rational=True) + pi',
             'Symbol("r", extended_real=True) + 1', 'Symbol("r", extended_real=True)*2',
             'Symbol("r", extended_positive=True)**2', 'Symbol("r", extended_positive=True)**-1',
             'Symbol("q", polar=True)*Symbol("w", polar=True)', 'Symbol("q", polar=True)**2', '2*Symbol("q", polar=True)',
             'x*Symbol("q", polar=True)', 'Mod(n, 3)', 'Symbol("c", composite=True)', 'Abs(x)', 'exp(x)', 'exp(p)',
             'log(p)', 'sin(x)']
    unknown_here: list[tuple[str, str]] = []
    for text in exprs:
        s = parse(text)
        c = C.as_expr(from_sympy(s))
        for name in _NAMES:
            got, expected = c._ask(name), getattr_(s, name)
            if got is not None:
                assert got == expected, (text, name, got, expected)
            elif expected is not None:
                unknown_here.append((text, name))
    assert not unknown_here, unknown_here


def test_known_decompositions() -> None:
    pairs = [('3*x**2', 'x'), ('3*x', 'x'), ('x**2 + x', 'x'), ('3', 'x'), ('3*x**2*y', 'x'), ('3/x', 'x'), ('x**y', 'x'),
             ('sqrt(x)*3', 'x')]
    for text, var in pairs:
        assert tuple(map(to_sympy, _cv(text).as_coeff_exponent(_cv(var)))) == parse(text).as_coeff_exponent(parse(var))
    for text, other in [('3*x*y', 'x'), ('3*x*y', 'y'), ('3*x*y', 'z'), ('2*E*x', 'E'), ('x + y', 'x'), ('2*x*y', 'x*y'),
                        ('x**2', 'x'), ('2*x', '2'), ('2*x', 'x + 1'), ('2*x*y', '2*x')]:
        got = _cv(text).as_coefficient(_cv(other))
        assert (None if got is None else to_sympy(got)) == parse(text).as_coefficient(parse(other)), (text, other)
    for text in ['2*x + 4*y', 'x/2 + y/3', '6*x', '-x - y', 'sqrt(8)', 'x**2', '-2*x', '6', '2*x + 1', '2*(x + 1)**2',
                 '-2*x - 4', '(2*x + 4)**2', 'x', '-6']:
        assert tuple(map(to_sympy, _cv(text).as_content_primitive())) == parse(text).as_content_primitive(), text
    for text, other in [('2*x + 3', 'x'), ('2*x + 3', '3'), ('2*x + 3', '5'), ('2*x + 3', 'x + 1'), ('2*x + 3', 'y'),
                        ('2*x', 'x'), ('x + y', 'x'), ('5', '2'), ('5', '7'), ('-x', 'x'), ('x', '0'), ('x', 'x')]:
        got = _cv(text).extract_additively(_cv(other))
        assert (None if got is None else to_sympy(got)) == parse(text).extract_additively(parse(other)), (text, other)
    for text, other in [('6*x*y', '2'), ('6*x*y', '2*x'), ('6*x*y', 'z'), ('6*x*y', '4'), ('x**2', 'x'), ('x**3', 'x**2'),
                        ('2*x + 4*y', '2'), ('2*x + 3*y', '2'), ('-x', '-1'), ('x', '-1'), ('6', '4'), ('6', 'Rational(1, 2)'),
                        ('sqrt(x)', 'x'), ('x*y', 'x'), ('2*x', 'x'), ('x', 'x'), ('3*x', '3*x'), ('-2*x', '2'), ('x/2', '2'),
                        ('Rational(3, 2)', '3'), ('-6', '2'), ('6', '-2'), ('x**-2', 'x**-1')]:
        got = _cv(text).extract_multiplicatively(_cv(other))
        assert (None if got is None else to_sympy(got)) == parse(text).extract_multiplicatively(parse(other)), (text, other)
    for text, other in [('x + 1', 'x + 1'), ('(x + 1)**2', 'x**2 + 2*x + 1'), ('x + 1', 'x + 2'), ('sqrt(2)*sqrt(3)', 'sqrt(6)'),
                        ('x', 'y'), ('sin(x)**2 + cos(x)**2', '1'), ('2', '2')]:
        assert _cv(text).equals(_cv(other)) == parse(text).equals(parse(other)), (text, other)
    for text in ['x**2 + 1', 'x + 1/y', 'sqrt(x)', 'x**n', '2**x', 'x**y', '2', 'x**-1', 'x**Rational(1, 2) + y**-1']:
        s, c = parse(text), _cv(text)
        assert (c.is_polynomial(), c.is_polynomial(cx), c.is_rational_function(cx), c.is_algebraic_expr(cx)) == \
            (s.is_polynomial(), s.is_polynomial(sx), s.is_rational_function(sx), s.is_algebraic_expr(sx)), text
    for text in ['x + 1', '2', 'x - x', 'pi']:
        assert _cv(text).is_constant() == parse(text).is_constant()
    for text in ['x', 'pi', 'pi + x', '2 + I', 'oo', '2', 'sqrt(2)', 'nan']:
        assert _cv(text).is_comparable == parse(text).is_comparable, text
    for text, digits in [('Float(2.5)', None), ('Float(3.14159)', 2), ('pi', 3), ('Float(2.5) + 3*I', None), ('3', None),
                         ('7', -1), ('Float(1.5)', None), ('Rational(7, 2)', None), ('Float(2.49)', None), ('Float(-2.5)', None)]:
        assert to_sympy(_cv(text).round(digits)) == parse(text).round(digits), text
    for text in ['x + 2*I', '2 + 3*I', 'I', 'x', 'Symbol("r", real=True)', 'x*I', '1/x', 'x**2']:
        assert tuple(map(to_sympy, _cv(text).as_real_imag())) == parse(text).as_real_imag(), text
    for text in ['x + 2*I', 'x', '2*x*I', 'x**2', 'Symbol("r", real=True)', 'sin(x)', 'pi', 'I']:
        assert to_sympy(_cv(text).conjugate()) == parse(text).conjugate(), text
    for text in ['x', 'Symbol("A", commutative=False)', 'Symbol("A", commutative=False)*x', 'I']:
        assert to_sympy(_cv(text).adjoint()) == parse(text).adjoint() and to_sympy(_cv(text).transpose()) == parse(text).transpose()
    for text in ['x**2', '2*x*y + 3']:
        assert [(str(t), r) for t, r in _cv(text).as_terms()[0]] == [(str(t), r) for t, r in parse(text).as_terms()[0]]
    assert [str(_cv(t).normal()) for t in ['2*x + 1', 'x/y + 1']] == [str(parse(t).normal()) for t in ['2*x + 1', 'x/y + 1']]
    for text in ['x + y', 'Lambda(x, x + y)']:
        assert str(from_sympy(parse_expr_any(text)).as_dummy()) == str(parse_expr_any(text).as_dummy())
    lam = [from_sympy(parse_expr_any(t)) for t in ['Lambda(x, x)', 'Lambda(y, y)', 'Lambda(y, y + 1)']]
    assert lam[0].dummy_eq(lam[1]) and not lam[0].dummy_eq(lam[2])
    assert [_cv('x + y').has_free(cx), _cv('x + y + z').has_free(cx + cy), _cv('x + y + z').has_xfree({cx + cy})] == \
        [(sx + sy).has_free(sx), (sx + sy + sz).has_free(sx + sy), (sx + sy + sz).has_xfree({sx + sy})]


def test_known_forwarded_methods() -> None:
    """The methods which go through SymPy give SymPy's results, and the
    expansions with hints and the exponential rules agree."""
    for text, n in [('sin(x)', 5), ('exp(x)', 3), ('1/(1 - x)', 3), ('cos(x)', 4), ('log(1 + x)', 4)]:
        assert str(_cv(text).series(cx, 0, n)) == str(parse(text).series(sx, 0, n)), text
    assert to_sympy(_cv('exp(x)').series(cx, 0, 3).removeO()) == sympy.exp(sx).series(sx, 0, 3).removeO()
    order_term = _cv('exp(x)').series(cx, 0, 3).getO()
    assert order_term is not None and to_sympy(order_term) == sympy.exp(sx).series(sx, 0, 3).getO()
    for text, point, direction in [('sin(x)/x', '0', '+'), ('1/x', '0', '-'), ('1/x', 'oo', '+'), ('(1 + 1/x)**x', 'oo', '+')]:
        assert to_sympy(_cv(text).limit(cx, _cv(point), direction)) == parse(text).limit(sx, parse(point), direction)
    checks: list[tuple[C.Basic, sympy.Basic]] = [
        (_cv('sin(x)**2 + cos(x)**2').simplify(), sympy.simplify(parse('sin(x)**2 + cos(x)**2'))),
        (_cv('x**2 + 2*x + 1').factor(), sympy.factor(parse('x**2 + 2*x + 1'))),
        (_cv('(x**2 - 1)/(x - 1)').cancel(), sympy.cancel(parse('(x**2 - 1)/(x - 1)'))),
        (_cv('1/(x*(x + 1))').apart(), sympy.apart(parse('1/(x*(x + 1))'))),
        (_cv('x*y + x').collect(cx), sympy.collect(parse('x*y + x'), sx)),
        (_cv('1/x + 1/y').together(), sympy.together(parse('1/x + 1/y'))),
        (_cv('x**2*x**y').powsimp(), sympy.powsimp(parse('x**2*x**y'))),
        (_cv('1/(sqrt(2) + 1)').radsimp(), sympy.radsimp(parse('1/(sqrt(2) + 1)'))),
        (_cv('Float(0.5)').nsimplify(), sympy.nsimplify(0.5)),
        (_cv('x**2').integrate(cx), sympy.integrate(sx**2, sx)),
        (_cv('x**2').integrate((cx, 0, 1)), sympy.integrate(sx**2, (sx, 0, 1))),
        (_cv('x**2 + 1').invert(cx), sympy.invert(sx**2 + 1, sx)),
        (_cv('gamma(x + 1)/gamma(x)').gammasimp(), sympy.gammasimp(parse('gamma(x + 1)/gamma(x)'))),
        (_cv('sin(x)').rewrite(C.SymPyFunction(sympy.exp)), sympy.sin(sx).rewrite(sympy.exp)),
        (_cv('sqrt(x**2)').refine(sympy.Q.positive(sx)), sympy.refine(sympy.sqrt(sx**2), sympy.Q.positive(sx))),
        (_cv('x + x**2').as_leading_term(cx), (sx + sx**2).as_leading_term(sx)),
        (_cv('sin(x)').taylor_term(3, cx), sympy.sin(sx).taylor_term(3, sx)),
        (_cv('exp(x)').nseries(cx, 0, 3), sympy.exp(sx).nseries(sx, 0, 3)),
        (C.Tuple(*islice(_cv('sin(x)').lseries(cx), 2)), sympy.Tuple(*islice(sympy.sin(sx).lseries(sx), 2))),
        (C.factor_terms(_cv('2*x + 4*y')), sympy.factor_terms(parse('2*x + 4*y'))),
        (C.gcd_terms(_cv('x/2 + y/4')), sympy.gcd_terms(parse('x/2 + y/4'))),
        (C.expand_trig(_cv('sin(2*x)')), sympy.expand_trig(parse('sin(2*x)'))),
        (C.expand_complex(_cv('x + I*y')), sympy.expand_complex(parse('x + I*y'))),
        (C.expand_func(_cv('gamma(x + 1)')), sympy.expand_func(parse('gamma(x + 1)'))),
        (C.expand_power_base(_cv('(x*y)**z'), force=True), sympy.expand_power_base(parse('(x*y)**z'), force=True)),
    ]
    for got, expected in checks:
        assert str(got) == str(expected), (got, expected)
        assert to_sympy(got) == expected or sympy.expand(sympy_expr(to_sympy(got)) - sympy_expr(expected)) == 0, (got, expected)
    assert str(_cv('x**2 + 1').as_poly()) == str(parse('x**2 + 1').as_poly())
    assert tuple(map(to_sympy, _cv('x**2 + x**3').leadterm(cx))) == (sx**2 + sx**3).leadterm(sx)
    hints: list[tuple[str, dict[str, bool]]] = [
        ('x*(y + 1)', {'mul': False}), ('(x + 1)**2', {'multinomial': False}), ('x*(y + z)', {}),
        ('2**(x + y)', {'power_exp': False}), ('2**(x + y)', {}), ('exp(x + y)', {}), ('sin(x + y)', {'trig': True}),
        ('(x + I*y)**2', {'complex': True}), ('gamma(x + 1)', {'func': True}), ('(x + 1)**2*(y + 1)', {'deep': False}),
        ('sin((x + 1)**2)', {'deep': False}), ('sin((x + 1)**2)', {}), ('x**(y + z)', {}), ('x**(n + 1)', {})]
    for text, hint in hints:
        assert str(_cv(text).expand(**hint)) == str(parse(text).expand(**hint)), (text, hint)
    for text in ['exp(x)*exp(y)', 'exp(x)*exp(-x)', 'exp(x)**2', 'exp(x)**y', 'exp(x)*exp(2*x)', 'exp(p)**x', 'sqrt(exp(x))',
                 'exp(-x)*2', 'exp(x) - exp(-x)', 'E**x', 'E**2', 'E**0', 'E**(I*pi)', 'E**log(x)', 'exp(x)/exp(y)',
                 'x*exp(x)*exp(-x)', 'exp(x)**-1']:
        assert str(_cv(text)) == str(parse(text)) and agrees(_cv(text), parse(text)), text
    assert str(cx**2*C.E**cx) == str(sx**2*sympy.E**sx) and str(C.E**cx*C.E**(-cx)) == '1' and str((C.E**cx)**2) == 'exp(2*x)'


def test_known_printing_and_round_trips_of_the_new_objects() -> None:
    for text in ['Mod(x, 3)', '2*Mod(x, 3)', 'Mod(x, 3)**2', '-Mod(x, 3)', 'Mod(x, 3) + 1', 'Order(x)', 'Order(x, (x, oo))',
                 'Order(x*y)', 'x + Order(x**2)', 'Subs(f(x), x, 2)', 'Subs(x**2 + y, (x, y), (1, 2))', 'Dict({x: 1, y: 2})',
                 'Wild("a")', 'WildFunction("F")', 'UnevaluatedExpr(x)*x', 'UnevaluatedExpr(2*x) + 1',
                 'EulerGamma + Catalan + GoldenRatio + TribonacciConstant', 'sin(x)*cos(x)', 'exp(x) - exp(-x)',
                 'x + exp(x) + log(x) + sin(x)', 're(x) + im(x)', 'Derivative(f(x), x) + f(x)', 'Integral(f(x), (x, 0, 1))',
                 'Piecewise((x, x > 0), (0, True))', 'Abs(x) + floor(x)', 'x**(2*y) + x**y', 'AlgebraicNumber(sqrt(2), [1, 1])',
                 'Order(1, (x, 1))', 'O(x**2 + y)', 'x**2 - x + O(x**3)', '2 + x + x**3 + O(x**5)', '1 + pi + I',
                 'E + pi + 1 + I + GoldenRatio', 'x + pi + I + 2', 'sqrt(2) + 1 + pi + E', 'x - 2*pi + 1', '-pi + I',
                 '3*pi + 2*I', '1 - pi', '-1 + pi', '-x - 2', 'x**2*y**2', '(2 - I)/5', '1/(-1 + I/4)', 'exp(I*x) - exp(-I*x)',
                 'sin(x) + sin(x - 1)', '-sin(x) + sin(x + 1)']:
        s = parse_expr_any(text)
        tree = from_sympy(s)
        assert str(tree) == str(s), (text, str(tree), str(s))
        # to_sympy evaluates: SymPy distributes 16/17 over -1 - I/4 when rebuilding
        assert same_structure(to_sympy(tree, evaluate=False), s) or (isinstance(tree, C.Expr) and agrees(tree, s)), text
        assert from_sympy(to_sympy(tree, evaluate=False)) == tree, text
    for text in ['And(x > 0, x < 1)', 'Sum(x, (x, 1, 10))', 'Interval(0, 1)', 'Function("h", real=True)(x)']:
        parsed = parse_expr_any(text)
        tree = from_sympy(parsed)
        assert to_sympy(tree) == parsed and from_sympy(to_sympy(tree)) == tree, text
    assert C.as_expr(from_sympy(sympy.Function('f')(2*sx).diff(sx))).dummy_eq(_cv('f(2*x)').diff(cx))
    for text in ['Subs(f(x), x, 2)', 'Subs(x**2 + y, (x, y), (1, 2))', 'Subs(Derivative(f(x), x), x, 2)',
                 'Subs(Derivative(f(x), x), x, y)', 'Subs(x, x, 2)', 'Integral(x**2, (x, 0, 1))', 'Sum(x, (x, 1, 10))',
                 'Mod(7, 3)', 'Order(x + x**2)', 'UnevaluatedExpr(x)**2']:
        assert str(from_sympy(parse(text)).doit()) == str(parse(text).doit()), text
    for text in ['f(2*x)', 'f(x**2)', 'f(x, 2*x)', 'Subs(f(x), x, y)', 'Mod(x, 3)', 'UnevaluatedExpr(x)', 'Order(x)']:
        assert str(_cv(text).diff(cx)) == str(parse(text).diff(sx)), text
    for text in ['EulerGamma', 'Catalan', 'GoldenRatio', 'TribonacciConstant', 'Mod(7, 3)', 'AlgebraicNumber(sqrt(2), [1, 1])',
                 'Integral(x**2, (x, 0, 1))']:
        assert str(C.as_expr(from_sympy(parse(text))).evalf()) == str(parse(text).evalf()), text
    assert str(C.N(C.pi, 20)) == str(sympy.N(sympy.pi, 20)) and str(C.N(cx + C.pi, 5)) == str(sympy.N(sx + sympy.pi, 5))
    assert str(C.nfloat(cx**2 + C.Rational(1, 3))) == str(sympy.nfloat(sx**2 + sympy.Rational(1, 3)))
    assert str(C.nfloat(C.sqrt(2)*cx, 3)) == str(sympy.nfloat(sympy.sqrt(2)*sx, 3))
    from sympy.core.numbers import comp as sympy_comp
    for z1, z2, tol in [(C.pi, 3.1416, None), (C.pi, 3.1416, 1e-3), (C.Float(1.0), 1.1, None), (C.Float(1.0), 1.05, 0.1)]:
        assert C.comp(z1, z2, tol) == sympy_comp(to_sympy(z1), z2, tol)
    for text in ['Order(x) + x', 'x + Order(x**2)', 'Order(x) + Order(x**2)', 'Order(x)*x', '2*Order(x)', 'Order(x)*Order(x)',
                 'Order(2*x)', 'Order(x**2 + y)', 'Order(x)**2', 'Order(x) + y']:
        assert str(_cv(text)) == str(parse(text)), text
    assert C.Order(cx).contains(cx**2) is True and C.Order(cx**2).contains(cx) is False
    assert str(C.Rel(cx, 1, '<')) == str(sympy.Rel(sx, 1, '<')) and str(C.Lt(cx, 1).negated) == str(sympy.Lt(sx, 1).negated)
    assert str(C.Gt(cx, 1).canonical) == str(sympy.Gt(sx, 1).canonical)


def parse_expr_any(text: str) -> sympy.Basic:
    from sympy.parsing.sympy_parser import parse_expr
    e = parse_expr(text, local_dict=dict(LOCALS, f=sympy.Function('f')))
    if not isinstance(e, sympy.Basic):
        raise TypeError("a SymPy object is expected, got %r" % (e,))
    return e


def test_known_helpers() -> None:
    intfunc = importlib.import_module('sympy.core.intfunc')
    assert (C.igcd(12, 18, 30), C.ilcm(4, 6), C.igcdex(10, 6), C.mod_inverse(3, 11), C.mod_inverse(-3, 11)) == \
        (sympy.igcd(12, 18, 30), sympy.ilcm(4, 6), intfunc.igcdex(10, 6), sympy.mod_inverse(3, 11), sympy.mod_inverse(-3, 11))
    for value, n in [(16, 2), (17, 2), (27, 3), (10**30 + 1, 7), (0, 2), (1, 5), (2**64, 4)]:
        assert C.integer_nthroot(value, n) == sympy.integer_nthroot(value, n)
    for value, base in [(100, 10), (99, 10), (1, 2), (7, 2), (3**20, 3)]:
        assert C.integer_log(value, base) == sympy.integer_log(value, base)
    assert (C.num_digits(1234), C.num_digits(255, 2), C.trailing(8), C.trailing(12), C.trailing(0), C.isqrt(17)) == \
        (intfunc.num_digits(1234), intfunc.num_digits(255, 2), intfunc.trailing(8), intfunc.trailing(12), intfunc.trailing(0), intfunc.isqrt(17))
    assert C.check_assumptions(cp, positive=True) == _sa.check_assumptions(sp, positive=True)
    assert C.check_assumptions(cx, positive=True) == _sa.check_assumptions(sx, positive=True)
    assert C.check_assumptions(C.Integer(-2), positive=True) == _sa.check_assumptions(sympy.S(-2), positive=True)
    assert C.check_assumptions(C.Integer(2), cn) == _sa.check_assumptions(sympy.S(2), sn)
    assert C.failing_assumptions(C.Integer(-2), positive=True, integer=True) == _sa.failing_assumptions(sympy.S(-2), positive=True, integer=True)
    assert C.common_assumptions([C.Integer(2), cp], check=['positive', 'zero']) == _sa.common_assumptions([sympy.S(2), sp], check=['positive', 'zero'])
    assert C.assumptions(cx) == _sa.assumptions(sx) and C.assumptions(cp, ['positive', 'real']) == _sa.assumptions(sp, ['positive', 'real'])
    for text in ['x', 'Symbol("A", commutative=False)', 'x + 1', 'Eq(x, 1)', '2', 'pi', 'f(x)', 'x < 1', 'Derivative(x**2, x)']:
        assert str(from_sympy(parse_expr_any(text)).kind) == str(parse_expr_any(text).kind), text


# ----------------------------------------------------------------------
# the logic, the new assumptions, the order of the arguments and the
# native algorithms, against SymPy


def test_known_connectives() -> None:
    """The evaluating connectives give what SymPy's constructors give,
    printed the same, on the cases probed (SymPy keeps
    ``Q.positive(x) & ~Q.positive(x)``, which is mirrored)."""
    from sympy.logic.boolalg import ITE as SITE
    S_NS: dict[str, object] = {'x': sx, 'y': sy, 'z': sz, 'And': sympy.And, 'Or': sympy.Or, 'Not': sympy.Not,
        'Xor': sympy.Xor, 'Implies': sympy.Implies, 'Equivalent': sympy.Equivalent, 'ITE': SITE, 'Eq': sympy.Eq,
        'Ne': sympy.Ne, 'true': sympy.true, 'Nand': sympy.Nand, 'Nor': sympy.Nor, 'Q': sympy.Q}
    C_NS: dict[str, object] = {'x': cx, 'y': cy, 'z': cz, 'And': C.and_, 'Or': C.or_, 'Not': C.not_, 'Xor': C.xor,
        'Implies': C.implies, 'Equivalent': C.equivalent, 'ITE': C.ite, 'Eq': C.Eq, 'Ne': C.Ne, 'true': C.true,
        'Nand': C.Nand, 'Nor': C.Nor, 'Q': C.Q}
    cases = ['And(x, y)', 'And(y, x)', 'And(x, And(y, z))', 'And(x, x)', 'And(x, True)', 'And(x, False)',
        'And(x, Not(x))', 'And()', 'Or()', 'And(x)', 'Or(x, Or(y, z))', 'Or(x, True)', 'Or(x, False)', 'Or(x, Not(x))',
        'Not(Not(x))', 'Not(True)', 'Not(x > 0)', 'Not(Eq(x, 1))', 'Not(And(x, y))', 'Not(Or(x, y))',
        'And(x > 0, x < 1)', 'Or(x > 0, Eq(x, 1))', '(x > 0) & (x < 1)', '~(x > 0)', '(x > 0) >> (x < 1)',
        '(x > 0) ^ (x < 1)', 'x & y', 'x | y', '~x', 'x >> y', 'x << y', 'x ^ y', 'And(z, y, x)', 'And(y > 0, x > 0)',
        'Or(z, x, y)', 'Xor(x, y)', 'Xor(x, x)', 'Xor(x, True)', 'Xor(x, y, z)', 'Xor()', 'Xor(x, y, x)', 'Xor(x, False)',
        'Implies(x, y)', 'Implies(True, y)', 'Implies(False, y)', 'Implies(x, True)', 'Implies(x, False)',
        'Implies(x, x)', 'Equivalent(x, y)', 'Equivalent(x, x)', 'Equivalent(x, True)', 'Equivalent(x, False)',
        'Equivalent(x, y, z)', 'Equivalent(x, Not(x))', 'Nand(x, y)', 'Nor(x, y)', 'ITE(x, y, z)', 'ITE(True, y, z)',
        'And(Or(x, y), z)', 'Or(And(x, y), z)', 'Implies(x, y) & z', 'And(x, Implies(y, z))', 'Not(Implies(x, y))',
        'Xor(x, y) & z', 'And(x, Not(y))', 'Or(x, Not(y))', 'And(x, Eq(y, 1))', 'Or(x, Eq(y, 1))', 'And(x, y) | z',
        'And(x, y) & z', 'Xor(x, y) | z', 'Not(Xor(x, y))', 'Not(ITE(x, y, z))', 'And(x > 0, x > 1)',
        'And(Eq(x, 1), Eq(x, 2))', 'Or(x > 0, x >= 0)', 'And(x > 0, x > 0, y)', 'Or(Eq(x, 1), Ne(x, 1))',
        'And(Eq(x, 1), Ne(x, 1))', 'And(x > 0, Not(x > 0))', 'And(Q.positive(x), Q.real(x))',
        'And(Q.positive(x), Not(Q.positive(x)))', 'Xor(x, y, Not(y))', 'Xor(x, y, True)', 'And(x, Xor(y, z))',
        'Equivalent(x, y, y)', 'Equivalent(x, True, y)', 'Equivalent(x, True, False)', 'Equivalent(True, True)',
        'Xor(x, y, Xor(y, z))', 'Not(Not(Not(x)))', '~(x >= 0)', '~(x <= 0)', 'Not(x).subs(x, y > 0)',
        'And(x, y, True, True)', 'Or(x, y, False)', 'And(x, y).subs({x: True, y: False})',
        'And(x > 0, y > 0).subs({x: 1, y: 2})', 'Or(x, y).subs(x, True)', 'Xor(x, y).subs(x, True)',
        'Equivalent(x, y).subs(x, True)', 'ITE(x, y, z).subs(x, True)', 'Not(x > y).subs(x, 1)', 'And(x > 0, y)',
        'str(Not(x) & y)', 'str(Not(x & y))', 'str(~(x | y) & z)', 'str(Implies(x, y) | z)', 'str(Nand(x, y) & z)',
        'str(Or(x, Implies(y, z)))', 'sorted(And(x, y).free_symbols, key=str)', 'And(x, y).count_ops()', 'true & x',
        'x | False']
    for text in cases:
        expected = eval(text, S_NS)
        got = eval(text, C_NS)
        if isinstance(got, C.Basic) and isinstance(expected, sympy.Basic):
            assert to_sympy(got) == expected, (text, got, expected)
        assert str(got) == str(expected), (text, got, expected)


def test_known_ask_and_refine() -> None:
    """``ask`` and ``refine`` answer as SymPy does on the cases probed:
    with conjunctive assumptions by the facts, with others by SymPy's
    solver, the errors included."""
    sf, cf = sympy.Function('f'), C.Function('f')
    S_NS: dict[str, object] = dict(vars(sympy), Q=sympy.Q, ask=sympy.ask, refine=sympy.refine, x=sx, y=sy, z=sz, f=sf)
    C_NS: dict[str, object] = dict(vars(C), x=cx, y=cy, z=cz, f=cf, sin=C.SymPyFunction(sympy.sin),
        exp=C.SymPyFunction(sympy.exp), Abs=C.SymPyFunction(sympy.Abs), log=C.SymPyFunction(sympy.log),
        sign=C.SymPyFunction(sympy.sign), re=C.SymPyFunction(sympy.re), im=C.SymPyFunction(sympy.im))
    cases = ['ask(Q.positive(x), Q.positive(x))', 'ask(Q.positive(x + 1), Q.positive(x))', 'ask(Q.integer(2))',
        'ask(Q.prime(x), Q.integer(x))', 'ask(Q.even(x), Q.odd(x))', 'ask(Q.positive(x), x > 0)',
        'ask(x > 0, Q.positive(x))', 'ask(Q.zero(x), x > 0)', 'ask(Q.is_true(x > 0), Q.positive(x))',
        'ask(Q.real(x), Q.positive(x) & Q.integer(y))', 'ask(Q.positive(x*y), Q.positive(x) & Q.positive(y))',
        'ask(Q.positive(x*y), Q.positive(x) & Q.negative(y))', 'ask(Q.real(x), Q.imaginary(x))',
        'ask(Q.integer(x), Q.even(x) | Q.odd(x))', 'ask(Q.positive(pi))', 'ask(Q.even(2*x), Q.integer(x))',
        'ask(Q.positive(x**2), Q.real(x) & Q.nonzero(x))', 'ask(Q.rational(x), Q.integer(x))',
        'ask(Q.positive(x), Q.nonnegative(x) & Q.nonzero(x))', 'ask(Q.negative_infinite(-oo))',
        'ask(Q.positive_infinite(oo))', 'ask(Q.finite(x), Q.real(x))', 'ask(Q.real(x), True)', 'ask(Q.real(x))',
        'ask(Q.commutative(x))', 'ask(Q.positive(x), Q.positive(x) | Q.negative(x))',
        'ask(Q.nonzero(x), Q.positive(x) | Q.negative(x))', 'ask(Q.positive(x) & Q.real(x), Q.positive(x))',
        'ask(Q.positive(x) | Q.real(x), Q.negative(x))', 'ask(Not(Q.positive(x)), Q.negative(x))',
        'ask(Implies(Q.positive(x), Q.real(x)))', 'ask(Q.positive(x) & Q.negative(x))', 'ask(true)', 'ask(false)',
        'ask(Eq(x, 1), Eq(x, 1))', 'ask(Q.positive(x), Eq(x, 1))', 'ask(Q.zero(x), Eq(x, 0))',
        'ask(Q.positive(x - 1), (x > 1))', 'ask(Q.positive(x), Q.positive(x) & Q.positive(y) & (z > 0))',
        'ask(Q.real(x), Q.positive(y))', 'ask(Q.positive(f(x)), Q.positive(f(x)))', 'ask(Q.positive(exp(x)), Q.real(x))',
        'ask(Q.positive(x + y), Q.positive(x) & Q.positive(y))', 'ask(Q.positive(x*y), Q.positive(x) & Q.real(y))',
        'ask(Q.integer(x/2), Q.even(x))', 'ask(Q.integer(x**2), Q.integer(x))', 'ask(Q.negative(-x), Q.positive(x))',
        'ask(Q.real(x + I), Q.real(x))', 'ask(Q.imaginary(x*I), Q.real(x) & Q.nonzero(x))',
        'ask(Q.even(x + 1), Q.odd(x))', 'ask(Q.prime(7))', 'ask(Q.composite(8))', 'ask(Q.positive(x), ~Q.negative(x))',
        'ask(Q.positive(x), ~Q.negative(x) & ~Q.zero(x) & Q.real(x))', 'ask(Q.extended_real(oo))',
        'ask(Q.positive(x), Q.positive_infinite(x))', 'ask(Q.extended_positive(x), Q.positive_infinite(x))',
        'ask(Q.finite(x), Q.positive_infinite(x))', 'ask(Q.positive(x).subs(x, 2))', 'ask(Q.gt(x, y), x > y)',
        'ask(Q.gt(2, 1))', 'ask(Q.eq(x, x))', 'ask(Q.positive(x), Q.is_true(x > 0))',
        'ask(Q.positive(x), Q.is_true(Q.positive(x)))', 'ask(Q.real(x), Q.positive(x) | Q.negative(x))',
        'ask(Q.positive(x), Implies(Q.real(x), Q.positive(x)) & Q.real(x))',
        'ask(Q.positive(x), Q.positive(x) ^ Q.negative(x))', 'ask(Q.real(x), Q.positive(x) ^ Q.negative(x))',
        'ask(Q.positive(f(x) + 1), Q.positive(f(x)))', 'ask(~Q.positive(x), Q.positive(x))', 'ask(~Q.positive(x))',
        'ask(Implies(Q.positive(x), Q.negative(x)), Q.positive(x))', 'ask(Xor(Q.positive(x), Q.negative(x)), Q.positive(x))',
        'ask(Q.positive(x) & Q.negative(y), Q.positive(x))', 'ask(Q.positive(x) | Q.negative(y), Q.negative(y))',
        'ask(Q.zero(x*y), Q.zero(y))', 'ask(Q.odd(x), ~Q.even(x) & Q.integer(x))',
        'ask(Q.rational(x), ~Q.irrational(x) & Q.real(x))', 'ask(Q.positive(x), True)',
        'ask(Q.even(x*y), Q.even(x) & Q.integer(y))', 'ask(Q.odd(x + y), Q.odd(x) & Q.even(y))',
        'ask(Q.prime(x + 1), Q.prime(x))', 'ask(Q.positive(Abs(x)), Q.nonzero(x))', 'ask(Q.nonnegative(Abs(x)))',
        'ask(Q.real(sin(x)), Q.real(x))', 'ask(Q.positive(exp(x)))', 'ask(Q.positive(x**2), Q.real(x))',
        'ask(Q.nonnegative(x**2), Q.real(x))', 'ask(Q.positive(2**x), Q.real(x))', 'ask(Q.integer(x/2), Q.integer(x))',
        'ask(Q.integer(x/y), Q.integer(x) & Q.integer(y))', 'ask(Q.positive(x - y), Q.positive(x) & Q.negative(y))',
        'ask(Q.positive(x - y), Q.positive(x) & Q.positive(y))', 'ask(Q.positive(x), Q.positive(x) & ~Q.positive(x))',
        'ask(Q.positive(x), Q.positive(x) & Q.zero(x))', 'ask(Q.positive(x), Q.even(x) & Q.odd(x))',
        'ask(Q.positive(x), False)', 'ask(Q.positive(x), Q.positive(x) & Q.negative(y) & Q.positive(y))',
        'ask(Q.positive(x), x)', 'ask(Q.positive(x), 2)', 'ask(2)', 'ask(Q.positive(x), Q.positive(x) | x)',
        'ask(Q.positive(x), Q.real(x) & ~Q.real(x))', 'ask(Q.positive(x), Q.positive(x) & Q.negative(x) | Q.zero(x))',
        'ask(Q.positive(x), Q.integer(x) & Q.irrational(x))', 'refine(sqrt(x**2), Q.positive(x))',
        'refine(Abs(x), Q.positive(x))', 'refine(sqrt(x**2), Q.real(x))', 'refine(sqrt(x**2), Q.positive(x) & Q.real(y))',
        'refine(Abs(x) + Abs(y), Q.positive(x) & Q.negative(y))', 'refine((-1)**x, Q.even(x))',
        'refine((-1)**x, Q.odd(x))', 'refine(exp(log(x)), Q.real(x))', 'refine(Eq(x, 0), Q.positive(x))',
        'refine(x > 0, Q.negative(x))', 'refine(sign(x), Q.negative(x))', 'refine(re(x), Q.real(x))',
        'refine(im(x), Q.imaginary(x))', 'refine(Abs(x*y), Q.positive(x))', 'refine(sqrt(x**2), Q.nonnegative(x))',
        'refine(sqrt(x**4), Q.real(x))']

    def run(ns: dict[str, object], text: str) -> str:
        try:
            return str(eval(text, ns))
        except (TypeError, ValueError) as e:
            return '%s: %s' % (type(e).__name__, e)
    for text in cases:
        assert run(C_NS, text) == run(S_NS, text), text


def test_known_order_of_arguments() -> None:
    """The arguments of sums and products are in SymPy's order, on random
    expressions; ``compare`` orders the classes as SymPy does."""
    from functools import cmp_to_key

    from sympy_extras.core.sorting import compare_key
    checked = 0
    for s, c in _samples(3, 150):
        for node in C.postorder_traversal(c):
            if isinstance(node, (C.Add, C.Mul)):
                sn = to_sympy(node)
                if isinstance(sn, (sympy.Add, sympy.Mul)) and len(sn.args) == len(node.args):
                    assert [str(to_sympy(a)) for a in node.args] == [str(a) for a in sn.args], node
                    checked += 1
    assert checked > 100
    items = [C.Integer(0), C.Integer(1), C.Rational(1, 2), C.oo, C.nan, C.Integer(-1), -C.oo, C.Integer(5),
        C.Rational(1, 3), C.Float(2.5), C.E, C.pi, C.I, cx, cn, C.Dummy('d'), C.Wild('w'), C.sqrt(cx), cx**2, cx*cy,
        cx + 1, _cv('Derivative(f(x), x)'), _cv('Integral(f(x), x)'), _cv('Abs(x)'), _cv('factorial(x)'),
        C.Lambda(cx, cx**2), C.Order(cx), C.Eq(cx, 1), C.Ne(cx, 1), cx > 1, cx < 1, cx >= 1, cx <= 1,
        C.AlgebraicNumber(C.sqrt(2), [1, 1]), C.Q.positive(cx), cx & cy, C.true, C.false, C.Catalan, C.zoo,
        C.Dict({cx: 1}), C.equivalent(cx, cy), C.EulerGamma, C.GoldenRatio, C.ite(cx, cy, cx), C.implies(cx, cy),
        C.Mod(cx, 2), ~cx, cx | cy, _cv('Piecewise((x, x > 0), (0, True))'), C.Subs(_cv('f(x)'), cx, 1),
        _cv('Sum(x, (x, 1, 2))'), C.TribonacciConstant, C.Tuple(cx), C.UnevaluatedExpr(cx), cx ^ cy,
        _cv('ceiling(x)'), _cv('exp(x)'), _cv('f(x)'), _cv('floor(x)'), _cv('log(x)'), _cv('sin(x)')]
    ours = sorted(items, key=compare_key)
    theirs = sorted([to_sympy(i) for i in items], key=cmp_to_key(sympy.Basic.compare))
    assert [str(i) for i in ours] == [str(i) for i in theirs]


def test_known_native_algorithms() -> None:
    """``factor_terms``, ``gcd_terms``, ``as_leading_term`` and
    ``leadterm`` agree with SymPy on the cases probed, ``Mod`` reduces
    ``n*(n + 1)`` modulo 2 as SymPy does, and the polynomial predicates
    give SymPy's three-valued answers."""
    for text in ['x**2 + x', '2*x + 4*y', '2*x + 4*y + 6', 'x*y + x*z', 'x/2 + y/2', 'x/2 + y/4', '-x - y', '-2*x - 4*y',
                 'x**2*y + x*y**2', '2*x**2 + 4*x**3', 'x*(x + 1) + 2*x', 'sin(x)*x + sin(x)*y', 'x + y', 'x**2 + x + 1',
                 'x*(2*y + 4*z)', '(2*x + 4*y)**2', '(2*x + 4*y)*(3*x + 6)', 'sqrt(2)*x + sqrt(2)*y', '2*sqrt(2)*x + 4*y',
                 'x/y + z/y', '2*x/y + 4*z/y', 'x*exp(x) + x', '2*exp(2*x) + 4*exp(x)', 'x**(n+1) + x**n', '2*x - 2',
                 '(x + 1)*(2*x + 2)', 'x*(x + y) + y*(x + y)', '6*x*y + 9*x*z + 3*x', 'x/3 + 1/3', '-x**2 - x', 'I*x + I*y',
                 '2*pi*x + 4*pi*y', 'x*(y + 1)/2 + x*(z + 1)/2', 'sqrt(x) + x', 'x**-1 + x**-2', '1/x + 1/y', '2/x + 4/x**2',
                 'x + x*y', 'f(2*x + 4*y)', 'sin(2*x + 4*y)', 'exp(2*x + 4*y)', 'Eq(2*x + 4, 2*y)', 'x + 1/x', '-x - 1',
                 '-2*x - 2', 'x/2 + x**2/2', '1/(2*x) + 1/(4*y)', '3/(2*x) + 1/x**2', 'x/(x + 1) + y/(x + 1)',
                 '-1/x - 1/y', 'exp(x)/x + exp(2*x)/x**2', 'x/2 + y', 'x/2 + 1/x']:
        s = parse(text) if 'Eq' not in text else sympy.Eq(parse('2*x + 4'), parse('2*y'))
        c = from_sympy(s)
        sft, sgt = untyped(sympy.factor_terms), untyped(sympy.gcd_terms)
        for ours, theirs in [(C.factor_terms(c), sft(s)), (C.gcd_terms(c), sgt(s)),
                             (C.factor_terms(c, fraction=True), sft(s, fraction=True)),
                             (C.factor_terms(c, sign=False), sft(s, sign=False)),
                             (C.gcd_terms(c, fraction=False), sgt(s, fraction=False))]:
            assert str(ours) == str(theirs), (text, ours, theirs)
    for text in ['x + x**2', '2*x**2 + x**3', '1 + x', 'x*y + x**2*y', 'x**-1 + x', '1/x + 1/x**2', 'x + y', 'x*(x + 1)',
                 '(x + 1)**2', 'x**2*(1 + x)', 'sin(x)', 'sin(x) + x', 'exp(x) - 1', 'cos(x) - 1', 'sqrt(x) + x',
                 '1/(1 + x)', '(1 + x)/x', 'x/(1 + x)', 'x**x', 'x**n', 'y', '2', 'x - x**2', 'x + x*y', 'y*(x + 1)',
                 'x**2 + 2*x*y + y**2', 'x + y + 1', 'I*x + x**2', '1/x + sin(x)', 'x*exp(x) + x**2', 'x**y',
                 '(x**2 + x)**2', '(x**2 + x)**(1/2)', '1/sqrt(x + x**2)', '(x + 1)**x', 'f(x)', 'x*f(x)',
                 'x**2*y + x*y**2 + y**3', '(x + 2)**3', 'x/(x**2 + 1) + 1', '(x + y)/(x - y)', '3*x**2 - 2*x']:
        se, ce = parse(text), _cv(text)
        pairs: list[tuple[Callable[[], object], Callable[[], object]]] = [
            (lambda: ce.as_leading_term(cx), lambda: se.as_leading_term(sx)),
            (lambda: ce.leadterm(cx), lambda: se.leadterm(sx)),
            (lambda: ce.as_leading_term(cx, cy), lambda: se.as_leading_term(sx, sy))]
        for ours_f, theirs_f in pairs:
            try:
                expected_text = str(theirs_f())
            except ValueError:
                expected_text = 'ValueError'
            try:
                got_text = str(ours_f())
            except ValueError:
                got_text = 'ValueError'
            assert got_text == expected_text, (text, got_text, expected_text)
    for text in ['n*(n + 1)', 'n*(n + 3)', '(n + 1)*(n + 4)', 'n*(n + 2)', 'n**2*(n + 1)', '3*n*(n + 1)', 'n*(n - 1)',
                 '(n - 1)*(n + 1)', 'n*(n + 1)*(n + 2)', 'n*(n + 1)*y', 'x*(x + 1)', 'n*(2*n + 1)', 'n*(n + 1)*(2*n + 1)']:
        for q in (2, 3, 4, 6):
            assert str(C.mod(_cv(text), q)) == str(sympy.Mod(parse(text), q)), (text, q)
    for text in ['sin(x)', 'sin(x) + 1', '2*sin(x)', 'sin(x)**2', 'sin(y)', 'f(x)', 'x + f(x)', 'sqrt(sin(x))',
                 'sin(x)/x', 'Abs(x)', 'exp(x)', 'sin(x)**-1', 'sin(x)**y', 'sin(x)**(1/2)', 'f(x)**2', '2**sin(x)',
                 'x**sin(x)', '(x + sin(x))**2', '1/sin(x)', 'sqrt(x)**2', 'x**-1', 'sqrt(x)', 'x**n', 'x**y', 'x**2.5',
                 'sin(y)**x', 'sin(y)*x', 'sin(y) + x', 'sin(y)**2', 'y**f(x)', 'f(y)**-1', 'f(y)**x', 'x**2 + 1',
                 'x + 1/y']:
        sp_, cp_ = parse(text), _cv(text)
        answers = (cp_.is_polynomial(), cp_.is_polynomial(cx), cp_.is_rational_function(cx),
                   cp_.is_algebraic_expr(cx), cp_.is_polynomial(cy), cp_.is_rational_function(cy),
                   cp_.is_algebraic_expr(cy))
        sympy_answers = (sp_.is_polynomial(), sp_.is_polynomial(sx), sp_.is_rational_function(sx),
                         sp_.is_algebraic_expr(sx), sp_.is_polynomial(sy), sp_.is_rational_function(sy),
                         sp_.is_algebraic_expr(sy))
        assert answers == sympy_answers, (text, answers, sympy_answers)


def test_known_methods_on_random_expressions() -> None:
    """Dozens of methods and properties of ``Expr`` give what SymPy gives
    on random expressions with functions (the method fuzz of the
    session): the printed results are compared, a fact known here and
    unknown to SymPy is allowed."""
    rng = random.Random(1)
    sfs = [sympy.sin, sympy.cos, sympy.exp, sympy.log, sympy.Abs]
    leaves = [(sx, cx), (sy, cy), (sz, cz), (sp, cp), (sn, cn), (sympy.pi, C.pi), (sympy.I, C.I), (sympy.E, C.E)]

    def leaf() -> Pair:
        r = rng.random()
        if r < 0.5:
            return rng.choice(leaves)
        if r < 0.8:
            k = rng.randint(-4, 4)
            return (sympy.Integer(k), C.Integer(k))
        a, b = rng.randint(-6, 6), rng.randint(1, 6)
        return (sympy.Rational(a, b), C.Rational(a, b))

    def build(depth: int) -> Pair:
        if depth == 0 or rng.random() < 0.25:
            return leaf()
        op = rng.choice(['add', 'add', 'mul', 'mul', 'pow', 'neg', 'sub', 'div', 'fun'])
        if op in ('add', 'mul'):
            parts = [build(depth - 1) for _ in range(rng.randint(2, 3))]
            if op == 'add':
                return (sympy_expr(sympy.Add(*[a for a, _ in parts])), C.add(*[b for _, b in parts]))
            return (sympy_expr(sympy.Mul(*[a for a, _ in parts])), C.mul(*[b for _, b in parts]))
        if op == 'pow':
            base = build(depth - 1)
            r = rng.random()
            if r < 0.5:
                k = rng.randint(-2, 3)
                e: Pair = (sympy.Integer(k), C.Integer(k))
            elif r < 0.8:
                q = rng.choice([2, 3])
                numerator = rng.choice([1, -1, 2])
                e = (sympy.Rational(numerator, q), C.Rational(numerator, q))
            else:
                e = leaf()
            return (sympy_expr(sympy.Pow(base[0], e[0])), C.pow(base[1], e[1]))
        if op == 'fun':
            g = rng.choice(sfs)
            inner = build(depth - 1)
            return (sympy_expr(g(inner[0])), C.SymPyFunction(g)(inner[1]).doit())
        if op == 'neg':
            inner = build(depth - 1)
            return (sympy_expr(-inner[0]), -inner[1])
        left, right = build(depth - 1), build(depth - 1)
        if op == 'sub':
            return (sympy_expr(left[0] - right[0]), left[1] - right[1])
        return (sympy_expr(left[0] / right[0]), left[1] / right[1])

    def shown(v: object) -> str:
        if isinstance(v, (tuple, list)):
            return '(' + ', '.join(shown(a) for a in v) + ')'
        if isinstance(v, (set, frozenset)):
            return '{' + ', '.join(sorted(str(a) for a in v)) + '}'
        if isinstance(v, dict):
            return '{' + ', '.join('%s: %s' % (k, shown(w)) for k, w in sorted(v.items(), key=lambda kv: str(kv[0]))) + '}'
        return str(v)
    facts = ['positive', 'real', 'integer', 'zero', 'finite', 'rational', 'nonnegative', 'even', 'imaginary',
             'algebraic']
    checks: list[tuple[str, Callable[[sympy.Expr], object], Callable[[C.Expr], object]]] = [
        ('str', lambda s: s, lambda c: c),
        ('is_number', lambda s: s.is_number, lambda c: c.is_number),
        ('srepr', lambda s: sympy.srepr(s), lambda c: C.srepr(c)),
        ('args', lambda s: s.args, lambda c: c.args),
        ('round trip', lambda s: True, lambda c: from_sympy(to_sympy(c)) == c),
        ('free_symbols', lambda s: s.free_symbols, lambda c: c.free_symbols),
        ('as_coeff_Mul', lambda s: s.as_coeff_Mul(), lambda c: c.as_coeff_Mul()),
        ('as_coeff_Add', lambda s: s.as_coeff_Add(), lambda c: c.as_coeff_Add()),
        ('as_base_exp', lambda s: s.as_base_exp(), lambda c: c.as_base_exp()),
        ('as_independent', lambda s: s.as_independent(sx), lambda c: c.as_independent(cx)),
        ('primitive', lambda s: s.primitive(), lambda c: c.primitive()),
        ('as_ordered_factors', lambda s: s.as_ordered_factors(), lambda c: c.as_ordered_factors()),
        ('as_ordered_terms', lambda s: s.as_ordered_terms(), lambda c: c.as_ordered_terms()),
        ('as_terms', lambda s: [str(t) for t, _ in s.as_terms()[0]], lambda c: [str(t) for t, _ in c.as_terms()[0]]),
        ('could_extract_minus_sign', lambda s: s.could_extract_minus_sign(), lambda c: c.could_extract_minus_sign()),
        ('diff', lambda s: sympy.diff(s, sx), lambda c: C.diff(c, cx)),
        ('subs', lambda s: s.subs(sx, sy + 1), lambda c: c.subs(cx, cy + 1)),
        ('subs2', lambda s: s.subs({sx: 2, sy: sz}), lambda c: c.subs({cx: C.Integer(2), cy: cz})),
        ('has', lambda s: s.has(sx), lambda c: c.has(cx)),
        ('atoms', lambda s: s.atoms(), lambda c: c.atoms()),
        ('ordered', lambda s: [str(t) for t in sympy.ordered(sympy.Add.make_args(s) + sympy.Mul.make_args(s))],
         lambda c: [str(t) for t in C.ordered(C.Add.make_args(c) + C.Mul.make_args(c))]),
        ('is_constant', lambda s: s.is_constant(), lambda c: c.is_constant()),
        ('conjugate', lambda s: sympy.conjugate(s), lambda c: c.conjugate()),
        ('together', lambda s: sympy.together(s), lambda c: c.together()),
        ('extract_multiplicatively', lambda s: s.extract_multiplicatively(sympy.Integer(2)),
         lambda c: c.extract_multiplicatively(C.Integer(2))),
        ('removeO', lambda s: s.removeO(), lambda c: c.removeO()),
        ('replace', lambda s: s.replace(sx, sy), lambda c: c.replace(cx, cy)),
        ('kind', lambda s: s.kind, lambda c: c.kind),
        ('is_comparable', lambda s: s.is_comparable, lambda c: c.is_comparable),
        ('compare', lambda s: s.compare(sx), lambda c: c.compare(cx)),
        ('sort_key', lambda s: s.sort_key(), lambda c: c.sort_key()),
        ('is_polynomial', lambda s: s.is_polynomial(), lambda c: c.is_polynomial()),
        ('is_rational_function', lambda s: s.is_rational_function(), lambda c: c.is_rational_function()),
    ]
    checked = 0
    for _ in range(60):
        s, c = build(3)
        if s.has(sympy.zoo, sympy.nan):
            continue
        checked += 1
        for name, fs, fc in checks:
            assert shown(fc(c)) == shown(fs(s)), (name, s)
        for name in facts:
            expected = getattr_(s, name)
            got = ask_structural(c, name)
            assert got == expected or expected is None, (name, s, got, expected)
    assert checked > 40
