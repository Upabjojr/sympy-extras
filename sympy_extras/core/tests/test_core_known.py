"""The experimental core against SymPy: the evaluated forms, the
derivatives, the expansions, the numerical values and the printed forms
of expressions computed by both, and the round trip of the conversions.

SymPy's results are the facts here: what ``Add``, ``Mul`` and ``Pow``
evaluate to is documented by SymPy's behaviour, and ``add``, ``mul`` and
``pow`` are expected to agree with it up to the order of the arguments
(and up to two quirks of SymPy noted in ``test_known_differences``).
"""
from __future__ import annotations

import random
from typing import Callable

import sympy

from sympy_extras import core as C
from sympy_extras.core import from_sympy, to_sympy

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
        (1/(2 + C.I), '2/5 - I/5'), (1/(1 - C.I), '1/2 + I/2'), ((2 + C.I)**-2, '(2 + I)**(-2)'),
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


def getattr_(s: sympy.Symbol, name: str) -> bool | None:
    """SymPy's ``is_<name>`` of a symbol, read without ``getattr``."""
    queries: dict[str, Callable[[sympy.Symbol], bool | None]] = {
        'positive': lambda e: e.is_positive, 'negative': lambda e: e.is_negative, 'zero': lambda e: e.is_zero,
        'nonzero': lambda e: e.is_nonzero, 'integer': lambda e: e.is_integer, 'rational': lambda e: e.is_rational,
        'real': lambda e: e.is_real, 'even': lambda e: e.is_even, 'odd': lambda e: e.is_odd,
        'prime': lambda e: e.is_prime, 'finite': lambda e: e.is_finite, 'commutative': lambda e: e.is_commutative,
    }
    return queries[name](s)
