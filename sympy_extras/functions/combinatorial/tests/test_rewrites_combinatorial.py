"""The ported rewrite rules of the combinatorial numbers (sums, products,
zeta, polygamma, the closed forms with radicals): the rewritten form has
the value of the node (they went to SymPy through the bridge)."""
from __future__ import annotations

from sympy_extras._numeric import reliable_complex
from sympy_extras.concrete.expr_with_limits import Limits, Product, Sum
from sympy_extras.core.add import add
from sympy_extras.core.expr import as_expr
from sympy_extras.core.mul import mul
from sympy_extras.core import Integer, Rational, Symbol
from sympy_extras.core.expr import Expr
from sympy_extras.core.numbers import GoldenRatio, TribonacciConstant
from sympy_extras.core.rewriting import RewriteTarget, has_rule
from sympy_extras.functions.combinatorial import factorials as f, numbers as nb
from sympy_extras.functions.special.gamma_functions import DiGamma, PolyGamma, TriGamma, UpperGamma
from sympy_extras.functions.special.hyper import Hyper
from sympy_extras.functions.special.zeta_functions import DirichletEta, PolyLog, Zeta

x, y = Symbol('x'), Symbol('y')
n = Symbol('n', integer=True, nonnegative=True)


def _close(a: Expr, b: Expr, values: dict[Symbol, Expr]) -> bool:
    va, vb = reliable_complex(a, 15, values), reliable_complex(b, 15, values)
    assert va is not None and vb is not None, (a, b, values)
    return abs(va - vb) <= 1e-9 * max(1.0, abs(va))


def _check(node: Expr, target: RewriteTarget, points: list[dict[Symbol, Expr]]) -> None:
    rewritten = node.rewrite(target)
    assert rewritten != node, (node, target)
    for values in points:
        assert _close(node, rewritten, values), (node, target, values)


def _at_integers(node: Expr, target: RewriteTarget, ns: list[int]) -> None:
    # the sums and products: equal to the node at integer points, written out
    rewritten = node.rewrite(target)
    assert rewritten != node, (node, target)
    for k in ns:
        values: dict[Symbol, Expr] = {n: Integer(k)}
        lhs = node.subs(values).doit()
        rhs = _explicit(rewritten.subs(values))
        va, vb = reliable_complex(lhs, 15), reliable_complex(rhs, 15)
        assert va is not None and vb is not None and abs(va - vb) <= 1e-9 * max(1.0, abs(va)), (node, k, lhs, rhs)


def _explicit(e: Expr) -> Expr:
    """``e`` with its sums and products of integer bounds written out term
    by term (``doit`` would evaluate the symbolic inner sums first, see
    the euler test)."""
    if isinstance(e, (Sum, Product)):
        return _unrolled(e.function, list(e.limits), isinstance(e, Sum))
    if not e.args:
        return e
    args = [_explicit(a) for a in e.args if isinstance(a, Expr)]
    if len(args) != len(e.args):
        return e
    return as_expr(e._evaluated(tuple(args)))


def _unrolled(f: Expr, limits: list[Limits], is_sum: bool) -> Expr:
    if not limits:
        return _explicit(f)
    v, a, b = limits[-1]
    assert isinstance(a, Integer) and isinstance(b, Integer), (a, b)
    terms: list[Expr] = []
    for i in range(a.p, b.p + 1):
        value = Integer(i)
        inner: list[Limits] = [(w, lo.subs(v, value), hi.subs(v, value)) for w, lo, hi in limits[:-1]]
        terms.append(_unrolled(f.subs(v, value), inner, is_sum))
    return add(*terms) if is_sum else mul(*terms)


def test_fibonacci_lucas_tribonacci_closed_forms() -> None:
    real: list[dict[Symbol, Expr]] = [{x: Integer(k)} for k in (0, 1, 5, 9)]
    _check(nb.Fibonacci(x), 'sqrt', real)
    _check(nb.Fibonacci(x), GoldenRatio, real)
    _check(nb.Lucas(x), 'sqrt', real)
    # SymPy's closed forms hold at the integers (the TribonacciConstant one is a floor)
    _check(nb.Tribonacci(x), 'sqrt', real)
    _check(nb.Tribonacci(x), TribonacciConstant, real)
    assert has_rule(nb.Tribonacci, 'sqrt') and has_rule(nb.Fibonacci, 'GoldenRatio')


def test_harmonic_bernoulli_genocchi_euler_andre_special_function_forms() -> None:
    points: list[dict[Symbol, Expr]] = [{x: Rational(7, 10), y: Integer(3)}, {x: Rational(23, 10), y: Integer(2)}]
    for node in (nb.Harmonic(x), nb.Harmonic(x, Integer(2)), nb.Harmonic(x, Integer(3))):
        for target in (PolyGamma, DiGamma, TriGamma, Zeta):
            _check(node, target, points)
    integers: list[dict[Symbol, Expr]] = [{x: Integer(4), y: Rational(1, 3)}, {x: Integer(3), y: Rational(5, 2)}]
    _check(nb.Bernoulli(x), Zeta, integers)
    _check(nb.Bernoulli(x, y), Zeta, integers)
    _check(nb.Genocchi(x), nb.Bernoulli, integers)
    _check(nb.Genocchi(x, y), nb.Bernoulli, integers)
    _check(nb.Genocchi(x), DirichletEta, integers)
    _check(nb.Euler(x), nb.Genocchi, integers + [{x: Integer(-1)}])
    _check(nb.Euler(x, y), nb.Genocchi, integers)
    _check(nb.Andre(x), Zeta, [{x: Integer(5)}, {x: Integer(4)}])
    _check(nb.Andre(x), PolyLog, [{x: Integer(5)}, {x: Integer(6)}])


def test_catalan_subfactorial_forms() -> None:
    points: list[dict[Symbol, Expr]] = [{x: Integer(4)}, {x: Rational(7, 10)}]
    for target in (f.Binomial, f.Factorial, Hyper):
        _check(nb.Catalan(x), target, [{x: Integer(4)}, {x: Integer(6)}] if target is Hyper else points)
    _check(f.Subfactorial(x), UpperGamma, [{x: Integer(4)}, {x: Integer(6)}])
    _check(f.Subfactorial(Integer(5)), f.Factorial, [{}])


def test_sums_and_products() -> None:
    _at_integers(f.Factorial(n), 'Product', [0, 1, 4])
    _at_integers(nb.Catalan(n), 'Product', [0, 1, 3, 5])
    _at_integers(nb.Harmonic(n), 'Sum', [1, 4])
    _at_integers(nb.Harmonic(n, Integer(2)), 'Sum', [1, 3])
    # SymPy's Sum((-1)**j*(k - 2*j)**3*binomial(k, j), (j, 0, k)).doit() is 0
    # for a symbolic k (the native port too): the sums are written out
    _at_integers(nb.Euler(2 * n), 'Sum', [1, 2])
    _at_integers(nb.Euler(n, Rational(1, 3)), 'Sum', [2, 3])
    # Dobinski's formula: an infinite sum, checked numerically
    s = nb.Bell(Integer(4)).rewrite('Sum')
    sums = [t for t in s.args if isinstance(t, Sum)]
    assert len(sums) == 1
    (k, lo, hi), = sums[0].limits
    partial = add(*[sums[0].function.subs(k, Integer(i)) for i in range(60)])
    value = reliable_complex(s.subs(sums[0], partial), 15)
    assert value is not None and abs(value - 15) < 1e-9
    # SymPy's rules which do not apply leave the node as it is
    assert f.Factorial(x).rewrite('Product') == f.Factorial(x)
    assert nb.Catalan(x).rewrite('Product') == nb.Catalan(x)
    assert nb.Bell(x).rewrite('Sum') == nb.Bell(x)
