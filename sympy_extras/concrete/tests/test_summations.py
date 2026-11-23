"""summation (Sum.doit) against SymPy (the oracle)."""
from __future__ import annotations

import sympy
from sympy import KroneckerDelta, Piecewise, Rational as R, binomial, exp, factorial, oo, sin

from sympy_extras.concrete.summations import summation
from sympy_extras.interop import to_sympy
from sympy_extras.interop import expr_from_sympy, from_sympy
from sympy_extras.core.symbol import Symbol

i = sympy.Symbol('i', integer=True)
k = sympy.Symbol('k', integer=True)
n = sympy.Symbol('n', integer=True)
m = sympy.Symbol('m', integer=True, positive=True)
x, y = sympy.symbols('x y')
j = sympy.Symbol('j')

CASES = [(i**2, (i, 1, n)), (i**3, (i, 0, n)), (x**i, (i, 0, n)), (2**i, (i, 1, n)), (1/(i*(i + 1)), (i, 1, n)),
         (i, (i, 1, 4)), (1/i**2, (i, 1, oo)), (1/i, (i, 1, n)), (1/i**3, (i, 1, n)), (binomial(n, i), (i, 0, n)),
         (i*binomial(n, i), (i, 0, n)), (x**i/factorial(i), (i, 0, oo)), (i*x**i, (i, 0, n)), (1/(i**2 - 1), (i, 2, n)),
         (sin(i), (i, 0, 3)), (i**2 + 3*i + 1, (i, 0, m)), (y, (i, 1, n)), (i*j, (i, 1, n)),
         (Piecewise((i, i < 5), (0, True)), (i, 0, 10)), (KroneckerDelta(i, k), (i, 0, n)),
         (1/((i + 1)*(i + 2)), (i, 0, oo)), (R(1, 2)**i, (i, 0, oo)), (factorial(i), (i, 0, n)), (i**4, (i, 1, n)),
         (1/i**2, (i, 1, n)), ((-1)**i/i, (i, 1, oo)), (exp(i), (i, 0, n)), (i/(i + 1), (i, 0, 5)), (x, (i, n, n)),
         (i, (i, 5, 3)), (i, (i, 1, 0)), (i**2, (i, -n, n)), (1/(2*i + 1), (i, 0, n)), (j**i, (i, 1, n)),
         (i*(i - 1)*binomial(n, i), (i, 0, n)), (binomial(m, i)*binomial(m, m - i), (i, 0, m)), (i**R(1, 2), (i, 1, n)),
         (1/i**(x + 1), (i, 1, oo)), (1/(i + 1)**2, (i, 0, oo))]


def test_summation_matches_sympy() -> None:
    for f, lim in CASES:
        v, a, b = lim
        var = from_sympy(v)
        assert isinstance(var, Symbol)
        ours = summation(expr_from_sympy(f), (var, expr_from_sympy(sympy.sympify(a)), expr_from_sympy(sympy.sympify(b))))
        assert str(to_sympy(ours, evaluate=False)) == str(sympy.summation(f, lim)), (f, lim)


def test_sum_doit_matches_sympy() -> None:
    for f, lim in CASES[:12]:
        s = sympy.Sum(f, lim)
        assert str(to_sympy(expr_from_sympy(s).doit(), evaluate=False)) == str(s.doit()), s


def test_hypergeometric_sums_match_sympy() -> None:
    # eval_sum_hyper: the sum as an expanded hypergeometric function on its
    # condition of convergence
    from sympy import rf
    from sympy.concrete.summations import eval_sum_hyper as sympy_eval_sum_hyper
    from sympy_extras.concrete.summations import eval_sum_hyper
    a = sympy.Symbol('a')
    cases = [(x**i/factorial(i), (i, 0, oo)), (R(1, 2)**i, (i, 0, oo)), ((-1)**i/(2*i + 1), (i, 0, oo)),
             (x**i*i, (i, 0, oo)), (x**(2*i)/factorial(2*i), (i, 0, oo)), (rf(a, i)*x**i/factorial(i), (i, 0, oo)),
             (binomial(n, i)**2, (i, 0, n)), (1/factorial(i), (i, 1, oo)), (x**i/factorial(i), (i, 2, oo)),
             (x**i/(i + 1), (i, 0, oo)), (binomial(2*i, i)*x**i, (i, 0, oo)), (2**i/factorial(i), (i, -oo, 0)),
             (x**i/factorial(i), (i, 0, n)), (1/2**i, (i, 3, oo)), (x**i, (i, -oo, oo)), (1/factorial(i)**2, (i, 0, oo))]
    for f, (v, lo, hi) in cases:
        var = from_sympy(v)
        assert isinstance(var, Symbol)
        limits = (var, expr_from_sympy(sympy.sympify(lo)), expr_from_sympy(sympy.sympify(hi)))
        ours = eval_sum_hyper(expr_from_sympy(f), limits)
        theirs = sympy_eval_sum_hyper(f, (v, lo, hi))
        assert (None if ours is None else str(to_sympy(ours, evaluate=False))) == (None if theirs is None else str(theirs))
        assert str(to_sympy(summation(expr_from_sympy(f), limits), evaluate=False)) == str(sympy.summation(f, (v, lo, hi)))


def test_convergence_statement_matches_sympy() -> None:
    from sympy_extras.functions.special.hyper import Hyper
    a, z = sympy.symbols('a z')
    for h in [sympy.hyper([1, 1], [2], z), sympy.hyper([a], [], z), sympy.hyper([a, 1], [3], z), sympy.hyper([], [1], z),
              sympy.hyper([1, 2, 3], [], z), sympy.hyper([1, 2], [3], 2*z), sympy.hyper([-n, -n, 1], [1, 1], 1)]:
        ours = expr_from_sympy(h)
        if not isinstance(ours, Hyper):
            continue
        c = ours.convergence_statement
        assert (str(c) if isinstance(c, bool) else str(to_sympy(c, evaluate=False))) == str(h.convergence_statement), h


def test_residue_sums_match_sympy() -> None:
    # eval_sum_residue; the Catalan sum is left out (the hypergeometric step
    # before it has a known difference in hyperexpand)
    from sympy.concrete.summations import eval_sum_residue as sympy_eval_sum_residue
    from sympy_extras.concrete.summations import eval_sum_residue
    cases = [(1/(i**2 + 1), (i, 0, oo)), (1/(i**2 + 1), (i, -oo, oo)), ((-1)**i/(i**2 + 1), (i, -oo, oo)),
             (1/(i**2 + 4), (i, 1, oo)), (1/(i**4 + 1), (i, -oo, oo)), (1/((i + 1)**2 + 1), (i, 0, oo)),
             (1/(i**2 - R(1, 4)), (i, 1, oo)), (1/(i**2 + i + 1), (i, -oo, oo)), (1/(i**2 + 1)**2, (i, -oo, oo)),
             (1/(4*i**2 + 1), (i, 0, oo)), (i**2/(i**4 + 4), (i, 1, oo)), (1/(i**3 + 2), (i, 1, oo)),
             (1/(i**2 + 2), (i, 3, oo)), (1/((i - 2)**2 + 1), (i, 5, oo)), ((-1)**i/(i**2 + 1), (i, 2, oo))]
    for f, (v, lo, hi) in cases:
        var = from_sympy(v)
        assert isinstance(var, Symbol)
        limits = (var, expr_from_sympy(sympy.sympify(lo)), expr_from_sympy(sympy.sympify(hi)))
        ours = eval_sum_residue(expr_from_sympy(f), limits)
        theirs = sympy_eval_sum_residue(f, (v, sympy.sympify(lo), sympy.sympify(hi)))
        assert (None if ours is None else str(to_sympy(ours, evaluate=False))) == (None if theirs is None else str(theirs))
        assert str(to_sympy(summation(expr_from_sympy(f), limits), evaluate=False)) == str(sympy.summation(f, (v, lo, hi)))


def test_residue_sums_at_crootof_poles() -> None:
    # the poles which are CRootOf are taken natively by the residue of a
    # pole at an algebraic point (they went to SymPy's eval_sum_residue,
    # which returns None here: the is_integer of its real CRootOf is None)
    import mpmath
    import pathlib
    from sympy_extras.concrete.summations import eval_sum_residue
    import sympy_extras.concrete as concrete
    i = sympy.Symbol('i')
    f = 1/(i**3 + i + 1)
    var = from_sympy(i)
    assert isinstance(var, Symbol)
    limits = (var, expr_from_sympy(-oo), expr_from_sympy(oo))
    ours = eval_sum_residue(expr_from_sympy(f), limits)
    assert ours is not None and ours.has(expr_from_sympy(sympy.CRootOf(i**3 + i + 1, 0)))
    with mpmath.workdps(30):
        # the tail beyond |k| = N is about 1/N**2: odd terms cancel
        partial = sum(mpmath.mpf(1)/(k**3 + k + 1) for k in range(-20000, 20001))
    assert abs(complex(ours.evalf(25)) - complex(partial)) < 1e-8
    # the poles of a double factor and the alternating sums
    f2 = 1/(i**3 + i + 1)**2
    ours2 = eval_sum_residue(expr_from_sympy(f2), limits)
    assert ours2 is not None
    with mpmath.workdps(30):
        partial2 = sum(mpmath.mpf(1)/(k**3 + k + 1)**2 for k in range(-2000, 2001))
    assert abs(complex(ours2.evalf(25)) - complex(partial2)) < 1e-12
    # SymPy's sum of 1/(i**4 + i + 1) over the integers is 0 (its series at a
    # CRootOf): the sum is positive, about 2.4920
    f3 = 1/(i**4 + i + 1)
    ours3 = eval_sum_residue(expr_from_sympy(f3), limits)
    assert ours3 is not None
    with mpmath.workdps(30):
        partial3 = sum(mpmath.mpf(1)/(k**4 + k + 1) for k in range(-2000, 2001))
    assert abs(complex(ours3.evalf(25)) - complex(partial3)) < 1e-9
    for path in pathlib.Path(concrete.__file__).parent.glob('*.py'):
        source = path.read_text()
        assert 'getattr(' not in source and 'import_module(' not in source, path


def test_residue_sums_over_ex() -> None:
    # coefficients our Poly does not take: the closed forms of ex_roots
    # (SymPy raises NotImplementedError: sorted roots not supported over EX)
    from sympy_extras.concrete.summations import eval_sum_residue
    i = sympy.Symbol('i', integer=True)
    var = from_sympy(i)
    assert isinstance(var, Symbol)
    for f, exact in ((1/(i**2 + sympy.pi), sympy.sqrt(sympy.pi)*sympy.coth(sympy.pi**R(3, 2))),
                     (1/(i**2 + sympy.sqrt(2)), sympy.pi/2**R(1, 4)*sympy.coth(sympy.pi*2**R(1, 4)))):
        ours = eval_sum_residue(expr_from_sympy(f), (var, expr_from_sympy(-oo), expr_from_sympy(oo)))
        assert ours is not None
        assert abs(sympy.N(to_sympy(ours) - exact, 30)) < 1e-25


def test_geometric_sum_with_ratio_one() -> None:
    # the bug: the sum of (-1)**(2*k) was nan, the geometric formula
    # (q**a - q**(b + 1))/(1 - q) at q = (-1)**2 = 1: the case Eq(q, 1) was
    # the unevaluated Eq(1, 1), which Piecewise drops as SymPy does for
    # Eq(1, 1, evaluate=False), where SymPy's Eq evaluates to True
    kk, nn = Symbol('k'), Symbol('n')
    from sympy_extras.core.numbers import Integer
    term = Integer(-1)**(2*kk)
    assert summation(term, (kk, 0, nn)) == nn + 1
    assert str(to_sympy(summation(term, (kk, 0, nn)))) == str(sympy.summation((-1)**(2*k), (k, 0, n)))


def test_gosper_certificate_poles_get_cases() -> None:
    # the bug (issue #75): Sum((-1)**j*(k - 2*j)**3*binomial(k, j), (j, 0, k))
    # was 0 for every k, but it is 2 at k = 1 and 48 at k = 3: the Gosper
    # certificates of the terms of the expanded summand (-j/k for
    # (-1)**j*binomial(k, j)) have poles at small k, where the telescoped
    # value does not hold (SymPy's Sum.doit gives 0 too)
    from sympy_extras.concrete.expr_with_limits import Sum
    from sympy_extras.core.add import add
    from sympy_extras.core.numbers import Integer
    from sympy_extras.functions.combinatorial.factorials import binomial as nbinomial
    jj = Symbol('j', integer=True, nonnegative=True)
    for positive in (False, True):
        kk = Symbol('k', integer=True, positive=True) if positive else Symbol('k', integer=True, nonnegative=True)
        for sign in (1, -1):
            for base in (kk - 2*jj, jj):
                for power in range(5):
                    term = Integer(sign)**jj * base**power * nbinomial(kk, jj)
                    value = Sum(term, (jj, 0, kk)).doit()
                    for kv in range(int(positive), 13):
                        direct = add(*[term.subs(kk, kv).subs(jj, jv) for jv in range(kv + 1)])
                        assert (value.subs(kk, Integer(kv)).doit() - direct).expand() == 0, (term, kv, value)
    kk = Symbol('k', integer=True, nonnegative=True)
    value = Sum((-1)**jj*(kk - 2*jj)**3*nbinomial(kk, jj), (jj, 0, kk)).doit()
    assert str(value) == 'Piecewise((2, Eq(k, 1)), (48, Eq(k, 3)), (0, True))'


def test_reversed_infinite_sums_follow_karr_convention() -> None:
    # sympy/sympy#31275, V46: a sum over (x, a, b) with a > b is minus the
    # sum over (x, b + 1, a - 1); SymPy forgets the sign when a is oo
    # (Sum(1/(x**2 + 1), (x, oo, 0)).doit() is positive there)
    import mpmath
    from sympy_extras.core import Integer, Rational as Rat, oo as inf, pi
    from sympy_extras.functions import exp as exp_, factorial as factorial_, log, tanh, zeta
    from sympy_extras.functions.elementary.piecewise import Piecewise as NativePiecewise
    from sympy_extras.concrete.expr_with_limits import Sum
    from sympy_extras.concrete.summations import eval_sum_residue
    t, c, s = Symbol('t'), Symbol('c'), Symbol('s')
    assert summation(1/(t**2 + 1), (t, inf, Integer(0))) == Rat(1, 2) - pi/(2*tanh(pi))
    assert Sum(1/(t**2 + 1), (t, inf, Integer(0))).doit() == Rat(1, 2) - pi/(2*tanh(pi))
    forward = mpmath.nsum(lambda v: 1/(v**2 + 1), [1, mpmath.inf])
    assert abs(complex((Rat(1, 2) - pi/(2*tanh(pi))).evalf()) + complex(forward)) < 1e-12
    assert (summation(c**t/factorial_(t), (t, inf, Integer(0))) - (1 - exp_(c))).expand() == 0
    assert summation(Integer(-1)**t/t, (t, inf, Integer(0))) == log(2)
    assert summation(Rat(1, 2)**t, (t, inf, Integer(-1))) == -2
    assert summation(1/t, (t, inf, Integer(0))) == -inf
    assert summation(Integer(-1)**t/(2*t + 1), (t, inf, Integer(-1))) == -pi/4
    reversed_zeta = summation(1/t**s, (t, inf, Integer(0)))
    assert isinstance(reversed_zeta, NativePiecewise) and reversed_zeta.args[0].expr == -zeta(s)
    assert eval_sum_residue(1/(t**2 + 1), (t, inf, Integer(-1))) == -Rat(1, 2) - pi/(2*tanh(pi))
    # the finite sums, right before: -(2 + 3 + 4)
    assert summation(t, (t, Integer(5), Integer(1))) == -9
    assert summation(t, (t, Integer(5), Integer(4))) == 0
