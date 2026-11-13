"""Oracle tests of the native rewrite (exp and cos targets) against SymPy."""
import random

import sympy

from sympy_extras.interop import expr_from_sympy, to_sympy
from sympy_extras.core.mul import neg
from sympy_extras.functions.elementary.exponential import Exp
from sympy_extras.functions.elementary.trigonometric import Cos, Sin

s = sympy
x, y = s.symbols('x y')
p = s.Symbol('p', positive=True)


def test_rewrite_exp_cos_and_sin_match_sympy() -> None:
    # the sin target is native too (it went to SymPy through the bridge)
    rng = random.Random(2)
    fs = [s.sin, s.cos, s.tan, s.cot, s.sec, s.csc, s.sinh, s.cosh, s.tanh, s.coth, s.sech, s.csch, s.exp, s.log,
          s.sqrt, s.gamma, s.asin, lambda a: a**2, lambda a: 2**a, lambda a: 1 / a]
    for _ in range(80):
        e = rng.choice([x, y, p, 2 * x, x + y, s.pi / 3, s.sqrt(3) / 2])
        for _ in range(rng.randint(1, 3)):
            e = rng.choice(fs)(e)
        for theirs, ours in ((s.exp, Exp), (s.cos, Cos), (s.sin, Sin)):
            assert str(to_sympy(expr_from_sympy(e).rewrite(ours), evaluate=False)) == str(e.rewrite(theirs)), e


def test_neg_does_not_flatten_like_sympy() -> None:
    # SymPy's __neg__ builds Mul._from_args: an unevaluated exp stays
    u = s.exp(2 * s.log(p), evaluate=False)
    assert str(to_sympy(neg(expr_from_sympy(u)), evaluate=False)) == str(-u)
    for e in (x, x + y, -2 * x, 2 * x * y, s.I * x, s.zoo * x):
        assert str(to_sympy(neg(expr_from_sympy(e)), evaluate=False)) == str(-e)


def test_count_ops_matches_sympy() -> None:
    from sympy_extras.core.function import count_ops
    rng = random.Random(9)
    atoms = [x, y, s.Integer(2), s.Rational(-1, 2), s.pi, -x]
    fs = [s.sin, s.exp, s.log, s.sqrt, lambda a: a**2, lambda a: 1 / a, lambda a: -a, lambda a: a**s.Rational(-3, 2)]

    def rnd(d: int) -> sympy.Expr:
        if d == 0 or rng.random() < .2:
            return rng.choice(atoms)
        q = rng.random()
        if q < .4:
            return rnd(d - 1) + rnd(d - 1)
        if q < .75:
            return rnd(d - 1) * rnd(d - 1)
        return rng.choice(fs)(rnd(d - 1))
    for _ in range(300):
        e = rnd(5)
        for visual in (False, True):
            assert str(count_ops(expr_from_sympy(e), visual=visual)) == str(s.count_ops(e, visual=visual)), e


def test_traversals_rebuild_only_changed_nodes() -> None:
    from sympy_extras.core.function import expand_log
    from sympy_extras.simplify.simplify import logcombine
    e = s.factor_terms(2 * y * s.log(2) + 4, sign=False)  # 2*(y*log(2) + 2)
    assert str(to_sympy(expand_log(expr_from_sympy(e), deep=True), evaluate=False)) == str(s.expand_log(e, deep=True))
    assert str(to_sympy(logcombine(expr_from_sympy(e)), evaluate=False)) == str(s.logcombine(e))


def test_add_recreates_nested_terms_and_normal_keeps_numeric_denominator() -> None:
    from sympy_extras.core.add import add
    nested = s.Mul._from_args((s.S.Half, x * y * (2 * y + 1)))
    ours = add(expr_from_sympy(nested), expr_from_sympy(y**2))
    assert [a.args for a in ours.args] == [tuple(expr_from_sympy(b) for b in a.args) for a in (nested + y**2).args]
    e = x * y * (y + s.S.Half) + y**2
    assert str(to_sympy(expr_from_sympy(e).normal(), evaluate=False)) == str(e.normal())


def test_rewrite_gamma_matches_sympy() -> None:
    from sympy_extras.functions.special.gamma_functions import Gamma
    n, k = s.symbols('n k', integer=True)
    m = s.Symbol('m', negative=True)
    cases: list[sympy.Expr] = [
        s.factorial(n) / s.factorial(n - 3), s.binomial(n, k), s.rf(x, k), s.ff(x, 3), s.rf(m, y)**2, s.ff(p, k),
        s.factorial2(n)**2, s.subfactorial(x), s.catalan(n), s.beta(x, p), s.rf(k, k)**2 + s.factorial(k)]
    for e in cases:
        for piecewise in (True, False):
            ours = to_sympy(expr_from_sympy(e).rewrite(Gamma, piecewise=piecewise), evaluate=False)
            assert str(ours) == str(e.rewrite(s.gamma, piecewise=piecewise)), (e, piecewise)


def test_expand_complex_matches_sympy() -> None:
    from sympy_extras.core.function import expand, expand_complex
    r = s.Symbol('r', real=True)
    cases: list[sympy.Expr] = [
        x + s.I * y, s.exp(x), s.sin(2 * p * x**2), s.log(p + y), y / (y + s.I), s.sqrt(x) + 1, p * (s.sqrt(x) + 1),
        s.cosh(r) * s.sqrt(s.Abs(x)), r**2 + x**2 + s.log(p + y), s.log(s.pi * r * x * (1 + s.I)), s.tan(s.pi * x * y / 2),
        2 * s.Abs(y)]
    for e in cases:
        assert str(to_sympy(expand_complex(expr_from_sympy(e)), evaluate=False)) == str(s.expand_complex(e)), e
        assert str(to_sympy(expand(expr_from_sympy(e), complex=True), evaluate=False)) == str(s.expand(e, complex=True)), e


def test_expand_trig_matches_sympy() -> None:
    from sympy_extras.core.function import expand, expand_trig
    z = s.Symbol('z')
    cases: list[sympy.Expr] = [
        s.sin(x + y), s.cos(3 * x), s.sin(4 * x), s.tan(x + y + z), s.cot(2 * x + y), s.tan(3 * x), s.cot(3 * x),
        s.sinh(2 * x + y), s.cosh(3 * x), s.tanh(x + y), s.tanh(3 * x), s.coth(x + y), s.coth(2 * x), s.sec(x + y),
        s.csc(2 * x), s.sech(x + y + z), s.csch(3 * x), x + s.cot(2 * x + y) * s.sech(x + y + z)]
    for e in cases:
        assert str(to_sympy(expand_trig(expr_from_sympy(e)), evaluate=False)) == str(s.expand_trig(e)), e
        assert str(to_sympy(expand(expr_from_sympy(e), trig=True), evaluate=False)) == str(s.expand(e, trig=True)), e


def test_expand_func_matches_sympy() -> None:
    from sympy_extras.core.function import expand, expand_func
    n, k = s.symbols('n k', integer=True)
    R = s.Rational
    cases: list[sympy.Expr] = [
        s.gamma(x + 3), s.gamma(R(7, 3)), s.gamma(x + R(5, 2)), s.polygamma(0, x + 2), s.polygamma(1, 3 * x),
        s.polygamma(0, R(1, 3)), s.polygamma(-1, x), s.loggamma(R(1, 3)), s.digamma(x - 2), s.trigamma(2 * x),
        s.beta(x, y), s.binomial(n, 3), s.binomial(x, k), s.harmonic(n + 2), s.harmonic(x - 3), s.erfc(x), s.erfi(x),
        s.erf2(x, y), s.expint(2, x), s.zeta(x, 3), s.polylog(-2, x), s.polylog(1, x), s.GoldenRatio * x,
        s.besselj(2, x) + s.gamma(x + 1), s.loggamma(R(7, 3))]
    for e in cases:
        assert str(to_sympy(expand_func(expr_from_sympy(e)), evaluate=False)) == str(s.expand_func(e)), e
        assert str(to_sympy(expand(expr_from_sympy(e), func=True), evaluate=False)) == str(s.expand(e, func=True)), e


def test_named_rules_table_matches_sympy() -> None:
    # the classes sent to SymPy for 'tractable'/'intractable' are exactly
    # those whose SymPy class has the rule
    from sympy_extras.core.rewriting import _SYMPY_NAMED_RULES
    from sympy_extras.interop import _NATIVE_TO_SYMPY, _fill_function_tables
    _fill_function_tables()
    for name, table in _SYMPY_NAMED_RULES.items():
        have = {n.__name__ for n, s in _NATIVE_TO_SYMPY.items()
                if any('_eval_rewrite_as_' + name in vars(k) for k in s.__mro__)}
        assert have == set(table), (name, have ^ set(table))


def test_own_meromorphic_table_matches_sympy() -> None:
    import sympy
    from sympy.core.function import Function as SympyFunction
    from sympy_extras.core.function import _SYMPY_OWN_MEROMORPHIC
    from sympy_extras.interop import _NATIVE_TO_SYMPY, _fill_function_tables
    _fill_function_tables()
    have = {n.__name__ for n, s in _NATIVE_TO_SYMPY.items()
            if any('_eval_is_meromorphic' in vars(k) for k in s.__mro__
                   if k not in (SympyFunction, sympy.Expr, sympy.Basic) and k.__module__.startswith('sympy.functions'))}
    assert have == set(_SYMPY_OWN_MEROMORPHIC), have ^ set(_SYMPY_OWN_MEROMORPHIC)


def test_rewrite_rules_table_matches_sympy() -> None:
    # sympy_extras.core._rewrite_rules: SymPy's _eval_rewrite_as_<name>
    # targets by class, and the SymPy names of the targets (function_name)
    from sympy.core.basic import Basic as SympyBasic
    from sympy_extras.core._rewrite_rules import OWN_REWRITE, REWRITE_TARGETS
    from sympy_extras.interop import _NATIVE_TO_SYMPY, _fill_function_tables
    _fill_function_tables()
    for native, theirs in _NATIVE_TO_SYMPY.items():
        targets = {m[len('_eval_rewrite_as_'):] for k in theirs.__mro__ for m in k.__dict__
                   if m.startswith('_eval_rewrite_as_')}
        assert REWRITE_TARGETS[native.__name__] == targets, native
        assert (native.__name__ in OWN_REWRITE) == any('_eval_rewrite' in k.__dict__ for k in theirs.__mro__
                                                       if k is not SympyBasic), native
        assert native.function_name == theirs.__name__, (native, theirs)


def test_expand_func_table_and_bessel_recurrence_match_sympy() -> None:
    import sympy
    from sympy.core.basic import Basic as SympyBasic
    from sympy_extras.core.function import _SYMPY_EXPAND_FUNC, expand_func
    from sympy_extras.interop import _NATIVE_TO_SYMPY, _fill_function_tables, expr_from_sympy, to_sympy
    _fill_function_tables()
    have = {n.__name__ for n, s in _NATIVE_TO_SYMPY.items()
            if any('_eval_expand_func' in k.__dict__ for k in s.__mro__ if k is not SympyBasic)}
    extra = {'Hyper', 'MeijerG', 'GoldenRatioType', 'TribonacciConstantType'}
    assert have | extra == set(_SYMPY_EXPAND_FUNC), have ^ (set(_SYMPY_EXPAND_FUNC) - extra)
    z, n = sympy.Symbol('z'), sympy.Symbol('n')
    half = sympy.Rational(1, 2)
    for f in (sympy.besselj, sympy.bessely, sympy.besseli, sympy.besselk, sympy.hankel1, sympy.hankel2):
        for nu in (0, 1, 2, 3, 5, -2, -3, half, 3*half, -5*half, n, n + 2):
            e = f(nu, z)
            assert str(to_sympy(expand_func(expr_from_sympy(e)), evaluate=False)) == str(sympy.expand_func(e)), e


def test_expand_func_loggamma_spherical_lerchphi_match_sympy() -> None:
    # the ported _eval_expand_func of loggamma (with a Sum), jn/yn/hn1/hn2
    # (spherical_bessel_fn) and lerchphi
    import re
    import sympy
    from sympy_extras.core.function import expand_func
    from sympy_extras.interop import expr_from_sympy, to_sympy
    z, s, a = sympy.symbols('z s a')
    R, I, pi = sympy.Rational, sympy.I, sympy.pi
    cases = [sympy.loggamma(R(7, 3)), sympy.loggamma(R(-5, 3)), sympy.loggamma(R(11, 4)), sympy.loggamma(R(-1, 2))]
    cases += [f(n, z) for f in (sympy.jn, sympy.yn, sympy.hn1, sympy.hn2) for n in range(-3, 4)]
    cases += [sympy.lerchphi(1, s, a), sympy.lerchphi(z, -2, 3), sympy.lerchphi(z, s, R(1, 2)), sympy.lerchphi(z, s, R(3, 2)),
              sympy.lerchphi(z, s, R(-1, 3)), sympy.lerchphi(-1, s, a), sympy.lerchphi(I, s, a),
              sympy.lerchphi(sympy.exp(2*pi*I/3), s, a), sympy.lerchphi(z, s, a)]

    def norm(text: str) -> str:
        return re.sub(r'_[kt]\w*', '_d', text)
    for e in cases:
        got = to_sympy(expand_func(expr_from_sympy(e)), evaluate=False)
        assert norm(str(got)) == norm(str(sympy.expand_func(e))), e


def test_log_and_special_rewrites_match_sympy() -> None:
    # functions/elementary/_rewrites._log_rules and functions/special/_rewrites
    import sympy
    from sympy_extras.functions.elementary.exponential import Log
    from sympy_extras.functions.elementary.trigonometric import Tan
    import sympy_extras.functions.special.bessel as B
    import sympy_extras.functions.special.error_functions as ER
    from sympy_extras.interop import expr_from_sympy, to_sympy
    x, z, n = sympy.symbols('x z n')
    half = sympy.Rational(1, 2)
    pairs: list[tuple[sympy.Expr, object, type]] = []
    for f in (sympy.asin, sympy.acos, sympy.atan, sympy.acot, sympy.asec, sympy.acsc, sympy.asinh, sympy.acosh,
              sympy.atanh, sympy.acoth, sympy.asech, sympy.acsch):
        for a in (x, 2*x + 1, sympy.Rational(1, 3)):
            pairs.append((f(a, evaluate=False), sympy.log, Log))
    pairs += [(sympy.cot(x), sympy.tan, Tan), (sympy.besselj(n, z), sympy.jn, B.SphericalJ),
              (sympy.besselj(n, z), sympy.besseli, B.BesselI), (sympy.besseli(n, z), sympy.besselj, B.BesselJ),
              (sympy.bessely(half, z), sympy.besselj, B.BesselJ), (sympy.bessely(n, z), sympy.besselj, B.BesselJ),
              (sympy.besselk(half, z), sympy.besselj, B.BesselJ), (sympy.erfc(z), sympy.erf, ER.Erf),
              (sympy.erfi(z), sympy.erf, ER.Erf), (sympy.erf2(z, 2*z), sympy.erf, ER.Erf),
              (sympy.expint(1, z), sympy.Ei, ER.EiNode), (sympy.expint(3, z), sympy.Ei, ER.EiNode),
              (sympy.Ei(z), sympy.expint, ER.Expint)]
    for e, theirs, ours in pairs:
        got = to_sympy(expr_from_sympy(e).rewrite(ours), evaluate=False)
        assert str(got) == str(e.rewrite(theirs)), (e, ours)


def test_pochhammer_binomial_and_eta_rewrites_are_native() -> None:
    # the rules of ff, rf, binomial between themselves and of dirichlet_eta
    # in zeta are ported (they went to SymPy through the bridge); checked
    # against SymPy's
    from sympy_extras.core.rewriting import _rule_for
    from sympy_extras.functions.combinatorial.factorials import Binomial, FallingFactorial, RisingFactorial
    from sympy_extras.functions.special.zeta_functions import DirichletEta, Zeta
    k = s.Symbol('k', integer=True)
    a = s.Symbol('a')
    cases = [(s.ff(x, k, evaluate=False), s.binomial, FallingFactorial, Binomial),
             (s.ff(x, k, evaluate=False), s.rf, FallingFactorial, RisingFactorial),
             (s.rf(x, k, evaluate=False), s.binomial, RisingFactorial, Binomial),
             (s.rf(x, k, evaluate=False), s.ff, RisingFactorial, FallingFactorial),
             (s.binomial(x, k, evaluate=False), s.ff, Binomial, FallingFactorial),
             (s.ff(x, y, evaluate=False), s.binomial, FallingFactorial, Binomial),
             (s.dirichlet_eta(y, evaluate=False), s.zeta, DirichletEta, Zeta),
             (s.dirichlet_eta(y, a, evaluate=False), s.zeta, DirichletEta, Zeta),
             (s.dirichlet_eta(3, evaluate=False), s.zeta, DirichletEta, Zeta)]
    for e, theirs, cls, target in cases:
        assert _rule_for(cls, target) is not None
        assert str(expr_from_sympy(e).rewrite(target)) == str(e.rewrite(theirs)), e


def test_own_aseries_table_matches_sympy() -> None:
    # the classes without an _eval_aseries of their own raise SymPy's
    # PoleError natively at an argument going to an infinity (they went
    # to SymPy's _eval_nseries through the bridge)
    from sympy.core.function import Function as SympyFunction
    from sympy_extras.core.function import _SYMPY_OWN_ASERIES
    from sympy_extras.interop import _NATIVE_TO_SYMPY, _fill_function_tables
    _fill_function_tables()
    have = {n.__name__ for n, theirs in _NATIVE_TO_SYMPY.items()
            if any('_eval_aseries' in vars(k) for k in theirs.__mro__ if k is not SympyFunction)}
    assert have == set(_SYMPY_OWN_ASERIES), have ^ set(_SYMPY_OWN_ASERIES)


def test_airy_series_at_infinity_raises_pole_error() -> None:
    # SymPy's airyai has no _eval_aseries: Function._eval_nseries raises
    # PoleError (natively now)
    from sympy.testing.pytest import raises
    from sympy_extras.core import Integer, Rational, Symbol
    from sympy_extras.core.coreerrors import PoleError
    from sympy_extras.functions.special.bessel import AiryAi
    w, a = Symbol('w', positive=True), Symbol('a', positive=True)
    node = AiryAi(-a / w**Rational(2, 3))
    raises(PoleError, lambda: node._eval_nseries(w, 2, None, Integer(0)))
    sw, sa = s.symbols('w a', positive=True)
    raises(s.PoleError, lambda: s.airyai(-sa / sw**s.Rational(2, 3))._eval_nseries(sw, 2, None, 0))


def test_sign_rewrites_as_abs_natively() -> None:
    # sign._eval_rewrite_as_Abs is ported (it went to SymPy through the
    # bridge); checked against SymPy's
    from sympy_extras.functions.elementary.complexes import Abs
    b, c = s.symbols('b c')
    for e in [s.sign(x), s.sign(s.sqrt(x**2 + y**2)), s.sign((b + 2*c*x)/c), s.sign(x) + s.sign(y)**2]:
        assert str(to_sympy(expr_from_sympy(e).rewrite(Abs), evaluate=False)) == str(e.rewrite(s.Abs)), e


def test_rewrite_table_of_the_nodes_with_parameters_matches_sympy() -> None:
    # Hyper, MeijerG, ExprCondPair, IndexedRoot, the sets, ... were not in
    # the table, so every rewrite of them asked SymPy through the bridge,
    # which answered None (they have no rule for a function target)
    from sympy.core.basic import Basic as SympyBasic
    from sympy.functions.elementary.piecewise import ExprCondPair as SympyExprCondPair
    from sympy_extras.core._rewrite_rules import OWN_REWRITE, REWRITE_TARGETS
    from sympy_extras.integrals.risch.risch_sympy import NonElementaryIntegral
    from sympy_extras.polys.cad.indexedroot_sympy import IndexedRoot
    from sympy_extras.solvers.isolation_sympy import TranscendentalRoot
    theirs: dict[str, type[SympyBasic]] = {
        'Hyper': s.hyper, 'MeijerG': s.meijerg, 'ExprCondPair': SympyExprCondPair, 'IndexedRoot': IndexedRoot,
        'ComplexRootOf': s.CRootOf, 'Integral': s.Integral, 'RootSum': s.RootSum, 'Max': s.Max, 'Min': s.Min,
        'UnevaluatedExpr': s.UnevaluatedExpr, 'AlgebraicNumber': s.AlgebraicNumber,
        'TranscendentalRoot': TranscendentalRoot, 'NonElementaryIntegral': NonElementaryIntegral,
        'ConditionSet': s.ConditionSet, 'ImageSet': s.ImageSet, 'Interval': s.Interval, 'FiniteSet': s.FiniteSet,
        'PointSet': s.FiniteSet, 'Union': s.Union, 'Intersection': s.Intersection, 'Complement': s.Complement,
        'Contains': s.Contains, 'ProductSet': s.ProductSet}
    for name, cls in theirs.items():
        targets = {m[len('_eval_rewrite_as_'):] for k in cls.__mro__ for m in k.__dict__
                   if m.startswith('_eval_rewrite_as_')}
        assert REWRITE_TARGETS[name] == targets, name
        assert (name in OWN_REWRITE) == any('_eval_rewrite' in k.__dict__ for k in cls.__mro__
                                            if k is not SympyBasic), name


def test_rewrite_of_nodes_without_a_rule_matches_sympy() -> None:
    # the nodes are rebuilt with their arguments rewritten, as SymPy does
    from sympy_extras.functions.elementary.trigonometric import Cos
    from sympy_extras.functions.special.gamma_functions import Gamma
    a, u, n = s.symbols('a u n')
    cases = [(s.hyper((a / 2, s.sin(a)), (s.Rational(1, 2),), -1), s.cos, Cos),
             (s.hyper((s.Rational(1, 6),), (s.Rational(1, 2),), s.factorial(a)), s.gamma, Gamma),
             (s.meijerg(((-1, n + 1), ()), ((), (-1, 0)), u / a * s.factorial(n)), s.gamma, Gamma),
             (s.Piecewise((s.factorial(a), a > 0), (0, True)), s.gamma, Gamma),
             (s.Max(s.factorial(a), x), s.gamma, Gamma)]
    for e, theirs, ours in cases:
        assert str(to_sympy(expr_from_sympy(e).rewrite(ours), evaluate=False)) == str(e.rewrite(theirs)), e


def _native_classes() -> dict[str, list[type]]:
    """The native classes of the package by name (the modules which define
    nodes imported)."""
    import importlib
    import pkgutil
    import sympy_extras
    from sympy_extras.core.basic import Basic
    for m in pkgutil.walk_packages(sympy_extras.__path__, 'sympy_extras.'):
        if 'tests' in m.name or m.name.endswith('_sympy') or 'interop' in m.name:
            continue
        try:
            importlib.import_module(m.name)
        except Exception:
            pass
    out: dict[str, list[type]] = {}
    todo: list[type] = [Basic]
    seen: set[type] = set()
    while todo:
        c = todo.pop()
        for sub in c.__subclasses__():
            if sub in seen:
                continue
            seen.add(sub)
            if sub.__module__.startswith('sympy_extras') and 'interop' not in sub.__module__:
                out.setdefault(sub.__name__, []).append(sub)
            todo.append(sub)
    return out


def test_every_sympy_rewrite_rule_is_native() -> None:
    # the nodes of a native class without a native rule for a target went
    # to SymPy's rule through the bridge; now every rule of SymPy 1.14 is
    # ported but those needing a class the core has not (a piecewise
    # function with boolean values, PowerSet); 'tractable' and
    # 'intractable' are rewrite_tractable's
    from sympy_extras.core._rewrite_rules import REWRITE_TARGETS
    from sympy_extras.core.rewriting import _SYMPY_NAMED_RULES, has_rule
    left_out = {('Piecewise', 'ITE'), ('ITE', 'Piecewise'), ('FiniteSet', 'PowerSet'), ('PointSet', 'PowerSet')}
    classes = _native_classes()
    missing: list[tuple[str, str]] = []
    for name, targets in REWRITE_TARGETS.items():
        for cls in classes.get(name, []):
            for t in targets:
                if t in _SYMPY_NAMED_RULES:
                    assert name in _SYMPY_NAMED_RULES[t], (name, t)
                elif not has_rule(cls, t) and (name, t) not in left_out:
                    missing.append((name, t))
    assert missing == [], missing


def test_every_sympy_expand_func_rule_is_native() -> None:
    from sympy_extras.core.function import _SYMPY_EXPAND_FUNC
    from sympy_extras.functions._expand_func import rules
    table = rules()
    classes = _native_classes()
    missing = [n for n in _SYMPY_EXPAND_FUNC for c in classes.get(n, []) if c not in table]
    assert missing == [], missing


def test_series_fallbacks_match_sympy() -> None:
    # the native functions without a series rule expand by the port of
    # Function._eval_nseries, as their SymPy classes do; the other nodes
    # without one raise NotImplementedError, as SymPy's Expr._eval_nseries
    # (they went to SymPy through the bridge)
    from sympy.core.function import Function as SympyFunction
    from sympy_extras.core.expr import Expr
    from sympy_extras.core.function import DefinedFunction
    from sympy_extras.functions._series_rules import rules
    from sympy_extras.interop import _NATIVE_TO_SYMPY, _fill_function_tables
    _fill_function_tables()
    series_rules = rules()
    for native, theirs in _NATIVE_TO_SYMPY.items():
        owner = next(k for k in native.__mro__ if '_eval_nseries' in k.__dict__)
        their_owner = next(k for k in theirs.__mro__ if '_eval_nseries' in k.__dict__)
        if owner is DefinedFunction and native.__name__ not in series_rules and not native.generic_nseries:
            assert their_owner is SympyFunction, (native, their_owner)
        if owner is Expr:
            assert their_owner is s.Expr, (native, their_owner)
