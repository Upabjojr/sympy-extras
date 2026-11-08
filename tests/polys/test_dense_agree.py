"""The Rust port of the dense polynomial modules against the Python
package: every function on random inputs over ``ZZ``, ``QQ``, ``GF(7)``
and ``GF(2**31 - 1)``, with equal results or the same exception."""

from __future__ import annotations

import importlib
import random
from fractions import Fraction
from typing import Any, Callable

import pytest

from ..conftest import load, stable_seed

#: (module, function, argument kinds); ``f``, ``g``: polynomials, ``c``: an
#: element, ``p``: a nonzero element, ``n``: a small nonnegative int
SIGNATURES: list[tuple[str, str, str]] = [
    ("densebasic", "dup_strip", "f"),
    ("densebasic", "dup_LC", "f"),
    ("densebasic", "dup_TC", "f"),
    ("densebasic", "dup_to_dict", "f"),
    ("densebasic", "dup_terms", "f"),
    ("densebasic", "dup_reverse", "f"),
    ("densebasic", "dup_slice", "fnn"),
    ("densebasic", "dup_monomial", "cn"),
    ("densebasic", "dup_is_one", "f"),
    ("densearith", "dup_add", "fg"),
    ("densearith", "dup_sub", "fg"),
    ("densearith", "dup_neg", "f"),
    ("densearith", "dup_mul", "fg"),
    ("densearith", "dup_sqr", "f"),
    ("densearith", "dup_pow", "fn"),
    ("densearith", "dup_add_term", "fcn"),
    ("densearith", "dup_sub_term", "fcn"),
    ("densearith", "dup_mul_term", "fcn"),
    ("densearith", "dup_add_ground", "fc"),
    ("densearith", "dup_mul_ground", "fc"),
    ("densearith", "dup_quo_ground", "fc"),
    ("densearith", "dup_exquo_ground", "fc"),
    ("densearith", "dup_lshift", "fn"),
    ("densearith", "dup_rshift", "fn"),
    ("densearith", "dup_div", "fg"),
    ("densearith", "dup_rem", "fg"),
    ("densearith", "dup_exquo", "fg"),
    ("densearith", "dup_pdiv", "fg"),
    ("densearith", "dup_pexquo", "fg"),
    ("densetools", "dup_diff", "fn"),
    ("densetools", "dup_eval", "fc"),
    ("densetools", "dup_shift", "fc"),
    ("densetools", "dup_compose", "fg"),
    ("densetools", "dup_mirror", "f"),
    ("densetools", "dup_scale", "fc"),
    ("densetools", "dup_content", "f"),
    ("densetools", "dup_primitive", "f"),
    ("densetools", "dup_decompose_content", "f"),
    ("densetools", "dup_monic", "f"),
    ("densetools", "dup_trunc", "fp"),
    ("densetools", "dup_integrate", "fn"),
    ("euclidtools", "dup_euclid", "fg"),
    ("euclidtools", "dup_prs", "fg"),
    ("euclidtools", "dup_subresultants", "fg"),
    ("euclidtools", "dup_gcd", "fg"),
    ("euclidtools", "dup_lcm", "fg"),
    ("euclidtools", "dup_gcdex", "fg"),
    ("euclidtools", "dup_invert", "fg"),
    ("euclidtools", "dup_resultant", "fg"),
    ("euclidtools", "dup_prs_resultant", "fg"),
    ("euclidtools", "dup_inner_subresultants", "fg"),
    ("euclidtools", "dup_discriminant", "f"),
    ("sqfreetools", "dup_sqf_list", "f"),
    ("sqfreetools", "dup_sqf_part", "f"),
    ("sqfreetools", "dup_sqf_p", "f"),
    ("sqfreetools", "dup_sqf_norm_content", "f"),
]

#: the functions of the ordered domains only
ORDERED: list[tuple[str, str, str]] = [
    ("densearith", "dup_max_norm", "f"),
    ("densearith", "dup_l1_norm", "f"),
    ("densetools", "dup_sign_variations", "f"),
]

DOMAINS: list[str] = ["ZZ", "QQ", "GF(7)", "GF(2147483647)"]

TRIALS: int = 60


def domain(package: str, name: str) -> Any:
    d = importlib.import_module(package + ".polys.domains")
    if name.startswith("GF("):
        return d.GF(int(name[3:-1]))
    return getattr(d, name)


def element(rng: random.Random, name: str, nonzero: bool = False) -> Any:
    while True:
        if name == "ZZ":
            value: Any = rng.randint(-9, 9)
        elif name == "QQ":
            value = Fraction(rng.randint(-9, 9), rng.randint(1, 5))
        else:
            value = rng.randrange(int(name[3:-1]))
        if value or not nonzero:
            return value


def polynomial(rng: random.Random, name: str) -> list[Any]:
    """A random polynomial without leading zeros, often with repeated
    factors or a common factor with another one (same seed)."""
    degree = rng.choice([-1, 0, 1, 2, 3, 4, 5, 6, 8])
    f = [element(rng, name) for _ in range(degree + 1)]
    while f and not f[0]:
        f.pop(0)
    return f


def structured(rng: random.Random, name: str, mul: Callable[[Any, Any], Any]) -> list[Any]:
    """A product of small polynomials, so that gcds and square-free parts
    are not trivial."""
    result: list[Any] = polynomial(rng, name) or [element(rng, name, True)]
    for _ in range(rng.randint(0, 2)):
        factor = polynomial(rng, name)
        if factor:
            result = mul(result, factor)
    return result


def outcome(call: Callable[[], Any]) -> Any:
    try:
        return ("ok", call())
    except (ValueError, ZeroDivisionError) as e:
        return ("raises", type(e).__name__)


#: the functions called on small random polynomials only
PLAIN: set[str] = {"dup_compose", "dup_pow", "dup_resultant", "dup_discriminant"}


def arguments(rng: random.Random, function: str, kinds: str, name: str, mul: Callable[[Any, Any], Any]) -> list[Any]:
    shared = structured(rng, name, mul)
    args: list[Any] = []
    if function in PLAIN:
        # compositions, powers and (over QQ, where the pseudo-remainders are
        # not normalised) resultants of products grow too fast: small inputs
        return [polynomial(rng, name) if kind in "fg" else rng.randint(0, 5) for kind in kinds]
    for kind in kinds:
        if kind in "fg":
            args.append(mul(shared, polynomial(rng, name) or [element(rng, name, True)]) if rng.random() < 0.5
                        else structured(rng, name, mul))
        elif kind == "c":
            args.append(element(rng, name))
        elif kind == "p":
            args.append(element(rng, name, nonzero=True))
        else:
            args.append(rng.randint(0, 5))
    return args


@pytest.mark.parametrize("name", DOMAINS)
@pytest.mark.parametrize("module, function, kinds", SIGNATURES + ORDERED,
                         ids=[s[1] for s in SIGNATURES + ORDERED])
def test_agree(rust: Any, module: str, function: str, kinds: str, name: str) -> None:
    if (module, function, kinds) in ORDERED and name.startswith("GF("):
        pytest.skip("not an ordered domain")
    K_py, K_rs = domain("sympy_extras", name), domain("sympy_extras_rs", name)
    python_function = getattr(importlib.import_module("sympy_extras.polys." + module), function)
    rust_function = getattr(importlib.import_module("sympy_extras_rs.polys." + module), function)
    from sympy_extras.polys.densearith import dup_mul
    rng = random.Random(stable_seed(module, function, name))
    for _ in range(TRIALS):
        args = arguments(rng, function, kinds, name, lambda f, g: dup_mul(f, g, K_py))
        expected = outcome(lambda: python_function(*args, K_py))
        got = outcome(lambda: rust_function(*args, K_rs))
        assert got == expected, (function, name, args)


@pytest.mark.parametrize("seed", range(5))
def test_qq_only(rust: Any, seed: int) -> None:
    from sympy_extras.polys.densetools import dup_clear_denoms
    from sympy_extras.polys.domains import QQ
    from sympy_extras.polys.euclidtools import dup_qq_gcd
    from sympy_extras.polys.densearith import dup_mul
    rs_t = load("sympy_extras_rs.polys.densetools")
    rs_e = load("sympy_extras_rs.polys.euclidtools")
    rs_QQ = load("sympy_extras_rs.polys.domains").QQ
    rng = random.Random(seed)
    for _ in range(TRIALS):
        h = structured(rng, "QQ", lambda f, g: dup_mul(f, g, QQ))
        f = dup_mul(h, polynomial(rng, "QQ") or [Fraction(1)], QQ)
        g = dup_mul(h, polynomial(rng, "QQ") or [Fraction(1)], QQ)
        assert rs_t.dup_clear_denoms(f, rs_QQ) == dup_clear_denoms(f, QQ)
        assert rs_e.dup_qq_gcd(f, g) == dup_qq_gcd(f, g)


def test_karatsuba(rust: Any) -> None:
    """Products above the threshold over QQ (the generic code path; ZZ and
    GF(p) use the dense kernels)."""
    from sympy_extras.polys.densearith import dup_mul, dup_sqr
    from sympy_extras.polys.domains import QQ
    rs = load("sympy_extras_rs.polys.densearith")
    rs_QQ = load("sympy_extras_rs.polys.domains").QQ
    rng = random.Random(1)
    for degree in (31, 32, 33, 64, 100):
        f = [Fraction(rng.randint(-9, 9), rng.randint(1, 4)) for _ in range(degree + 1)]
        g = [Fraction(rng.randint(-9, 9), rng.randint(1, 4)) for _ in range(degree + 7)]
        f[0], g[0] = Fraction(1), Fraction(2)
        assert rs.dup_mul(f, g, rs_QQ) == dup_mul(f, g, QQ)
        assert rs.dup_sqr(f, rs_QQ) == dup_sqr(f, QQ)


@pytest.mark.parametrize("source", DOMAINS)
@pytest.mark.parametrize("target", DOMAINS)
def test_convert(rust: Any, source: str, target: str) -> None:
    from sympy_extras.polys.densebasic import dup_convert
    rs = load("sympy_extras_rs.polys.densebasic")
    rng = random.Random(7)
    for _ in range(TRIALS):
        f = polynomial(rng, source)
        expected = outcome(lambda: dup_convert(f, domain("sympy_extras", source), domain("sympy_extras", target)))
        got = outcome(lambda: rs.dup_convert(f, domain("sympy_extras_rs", source), domain("sympy_extras_rs", target)))
        assert got == expected, (source, target, f)


@pytest.mark.parametrize("name", DOMAINS)
@pytest.mark.parametrize("method", ["add", "sub", "mul", "exquo", "quo", "rem", "divmod", "gcd", "lcm", "gcdex",
                                    "invert", "canonical_unit", "neg", "pow"])
def test_domain_methods(rust: Any, name: str, method: str) -> None:
    K_py, K_rs = domain("sympy_extras", name), domain("sympy_extras_rs", name)
    rng = random.Random(3)
    unary = method in ("invert", "canonical_unit", "neg")
    for _ in range(TRIALS):
        a, b = element(rng, name), element(rng, name)
        if method == "pow":
            n = rng.randint(-3, 6)
            args: tuple[Any, ...] = (a, n)
        elif unary:
            args = (a,)
        else:
            args = (a, b)
        expected = outcome(lambda: getattr(K_py, method)(*args))
        got = outcome(lambda: getattr(K_rs, method)(*args))
        assert got == expected, (method, name, args)


def test_integer_round_trip(rust: Any) -> None:
    """Python ints through the two's complement bytes of the binding, at the
    byte and word boundaries and for large values."""
    ZZ, QQ = load("sympy_extras_rs.polys.domains").ZZ, load("sympy_extras_rs.polys.domains").QQ
    values = [0, 1, -1, 127, 128, -128, -129, 255, 256, -256, 2**31, 2**63 - 1, 2**63, -2**63, -2**63 - 1,
              2**64, -2**64, 2**127, -2**128 + 1, 3**500, -7**333]
    for n in values:
        assert ZZ.add(n, 0) == n, n
        assert QQ.add(Fraction(n, 3), 0) == Fraction(n, 3), n
