"""The Rust port of ``domains/polynomialring`` and ``domains/fractionfield``
against the Python package: the domain methods of ``K[y]`` and ``K(y)``, and
the dense algorithms over ``K[y]`` (bivariate polynomials as polynomials
over a polynomial ring)."""

from __future__ import annotations

import importlib
import random
from fractions import Fraction
from typing import Any, Callable

import pytest

from ..conftest import load


def domains(impl: str, name: str = "") -> Any:
    load(impl)
    return importlib.import_module(impl + ".polys.domains" + ("." + name if name else ""))


def ground(impl: str, kind: str) -> Any:
    d = domains(impl)
    return d.GF(7) if kind == "GF(7)" else getattr(d, kind)


def poly_ring(impl: str, kind: str) -> Any:
    return domains(impl, "polynomialring").PolynomialRing(ground(impl, kind), 'y')


def frac_field(impl: str, kind: str) -> Any:
    return domains(impl, "fractionfield").FractionField(poly_ring(impl, kind))


def coefficient(rng: random.Random, kind: str) -> Any:
    if kind == "ZZ":
        return rng.randint(-4, 4)
    if kind == "QQ":
        return Fraction(rng.randint(-4, 4), rng.randint(1, 3))
    return rng.randrange(7)


def ground_poly(rng: random.Random, kind: str, degree: int = 3) -> list[Any]:
    f = [coefficient(rng, kind) for _ in range(rng.randint(0, degree + 1))]
    while f and not f[0]:
        f.pop(0)
    return f


def normal(value: Any) -> Any:
    """Fractions as ``(numer, denom)`` pairs, recursively."""
    if hasattr(value, "numer") and hasattr(value, "denom") and not isinstance(value, Fraction):
        return ("frac", normal(value.numer), normal(value.denom))
    if isinstance(value, (list, tuple)):
        return type(value)(normal(v) for v in value)
    return value


def outcome(call: Callable[[], Any]) -> Any:
    try:
        return ("ok", normal(call()))
    except (ValueError, ZeroDivisionError, NotImplementedError) as e:
        return ("raises", type(e).__name__)


def test_known(impl: str) -> None:
    R = poly_ring(impl, "ZZ")
    assert R.name == "ZZ[y]" and repr(R) == "ZZ[y]"
    assert R.mul([1, 1], [1, -1]) == [1, 0, -1]
    assert R.gcd([2, 0, -2], [4, 4]) == [2, 2]
    F = frac_field(impl, "ZZ")
    assert F.name == "ZZ(y)"
    assert normal(F.new([1, 0, -1], [1, 1])) == ("frac", [1, -1], [1])
    assert normal(F.add(F.new([1], [1, 0]), F.new([1], [1, 1]))) == ("frac", [2, 1], [1, 1, 0])
    assert normal(F.invert(F.new([2, 0], [1, 1]))) == ("frac", [1, 1], [2, 0])


METHODS = ["add", "sub", "mul", "exquo", "quo", "rem", "gcd", "lcm", "gcdex", "invert", "canonical_unit", "neg",
           "is_unit", "is_one", "is_zero"]


@pytest.mark.parametrize("kind", ["ZZ", "QQ", "GF(7)"])
@pytest.mark.parametrize("which", ["ring", "field"])
@pytest.mark.parametrize("method", METHODS)
def test_domain_methods_agree(rust: Any, kind: str, which: str, method: str) -> None:
    make = poly_ring if which == "ring" else frac_field
    Ks = [make(impl, kind) for impl in ("sympy_extras", "sympy_extras_rs")]
    rng = random.Random(hash((kind, which, method)) & 0xffff)
    unary = method in ("invert", "canonical_unit", "neg", "is_unit", "is_one", "is_zero")
    for _ in range(40):
        a, b = ground_poly(rng, kind), ground_poly(rng, kind)
        # the elements of QQ are Fractions (a plain int would make Python's
        # QQ.canonical_unit return a float)
        unit = Fraction(1) if kind == "QQ" else 1
        c, d = ground_poly(rng, kind) or [unit], ground_poly(rng, kind) or [unit]
        results = []
        for K in Ks:
            if which == "ring":
                x, y = a, b
            else:
                try:
                    x, y = K.new(a, c), K.new(b, d)
                except ZeroDivisionError:
                    results.append("zero denominator")
                    continue
            args = (x,) if unary else (x, y)
            results.append(outcome(lambda: getattr(K, method)(*args)))
        assert results[0] == results[1], (method, kind, which, a, b, c, d)


FUNCTIONS: list[tuple[str, str, str]] = [
    ("densearith", "dup_add", "fg"), ("densearith", "dup_mul", "fg"), ("densearith", "dup_sub", "fg"),
    ("densearith", "dup_div", "fg"), ("densearith", "dup_pdiv", "fg"), ("densearith", "dup_exquo", "fg"),
    ("densetools", "dup_diff", "fn"), ("densetools", "dup_compose", "fg"), ("densetools", "dup_content", "f"),
    ("densetools", "dup_primitive", "f"), ("densetools", "dup_monic", "f"),
    ("euclidtools", "dup_gcd", "fg"), ("euclidtools", "dup_lcm", "fg"), ("euclidtools", "dup_resultant", "fg"),
    ("euclidtools", "dup_prs_resultant", "fg"), ("euclidtools", "dup_subresultants", "fg"),
    ("euclidtools", "dup_discriminant", "f"),
    ("sqfreetools", "dup_sqf_list", "f"), ("sqfreetools", "dup_sqf_part", "f"), ("sqfreetools", "dup_sqf_p", "f"),
]


@pytest.mark.parametrize("kind", ["ZZ", "QQ", "GF(7)"])
@pytest.mark.parametrize("module, function, kinds", FUNCTIONS, ids=[f[1] for f in FUNCTIONS])
def test_dense_over_polynomial_ring_agrees(rust: Any, kind: str, module: str, function: str, kinds: str) -> None:
    """Bivariate polynomials in ``K[y][x]``: products sharing a factor, so
    that the gcds and the square-free parts are not trivial."""
    Ks = [poly_ring(impl, kind) for impl in ("sympy_extras", "sympy_extras_rs")]
    functions = [getattr(importlib.import_module(impl + ".polys." + module), function)
                 for impl in ("sympy_extras", "sympy_extras_rs")]
    from sympy_extras.polys.densearith import dup_mul
    rng = random.Random(hash((kind, function)) & 0xffff)

    def bivariate(degree: int) -> list[list[Any]]:
        f = [ground_poly(rng, kind, 2) for _ in range(rng.randint(1, degree + 1))]
        while f and not f[0]:
            f.pop(0)
        return f

    for _ in range(25):
        common = bivariate(2)
        args: list[Any] = []
        for k in kinds:
            if k in "fg":
                p = bivariate(2)
                args.append(dup_mul(p, common, Ks[0]) if common and p and rng.random() < 0.7 else p)
            else:
                args.append(rng.randint(0, 2))
        results = [outcome(lambda: fn(*args, K)) for fn, K in zip(functions, Ks)]
        assert results[0] == results[1], (function, kind, args)
