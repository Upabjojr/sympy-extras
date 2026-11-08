"""Oracle tests of trigsimp and exptrigsimp against SymPy."""
import sympy

from sympy_extras.core.interop import expr_from_sympy, to_sympy
from sympy_extras.simplify.trigsimp import exptrigsimp, trigsimp

sx, sy, sa = sympy.symbols('x y a')


def _same(ours: object, theirs: object) -> bool:
    return str(ours) == str(theirs)


def test_trigsimp_matches_sympy() -> None:
    s = sympy
    cases: list[sympy.Expr] = [
        s.sin(sx)**2 + s.cos(sx)**2, 2 * s.sin(sx) * s.cos(sx), s.sin(sx) / s.cos(sx),
        s.cosh(sx)**2 - s.sinh(sx)**2, 1 - s.cos(sx)**2, s.sin(sx)**4 - s.cos(sx)**4, s.tan(sx)**2 + 1,
        s.sin(sx) * s.cos(sy) + s.cos(sx) * s.sin(sy), sa * s.sin(sx)**2 + sa * s.cos(sx)**2,
        s.cos(2 * sx) + 2 * s.sin(sx)**2, s.sec(sx)**2 - 1, (1 + s.tan(sx)**2) * s.cos(sx)**2,
        s.sin(sx)**3 * s.cos(sx) + s.sin(sx) * s.cos(sx)**3, s.sinh(sx) / s.cosh(sx),
        s.log(2 * s.sin(sx) * s.cos(sx)), s.sqrt(1 - s.sin(sx)**2), s.cos(sx)**2 * s.tan(sx),
        3 * s.sin(sx)**2 + 3 * s.cos(sx)**2 + sy]
    for e in cases:
        assert _same(to_sympy(trigsimp(expr_from_sympy(e)), evaluate=False), s.trigsimp(e)), e


def test_exptrigsimp_matches_sympy() -> None:
    s = sympy
    cases: list[sympy.Expr] = [
        s.exp(sx) + s.exp(-sx), (s.exp(sx) - s.exp(-sx)) / 2, s.exp(2 * sx) / (1 + s.exp(2 * sx)),
        s.cosh(sx) + s.sinh(sx), (1 - s.exp(-sx)) / (1 + s.exp(-sx)), s.exp(sx) * s.cos(sy)]
    for e in cases:
        assert _same(to_sympy(exptrigsimp(expr_from_sympy(e)), evaluate=False), s.exptrigsimp(e)), e
