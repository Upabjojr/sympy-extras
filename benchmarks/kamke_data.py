"""The Kamke and Murphy collections of ordinary differential equations as
transcribed in the test suite of Maxima's ``contrib_ode`` package.

The equations are downloaded at run time from Maxima's source repository
(SourceForge) and cached; nothing of the GPL-licensed files is stored in
this repository. Only the equations themselves (mathematical facts from
E. Kamke, *Differentialgleichungen: Lösungsmethoden und Lösungen*, and
G. M. Murphy, *Ordinary Differential Equations and Their Solutions*) are
used, not Maxima's solutions.
"""
from __future__ import annotations

import pathlib
import re
import urllib.request
from typing import Optional

from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.function import Function, Derivative
from sympy.core.symbol import Symbol
from sympy.parsing.sympy_parser import parse_expr

BASE = ("https://sourceforge.net/p/maxima/code/ci/master/tree/share/contrib/"
        "diffequations/tests/%s.mac?format=raw")
FILES = {
    'kamke1': ['rtestode_kamke_1_%d' % k for k in range(1, 7)],
    'kamke2': ['rtestode_kamke_2_%d' % k for k in range(1, 6)],
    'murphy1': ['rtestode_murphy_1_%d' % k for k in range(1, 7)],
    'murphy2': ['rtestode_murphy_2_%d' % k for k in range(1, 6)],
}
CACHE = pathlib.Path(__file__).parent / '.cache' / 'kamke'

_ENTRY = re.compile(r"\(pn_\(\s*(\d+)\s*\),ans:contrib_ode\(eqn:(.*?),y,x\)", re.S)
_COMMENT = re.compile(r"/\*.*?\*/", re.S)


class ODEEntry:
    """One equation of a collection."""

    def __init__(self, collection: str, number: int, source: str, equation: Expr, order: int) -> None:
        self.collection = collection
        self.number = number
        self.source = source
        self.equation = equation
        self.order = order

    @property
    def name(self) -> str:
        return "%s.%d" % (self.collection, self.number)

    def __repr__(self) -> str:
        return "ODEEntry(%s, %s)" % (self.name, self.equation)


def fetch(name: str) -> Optional[str]:
    """The text of a test file, from the cache or the network."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / (name + '.mac')
    if path.exists():
        return path.read_text()
    try:
        with urllib.request.urlopen(BASE % name, timeout=60) as response:
            text = response.read().decode('utf-8', errors='replace')
    except OSError:
        return None
    if 'contrib_ode' not in text:
        return None
    path.write_text(text)
    return text


x = Symbol('x')
y = Function('y')
_FUNCTIONS: dict[str, object] = {name: Function(name) for name in ('f', 'g', 'h', 'phi', 'psi', 'tg')}
_NAMES = {
    'bessel_j': 'besselj', 'bessel_y': 'bessely', 'bessel_i': 'besseli',
    'bessel_k': 'besselk', 'abs': 'Abs', '%e': 'E', '%i': 'I', '%pi': 'pi',
    'atan': 'atan', 'asin': 'asin', 'acos': 'acos', 'sinh': 'sinh', 'cosh': 'cosh',
    'tanh': 'tanh', 'erf': 'erf', 'gamma': 'gamma',
}


def translate(text: str) -> Optional[Expr]:
    """A Maxima expression of the collections as a SymPy expression in
    ``y(x)``, or ``None`` when it cannot be translated."""
    s = text.strip()
    s = re.sub(r"'diff\(\s*y\s*,\s*x\s*,\s*(\d+)\s*\)", r"Derivative(y(x), (x, \1))", s)
    s = re.sub(r"'diff\(\s*y\s*,\s*x\s*\)", r"Derivative(y(x), x)", s)
    s = re.sub(r"'diff\(\s*(\w+)\(x\)\s*,\s*x\s*,\s*(\d+)\s*\)", r"Derivative(\1(x), (x, \2))", s)
    s = re.sub(r"'diff\(\s*(\w+)\(x\)\s*,\s*x\s*\)", r"Derivative(\1(x), x)", s)
    if "'" in s or "diff" in s:
        return None
    s = re.sub(r"\by\b(?!\()", "y(x)", s)
    for old, new in _NAMES.items():
        s = re.sub(re.escape(old) + r"(?![A-Za-z_0-9])", new, s)
    s = s.replace('^', '**')
    local: dict[str, object] = dict(_FUNCTIONS)
    local['y'] = y
    local['x'] = x
    local['Derivative'] = Derivative
    try:
        expr = parse_expr(s, local_dict=local)
    except Exception:
        return None
    if not isinstance(expr, Expr) or not expr.has(y(x)):
        return None
    return expr


def _order(expr: Basic) -> int:
    orders = [sum(int(n) for _, n in d.variable_count) for d in expr.atoms(Derivative) if d.expr == y(x)]
    return max(orders, default=0)


def load(collections: tuple[str, ...] = ('kamke1', 'kamke2')) -> list[ODEEntry]:
    """The entries of the collections which could be downloaded and
    translated."""
    entries: list[ODEEntry] = []
    for collection in collections:
        for name in FILES[collection]:
            text = fetch(name)
            if text is None:
                continue
            text = _COMMENT.sub('', text)
            for number, source in _ENTRY.findall(text):
                expr = translate(source)
                if expr is None:
                    continue
                order = _order(expr)
                if order == 0:
                    continue
                entries.append(ODEEntry(collection, int(number), source, expr, order))
    return entries


if __name__ == '__main__':
    loaded = load(('kamke1', 'kamke2', 'murphy1', 'murphy2'))
    by: dict[str, int] = {}
    for e in loaded:
        by.setdefault(e.collection, 0)
        by[e.collection] += 1
    print(by, sum(by.values()))
