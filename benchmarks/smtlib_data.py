"""SMT-LIB ``QF_NRA`` problems as SymPy formulas.

The Meti-Tarski family of SMT-LIB (proof obligations of MetiTarski on
polynomial bounds of special functions, submitted by Jovanovic and
de Moura) is mirrored in the ``dreal/benchmarks`` repository on GitHub,
which is cloned sparsely on first use (about 7700 files, each with its
``:status sat|unsat``). Only the formulas are read; nothing is stored in
this repository.
"""
from __future__ import annotations

import pathlib
import subprocess
from typing import Optional, Union

from sympy import Rational, Symbol, Eq, Ne, And, Or, Not, Implies, true, false
from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.logic.boolalg import Boolean, ITE

REPOSITORY = "https://github.com/dreal/benchmarks"
SUBDIRECTORY = "smt2/smt-lib/meti-tarski"
CACHE = pathlib.Path(__file__).parent / '.cache' / 'dreal-benchmarks'

SExpr = Union[str, list['SExpr']]


def fetch() -> Optional[pathlib.Path]:
    """The directory of the Meti-Tarski files, cloned on first use."""
    target = CACHE / SUBDIRECTORY
    if target.is_dir() and any(target.iterdir()):
        return target
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(['git', 'clone', '--depth', '1', '--filter=blob:none', '--sparse',
                        REPOSITORY, str(CACHE)], check=True, capture_output=True, timeout=600)
        subprocess.run(['git', '-C', str(CACHE), 'sparse-checkout', 'set', SUBDIRECTORY],
                       check=True, capture_output=True, timeout=600)
    except (subprocess.SubprocessError, OSError):
        return None
    return target if target.is_dir() else None


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c.isspace():
            i += 1
        elif c == ';':
            while i < n and text[i] != '\n':
                i += 1
        elif c in '()':
            tokens.append(c)
            i += 1
        elif c == '|':
            j = text.index('|', i + 1)
            tokens.append(text[i:j + 1])
            i = j + 1
        elif c == '"':
            j = text.index('"', i + 1)
            tokens.append(text[i:j + 1])
            i = j + 1
        else:
            j = i
            while j < n and not text[j].isspace() and text[j] not in '()':
                j += 1
            tokens.append(text[i:j])
            i = j
    return tokens


def parse(tokens: list[str]) -> list[SExpr]:
    """All top level s-expressions."""
    stack: list[list[SExpr]] = [[]]
    for token in tokens:
        if token == '(':
            stack.append([])
        elif token == ')':
            done = stack.pop()
            stack[-1].append(done)
        else:
            stack[-1].append(token)
    return stack[0]


class Problem:
    """A parsed problem: its formula, variables and expected status."""

    def __init__(self, name: str, formula: Boolean, variables: list[Symbol], status: str) -> None:
        self.name = name
        self.formula = formula
        self.variables = variables
        self.status = status

    def __repr__(self) -> str:
        return "Problem(%s, %s, %d variables)" % (self.name, self.status, len(self.variables))


def _number(token: str) -> Optional[Rational]:
    try:
        return Rational(token)
    except (ValueError, TypeError):
        return None


class _Translator:
    def __init__(self) -> None:
        self.symbols: dict[str, Symbol] = {}
        self.bindings: list[dict[str, Basic]] = []

    def lookup(self, name: str) -> Optional[Basic]:
        for scope in reversed(self.bindings):
            if name in scope:
                return scope[name]
        return self.symbols.get(name)

    def term(self, e: SExpr) -> Expr:
        if isinstance(e, str):
            value = _number(e)
            if value is not None:
                return value
            bound = self.lookup(e)
            if isinstance(bound, Expr):
                return bound
            raise ValueError("unknown term %s" % e)
        head, args = e[0], e[1:]
        if head == 'let':
            self._bind(args[0])
            try:
                return self.term(args[1])
            finally:
                self.bindings.pop()
        if head == '+':
            return sum((self.term(a) for a in args), Rational(0))
        if head == '*':
            result: Expr = Rational(1)
            for a in args:
                result = result*self.term(a)
            return result
        if head == '-':
            if len(args) == 1:
                return -self.term(args[0])
            result = self.term(args[0])
            for a in args[1:]:
                result = result - self.term(a)
            return result
        if head == '/':
            result = self.term(args[0])
            for a in args[1:]:
                result = result/self.term(a)
            return result
        if head == 'ite':
            raise ValueError("ite terms are not supported")
        raise ValueError("unknown operator %s" % head)

    def _bind(self, bindings: SExpr) -> None:
        """Push the scope of a ``let``."""
        scope: dict[str, Basic] = {}
        if not isinstance(bindings, list):
            raise ValueError("bad let")
        for binding in bindings:
            if not isinstance(binding, list) or len(binding) != 2 or not isinstance(binding[0], str):
                raise ValueError("bad let binding")
            name, value = binding
            if not isinstance(name, str):
                raise ValueError("bad let binding")
            try:
                scope[name] = self.term(value)
            except ValueError:
                scope[name] = self.formula(value)
        self.bindings.append(scope)

    def formula(self, e: SExpr) -> Boolean:
        if isinstance(e, str):
            if e == 'true':
                return true
            if e == 'false':
                return false
            bound = self.lookup(e)
            if isinstance(bound, Boolean):
                return bound
            raise ValueError("unknown formula %s" % e)
        head, args = e[0], e[1:]
        if head == 'let':
            self._bind(args[0])
            try:
                return self.formula(args[1])
            finally:
                self.bindings.pop()
        if head in ('and', 'or'):
            parts = [self.formula(a) for a in args]
            return And(*parts) if head == 'and' else Or(*parts)
        if head == 'not':
            return Not(self.formula(args[0]))
        if head == '=>':
            return Implies(self.formula(args[0]), self.formula(args[1]))
        if head == 'ite':
            return ITE(self.formula(args[0]), self.formula(args[1]), self.formula(args[2]))
        if head in ('<=', '<', '>=', '>', '=', 'distinct'):
            terms = [self.term(a) for a in args]
            relations: list[Boolean] = []
            for a, b in zip(terms, terms[1:]):
                if head == '<=':
                    relations.append(a <= b)
                elif head == '<':
                    relations.append(a < b)
                elif head == '>=':
                    relations.append(a >= b)
                elif head == '>':
                    relations.append(a > b)
                elif head == '=':
                    relations.append(Eq(a, b))
                else:
                    relations.append(Ne(a, b))
            return And(*relations)
        raise ValueError("unknown connective %s" % head)


def translate(text: str, name: str = '') -> Optional[Problem]:
    """The problem in an SMT-LIB file, or ``None`` when it uses something
    beyond ``QF_NRA`` formulas."""
    try:
        commands = parse(tokenize(text))
    except (ValueError, IndexError):
        return None
    translator = _Translator()
    status = 'unknown'
    assertions: list[Boolean] = []
    for command in commands:
        if not isinstance(command, list) or not command:
            continue
        head = command[0]
        if head == 'set-info' and len(command) >= 3 and command[1] == ':status' and isinstance(command[2], str):
            status = command[2]
        elif head == 'declare-fun' and len(command) == 4 and isinstance(command[1], str):
            if command[2] != [] or command[3] != 'Real':
                return None
            translator.symbols[command[1]] = Symbol(command[1])
        elif head == 'declare-const' and len(command) == 3 and isinstance(command[1], str):
            if command[2] != 'Real':
                return None
            translator.symbols[command[1]] = Symbol(command[1])
        elif head == 'assert':
            try:
                assertions.append(translator.formula(command[1]))
            except (ValueError, NotImplementedError, ZeroDivisionError):
                return None
    if not assertions:
        return None
    variables = sorted(translator.symbols.values(), key=lambda s: s.name)
    return Problem(name, And(*assertions), variables, status)


def load(limit: int = 0) -> list[pathlib.Path]:
    """The paths of the problem files."""
    directory = fetch()
    if directory is None:
        return []
    paths = sorted(directory.glob('*.smt2'))
    return paths[:limit] if limit else paths
