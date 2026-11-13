"""The ``rewrite`` rules of the orthogonal polynomials (SymPy's
``_eval_rewrite_as_Sum`` and ``_eval_rewrite_as_polynomial``), as a typed
table for :func:`sympy_extras.core.rewriting.rewrite`.
"""
from __future__ import annotations

from sympy_extras.core.rewriting import Rule, RuleKey

__all__ = ['rules']


def rules() -> dict[RuleKey, Rule]:
    """The rules keyed by (the class of the node, the target class or
    SymPy name)."""
    table: dict[RuleKey, Rule] = {}
    return table
