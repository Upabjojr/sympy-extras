"""The ``rewrite`` rules of the Bessel, spherical Bessel and Airy functions (SymPy's
``_eval_rewrite_as_<name>`` methods of bessel.py), as a typed
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
