"""The ``rewrite`` rules of the complex, integer, piecewise and
discontinuous functions (``Abs``, ``sign``, ``floor``, ``Max``,
``Heaviside``, ``DiracDelta``, ``SingularityFunction``, ...) and of the
nodes which are not functions (the logical connectives, the sums and
products), as typed tables for :func:`sympy_extras.core.rewriting.rewrite`.
"""
from __future__ import annotations

from sympy_extras.core.rewriting import NodeRule, Rule, RuleKey

__all__ = ['rules', 'node_rules']


def rules() -> dict[RuleKey, Rule]:
    """The rules of expressions keyed by (the class of the node, the
    target class or SymPy name)."""
    table: dict[RuleKey, Rule] = {}
    return table


def node_rules() -> dict[RuleKey, NodeRule]:
    """The rules of the nodes which are not expressions, or whose result
    is not one, keyed by (the class of the node, the target)."""
    table: dict[RuleKey, NodeRule] = {}
    return table
