r"""The engine of the step-by-step solutions: rules as generators, steps
as dataclasses, and the solver which drives them.

The design is that of ``sympy.integrals.manualintegrate`` in SymPy's
development branch [1]_, made generic over the kind of problem and typed:

* A **problem** is a small immutable, hashable value (a derivative to
  take, an integral, an equation).
* A **step** is a frozen dataclass, a subclass of :class:`Step`, which
  records one operation: the rule is the class, the fields are what the
  rule needs (the factors of a product rule, the substitution of a chain
  rule) and the steps of its subproblems. The result of a derivation is
  therefore a *tree of steps*. Steps carry no sentences: the name of the
  class is the identifier of the rule, and the wording belongs to
  whatever renders the tree.
* A **rule** is a *generator function*. It does not call the solver:
  when it needs a subproblem solved it ``yield``\ s that problem and
  receives the solved step at the ``yield`` expression; a rule may yield
  any number of times, deciding what to ask next from the answers so far.
  Its ``return`` value is the step it builds, or ``None`` when it does
  not apply.
* The :class:`StepSolver` owns the state of one run and performs the
  recursive calls on behalf of the rules (``generator.send``): the
  subproblems on the current path (a rule which leads back to a problem
  being solved is cut: the cyclic integrations by parts), the steps
  already found (a subproblem reached twice is solved once), and the
  limit on the depth.

**Alternatives.** For each problem the domain proposes its candidate
rules in order. By default the first rule which solves the problem
completely is kept. With ``branch=True`` every candidate which works is
kept and the results are collected in an alternative step (a subclass of
:class:`AlternativeStep`), at every level of the tree.

**Collapsible detail.** :func:`format_steps` prints a tree down to a
given depth, each level indented under its parent; steps marked
``minor`` (arithmetic, rewriting) are skipped unless asked for, their
substeps taking their place.

References
==========

.. [1] SymPy Development Team, ``sympy/integrals/manualintegrate.py``
       (the ``IntegrationSolver`` and its generator rules), SymPy master,
       2026; SymPy issue 11088, *Step by Step solutions*.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, ClassVar, Generator, Generic, Optional, Sequence, TypeVar

__all__ = ['Step', 'AlternativeStep', 'Context', 'StepSolver', 'atomic', 'format_steps', 'count_steps']


class Step:
    """One operation of a derivation. The subclasses are frozen
    dataclasses; ``rule`` (the name of the class) identifies the rule."""

    #: a step of little interest (arithmetic, a rewriting), hidden by default when the tree is printed
    minor: ClassVar[bool] = False

    def substeps(self) -> tuple[Step, ...]:
        """The steps of the subproblems, in order."""
        return ()

    def contains_dont_know(self) -> bool:
        """Whether some part of the derivation was left undone."""
        return any(step.contains_dont_know() for step in self.substeps())

    @property
    def rule(self) -> str:
        """The identifier of the rule."""
        return type(self).__name__

    def statement(self) -> str:
        """What the step establishes, as a formula without words (``d/dx
        sin(x) = cos(x)``)."""
        raise NotImplementedError


class AlternativeStep(Step):
    """Several derivations of the same problem: ``substeps()`` are the
    alternatives, the first one being the default."""


P = TypeVar('P')
S = TypeVar('S', bound=Step)


@dataclass(frozen=True)
class Context:
    """What a rule may know about the run: whether alternatives are
    collected, and the depth of the problem in the tree."""

    branch: bool
    depth: int


#: a rule: a generator which yields subproblems, receives their steps and returns its own step or ``None``
RuleGenerator = Generator[P, S, Optional[S]]
Rule = Callable[[P, Context], Generator[P, S, Optional[S]]]


def atomic(function: Callable[[P], Optional[S]]) -> Callable[[P, Context], Generator[P, S, Optional[S]]]:
    """A rule without subproblems, from a plain function of the
    problem."""
    def rule(problem: P, context: Context) -> Generator[P, S, Optional[S]]:
        return function(problem)
        yield problem               # never reached: makes the function a generator
    rule.__name__ = function.__name__
    return rule


class StepSolver(Generic[P, S]):
    """Solves problems of type ``P`` into steps of type ``S``.

    Parameters
    ==========

    candidates : the rules to try on a problem, in order of preference.
    dont_know : the step which leaves a problem undone.
    alternative : the step which holds several derivations of a problem.
    branch : collect every workable rule (``True``) or keep the first.
    max_depth : the limit on the depth of nested subproblems (``None``
        for no limit); a deeper subproblem is left undone.
    """

    def __init__(self, candidates: Callable[[P], Sequence[Callable[[P, Context], Generator[P, S, Optional[S]]]]],
                 dont_know: Callable[[P], S], alternative: Callable[[P, Sequence[S]], S], branch: bool = False,
                 max_depth: Optional[int] = 100) -> None:
        self.candidates = candidates
        self.dont_know = dont_know
        self.alternative = alternative
        self.branch: bool = branch
        self.max_depth: Optional[int] = max_depth
        self._active: list[P] = []
        self._solved: dict[P, S] = {}
        self._starved: bool = False

    def solve(self, problem: P) -> S:
        """The step tree of a problem (possibly with parts left undone:
        see :meth:`Step.contains_dont_know`)."""
        known = self._solved.get(problem)
        if known is not None:
            return known
        if self.max_depth is not None and len(self._active) >= self.max_depth:
            self._starved = True
            return self.dont_know(problem)
        if problem in self._active:
            # the attempt leads around in a loop
            return self.dont_know(problem)
        outer, self._starved = self._starved if self._active else False, False
        self._active.append(problem)
        try:
            result = self._apply(problem)
            if not self._starved:
                # a result found without running out of depth does not depend on where it was asked
                self._solved[problem] = result
        finally:
            self._active.pop()
            self._starved = outer or self._starved
        return result

    def _apply(self, problem: P) -> S:
        context = Context(self.branch, len(self._active) - 1)
        complete: list[S] = []
        partial: Optional[S] = None
        for rule in self.candidates(problem):
            result = self._run(rule(problem, context))
            if result is None:
                continue
            if result.contains_dont_know():
                if partial is None:
                    partial = result
                continue
            if not self.branch:
                return result
            if result not in complete:
                complete.append(result)
        if len(complete) == 1:
            return complete[0]
        if complete:
            return self.alternative(problem, complete)
        return partial if partial is not None else self.dont_know(problem)

    def _run(self, generator: Generator[P, S, Optional[S]]) -> Optional[S]:
        """Drive a rule: solve the subproblems it yields and send the
        steps back, until it returns."""
        try:
            request = next(generator)
            while True:
                request = generator.send(self.solve(request))
        except StopIteration as finished:
            value: Optional[S] = finished.value
            return value


def count_steps(step: Step) -> int:
    """The number of steps of a tree (the alternatives all counted)."""
    return 1 + sum(count_steps(child) for child in step.substeps())


def format_steps(step: Step, depth: Optional[int] = None, minor: bool = False, indent: str = '  ') -> str:
    """The tree as text: one line ``Rule: statement`` for each step, the
    substeps indented below it, down to ``depth`` levels (all of them for
    ``None``). Steps marked ``minor`` are left out, their substeps taking
    their place, unless ``minor`` is true. The alternatives are numbered.
    """
    lines: list[str] = []

    def walk(current: Step, level: int, prefix: str) -> None:
        if depth is not None and level > depth:
            return
        shown = minor or not type(current).minor or not current.substeps()
        if shown:
            lines.append('%s%s%s: %s' % (indent*level, prefix, current.rule, current.statement()))
        below = level + 1 if shown else level
        if isinstance(current, AlternativeStep):
            for number, alternative in enumerate(current.substeps(), start=1):
                walk(alternative, below, '[%d] ' % number)
        else:
            for child in current.substeps():
                walk(child, below, '' if shown else prefix)

    walk(step, 0, '')
    return '\n'.join(lines)
