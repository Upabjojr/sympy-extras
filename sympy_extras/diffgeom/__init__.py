"""Differential geometry on the typed core: the curvature of a metric
given by its components (:mod:`sympy_extras.diffgeom.metric`), tensors
with index positions on it (:mod:`sympy_extras.diffgeom.tensors`), curves
and surfaces in space (:mod:`sympy_extras.diffgeom.classical`)."""
from .classical import Curve, Surface
from .metric import Metric
from .tensors import Tensor

__all__ = ['Metric', 'Tensor', 'Curve', 'Surface']
