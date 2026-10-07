"""Stage-one fixed-reference transverse-isotropic material support.

The package is deliberately independent from the legacy MPM kernels while the
center-state plumbing is being validated.  The NumPy functions are the
reference implementation used by tests and small material-point experiments;
the Warp helpers in :mod:`kernels` expose the same fiber terms to future
matrix-free kernels.
"""

from .types import AnisotropicMaterialParams, AnisotropicCenterState
from .state import AnisotropicCenterBuffer
from .implicit import center_trial_F, center_velocity_gradient, center_residual, center_matvec, matrix_free_pcg
from .adapter import AnisotropicCenterAdapter
from .solver import AnisotropicLiteImplicitSolver
from .device import query_gpu_memory, select_lowest_memory_device
from .constitutive import (
    a0_to_A0,
    average_structure_tensors,
    structure_tensor_mixing,
    fiber_invariant,
    energy,
    pk1,
    kirchhoff,
    dP_fiber_apply,
    dP_apply_finite_difference,
)

__all__ = [
    "AnisotropicMaterialParams",
    "AnisotropicCenterState",
    "AnisotropicCenterBuffer",
    "center_trial_F",
    "center_velocity_gradient",
    "center_residual",
    "center_matvec",
    "matrix_free_pcg",
    "AnisotropicCenterAdapter",
    "AnisotropicLiteImplicitSolver",
    "select_lowest_memory_device",
    "query_gpu_memory",
    "a0_to_A0",
    "average_structure_tensors",
    "structure_tensor_mixing",
    "fiber_invariant",
    "energy",
    "pk1",
    "kirchhoff",
    "dP_fiber_apply",
    "dP_apply_finite_difference",
]
