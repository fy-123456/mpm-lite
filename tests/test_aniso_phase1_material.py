import numpy as np
import unittest

from engine.aniso_phase1 import (
    AnisotropicCenterState,
    AnisotropicMaterialParams,
    a0_to_A0,
    average_structure_tensors,
    dP_fiber_apply,
    energy,
    kirchhoff,
    pk1,
    structure_tensor_mixing,
)


class TestAnisoPhase1Material(unittest.TestCase):
  def test_direction_is_normalized_and_unoriented(self):
    A = a0_to_A0([2.0, -1.0, 0.5])
    assert np.allclose(A, a0_to_A0([-2.0, 1.0, -0.5]))
    assert np.allclose(A, A.T)
    assert np.linalg.eigvalsh(A).min() >= -1.0e-14


  def test_zero_direction_is_rejected(self):
    with self.assertRaisesRegex(ValueError, "non-zero"):
        AnisotropicMaterialParams(1.0, 2.0, 3.0, [0.0, 0.0, 0.0])


  def test_structure_tensor_average_preserves_uniform_direction(self):
    A = a0_to_A0([1.0, 2.0, 3.0])
    out = average_structure_tensors(np.stack([A, A]), np.array([2.0, 1.0]))
    assert np.allclose(out, A)


  def test_structure_tensor_mixing_detects_multiple_directions(self):
    pure = a0_to_A0([1.0, 0.0, 0.0])
    mixed = average_structure_tensors(
        np.stack([a0_to_A0([1.0, 0.0, 0.0]), a0_to_A0([0.0, 1.0, 0.0])]),
        np.array([1.0, 1.0]),
    )
    self.assertAlmostEqual(structure_tensor_mixing(pure), 0.0, places=14)
    self.assertAlmostEqual(structure_tensor_mixing(mixed), 0.5, places=14)


  def test_material_point_stress_and_reference_state(self):
    params = AnisotropicMaterialParams(2.0, 3.0, 10.0, [1.0, 0.0, 0.0])
    I = np.eye(3)
    assert np.allclose(pk1(I, params.A0, params), 0.0, atol=1.0e-12)
    assert np.isclose(energy(I, params.A0, params), 0.0, atol=1.0e-12)
    F = np.diag([1.2, 1.0, 1.0])
    # Fiber contribution is 2*k_f*(I4-1)*F*A0 = 10.56 e_x⊗e_x.
    assert np.isclose(pk1(F, params.A0, params)[0, 0], 10.56 + pk1(F, np.zeros((3, 3)), params)[0, 0])


  def test_fiber_directional_derivative_matches_finite_difference(self):
    params = AnisotropicMaterialParams(7.0, 11.0, 13.0, [1.0, 2.0, -1.0])
    F = np.array([[1.2, 0.1, 0.0], [0.0, 0.9, 0.2], [0.1, 0.0, 1.1]])
    dF = np.array([[0.1, -0.2, 0.3], [0.0, 0.2, -0.1], [0.05, 0.1, 0.0]])
    eps = 1.0e-6
    numerical = (pk1(F + eps * dF, params.A0, params) - pk1(F - eps * dF, params.A0, params)) / (2.0 * eps)
    analytic_fiber = dP_fiber_apply(F, params.A0, dF, params)
    numerical_fiber = (2.0 * params.k_f * (np.einsum("ij,ij", (F + eps * dF) @ params.A0, F + eps * dF) - 1.0) * ((F + eps * dF) @ params.A0) - 2.0 * params.k_f * (np.einsum("ij,ij", (F - eps * dF) @ params.A0, F - eps * dF) - 1.0) * ((F - eps * dF) @ params.A0)) / (2.0 * eps)
    assert np.allclose(analytic_fiber, numerical_fiber, rtol=2.0e-5, atol=2.0e-7)
    assert np.isfinite(numerical).all()


  def test_rigid_rotation_is_objective_for_fiber_energy(self):
    params = AnisotropicMaterialParams(2.0, 3.0, 9.0, [1.0, 0.0, 0.0])
    theta = 0.73
    R = np.array([[np.cos(theta), -np.sin(theta), 0.0], [np.sin(theta), np.cos(theta), 0.0], [0.0, 0.0, 1.0]])
    F = np.diag([1.25, 0.9, 1.1])
    assert np.isclose(energy(R @ F, params.A0, params), energy(F, params.A0, params))
    assert np.allclose(kirchhoff(R @ F, params.A0, params), R @ kirchhoff(F, params.A0, params) @ R.T, atol=1.0e-10)

  def test_uniaxial_directional_stiffness_is_distinct(self):
    F = np.diag([1.1, 1.0, 1.0])
    angles = (0.0, np.pi / 4.0, np.pi / 2.0)
    stresses = []
    for angle in angles:
        params = AnisotropicMaterialParams(2.0, 3.0, 40.0, [np.cos(angle), np.sin(angle), 0.0])
        stresses.append(pk1(F, params.A0, params)[0, 0])
    self.assertGreater(stresses[0], stresses[1])
    self.assertGreater(stresses[1], stresses[2])


  def test_simple_shear_and_elastic_unloading(self):
    params = AnisotropicMaterialParams(4.0, 6.0, 18.0, [1.0, 0.0, 0.0])
    loaded = np.array([[1.0, 0.25, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    self.assertGreater(energy(loaded, params.A0, params), 0.0)
    self.assertTrue(np.isfinite(pk1(loaded, params.A0, params)).all())
    unloaded = np.eye(3)
    self.assertAlmostEqual(energy(unloaded, params.A0, params), 0.0, places=12)
    self.assertTrue(np.allclose(pk1(unloaded, params.A0, params), 0.0, atol=1.0e-12))


  def test_invalid_orientation_or_negative_jacobian_is_rejected(self):
    params = AnisotropicMaterialParams(2.0, 3.0, 4.0, [1.0, 0.0, 0.0])
    with self.assertRaises(ValueError):
      pk1(np.diag([-1.0, 1.0, 1.0]), params.A0, params)


  def test_center_trial_commit_and_rollback_lifecycle(self):
    state = AnisotropicCenterState(A0=a0_to_A0([0.0, 1.0, 0.0]))
    old = state.committed_F.copy()
    state.begin_trial(np.diag([1.0, 0.0, 0.0]), 0.1)
    assert not np.allclose(state.trial_F, old)
    state.rollback()
    assert np.allclose(state.trial_F, old)
    state.begin_trial(np.diag([1.0, 0.0, 0.0]), 0.1)
    state.commit()
    assert state.valid and np.allclose(state.committed_F, state.trial_F)


  def test_default_center_direction_is_uniaxial(self):
    state = AnisotropicCenterState()
    assert np.allclose(state.A0, np.diag([1.0, 0.0, 0.0]))


if __name__ == "__main__":
    unittest.main()
