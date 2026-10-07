"""Visual transforms must not change physics or the fixed reference direction."""

import unittest

import numpy as np

from demos.aniso import Config, Scene, directional_curves, display_geometry


class AnisoDemoTests(unittest.TestCase):
    def test_magnification_does_not_mutate_simulation_or_fibers(self):
        reference = np.array([[.3, .3, .3], [.5, .5, .5]])
        points = reference + .001
        F = np.repeat(np.array([[[1., .1, 0.], [.2, 1., 0.], [0., 0., 1.]]]), 2, axis=0)
        original = points.copy(), F.copy()
        shown, segments, _ = display_geometry(reference, points, F, np.array([1., 0., 0.]), 20.)
        np.testing.assert_allclose(shown, reference + .02)
        line = segments[:, 1] - segments[:, 0]
        np.testing.assert_allclose(line / np.linalg.norm(line, axis=1, keepdims=True),
                                   np.tile(np.array([1., .2, 0.]) / np.sqrt(1.04), (2, 1)), atol=2e-6)
        np.testing.assert_array_equal(points, original[0])
        np.testing.assert_array_equal(F, original[1])

    def test_material_unload_returns_to_reference(self):
        scene = Scene(Config(scene="material", fiber_angle=45.), "cpu")
        points, _ = scene.frame(strain=.1, shear=.2)
        self.assertGreater(np.linalg.norm(points - scene.reference), 0.)
        self.assertGreater(scene.metrics()["energy_density"], 0.)
        points, F = scene.frame(strain=0., shear=0.)
        np.testing.assert_allclose(points, scene.reference)
        np.testing.assert_allclose(F, np.tile(np.eye(3), (len(points), 1, 1)))
        self.assertEqual(scene.metrics()["energy_density"], 0.)
        self.assertEqual(scene.metrics()["P11"], 0.)

    def test_directional_plot_has_isotropic_limit_and_order(self):
        strain, curves = directional_curves(0.)
        np.testing.assert_allclose(curves[0], curves[1])
        np.testing.assert_allclose(curves[1], curves[2])
        _, curves = directional_curves(200.)
        self.assertTrue(np.all(curves[0, 1:] > curves[1, 1:]))
        self.assertTrue(np.all(curves[1, 1:] > curves[2, 1:]))
        np.testing.assert_array_equal(curves[:, 0], 0.)
        self.assertEqual(strain[-1], .2)


if __name__ == "__main__":
    unittest.main()
