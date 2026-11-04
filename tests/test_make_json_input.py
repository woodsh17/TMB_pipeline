"""CPU tests for the TMB_MPNN JSON input preparation script.

make_json_input_for_TMB_MPNN.py is the first thing a user runs: it turns backbone
PDBs into the two .jsonl files that submit_TMB_MPNN.sh feeds to run.py. It needs
no model weights, but a mistake in it silently mislabels which residues face the
membrane interface, so the resulting sequences would be wrong rather than absent.

NOTE: this script is duplicated, byte for byte apart from the output filenames it
is paired with, at
    TMB_MPNN/run_TMB_MPNN_longer_hydrophobic_example/make_json_input_longer_hydrophobic_thickness_for_TMB_MPNN.py
Only the copy under run_TMB_MPNN_example is tested here. Deduplicating the two is
a refactor that belongs in its own change; until then, fixes must be applied twice.

The script has no main() guard: importing it executes the whole pipeline against
sys.argv and writes its output into the current directory. load_script_module()
therefore imports it inside a temporary directory with an argv that names no PDBs.
"""

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = ROOT / "TMB_MPNN/run_TMB_MPNN_example"
SCRIPT = EXAMPLE_DIR / "make_json_input_for_TMB_MPNN.py"
BACKBONE_DIR = EXAMPLE_DIR / "backbones"

TM_DEPTH = "100"
INTERFACE_FRACTION = "0.25"


def load_script_module(directory):
    """Import the script with a harmless argv, from inside `directory`."""
    original_argv, original_cwd = sys.argv, os.getcwd()
    sys.argv = [SCRIPT.name, TM_DEPTH, INTERFACE_FRACTION]
    os.chdir(directory)
    try:
        spec = importlib.util.spec_from_file_location("tmb_make_json_input", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        os.chdir(original_cwd)
        sys.argv = original_argv


class ScriptModuleTestCase(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.directory)
        self.script = load_script_module(self.directory)


class GeometryHelperTests(ScriptModuleTestCase):
    def test_scale_func_passes_through_short_radii_and_suppresses_long_ones(self):
        self.assertEqual(self.script.scale_func(3.0), 3.0)
        self.assertEqual(self.script.scale_func(5.0), 5.0)
        self.assertAlmostEqual(self.script.scale_func(10.0), 1 / 1000.0)

    def test_mod_coords_scales_each_row_by_its_own_radius(self):
        coords = np.array([[3.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        result = self.script.mod_coords(coords)
        np.testing.assert_allclose(result[0], [9.0, 0.0, 0.0])
        np.testing.assert_allclose(result[1], [10.0 / 1000.0, 0.0, 0.0])

    def test_make_rotmat_sends_its_input_axis_onto_z(self):
        vector = np.array([1.0, 2.0, -3.0])
        unit = vector / np.linalg.norm(vector)
        rotation = self.script.make_rotmat(unit)
        np.testing.assert_allclose(rotation @ unit, [0.0, 0.0, 1.0], atol=1e-9)
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-9)
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=9)

    def test_make_rotmat_is_identity_for_an_axis_already_on_z(self):
        np.testing.assert_allclose(
            self.script.make_rotmat(np.array([0.0, 0.0, 1.0])), np.eye(3), atol=1e-12
        )

    def test_invert_pdb_negates_x_and_z_but_preserves_y(self):
        coords = np.array([[1.0, 2.0, 3.0]])
        first, second, third = self.script.invert_pdb(coords, coords * 2, coords * 3)
        np.testing.assert_allclose(first, [[-1.0, 2.0, -3.0]])
        np.testing.assert_allclose(second, [[-2.0, 4.0, -6.0]])
        np.testing.assert_allclose(third, [[-3.0, 6.0, -9.0]])


class CbetaGeometryTests(ScriptModuleTestCase):
    def setUp(self):
        super().setUp()
        # Two residues with realistic N/CA/C geometry, translated apart.
        self.ca = np.array([[0.0, 0.0, 0.0], [3.8, 0.0, 0.0]])
        self.n = self.ca + np.array([-0.525, 1.363, 0.0])
        self.c = self.ca + np.array([1.526, 0.0, 0.0])

    def test_cbeta_sits_one_n_bond_length_from_calpha(self):
        cb = self.script.calculate_cbeta_coordinates(self.n, self.c, self.ca)
        # The CA->N vector is rotated, not rescaled, so the length is preserved.
        np.testing.assert_allclose(
            np.linalg.norm(cb - self.ca, axis=-1),
            np.linalg.norm(self.n - self.ca, axis=-1),
            atol=1e-9,
        )

    def test_cbeta_keeps_the_n_ca_c_angle_because_it_rotates_about_ca_c(self):
        cb = self.script.calculate_cbeta_coordinates(self.n, self.c, self.ca)

        def angle_to_c(vectors):
            axis = (self.c - self.ca) / np.linalg.norm(self.c - self.ca, axis=-1, keepdims=True)
            unit = vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)
            return np.sum(axis * unit, axis=-1)

        np.testing.assert_allclose(angle_to_c(cb - self.ca), angle_to_c(self.n - self.ca), atol=1e-9)

    def test_cbeta_is_not_simply_the_nitrogen_position(self):
        cb = self.script.calculate_cbeta_coordinates(self.n, self.c, self.ca)
        self.assertGreater(float(np.min(np.linalg.norm(cb - self.n, axis=-1))), 1.0)


class PoreAndSurfaceTests(ScriptModuleTestCase):
    @staticmethod
    def barrel(n_residues=12, radius=10.0):
        """A ring of residues around the z-axis with alternating CB directions."""
        angles = np.linspace(0, 2 * np.pi, n_residues, endpoint=False)
        ca = np.stack([radius * np.cos(angles), radius * np.sin(angles),
                       np.zeros(n_residues)], axis=-1)
        # Offsets chosen only to be non-degenerate; direction comes out of the
        # idealized CB construction, which is what we want to exercise.
        n = ca + np.array([-0.525, 1.363, 0.0])
        c = ca + np.array([1.526, 0.0, 0.3])
        return ca, n, c

    def test_pore_and_surface_sets_are_disjoint_and_within_range(self):
        ca, n, c = self.barrel()
        pore, surface = self.script.find_pore_and_surf_residues(ca, n, c)
        self.assertEqual(set(pore) & set(surface), set())
        self.assertTrue(set(pore) | set(surface) <= set(range(len(ca))))

    def test_classification_is_unchanged_by_rotation_about_the_pore_axis(self):
        ca, n, c = self.barrel()
        pore, surface = self.script.find_pore_and_surf_residues(ca, n, c)
        angle = 0.7
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0.0],
                             [np.sin(angle), np.cos(angle), 0.0],
                             [0.0, 0.0, 1.0]])
        rotated = self.script.find_pore_and_surf_residues(ca @ rotation.T, n @ rotation.T,
                                                          c @ rotation.T)
        np.testing.assert_array_equal(rotated[0], pore)
        np.testing.assert_array_equal(rotated[1], surface)

    def test_classification_is_unchanged_by_translation_along_the_pore_axis(self):
        ca, n, c = self.barrel()
        pore, surface = self.script.find_pore_and_surf_residues(ca, n, c)
        shift = np.array([0.0, 0.0, 25.0])
        shifted = self.script.find_pore_and_surf_residues(ca + shift, n + shift, c + shift)
        np.testing.assert_array_equal(shifted[0], pore)
        np.testing.assert_array_equal(shifted[1], surface)


class InterfaceLabelTests(ScriptModuleTestCase):
    @staticmethod
    def stack(z_values):
        return np.array([[0.0, 0.0, z] for z in z_values])

    def test_labels_cover_every_residue_and_use_only_the_three_codes(self):
        coords = self.stack([0.0, 1.0, 2.0, 8.0, 9.0, 10.0])
        labels = self.script.get_intf_buried_resi(coords, 100.0, 0.25, np.arange(len(coords)))
        self.assertEqual(len(labels), len(coords))
        self.assertTrue(set(float(value) for value in labels) <= {0.0, 1.0, 2.0})

    def test_nothing_is_labelled_when_no_residue_faces_the_surface(self):
        coords = self.stack([0.0, 1.0, 2.0, 8.0, 9.0, 10.0])
        labels = self.script.get_intf_buried_resi(coords, 100.0, 0.25, np.array([], dtype=int))
        self.assertEqual([float(value) for value in labels], [0.0] * len(coords))

    def test_residues_at_both_membrane_edges_are_labelled_interface(self):
        coords = self.stack([0.0, 1.0, 5.0, 9.0, 10.0])
        labels = self.script.get_intf_buried_resi(coords, 100.0, 0.25, np.arange(len(coords)))
        # The mid-slab residue is buried, the outermost ones are interface.
        self.assertEqual(float(labels[4]), 1.0)
        self.assertEqual(float(labels[2]), 2.0)

    @unittest.expectedFailure
    def test_first_residue_can_be_labelled(self):
        """Regression record: residue index 0 is never labelled.

        get_intf_buried_resi builds its index lists with comprehensions that drop
        any entry equal to 0.0, which discards the genuine residue at index 0 along
        with the zeros numpy.where() uses as padding. The first residue therefore
        always reaches TMB_MPNN with no surface condition.

        Demonstrated below: index 0 and index 1 hold identical z, both are declared
        surface-facing, and index 1 is labelled interface while index 0 is not.
        Remove this decorator once the padding sentinel is replaced with a mask.
        """
        coords = self.stack([0.0, 0.0, 1.0, 2.0, 9.0, 10.0])
        labels = self.script.get_intf_buried_resi(coords, 100.0, 0.25, np.arange(len(coords)))
        self.assertEqual(float(labels[0]), float(labels[1]))


class EndToEndTests(unittest.TestCase):
    """Run the script the way the README tells users to and check its output."""

    @classmethod
    def setUpClass(cls):
        cls.backbones = sorted(BACKBONE_DIR.glob("*.pdb"))
        if not cls.backbones:
            raise unittest.SkipTest(f"no example backbones in {BACKBONE_DIR}")

        cls.directory = Path(tempfile.mkdtemp())
        cls.addClassCleanup(shutil.rmtree, cls.directory)
        result = subprocess.run(
            [sys.executable, str(SCRIPT), TM_DEPTH, INTERFACE_FRACTION]
            + [str(path) for path in cls.backbones],
            cwd=cls.directory, capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise AssertionError(result.stdout + result.stderr)

    def load(self, name):
        return json.loads((self.directory / name).read_text())

    def test_writes_both_json_files_keyed_by_resolved_pdb_path(self):
        expected = {str(path.resolve()) for path in self.backbones}
        self.assertEqual(set(self.load("pdbs_multi_json.jsonl")), expected)
        self.assertEqual(set(self.load("intf_buried.jsonl")), expected)

    def test_pdb_entries_are_empty_lists_as_run_py_expects(self):
        self.assertEqual(list(self.load("pdbs_multi_json.jsonl").values()),
                         [[] for _ in self.backbones])

    def test_one_label_per_residue_using_only_the_three_codes(self):
        for path, labels in self.load("intf_buried.jsonl").items():
            with self.subTest(pdb=Path(path).name):
                ca_count = sum(1 for line in Path(path).read_text().splitlines()
                               if line.startswith("ATOM") and line[12:16].strip() == "CA")
                self.assertEqual(len(labels), ca_count)
                self.assertTrue(set(labels) <= {0.0, 1.0, 2.0})

    def test_both_interface_and_buried_residues_are_identified(self):
        for path, labels in self.load("intf_buried.jsonl").items():
            with self.subTest(pdb=Path(path).name):
                self.assertIn(1.0, labels, "no interface residues labelled")
                self.assertIn(2.0, labels, "no buried residues labelled")


if __name__ == "__main__":
    unittest.main()
