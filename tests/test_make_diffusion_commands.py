"""CPU tests for the TMB_RFD2 inference-command generator.

make_diffusion_commands.sh is pure command assembly: it derives glycine-twist
residue indices, writes the conditioning .pt files, and emits inference_commands.txt.
None of that needs RFD2 weights or a GPU, but a typo in it silently produces a
wrong run, so it is worth pinning down.

The script insists on a specific directory layout and refuses to run without the
containers and checkpoint, so each test builds a miniature pipeline tree in a
temporary directory with empty placeholder files.
"""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_NAME = "make_diffusion_commands.sh"

# Defaults baked into the script's user-parameter block.
N_STRANDS = 14
RESIDUES_PER_STRAND = 16
TWIST_STRANDS = [2, 7]


def expected_twist_residues(strands, residues_per_strand):
    """Mirror of the index arithmetic in the script: five residues per strand."""
    residues = []
    for strand in strands:
        start = (strand - 1) * residues_per_strand + residues_per_strand // 2 - 2
        residues.extend(range(start, start + 5))
    return residues


class DiffusionCommandTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.directory)

        self.pipeline = self.directory / "TMB_pipeline"
        self.repo_root = self.pipeline / "TMB_RFD2"
        (self.repo_root / "TMB_RFD2_code" / "rf_diffusion").mkdir(parents=True)

        for name in [SCRIPT_NAME, "adj_ss_wrapper_glytwist.py", "dummy.pdb"]:
            shutil.copy(ROOT / "TMB_RFD2" / name, self.repo_root / name)

        self.weights = self.pipeline / "weights"
        self.environments = self.pipeline / "environments"
        self.weights.mkdir()
        self.environments.mkdir()
        self.checkpoint = self.weights / "TMB_RFD2.pt"
        self.checkpoint.write_bytes(b"placeholder checkpoint")
        self.modulus_sif = self.environments / "modulus_latest.sif"
        self.diffusion_sif = self.environments / "TMB_RFD2.sif"
        for sif in [self.modulus_sif, self.diffusion_sif]:
            sif.write_bytes(b"placeholder container")

        # The script invokes a bare `python`, which need not exist on PATH (this
        # repo's own container only provides `python3.12`). Shim it to the
        # interpreter running the tests, which is the one that has torch.
        self.shim = self.directory / "shim"
        self.shim.mkdir()
        # A symlink would break virtualenv detection, so exec the real path.
        shim_path = self.shim / "python"
        shim_path.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
        shim_path.chmod(0o755)

        self.output_file = self.repo_root / "inference_commands.txt"

    def run_script(self, expect_success=True):
        environment = dict(os.environ)
        environment["WEIGHTS_DIR"] = str(self.weights)
        environment["ENV_DIR"] = str(self.environments)
        environment["PATH"] = f"{self.shim}{os.pathsep}{environment['PATH']}"
        result = subprocess.run(
            ["bash", SCRIPT_NAME],
            cwd=self.repo_root,
            env=environment,
            capture_output=True,
            text=True,
        )
        if expect_success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def read_command(self):
        lines = [line for line in self.output_file.read_text().splitlines() if line.strip()]
        self.assertEqual(len(lines), 1, f"expected one command, got {len(lines)}")
        return lines[0]

    def test_generates_one_command_carrying_the_documented_inference_flags(self):
        self.run_script()
        command = self.read_command()
        for fragment in [
            '--config-name="aa_ppi"',
            'contigmap.has_termini="[True]"',
            "contigmap.contigs=\"['1-1']\"",
            'inference.num_designs="10"',
            '++transforms.configs.CenterPostTransform.center_type="all"',
            f'inference.ckpt_path="{self.checkpoint}"',
            f'inference.input_pdb="{self.repo_root / "dummy.pdb"}"',
            str(self.diffusion_sif),
            "rf_diffusion/run_inference.py",
        ]:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, command)

    def test_twist_residue_indices_follow_the_five_residue_window_rule(self):
        self.run_script()
        command = self.read_command()
        expected = ",".join(str(index) for index in
                            expected_twist_residues(TWIST_STRANDS, RESIDUES_PER_STRAND))
        self.assertIn(f"++glycine_twists.glycine_twists_res='\"{expected}\"'", command)

    def test_conditioning_tensors_are_written_where_the_command_points(self):
        self.run_script()
        command = self.read_command()
        scaffold_dir = command.split('scaffoldguided.scaffold_dir="', 1)[1].split('"', 1)[0]
        prefix = f"n{N_STRANDS}_nres{RESIDUES_PER_STRAND}"
        for suffix in ["adj", "ss", "twist"]:
            with self.subTest(tensor=suffix):
                self.assertTrue(Path(scaffold_dir, f"{prefix}_{suffix}.pt").is_file())

    def test_command_twist_indices_match_the_generated_twist_tensor(self):
        """The shell arithmetic and the Python wrapper must agree.

        The script computes twist residue indices itself for the inference flag,
        and separately asks adj_ss_wrapper_glytwist.py to write a twist tensor.
        Nothing forces those two to stay in step, so compare them directly.
        """
        self.run_script()
        command = self.read_command()
        scaffold_dir = command.split('scaffoldguided.scaffold_dir="', 1)[1].split('"', 1)[0]
        twist = torch.load(
            Path(scaffold_dir, f"n{N_STRANDS}_nres{RESIDUES_PER_STRAND}_twist.pt"),
            weights_only=True,
        )
        from_tensor = sorted(int(index) for index in twist.nonzero().flatten())
        flag = command.split("glycine_twists_res='\"", 1)[1].split('"', 1)[0]
        from_command = sorted(int(index) for index in flag.split(","))
        self.assertEqual(from_command, from_tensor)
        self.assertEqual(len(from_tensor), 5 * len(TWIST_STRANDS))

    def test_rerunning_reuses_existing_tensors_and_rewrites_one_command(self):
        self.run_script()
        first = self.output_file.read_text()
        result = self.run_script()
        self.assertIn("Files already exist", result.stdout)
        self.assertEqual(self.output_file.read_text(), first)
        self.read_command()

    def test_missing_container_fails_before_writing_any_command(self):
        self.diffusion_sif.unlink()
        result = self.run_script(expect_success=False)
        self.assertEqual(result.returncode, 1)
        self.assertIn("diffusion SIF not found", result.stdout)
        self.assertFalse(self.output_file.exists())

    def test_missing_checkpoint_reports_the_weights_directory(self):
        self.checkpoint.unlink()
        result = self.run_script(expect_success=False)
        self.assertEqual(result.returncode, 1)
        self.assertIn("checkpoint not found", result.stdout)
        self.assertIn(str(self.weights), result.stdout)

    def test_missing_rf_diffusion_tree_is_rejected(self):
        shutil.rmtree(self.repo_root / "TMB_RFD2_code" / "rf_diffusion")
        result = self.run_script(expect_success=False)
        self.assertEqual(result.returncode, 1)
        self.assertIn("rf_diffusion directory not found", result.stdout)


if __name__ == "__main__":
    unittest.main()
