"""CPU tests for a3m parsing and MSA subsampling; no model weights required."""

import gzip
from pathlib import Path
import random
import shutil
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "TMB_MPNN/TMB_MPNN_code/v2.1.1"))

from mpnn.datasets.fused_data_utils import parse_a3m, subsample_msa

ALPHABET = "ARNDCQEGHILKMFPSTWYV-"
CODE = {letter: index for index, letter in enumerate(ALPHABET)}
GAP = CODE["-"]


def encode(sequence):
    """Encode an uppercase sequence the way parse_a3m is expected to."""
    return np.array([CODE[letter] for letter in sequence], dtype=np.uint8)


class ParseA3MTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.directory)

    def write_a3m(self, text, name="alignment.a3m"):
        path = self.directory / name
        path.write_text(text)
        return str(path)

    def test_match_columns_encode_to_alphabet_indices_and_gaps(self):
        path = self.write_a3m(">query\nARND\n>hit1\nARN-\n")
        msa, ins = parse_a3m(path)
        self.assertEqual(msa.shape, (2, 4))
        np.testing.assert_array_equal(msa[0], encode("ARND"))
        np.testing.assert_array_equal(msa[1], np.array([0, 1, 2, GAP], dtype=np.uint8))
        np.testing.assert_array_equal(ins, np.zeros((2, 4), dtype=np.uint8))

    def test_lowercase_columns_are_stripped_and_counted_at_one_site(self):
        # Two lowercase residues inserted after match column 1 (0-based).
        path = self.write_a3m(">query\nARND\n>hit1\nARndND\n")
        msa, ins = parse_a3m(path)
        self.assertEqual(msa.shape, (2, 4))
        np.testing.assert_array_equal(msa[1], encode("ARND"))
        np.testing.assert_array_equal(ins[1], np.array([0, 0, 2, 0], dtype=np.uint8))

    def test_insertions_at_two_separate_sites_are_counted_separately(self):
        path = self.write_a3m(">query\nARND\n>hit1\nAxRNyD\n")
        _, ins = parse_a3m(path)
        np.testing.assert_array_equal(ins[1], np.array([0, 1, 0, 1], dtype=np.uint8))

    def test_maxseq_stops_reading_after_that_many_sequences(self):
        path = self.write_a3m(">q\nARND\n>h1\nARND\n>h2\nARND\n>h3\nARND\n")
        msa, ins = parse_a3m(path, maxseq=2)
        self.assertEqual(msa.shape, (2, 4))
        self.assertEqual(ins.shape, (2, 4))

    def test_unknown_residue_letters_are_treated_as_gaps(self):
        path = self.write_a3m(">query\nARXD\n")
        msa, _ = parse_a3m(path)
        # X is not in the 21-letter alphabet and must collapse onto the gap code.
        self.assertEqual(int(msa[0, 2]), GAP)

    def test_blank_lines_and_label_lines_are_skipped(self):
        path = self.write_a3m(">query\nARND\n\n>hit1\n\nARND\n")
        msa, _ = parse_a3m(path)
        self.assertEqual(msa.shape, (2, 4))

    def test_gzipped_alignment_matches_the_plain_one(self):
        text = ">query\nARND\n>hit1\nARndND\n"
        plain_msa, plain_ins = parse_a3m(self.write_a3m(text))
        gz_path = self.directory / "alignment.a3m.gz"
        with gzip.open(gz_path, "wt") as handle:
            handle.write(text)
        gz_msa, gz_ins = parse_a3m(str(gz_path))
        np.testing.assert_array_equal(gz_msa, plain_msa)
        np.testing.assert_array_equal(gz_ins, plain_ins)


class SubsampleMsaTests(unittest.TestCase):
    @staticmethod
    def make_msa(rows, columns=6):
        """Each row is filled with its own index so rows stay identifiable."""
        msa = np.tile(np.arange(rows, dtype=np.uint8)[:, None], (1, columns))
        return msa, np.zeros((rows, columns), dtype=np.uint8)

    def test_shallow_alignments_are_returned_unchanged(self):
        msa, ins = self.make_msa(2)
        out_msa, out_ins = subsample_msa(msa, ins, maxseq=100)
        np.testing.assert_array_equal(out_msa, msa)
        np.testing.assert_array_equal(out_ins, ins)

    def test_query_is_kept_first_and_every_row_comes_from_the_input(self):
        msa, ins = self.make_msa(40)
        np.random.seed(0)
        random.seed(0)
        out_msa, out_ins = subsample_msa(msa, ins, maxseq=100)
        np.testing.assert_array_equal(out_msa[0], msa[0])
        self.assertEqual(out_msa.shape[1], msa.shape[1])
        self.assertEqual(out_msa.shape, out_ins.shape)
        kept = {int(row[0]) for row in out_msa}
        self.assertTrue(kept.issubset(set(range(40))))
        # The query is row 0 and must not be duplicated by the sample.
        self.assertEqual(len(kept), out_msa.shape[0])

    def test_maxseq_caps_how_many_rows_are_considered(self):
        msa, ins = self.make_msa(40)
        np.random.seed(0)
        random.seed(0)
        out_msa, _ = subsample_msa(msa, ins, maxseq=12)
        self.assertLessEqual(out_msa.shape[0], 12)
        # Rows beyond maxseq must never be selected.
        self.assertTrue(all(int(row[0]) < 12 for row in out_msa))

    @unittest.expectedFailure
    def test_small_alignments_do_not_crash(self):
        """Regression record: subsample_msa raises for 3-8 sequence alignments.

        n is drawn from np.random.uniform(9, nr-1), which for a small nr yields a
        value far larger than the population random.sample() then draws from, so
        the call dies with "Sample larger than population". Measured failure rate
        with 300 seeds: nr=3 87%, nr=5 81%, nr=8 51%, nr>=9 none.

        Remove this decorator once the sampling bound is fixed.
        """
        msa, ins = self.make_msa(5)
        np.random.seed(0)
        random.seed(0)
        subsample_msa(msa, ins, maxseq=100, sub_type="UNI")

    @unittest.expectedFailure
    def test_unknown_subsampling_mode_exits_with_a_message(self):
        """Regression record: the error path itself is broken.

        sys.exit() is called with two arguments, so an unrecognised sub_type
        raises TypeError instead of exiting with the intended message.
        Remove this decorator once the call passes a single string.
        """
        msa, ins = self.make_msa(20)
        with self.assertRaises(SystemExit):
            subsample_msa(msa, ins, maxseq=100, sub_type="NOT_A_MODE")


if __name__ == "__main__":
    unittest.main()
