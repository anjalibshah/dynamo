# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for prompt synthesis — the shared-prefix property is the point."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prompt_synth import shared_prefix_len, synth_prompt  # noqa: E402


class TestSharedPrefix(unittest.TestCase):
    def test_shared_ids_give_shared_text_prefix(self):
        # Two branches sharing an 8-block prefix, each with a distinct extra block.
        prefix = list(range(8))
        a = synth_prompt(prefix + [100], words_per_block=50)
        b = synth_prompt(prefix + [200], words_per_block=50)
        # The shared region must be a real, long common prefix.
        common = shared_prefix_len(a, b)
        prefix_only = synth_prompt(prefix, words_per_block=50)
        self.assertGreaterEqual(common, len(prefix_only))

    def test_divergence_after_shared_prefix(self):
        a = synth_prompt([1, 2, 3, 100], words_per_block=20)
        b = synth_prompt([1, 2, 3, 200], words_per_block=20)
        self.assertNotEqual(a, b)  # different tail

    def test_deterministic_across_calls(self):
        self.assertEqual(synth_prompt([7, 8, 9]), synth_prompt([7, 8, 9]))

    def test_no_shared_prefix_when_first_id_differs(self):
        a = synth_prompt([1, 2, 3], words_per_block=10)
        b = synth_prompt([9, 2, 3], words_per_block=10)
        # First block differs => common prefix shorter than one block.
        one_block = len(synth_prompt([1], words_per_block=10))
        self.assertLess(shared_prefix_len(a, b), one_block)

    def test_length_scales_with_blocks(self):
        short = synth_prompt([1], words_per_block=100)
        long = synth_prompt([1, 2, 3], words_per_block=100)
        self.assertGreater(len(long), len(short))


if __name__ == "__main__":
    unittest.main()
