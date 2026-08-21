# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic prompt synthesis from a row's ``hash_ids``.

On a *real* engine, branches co-locate only if they share an identical **text**
prefix long enough to fill KV blocks — the engine's prefix cache keys on tokens,
not on the trace's ``hash_ids`` (those are for offline tooling). So the replay
client must turn ``hash_ids`` into text where two rows sharing a leading run of
ids produce a byte-identical leading string.

We map each hash id to a fixed block of pseudo-word tokens derived only from the
integer id. Concatenated in order, a shared id-prefix yields a shared text
prefix. The mapping is process-independent (seeded by the id alone), so every
arm and every worker reconstructs the same text for the same id.

Token-count note: one "word" ≈ one token is an approximation; exact ISL control
would require sending token ids. For the P0 question — does a shared prefix
co-batch, and does the gate move the tail — approximate lengths are fine, and
``datagen analyze`` on the trace already reports the intended block structure.
The words-per-block ratio is configurable so it can be tuned per tokenizer once
measured on the box.
"""

from __future__ import annotations

import hashlib

# Deterministic vocabulary. Fixed list so a given id always yields the same
# words regardless of Python hash randomization.
_VOCAB = [
    f"tok{n:04d}" for n in range(4096)
]


def _block_words(hash_id: int, words_per_block: int) -> list[str]:
    """Deterministic list of words for one KV block, keyed only by ``hash_id``."""
    out = []
    # Seed a simple counter-mode expansion from the id; stable across processes.
    for w in range(words_per_block):
        h = hashlib.blake2b(f"{hash_id}:{w}".encode(), digest_size=4).digest()
        idx = int.from_bytes(h, "big") % len(_VOCAB)
        out.append(_VOCAB[idx])
    return out


def synth_prompt(hash_ids: list[int], words_per_block: int = 400) -> str:
    """Build a prompt whose text prefix is shared iff the id prefix is shared.

    ``words_per_block`` approximates ``block_size`` tokens; tune once the
    tokenizer's words/token ratio is measured on the target model.
    """
    words: list[str] = []
    for hid in hash_ids:
        words.extend(_block_words(hid, words_per_block))
    return " ".join(words)


def shared_prefix_len(a: str, b: str) -> int:
    """Length of the common leading substring — used in tests/assertions."""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i
