# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI parsing tests for taper_router.

Regression guard: --shadow-mode was once declared with arg_type=bool, so
`--shadow-mode false` parsed as True (bool("false") is True) and a live
benchmark ran with the gate silently in shadow mode.
"""

from __future__ import annotations

import argparse

import pytest

from dynamo.taper_router.args import TaperArgGroup

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    TaperArgGroup().add_arguments(parser)
    return parser.parse_args(argv)


def test_shadow_mode_defaults_on(monkeypatch):
    monkeypatch.delenv("DYN_TAPER_SHADOW_MODE", raising=False)
    assert _parse([]).shadow_mode is True


def test_no_shadow_mode_flag_turns_it_off(monkeypatch):
    monkeypatch.delenv("DYN_TAPER_SHADOW_MODE", raising=False)
    assert _parse(["--no-shadow-mode"]).shadow_mode is False


@pytest.mark.parametrize("value,expected", [("false", False), ("true", True)])
def test_shadow_mode_env_var_is_parsed_as_bool(monkeypatch, value, expected):
    monkeypatch.setenv("DYN_TAPER_SHADOW_MODE", value)
    assert _parse([]).shadow_mode is expected


def test_release_smoothing_defaults_and_overrides(monkeypatch):
    monkeypatch.delenv("DYN_TAPER_ADMIT_SETTLE_SECONDS", raising=False)
    monkeypatch.delenv("DYN_TAPER_MAX_RELEASE_PER_TICK", raising=False)
    ns = _parse([])
    assert (ns.admit_settle_seconds, ns.max_release_per_tick) == (0.5, 4)
    ns = _parse(["--admit-settle-seconds", "0.25", "--max-release-per-tick", "8"])
    assert (ns.admit_settle_seconds, ns.max_release_per_tick) == (0.25, 8)
