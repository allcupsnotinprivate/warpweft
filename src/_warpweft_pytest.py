"""Pytest plugin for warpweft: ready-made fixtures over the testing helpers.

A standalone top-level module, loaded through the ``pytest11`` entry point.
It lives *outside* the ``warpweft`` package on purpose: pytest loads plugin
entry points at startup, before a coverage plugin begins measuring, so a
plugin under ``warpweft`` would drag the whole package's import-time code in
too early and skew its coverage. Here, importing the plugin only imports
``pytest``; the warpweft imports are deferred into the fixture bodies, which
run once a session (and any coverage) is already under way.

Deliberately minimal - a fresh clock per test. ``drive`` / ``drive_policy``
stay plain functions: wrapping stateless helpers in a fixture would add
nothing.
"""

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from warpweft.core.clock import ManualClock
    from warpweft.testing import InstantClock


@pytest.fixture
def instant_clock() -> "InstantClock":
    """A fresh ``InstantClock``: sleeps advance virtual time, never block."""
    from warpweft.testing import InstantClock

    return InstantClock()


@pytest.fixture
def manual_clock() -> "ManualClock":
    """A fresh ``ManualClock``: the test advances time explicitly."""
    from warpweft.testing import ManualClock

    return ManualClock()
