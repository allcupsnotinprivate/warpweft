"""The pytest11 plugin: instant_clock / manual_clock fixtures.

The plugin auto-loads through warpweft's ``pytest11`` entry point, so these
fixtures resolve in warpweft's own suite as well as in a downstream project.
"""

import sys

import pytest

from warpweft.core.clock import ManualClock
from warpweft.testing import InstantClock, drive_policy, fails_then_succeeds

pytestmark = pytest.mark.unit


def test_instant_clock_fixture_is_fresh(instant_clock: InstantClock) -> None:
    assert isinstance(instant_clock, InstantClock)
    assert instant_clock.slept == []  # a fresh clock records no sleeps


def test_manual_clock_fixture_is_fresh(manual_clock: ManualClock) -> None:
    assert isinstance(manual_clock, ManualClock)


@pytest.mark.anyio
async def test_instant_clock_drives_a_policy_without_real_time(instant_clock: InstantClock) -> None:
    outcome = await drive_policy(
        {"retry": {"attempts": 5, "base_delay": 1.0, "max_delay": 8.0, "jitter": False}},
        fails_then_succeeds(2, value="ok"),
        clock=instant_clock,
    )
    assert outcome.value == "ok"
    assert outcome.attempts == 3  # two failures, then success
    assert len(instant_clock.slept) == 2  # one virtual backoff before each retry


def test_fixtures_resolve_in_an_isolated_pytest_run(pytester: pytest.Pytester) -> None:
    """A downstream project gets the fixtures purely from the entry point."""
    pytester.makepyfile(
        """
        from warpweft.testing import InstantClock
        from warpweft.core.clock import ManualClock

        def test_instant(instant_clock):
            assert isinstance(instant_clock, InstantClock)

        def test_manual(manual_clock):
            assert isinstance(manual_clock, ManualClock)
        """
    )
    result = pytester.runpytest()
    result.assert_outcomes(passed=2)


def test_importing_warpweft_does_not_import_pytest(pytester: pytest.Pytester) -> None:
    """Plain use of warpweft must not drag pytest into the process."""
    result = pytester.run(
        sys.executable,
        "-c",
        "import sys, warpweft, warpweft.testing; sys.exit(1 if 'pytest' in sys.modules else 0)",
    )
    assert result.ret == 0
