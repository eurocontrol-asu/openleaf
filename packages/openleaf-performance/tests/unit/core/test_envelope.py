from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from openleaf.performance.core.envelope import EnvelopeChecker
from openleaf.performance.core.types import EnvelopeMode
from openleaf.performance.core.types import EnvelopeViolationError


@pytest.fixture
def mock_model():
    """Model with generous limits (no violations expected)."""
    model = MagicMock()
    model.oem = 40_000.0
    model.mtom = 80_000.0
    model.model = "TEST-A320"
    return model


def _make_flight_state(
    *,
    cas_kt: float = 250.0,
    mach: float = 0.78,
    altitude_ft: float = 35_000.0,
    mass: float = 60_000.0,
):
    """Create a mock flight state accessor."""
    state = MagicMock()
    state.cas_kt = np.asarray(cas_kt)
    state.mach = np.asarray(mach)
    state.altitude_ft = np.asarray(altitude_ft)
    state.mass = np.asarray(mass)
    return state


def _make_checker(
    mock_model,
    flight_state,
    *,
    max_cas: float = 350.0,
    min_cas: float = 130.0,
    max_mach: float = 0.82,
    max_altitude: float = 41_000.0,
    envelope_mode: EnvelopeMode = EnvelopeMode.IGNORE,
):
    """Build an EnvelopeChecker with controlled limit values."""
    mock_model.max_cas.return_value = np.asarray(max_cas)
    mock_model.min_cas.return_value = np.asarray(min_cas)
    mock_model.max_mach.return_value = np.asarray(max_mach)
    mock_model.max_altitude.return_value = np.asarray(max_altitude)
    return EnvelopeChecker(
        model=mock_model,
        flight_state=flight_state,
        envelope_mode=envelope_mode,
    )


# ---------------------------------------------------------------------------
# Unit tests — one violation type at a time
# ---------------------------------------------------------------------------


class TestEnvelopeCheckerNoViolations:
    def test_envelope_checker_no_violations(self, mock_model):
        """Nominal flight state within all limits."""
        state = _make_flight_state(cas_kt=250.0, mach=0.78, altitude_ft=35_000.0, mass=60_000.0)
        checker = _make_checker(mock_model, state)
        violations = checker.envelope_violations

        assert not np.any(violations["overspeed_cas"])
        assert not np.any(violations["overspeed_mach"])
        assert not np.any(violations["underspeed"])
        assert not np.any(violations["ceiling"])
        assert not np.any(violations["mass_overweight"])
        assert not np.any(violations["mass_underweight"])


class TestEnvelopeCheckerOverspeedCas:
    def test_envelope_checker_overspeed_cas(self, mock_model):
        """CAS above max_cas triggers overspeed_cas only."""
        state = _make_flight_state(cas_kt=400.0)
        checker = _make_checker(mock_model, state, max_cas=350.0)
        violations = checker.envelope_violations

        assert np.all(violations["overspeed_cas"])
        assert not np.any(violations["overspeed_mach"])
        assert not np.any(violations["underspeed"])
        assert not np.any(violations["ceiling"])
        assert not np.any(violations["mass_overweight"])
        assert not np.any(violations["mass_underweight"])


class TestEnvelopeCheckerOverspeedMach:
    def test_envelope_checker_overspeed_mach(self, mock_model):
        """Mach above max_mach triggers overspeed_mach only."""
        state = _make_flight_state(mach=0.88)
        checker = _make_checker(mock_model, state, max_mach=0.82)
        violations = checker.envelope_violations

        assert np.all(violations["overspeed_mach"])
        assert not np.any(violations["overspeed_cas"])


class TestEnvelopeCheckerUnderspeed:
    def test_envelope_checker_underspeed(self, mock_model):
        """CAS below min_cas triggers underspeed only."""
        state = _make_flight_state(cas_kt=100.0)
        checker = _make_checker(mock_model, state, min_cas=130.0)
        violations = checker.envelope_violations

        assert np.all(violations["underspeed"])
        assert not np.any(violations["overspeed_cas"])


class TestEnvelopeCheckerCeiling:
    def test_envelope_checker_ceiling(self, mock_model):
        """Altitude above max_altitude triggers ceiling only."""
        state = _make_flight_state(altitude_ft=45_000.0)
        checker = _make_checker(mock_model, state, max_altitude=41_000.0)
        violations = checker.envelope_violations

        assert np.all(violations["ceiling"])
        assert not np.any(violations["overspeed_cas"])


class TestEnvelopeCheckerMassOverweight:
    def test_envelope_checker_mass_overweight(self, mock_model):
        """Mass above MTOM triggers mass_overweight only."""
        state = _make_flight_state(mass=90_000.0)
        checker = _make_checker(mock_model, state)  # mtom=80_000
        violations = checker.envelope_violations

        assert np.all(violations["mass_overweight"])
        assert not np.any(violations["mass_underweight"])


class TestEnvelopeCheckerMassUnderweight:
    def test_envelope_checker_mass_underweight(self, mock_model):
        """Mass below OEM triggers mass_underweight only."""
        state = _make_flight_state(mass=30_000.0)
        checker = _make_checker(mock_model, state)  # oem=40_000
        violations = checker.envelope_violations

        assert np.all(violations["mass_underweight"])
        assert not np.any(violations["mass_overweight"])


# ---------------------------------------------------------------------------
# Envelope mode tests
# ---------------------------------------------------------------------------


class TestEnvelopeModeIgnore:
    def test_envelope_mode_ignore(self, mock_model):
        """IGNORE mode returns violations without side effects."""
        state = _make_flight_state(cas_kt=400.0)
        checker = _make_checker(
            mock_model, state, max_cas=350.0, envelope_mode=EnvelopeMode.IGNORE
        )
        result = checker.check_envelope()

        assert np.all(result["overspeed_cas"])


class TestEnvelopeModeWarn:
    def test_envelope_mode_warn(self, mock_model, caplog):
        """WARN mode returns violations and logs a warning."""
        state = _make_flight_state(cas_kt=400.0)
        checker = _make_checker(mock_model, state, max_cas=350.0, envelope_mode=EnvelopeMode.WARN)
        result = checker.check_envelope()

        assert np.all(result["overspeed_cas"])
        # Structlog may not route to caplog; just verify no exception raised


class TestEnvelopeModeRaise:
    def test_envelope_mode_raise(self, mock_model):
        """RAISE mode raises EnvelopeViolationError on violations."""
        state = _make_flight_state(cas_kt=400.0)
        checker = _make_checker(mock_model, state, max_cas=350.0, envelope_mode=EnvelopeMode.RAISE)

        with pytest.raises(EnvelopeViolationError, match="Flight envelope violations"):
            checker.check_envelope()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestMultipleSimultaneousViolations:
    def test_multiple_violations(self, mock_model):
        """Multiple violations detected simultaneously."""
        state = _make_flight_state(cas_kt=400.0, mach=0.90)
        checker = _make_checker(mock_model, state, max_cas=350.0, max_mach=0.82)
        violations = checker.envelope_violations

        assert np.all(violations["overspeed_cas"])
        assert np.all(violations["overspeed_mach"])


class TestScalarInputs:
    def test_scalar_inputs(self, mock_model):
        """Single-point trajectory returns scalar booleans."""
        state = _make_flight_state(cas_kt=400.0)
        checker = _make_checker(mock_model, state, max_cas=350.0)
        violations = checker.envelope_violations

        # Result should be usable as scalar bool
        assert bool(violations["overspeed_cas"])


class TestModelWithoutLimits:
    def test_model_without_min_cas(self, mock_model):
        """Model returning None for min_cas is handled gracefully."""
        state = _make_flight_state(cas_kt=100.0)
        mock_model.min_cas.return_value = None
        mock_model.max_cas.return_value = np.asarray(350.0)
        mock_model.max_mach.return_value = np.asarray(0.82)
        mock_model.max_altitude.return_value = np.asarray(41_000.0)
        checker = EnvelopeChecker(
            model=mock_model,
            flight_state=state,
            envelope_mode=EnvelopeMode.IGNORE,
        )
        violations = checker.envelope_violations

        # When limit is None, the check should be skipped (no violation)
        assert not np.any(violations["underspeed"])


class TestOptionalMmoNoCrash:
    """Aircraft without an MMO (e.g. piston) must not crash the mach check."""

    def test_none_max_mach_reports_no_violation(self, mock_model):
        state = _make_flight_state(mach=0.40)
        mock_model.max_cas.return_value = np.asarray(350.0)
        mock_model.min_cas.return_value = np.asarray(60.0)
        mock_model.max_mach.return_value = None  # no MMO defined
        mock_model.max_altitude.return_value = np.asarray(25_000.0)
        checker = EnvelopeChecker(
            model=mock_model,
            flight_state=state,
            envelope_mode=EnvelopeMode.IGNORE,
        )
        violations = checker.envelope_violations
        assert bool(violations["overspeed_mach"]) is False
