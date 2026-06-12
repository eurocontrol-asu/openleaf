"""Parser regression tests against a fully SYNTHETIC OPF file.

All numbers below are fabricated for testing the parser's field ordering and
unit handling. They are NOT BADA coefficients for any real aircraft, so this
file is safe to publish and runs without any proprietary BADA data.

The fixture's values are deliberately all distinct so that any column-order
regression (e.g. swapped Desc(low)/Desc(high), or buffet coefficients read
from the wrong position on the shared aerodynamics line) fails loudly.
"""

from __future__ import annotations

import pytest

from openleaf.performance.core.types import AerodynamicConfig
from openleaf.performance.models.bada3.limits import BADA3LimitsModel
from openleaf.performance.models.bada3.parsing import parse_bada3_file


SYNTHETIC_OPF = """\
CD XXSYN_ 2 engines Jet M
CC Aerodynamics: Wing Area and Buffet coefficients (SIM)
CD   3    .90000E+02   .15000E+01   .50000E+00   .00000E+00
CC Configuration characteristics
CD 1 CR Clean      .14500E+03   .25000E-01   .40000E-01
CD 2 IC Initial    .12000E+03   .27000E-01   .42000E-01
CD 3 TO Takeoff    .11000E+03   .30000E-01   .45000E-01
CD 4 AP Approach   .10000E+03   .50000E-01   .47000E-01
CD 5 LD Landing    .95000E+02   .80000E-01   .50000E-01
CC Gear
CD DOWN   .20000E-01
CC Max climb thrust coefficients (SIM)
CD   .14000E+06   .50000E+05   .60000E-10   .90000E+01   .10000E-01
CC Desc(low)      Desc(high)   Desc level   Desc(app)    Desc(ld) (SIM)
CD   .10000E+00   .50000E-01   .20000E+05   .15000E+00   .30000E+00
CC Thrust Specific Fuel Consumption Coefficients (SIM)
CD   .70000E+00   .20000E+04
CC Descent Fuel Flow Coefficients (SIM)
CD   .10000E+02   .60000E+05
CC Cruise Corr. (SIM)
CD   .95000E+00
CC Mass (t)
CD   .62000E+02   .40000E+02   .75000E+02   .20000E+02   .50000E+00
CC Flight envelope
CD   .34000E+03   .82000E+00   .39000E+05   .35000E+05  -.80000E+02   .45000E+00
"""


@pytest.fixture(scope="module")
def parsed(tmp_path_factory):
    path = tmp_path_factory.mktemp("opf") / "XXSYN_.OPF"
    path.write_text(SYNTHETIC_OPF, encoding="utf-8")
    return parse_bada3_file(path)


class TestSyntheticAircraft:
    def test_header(self, parsed):
        assert parsed.aircraft.model == "XXSYN_"
        assert parsed.aircraft.engines == 2
        assert parsed.aircraft.wtc == "M"


class TestSyntheticEngine:
    def test_max_thrust_coefficients(self, parsed):
        eng = parsed.engine
        assert eng.ct_1 == pytest.approx(1.4e5)
        assert eng.ct_2 == pytest.approx(5.0e4)
        assert eng.ct_3 == pytest.approx(6.0e-11)
        assert eng.ct_4 == pytest.approx(9.0)
        assert eng.ct_5 == pytest.approx(1.0e-2)

    def test_descent_thrust_column_order(self, parsed):
        """OPF order is Desc(low) Desc(high) Desc-level Desc(app) Desc(ld).

        Regression guard: these were unpacked with low and high swapped,
        inverting idle thrust above/below the transition altitude.
        """
        eng = parsed.engine
        assert eng.ct_des_low == pytest.approx(0.10)
        assert eng.ct_des_high == pytest.approx(0.05)
        assert eng.h_des == pytest.approx(20_000.0)
        assert eng.ct_des_app == pytest.approx(0.15)
        assert eng.ct_des_ld == pytest.approx(0.30)

    def test_fuel_coefficients(self, parsed):
        eng = parsed.engine
        assert eng.cf_1 == pytest.approx(0.70)
        assert eng.cf_2 == pytest.approx(2.0e3)
        assert eng.cf_3 == pytest.approx(10.0)
        assert eng.cf_4 == pytest.approx(6.0e4)
        assert eng.cf_crz == pytest.approx(0.95)


class TestSyntheticAerodynamics:
    def test_wing_area_and_polars(self, parsed):
        aero = parsed.aerodynamics
        assert aero.surface_area == pytest.approx(90.0)
        assert aero.cd_0[AerodynamicConfig.CR] == pytest.approx(0.025)
        assert aero.cd_2[AerodynamicConfig.CR] == pytest.approx(0.040)
        # Landing CD0 includes the gear increment (0.080 + 0.020)
        assert aero.cd_0[AerodynamicConfig.LD] == pytest.approx(0.100)

    def test_stall_speeds(self, parsed):
        assert parsed.limits.vstall[AerodynamicConfig.CR] == pytest.approx(145.0)
        assert parsed.limits.vstall[AerodynamicConfig.AP] == pytest.approx(100.0)


class TestSyntheticLimits:
    def test_masses_converted_to_kg(self, parsed):
        lim = parsed.limits
        assert lim.mref == pytest.approx(62_000.0)
        assert lim.oem == pytest.approx(40_000.0)
        assert lim.mtom == pytest.approx(75_000.0)
        assert lim.mpl == pytest.approx(20_000.0)
        assert lim.mmax == pytest.approx(lim.mtom)

    def test_envelope(self, parsed):
        lim = parsed.limits
        assert lim.vmo == pytest.approx(340.0)
        assert lim.mmo == pytest.approx(0.82)
        assert lim.hmo == pytest.approx(39_000.0)
        assert lim.hmax == pytest.approx(35_000.0)
        assert lim.gt == pytest.approx(-80.0)
        assert lim.gw == pytest.approx(0.45)

    def test_buffet_coefficients_skip_wing_area(self, parsed):
        """Clbo and k are the 2nd and 3rd sci-floats on the shared aero line.

        Regression guard: the parser used to capture (S, Clbo) as (clbo, k),
        i.e. the wing area ended up as the buffet onset lift coefficient.
        """
        lim = parsed.limits
        assert lim.clbo == pytest.approx(1.5)
        assert lim.k == pytest.approx(0.5)

    def test_buffet_onset_cl_is_linear_in_mach(self, parsed):
        model = BADA3LimitsModel(parsed.limits)
        # CL_bo(M) = Clbo - k * M = 1.5 - 0.5 * 0.5
        assert model.buffet_onset_cl(0.5) == pytest.approx(1.25)
