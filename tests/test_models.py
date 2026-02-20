"""Tests for Pydantic parameter models and their coerce/output methods."""

from __future__ import annotations

import pytest

from dara.cif2str import STRPhaseParameters
from dara.generate_control_file import RefinementParameters
from dara.refine import RefinementPhase

# ---------------------------------------------------------------------------
# STRPhaseParameters
# ---------------------------------------------------------------------------


class TestSTRPhaseParameters:
    """Tests for STRPhaseParameters defaults, validation, and coerce."""

    def test_defaults(self):
        p = STRPhaseParameters()
        assert p.lattice_range == 0.01
        assert p.rp == 4
        assert p.k1 == "0_0^0.01"
        assert p.k2 == "fixed"
        assert p.k3 is None
        assert p.b1 == "0_0^0.005"
        assert p.b2 is None
        assert p.sk is None
        assert p.gewicht == "0_0"
        assert p.lebail is False

    def test_coerce_none(self):
        result = STRPhaseParameters.coerce(None)
        assert isinstance(result, STRPhaseParameters)
        assert result == STRPhaseParameters()

    def test_coerce_dict(self):
        result = STRPhaseParameters.coerce({"k1": "0_0^0.05", "lebail": True})
        assert isinstance(result, STRPhaseParameters)
        assert result.k1 == "0_0^0.05"
        assert result.lebail is True
        # other fields keep defaults
        assert result.rp == 4

    def test_coerce_passthrough(self):
        original = STRPhaseParameters(k2="0_0^0.01")
        result = STRPhaseParameters.coerce(original)
        assert result is original

    def test_validation_good_params(self):
        """Various valid refinement parameter formats should be accepted."""
        STRPhaseParameters(k1="0_0^0.01")
        STRPhaseParameters(k1="0.005_0.0^0.02")
        STRPhaseParameters(k1="fixed")
        STRPhaseParameters(k1="0")
        STRPhaseParameters(b1="0_0^0.005")

    def test_validation_bad_param(self):
        with pytest.raises(ValueError, match="Invalid refinement parameter"):
            STRPhaseParameters(k1="not_a_number")

    def test_validation_none_optional_fields(self):
        """None should be accepted for optional fields (k3, b2, sk)."""
        p = STRPhaseParameters(k3=None, b2=None, sk=None)
        assert p.k3 is None
        assert p.b2 is None
        assert p.sk is None

    def test_lattice_range_fixed(self):
        p = STRPhaseParameters(lattice_range="fixed")
        assert p.lattice_range == "fixed"


# ---------------------------------------------------------------------------
# RefinementParameters
# ---------------------------------------------------------------------------


class TestRefinementParameters:
    """Tests for RefinementParameters defaults, coerce, and SAV output."""

    def test_defaults(self):
        p = RefinementParameters()
        assert p.n_threads == 8
        assert p.wavelength == "Cu"
        assert p.eps1 == 0.0
        assert p.eps2 == "0_-0.05^0.05"
        assert p.protokoll is True
        assert p.wmin is None
        assert p.wmax is None

    def test_coerce_none(self):
        result = RefinementParameters.coerce(None)
        assert isinstance(result, RefinementParameters)
        assert result == RefinementParameters()

    def test_coerce_dict(self):
        result = RefinementParameters.coerce({"wavelength": "Co", "n_threads": 4})
        assert isinstance(result, RefinementParameters)
        assert result.wavelength == "Co"
        assert result.n_threads == 4

    def test_coerce_passthrough(self):
        original = RefinementParameters(wavelength="Mo")
        result = RefinementParameters.coerce(original)
        assert result is original

    # -- to_sav_lines tests --

    def test_sav_lines_defaults(self):
        """Default parameters should produce valid SAV lines."""
        lines = RefinementParameters().to_sav_lines()
        assert "LAMBDA=CU" in lines
        assert "EPS1=0.0" in lines
        assert any(l.startswith("PARAM[") and "EPS2=" in l for l in lines)
        assert "NTHREADS=8" in lines
        assert "PROTOKOLL=Y" in lines

    def test_sav_lines_synchrotron(self):
        p = RefinementParameters(wavelength=0.1234)
        lines = p.to_sav_lines()
        assert "SYNCHROTRON=0.1234" in lines
        assert not any("LAMBDA" in l for l in lines)

    def test_sav_lines_wavelength_variants(self):
        for wl in ("Cu", "Co", "Cr", "Fe", "Mo"):
            lines = RefinementParameters(wavelength=wl).to_sav_lines()
            assert f"LAMBDA={wl.upper()}" in lines

    def test_sav_lines_optional_fields(self):
        """Optional fields should only appear when set."""
        p_default = RefinementParameters()
        lines_default = p_default.to_sav_lines()

        # These should NOT appear with defaults
        assert not any("WMIN=" in l for l in lines_default)
        assert not any("WMAX=" in l for l in lines_default)
        assert not any("ITMAX=" in l for l in lines_default)
        assert not any("CUT=" in l for l in lines_default)
        assert not any("SAVE=" in l for l in lines_default)

        # These SHOULD appear when set
        p = RefinementParameters(wmin=5.0, wmax=80.0, itmax=200, cut=0.5, save="Y")
        lines = p.to_sav_lines()
        assert "WMIN=5.0" in lines
        assert "WMAX=80.0" in lines
        assert "ITMAX=200" in lines
        assert "CUT=0.5" in lines
        assert "SAVE=Y" in lines

    def test_sav_lines_eps_refinable(self):
        """EPS parameters given as strings should get PARAM[] wrappers."""
        p = RefinementParameters(eps1="0_-0.01^0.01", eps2="0_-0.05^0.05")
        lines = p.to_sav_lines()
        eps1_line = [l for l in lines if "EPS1=" in l][0]
        eps2_line = [l for l in lines if "EPS2=" in l][0]
        assert eps1_line.startswith("PARAM[")
        assert eps2_line.startswith("PARAM[")

    def test_sav_lines_eps_fixed(self):
        """EPS parameters given as floats should NOT get PARAM[] wrappers."""
        p = RefinementParameters(eps1=0.0, eps2=0.0)
        lines = p.to_sav_lines()
        eps1_line = [l for l in lines if "EPS1=" in l][0]
        eps2_line = [l for l in lines if "EPS2=" in l][0]
        assert not eps1_line.startswith("PARAM[")
        assert not eps2_line.startswith("PARAM[")

    def test_sav_lines_limits(self):
        p = RefinementParameters(limit2=0.5, limit4=0.3)
        lines = p.to_sav_lines()
        assert "LIMIT2=0.5" in lines
        assert "LIMIT4=0.3" in lines

    def test_sav_lines_onlyiso(self):
        lines_on = RefinementParameters(onlyiso=True).to_sav_lines()
        lines_off = RefinementParameters(onlyiso=False).to_sav_lines()
        assert "ONLYISO=Y" in lines_on
        assert "ONLYISO=N" in lines_off


# ---------------------------------------------------------------------------
# RefinementPhase
# ---------------------------------------------------------------------------


class TestRefinementPhase:
    """Tests for RefinementPhase construction and helpers."""

    def test_make_from_string(self, tmp_path):
        cif = tmp_path / "test.cif"
        cif.touch()
        phase = RefinementPhase.make(str(cif))
        assert phase.path == cif
        assert isinstance(phase.params, STRPhaseParameters)

    def test_make_from_path(self, tmp_path):
        cif = tmp_path / "test.cif"
        cif.touch()
        phase = RefinementPhase.make(cif)
        assert phase.path == cif

    def test_make_with_params_dict(self, tmp_path):
        cif = tmp_path / "test.cif"
        cif.touch()
        phase = RefinementPhase.make(cif, params={"k1": "0_0^0.05"})
        assert phase.params.k1 == "0_0^0.05"

    def test_make_with_kwargs(self, tmp_path):
        cif = tmp_path / "test.cif"
        cif.touch()
        phase = RefinementPhase.make(cif, k1="0_0^0.05")
        assert phase.params.k1 == "0_0^0.05"

    def test_make_rejects_both_params_and_kwargs(self, tmp_path):
        cif = tmp_path / "test.cif"
        cif.touch()
        with pytest.raises(ValueError, match="Cannot pass both"):
            RefinementPhase.make(cif, params={"k1": "0_0^0.05"}, k2="fixed")

    def test_make_passthrough(self, tmp_path):
        cif = tmp_path / "test.cif"
        cif.touch()
        original = RefinementPhase(path=cif)
        result = RefinementPhase.make(original)
        assert result.path == original.path

    def test_with_params(self, tmp_path):
        cif = tmp_path / "test.cif"
        cif.touch()
        phase = RefinementPhase(path=cif)
        updated = phase.with_params(k1="0_0^0.1")
        assert updated.params.k1 == "0_0^0.1"
        # original is unchanged (frozen model)
        assert phase.params.k1 == "0_0^0.01"

    def test_coerce_params_from_dict(self, tmp_path):
        cif = tmp_path / "test.cif"
        cif.touch()
        phase = RefinementPhase(path=cif, params={"lebail": True})
        assert phase.params.lebail is True

    def test_hash_and_equality(self, tmp_path):
        cif = tmp_path / "test.cif"
        cif.touch()
        a = RefinementPhase(path=cif)
        b = RefinementPhase(path=cif, params={"k1": "0_0^0.1"})
        assert a == b  # equality is path-based
        assert hash(a) == hash(b)
