import tempfile
from pathlib import Path

import pytest

from dara.cif2str import STRPhaseParameters
from dara.generate_control_file import RefinementParameters
from dara.refine import do_refinement

TEST_DATA = Path(__file__).parent / "test_data"
# Every case gets itmax=1 so BGMN parses the file but barely iterates.
_FAST = {"itmax": 1}

_REFINEMENT_PARAM_CASES = {
    "eps1_refinable": {**_FAST, "eps1": "0_-0.01^0.01"},
    "eps2_fixed": {**_FAST, "eps2": 0.0},
    "eps3_enabled": {**_FAST, "eps3": "0_-0.01^0.01"},
    "eps4_enabled": {**_FAST, "eps4": 0.0},
    "wmin_wmax": {**_FAST, "wmin": 20.0, "wmax": 70.0},
    "onlyiso": {**_FAST, "onlyiso": True},
    "ru": {**_FAST, "ru": 8},
    "cut": {**_FAST, "cut": 0.5},
    "pol": {**_FAST, "pol": 0.8},
    "betaratio": {**_FAST, "betaratio": 0.001},
    "alpha3ratio": {**_FAST, "alpha3ratio": 0.001},
    "protokoll_off": {**_FAST, "protokoll": False},
    "save": {**_FAST, "save": "N"},
    "limits": {**_FAST, "limit2": 0.5, "limit4": 0.3, "limit6": 0.2, "limit8": 0.1, "limit10": 0.05},
    "anisolimits": {**_FAST, "anisolimit": 0.01, "aniso4limit": 0.005},
    "kitchen_sink": {
        **_FAST,
        "eps1": "0_-0.01^0.01",
        "eps2": "0_-0.05^0.05",
        "eps3": "0_-0.02^0.02",
        "eps4": "0_-0.01^0.01",
        "pol": 0.7,
        "betaratio": 0.001,
        "alpha3ratio": 0.001,
        "onlyiso": True,
        "ru": 6,
        "anisolimit": 0.02,
        "aniso4limit": 0.01,
        "limit2": 0.5,
        "limit4": 0.4,
        "limit6": 0.3,
    },
}

_PHASE_PARAM_CASES = {
    "defaults": {},
    "k2_enabled": {"k2": "0_0^0.01"},
    "k3_enabled": {"k3": "0_0^0.005"},
    "b2_enabled": {"b2": "0_0^0.005"},
    "sk_enabled": {"sk": "0_0^0.01"},
    "gewicht_sphar4": {"gewicht": "SPHAR4"},
    "lattice_fixed": {"lattice_range": "fixed"},
    "lebail": {"lebail": True},
    "kitchen_sink": {
        "k1": "0_0^0.05",
        "k2": "0_0^0.02",
        "b1": "0_0^0.01",
        "b2": "0_0^0.005",
        "gewicht": "SPHAR4",
    },
}


@pytest.fixture(scope="module")
def test_data():
    cif_paths = list(TEST_DATA.glob("*.cif"))
    pattern_path = TEST_DATA / "BiFeO3.xy"
    return pattern_path, cif_paths


def test_default_refinement(test_data):
    """Test the refinement function."""
    pattern_path, cif_paths = test_data

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        result = do_refinement(
            pattern_path,
            cif_paths,
            instrument_profile="Aeris-fds-Pixcel1d-Medipix3",
            working_dir=tmpdir,
        )
        assert result.lst_data.rwp < 8


@pytest.mark.parametrize("case_name, params", list(_REFINEMENT_PARAM_CASES.items()))
def test_refinement_params_accepted_by_bgmn(test_data, case_name, params, tmp_path):
    """BGMN should accept the .sav file produced by each RefinementParameters combo."""
    pattern_path, cif_paths = test_data
    refinement_params = RefinementParameters(**params)

    result = do_refinement(
        pattern_path,
        cif_paths,
        instrument_profile="Aeris-fds-Pixcel1d-Medipix3",
        working_dir=tmp_path / case_name,
        refinement_params=refinement_params,
    )
    # If BGMN rejected the file it would raise; a finite RWP means it ran.
    assert result.lst_data.rwp > 0, f"RWP should be positive for case '{case_name}'"


@pytest.mark.parametrize("case_name, params", list(_PHASE_PARAM_CASES.items()))
def test_phase_params_accepted_by_bgmn(test_data, case_name, params, tmp_path):
    """BGMN should accept the .str file produced by each STRPhaseParameters combo."""
    pattern_path, cif_paths = test_data
    phase_params = STRPhaseParameters(**params)

    result = do_refinement(
        pattern_path,
        cif_paths,
        instrument_profile="Aeris-fds-Pixcel1d-Medipix3",
        working_dir=tmp_path / case_name,
        phase_params=phase_params,
        refinement_params=RefinementParameters(itmax=1),
    )
    assert result.lst_data.rwp > 0, f"RWP should be positive for case '{case_name}'"
