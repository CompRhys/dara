"""Perform refinements with BGMN."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from dara.bgmn_worker import BGMNWorker
from dara.cif2str import STRPhaseParameters, cif2str
from dara.generate_control_file import RefinementParameters, generate_control_file
from dara.result import RefinementResult, get_result
from dara.xrd import rasx2xy, raw2xy, xrdml2xy


class RefinementPhase(BaseModel, frozen=True):
    """
    Input phase for refinement.

    Contains the path to the phase file and the specific parameters for the phase.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    path: Path = Field(..., description="The path to the phase file.")
    params: STRPhaseParameters = Field(
        default_factory=STRPhaseParameters,
        kw_only=True,
        description="Per-phase STR parameters.",
    )

    @field_validator("params", mode="before")
    @classmethod
    def _coerce_params(cls, v):
        """Accept a STRPhaseParameters instance or a dict."""
        if isinstance(v, dict):
            return STRPhaseParameters(**v)
        return v

    @field_validator("path", mode="before")
    @classmethod
    def _validate_path(cls, v):
        return Path(v)

    def with_params(self, **kwargs) -> RefinementPhase:
        """Return a new RefinementPhase with updated STR parameters."""
        params = self.params.model_copy(update=kwargs)
        return self.model_copy(update={"params": params})

    def __hash__(self):
        return hash(self.path)

    def __eq__(self, other: RefinementPhase):
        return self.path == other.path

    @classmethod
    def make(
        cls,
        path_obj: RefinementPhase | Path | str,
        params: STRPhaseParameters | dict | None = None,
        **kwargs,
    ) -> RefinementPhase:
        """
        Make an RefinementPhase object from a path object. If the path object is already an
        RefinementPhase object, return it (optionally updating its params).
        If the path object is a string or Path object, create an RefinementPhase object
        with the path object.

        Args:
            path_obj: the path object, can be a string, Path object, or RefinementPhase object.
            params: optional STR phase parameters (dict, STRPhaseParameters, or keyword arguments).
            **kwargs: STR phase parameter fields passed as keyword arguments (cannot be combined with params).

        Returns
        -------
            RefinementPhase object
        """
        if params is not None and kwargs:
            raise ValueError("Cannot pass both 'params' and keyword arguments.")

        phase = path_obj if isinstance(path_obj, RefinementPhase) else cls(path=Path(path_obj))
        params = kwargs or params
        params = STRPhaseParameters(**(params or {})) if not isinstance(params, STRPhaseParameters) else params
        return phase.model_copy(update={"params": params})


def _merge_phase_params(defaults: STRPhaseParameters, overrides: STRPhaseParameters) -> STRPhaseParameters:
    """Overlay per-phase overrides onto the default phase parameters."""
    base = defaults.model_dump()
    base.update(overrides.model_dump(exclude_defaults=True))
    return STRPhaseParameters(**base)


def do_refinement(
    pattern_path: Path | str,
    phases: list[RefinementPhase | Path | str],
    instrument_profile: str | Path = "Aeris-fds-Pixcel1d-Medipix3",
    working_dir: Path | str | None = None,
    phase_params: STRPhaseParameters | dict | None = None,
    refinement_params: RefinementParameters | dict | None = None,
    show_progress: bool = False,
) -> RefinementResult:
    """Refine the structure using BGMN."""
    refinement_params = RefinementParameters.coerce(refinement_params)

    pattern_path = Path(pattern_path)
    working_dir = (
        Path(working_dir) if working_dir is not None else pattern_path.parent / f"refinement_{pattern_path.stem}"
    )

    if not working_dir.exists():
        working_dir.mkdir(exist_ok=True, parents=True)

    phase_params = STRPhaseParameters.coerce(phase_params)

    if pattern_path.suffix == ".xrdml":
        pattern_path = xrdml2xy(pattern_path, working_dir)
    elif pattern_path.suffix == ".raw":
        pattern_path = raw2xy(pattern_path, working_dir)
    elif pattern_path.suffix == ".rasx":
        pattern_path = rasx2xy(pattern_path, working_dir)

    str_paths = []
    for phase_path in phases:
        phase = RefinementPhase.make(phase_path)
        phase_path_ = phase.path
        # Merge default phase params with per-phase overrides
        merged_params = _merge_phase_params(phase_params, phase.params)
        if phase_path_.suffix == ".cif":
            str_path = cif2str(phase_path_, "", working_dir, phase_params=merged_params)
        else:
            if phase_path_.parent != working_dir:
                shutil.copy(phase_path_, working_dir)
            str_path = working_dir / phase_path_.name
        str_paths.append(str_path)

    control_file_path = generate_control_file(
        pattern_path=pattern_path,
        str_paths=str_paths,
        instrument_profile=instrument_profile,
        working_dir=working_dir,
        refinement_params=refinement_params,
    )

    bgmn_worker = BGMNWorker()
    bgmn_worker.run_refinement_cmd(control_file_path, show_progress=show_progress)
    return get_result(control_file_path)


def do_refinement_no_saving(
    pattern_path: Path,
    phases: list[RefinementPhase | Path | str],
    instrument_profile: str | Path = "Aeris-fds-Pixcel1d-Medipix3",
    phase_params: STRPhaseParameters | dict | None = None,
    refinement_params: RefinementParameters | dict | None = None,
    show_progress: bool = False,
) -> RefinementResult:
    """Refine the structure using BGMN in a temporary directory without saving."""
    with tempfile.TemporaryDirectory() as tmpdir:
        working_dir = Path(tmpdir)

        return do_refinement(
            pattern_path=pattern_path,
            phases=phases,
            instrument_profile=instrument_profile,
            working_dir=working_dir,
            phase_params=phase_params,
            refinement_params=refinement_params,
            show_progress=show_progress,
        )
