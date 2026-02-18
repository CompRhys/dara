"""Generate a control file for BGMN."""

from __future__ import annotations

import re
import shutil
import warnings
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from dara.utils import read_phase_name_from_str


class RefinementParametersParameters(BaseModel):
    """BGMN SAV control file parameters.

    See http://www.bgmn.de/variables.html and http://www.bgmn.de/calculat.html
    for full documentation.
    """

    # --- Threading ---
    n_threads: int = Field(default=8, description="Number of threads for parallel computation.")

    # --- Angular range ---
    wmin: float | None = Field(default=None, description="Minimum 2-theta angle.")
    wmax: float | None = Field(default=None, description="Maximum 2-theta angle.")

    # --- Wavelength ---
    wavelength: Literal["Cu", "Co", "Cr", "Fe", "Mo"] | float = Field(
        default="Cu",
        description="Wavelength: tube target material string or synchrotron wavelength in nm.",
    )
    betaratio: float | None = Field(default=None, description="Kb/Ka intensity ratio.")
    alpha3ratio: float | None = Field(default=None, description="Ka3/Ka1 intensity ratio.")

    # --- Instrument corrections ---
    eps1: float | str = Field(default=0.0, description="Zero-point correction.")
    eps2: float | str = Field(default="0_-0.05^0.05", description="Sample displacement correction.")
    eps3: float | str | None = Field(default=None, description="Specimen transparency correction.")
    eps4: float | str | None = Field(default=None, description="Fourth instrument correction parameter.")

    # --- Polarization ---
    pol: float | str | None = Field(default=None, description="Polarization factor (e.g. for monochromator setups).")

    # --- Refinement control ---
    itmax: int | None = Field(default=None, description="Maximum number of refinement iterations.")
    cut: float | None = Field(default=None, description="Peak cutoff distance from center.")
    onlyiso: bool | None = Field(default=None, description="Restrict to isotropic displacement parameters only.")

    # --- Background ---
    unt: str | None = Field(default=None, description="Path to measured background file.")
    ru: int | None = Field(default=None, description="Number of background polynomial terms.")

    # --- Anisotropy limits ---
    anisolimit: float | None = Field(default=None, description="Anisotropic broadening threshold.")
    aniso4limit: float | None = Field(default=None, description="4th-order anisotropic broadening threshold.")

    # --- Peak limits ---
    limit2: float | None = Field(default=None, description="SPHAR2 preferred orientation limit.")
    limit4: float | None = Field(default=None, description="SPHAR4 preferred orientation limit.")
    limit6: float | None = Field(default=None, description="SPHAR6 preferred orientation limit.")
    limit8: float | None = Field(default=None, description="SPHAR8 preferred orientation limit.")
    limit10: float | None = Field(default=None, description="SPHAR10 preferred orientation limit.")

    # --- Output ---
    protokoll: bool = Field(default=True, description="Enable verbose refinement progress output.")
    save: str | None = Field(default=None, description="Save intermediate results (Y/N).")

    @classmethod
    def coerce(cls, value: RefinementParametersParameters | dict | None) -> RefinementParametersParameters:
        """Normalise *value* to a ``RefinementParametersParameters`` instance."""
        if value is None:
            return cls()
        if isinstance(value, dict):
            return cls(**value)
        return value

    def to_sav_lines(self) -> list[str]:
        """Return non-None parameters as BGMN SAV-file lines."""
        lines: list[str] = []

        # Wavelength
        if isinstance(self.wavelength, str):
            lines.append(f"LAMBDA={self.wavelength.upper()}")
        else:
            lines.append(f"SYNCHROTRON={self.wavelength:.4f}")

        if self.wmin is not None:
            lines.append(f"WMIN={self.wmin}")
        if self.wmax is not None:
            lines.append(f"WMAX={self.wmax}")

        # EPS parameters — refinable ones get PARAM[] wrappers
        param_idx = 1
        if isinstance(self.eps1, str):
            lines.append(f"PARAM[{param_idx}]=EPS1={self.eps1}")
            param_idx += 1
        else:
            lines.append(f"EPS1={self.eps1}")

        if isinstance(self.eps2, str):
            lines.append(f"PARAM[{param_idx}]=EPS2={self.eps2}")
            param_idx += 1
        else:
            lines.append(f"EPS2={self.eps2}")

        if self.eps3 is not None:
            if isinstance(self.eps3, str):
                lines.append(f"PARAM[{param_idx}]=EPS3={self.eps3}")
                param_idx += 1
            else:
                lines.append(f"EPS3={self.eps3}")

        if self.eps4 is not None:
            if isinstance(self.eps4, str):
                lines.append(f"PARAM[{param_idx}]=EPS4={self.eps4}")
                param_idx += 1
            else:
                lines.append(f"EPS4={self.eps4}")

        # Spectral line ratios
        if self.betaratio is not None:
            lines.append(f"betaratio={self.betaratio}")
        if self.alpha3ratio is not None:
            lines.append(f"alpha3ratio={self.alpha3ratio}")

        # Polarization
        if self.pol is not None:
            lines.append(f"POL={self.pol}")

        # Refinement control
        if self.itmax is not None:
            lines.append(f"ITMAX={self.itmax}")
        if self.cut is not None:
            lines.append(f"CUT={self.cut}")
        if self.onlyiso is not None:
            lines.append(f"ONLYISO={'Y' if self.onlyiso else 'N'}")

        # Background
        if self.unt is not None:
            lines.append(f"UNT={self.unt}")
        if self.ru is not None:
            lines.append(f"RU={self.ru}")

        # Anisotropy limits
        if self.anisolimit is not None:
            lines.append(f"ANISOLIMIT={self.anisolimit}")
        if self.aniso4limit is not None:
            lines.append(f"ANISO4LIMIT={self.aniso4limit}")

        # Peak limits
        for n in (2, 4, 6, 8, 10):
            val = getattr(self, f"limit{n}")
            if val is not None:
                lines.append(f"LIMIT{n}={val}")

        # Threading & output
        lines.append(f"NTHREADS={self.n_threads}")
        lines.append(f"PROTOKOLL={'Y' if self.protokoll else 'N'}")
        if self.save is not None:
            lines.append(f"SAVE={self.save}")

        return lines


def copy_instrument_files(instrument_profile: str | Path, working_dir: Path) -> str:
    """
    Copy the instrument file (.geq) to the working directory.

    Args:
        working_dir: the working directory

    Returns
    -------
        The name of the instrument
    """
    default_instrument_path = Path(__file__).parent / "data" / "BGMN-Templates" / "Devices"
    instrument_path = Path(instrument_profile)  # try to parse as a path
    if instrument_path.suffix != ".geq" or not instrument_path.exists():
        instrument_profile = instrument_path.name.removesuffix(".geq")
        instrument_path = default_instrument_path / f"{instrument_profile}.geq"

    if not instrument_path.exists():
        raise FileNotFoundError(
            f"Could not find the instrument file ({instrument_profile} in both "
            f"the provided path and the default path ({default_instrument_path})."
        )

    shutil.copy(instrument_path, working_dir)
    return instrument_path.stem


def copy_xy_pattern(pattern_path: Path, working_dir: Path) -> Path:
    """Copy the xy pattern to the working directory."""
    # if same directory, do nothing
    if pattern_path.parent != working_dir:
        shutil.copy(pattern_path, working_dir)
    return working_dir / pattern_path.name


def trim_pattern(xy_content: np.ndarray) -> np.ndarray:
    """Trim the pattern to remove negative intensities."""
    if xy_content[:, 1].min() <= 0:
        warnings.warn("Pattern contains negative or zero intensities. Setting them to 1e-6.")
        xy_content[:, 1] = np.clip(xy_content[:, 1], 1e-6, None)

    if xy_content[:, 0].min() < 1.0:
        warnings.warn("Pattern contains 2-theta values below 1.0. Remove them.")
        xy_content = xy_content[xy_content[:, 0] >= 1.0]

    return xy_content


def generate_control_file(
    pattern_path: Path,
    str_paths: list[Path],
    instrument_profile: str | Path,
    working_dir: Path | None = None,
    *,
    refinement_params: RefinementParametersParameters | None = None,
    **kwargs,
) -> Path:
    """
    Generate a control file for BGMN.

    Args:
        pattern_path: the path to the pattern file. It has to be in `.xy` format
        str_paths: the paths to the STR files
        instrument_profile: the name of the instrument, if it is a path, it must be ended with `.geq`
        working_dir: the working directory
        refinement_params: RefinementParametersParameters instance. If not provided, one is built from **kwargs.
        **kwargs: forwarded to RefinementParametersParameters if refinement_params is not given.

    """
    if refinement_params is None:
        refinement_params = RefinementParametersParameters(**kwargs)

    if working_dir is None:
        control_file_path = pattern_path.parent / f"{pattern_path.stem}.sav"
    else:
        control_file_path = working_dir / f"{pattern_path.stem}.sav"

    copy_xy_pattern(pattern_path, control_file_path.parent)
    instrument_name = copy_instrument_files(instrument_profile, control_file_path.parent)

    xy_pattern_path = control_file_path.parent / pattern_path.name

    try:
        xy_content = np.loadtxt(pattern_path)
    except ValueError as e:
        raise ValueError(f"Could not load pattern file {pattern_path}") from e

    xy_content = trim_pattern(xy_content)
    np.savetxt(xy_pattern_path, xy_content, fmt="%.6f")

    phases_str = "\n".join([f"STRUC[{i}]={str_path.name}" for i, str_path in enumerate(str_paths, start=1)])

    phase_names = [read_phase_name_from_str(str_path) for str_path in str_paths]
    phase_fraction_str = "\n".join([f"Q{phase_name}={phase_name}/sum" for phase_name in phase_names])
    goal_str = "\n".join([f"GOAL[{i}]=Q{phase_name}" for i, phase_name in enumerate(phase_names, start=1)])

    param_lines = "\n".join(refinement_params.to_sav_lines())

    control_file = f"""
    % Theoretical instrumental function
    VERZERR={instrument_name}.geq
    % Phases
    {phases_str}
    % Measured data
    VAL[1]={pattern_path.name}
    % Result list output
    LIST={pattern_path.stem}.lst
    % Peak list output
    OUTPUT={pattern_path.stem}.par
    % Diagram output
    DIAGRAMM={pattern_path.stem}.dia
    % Global parameters
    {param_lines}
    sum={"+".join(phase_name for phase_name in phase_names)}
    {phase_fraction_str}
    {goal_str}
    """
    control_file = re.sub(r"^\s+", "", control_file, flags=re.MULTILINE)

    with open(control_file_path, "w") as f:
        f.write(control_file)

    return control_file_path
