"""Automatically configures DARA settings."""

from importlib.metadata import version

from dara.cif2str import STRPhaseParameters
from dara.generate_control_file import RefinementParametersParameters
from dara.refine import RefinementPhase, do_refinement, do_refinement_no_saving
from dara.search import search_phases
from dara.settings import DaraSettings

__version__ = version("dara-xrd")
SETTINGS = DaraSettings()

__all__ = [
    "SETTINGS",
    "RefinementParametersParameters",
    "RefinementPhase",
    "STRPhaseParameters",
    "do_refinement",
    "do_refinement_no_saving",
    "search_phases",
]
