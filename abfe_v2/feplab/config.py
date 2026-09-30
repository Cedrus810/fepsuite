"""Configuration handling.

The pipeline reads the user configuration from ``para_conf.zsh`` files.
Keeping the zsh format has two advantages:

* the values stay readable and editable for people coming from v1, and
* ``controller.zsh`` sources the very same files to compute job resources,
  so a single configuration can never disagree with itself.

Only a restricted, well-defined subset of the shell syntax is accepted:
``KEY = value`` assignments with numbers or (optionally quoted) strings.
Anything dynamic (arrays, ``$``-expansions, command substitution) is
rejected with a clear error instead of being silently misparsed.
"""

import os
import re
from dataclasses import dataclass
from pathlib import Path

from .errors import ConfigError

_ASSIGN_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(\S.*)$")


def parse_para_conf(path) -> dict[str, str]:
    """Parse a ``para_conf.zsh``-style file into a plain string dict."""
    values: dict[str, str] = {}
    with open(path) as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if line == "" or line.startswith("#") or line.startswith(";"):
                continue
            m = _ASSIGN_RE.match(line)
            if m is None:
                raise ConfigError(f"{path}:{lineno}: cannot parse line: {raw.rstrip()}")
            key, value = m.group(1), m.group(2).strip()
            if value.startswith("("):
                raise ConfigError(
                    f"{path}:{lineno}: array values are not supported ({key})")
            if "$" in value or "`" in value:
                raise ConfigError(
                    f"{path}:{lineno}: variable/command expansion is not supported ({key})")
            if value[:1] in ("'", '"'):
                quote = value[:1]
                if len(value) < 2 or not value.endswith(quote):
                    raise ConfigError(
                        f"{path}:{lineno}: unbalanced quote in {key}={value}")
                value = value[1:-1]
            else:
                # In zsh, "KEY=value # comment" assigns only the first word.
                cut = value.find(" #")
                if cut >= 0:
                    value = value[:cut].strip()
            values[key] = value
    return values


class Config:
    """Typed view over the parsed configuration."""

    def __init__(self, values: dict[str, str], sources: list[str]):
        self._values = values
        self.sources = sources

    @classmethod
    def load(cls, rundir: Path, run_id: str | None = None,
             extra_path: str | None = None) -> "Config":
        """Load rundir/para_conf.zsh, overridden by rundir/<run_id>/para_conf.zsh.

        ``extra_path`` (typically from FEPLAB_PARA_CONF) replaces the base
        file when given.  This mirrors what controller.zsh sources.
        """
        values: dict[str, str] = {}
        sources: list[str] = []
        base = Path(extra_path) if extra_path else rundir / "para_conf.zsh"
        if base.exists():
            values.update(parse_para_conf(base))
            sources.append(str(base))
        if run_id:
            per_id = rundir / run_id / "para_conf.zsh"
            if per_id.exists():
                values.update(parse_para_conf(per_id))
                sources.append(str(per_id))
        return cls(values, sources)

    # ---- typed access -------------------------------------------------
    def get_str(self, key: str, default: str | None = None) -> str:
        if key in self._values:
            return self._values[key]
        env = os.environ.get(key)
        if env is not None and env != "":
            return env
        if default is None:
            raise ConfigError(f"configuration key {key} is required (set it in para_conf.zsh)")
        return default

    def get_int(self, key: str, default: int | None = None) -> int:
        raw = self.get_str(key, None if default is None else str(default))
        try:
            return int(float(raw))
        except ValueError:
            raise ConfigError(f"configuration key {key} must be an integer, got {raw!r}") from None

    def get_float(self, key: str, default: float | None = None) -> float:
        raw = self.get_str(key, None if default is None else str(default))
        try:
            return float(raw)
        except ValueError:
            raise ConfigError(f"configuration key {key} must be a number, got {raw!r}") from None

    def get_bool(self, key: str, default: bool) -> bool:
        raw = self._values.get(key)
        if raw is None:
            return default
        value = raw.strip().lower()
        if value in ("yes", "true", "on", "1"):
            return True
        if value in ("no", "false", "off", "0"):
            return False
        raise ConfigError(
            f"configuration key {key} must be yes/no, got {raw!r}")

    # ---- typed properties ---------------------------------------------
    # Replicas per phase
    @property
    def nrestr(self) -> int:
        return self.get_int("NRESTR", 4)

    @property
    def ncharge(self) -> int:
        return self.get_int("NCHARGE", 12)

    @property
    def nannih(self) -> int:
        return self.get_int("NANNIH", 12)

    # Parallelization
    @property
    def lig_para(self) -> int:
        return self.get_int("LIG_PARA")

    @property
    def complex_para(self) -> int:
        return self.get_int("COMPLEX_PARA")

    @property
    def tpp(self) -> int:
        return self.get_int("TPP", 1)

    # Simulation lengths
    @property
    def run_prod(self) -> float:
        return self.get_float("RUN_PROD", 2000.0)

    # System setup
    @property
    def lig_gmx(self) -> str:
        return self.get_str("LIG_GMX", "MOL")

    @property
    def receptor_sel(self) -> str:
        return self.get_str("RECEPTOR_MDTRAJ", "protein")

    @property
    def solvent_name(self) -> str:
        return self.get_str("SOLVENT", "SOL")

    @property
    def water_structure(self) -> str:
        return self.get_str("WATER_STRUCTURE", "spc216")

    @property
    def water_thickness(self) -> float:
        return self.get_float("WATER_THICKNESS", 1.0)

    @property
    def ionic_strength(self) -> float:
        return self.get_float("IONIC_STRENGTH", 0.150)

    @property
    def ion_positive(self) -> str:
        return self.get_str("ION_POSITIVE", "NA")

    @property
    def ion_negative(self) -> str:
        return self.get_str("ION_NEGATIVE", "CL")

    # Thresholding / sanity
    @property
    def eq_rmsd_cutoff(self) -> float:
        return self.get_float("EQ_RMSD_CUTOFF", 0.4)

    @property
    def ligand_diameter(self) -> float:
        return self.get_float("LIGAND_DIAMETER", 0.0)

    # Lambda optimization
    @property
    def annih_lambda_opt(self) -> int:
        return self.get_int("ANNIH_LAMBDA_OPT", 5)

    @property
    def annih_lambda_opt_length(self) -> float:
        return self.get_float("ANNIH_LAMBDA_OPT_LENGTH", 50.0)

    # Charge correction
    @property
    def apbs(self) -> str:
        return self.get_str("APBS", "apbs")

    @property
    def charge_correction_nsamp(self) -> int:
        return self.get_int("CHARGE_CORRECTION_NSAMP", 5)

    # Optional mdp key override (v1 supported NSTLIST but never defined it;
    # kept for compatibility).
    @property
    def nstlist(self) -> int | None:
        if "NSTLIST" in self._values:
            return self.get_int("NSTLIST")
        return None

    # ---- resource auto-tuning -------------------------------------------
    @property
    def auto_resource(self) -> bool:
        """Detect GPUs at runtime and derive the mdrun rank layout."""
        return self.get_bool("AUTO_RESOURCE", True)

    @property
    def prep_nonmpi(self) -> bool:
        """Run the pre-equilibration chain without MPI (plain gmx + threads)."""
        return self.get_bool("PREP_NONMPI", True)

    @property
    def gmx_nompi(self) -> str:
        """GROMACS binary for the non-MPI pre-equilibration runs."""
        return self.get_str("GMX_NOMPI", "gmx")

    # ---- long-range correction -------------------------------------------
    @property
    def lrc_sanity_limit(self) -> float:
        """Hard limit on |LRC term| in kcal/mol before the analysis fails.

        The PME re-evaluation of the soft-core decoupled annihilation
        endpoints is known to produce garbage `LJ recip.` energies on
        GROMACS >= 2024-ish builds (see docs/issue-lrc-gmx2026.md); the
        guard turns that into a loud failure instead of a poisoned
        result.txt.
        """
        return self.get_float("LRC_SANITY_LIMIT", 50.0)

    @property
    def skip_annihilation_lrc(self) -> bool:
        """Skip the LRC re-evaluation of the decoupled annihilation endpoints.

        Physically the lambda=1 state has no ligand-environment dispersion
        to correct; the two endpoint terms nearly cancel in the cycle
        (v1 measured ~0.1 kcal/mol each), so skipping them is a sane
        workaround for the GROMACS 2026 soft-core/PME-LJ breakage.
        """
        return self.get_bool("SKIP_ANNIHILATION_LRC", False)

    @property
    def lrc_energy_terms(self) -> list[str] | None:
        """Override the EDR energy-term names summed for the LRC."""
        raw = self._values.get("LRC_ENERGY_TERMS")
        if not raw:
            return None
        return [x.strip() for x in raw.split(",") if x.strip()]

    def validate(self) -> None:
        """Fail fast on values that would break the pipeline later."""
        for value, key in ((self.nrestr, "NRESTR"), (self.ncharge, "NCHARGE"),
                           (self.nannih, "NANNIH")):
            if value < 2:
                raise ConfigError(f"{key} must be >= 2 (BAR needs at least two states)")
        if self.annih_lambda_opt < 1:
            raise ConfigError(
                "ANNIH_LAMBDA_OPT must be >= 1: the annihilation-complex phase "
                "seeds its lambda schedule from the annihilation-lig prerun")
        if self.complex_para < 1 or self.lig_para < 1:
            raise ConfigError("LIG_PARA and COMPLEX_PARA must be >= 1")
        if self.charge_correction_nsamp < 0:
            raise ConfigError("CHARGE_CORRECTION_NSAMP must be >= 0 (0 disables the correction)")
