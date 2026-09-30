"""Configuration handling.

The pipeline reads the user configuration from ``para_conf.zsh`` files.
Keeping the zsh format means ``controller.zsh`` sources the very same
files to compute job resources, so a single configuration can never
disagree with itself.

Only a restricted subset of the shell syntax is accepted: ``KEY = value``
assignments with literal numbers or (optionally quoted) strings.  Values
with ``$``-expansions are rejected with a clear error — the same file is
parsed by both zsh and Python, so dynamic syntax would be a trap.  This
matters when carrying over an old ``feprest`` ``para_conf.zsh``: the v1
template used ``REFINIT=$BASECONF``, which must be spelled out literally
(``REFINIT=conf_ionized.pdb``) for v2.
"""

import os
import re
from dataclasses import dataclass
from pathlib import Path

from .errors import ConfigError

_ASSIGN_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(\S.*)?$")


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
            key, value = m.group(1), (m.group(2) or "").strip()
            if value == "":
                values[key] = value  # "KEY=" assigns the empty string in zsh
                continue
            if value.startswith("("):
                raise ConfigError(
                    f"{path}:{lineno}: array values are not supported ({key})")
            if "$" in value or "`" in value:
                raise ConfigError(
                    f"{path}:{lineno}: variable/command expansion is not supported "
                    f"({key}); write the literal value instead")
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

        ``extra_path`` (typically from RESTLAB_PARA_CONF) replaces the base
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

    # ---- input structure -----------------------------------------------
    @property
    def baseconf(self) -> str:
        return self.get_str("BASECONF")

    @property
    def basetop(self) -> str:
        return self.get_str("BASETOP")

    # ---- REST2 / mutation setup ------------------------------------------
    @property
    def charge_mode(self) -> str:
        return self.get_str("CHARGE", "auto")

    @property
    def ff(self) -> str:
        return self.get_str("FF", "amber")

    @property
    def water_moltype(self) -> str:
        return self.get_str("WATER_MOLTYPE", "SOL")

    @property
    def rest2_region_distance(self) -> float:
        return self.get_float("REST2_REGION_DISTANCE", 0.4)

    @property
    def rest2_temp(self) -> float:
        return self.get_float("REST2_TEMP", 1200.0)

    @property
    def rest2_temp0(self) -> float:
        return self.get_float("REST2_TEMP0", 300.0)

    # ---- replicas and parallelization ------------------------------------
    @property
    def nrep(self) -> int:
        return self.get_int("NREP", 32)

    @property
    def para(self) -> int:
        return self.get_int("PARA", 8)

    @property
    def tpp(self) -> int:
        return self.get_int("TPP", 1)

    # ---- production -------------------------------------------------------
    @property
    def ntune(self) -> int:
        return self.get_int("NTUNE", 5)

    @property
    def simlength(self) -> float:
        return self.get_float("SIMLENGTH", 4000.0)

    @property
    def replica_interval(self) -> int:
        return self.get_int("REPLICA_INTERVAL", 1000)

    @property
    def sampling_interval(self) -> int:
        return self.get_int("SAMPLING_INTERVAL", 100)

    @property
    def basewarn(self) -> int:
        return self.get_int("BASEWARN", 1)

    @property
    def refinit(self) -> str | None:
        """Reference coordinates of the minimization/eq chain (unset: none)."""
        value = self.get_str("REFINIT", "")
        return value or None

    @property
    def refcrd(self) -> str | None:
        """Extra restraints during FEP (useful for protein complex)."""
        value = self.get_str("REFCRD", "")
        return value or None

    @property
    def domain_shrink(self) -> float:
        return self.get_float("DOMAIN_SHRINK", 0.6)

    # ---- replica-parameter tuning ------------------------------------------
    @property
    def run_tpr_parallel(self) -> bool:
        """grompp/convert-tpr of all replicas concurrently (they are slow)."""
        return self.get_bool("RUN_TPR_PARALLEL", True)

    @property
    def repopt_extend_run_length(self) -> float:
        return self.get_float("EXTEND_RUN_LENGTH", 50.0)

    @property
    def repopt_replex_interval(self) -> int:
        return self.get_int("REPOPT_REPLEX_INTERVAL", 100)

    # ---- optional mdrun flags ----------------------------------------------
    @property
    def nstlist_cmd(self) -> str:
        return self.get_str("NSTLIST_CMD", "")

    # ---- hot-region detection (add_underline) -------------------------------
    @property
    def non_perturbed_moleculetype(self) -> str:
        return self.get_str("NON_PERTURBED_MOLECULETYPE", "SOL SOL2pos SOL2neg")

    @property
    def target_molecule(self) -> str:
        return self.get_str("TARGET_MOLECULE",
                            "not (resname HOH SOL NA CL Na Cl K SOD CLA)")

    def validate(self) -> None:
        """Fail fast on values that would break the pipeline later."""
        if self.nrep < 2:
            raise ConfigError("NREP must be >= 2 (BAR needs at least two states)")
        if self.para < 1:
            raise ConfigError("PARA must be >= 1")
        if self.tpp < 1:
            raise ConfigError("TPP must be >= 1")
        if self.ntune < 1:
            raise ConfigError("NTUNE must be >= 1 (the npt-replex stage picks "
                              "up the states tuned in the last cycle)")
        if self.simlength <= 0:
            raise ConfigError("SIMLENGTH must be > 0")
        if self.replica_interval <= 0 or self.sampling_interval <= 0:
            raise ConfigError("REPLICA_INTERVAL and SAMPLING_INTERVAL must be > 0")
        if self.charge_mode not in ("auto", "no", "posonly"):
            raise ConfigError(
                f"CHARGE must be one of auto/no/posonly, got {self.charge_mode!r}")
        if not self.rest2_region_distance > 0:
            raise ConfigError("REST2_REGION_DISTANCE must be > 0")
        if self.rest2_temp <= self.rest2_temp0:
            raise ConfigError(
                "REST2_TEMP must be higher than REST2_TEMP0 "
                f"({self.rest2_temp} <= {self.rest2_temp0})")
