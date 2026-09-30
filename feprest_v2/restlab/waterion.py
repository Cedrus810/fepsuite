"""Flexible-water topology restoration — library port of
``feprest/recover-water.py``.

``gmx grompp -pp`` bakes the processed water moleculetype (rigid, SETTLE)
into the preprocessed topology; the minimization steps with
``-DFLEXIBLE`` need the original flexible water back, so the water
moleculetype section is replaced by the contents of
``<water_dir>/<ff>.water.itp``.
"""

from __future__ import annotations

import os
from pathlib import Path

# shipped alongside this package (copied from feprest/water_ion_models)
WATER_ION_MODELS_DIR = Path(__file__).resolve().parent.parent / "water_ion_models"


def recover_water(topology_path, output_path, ff: str,
                  water_moltype: str = "SOL",
                  water_dir: str | os.PathLike | None = None) -> None:
    if water_dir is None:
        water_dir = WATER_ION_MODELS_DIR
    outbuf = []
    ignore = False
    with open(topology_path) as fh:
        section = None
        for l in fh:
            if not ignore:
                outbuf.append(l)
            if l.startswith("["):
                section = l.split()[1]
                if section == "moleculetype":
                    if ignore:
                        outbuf.append(l)
                    ignore = False  # end ignoring
                continue
            if ';' in l:
                l = l.split(';')[0]
            if l.strip() == "":
                continue
            if section == "moleculetype":
                molname = l.split()[0]
                if molname == water_moltype:
                    del outbuf[-1]  # prevent dup
                    ignore = True
                    # Copy contents of *.water.itp
                    with open(os.path.join(str(water_dir), f"{ff}.water.itp")) as itpfh:
                        for li in itpfh:
                            if li.startswith('[') and li.split()[1] == "moleculetype":
                                pass  # prevent dup
                            else:
                                outbuf.append(li)
    with open(output_path, "w") as ofh:
        for l in outbuf:
            ofh.write(l)
