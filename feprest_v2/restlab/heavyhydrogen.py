"""Hydrogen mass repartitioning of the equilibration topology — library
port of ``feprest/turn-heavy.py``.

The state-A equilibration runs (``nvt``/``npt`` stages) use a topology
in which every atom lighter than ``threshold`` amu gets its mass (both
A and B states) replaced by ``weight`` amu, so the equilibration
timesteps stay stable despite hydrogen motions.  Solvent moleculetypes
listed in ``ignore_moleculetype`` (default SOL) keep their masses.
"""

from __future__ import annotations


def turn_heavy(topology_path, output_path, ignore_moleculetype: str = "SOL",
               threshold: float = 3.5, weight: float = 8.0) -> None:
    ignored_molecules = set(ignore_moleculetype.split())
    section = None
    molecule = None
    outbuf = []
    with open(topology_path) as fh:
        for l in fh:
            outbuf.append(l)
            if section is None:
                if l.startswith("*"):
                    continue
            l = l.rstrip()
            lcomment = l.split(';', 1)
            l = lcomment[0]
            l = l.strip()
            if l == "":
                continue
            elif l.startswith("#"):
                continue
            elif l.startswith("["):
                sectionstr = l.lstrip("[")
                sectionstr = sectionstr.rstrip("]")
                sectionstr = sectionstr.strip()
                section = sectionstr
                continue
            ls = l.split()
            if ls == []:
                continue
            if section == "moleculetype":
                molecule = l.split()[0]
            elif section == "atoms" and molecule not in ignored_molecules:
                if len(ls) > 7:
                    # only when mass exists. Otherwise we just ignore.
                    to_be_replaced = False
                    for replace_at in (7, 10):  # state A, state B
                        if len(ls) > replace_at:
                            mass = float(ls[replace_at])
                            if mass < threshold:
                                ls[replace_at] = str(weight)
                                to_be_replaced = True
                    if to_be_replaced:
                        # replace the line (v1 drops the trailing comment here)
                        del outbuf[-1]
                        outbuf.append(" ".join(ls) + "\n")
    with open(output_path, "w") as ofh:
        for l in outbuf:
            ofh.write(l)
