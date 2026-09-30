"""GROMACS topology / index file parsing.

Port of ``abfe/common_gmx_files.py`` (used by index generation, ligand
extraction, restraint search, and the charge correction), with the same
tolerance for the various shapes of ``[ atomtypes ]`` lines that GROMACS
emits across versions.  Topologies must be preprocessed (``gmx grompp
-pp``); ``#include``/``#define`` lines are rejected with a clear error.
"""

from typing import NamedTuple


class AtomRow(NamedTuple):
    atomtype: str
    resnr: int
    resname: str
    atomname: str
    charge: float | None


class Topology:
    def __init__(self):
        self.atomtypes: dict[str, tuple] = {}
        self.system: list[tuple[str, int]] = []
        self.moleculetypes: dict[str, list[AtomRow]] = {}
        self.defaults: list | None = None


def parse_top(topfile) -> Topology:
    top = Topology()
    section = None
    molecule = None
    atomlist: list[AtomRow] = []
    with open(topfile) as fh:
        for raw in fh:
            if section is None and raw.startswith("*"):
                continue
            line = raw.split(";", 1)[0].strip()
            if line == "":
                continue
            if line.startswith("#"):
                raise RuntimeError("topology is not preprocessed")
            if line.startswith("["):
                section = line.lstrip("[").rstrip("]").strip()
                if section in ("molecules", "system", "moleculetype"):
                    # "molecules"/"system" come after all molecular sections;
                    # moleculetype starts a new molecule.
                    if molecule is not None:
                        top.moleculetypes[molecule] = atomlist
                        molecule = None
                        atomlist = []
                continue
            ls = line.split()
            if not ls:
                continue
            if section == "defaults":
                defaults = ls
                defaults[0] = int(defaults[0])  # nbfunc
                defaults[1] = int(defaults[1])  # comb-rule
                defaults[3] = float(defaults[3])  # fudgeLJ
                defaults[4] = float(defaults[4])  # fudgeQQ
                top.defaults = defaults
            elif section == "atomtypes":
                # The field layout of [ atomtypes ] varies across GROMACS
                # versions; toppush.cpp's parser is famously messy.  The
                # following mirrors common_gmx_files.py of v1.
                if len(ls[5]) == 1 and ls[5].isalpha():
                    # "If field 5 is a single char we have both."
                    have_bonded_type = True
                    have_atomic_number = True
                elif len(ls[3]) == 1 and ls[3].isalpha():
                    # "If field 3 (starting from 0) is a single char,
                    #  we have neither bonded_type or atomic numbers."
                    have_bonded_type = False
                    have_atomic_number = False
                else:
                    # GROMACS issue 4120 changed the layout: field 1 is an
                    # atomic number when it is fully numeric.  int() is more
                    # permissive than C atoi() (e.g. "3_10"), hence isdigit.
                    have_atomic_number = all(c.isdigit() for c in ls[1])
                    have_bonded_type = not have_atomic_number

                atomtype = ls[0]
                (mass, charge, particle, sigc6, epsc12) = \
                    ls[1 + int(have_bonded_type) + int(have_atomic_number):]
                mass = float(mass)
                charge = float(charge)
                sigc6 = float(sigc6)
                epsc12 = float(epsc12)

                if have_bonded_type:
                    bondtype = ls[1]
                    if all(c.isdigit() for c in ls[1]):
                        raise RuntimeError(
                            f'[ atomtypes ] contains bondtype "{atomtype}", but the bondtype '
                            "for this atomtype consists of all digits.\n"
                            "This is considered invalid atomtype in GROMACS.")
                else:
                    bondtype = atomtype
                if have_atomic_number:
                    atomic_number = int(ls[1 + int(have_bonded_type)])
                else:
                    atomic_number = 0

                if top.defaults is None:
                    raise RuntimeError("[ atomtypes ] appeared before [ defaults ]")
                if top.defaults[1] == 1:
                    # geometric combination rule with C6/C12
                    if sigc6 == 0.0 or epsc12 == 0.0:
                        sigma = 0.0
                        eps = 0.0
                    else:
                        sigma = (epsc12 / sigc6) ** (1.0 / 6.0)
                        eps = sigc6 / (4.0 * (sigma ** 6))
                elif top.defaults[1] in (2, 3):
                    # (2) arithmetic / (3) geometric mean with sigma/eps
                    eps = epsc12
                    sigma = sigc6
                else:
                    raise RuntimeError(f"Unknown [ defaults ] comb-rule: {top.defaults[1]}")
                top.atomtypes[atomtype] = (bondtype, particle, atomic_number, mass,
                                           charge, sigma, eps)
            elif section == "moleculetype":
                molecule = ls[0]
                atomlist = []
            elif section == "atoms":
                charge = float(ls[6]) if len(ls) >= 7 else None
                atomlist.append(AtomRow(ls[1], int(ls[2]), ls[3], ls[4], charge))
            elif section == "molecules":
                top.system.append((ls[0], int(ls[1])))
    if molecule is not None:
        top.moleculetypes[molecule] = atomlist
    return top


def parse_index(ndx) -> dict[str, list[int]]:
    """Parse an ndx file into group name -> 0-based atom indices."""
    group = None
    ret: dict[str, list[int]] = {}
    with open(ndx) as fh:
        for line in fh:
            ls = line.split()
            if line.startswith("["):
                group = ls[1]
                if group in ret:
                    raise RuntimeError("Same group appeared more than once in the index file")
                ret[group] = []
            elif group is not None:
                ret[group].extend(int(x) - 1 for x in ls)
    return ret


def write_ndx_group(fh, name: str, indices) -> None:
    print(f"[ {name} ]", file=fh)
    buf = ""
    for i, ind in enumerate(indices):
        buf += str(ind + 1)
        buf += "\n" if i % 10 == 9 else " "
    print(buf.rstrip() + "\n", file=fh)


def find_ligand_range(top: Topology, molname: str) -> list[int]:
    """0-based atom indices of the single ligand molecule in the system."""
    atomptr = 0
    found: list[int] | None = None
    for mol, count in top.system:
        natom = len(top.moleculetypes[mol])
        if mol == molname:
            if count == 0:
                continue
            if count != 1 or found is not None:
                raise RuntimeError("More than one ligands are not supported")
            found = list(range(atomptr, atomptr + natom))
        atomptr += natom * count
    if found is None:
        raise RuntimeError("Ligand not found: %s (candidates are: %s)"
                           % (molname, " ".join(top.moleculetypes.keys())))
    return found
