"""Charge neutralization of the perturbed state — library port of
``feprest/neutralize.py``.

The A→B perturbation generally changes the total charge; leaving it
neutralized avoids the artifacts of a net-charged box.  The v1 scheme
mutates bulk water molecules into counter-ions *through* the FEP
states: ``[ atoms ]`` of a few SOL molecules become the SOL2pos/SOL2neg
hybrid moleculetypes (from ``<water_dir>/<ff>.ion.itp``), so state A is
pure water and state B contains the ion (or the reverse, in ``posonly``
mode, for multi-reference-state setups).

The port keeps the v1 behavior, including choosing the last sufficiently
large solvent block of ``[ molecules ]`` (crystalline-water blocks listed
earlier are left alone) and picking the converted waters randomly among
those within ``exclude_distance`` of any non-solvent atom.  The code is
split into a topology pass (:func:`neutralize_topology`) and a
coordinate pass (:func:`rewrite_gro`) so both are testable; a run-seed
parameter was added for reproducibility (v1 used the global ``random``).
"""

from __future__ import annotations

import os
import random
import sys
from dataclasses import dataclass, field

from .waterion import WATER_ION_MODELS_DIR


class WaterModel:
    """Parser for the ``*.ion.itp`` water/ion hybrid models.

    The files carry two magic comments: ``; SOLINFO <natom> <posresname>
    <negresname>`` and one ``; SOLCOORD x y z`` per hybrid atom.
    """

    def __init__(self, itpfile):
        self.natom = None
        self.posresname = None
        self.negresname = None
        self.atomnames = None
        self.coord = None
        self.contents = []

        self.parse_itpfile(itpfile)

    def parse_itpfile(self, file):
        self.contents = []
        with open(file) as fh:
            section = None
            molecule = None
            for l in fh:
                self.contents.append(l.rstrip())
                if l.startswith("; SOLINFO"):
                    ls = l.split()
                    assert ls[0] == ";" and ls[1] == "SOLINFO"
                    assert ls[1] == "SOLINFO", "Space/tab must be inserted after SOLINFO"
                    assert len(ls) == 5, "wrong SOLINFO fomat"
                    self.natom = int(ls[2])
                    self.posresname = ls[3]
                    self.negresname = ls[4]
                elif l.startswith("; SOLCOORD"):
                    ls = l.split()
                    assert ls[0] == ";" and ls[1] == "SOLCOORD"
                    assert ls[1] == "SOLCOORD", "Space/tab must be inserted after SOLCOORD"
                    assert self.natom is not None, "SOLCOORD must be after SOLINFO"
                    assert len(ls) == 2 + 3 * self.natom, "SOLCOORD atom number mismatch with SOLINFO"
                    self.coord = []
                    for ix in range(2, 2 + 3 * self.natom, 3):
                        self.coord.append([float(ls[ix]), float(ls[ix + 1]), float(ls[ix + 2])])
                else:
                    lsc = l.split(';', 1)
                    lcmd = lsc[0]
                    if lcmd.strip() == "":
                        continue
                    lcmd = lcmd.lstrip()
                    if lcmd.startswith('['):
                        ls = lcmd.split()
                        section = ls[1]
                        continue
                    if section == "moleculetype":
                        ls = lcmd.split()
                        molecule = ls[0]
                        if molecule == "SOL2pos":
                            self.atomnames = []
                        continue
                    if section == "atoms" and molecule == "SOL2pos":
                        ls = lcmd.split()
                        assert int(ls[0]) == len(self.atomnames) + 1, \
                            "[ atoms ] section atom number mismatch"
                        self.atomnames.append(ls[4])


def sign(x):
    if x < 0:
        return -1
    elif x > 0:
        return 1
    else:
        return 0


@dataclass
class NeutralizationPlan:
    """Outcome of the topology pass, consumed by :func:`rewrite_gro`."""

    dtot: float
    nchg: int
    frommol: str | None
    nfrom: int
    fepmol: str | None
    tomol: str | None
    nto: int
    exchange_mol: int | None
    mol_begin: list[int] = field(default_factory=list)


def neutralize_topology(topology_path, output_topology_path,
                        water_model: WaterModel, mode: str) -> NeutralizationPlan:
    """Rewrite ``[ molecules ]`` of the preprocessed topology so the charge
    change of the perturbation is neutralized (v1 topology pass)."""
    dmol = {}
    natoms = {}
    molecules = []
    with open(topology_path) as fh, open(output_topology_path, 'w') as ofh:
        sectionname = None
        moleculetype = None
        remain = []
        for lraw in fh:
            l = lraw.split(";")[0].strip()
            ls = l.split()
            if l == "":
                pass
            elif l.startswith("["):
                if sectionname == "molecules":
                    # section name after molecules
                    remain.append(lraw)
                    for lraw in fh:
                        remain.append(lraw)
                    break
                sectionname = l.split()[1]
                if sectionname == "system":
                    for l in water_model.contents:
                        ofh.write(l + "\n")
            elif l.startswith("#"):
                sys.stderr.write("Warning: input file must be preprocessed, but seems not\n")
            elif sectionname == "moleculetype":
                moleculetype = ls[0]
                _nrexcl = int(ls[1])
                dmol[moleculetype] = 0.
                natoms[moleculetype] = 0
            elif sectionname == "atoms":
                # sum up charges
                natoms[moleculetype] += 1
                if len(ls) >= 10:
                    # no charge difference unless atoms are perturbed
                    chga = float(ls[6])
                    chgb = float(ls[9])
                    dmol[moleculetype] += (chgb - chga)
            elif sectionname == "molecules":
                mol = ls[0]
                nmol = int(ls[1])
                molecules.append((mol, nmol))
                # do not write [ molecules ] yet, because we need to modify them
                continue
            ofh.write(lraw)

        dtot = 0
        mol_begin = [0]
        for (mol, nmol) in molecules:
            dtot += dmol[mol] * nmol
            mol_begin.append(mol_begin[-1] + natoms[mol] * nmol)
        print(f"Total charge change: {dtot:.4f}")
        nchg = abs(round(dtot))
        if abs(dtot) - nchg > 1e-3:
            raise RuntimeError("Noninteger charge perturbation")
        sgn = sign(dtot)
        if sgn == -1:
            print("Perturbing water(s) into positive ion(s)")
            # easy case: auto and posonly both use "SOL2pos"
            frommol = "SOL"
            nfrom = water_model.natom
            fepmol = "SOL2pos"
            tomol = water_model.posresname
        elif sgn == 1:
            if mode == "auto":
                print("Perturbing water(s) into negative ion(s)")
                frommol = "SOL"
                nfrom = water_model.natom
                fepmol = "SOL2neg"
                tomol = water_model.negresname
            elif mode == "posonly":
                print("Perturbing positive ion(s) into water(s)")
                frommol = water_model.posresname
                nfrom = 1
                fepmol = "pos2SOL"
                tomol = "SOL"
            else:
                raise RuntimeError(f"Unsupported mode: {mode}")
        else:
            print("No perturbation required")
            frommol = None
            nfrom = water_model.natom
            fepmol = None
            tomol = None
        nto = water_model.natom  # perturbed molecules always take this size

        output_mol = []
        exchange_mol = None
        # Find the last (frommol) molecule section. This peculiar feature is to
        # support the following case:
        # [ molecules ]
        # Protein 1
        # SOL 35
        # SOL 13367
        # In the above case, SOL 35 is likely to be crystalline water and may
        # not be preferable to be converted nor shuffled.
        for (i, (mol, nmol)) in list(enumerate(molecules))[::-1]:
            if exchange_mol is not None:
                output_mol.append((mol, nmol))
            else:
                if mol == frommol and nmol >= nchg:
                    exchange_mol = i
                    if nmol > nchg:
                        output_mol.append((mol, nmol - nchg))
                    output_mol.append((fepmol, nchg))
                else:
                    output_mol.append((mol, nmol))
        if exchange_mol is None and frommol is not None:
            raise RuntimeError(
                f"Unable to find {nchg} consecutive {frommol} molecules in the system ")

        output_mol = output_mol[::-1]
        for (mol, nmol) in output_mol:
            ofh.write(f"{mol}     {nmol}\n")
        for l in remain:
            ofh.write(l)
    # end of topology read/write fh/ofh

    return NeutralizationPlan(dtot=dtot, nchg=nchg, frommol=frommol,
                              nfrom=nfrom, fepmol=fepmol, tomol=tomol,
                              nto=nto, exchange_mol=exchange_mol,
                              mol_begin=mol_begin)


def rewrite_gro(gro_path, output_gro_path, plan: NeutralizationPlan,
                water_model: WaterModel, exclude_distance: float = 0.4,
                rng=random) -> None:
    """Coordinate pass: expand the chosen solvent molecules to the hybrid
    ion/water geometry (v1 gro pass)."""
    import mdtraj

    structure = mdtraj.load(gro_path)
    nonsolvent = structure.topology.select(
        f"not (resname SOL {water_model.posresname} {water_model.negresname})")
    if plan.exchange_mol is None:
        bsolv = 0
        esolv = 0
        far_chosen = []
        target_unchosen = []
    else:
        bsolv = plan.mol_begin[plan.exchange_mol]
        esolv = plan.mol_begin[plan.exchange_mol + 1]
        targets = list(range(bsolv, esolv, plan.nfrom))
        far = mdtraj.compute_neighbors(structure, exclude_distance,
                                       nonsolvent, targets)[0]
        far_chosen = rng.sample(list(far), plan.nchg)
        target_unchosen = sorted(set(targets).difference(set(far_chosen)))
    with open(gro_path) as fh, open(output_gro_path, 'w') as ofh:
        title = next(fh)
        ofh.write(title)
        lraw = next(fh)
        natom = int(lraw.strip())
        nshift = plan.nchg * (plan.nto - plan.nfrom)
        new_natom = natom + nshift  # may or may not change
        ofh.write(f"{new_natom:5d}\n")  # natom may exceed 5 digits but still this is OK
        outputbuf = []
        coords = []
        for lraw in fh:
            coords.append(lraw)
        box = coords[-1]
        del coords[-1]

        # output as-is until we hit the "solvent" region
        for i in range(0, bsolv):
            outputbuf.append(coords[i])

        # output sampled ions (perturbed)
        for c in far_chosen:
            if plan.nfrom < plan.nto:
                assert plan.nfrom == 1  # other cases are not considered below
                l = coords[c]
                # need to complement atoms not existing in the input file
                _resid = l[0:5]
                _resname = l[5:10]
                _atomname = l[10:15]
                atomno = l[15:20].strip()
                x = float(l[20:28].strip())
                y = float(l[28:36].strip())
                z = float(l[36:44].strip())
                basecrd = [x, y, z]
                remain = "\n"
                if len(l) > 44:
                    remain = l[44:]  # may have velocity info
                for i in range(plan.nto):
                    crd = [basecrd[0] + water_model.coord[i][0],
                           basecrd[1] + water_model.coord[i][1],
                           basecrd[2] + water_model.coord[i][2]]
                    # atomname is the same but gromacs will not complain
                    newline = l[0:5] + "%-5s" % "SOL" + "%5s" % water_model.atomnames[i] \
                        + l[15:20] + "%8.3f%8.3f%8.3f" % tuple(crd) + remain
                    outputbuf.append(newline)
            else:
                assert plan.nfrom == plan.nto
                for i in range(plan.nfrom):
                    outputbuf.append(coords[c + i])

        # output ions not chosen (as-is)
        for c in target_unchosen:
            for i in range(plan.nfrom):
                outputbuf.append(coords[c + i])

        # to the end
        for i in range(esolv, len(coords)):
            outputbuf.append(coords[i])

        for lw in outputbuf:
            ofh.write(lw)
        ofh.write(box)


def neutralize(topology, gro, output_topology, output_gro, mode: str = "auto",
               ff: str = "amber", water_dir: str | os.PathLike | None = None,
               exclude_distance: float = 0.4, rng=random) -> NeutralizationPlan:
    """Full port of neutralize.py main()."""
    if water_dir is None:
        water_dir = WATER_ION_MODELS_DIR
    water_model = WaterModel(os.path.join(str(water_dir), ff + ".ion.itp"))
    plan = neutralize_topology(topology, output_topology, water_model, mode)
    rewrite_gro(gro, output_gro, plan, water_model,
                exclude_distance=exclude_distance, rng=rng)
    return plan
