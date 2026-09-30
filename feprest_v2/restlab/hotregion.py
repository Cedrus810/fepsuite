"""REST2 hot-region detection — library port of ``feprest/add_underline.py``
and ``feprest/underlined_group.py``.

:func:`underline_topology` finds every perturbed atom of the dual
topology (A/B atomtype or charge differs), grows the set to whole
residues within ``distance`` nm of any perturbed atom (mdtraj
neighbor search against the target molecules), and rewrites the
preprocessed topology appending ``_`` to the atomtype of the hot
atoms.  The suffixed types are what :mod:`restlab.rest2.scaling` later
duplicates with scaled parameters.

:func:`write_hot_index` emits the ``[ hot ]`` index group (every atom
whose type is underlined or which carries an A/B difference), consumed
by the HREX patch as the ``energygrps = hot`` group.
"""

from __future__ import annotations

from collections import defaultdict

from .errors import PipelineError


def parse_perturbed_atoms(topology_path, ignored_moleculetypes: set[str]):
    """Perturbed (molecule, atom0-based-index) pairs + composition tables.

    An atom is perturbed when its A and B atomtypes or charges differ
    (``[ atoms ]`` fields 8/9 vs 1/6), except inside moleculetypes listed
    in ``ignored_moleculetypes``.
    """
    perturbed = []
    molcomposition = []
    natom_moleculetypes = {}
    section = None
    molecule = None
    natom = 0
    with open(topology_path) as fh:
        for l in fh:
            if l.startswith("["):
                section = l.split()[1]
                if section in ["molecules", "system", "moleculetype"]:
                    # "molecules" / "system" must be after the molecular sections
                    if molecule is not None:
                        natom_moleculetypes[molecule] = natom
                        molecule = None
                        natom = 0
                continue
            if ';' in l:
                l = l.split(';')[0]
            if l.strip() == "":
                continue
            ls = l.split()
            if section == "moleculetype":
                molecule = ls[0]
                continue
            if section == "atoms":
                atomno = int(ls[0]) - 1
                atype = ls[1]
                btype = ls[8] if len(ls) > 8 else atype
                acharge = float(ls[6])
                bcharge = float(ls[9]) if len(ls) > 9 else acharge
                if (atype != btype or acharge != bcharge) \
                        and molecule not in ignored_moleculetypes:
                    perturbed.append((molecule, atomno))
                natom += 1
            elif section == "molecules":
                molecule = ls[0]
                nmol = int(ls[1])
                molcomposition.append((molecule, nmol))
    return perturbed, molcomposition, natom_moleculetypes


def molecule_atom_starts(molcomposition, natom_moleculetypes):
    """First-atom offsets: per-molecule list and a by-name default dict."""
    atomno_starts = []
    curatomno = 0
    starts_by_name = defaultdict(list)
    for (m, n) in molcomposition:
        atomno_starts.append(curatomno)
        natom_mol = natom_moleculetypes[m]
        for k in range(n):
            starts_by_name[m].append(curatomno + natom_mol * k)
        curatomno += natom_mol * n
    return atomno_starts, starts_by_name


def system_perturbed_atoms(perturbed, starts_by_name,
                           ignore_perturbing_multiple_molecules: bool = False):
    """System-wide indices of perturbed atoms."""
    perturbed_atomno = []
    for (m, a) in perturbed:
        if len(starts_by_name[m]) > 1 and not ignore_perturbing_multiple_molecules:
            raise PipelineError(
                f"Molecule named {m} are perturbed but there are multiple "
                f"({len(starts_by_name[m])}) molecules in the system")
        for ba in starts_by_name[m]:
            perturbed_atomno.append(ba + a)
    perturbed_atomno.sort()
    return perturbed_atomno


def compute_hot_atoms(confname, perturbed_atomno, distance: float,
                      target_molecule: str) -> set[int]:
    """Whole residues within ``distance`` nm of any perturbed atom."""
    import mdtraj
    gro = mdtraj.load(confname)
    protein = gro.topology.select(target_molecule)
    neighbors = mdtraj.compute_neighbors(gro, distance,
                                         query_indices=perturbed_atomno,
                                         haystack_indices=protein)
    neighbors = neighbors[0]
    rest_residues = set()
    rest_atoms = set()

    # list residues that need to be RESTed
    for n in neighbors:
        res = gro.topology.atom(n).residue
        rest_residues.add(res)

    # list atoms that need to be perturbed
    for r in rest_residues:
        for a in r.atoms:
            rest_atoms.add(a.index)
    return rest_atoms, rest_residues


def underline_topology(topology_path, output_path, starts_by_name,
                       rest_atoms: set[int], ignored_moleculetypes: set[str],
                       header: str | None = None):
    """Rewrite the topology appending ``_`` to hot atoms' atomtypes."""
    with open(topology_path) as fh, open(output_path, "w") as ofh:
        if header is not None:
            print(header, file=ofh)
        section = None
        molecule = None
        molecule_offset = None
        for l in fh:
            comment = ""
            if l.startswith("["):
                section = l.split()[1]
            elif ';' in l:
                (l, comment) = l.split(';', 1)
                comment = "; " + comment
            elif l.strip() == "":
                pass
            elif section == "moleculetype":
                molecule = l.split()[0]
                if molecule in starts_by_name and len(starts_by_name[molecule]) >= 1:
                    molecule_offset = starts_by_name[molecule][0]
                else:
                    molecule_offset = None
            elif section == "atoms" and molecule not in ignored_moleculetypes \
                    and molecule_offset is not None:
                # 3rd condition == molecule is actually listed in [ molecules ]
                ls = l.split()
                atomno = int(ls[0]) - 1
                if molecule_offset + atomno in rest_atoms:
                    l = " ".join([ls[0], ls[1] + "_"] + ls[2:]) + "\n"
            print(l.rstrip() + comment, file=ofh)


def add_underline(structure, topology, output, distance: float,
                  target_molecule: str =
                  "not (resname HOH SOL NA CL Na Cl K SOD CLA)",
                  non_perturbed_moleculetype: str = "SOL SOL2pos SOL2neg",
                  ignore_perturbing_multiple_molecules: bool = False) -> None:
    """Full port of add_underline.py main()."""
    ignored_sections = set(non_perturbed_moleculetype.split())

    perturbed, molcomposition, natom_moleculetypes = \
        parse_perturbed_atoms(topology, ignored_sections)
    _, starts_by_name = molecule_atom_starts(molcomposition, natom_moleculetypes)
    perturbed_atomno = system_perturbed_atoms(
        perturbed, starts_by_name, ignore_perturbing_multiple_molecules)
    rest_atoms, rest_residues = compute_hot_atoms(
        structure, perturbed_atomno, distance, target_molecule)

    # residue ids (v1 typo "undelined" kept: the line appears in the output)
    residue_ids = [r.resSeq for r in rest_residues]
    header = "; undelined resids =  " + repr(sorted(residue_ids))
    underline_topology(topology, output, starts_by_name, rest_atoms,
                       ignored_sections, header=header)


def collect_scaled_atoms(topology_path) -> tuple[int, list[int]]:
    """Total atom count and the system-wide indices of REST2-scaled atoms.

    Port of underlined_group.py's scan: an atom is scaled when its
    (already underlined) atomtype ends with ``_`` or its B-state
    type/charge differs from the A state.
    """
    moltable = {}
    scaled_indices = []
    tot_natom = 0
    with open(topology_path) as fh:
        section = None
        molname = None
        scaledatoms = []
        natom = 0
        for l in fh:
            if l.startswith("["):
                if section == "atoms":
                    moltable[molname] = (natom, scaledatoms)
                section = l.split()[1]
                if section == "atoms":
                    scaledatoms = []
                    natom = 0
                continue
            if ';' in l:
                l = l.split(';')[0]
            if l.strip() == "":
                continue
            if section == "moleculetype":
                molname = l.split()[0]
                continue
            if section == "atoms":
                ls = l.split()
                aindex = int(ls[0])
                atype = ls[1]
                if atype.endswith("_") or (len(ls) >= 10 and
                                           (ls[8] != atype or
                                            float(ls[9]) != float(ls[6]))):
                    scaledatoms.append(aindex - 1)
                natom += 1
            if section == "molecules":
                ls = l.split()
                molname = ls[0]
                molcount = int(ls[1])
                (natom, scaledatoms) = moltable[molname]
                for i in range(molcount):
                    for c in scaledatoms:
                        scaled_indices.append(c + i * natom + tot_natom)
                tot_natom += natom * molcount
                continue
    return tot_natom, scaled_indices


def write_hot_index(topology_path, output_path) -> None:
    """Write the [ System ] / [ hot ] index file (underlined_group.py)."""
    tot_natom, scaled_indices = collect_scaled_atoms(topology_path)
    with open(output_path, "w") as ofh:
        print("[ System ]", file=ofh)
        for i in range(tot_natom):
            print(i + 1, file=ofh)
        print(file=ofh)
        print("[ hot ]", file=ofh)
        for i in scaled_indices:
            print(i + 1, file=ofh)
        print(file=ofh)
