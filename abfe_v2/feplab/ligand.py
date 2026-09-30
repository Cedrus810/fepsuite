"""Ligand-only system construction and related utilities.

Ports of v1 ``generate_ligand_topology.py``, ``ligand_diameter.py``,
``resurrect_flexible.py``, ``make_ndx.py`` and ``rms_check.py``.
"""

from __future__ import annotations

import itertools

import numpy

from .errors import PipelineError
from .topology import find_ligand_range, parse_index, parse_top, write_ndx_group


# ---------------------------------------------------------------------------
# Ligand extraction (port of generate_ligand_topology.py)
# ---------------------------------------------------------------------------

def _write_topology_with_only_ligand(topology_in, molname, topology_out) -> None:
    section = None
    with open(topology_in) as fh, open(topology_out, "w") as ofh:
        for raw in fh:
            if section is None and raw.startswith("*"):
                ofh.write(raw)
                continue
            line = raw.split(";", 1)[0].strip()
            ls = line.split()
            if line.startswith("["):
                section = line.lstrip("[").rstrip("]").strip()
                ofh.write(raw)
                continue
            if line == "":
                pass
            elif line.startswith("#"):
                raise PipelineError("topology is not preprocessed")
            elif section == "molecules":
                if ls and ls[0] == molname:
                    ofh.write(raw)
                continue
            ofh.write(raw)


def extract_ligand(*, topology, mol, structure, index=None,
                   ligand_group="Ligand", output_ligand_structure,
                   output_ligand_topology, total_charge) -> None:
    """Extract the single ligand molecule into its own structure/topology."""
    import mdtraj  # lazy heavy import

    ptop = parse_top(topology)
    molcomposition = ptop.system
    moleculetypes = ptop.moleculetypes

    # Validation 1: the ligand must appear exactly once in [ molecules ].
    totcount = sum(count for (m, count) in molcomposition if m == mol)
    if totcount == 0:
        raise PipelineError("Ligand did not appear in topology")
    if totcount > 1:
        raise PipelineError("Ligand appeared more than once in topology")

    # Validation 2: the index group must match the topology layout.
    selected_from_top = None
    atomptr = 0
    for (m, c) in molcomposition:
        natom = len(moleculetypes[m])
        if m == mol:
            selected_from_top = [atomptr + i for i in range(natom)]
        atomptr += natom * c
    if index:
        indexinfo = parse_index(index)
        selected = indexinfo[ligand_group]
        if selected_from_top != selected:
            raise PipelineError("Index file mismatched with topology")
    else:
        selected = selected_from_top

    structure_trj = mdtraj.load(structure)
    outstr = structure_trj.atom_slice(selected)
    outstr.save(output_ligand_structure)

    _write_topology_with_only_ligand(topology, mol, output_ligand_topology)

    totalcharge = sum((a.charge or 0.0) for a in moleculetypes[mol])
    with open(total_charge, "w") as ofh:
        print("%.5f" % totalcharge, file=ofh)


# ---------------------------------------------------------------------------
# Ligand diameter (port of ligand_diameter.py)
# ---------------------------------------------------------------------------

def ligand_diameter(*, structure, trajectory=None, index=None,
                    ligand_mol="Ligand") -> tuple[float, float, float]:
    """Maximum interatomic distance statistics; returns (avg, max, safe).

    ``safe`` is a normal-tail extrapolation of the diameter distribution
    (the quantile regression of v1), used to size the pair-list cutoff.
    """
    import mdtraj  # lazy heavy import

    trajfile = str(trajectory) if trajectory is not None else str(structure)
    structure = str(structure)
    if index:
        ndx = parse_index(index)
        ligand = ndx[ligand_mol]
    else:
        ligand = list(range(mdtraj.load(structure).n_atoms))

    distpair = list(itertools.product(ligand, ligand))
    diameters = numpy.zeros((0,))
    for chunk in mdtraj.iterload(trajfile, top=structure):
        distances = mdtraj.compute_distances(chunk, distpair)
        diameters = numpy.concatenate((diameters, numpy.amax(distances, axis=1)))

    if len(diameters) == 1:
        safe = float(diameters[0]) * 1.5
    else:
        # Regression assuming the tail behaves like a normal distribution.
        obs_points = numpy.array([0.90, 0.93, 0.95, 0.97, 0.99])
        invsf = [1.2815515655446004, 1.4757910281791706, 1.6448536269514729,
                 1.8807936081512511, 2.3263478740408408]  # scipy.stats.norm.isf(1-p)
        target_invsf = 3.7190164854556804  # isf(1e-4)
        quantiles = numpy.quantile(diameters, obs_points)
        A = numpy.vstack([invsf, numpy.ones_like(invsf)]).T
        slope, intercept = numpy.linalg.lstsq(A, quantiles, rcond=None)[0]
        safe = float(slope * target_invsf + intercept)

    return (float(numpy.mean(diameters)), float(numpy.amax(diameters)), safe)


def write_diameter_txt(path, avg: float, max_: float, safe: float) -> None:
    with open(path, "w") as ofh:
        print("avg", avg, file=ofh)
        print("max", max_, file=ofh)
        print("safe", safe, file=ofh)


def read_safe_diameter(path) -> float | None:
    """Parse the ``safe`` value from a diameter.txt (None when missing)."""
    try:
        with open(path) as fh:
            for line in fh:
                ls = line.split()
                if ls and ls[0] == "safe":
                    return float(ls[1])
    except OSError:
        pass
    return None


# ---------------------------------------------------------------------------
# Flexible-solvent topology switching (port of resurrect_flexible.py)
# ---------------------------------------------------------------------------

def resurrect_flexible(*, flexible, topology, output, solvent="SOL") -> None:
    """Wrap the solvent moleculetype with #ifndef FLEXIBLE switches.

    The flexible (``-DFLEXIBLE``) solvent definition is copied from the
    preprocessed flexible topology so minimization can relax water while
    the production runs keep rigid water.
    """
    flexsol: list[str] = []

    def scan(flexpath):
        section = None
        in_solvent = False
        with open(flexpath) as fh:
            for raw in fh:
                if section is None and raw.startswith("*"):
                    continue
                line = raw.split(";", 1)[0].strip()
                ls = line.split()
                if line.startswith("["):
                    section = line.lstrip("[").rstrip("]").strip()
                    if in_solvent and section == "moleculetype":
                        in_solvent = False
                    elif in_solvent:
                        flexsol.append(raw)
                    continue
                if in_solvent:
                    flexsol.append(raw)
                if line == "":
                    pass
                elif line.startswith("#"):
                    raise PipelineError("topology is not preprocessed")
                elif section == "moleculetype":
                    if ls and ls[0] == solvent:
                        in_solvent = True

    scan(flexible)

    with open(topology) as fh, open(output, "w") as ofh:
        section = None
        in_solvent = False
        for raw in fh:
            if section is None and raw.startswith("*"):
                continue
            line = raw.split(";", 1)[0].strip()
            ls = line.split()
            if line.startswith("["):
                section = line.lstrip("[").rstrip("]").strip()
                if in_solvent and section == "moleculetype":
                    in_solvent = False
                    ofh.write("#else\n")
                    for lflex in flexsol:
                        ofh.write(lflex)
                    ofh.write("#endif\n")
                ofh.write(raw)
                continue
            ofh.write(raw)
            if line == "":
                pass
            elif line.startswith("#"):
                raise PipelineError("topology is not preprocessed")
            elif section == "moleculetype":
                if ls and ls[0] == solvent:
                    in_solvent = True
                    ofh.write("#ifndef FLEXIBLE\n")


# ---------------------------------------------------------------------------
# Index file generation (port of make_ndx.py)
# ---------------------------------------------------------------------------

def _find_center_atom(structure, ndxs: list[int]) -> int:
    """Non-hydrogen atom of the group closest to its centroid."""
    import mdtraj  # lazy heavy import

    centroid = mdtraj.compute_center_of_geometry(structure.atom_slice(ndxs))[0, :]
    nonh = structure.topology.select("not element H")
    nset = sorted(set(ndxs) & set(int(x) for x in nonh))
    if len(nset) == 0:
        raise PipelineError("no non-hydrogen atom in the group")
    dists2 = numpy.sum((structure.xyz[0, nset, :] - centroid[numpy.newaxis, :]) ** 2, axis=1)
    return nset[int(numpy.argmin(dists2))]


def make_ndx(*, structure, topology, output, ligand, receptor=None) -> None:
    """Write the pipeline index file (System/Ligand/Receptor groups)."""
    import mdtraj  # lazy heavy import

    gmxtop = parse_top(topology)
    structure_trj = mdtraj.load(structure)
    ligndx = find_ligand_range(gmxtop, ligand)
    with open(output, "w") as ofh:
        write_ndx_group(ofh, "System", structure_trj.topology.select("all"))
        write_ndx_group(ofh, "Ligand", ligndx)
        write_ndx_group(ofh, "Ligand_center", [_find_center_atom(structure_trj, ligndx)])
        if receptor is not None:
            rec = [int(x) for x in structure_trj.topology.select(receptor)]
            write_ndx_group(ofh, "Receptor", rec)
            write_ndx_group(ofh, "Receptor_center", [_find_center_atom(structure_trj, rec)])
            write_ndx_group(ofh, "Ligand+Receptor", sorted(set(ligndx) | set(rec)))


# ---------------------------------------------------------------------------
# (check_rms_average lives in feplab.restraints next to the other
# equilibration gates.)
# ---------------------------------------------------------------------------
