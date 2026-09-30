"""Boresch restraint handling for the ``restraints`` stage.

Three pieces, ported from v1 (``find_restr_from_md.py``,
``generate_restr.py``, ``rms_check.py``) with the numerics unchanged:

* automatic selection of the six anchor atoms and the restraint
  coordinates from the equilibration trajectory,
* generation of the GROMACS pull-code mdp / index files (and the
  intermolecular-interaction itp variant),
* the analytical standard-state correction of Boresch et al.
  (J. Phys. Chem. B 107, 9535 (2003)), used by the final report,
* the equilibration RMSD gate.

The spring constants are defined once here and shared with the report
(v1 duplicated them with slightly different values 41.848 vs 41.84).
"""

from __future__ import annotations

import itertools
import math

import numpy

from .errors import PipelineError
from .units import V0_STANDARD_STATE, kbt

# Restraint force constants (v1 values; angle/dihedral standardized on
# 41.84 which is what the pipeline actually used).
SPRING_DISTANCE = 4184.0  # kJ/mol/nm^2
SPRING_ANGLE = 41.84      # kJ/mol/rad^2
SPRING_DIHEDRAL = 41.84   # kJ/mol/rad^2


def read_restrinfo(path) -> tuple[list[int], list[float]]:
    """Parse a restrinfo file.

    Returns the six 0-based anchor atom indices and the average
    coordinates ``[dist, angle_a, angle_b, dihed_a, dihed_b, dihed_c]``
    in nm / radians.
    """
    with open(path) as fh:
        lines = [x for x in fh.readlines() if not x.startswith("#")]
    if len(lines) < 2:
        raise PipelineError(f"malformed restrinfo file: {path}")
    anchors = [int(x) for x in lines[0].split()]
    avgs = [float(x) for x in lines[1].split()]
    if len(anchors) != 6 or len(avgs) != 6:
        raise PipelineError(f"malformed restrinfo file: {path}")
    return anchors, avgs


def analytical_restraint_free_energy(avgs, *, temp: float,
                                     distance_spring: float = SPRING_DISTANCE,
                                     angle_spring: float = SPRING_ANGLE,
                                     dihedral_spring: float = SPRING_DIHEDRAL) -> float:
    """Analytical standard-state correction, in kJ/mol (v1 read_restr).

    ``avgs`` = [dist, angle_a, angle_b, ...] with angles in radians;
    returns ``-RT * mdeltaf`` with

        mdeltaf = log(8 pi^2 V0) + 0.5 log(Kr) + log(Ktheta)
                  + 1.5 log(Kphi) - 2 log(r) - log(sin theta)
                  - log(sin phi) - 3 log(2 pi RT)
    """
    v0 = V0_STANDARD_STATE  # 1 M standard state in nm^3
    RT = kbt(temp)
    mdeltaf = (math.log(8 * math.pi ** 2 * v0)
               + 0.5 * math.log(distance_spring)
               + 1.0 * math.log(angle_spring)
               + 1.5 * math.log(dihedral_spring)
               - 2.0 * math.log(avgs[0])
               - math.log(math.sin(avgs[1]))
               - math.log(math.sin(avgs[2]))
               - 3.0 * math.log(2 * math.pi * RT))
    return -RT * mdeltaf


# ---------------------------------------------------------------------------
# RMSD gate of the equilibration
# ---------------------------------------------------------------------------

def check_rms_average(xvg_path, threshold: float = 0.4) -> float:
    """Average the xvg second column; fail when it exceeds the threshold."""
    xsum = 0.0
    nsum = 0
    with open(xvg_path) as fh:
        for line in fh:
            if line.startswith("@") or line.startswith("#"):
                continue
            ls = line.split()
            if not ls:
                continue
            xsum += float(ls[1])
            nsum += 1
    if nsum == 0:
        raise PipelineError(f"no data points in {xvg_path}")
    average = xsum / nsum
    if average > threshold:
        raise PipelineError(
            f"RMS was too large (average {average:f}, threshold {threshold:f})")
    return average


# ---------------------------------------------------------------------------
# Anchor-atom search (port of find_restr_from_md.py)
# ---------------------------------------------------------------------------

def find_restraints(*, topology, trajectory, index, prot_sel="Receptor",
                    lig_sel="Ligand", search_dist=0.5,
                    anchor_atoms="CB,CA,C,N,O", distance_weight=4184.0,
                    angle_weight=41.84, dihedral_weight=41.84, output) -> None:
    """Choose the Boresch anchor atoms and restraint averages from MD.

    The score is the weighted sum of coordinate variances; tuples with
    average angles outside 45-135 degrees are banned (the log sin term
    of the analytical correction becomes unstable there).
    """
    import mdtraj  # heavy dependency, imported lazily

    refstructure = mdtraj.load(topology)
    mdtop = refstructure.topology

    def iter_trajectory():
        return mdtraj.iterload(trajectory, top=mdtop)

    indices = None
    from .topology import parse_index
    indices = parse_index(index)
    protix = indices[prot_sel]
    ligix = indices[lig_sel]
    ligheavyix = list(set(mdtop.select("not type H")).intersection(set(ligix)))

    # Validate the anchor atom names before scanning the trajectory.
    anchor_names = anchor_atoms.split(",")
    anchors = list(set(mdtop.select(f'name {" ".join(anchor_names)}')).intersection(protix))
    if len(anchors) == 0:
        raise PipelineError("No match for anchor atoms in system")
    if len(set(ligheavyix).intersection(anchors)) > 0:
        raise PipelineError("Ligand and receptor atoms are overlapping")

    # Candidate atoms: neighbors on the other side of the interface.
    neighbors_lig = mdtraj.compute_neighbors(refstructure, search_dist, anchors,
                                             haystack_indices=ligheavyix)[0]
    neighbors_anchor = mdtraj.compute_neighbors(refstructure, search_dist, ligheavyix,
                                                haystack_indices=anchors)[0]
    if len(neighbors_lig) == 0 or len(neighbors_anchor) == 0:
        raise PipelineError(
            f"Number of neighbor atoms in ligand was {len(neighbors_lig)}, "
            f"in anchor was {len(neighbors_anchor)}")

    atom_pairs = list(itertools.product(neighbors_anchor, neighbors_lig))
    distances = mdtraj.compute_distances(refstructure, atom_pairs)[0, :]
    filtered_pairs = [p for p, d in zip(atom_pairs, distances) if d <= search_dist]

    print(f"There are {len(filtered_pairs)} pairs of atoms considered")
    if len(filtered_pairs) == 0:
        raise PipelineError(f"Could not find pairs within {search_dist} nm")

    def find_bonded(a, searchindex):
        threshold = 0.22  # nm, sufficiently long for any bond
        found = mdtraj.compute_neighbors(refstructure, threshold, [a],
                                         [x for x in searchindex if x != a])[0]
        return found

    def find_two_bonds(a, searchindex):
        bonded = find_bonded(a, searchindex)
        ret = []
        for b in bonded:
            angled = [x for x in find_bonded(b, searchindex) if x != a]
            for c in angled:
                ret.append((a, b, c))
        return ret

    # Enumerate every possible 6-atom combination around each contact pair.
    anclig_list = []
    for (anc, lig) in filtered_pairs:
        anc_angles = find_two_bonds(anc, anchors)
        lig_angles = find_two_bonds(lig, ligheavyix)
        anc_rev_angles = [(c, b, a) for (a, b, c) in anc_angles]
        anclig_list.extend(itertools.product(anc_rev_angles, lig_angles))

    print(f"There are {len(anclig_list)} sets of 6-atom tuples considered")

    dihed_indices_a = [[anc[0], anc[1], anc[2], lig[0]] for anc, lig in anclig_list]
    dihed_indices_b = [[anc[1], anc[2], lig[0], lig[1]] for anc, lig in anclig_list]
    dihed_indices_c = [[anc[2], lig[0], lig[1], lig[2]] for anc, lig in anclig_list]
    angle_indices_a = [[anc[1], anc[2], lig[0]] for anc, lig in anclig_list]
    angle_indices_b = [[anc[2], lig[0], lig[1]] for anc, lig in anclig_list]
    dist_indices = [[anc[2], lig[0]] for anc, lig in anclig_list]

    npairs = len(dist_indices)
    diheds_a = numpy.zeros((0, npairs))
    diheds_b = numpy.zeros((0, npairs))
    diheds_c = numpy.zeros((0, npairs))
    angles_a = numpy.zeros((0, npairs))
    angles_b = numpy.zeros((0, npairs))
    dists = numpy.zeros((0, npairs))
    for chunk in iter_trajectory():
        dists = numpy.concatenate((dists, mdtraj.compute_distances(chunk, dist_indices)), axis=0)
        angles_a = numpy.concatenate((angles_a, mdtraj.compute_angles(chunk, angle_indices_a)), axis=0)
        angles_b = numpy.concatenate((angles_b, mdtraj.compute_angles(chunk, angle_indices_b)), axis=0)
        diheds_a = numpy.concatenate((diheds_a, mdtraj.compute_dihedrals(chunk, dihed_indices_a)), axis=0)
        diheds_b = numpy.concatenate((diheds_b, mdtraj.compute_dihedrals(chunk, dihed_indices_b)), axis=0)
        diheds_c = numpy.concatenate((diheds_c, mdtraj.compute_dihedrals(chunk, dihed_indices_c)), axis=0)

    def periodic_diheds(dh):
        def periodic_impl(dh, baseval):
            diff = dh[:, :] - baseval[numpy.newaxis, :]
            dh[:, :] -= (2 * math.pi) * numpy.round(diff / (2 * math.pi))
        periodic_impl(dh, dh[0, :])
        xmean = numpy.mean(dh, axis=0)
        xmean -= (2 * math.pi) * numpy.round(xmean / (2 * math.pi))
        periodic_impl(dh, xmean)

    periodic_diheds(diheds_a)
    periodic_diheds(diheds_b)
    periodic_diheds(diheds_c)

    total_weights = (
        distance_weight * numpy.var(dists, axis=0)
        + angle_weight * (numpy.var(angles_a, axis=0) + numpy.var(angles_b, axis=0))
        + dihedral_weight * (numpy.var(diheds_a, axis=0) + numpy.var(diheds_b, axis=0)
                             + numpy.var(diheds_c, axis=0)))
    avgangle_a_all = numpy.mean(angles_a, axis=0)
    avgangle_b_all = numpy.mean(angles_b, axis=0)
    banned_a = numpy.logical_or(avgangle_a_all < (math.pi * 45.0 / 180.0),
                                avgangle_a_all > (math.pi * 135.0 / 180.0))
    banned_b = numpy.logical_or(avgangle_b_all < (math.pi * 45.0 / 180.0),
                                avgangle_b_all > (math.pi * 135.0 / 180.0))
    banned = numpy.logical_or(banned_a, banned_b)
    print(f"Removed {numpy.sum(banned)} tuples due to bad angles")
    total_weights[banned] = 1e+6

    bestcand = int(numpy.argmin(total_weights))
    avgdist = float(numpy.mean(dists[:, bestcand]))
    avgangle_a = float(numpy.mean(angles_a[:, bestcand]))
    avgangle_b = float(numpy.mean(angles_b[:, bestcand]))
    avgdihed_a = float(numpy.mean(diheds_a[:, bestcand]))
    avgdihed_b = float(numpy.mean(diheds_b[:, bestcand]))
    avgdihed_c = float(numpy.mean(diheds_c[:, bestcand]))
    bestanclig = anclig_list[bestcand]

    with open(output, "w") as ofh:
        print("# ancA ancB ancC ligA ligB ligC", file=ofh)
        print(bestanclig[0][0], bestanclig[0][1], bestanclig[0][2],
              bestanclig[1][0], bestanclig[1][1], bestanclig[1][2], file=ofh)
        print("#",
              mdtop.atom(bestanclig[0][0]), mdtop.atom(bestanclig[0][1]),
              mdtop.atom(bestanclig[0][2]), mdtop.atom(bestanclig[1][0]),
              mdtop.atom(bestanclig[1][1]), mdtop.atom(bestanclig[1][2]), file=ofh)
        print("# avg dist / angle-a,b / dihed a,b,c (nm or radian)", file=ofh)
        print(avgdist, avgangle_a, avgangle_b, avgdihed_a, avgdihed_b, avgdihed_c, file=ofh)
        print('#', float(numpy.std(dists[:, bestcand])), float(numpy.std(angles_a[:, bestcand])),
              float(numpy.std(angles_b[:, bestcand])), float(numpy.std(diheds_a[:, bestcand])),
              float(numpy.std(diheds_b[:, bestcand])), float(numpy.std(diheds_c[:, bestcand])),
              file=ofh)


# ---------------------------------------------------------------------------
# Restraint file generation (port of generate_restr.py)
# ---------------------------------------------------------------------------

def _anchor_letter(i: int) -> str:
    return chr(ord("A") + i)


def _load_restraint_params(restrinfo, distance_spring, angle_spring,
                           dihedral_spring) -> tuple[list[int], list[float]]:
    anchors, avgs = read_restrinfo(restrinfo)
    for i in range(1, 6):
        avgs[i] = avgs[i] * 180.0 / math.pi  # pull/itp formats use degrees
        avgs[i] -= 360.0 * round(avgs[i] / 360.0)  # normalize to [-180, 180]
    return anchors, avgs


def generate_pull_restraint(*, restrinfo, mdp=None, ndx=None, decouple_B=False,
                            distance_spring: float = SPRING_DISTANCE,
                            angle_spring: float = SPRING_ANGLE,
                            dihedral_spring: float = SPRING_DIHEDRAL) -> None:
    """Write the pull-code mdp and the anchor index file (v1 mdp+ndx mode)."""
    if mdp is None or ndx is None:
        raise PipelineError("generate_pull_restraint requires both mdp and ndx paths")
    anchor_atoms, avgs = _load_restraint_params(restrinfo, distance_spring,
                                                angle_spring, dihedral_spring)
    with open(ndx, "w") as ofh:
        print("".join(["[ anchor%s ]\n%d\n" % (_anchor_letter(i), a)
                       for (i, a) in enumerate(anchor_atoms)]), file=ofh)
    with open(mdp, "w") as ofh:
        print("""
            pull = yes
            pull-nstxout = 0
            pull-nstfout = 0
            pull-ngroups = 6 ; A-F
            pull-ncoords = 6 ; 1 bond, 2 angles, 3 dihedrals
            pull-pbc-ref-prev-step-com = yes
            """, file=ofh)
        for i in range(6):
            print("pull-group%d-name    = anchor%s" % (i + 1, _anchor_letter(i)), file=ofh)
            print("pull-group%d-pbcatom = %d" % (i + 1, anchor_atoms[i]), file=ofh)

        dref = {
            "distref_23": avgs[0],
            "distK_23_A": distance_spring,
            "distK_23_B": distance_spring,
            "angleref_123": avgs[1],
            "angleK_123_A": angle_spring,
            "angleK_123_B": angle_spring,
            "angleref_234": avgs[2],
            "angleK_234_A": angle_spring,
            "angleK_234_B": angle_spring,
            "dihedralref_0123": avgs[3],
            "dihedralK_0123_A": dihedral_spring,
            "dihedralK_0123_B": dihedral_spring,
            "dihedralref_1234": avgs[4],
            "dihedralK_1234_A": dihedral_spring,
            "dihedralK_1234_B": dihedral_spring,
            "dihedralref_2345": avgs[5],
            "dihedralK_2345_A": dihedral_spring,
            "dihedralK_2345_B": dihedral_spring,
        }
        if decouple_B:
            dref["distK_23_B"] = 0.0
            dref["angleK_123_B"] = 0.0
            dref["angleK_234_B"] = 0.0
            dref["dihedralK_0123_B"] = 0.0
            dref["dihedralK_1234_B"] = 0.0
            dref["dihedralK_2345_B"] = 0.0

        print("""
            pull-coord1-type = umbrella
            pull-coord1-geometry = distance
            pull-coord1-groups = 3 4
            pull-coord1-dim = Y Y Y
            pull-coord1-k = {distK_23_A}
            pull-coord1-kB = {distK_23_B}
            pull-coord1-init = {distref_23}

            pull-coord2-type = umbrella
            pull-coord2-geometry = angle
            pull-coord2-groups = 3 2 3 4
            pull-coord2-dim = Y Y Y
            pull-coord2-k = {angleK_123_A}
            pull-coord2-kB = {angleK_123_B}
            pull-coord2-init = {angleref_123}

            pull-coord3-type = umbrella
            pull-coord3-geometry = angle
            pull-coord3-groups = 4 3 4 5
            pull-coord3-dim = Y Y Y
            pull-coord3-k = {angleK_234_A}
            pull-coord3-kB = {angleK_234_B}
            pull-coord3-init = {angleref_234}

            pull-coord4-type = umbrella
            pull-coord4-geometry = dihedral
            pull-coord4-groups = 1 2 2 3 3 4
            pull-coord4-dim = Y Y Y
            pull-coord4-k = {dihedralK_0123_A}
            pull-coord4-kB = {dihedralK_0123_B}
            pull-coord4-init = {dihedralref_0123}

            pull-coord5-type = umbrella
            pull-coord5-geometry = dihedral
            pull-coord5-groups = 2 3 3 4 4 5
            pull-coord5-dim = Y Y Y
            pull-coord5-k = {dihedralK_1234_A}
            pull-coord5-kB = {dihedralK_1234_B}
            pull-coord5-init = {dihedralref_1234}

            pull-coord6-type = umbrella
            pull-coord6-geometry = dihedral
            pull-coord6-groups = 3 4 4 5 5 6
            pull-coord6-dim = Y Y Y
            pull-coord6-k = {dihedralK_2345_A}
            pull-coord6-kB = {dihedralK_2345_B}
            pull-coord6-init = {dihedralref_2345}

            """.format(**dref), file=ofh)


def generate_restraint_itp(*, restrinfo, itp, decouple_B=False,
                           distance_spring: float = SPRING_DISTANCE,
                           angle_spring: float = SPRING_ANGLE,
                           dihedral_spring: float = SPRING_DIHEDRAL) -> None:
    """Write the restraint as [ intermolecular_interactions ] (v1 itp mode)."""
    anchor_atoms, avgs = _load_restraint_params(restrinfo, distance_spring,
                                                angle_spring, dihedral_spring)
    a = anchor_atoms
    with open(itp, "w") as ofh:
        print("[ intermolecular_interactions ]", file=ofh)
        print("[ bonds ]", file=ofh)
        if decouple_B:
            print("%d %d 6 %.3f %.3f %.3f 0.000"
                  % (a[2], a[3], avgs[0], distance_spring, avgs[0]), file=ofh)
        else:
            print("%d %d 6 %.3f %.3f" % (a[2], a[3], avgs[0], distance_spring), file=ofh)
        print(file=ofh)
        print("[ angles ]", file=ofh)
        if decouple_B:
            print("%d %d %d 1 %.3f %.3f %.3f 0.000"
                  % (a[1], a[2], a[3], avgs[1], angle_spring, avgs[1]), file=ofh)
            print("%d %d %d 1 %.3f %.3f %.3f 0.000"
                  % (a[2], a[3], a[4], avgs[2], angle_spring, avgs[2]), file=ofh)
        else:
            print("%d %d %d 1 %.3f %.3f" % (a[1], a[2], a[3], avgs[1], angle_spring), file=ofh)
            print("%d %d %d 1 %.3f %.3f" % (a[2], a[3], a[4], avgs[2], angle_spring), file=ofh)
        print(file=ofh)
        print("[ dihedrals ]", file=ofh)
        if decouple_B:
            for (i0, i1, i2, i3), avg in (((a[0], a[1], a[2], a[3]), avgs[3]),
                                          ((a[1], a[2], a[3], a[4]), avgs[4]),
                                          ((a[2], a[3], a[4], a[5]), avgs[5])):
                print("%d %d %d %d 2 %.3f %.3f %.3f 0.000"
                      % (i0, i1, i2, i3, avg, dihedral_spring, avg), file=ofh)
        else:
            for (i0, i1, i2, i3), avg in (((a[0], a[1], a[2], a[3]), avgs[3]),
                                          ((a[1], a[2], a[3], a[4]), avgs[4]),
                                          ((a[2], a[3], a[4], a[5]), avgs[5])):
                print("%d %d %d %d 2 %.3f %.3f"
                      % (i0, i1, i2, i3, avg, dihedral_spring), file=ofh)
        print(file=ofh)
