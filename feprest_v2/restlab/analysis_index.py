"""Index file for the analysis trajectories — library port of
``feprest/make_ndx_trjconv_analysis.py``.

Produces a ``[ centering ]`` group (the atom closest to the
center of mass) and an ``[ output ]`` group (every atom) so that
``gmx trjconv -center`` can compact-PBC-center the state A/B
trajectories of the trajectory stage.
"""

from __future__ import annotations

import numpy


def write_centering_ndx(structure_path, output_path) -> None:
    import mdtraj

    structure = mdtraj.load(structure_path)

    com = mdtraj.compute_center_of_mass(structure)[0, :]

    displs = structure.xyz[0, :, :] - com[numpy.newaxis, :]
    dist2 = numpy.sum(displs[:, :] * displs[:, :], axis=1)
    closest = numpy.argmin(dist2)

    with open(output_path, "w") as ofh:
        print("[ centering ]", file=ofh)
        print("%d" % (closest + 1), file=ofh)
        print(file=ofh)

        print("[ output ]", file=ofh)
        for i in range(structure.n_atoms):
            print("%d" % (i + 1), file=ofh)
        print(file=ofh)
