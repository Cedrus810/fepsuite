"""Topology canonicalization — library port of
``feprest/rest2py/canonicalize_top.py``.

Copyright 2018-2023 Shun Sakuraba, BSD 3-clause (see the header of the
original file).  Removes ``[ bondtypes ]`` / ``[ angletypes ]`` /
``[ dihedraltypes ]`` from a preprocessed topology by looking up each
``[ bonds ]`` / ``[ angles ]`` / ``[ dihedrals ]`` entry's parameters and
writing them out explicitly.  Utility for topology debugging; not used
by the pipeline stages themselves.
"""

from __future__ import annotations

import copy

from .scaling import find_matching_dihedral, next_permutation  # noqa: F401 (re-export)


def find_matching_bond(bondtype_params, ai, aj, fun):
    for key in [(ai, aj, fun), (aj, ai, fun)]:
        if key in bondtype_params:
            return bondtype_params[key]
    raise RuntimeError("Could not find bond for %s-%s-%d" % (ai, aj, fun))


def find_matching_angle(angletype_params, ai, aj, ak, fun):
    for key in [(ai, aj, ak, fun), (ak, aj, ai, fun)]:
        if key in angletype_params:
            return angletype_params[key]
    raise RuntimeError("Could not find angle for %s-%s-%d" % (ai, aj, ak, fun))


# dihedral function -> (#non-fep params, #fep params, integer indices)
_DIHED_TABLE = {
    1: (3, 2, [2]),
    2: (2, 2, []),
    3: (6, 6, []),
    4: (3, 2, [2]),
    5: (4, 4, []),
    8: (2, 1, [0]),
    9: (3, 2, [2]),
    10: (1, 0, []),
    11: (4, 0, []),
}

# the same for [ angles ]
_ANGLE_TABLE = {
    1: (2, 2, []),
    2: (2, 2, []),
    3: (3, 0, []),
    4: (4, 0, []),
    5: (4, 4, []),
    6: (6, 0, []),
    8: (2, 1, []),
    10: (2, 0, []),
}

# the same for [ bonds ]
_BOND_TABLE = {
    1: (2, 2, []),
    2: (2, 2, []),
    3: (2, 2, []),
    4: (3, 0, []),
    5: (0, 0, []),
    6: (2, 2, []),
    7: (2, 0, []),
    8: (2, 1, [0]),
    9: (2, 1, [0]),
    10: (4, 4, []),
}


def canonicalize_topology(topology_path, output_path,
                          ignore_noninteger_periodicity: bool = False) -> None:
    with open(topology_path) as fh, open(output_path, "w") as ofh:
        bondtype_of_atomtype = {}  # confusing but this is atom->bondatom mapping
        atomtype_info = {}
        dummy_atomtypes = set()
        dihtype = {}
        angtype = {}
        bondtype_params = {}
        molecule = None
        sectiontype = None
        residues = None
        atomnames = None

        def print_parameters(params, n_nonfep, n_fep, intind, atoms, funcno):
            if len(params) not in [n_nonfep, n_nonfep + n_fep]:
                directive = [None, None, "bonds", "angles", "dihedrals"][len(atoms)]
                raise RuntimeError("Number of args in %s: expected %d or %d, but was %d (%s:%d)" %
                                   (directive,
                                    n_nonfep, n_nonfep + n_fep, len(params),
                                    "-".join([str(x) for x in atoms]), funcno))
            for (i, v) in enumerate(params):
                if i in intind:
                    try:
                        v = int(v)
                    except ValueError:
                        print(params)
                        if ignore_noninteger_periodicity:
                            vf = float(v)
                            vi = int(vf)
                            if abs(vi - vf) > 1e-2:
                                raise RuntimeError("Periodicity should be integer but was %f" % vf)
                            v = vi
                        else:
                            raise RuntimeError("Periodicity should be integer")
                    fmts = "%1d" % v
                else:
                    v = float(v)
                    fmts = "%16.8e" % v
                ofh.write(" %s" % fmts)
            ofh.write("\n")

        for lraw in fh:
            ltmp = lraw.split(';', 1)
            if len(ltmp) == 1:
                l = ltmp[0]
                comment = ""
            else:
                l = ltmp[0]
                comment = ";" + ltmp[1]
            l = l.strip()
            ls = l.split()
            if l.startswith('#'):
                raise RuntimeError("The topology file is not preprocessed")
            if l.startswith('['):
                sectiontype = ls[1]
                if sectiontype not in ["bondtypes", "angletypes", "dihedraltypes"]:
                    ofh.write(lraw)
                continue

            # blank line
            if len(ls) == 0:
                ofh.write(lraw)
                continue

            if sectiontype is None:
                pass
            elif sectiontype == 'defaults':
                pass
            elif sectiontype == 'atomtypes':
                if len(ls) < 6:
                    raise RuntimeError("Atomtype contains < 6 fields")
                # here everything is mess but toppush.cpp is actually super mess
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
                    # After GROMACS resolved issue 4120, the logic changed:
                    # "Attempt parsing field 1 to integer. If conversion fails,
                    # we do not have an atomic number but a bonded type."
                    # int() in Python is more permissive (e.g. "3_10" is 310),
                    # so this part is more restrictive.
                    have_atomic_number = all((c.isdigit() for c in ls[1]))
                    have_bonded_type = not have_atomic_number

                atomtype = ls[0]
                (mass, charge, particle, sigc6, epsc12) = \
                    ls[1 + int(have_bonded_type) + int(have_atomic_number):]

                if have_bonded_type:
                    bondtype = ls[1]
                    if all((c.isdigit() for c in ls[1])):
                        raise RuntimeError(
                            f"""[ atomtypes ] contains bondtype "{atomtype}", but the bondtype for this atomtype consists of only digits.
This is considered invalid atomtype in GROMACS.""")
                else:
                    bondtype = atomtype
                if have_atomic_number:
                    atomic_ix = 1 + int(have_bonded_type)
                    atomic_number = int(ls[atomic_ix])
                else:
                    atomic_number = 0  # ??

                # store this because we use in [ atoms ] section
                bondtype_of_atomtype[atomtype] = bondtype

                mass = float(mass)
                charge = float(charge)
                sigc6 = float(sigc6)
                epsc12 = float(epsc12)
                is_dummy = epsc12 == 0.

                atomtype_info[atomtype] = (charge, mass)
                if is_dummy:
                    dummy_atomtypes.add(atomtype)

            elif sectiontype == 'dihedraltypes':
                (ai, aj, ak, al) = ls[0:4]
                dihfun = int(ls[4])
                values = ls[5:]
                key = (ai, aj, ak, al, dihfun)
                if dihfun == 9:
                    # allows multiple dihedraltype for fn = 9
                    if key not in dihtype:
                        dihtype[key] = []
                    dihtype[key].append(values)
                else:
                    if key in dihtype:
                        for (i, e) in enumerate(dihtype[key]):
                            d = abs(float(values[i]) - float(e))
                            if d > 1e-20:
                                raise RuntimeError("Multiple dihedral for dihfun = %d, %s-%s-%s-%s"
                                                   % (dihfun, ai, aj, ak, al))
                    else:
                        dihtype[key] = values
                continue  # suppress printing, we won't use dihedraltypes.
            elif sectiontype == 'angletypes':
                (ai, aj, ak) = ls[0:3]
                anglefun = int(ls[3])
                values = ls[4:]
                key = (ai, aj, ak, anglefun)
                angtype[key] = values
                continue  # suppress printing
            elif sectiontype == 'bondtypes':
                (ai, aj) = ls[0:2]
                bondfun = int(ls[2])
                values = ls[3:]
                key = (ai, aj, bondfun)
                bondtype_params[key] = values
                continue  # suppress printing
            elif sectiontype == 'moleculetype':
                molecule = ls[0]
                # These None are sentinels for 1-origin access
                bondtype_list = [None]
                scaled = [None]
                atomnames = []
                residues = []
            elif sectiontype == 'atoms':
                aindex = int(ls[0])
                atomtype = ls[1]
                bondtype_list.append(bondtype_of_atomtype[atomtype])

                # charge & mass is optional parameters, oof...
                charge, mass = atomtype_info[atomtype]
                if len(ls) > 6:
                    charge = float(ls[6])
                if len(ls) > 7:
                    mass = float(ls[7])

                if len(ls) > 8:
                    atomtypeB = ls[8]
                    (chargeB, massB) = atomtype_info[atomtypeB]
                    fep = True
                else:
                    fep = False
                    chargeB = 0.
                    massB = 0.
                    atomtypeB = None
                if len(ls) > 9:
                    chargeB = float(ls[9])
                if len(ls) > 10:
                    massB = float(ls[10])

                _resnr = ls[2]
                resid = ls[3]
                residues.append(resid)
                atomname = ls[4]
                atomnames.append(atomname)

                ofh.write("%5d %4s %4s %4s %4s %5s %16.8e %16.8e" % (aindex, atomtype, ls[2], ls[3], ls[4], ls[5], charge, mass))
                if fep:
                    ofh.write(" %4s %16.8e %16.8e%s\n" % (atomtypeB, chargeB, massB, comment.rstrip()))
                else:
                    ofh.write(" %s\n" % comment.rstrip())
                continue
            elif sectiontype == 'dihedrals':
                dihfun = int(ls[4])
                (ai, aj, ak, al) = [int(x) for x in ls[0:4]]
                if len(ls) == 5:
                    # must load dihedral table
                    (ti, tj, tk, tl) = [bondtype_list[x] for x in [ai, aj, ak, al]]
                    ofh.write("; parameters for %s-%s-%s-%s, fn=%d\n" % (ti, tj, tk, tl, dihfun))

                    params_tmp = find_matching_dihedral(dihtype, ti, tj, tk, tl, dihfun)
                    matched = True
                else:
                    params_tmp = ls[5:]
                    matched = False
                if dihfun == 9 and matched:
                    params_list = params_tmp
                else:
                    params_list = [params_tmp]

                for params in params_list:
                    ofh.write("%5d %5d %5d %5d %2d" % (ai, aj, ak, al, dihfun))
                    (n_nonfep, n_fep, intind) = _DIHED_TABLE[dihfun]
                    print_parameters(params, n_nonfep, n_fep, intind,
                                     [ai, aj, ak, al], dihfun)
                continue
            elif sectiontype == 'angles':
                angfun = int(ls[3])
                (ai, aj, ak) = [int(x) for x in ls[0:3]]
                if len(ls) == 4:
                    # must load angle table
                    (ti, tj, tk) = [bondtype_list[x] for x in [ai, aj, ak]]
                    ofh.write("; parameters for %s-%s-%s, fn=%d\n" % (ti, tj, tk, angfun))

                    params = find_matching_angle(angtype, ti, tj, tk, angfun)
                    matched = True
                else:
                    params = ls[4:]
                    matched = False
                ofh.write("%5d %5d %5d %2d" % (ai, aj, ak, angfun))
                (n_nonfep, n_fep, intind) = _ANGLE_TABLE[angfun]
                print_parameters(params, n_nonfep, n_fep, intind,
                                 [ai, aj, ak], angfun)
                continue

            elif sectiontype == 'bonds':
                bondfun = int(ls[2])
                (ai, aj) = [int(x) for x in ls[0:2]]
                if len(ls) == 3:
                    # must load bond table
                    (ti, tj) = [bondtype_list[x] for x in [ai, aj]]
                    ofh.write("; parameters for %s-%s, fn=%d\n" % (ti, tj, bondfun))

                    params = find_matching_bond(bondtype_params, ti, tj, bondfun)
                    matched = True
                else:
                    params = ls[3:]
                    matched = False
                ofh.write("%5d %5d %2d" % (ai, aj, bondfun))
                (n_nonfep, n_fep, intind) = _BOND_TABLE[bondfun]
                print_parameters(params, n_nonfep, n_fep, intind,
                                 [ai, aj], bondfun)
                continue

            # With a few exceptions, we just print as-is
            ofh.write(lraw)
