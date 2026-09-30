# FEP-ABFE v2: refactored absolute binding free energy pipeline

This is a ground-up reorganization of the [v1 ABFE pipeline](../abfe/README.md).
The **scientific algorithm is unchanged** — the same Aldeghi et al. (2016)
thermodynamic cycle with Boresch restraints, evaluated with GROMACS — but the
implementation is restructured for maintainability. The old `abfe/` pipeline
stays in place untouched; both can coexist.

## What changed compared to v1

| Aspect | v1 (`abfe/`) | v2 (`abfe_v2/`) |
|---|---|---|
| Orchestration | 648-line zsh `pipeline.zsh`, 12 numbered stages | Python package `feplab`, **named stages** |
| Stage selection | `./run.zsh mol1 all` / `3 4 5` (numbers) | `./run.zsh mol1 all` / `setup equilibrate ...` (names) |
| Job systems | `submit_scripts/*.zsh` | **unchanged** — `pipeline.zsh` + `jobbridge.zsh` keep the same controller protocol |
| Configuration | `para_conf.zsh` | same file format, parsed by Python (same keys, see below) |
| Lambda optimization | buried in `generate_decoupling.py` | `feplab/lambdas.py` |
| Charge correction | wired to scrape the generator's stdout with regexes (and the call was commented out — charged ligands crashed stage 12) | in-process library call, structured JSON results, **fixed** |
| MDP assembly | `cat` + `sed` pipelines | `feplab/mdp.py` (unit-tested) |
| Dead code | `generate_warpdrive.py`, `abfe/template/`, `index.py`, `MAX_BONDED_INTERACTION_DIST` | removed |

Other behavioral fixes (documented in the code):

* **LRC time alignment**: the original EDR and the PME re-evaluation EDR
  generally have different time grids (the production run writes energies
  every `nstenergy` of its own clock, a `mdrun -rerun` writes at the
  trajectory frame times, whose timestamps pass through float32 in the
  xtc).  v1 paired the frames by position and required exact timestamp
  equality; v2 aligns frames by nearest timestamp with a 0.002 ps
  tolerance and fails with a diagnostic when the grids share no frames;
* **LRC sanity guard and term matching**: energy terms are matched by
  normalized names (immune to GROMACS renames like `LJ (SR)` →
  `LJ-(SR)`), the per-term perturbation decomposition is printed to the
  log, and any LRC term beyond `LRC_SANITY_LIMIT` (default 50 kcal/mol)
  aborts the analysis instead of poisoning `result.txt`.  This fires on
  the GROMACS 2026.3 soft-core + PME-LJ breakage of the annihilation
  endpoint re-evaluations — see
  [docs/issue-lrc-gmx2026.md](docs/issue-lrc-gmx2026.md) for the full
  investigation; `SKIP_ANNIHILATION_LRC=yes` is the workaround
  (physically the decoupled endpoints have no dispersion to correct);
* the Boresch spring constants are defined once (`feplab/restraints.py`);
  v1 used inconsistent values (41.848 vs 41.84) between analysis and generation;
* the charge-correction ddG formulas consistently use the box length `a`
  (v1 mixed `max(a,b,c)` and `a`; identical for dodecahedron boxes);
* a single charge-correction sample no longer divides by zero;
* `gmx check` parsing and other shell scrapes are replaced by explicit,
  error-checked parsing.

## Resource auto-tuning (new in v2)

Two mechanisms reduce the mismatch between what the queueing system
allocates and what mdrun actually uses.  Both are on by default and can
be turned off in `para_conf.zsh`.

**`AUTO_RESOURCE=yes` (default)** — at every MPI mdrun launch inside the
job, the pipeline detects how many GPUs the allocation actually provided
(`qstat -F json -f $PBS_JOBID` → `PBS_GPUFILE` → `CUDA_VISIBLE_DEVICES`
→ `nvidia-smi -L`, first hit wins) and derives the rank layout:

```
ranks_per_replica = max(1, gpus // replicas)
NP                = replicas * ranks_per_replica
```

Fewer GPUs than replicas → every replica keeps one rank and shares the
GPUs; more GPUs than replicas → extra ranks per replica.  When nothing
can be detected (e.g. `JOBSYSTEM=none` on a login node) the static
`MULTI x *_PARA` layout from `para_conf.zsh` is used, so behavior
degrades to v1.

**`PREP_NONMPI=yes` (default)** — the pre-equilibration chain
(`steep`/`cg`/`nvt`/`npt` of both systems and the ligand diameter
sampling) runs on a single non-MPI rank with OMP threads instead of a
full MPI layout.  The corresponding stages (`setup`, `restraints`,
`prep-charging-lig`) then request only one rank.  `equilibrate` and the
production/LRC runs keep MPI.  `GMX_NOMPI` (default `gmx`) selects the
binary, so MPI-only GROMACS installs can point at a thread build.

Note: `PREP_NONMPI` / `AUTO_RESOURCE` are read both at submit time
(resource request) and at run time; per-`$ID` overrides of these two
keys are not seen at submit time, so set them in the base
`para_conf.zsh` only.  The long NPT `equilibrate` stage is deliberately
*not* prep-like — it is production-length and benefits from MPI.

## Large-ligand support

Ligands much larger than the cutoff need an explicit ``rlist`` because
``couple-intramol = no`` puts intra-ligand exclusions beyond the
pair-list cutoff (GROMACS cannot auto-estimate the buffer).  v2 layers
four defenses where v1 had two:

1. the ligand diameter is measured in the ``restraints`` stage (tail
   extrapolation of the max interatomic distance, written to
   ``diameter.txt``) and ``rlist = max(1.2*rvdw, safe_diameter)`` is set
   for every decoupling/eval mdp — as in v1;
2. the solvated ligand box is grown to ``max(WATER_THICKNESS,
   safe_diameter + 0.1)`` — as in v1;
3. **new**: when mdrun fails with the GROMACS pair-list cutoff error,
   the pipeline enlarges rlist (to 1.1x the largest distance the log
   reports in nm, or 1.2x the current value otherwise), rewrites and
   re-grompps **all** replica mdps of the phase, and retries — instead
   of failing the job.  It gives up (with a pointer to
   ``LIGAND_DIAMETER``) only when rlist would have to exceed 10 nm;
4. ``LIGAND_DIAMETER`` in ``para_conf.zsh`` still overrides everything,
   as in v1.

The auto-retry applies to both the replex product runs and the PME
re-evaluation runs.

## Requirements

* GROMACS >= 2022.5 (2022.4 and older do **not** work)
* Python 3.11+ with `numpy`, `mdtraj`, `pyedr` (`pip3 install cython mdtraj pyedr`)
* [APBS](https://server.poissonboltzmann.org) **only for charged ligands**
* zsh (the controller/job layer of FEP-suite)

## Usage

The workflow is the same as v1, with named stages instead of numbers:

1. Prepare a solvated, neutralized system: `topol_ionized.top` +
   `conf_ionized.pdb` (or `.gro`) under e.g. `abfecalc/mol1/`.
2. Copy `abfe_v2/rundir_template/*` to `abfecalc/`.
3. Edit `run.zsh` (`FEPSUITE_ROOT`, `GROMACS_DIR`, `JOBTYPE=abfe_v2`,
   `JOBSYSTEM`) and `para_conf.zsh`.
4. Run stages:

```sh
cd abfecalc
./run.zsh mol1 all                        # everything, in dependency order
./run.zsh mol1 setup equilibrate          # or selected stages
./run.zsh mol1 query                      # (v2 extra) list the named stages
```

Result: `abfecalc/mol1/result.txt`, in exactly the v1 format.

### Stages

| stage | replaces | what it does |
|---|---|---|
| `setup` | 1 | version gate, preprocessing, complex index, restrained min/eq chain |
| `equilibrate` | 2 | unrestrained NPT equilibration |
| `restraints` | 3 | PBC-fixed trajectory analysis, Boresch anchor search, ligand extraction/solvation |
| `prep-charging-lig` | 4 | min/eq chain of the ligand-only system |
| `charging-lig` | 5 | ligand charge-discharge replex run |
| `annihilation-lig` | 6 | ligand annihilation replex run (with lambda-optimization preruns) |
| `lrc-lig` | 7 | PME long-range re-evaluation for the ligand legs |
| `charging-complex` | 8 | complex charging replex run |
| `annihilation-complex` | 9 | complex annihilation replex run |
| `restraint-decouple` | 10 | Boresch restraint decoupling replex run |
| `lrc-complex` | 11 | PME long-range re-evaluation for the complex legs |
| `analysis` | 12 | BAR, LRC, charge correction, `result.txt` |

Stage dependencies (declared in `feplab/phases.py`, forwarded to
`controller.zsh` which maps them onto the queueing system's dependency
mechanism):

```
setup -> equilibrate -> restraints -+-> prep-charging-lig -> charging-lig -> annihilation-lig --+
                                    |                                       ^                   |
                                    |                                       |                   v
                                    |                                  (charging-lig)          lrc-lig
                                    +-> charging-complex -> annihilation-complex -----------------+
                                    |                          ^
                                    |                          |
                                    |                    (annihilation-lig)
                                    +-> restraint-decouple ----------------------------------> lrc-complex
                                    (all) --------------------------------------------------> analysis
```

### Configuration (`para_conf.zsh`)

Same keys and defaults as v1: `NRESTR` / `NCHARGE` / `NANNIH` (replica
counts), `LIG_PARA` / `COMPLEX_PARA` / `TPP` (parallelization), `RUN_PROD`,
`LIG_GMX`, `RECEPTOR_MDTRAJ`, `EQ_RMSD_CUTOFF`, `LIGAND_DIAMETER`,
`ANNIH_LAMBDA_OPT` / `ANNIH_LAMBDA_OPT_LENGTH`, `WATER_STRUCTURE`,
`WATER_THICKNESS`, `IONIC_STRENGTH`, `ION_POSITIVE` / `ION_NEGATIVE`,
`APBS`, `CHARGE_CORRECTION_NSAMP`.

Differences:

* `MAX_BONDED_INTERACTION_DIST` is gone (it was never referenced by v1);
* new optional `SOLVENT` (default `SOL`) — the solvent moleculetype used by
  `gmx genion` and the flexible-solvent switch;
* `APBS` is now an ordinary key (default `apbs`);
* `CHARGE_CORRECTION_NSAMP=0` explicitly disables the charge correction;
* values must be literal numbers/strings — shell expansions are rejected
  with a clear error (the same file is parsed by both zsh and Python, so
  dynamic syntax would be a trap).

## Layout

```
abfe_v2/
├── README.md            this file
├── pipeline.zsh         thin adapter for controller.zsh (query/run protocol)
├── jobbridge.zsh        lets Python call job_singlerun / job_mpirun
├── apbs_input.template  APBS input for the charge correction
├── template/            per-phase FEP mdp addenda ({lambdas_formatted}, ...)
├── rundir_template/     run.zsh, para_conf.zsh, mdp/ (copy to your calc dir)
├── feplab/              the pipeline (Python)
│   ├── cli.py           entry point (query/run)
│   ├── phases.py        stage DAG, phase table, thermodynamic-cycle bookkeeping
│   ├── stages.py        the twelve stages (port of v1 pipeline.zsh)
│   ├── runner.py        GROMACS execution layer + NP-search retry + sanity checks
│   ├── config.py        para_conf.zsh parser + typed config
│   ├── mdp.py           mdp read/merge/write, LRCONLY policy, rlist
│   ├── topology.py      topology / index parsing
│   ├── lambdas.py       schedules + replex-probability optimization
│   ├── restraints.py    Boresch anchor search, restraint generation, analytical term
│   ├── ligand.py        ligand extraction, diameter, index generation, flexible solvent
│   ├── lrc.py           long-range dispersion correction (pyedr)
│   ├── charge_correction.py  Rocklin et al. finite-size correction (APBS)
│   └── analysis.py      BAR/LRC aggregation and result.txt
└── tests/               unit tests (no GROMACS needed)
```

The job-system layer is intentionally kept in zsh: `pipeline.zsh` forwards
the controller protocol and exports the job context; `jobbridge.zsh` sources
`submit_scripts/$JOBSYSTEM.zsh` and calls its `job_singlerun` /
`job_mpirun`, so every existing cluster definition (kudpc, tsukuba-pegasus,
... ) works without modification.

## Tests

```sh
cd abfe_v2
python3 -m pytest tests/ -q
```

76 tests cover the config parser, mdp assembly (including the real
templates), topology/index parsing, lambda schedules and the replex-based
optimizer, Boresch restraint generation and the analytical correction, the
charge-correction geometry/chemistry parts, the BAR/LRC report math, and the
stage DAG. They run without GROMACS.

## Provenance

The v2 pipeline was rewritten (2026) by
[Cedrus810](https://github.com/Cedrus810) on top of the scientific
algorithms of FEP-suite v1 by Shun Sakuraba (QST). As a derivative work of
GPL-3.0-or-later code it is distributed under the same license (see
[COPYING](../COPYING)): the v2 implementation is © 2026 Cedrus810, while
the v1 pipeline ([../abfe](../abfe)) remains © Shun Sakuraba. The
GROMACS 2026.3 LRC investigation above is filed upstream as
[shunsakuraba/fepsuite#9](https://github.com/shunsakuraba/fepsuite/issues/9).

## References

The algorithm and its references are identical to v1; see
[abfe/README.md](../abfe/README.md) (Aldeghi et al. 2016, Boresch et al.
2003, Rocklin et al. 2013, Chen et al. 2018).
