# FEP/REST2 v2: refactored relative-FEP pipeline (RBFE + mutation FEP)

This is a ground-up reorganization of the [v1 FEP/REST pipeline](../feprest/README.md).
The **scientific algorithm is unchanged** — the same REST2-enhanced
Hamiltonian replica exchange over a single/dual-topology FEP, evaluated
with the HREX-patched GROMACS 2020 — but the implementation is
restructured for maintainability.  The old `feprest/` pipeline stays in
place untouched; both can coexist.

One engine serves both relative-FEP applications of FEP-suite:

* **FEP (mutation stability)** — ΔΔG of folding upon a point mutation,
  prepared with `feprest/tools/prep_mutation_fep.py` (FASPR) exactly as
  before;
* **RBFE (relative binding)** — ΔΔG of binding upon transforming one
  ligand into the other, prepared with `feprest/fepgen` exactly as
  before.

The two use cases differ only in the offline system preparation (both
produce the same `fepbase.pdb` / `fepbase.top` mixed-topology inputs);
the run pipeline, the REST2 machinery and the BAR analysis are shared,
so refactoring the pipeline once delivers both workflows.

## What changed compared to v1

| Aspect | v1 (`feprest/`) | v2 (`feprest_v2/`) |
|---|---|---|
| Orchestration | 395-line zsh `pipeline.zsh`, numbered stages 1-8/999 | Python package `restlab`, **named stages** |
| Stage selection | `./run.zsh mol 3 4` (numbers) | `./run.zsh mol rest2-setup tune` (names) |
| Job systems | `submit_scripts/*.zsh` | **unchanged** — `pipeline.zsh` + `jobbridge.zsh` keep the same controller protocol |
| Configuration | `para_conf.zsh` sourced by zsh | same file format, parsed by Python (values must be literal; `$BASECONF`-style expansions are rejected with a clear error) |
| Helper scripts | 8 scattered argparse scripts | `restlab` modules called in-process (v1 spawned a `python3 rest2py.py …` subprocess per replica per tuning cycle) |
| grompp/convert-tpr of replicas | sequential unless `RUN_TPR_PARALLEL=yes` was exported | **concurrent by default** (they dominate the stage wall time); `RUN_TPR_PARALLEL=no` restores sequential |
| Re-running production chunks | stage numbers `8`, `9`, `10`, … | re-run the `prodrun` stage; each invocation extends by `SIMLENGTH` ps |

Other behavioral notes (documented in the code):

* the domain-decomposition check of the mdrun NP-search retry now
  inspects the log of *each* replica directory; v1 tailed
  `$log_basename.log` relative to the run root even for multidir runs
  (an unused loop variable), so the check read a nonexistent path;
* the tuning blend step is clamped to 1 exactly as in v1
  (`NINITIALTUNE` equals `NTUNE`), i.e. every tuning cycle blends the
  optimized and previous replica coordinates 50/50 — the formula is
  kept, now with a comment;
* unknown replica modes fail explicitly in `replica init` (v1
  silently produced a `NameError` later);
* `replica_states`-parsing fails with a diagnostic when the HREX log
  has no `Repl  average probabilities:` section;
* the `checkpoint_7`/`checkpoint_8`… directories of v1 become
  `checkpoint_init` / `checkpoint_ph<N>` (they were named after stage
  numbers);
* latent format-string bugs of `rest2py.py` / `canonicalize_top.py`
  ("Could not find dihedral for …" raising `TypeError` instead of the
  message) are fixed.

## Requirements

* the HREX-patched GROMACS 2020 (see
  [feprest/README.md](../feprest/README.md), "Preparation 4") — the
  patch, `fepgen` and the input-preparation tools are **not duplicated**
  and keep living under `feprest/`
* Python 3 with `numpy`, `mdtraj` and `pymbar` 3.0.3
* zsh (the controller/job layer of FEP-suite)

## Usage

The workflow is the same as v1, with named stages instead of numbers:

1. Prepare the mixed-topology input directories (`wt_L99A`,
   `wt_L99A_ref`, …) with `feprest/tools/prep_mutation_fep.py`
   (mutation) or `feprest/fepgen` (ligand transformation) — see the
   [v1 README](../feprest/README.md); the preparation side is unchanged.
2. Copy `feprest_v2/rundir_template/*` into the calculation directory
   (next to the input directories).
3. Edit `run.zsh` (`FEPSUITE_ROOT`, `GROMACS_DIR` (the patched 2020),
   `JOBTYPE=feprest_v2`, `JOBSYSTEM`) and `para_conf.zsh`.  If you carry
   over a v1 `para_conf.zsh`, replace `REFINIT=$BASECONF` with the
   literal filename (`REFINIT=conf_ionized.pdb`).
4. Run stages:

```sh
cd feprest-calc
./run.zsh wt_L99A all                        # everything, in order
./run.zsh wt_L99A minimize nvt               # or selected stages
./run.zsh wt_L99A prodrun                    # extend the sampling by SIMLENGTH ps
```

Result: `wt_L99A/bar1.log` (and `bar/results-*.pickle`), in exactly the
v1 format — subtract the reference-state log to obtain the mutation /
transformation free energy, as in v1.

### Stages

| stage | replaces | what it does |
|---|---|---|
| `minimize` | 1 | state-A steep/cg minimization (FEP off, `cginit.mdp`) |
| `nvt` | 2 | state-A NVT with hydrogen-massed topology (`turn-heavy`), emits the preprocessed `fep_pp.top` |
| `npt` | 3 | state-A NPT equilibration (10 ns) |
| `rest2-setup` | 4 | hot-region underline (`add_underline`), charge neutralization, `for_rest.ndx`, per-replica REST2 topologies/mdps, per-replica minimization |
| `tune` | 5 | `NTUNE` replica-exchange tuning cycles (replex + ladder optimization) |
| `npt-replex` | 6 | multi-replica NPT |
| `prodrun-init` | 7 | first production chunk with `deltae` sampling + checkpoints |
| `prodrun` | 8+ | one `SIMLENGTH` ps extension chunk + BAR analysis; **re-run to extend** |
| `trajectory` | 999 | centered state-A/B trajectories (`trjconv`; auxiliary, not part of `all`) |

Every mdrun goes through the job-system MPI wrapper with the NP-search
retry loop (v1 `mdrun_find_possible_np`, including the `-ntomp 1` nudge
and the optional `NSTLIST_CMD` appendix), and the BAR analysis failure
stays non-fatal with v1's message so a chunk can simply be extended.

### Configuration (`para_conf.zsh`)

Same keys and defaults as v1: `BASECONF` / `BASETOP`, `CHARGE`,
`FF`, `NREP`, `PARA`, `TPP`, `NTUNE`, `SIMLENGTH`, `REPLICA_INTERVAL`,
`SAMPLING_INTERVAL`, `BASEWARN`, `REFINIT`, `REFCRD`, `DOMAIN_SHRINK`,
`REST2_REGION_DISTANCE`, `REST2_TEMP`.

Differences:

* values must be literal numbers/strings — shell expansions are
  rejected with a clear error (the same file is parsed by both zsh and
  Python);
* new optional keys (defaults match v1's behavior): `RUN_TPR_PARALLEL`
  (default `yes`), `NON_PERTURBED_MOLECULETYPE`, `TARGET_MOLECULE`,
  `WATER_MOLTYPE`, `REST2_TEMP0` (default 300), `EXTEND_RUN_LENGTH`,
  `REPOPT_REPLEX_INTERVAL`, `NSTLIST_CMD`;
* `NTUNE` must be >= 1 (v1 broke unintuitively at 0).

## Layout

```
feprest_v2/
├── README.md            this file
├── pipeline.zsh         thin adapter for controller.zsh (query/run protocol)
├── jobbridge.zsh        lets Python call job_singlerun / job_mpirun
├── itp_addenda/         tip3p itps linked into run directories (stage 4)
├── water_ion_models/    water/ion hybrid models for neutralize/recover-water
├── rundir_template/     run.zsh, para_conf.zsh, mdp/ (copy to your calc dir)
├── restlab/             the pipeline (Python)
│   ├── cli.py           entry point (query/run)
│   ├── phases.py        stage table + controller protocol lines
│   ├── stages.py        the nine stages (port of v1 pipeline.zsh)
│   ├── runner.py        GROMACS execution layer + NP-search retry
│   ├── config.py        para_conf.zsh parser + typed config
│   ├── mdp.py           mdp assembly (sed-pipeline ports), lambda rendering
│   ├── hotregion.py     REST2 hot region: add_underline + underlined_group
│   ├── heavyhydrogen.py turn-heavy
│   ├── waterion.py      recover-water (flexible water restoration)
│   ├── neutralize.py    water⇄ion charge neutralization (topology + gro)
│   ├── analysis_index.py  trjconv centering index
│   ├── bar.py           BAR analysis of deltae.xvg (pymbar)
│   └── rest2/
│       ├── scaling.py   REST2 topology scaler (rest2py.py)
│       ├── replica.py   replica ladder state + replex-probability optimizer
│       └── canonical.py topology canonicalizer (utility)
└── tests/               unit tests (no GROMACS needed)
```

The job-system layer is intentionally kept in zsh: `pipeline.zsh`
forwards the controller protocol and exports the job context;
`jobbridge.zsh` sources `submit_scripts/$JOBSYSTEM.zsh` and calls its
`job_singlerun` / `job_mpirun`, so every existing cluster definition
(kudpc, tsukuba-pegasus, …) works without modification.

## Tests

```sh
cd feprest_v2
python3 -m pytest tests/ -q
```

68 tests cover the config parser, the stage table and controller
protocol lines, the mdp transformations against the real templates, the
hot-region detection (mdtraj) and index generation, the hydrogen-mass
and flexible-water topology edits, the water⇄ion neutralization
(topology and coordinate passes), the REST2 topology scaling, the
replica-ladder optimization math (including the exchange-probability
log format of the HREX patch), the BAR pipeline orchestration, and the
mdrun log/error parsing.

## Provenance

The v2 pipeline was rewritten (2026) by
[Cedrus810](https://github.com/Cedrus810) on top of the scientific
algorithms of FEP-suite v1 by Shun Sakuraba (QST). As a derivative work of
GPL-3.0-or-later code it is distributed under the same license (see
[COPYING](../COPYING)): the v2 implementation is © 2026 Cedrus810, while
the v1 pipeline ([../feprest](../feprest)) remains © Shun Sakuraba.

## References

The algorithm and its references are identical to v1; see
[feprest/README.md](../feprest/README.md).
