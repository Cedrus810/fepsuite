````
    ________________                   _ __     
   / ____/ ____/ __ \      _______  __(_) /____ 
  / /_  / __/ / /_/ /_____/ ___/ / / / / __/ _ \
 / __/ / /___/ ____/_____(__  ) /_/ / / /_/  __/
/_/   /_____/_/         /____/\__,_/_/\__/\___/ (beta)
````

# FEP-suite v2 — a ground-up rewrite of the FEP-suite pipelines

This repository is a fork of
[shunsakuraba/fepsuite](https://github.com/shunsakuraba/fepsuite). The two
pipelines in `abfe_v2/` and `feprest_v2/` are a **complete rewrite by
[Cedrus810](https://github.com/Cedrus810) (2026)** of the original FEP-suite
v1 pipelines by Shun Sakuraba: the scientific algorithms are unchanged, but
the implementation is new — restructured, unit-tested, and hardened with the
fixes and safeguards below.

| pipeline | v1 (original) | v2 (this rewrite) |
|---|---|---|
| ABFE — absolute binding free energy (protein–ligand) | [abfe/](abfe/README.md) | [abfe_v2/](abfe_v2/README.md) |
| FEP/REST2 — relative FEP: ligand RBFE and mutation stability | [feprest/](feprest/README.md) | [feprest_v2/](feprest_v2/README.md) |

The v1 pipelines remain in place untouched, so both generations can coexist.
The v2 workflow, job-system layer (`controller.zsh` + `submit_scripts/`) and
result formats are the same as v1, so existing calculation directories and
cluster definitions keep working.

## Highlights of the rewrite

**Large-ligand safety.** Ligands larger than the nonbonded cutoff break
GROMACS' auto-estimated pair-list buffer under `couple-intramol = no`. v2
measures the ligand diameter during setup, grows the solvated ligand box
accordingly, and — when mdrun still fails with the pair-list cutoff error —
automatically enlarges `rlist`, rewrites and re-`grompp`s **all** replica mdps
of the phase and retries instead of failing the job (details:
[abfe_v2/README.md](abfe_v2/README.md#large-ligand-support)).

**The GROMACS 2026.3 LRC breakage is contained automatically.** With current
GROMACS, the PME long-range-dispersion re-evaluation of the annihilation
endpoints returns garbage `LJ recip.` terms that would silently poison
`result.txt` (reported upstream as
[issue #9](https://github.com/shunsakuraba/fepsuite/issues/9), full
investigation in
[abfe_v2/docs/issue-lrc-gmx2026.md](abfe_v2/docs/issue-lrc-gmx2026.md)). The
v2 analysis sanity-checks every LRC term (default limit 50 kcal/mol), aborts
with a diagnostic naming the offending term instead of writing bogus results,
and supports `SKIP_ANNIHILATION_LRC=yes` — the physically motivated
workaround, since decoupled endpoints have no dispersion left to correct.

**A maintainable, tested codebase.** The numbered zsh stages became named
stages of Python packages (`feplab`: 12 stages, `restlab`: 9 stages) with 144
unit tests that run without GROMACS; `para_conf.zsh` is parsed structurally
and shell-expansion traps are rejected with clear errors; the charge
correction is an in-process library call with structured JSON results (v1
scraped the generator's stdout with regexes, and charged ligands crashed the
final stage); and a series of v1 bugs is fixed — inconsistent Boresch spring
constants, LRC time-grid misalignment, the mdrun domain-decomposition retry
reading a wrong log path, `rest2py` format-string crashes, and more (see the
per-pipeline READMEs for the complete lists).

## Provenance & license

The `abfe_v2/` and `feprest_v2/` pipelines are a ground-up rewrite by
[Cedrus810](https://github.com/Cedrus810) (2026), implemented on top of the
scientific algorithms of FEP-suite v1 by Shun Sakuraba (QST). As a derivative
work of GPL-3.0-or-later code, this project is distributed under
GPL-3.0-or-later as well (see [COPYING](COPYING)): the copyright of the v2
implementation is © 2026 Cedrus810, and the original v1 pipelines
(`abfe/`, `feprest/`) remain © Shun Sakuraba, unchanged.

## Original v1 pipelines (by Shun Sakuraba)

FEP-suite is a collection of programs for calculating various free energy
differences, using GROMACS as a backend. See
[abfe/README.md](abfe/README.md) and [feprest/README.md](feprest/README.md)
for the v1 pipelines and how to use them. The program is currently in
"beta" version.

# License

GPL 3 or later (see [COPYING](COPYING) for details)

feprest/fepgen/cmdline.h is licensed under BSD 3-clause. See the file for details.

# Authors

* v2 rewrite, the abfe_v2 and feprest_v2 pipelines: [Cedrus810](https://github.com/Cedrus810)
* original v1 pipelines, abfe and feprest: Shun Sakuraba (National Institutes for Quantum Science and Technology, Japan)

# Acknowledgements

This software was supported by following fundings:
* a Grant-in-aid for Young Scientists (B) (grant no. 16K17778) from Japan Society for the Promotion of Science (JSPS).
* a Grant-in-aid for Scientific Research on Innovative Areas "Molecular Engine" (grant no. 19H05410) from JSPS.
* Platform Project for Supporting Drug Discovery and Life Science Research (Basis for Supporting Innovative Drug Discovery and Life Science Research (BINDS)) under Grant Number JP21am0101106, Agency for Medical Research and Development (AMED), Japan
* Project Focused on Developing Key Technology for Discovering and Manufacturing Drugs for Next-Generation Treatment and Diagnosis from AMED (JP21ae0121005s0101).
