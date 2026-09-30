# LRC explodes to 10^4..10^10 kcal/mol with GROMACS 2026.3: `mdrun -rerun` with `vdwtype = PME` + soft-core decoupling produces garbage `LJ recip.`

*(filed upstream as
[shunsakuraba/fepsuite#9](https://github.com/shunsakuraba/fepsuite/issues/9);
this document is the full investigation and the v2 mitigation record.)*

## Summary

With GROMACS 2026.3, the long-range-correction (LRC) re-evaluation of the
**annihilation endpoints** of the ABFE thermodynamic cycle returns
astronomically wrong energies, poisoning `result.txt`:

| system | annihilation | charging | restraint | **long-range-correction** | total |
|---|---|---|---|---|---|
| brd4_ligand1 | -19.209 ± 0.191 | -0.038 ± 0.050 | 6.451 ± 0.005 | **-32641.052 ± 1828.196** | -32653.848 ± 1828.196 |
| brd4_ligand2 | -16.448 ± 0.398 | -1.207 ± 0.063 | 6.552 ± 0.017 | **1.07e10 ± 9.78e9** | 1.07e10 ± 9.78e9 |

(kcal/mol; annihilation/charging/restraint are sane, only the LRC is broken.)

## Environment

* FEP-suite `abfe` pipeline (v1; the re-evaluation logic is shared by v2)
* GROMACS **2026.3** for both the production runs and the `mdrun -rerun`
  re-evaluations
* Hamiltonian replica exchange, `vdwtype = PME` re-evaluation mdps

## Evidence

Per-term decomposition of the perturbation (`long - short`, kJ/mol) over
2001 aligned frames, from the actual run directories
(`gmx energy`/pyedr on `brd4_ligand1`):

| term | lr-complex (sane) | lr-annihilation-lig (broken) | lr-annihilation-complex (broken) |
|---|---|---|---|
| `LJ (SR)` diff | +38587 ± 141 | +5325 ± 41 | +39016 ± 2383 |
| `LJ recip.` (only in re-eval) | **-43657 ± ...** | **-784228 ± 65522** (min -1174292) | **-256490 ± ...** |
| `Disper. corr.` (only in original) | +5021 | +717 | +5001 |
| resulting LRC | **-1.910 kcal (sane)** | **-41245 kcal** | **+8607 kcal** |

Observations:

1. Only the two `lr-annihilation-*` re-evaluations are broken. These are
   exactly the ones whose mdp carries the **soft-core FEP block** into the
   re-evaluation (`couple_moltype`, `couple-lambda0 = vdw`,
   `couple-lambda1 = none`, `sc-alpha = 0.5`, `init-lambda-state = 0` with
   `fep-lambdas = 1`, i.e. the fully decoupled state).
   `lr-complex`/`lr-lig` (plain coupled re-evaluations, no FEP block) are sane.
2. The garbage is in the **reciprocal-space LJ term**: for the *ligand-only*
   box, `LJ recip.` averages **-784,228 kJ/mol** with a 65,522 kJ standard
   deviation — roughly 18x the (already large) value of the full complex box,
   for a system ~10x smaller. The decoupled endpoint has almost no
   ligand-environment LJ to re-evaluate at all, so the correct value is ~0.
3. `grompp`/`mdrun` emit **no warning**; the run "succeeds".
4. The same workflow on GROMACS 2022.5 produced ~±0.1 kcal/mol for these
   terms (v1 README example), so this is a behavior change in newer GROMACS,
   most likely in the PME-LJ handling of soft-core-scaled / decoupled
   (`couple-lambda1 = none`) pair exclusions in reciprocal space.

Side note (gotcha): `gmx energy` of 2026.3 *displays* renamed terms
(`LJ-(SR)`, `LJ-recip.`), while the EDR files store the classic names
(`LJ (SR)`, `LJ recip.`); term-name matching must normalize both.

## Impact

Any ABFE calculation whose LRC re-evaluations run under a GROMACS build
with this behavior gets a poisoned `total` in `result.txt`. Because the
free-energy estimator is an exponential average, even a handful of extreme
frames destroys the estimate (the two systems above differ by 6 orders of
magnitude).

## Proposed fixes

In the pipeline (implemented in `abfe_v2`):

1. **Sanity guard**: abort the analysis (instead of writing `result.txt`)
   when any LRC term exceeds `LRC_SANITY_LIMIT` (default 50 kcal/mol).
2. **`SKIP_ANNIHILATION_LRC=yes`**: skip the re-evaluation of the
   decoupled endpoints entirely. Physically the lambda=1 state has no
   ligand-environment dispersion to correct; v1 measured ~0.1 kcal/mol for
   each endpoint and the two nearly cancel in the cycle, so recording them
   as zero is a sound workaround.
3. Normalized EDR term matching + configurable term list
   (`LRC_ENERGY_TERMS`) against GROMACS renames.

In GROMACS (needs upstream confirmation):

* check the reciprocal-space dispersion contribution for systems with
  soft-core-scaled / fully-decoupled `couple_moltype` pairs under
  `vdwtype = PME` — in particular whether exclusion corrections for
  decoupled pairs are applied with the correct sign/magnitude in 2026.3;
* consider a grompp warning/error for `vdwtype = PME` combined with
  soft-core decoupling if the combination is not supported.

Possible pipeline-level alternative (untested): drop soft-core from the
endpoint re-evaluations (`sc-alpha = 0`), since at the fully decoupled
endpoint no pair is partially scaled and soft-core should be unnecessary.
