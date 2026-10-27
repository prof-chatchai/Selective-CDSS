# Protocol addendum A6: subgroup retained FNR on real ICU data (pre-run)

Written 2026-10-27 before any A6 result was produced. Motivation: the guarantee is
marginal; the simulated ACS subgroup failed. A6 measures how the certified rule
behaves in clinically defined subgroups of the real data.

- Data, features, splits, seeds (1000 + run), LR model and certified rule exactly as
  run_realdata.py; alpha = 0.05, delta = 0.10, m = 0.10; within site A and within site B;
  50 runs; the threshold is calibrated on the whole calibration fold (marginal).
- Subgroups on the test fold: sex (Gender 0 / 1), age < 65 / >= 65, and ICU unit
  (Unit1 = 1 medical; Unit2 = 1 surgical; neither recorded).
- Outcome per subgroup and run: retained FNR and violation (> alpha), retained
  positives; summary = violations out of served runs with Clopper-Pearson 95% CI and
  mean retained FNR. Subgroups with fewer than 10 retained test positives in a run are
  counted as "too small" in that run.
- Secondary: the same rule calibrated per subgroup (Mondrian), refusals counted.
- No subgroup or outcome will be dropped after results are seen.
