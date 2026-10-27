# Protocol addendum A5: selective-prediction baselines (pre-run)

Written 2026-10-27 before any A5 result was produced. Purpose: compare the selective
certified LTT rule with standard selective-prediction and conformal baselines on the
same calibration/test pairs.

## Rules compared (scripts/mlwa/baselines.py, scripts/realdata/rules.py)
1. LTT-selective (ours), band m = 0.10, delta = 0.10.
2. Empirical (uncorrected) threshold with the same band.
3. Softmax response (SR): keep max(p, 1-p) >= c, predict p >= 0.5.
   - SR-emp: c = smallest cut with calibration retained FNR <= alpha (no correction).
   - SR-cov70: c = calibration quantile giving 70% coverage.
4. SGR (Geifman & El-Yaniv 2017), binary-search certified SR (existing implementation).
5. Split conformal, LAC score, marginal and class-conditional (Mondrian) at level alpha;
   singleton sets retained, empty or two-label sets deferred.
6. Conformal risk control (CRC) for FNR, no abstention.
7. SelectiveNet (Geifman & El-Yaniv 2019), two hidden layers of 64 units, target
   coverage 0.70, lambda = 32, auxiliary-head weight 0.5, 40 epochs, Adam 1e-3,
   positive class weighted n_neg/n_pos, trained on the training fold only.
   - SN-cov70: selection cut = calibration quantile for 70% coverage.
   - SN-emp: smallest cut with calibration retained FNR <= alpha.

## Data and runs
- Real: PhysioNet/CinC 2019, same features, 40/20/40 splits and seeds (1000 + run) as
  run_realdata.py; LR base model; settings within A, within B, cross A->B, cross B->A,
  and 100-patient calibration subsample at site A; alpha in {0.05, 0.10}; 50 runs.
- Simulation: fresh cohort per run (generator seed 777000 + run), independent
  9,000-patient pool (seed 20261027), RF base model as in r6; moderate MCAR, severe MCAR,
  MNAR; full and half calibration fold; alpha = 0.05; 50 runs.

## Outcomes
Coverage, retained FNR, retained FPR, selective 0-1 error, AURC (0-1 and retained-FNR
risk), refusals, and violations (retained FNR > alpha on the test fold / pool), with
Clopper-Pearson 95% intervals on violation counts. No outcome or setting will be dropped
after results are seen; deviations will be logged below.

## Deviations
(none yet)
