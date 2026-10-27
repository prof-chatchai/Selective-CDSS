"""MLWA baseline comparison (protocol addendum A5, scripts/mlwa/PROTOCOL_A5.md).

Usage:
  python -I run_mlwa.py real <repo> <features_w6.csv.gz> <out_dir> [n_runs=50]
  python -I run_mlwa.py sim  <repo> <out_dir> [n_runs=50]
Resumable: one JSON per run in <out_dir>/parts/. summarise.py builds the tables.
"""
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "realdata"))
import baselines as B  # noqa: E402
import rules as RL  # noqa: E402

DELTA, M, COV = 0.10, 0.10, 0.70


def evaluate(pc, yc, pt, yt, alpha, snc=None, snt=None):
    """All rules on one calibration/test pair. snc/snt = (f, g) SelectiveNet outputs."""
    out = {}

    def put(name, ret, pred, conf=None, refused=False):
        if refused:
            out[name] = dict(refused=True); return
        ret = np.asarray(ret, bool); pred = np.asarray(pred).astype(int)
        rp = int(((yt == 1) & ret).sum()); fn = int(((yt == 1) & ret & (pred == 0)).sum())
        rn = int(((yt == 0) & ret).sum()); fp = int(((yt == 0) & ret & (pred == 1)).sum())
        d = dict(refused=False, coverage=float(ret.mean()), retained_pos=rp,
                 rfnr=fn / rp if rp else 0.0, rfpr=fp / rn if rn else 0.0,
                 sel_err=float((pred[ret] != yt[ret]).mean()) if ret.any() else 0.0)
        if conf is not None:
            d["aurc01"], d["aurc_fnr"] = B.aurc(conf, yt, pred)
        out[name] = d

    t = RL.tau_selcert(pc, yc, alpha, DELTA, M)
    if t is None:
        put("ltt_selective", None, None, refused=True)
    else:
        put("ltt_selective", np.abs(pt - t) >= M, pt >= t, np.abs(pt - t))
    t = RL.tau_emp(pc, yc, alpha)
    if t is None:
        put("empirical", None, None, refused=True)
    else:
        put("empirical", np.abs(pt - t) >= M, pt >= t, np.abs(pt - t))
    conf_t = np.maximum(pt, 1 - pt)
    c = B.sr_emp(pc, yc, alpha)
    if c is None:
        put("sr_emp", None, None, refused=True)
    else:
        put("sr_emp", conf_t >= c, pt >= .5, conf_t)
    put("sr_cov70", conf_t >= B.sr_cov(pc, COV), pt >= .5, conf_t)
    c = RL.sgr_binary(pc, yc, alpha, DELTA)
    if c is None:
        put("sgr", None, None, refused=True)
    else:
        put("sgr", conf_t >= c, pt >= .5, conf_t)
    for nm, mon in (("conformal", False), ("conformal_mondrian", True)):
        r, p = B.conformal_apply(pt, B.conformal_q(pc, yc, alpha, mon)); put(nm, r, p)
    t = RL.crc(pc, yc, alpha)
    if t is None:
        put("crc", None, None, refused=True)
    else:
        put("crc", np.ones(len(pt), bool), pt >= t)
    if snc is not None:
        fc, gc = snc; ft, gt = snt
        put("selectivenet_cov70", gt >= float(np.quantile(gc, 1 - COV)), ft >= .5, gt)
        c = B.sn_emp(fc, gc, yc, alpha)
        if c is None:
            put("selectivenet_emp", None, None, refused=True)
        else:
            put("selectivenet_emp", gt >= c, ft >= .5, gt)
    return out


# ---------------------------------------------------------------- real data
def run_real(repo, feats_path, out, n_runs):
    import pandas as pd
    import run_realdata as RR
    df = pd.read_csv(feats_path); df = df[df.included == 1]
    feats = [c for c in df.columns if c not in RR.NON_FEAT]
    dropped = [c for c in feats if df.groupby("site")[c].apply(lambda v: v.notna().sum()).min() == 0]
    feats = [c for c in feats if c not in dropped]
    S = {s: (df[df.site == s][feats].to_numpy(float), df[df.site == s]["y"].to_numpy(int)) for s in "AB"}
    parts = out / "parts"; parts.mkdir(parents=True, exist_ok=True)
    for run in range(n_runs):
        fp = parts / f"run_{run:02d}.json"
        if fp.exists():
            continue
        seed = 1000 + run
        sp = {s: RR.split(S[s][1], seed) for s in "AB"}
        P, SN = {}, {}
        for s in "AB":
            X, y = S[s]; tr = sp[s][0]
            clf = RR.model("LR", seed).fit(X[tr], y[tr])
            med = np.nanmedian(X[tr], axis=0); med = np.where(np.isnan(med), 0.0, med)
            imp = lambda Z: np.where(np.isnan(Z), med, Z)  # noqa: E731
            obs = np.isnan(X[tr]).all(0)
            sn = B.train_selectivenet(imp(X[tr])[:, ~obs], y[tr], seed)
            for t in "AB":
                P[(s, t)] = clf.predict_proba(S[t][0])[:, 1]
                SN[(s, t)] = sn(imp(S[t][0])[:, ~obs])
        res = {}
        for alpha in (0.05, 0.10):
            for s in "AB":
                for t in "AB":
                    ca, te = sp[s][1], sp[t][2]; ys, yt = S[s][1], S[t][1]
                    key = f"{'within' if s == t else 'cross'}_{s}to{t}|{alpha}"
                    f_c, g_c = SN[(s, s)]; f_t, g_t = SN[(s, t)]
                    res[key] = evaluate(P[(s, s)][ca], ys[ca], P[(s, t)][te], yt[te], alpha,
                                        (f_c[ca], g_c[ca]), (f_t[te], g_t[te]))
            # small calibration: 100 patients from site A
            rng = np.random.default_rng(seed); sub = rng.choice(sp["A"][1], 100, replace=False)
            te = sp["A"][2]; yA = S["A"][1]; f_a, g_a = SN[("A", "A")]
            res[f"small100_A|{alpha}"] = evaluate(P[("A", "A")][sub], yA[sub], P[("A", "A")][te], yA[te], alpha,
                                                  (f_a[sub], g_a[sub]), (f_a[te], g_a[te]))
        fp.write_text(json.dumps(res)); print("run", run, flush=True)


# ---------------------------------------------------------------- simulation
def run_sim(repo, out, n_runs):
    sys.path.insert(0, repo); sys.path.insert(0, repo + "/src")
    import pandas as pd
    import run_eval as R
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    SET = {"moderate": ("mcar", .35, .6), "severe": ("mcar", .5, .9), "mnar": ("mnar", .35, .6)}

    def gen(n_per):
        rows, sev = [], []
        for _, (cls, spec, lat) in R.BASELINES.items():
            m = cls()
            for _ in range(n_per):
                f, s = R.make_patient(m, spec, lat); rows.append(f); sev.append(s)
        return pd.DataFrame(rows), np.array(sev)
    parts = out / "parts"; parts.mkdir(parents=True, exist_ok=True)
    R.rng = np.random.default_rng(20261027)  # A5 stream; independent of r5/r6
    dev, sev0 = gen(R.N_PER); CUT = float(np.quantile(sev0, 0.66)); cols = list(dev.columns)
    pdf, psev = gen(3000)
    Xp = pdf[cols].apply(lambda c: c.fillna(c.median())).values; yp = (psev >= CUT).astype(int)
    for s in range(n_runs):
        fp = parts / f"run_{s:02d}.json"
        R.rng = np.random.default_rng(777000 + s)  # fresh cohort per run, reproducible on resume
        df, sv = gen(R.N_PER)
        if fp.exists():
            continue
        X = df[cols].apply(lambda c: c.fillna(c.median())).values; y = (sv >= CUT).astype(int)
        idx = np.arange(len(y))
        a, b = train_test_split(idx, test_size=0.5, random_state=s, stratify=y)
        c_, _ = train_test_split(b, test_size=0.5, random_state=s, stratify=y[b])
        med = {c: float(np.median(X[a, j])) for j, c in enumerate(cols)}
        clf = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=s, n_jobs=-1).fit(X[a], y[a])
        sn = B.train_selectivenet(X[a], y[a], s)
        yc = y[c_]
        half = train_test_split(np.arange(len(yc)), test_size=0.5, random_state=s, stratify=yc)[0]
        res = {}
        for nm, (mech, rate, sig) in SET.items():
            Xc = R.degrade(X[c_], cols, med, mech, rate, sig, seed=s)
            Xq = R.degrade(Xp, cols, med, mech, rate, sig, seed=1000 + s)
            qc, qp = clf.predict_proba(Xc)[:, 1], clf.predict_proba(Xq)[:, 1]
            fc, gc = sn(Xc); fq, gq = sn(Xq)
            for cal, sel in (("full", np.arange(len(yc))), ("half", half)):
                res[f"{nm}_{cal}|0.05"] = evaluate(qc[sel], yc[sel], qp, yp, 0.05, (fc[sel], gc[sel]), (fq, gq))
        fp.write_text(json.dumps(res)); print("run", s, flush=True)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "real":
        run_real(sys.argv[2], sys.argv[3], Path(sys.argv[4]), int(sys.argv[5]) if len(sys.argv) > 5 else 50)
    else:
        run_sim(sys.argv[2], Path(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 50)
