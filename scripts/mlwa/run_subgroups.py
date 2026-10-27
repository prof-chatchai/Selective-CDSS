"""Addendum A6: subgroup retained FNR of the certified rule on PhysioNet/CinC 2019.
Usage: python -I run_subgroups.py <features_w6.csv.gz> <out_dir> [n_runs=50]"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "realdata"))
import rules as RL  # noqa: E402
import run_realdata as RR  # noqa: E402

A, D, M = 0.05, 0.10, 0.10


def groups(df):
    return {"female": df.Gender.values == 0, "male": df.Gender.values == 1,
            "age<65": df.Age.values < 65, "age>=65": df.Age.values >= 65,
            "unit_medical": df.Unit1.values == 1, "unit_surgical": df.Unit2.values == 1,
            "unit_unknown": df.Unit_missing.values == 1}


def main():
    df = pd.read_csv(sys.argv[1]); df = df[df.included == 1]
    out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
    n_runs = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    feats = [c for c in df.columns if c not in RR.NON_FEAT]
    dropped = [c for c in feats if df.groupby("site")[c].apply(lambda v: v.notna().sum()).min() == 0]
    feats = [c for c in feats if c not in dropped]
    rows = []
    for s in "AB":
        d = df[df.site == s].reset_index(drop=True)
        X, y = d[feats].to_numpy(float), d.y.to_numpy(int); G = groups(d)
        for run in range(n_runs):
            seed = 1000 + run
            tr, ca, te = RR.split(y, seed)
            p = RR.model("LR", seed).fit(X[tr], y[tr]).predict_proba(X)[:, 1]
            tau = RL.tau_selcert(p[ca], y[ca], A, D, M)
            for g, mask in G.items():
                base = dict(site=s, run=run, group=g)
                # marginal threshold, evaluated in subgroup
                if tau is None:
                    rows.append(dict(base, mode="marginal", refused=True));
                else:
                    t = te[mask[te]]; ret = np.abs(p[t] - tau) >= M
                    rp = int(((y[t] == 1) & ret).sum()); fn = int(((y[t] == 1) & ret & (p[t] < tau)).sum())
                    rows.append(dict(base, mode="marginal", refused=False, retained_pos=rp,
                                     rfnr=fn / rp if rp else np.nan, small=rp < 10))
                # per-subgroup (Mondrian) calibration
                c = ca[mask[ca]]; tg = RL.tau_selcert(p[c], y[c], A, D, M)
                if tg is None:
                    rows.append(dict(base, mode="mondrian", refused=True))
                else:
                    t = te[mask[te]]; ret = np.abs(p[t] - tg) >= M
                    rp = int(((y[t] == 1) & ret).sum()); fn = int(((y[t] == 1) & ret & (p[t] < tg)).sum())
                    rows.append(dict(base, mode="mondrian", refused=False, retained_pos=rp,
                                     rfnr=fn / rp if rp else np.nan, small=rp < 10))
            print(s, run, flush=True)
    R = pd.DataFrame(rows); R.to_csv(out / "runs.csv", index=False)
    summ = []
    for (s, g, mode), x in R.groupby(["site", "group", "mode"]):
        ok = x[~x.refused.astype(bool)]; ev = ok[~ok.small.astype(bool)]
        k = int((ev.rfnr > A).sum()); n = len(ev)
        lo, hi = RL.cp(k, n) if n else [None, None]
        summ.append(dict(site=s, group=g, mode=mode, runs=len(x), refused=int(x.refused.sum()),
                         too_small=int(ok.small.sum()), evaluated=n, violations=k, ci=[lo, hi],
                         rfnr_mean=float(ev.rfnr.mean()) if n else None,
                         retained_pos_mean=float(ok.retained_pos.mean()) if len(ok) else None))
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    for r in summ:
        print(r)


if __name__ == "__main__":
    main()
