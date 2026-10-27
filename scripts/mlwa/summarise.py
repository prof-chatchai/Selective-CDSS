"""Summarise A5 parts -> results/mlwa/summary.json.
Usage: python -I summarise.py <results/mlwa>"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import beta


def cp(k, n):
    lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    return [lo, hi]


def summ(root, alpha_of):
    parts = sorted((root / "parts").glob("run_*.json"))
    runs = [json.loads(p.read_text()) for p in parts]
    out = {"n_runs": len(runs), "settings": {}}
    for key in runs[0]:
        alpha = float(key.split("|")[1]); S = {}
        for rule in runs[0][key]:
            v = [r[key][rule] for r in runs]
            ok = [d for d in v if not d["refused"]]
            k = sum(d["rfnr"] > alpha for d in ok); n = len(v)
            f = lambda m: float(np.mean([d[m] for d in ok])) if ok and m in ok[0] else None  # noqa: E731
            S[rule] = dict(runs=n, refused=n - len(ok), violations=int(k), viol_ci=cp(int(k), n),
                           rfnr_mean=f("rfnr"), rfnr_max=float(max(d["rfnr"] for d in ok)) if ok else None,
                           coverage_mean=f("coverage"), rfpr_mean=f("rfpr"), sel_err_mean=f("sel_err"),
                           aurc01_mean=f("aurc01"), aurc_fnr_mean=f("aurc_fnr"),
                           retained_pos_mean=f("retained_pos"))
        out["settings"][key] = S
    return out


if __name__ == "__main__":
    root = Path(sys.argv[1]); res = {}
    for d in ("real", "sim"):
        if (root / d / "parts").exists():
            res[d] = summ(root / d, None)
    (root / "summary.json").write_text(json.dumps(res, indent=1))
    for d, r in res.items():
        print(d, r["n_runs"])
        for key, S in r["settings"].items():
            print(" ", key)
            for rule, s in S.items():
                print("   %-20s ref %2d viol %2d rFNR %s cov %s rFPR %s AURC %s" % (
                    rule, s["refused"], s["violations"],
                    None if s["rfnr_mean"] is None else round(100 * s["rfnr_mean"], 1),
                    None if s["coverage_mean"] is None else round(100 * s["coverage_mean"], 1),
                    None if s["rfpr_mean"] is None else round(100 * s["rfpr_mean"], 1),
                    None if s["aurc_fnr_mean"] is None else round(100 * s["aurc_fnr_mean"], 1)))
