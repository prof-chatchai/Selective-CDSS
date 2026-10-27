"""Selective-prediction baselines for the MLWA comparison (protocol addendum A5).

All rules take calibration scores/labels and return a callable that maps test scores
(and, for SelectiveNet, selection scores) to (retained mask, predicted label), or None
when the rule refuses to certify.
"""
import math
import numpy as np
import torch
from torch import nn

GRID = np.linspace(0.01, 0.99, 99)


def rfnr(y, ret, pred):
    pos = (y == 1) & ret
    return float((pos & (pred == 0)).sum() / pos.sum()) if pos.any() else 0.0


# ---------------------------------------------------------------- softmax response
def sr_emp(pc, yc, alpha):
    """Softmax response (Geifman & El-Yaniv 2017): keep max(p,1-p) >= c, predict p>=0.5.
    c = smallest cut (largest coverage) whose calibration retained FNR <= alpha."""
    conf = np.maximum(pc, 1 - pc); pred = pc >= .5
    for c in np.unique(np.round(np.r_[0.5, conf], 6)):
        ret = conf >= c
        if rfnr(yc, ret, pred) <= alpha:
            return float(c)
    return None


def sr_cov(pc, cov):
    """Softmax response at a fixed target coverage (cut = calibration quantile)."""
    conf = np.maximum(pc, 1 - pc)
    return float(np.quantile(conf, 1 - cov))


# ---------------------------------------------------------------- split conformal
def conformal_q(pc, yc, alpha, mondrian=False):
    """LAC score s = 1 - p_hat(y|x). Marginal or class-conditional (Mondrian) quantile."""
    s = np.where(yc == 1, 1 - pc, pc)

    def q(v):
        n = len(v)
        if n == 0:
            return 1.0
        k = math.ceil((n + 1) * (1 - alpha))
        return 1.0 if k > n else float(np.sort(v)[k - 1])
    if mondrian:
        return (q(s[yc == 0]), q(s[yc == 1]))
    t = q(s); return (t, t)


def conformal_apply(pt, qs):
    """Prediction set {y: s_y <= q_y}; retain singletons and predict their label.
    Empty sets are deferred too."""
    in1 = (1 - pt) <= qs[1]; in0 = pt <= qs[0]
    ret = in1 ^ in0
    return ret, in1.astype(int)


# ---------------------------------------------------------------- SelectiveNet
class _SN(nn.Module):
    def __init__(self, d, h=64):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.1), nn.Linear(h, h), nn.ReLU())
        self.f = nn.Linear(h, 1); self.g = nn.Sequential(nn.Linear(h, 16), nn.ReLU(), nn.Linear(16, 1))
        self.aux = nn.Linear(h, 1)

    def forward(self, x):
        z = self.body(x)
        return self.f(z).squeeze(1), torch.sigmoid(self.g(z)).squeeze(1), self.aux(z).squeeze(1)


def train_selectivenet(X, y, seed, cov=0.7, lam=32.0, mix=0.5, epochs=40, bs=256):
    """SelectiveNet (Geifman & El-Yaniv 2019) for binary outcomes: selective risk
    + lam * max(0, cov - phi(g))^2, mixed with an auxiliary full-coverage head.
    Positive class weighted by n_neg/n_pos (as class_weight='balanced')."""
    torch.manual_seed(seed); np.random.seed(seed)
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Xt = torch.tensor((X - mu) / sd, dtype=torch.float32); yt = torch.tensor(y, dtype=torch.float32)
    w_pos = float((y == 0).sum() / max((y == 1).sum(), 1))
    net = _SN(X.shape[1]); opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    n = len(y); g_ = torch.Generator().manual_seed(seed)
    net.train()
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g_)
        for i in range(0, n, bs):
            b = perm[i:i + bs]; xb, yb = Xt[b], yt[b]
            wb = torch.where(yb == 1, torch.tensor(w_pos), torch.tensor(1.0))
            fo, go, ao = net(xb)
            l = bce(fo, yb) * wb
            phi = go.mean()
            sel = (l * go).mean() / phi.clamp_min(1e-6) + lam * torch.clamp(cov - phi, min=0) ** 2
            loss = mix * sel + (1 - mix) * (bce(ao, yb) * wb).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()

    def predict(Xn):
        with torch.no_grad():
            fo, go, _ = net(torch.tensor((Xn - mu) / sd, dtype=torch.float32))
        return torch.sigmoid(fo).numpy(), go.numpy()
    return predict


def sn_emp(fc, gc, yc, alpha):
    """Selection cut on g: smallest cut whose calibration retained FNR <= alpha."""
    pred = fc >= .5
    for c in np.unique(np.round(np.r_[0.0, gc], 6)):
        if rfnr(yc, gc >= c, pred) <= alpha:
            return float(c)
    return None


# ---------------------------------------------------------------- AURC
def aurc(conf, y, pred):
    """Area under the risk-coverage curve, 0-1 selective risk and retained-FNR risk,
    ranking by conf (descending); mean over all coverage levels k/n."""
    o = np.argsort(-conf, kind="stable"); yy = y[o]; err = (pred[o] != yy).astype(float)
    k = np.arange(1, len(y) + 1)
    r01 = np.cumsum(err) / k
    pos = np.cumsum(yy == 1); miss = np.cumsum((yy == 1) & (pred[o] == 0))
    ok = pos > 0
    rf = miss[ok] / pos[ok]
    return float(r01.mean()), float(rf.mean())
