"""TerraBalance - Steps 7 to 13.

  Step 7   rollup_pins()      one row per PIN + majority owner
  Step 8   Problem.score()    the four-term score and the hard rules
  Step 9   solve()            start at majority owner, move PINs while the score improves
  Step 10  PRESETS            weight settings to compare
  Step 11  rate_reps(), rollups()
  Step 12  explain_moves()
  Step 13  solve(..., unavailable={...}) re-aligns after a rep leaves

Run directly for a text report:  python optimiser.py
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from metrics import haversine, load

TRAVEL_SCALE_KM = 10.0   # travel term = average km per visit / 10, so it sits near the CV terms


# =============================== Step 7: PIN roll-up ===============================
def rollup_pins(customers, reps, pins):
    """One row per PIN: customers, workload, potential, majority owner.
    Also returns cnt[pin, rep] = how many customers of that PIN each rep owns today."""
    g = customers.groupby("pin").agg(customers=("customer_id", "count"),
                                     workload=("visits", "sum"),
                                     potential=("potential", "sum"))
    p = pins[["pin", "region", "city", "lat", "lon"]].merge(g, on="pin", how="left")
    p = p[p.customers.notna()].reset_index(drop=True)          # drop PINs with no customers
    p[["customers", "workload", "potential"]] = p[["customers", "workload", "potential"]].astype(int)
    cnt = (pd.crosstab(customers.pin, customers.current_owner)
           .reindex(index=p.pin, columns=reps.rep_id, fill_value=0))
    p["majority_owner"] = cnt.idxmax(axis=1).values
    p["majority_share"] = (cnt.max(axis=1) / cnt.sum(axis=1)).values
    p["owner_mix"] = [", ".join(f"{r}:{n}" for r, n in row[row > 0].sort_values(ascending=False).items())
                      for _, row in cnt.iterrows()]
    return p, cnt.values


# =============================== Step 8: the score ===============================
@dataclass
class Weights:
    travel: float = 1.0
    workload: float = 1.0
    potential: float = 1.0
    relationship: float = 1.0


def cv(x):
    return float(x.std() / x.mean()) if x.mean() > 0 else 0.0


class Problem:
    def __init__(self, customers, reps, pins, cnt):
        self.reps, self.pins, self.cnt = reps, pins, cnt
        self.rep_ids = reps.rep_id.tolist()
        self.pin_idx = {p: i for i, p in enumerate(pins.pin)}
        self.wk = pins.workload.values.astype(float)
        self.pt = pins.potential.values.astype(float)
        self.N = float(cnt.sum())
        self.W = float(self.wk.sum())
        self.D = haversine(pins.lat.values[:, None], pins.lon.values[:, None],
                           reps.home_lat.values[None, :], reps.home_lon.values[None, :])
        self.same_city = pins.city.to_numpy(object)[:, None] == reps.city.to_numpy(object)[None, :]

    def active_mask(self, unavailable=()):
        return ~np.isin(self.rep_ids, list(unavailable))

    def allowed(self, unavailable=()):
        """Hard rule: same city AND rep is available."""
        return self.same_city & self.active_mask(unavailable)[None, :]

    def _cost(self, lw, lp, trav, kept, active, w):
        t = trav / self.W / TRAVEL_SCALE_KM
        a, b = cv(lw[active]), cv(lp[active])
        r = 1 - kept / self.N
        total = w.travel * t + w.workload * a + w.potential * b + w.relationship * r
        return total, dict(travel=t, workload_cv=a, potential_cv=b, relationship=r)

    def score(self, assign, w, unavailable=()):
        """Score of a full assignment (lower is better). assign[i] = rep index of PIN i."""
        n, R = len(assign), len(self.rep_ids)
        lw = np.bincount(assign, weights=self.wk, minlength=R)
        lp = np.bincount(assign, weights=self.pt, minlength=R)
        trav = float((self.wk * self.D[np.arange(n), assign]).sum())
        kept = float(self.cnt[np.arange(n), assign].sum())
        return self._cost(lw, lp, trav, kept, self.active_mask(unavailable), w)

    def violations(self, assign, unavailable=()):
        """Number of PINs breaking the hard rule (should always be 0)."""
        return int((~self.allowed(unavailable)[np.arange(len(assign)), assign]).sum())

    def owner_ids(self, assign):
        return np.array(self.rep_ids)[assign]


# =============================== Step 9: the optimiser ===============================
def majority_start(problem, unavailable=()):
    a = problem.cnt.argmax(axis=1)
    return fix_forbidden(problem, a, unavailable)


def fix_forbidden(problem, assign, unavailable=()):
    """Any PIN sitting with a forbidden rep goes to its nearest allowed rep."""
    a = assign.copy()
    ok = problem.allowed(unavailable)
    for i in range(len(a)):
        if not ok[i, a[i]]:
            d = np.where(ok[i], problem.D[i], np.inf)
            a[i] = int(d.argmin())
    return a


def solve(problem, w, start=None, unavailable=(), k=4, max_pass=50):
    """Local search: for each PIN try its k nearest allowed reps (plus where it is now),
    keep the best move if the score improves. Repeat until a full pass changes nothing."""
    n, R = len(problem.pins), len(problem.rep_ids)
    ok = problem.allowed(unavailable)
    active = problem.active_mask(unavailable)
    a = fix_forbidden(problem, majority_start(problem) if start is None else np.asarray(start), unavailable)
    start_assign = a.copy()
    lw = np.bincount(a, weights=problem.wk, minlength=R)
    lp = np.bincount(a, weights=problem.pt, minlength=R)
    trav = float((problem.wk * problem.D[np.arange(n), a]).sum())
    kept = float(problem.cnt[np.arange(n), a].sum())
    cur, _ = problem._cost(lw, lp, trav, kept, active, w)
    history = [cur]
    order = np.argsort(-problem.wk)                      # heavy PINs first
    nearest = [np.where(ok[i])[0][np.argsort(problem.D[i][ok[i]])][:k] for i in range(n)]

    for _ in range(max_pass):
        changed = 0
        for i in order:
            old = a[i]
            best_r, best_c = old, cur
            for r in set(nearest[i].tolist()) | {old}:
                if r == old:
                    continue
                lw2, lp2 = lw.copy(), lp.copy()
                lw2[old] -= problem.wk[i]; lw2[r] += problem.wk[i]
                lp2[old] -= problem.pt[i]; lp2[r] += problem.pt[i]
                trav2 = trav + problem.wk[i] * (problem.D[i, r] - problem.D[i, old])
                kept2 = kept + problem.cnt[i, r] - problem.cnt[i, old]
                c, _ = problem._cost(lw2, lp2, trav2, kept2, active, w)
                if c < best_c - 1e-9:
                    best_r, best_c = r, c
            if best_r != old:
                lw[old] -= problem.wk[i]; lw[best_r] += problem.wk[i]
                lp[old] -= problem.pt[i]; lp[best_r] += problem.pt[i]
                trav += problem.wk[i] * (problem.D[i, best_r] - problem.D[i, old])
                kept += problem.cnt[i, best_r] - problem.cnt[i, old]
                a[i], cur = best_r, best_c
                changed += 1
        history.append(cur)
        if changed == 0:
            break
    return a, start_assign, history


# =============================== Step 10: weight settings ===============================
PRESETS = {
    "Stability first": Weights(travel=0.5, workload=0.5, potential=0.5, relationship=3.0),
    "Gentle":          Weights(travel=1.0, workload=1.0, potential=1.0, relationship=1.5),
    "Balanced (recommended)": Weights(travel=2.0, workload=2.0, potential=1.0, relationship=1.5),
    "Balance first":   Weights(travel=0.5, workload=3.0, potential=2.0, relationship=0.5),
    "Travel first":    Weights(travel=3.0, workload=1.0, potential=0.5, relationship=0.5),
}
DEFAULT_PRESET = "Balanced (recommended)"


def apply_assignment(customers, problem, assign, col="new_owner"):
    c = customers.copy()
    c[col] = problem.owner_ids(assign)[c.pin.map(problem.pin_idx).values]
    return c


# =============================== Step 12: explanations ===============================
def explain_moves(problem, start, final, customers):
    """One row per PIN that changed rep, with numbers and a one-line reason."""
    R = len(problem.rep_ids)
    lw0 = np.bincount(start, weights=problem.wk, minlength=R)
    lw1 = np.bincount(final, weights=problem.wk, minlength=R)
    team_avg = lw1[lw1 > 0].mean()
    name = problem.reps.set_index("rep_id").name.to_dict()
    rows = []
    for i in np.where(start != final)[0]:
        o, n = start[i], final[i]
        d_old, d_new = problem.D[i, o], problem.D[i, n]
        keep_new = int(problem.cnt[i, n])
        total = int(problem.pins.customers[i])
        bits = []
        if d_new < d_old - 0.5:
            bits.append(f"cuts the distance by {d_old - d_new:.1f} km ({d_old:.1f} to {d_new:.1f} km)")
        elif d_new > d_old + 0.5:
            bits.append(f"adds {d_new - d_old:.1f} km of distance ({d_old:.1f} to {d_new:.1f} km) to even out load")
        if lw0[o] > 1.1 * team_avg and lw0[n] < lw0[o]:
            bits.append(f"takes load off {name[problem.rep_ids[o]]}, who was above the team average")
        if lw0[n] < 0.9 * team_avg:
            bits.append(f"gives {name[problem.rep_ids[n]]}, who was below average, more work")
        if not bits:
            bits.append("improves the overall score")
        sentence = (f"PIN {problem.pins.pin[i]} moves from {name[problem.rep_ids[o]]} to "
                    f"{name[problem.rep_ids[n]]}: it " + "; it ".join(bits) + ". "
                    f"{name[problem.rep_ids[o]]}'s load goes {lw0[o]:.0f} to {lw1[o]:.0f} visits/month and "
                    f"{name[problem.rep_ids[n]]}'s goes {lw0[n]:.0f} to {lw1[n]:.0f} (team average {team_avg:.0f}). "
                    f"{keep_new} of its {total} customers already belonged to {name[problem.rep_ids[n]]}; "
                    f"{total - keep_new} change rep.")
        rows.append(dict(pin=int(problem.pins.pin[i]), customers=total, from_rep=problem.rep_ids[o],
                         to_rep=problem.rep_ids[n], km_before=round(d_old, 1), km_after=round(d_new, 1),
                         from_load_before=lw0[o], from_load_after=lw1[o],
                         to_load_before=lw0[n], to_load_after=lw1[n],
                         already_theirs=keep_new, reason=sentence))
    return pd.DataFrame(rows)


# =============================== Step 11: rep table, flags, ratings ===============================
RATING_WEIGHTS = {"Workload fairness": 25, "Potential fairness": 15, "Travel": 25,
                  "Relationships kept": 20, "Small new book": 15}
TRAVEL_CAP_KM = 15.0     # 0 km = full marks, 15 km per visit or more = zero


def rate_reps(customers, reps, owner_col, active=None):
    """Per-rep table with flags and a 1-5 star territory rating.

    The rating scores the TERRITORY a rep ends up with (fair load, fair potential, short drives,
    relationships kept, few new customers to win over). It is not a performance review."""
    active = reps.rep_id.tolist() if active is None else active
    c = customers.copy()
    homes = reps.set_index("rep_id")
    h = homes.loc[c[owner_col]]
    c["km"] = haversine(c.lat.values, c.lon.values, h.home_lat.values, h.home_lon.values)
    c["vkm"] = c.km * c.visits
    mine = c[owner_col] == c.current_owner

    g = c.groupby(owner_col).agg(customers=("customer_id", "count"), visits=("visits", "sum"),
                                 potential=("potential", "sum"), vkm=("vkm", "sum"))
    g["kept"] = c[mine].groupby(owner_col).size()
    t = g.reindex(reps.rep_id).fillna(0)
    t["avg_km_per_visit"] = (t.vkm / t.visits.replace(0, np.nan)).fillna(0)
    t["had_before"] = c.groupby("current_owner").size().reindex(reps.rep_id).fillna(0)
    t["gained"] = t.customers - t.kept
    t["lost"] = t.had_before - t.kept
    t["retention"] = (t.kept / t.had_before.replace(0, np.nan)).fillna(1.0)
    t["new_share"] = (t.gained / t.customers.replace(0, np.nan)).fillna(0)
    t = t.drop(columns="vkm")
    t.insert(0, "name", reps.set_index("rep_id").name)
    t.insert(1, "area_manager", reps.set_index("rep_id").area_manager)
    t["active"] = t.index.isin(active)

    a = t[t.active]
    mw, mp, med_km = a.visits.mean(), a.potential.mean(), a.avg_km_per_visit.median()
    flags = []
    for rid, r in t.iterrows():
        f = []
        if r.active:
            if r.visits > 1.2 * mw: f.append("Overloaded")
            if r.visits < 0.8 * mw: f.append("Underloaded")
            if r.potential > 1.2 * mp: f.append("High potential")
            if r.potential < 0.8 * mp: f.append("Low potential")
            if r.avg_km_per_visit > 1.5 * med_km: f.append("Sprawling")
        else:
            f.append("Unavailable")
        flags.append(", ".join(f) if f else "OK")
    t["flags"] = flags

    close = lambda x, m: np.clip(1 - np.abs(x / m - 1) / 0.5, 0, 1)
    sub = pd.DataFrame({
        "Workload fairness": close(t.visits, mw),
        "Potential fairness": close(t.potential, mp),
        "Travel": np.clip(1 - t.avg_km_per_visit / TRAVEL_CAP_KM, 0, 1),
        "Relationships kept": t.retention,
        "Small new book": 1 - t.new_share}, index=t.index)
    pts = sum(sub[k] * v for k, v in RATING_WEIGHTS.items())            # 0-100
    t["rating"] = np.where(t.active, (1 + 4 * pts / 100).round(1), np.nan)
    t["stars"] = [("★" * int(round(x)) + "☆" * (5 - int(round(x)))) if x == x else "-" for x in t.rating]
    t = t.join(sub.add_prefix("s_"))
    return t


def rollups(rep_table):
    """Area Manager and National roll-ups, plus a check that the levels add up."""
    t = rep_table[rep_table.active]

    def agg(g):
        return pd.Series({"reps": len(g), "customers": g.customers.sum(), "visits": g.visits.sum(),
                          "potential": g.potential.sum(),
                          "avg_km_per_visit": (g.avg_km_per_visit * g.visits).sum() / g.visits.sum(),
                          "workload_cv": cv(g.visits) if len(g) > 1 else 0.0,
                          "customers_kept_%": 100 * g.kept.sum() / g.customers.sum(),
                          "avg_stars": g.rating.mean()})

    am = t.groupby("area_manager").apply(agg, include_groups=False)
    national = agg(t).to_frame("National").T
    checks = {"customers add up": int(am.customers.sum()) == int(national.customers.iloc[0]) == int(t.customers.sum()),
              "visits add up": int(am.visits.sum()) == int(national.visits.iloc[0]),
              "potential adds up": int(am.potential.sum()) == int(national.potential.iloc[0]),
              "reps add up": int(am.reps.sum()) == int(national.reps.iloc[0])}
    return am, national, checks


if __name__ == "__main__":
    from metrics import compute_metrics, nearest_rep
    customers, reps, pins_raw = load()
    customers["baseline_owner"] = nearest_rep(customers, reps)
    pins, cnt = rollup_pins(customers, reps, pins_raw)
    prob = Problem(customers, reps, pins, cnt)
    rows = {"Current": compute_metrics(customers, reps, "current_owner"),
            "Nearest-rep": compute_metrics(customers, reps, "baseline_owner")}
    for name, w in PRESETS.items():
        a, s, hist = solve(prob, w)
        c = apply_assignment(customers, prob, a)
        m = compute_metrics(c, reps, "new_owner")
        m["PINs moved"] = int((a != s).sum()); m["passes"] = len(hist) - 1
        m["violations"] = prob.violations(a)
        rows[name] = m
    pd.set_option("display.width", 200)
    print(pd.DataFrame(rows).round(2))
