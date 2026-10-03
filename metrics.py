"""TerraBalance - data loading, nearest-rep baseline, and the four metrics.

Imported by app.py. Can also be run directly:  python metrics.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data"


def haversine(lat1, lon1, lat2, lon2):
    """Straight-line distance in km (numpy arrays or scalars)."""
    p = np.pi / 180
    a = (np.sin((lat2 - lat1) * p / 2) ** 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2)
    return 12742 * np.arcsin(np.sqrt(a))


def load():
    customers = pd.read_csv(DATA / "customers.csv")
    reps = pd.read_csv(DATA / "reps.csv")
    pins = pd.read_csv(DATA / "pins.csv")
    return customers, reps, pins


# ---------- Using your own data: checks, defaults, and a friendly error if something is wrong ----------
class DataError(ValueError):
    """Raised with a plain-English message when an uploaded file cannot be used."""


CUSTOMER_COLS = ["customer_id", "pin", "lat", "lon", "grade", "city", "current_owner"]
REP_COLS = ["rep_id", "name", "city", "home_lat", "home_lon", "area_manager"]
VISITS = {"A": 4, "B": 3, "C": 2, "D": 1}        # visits per month by grade
POTENTIAL = {"A": 8, "B": 4, "C": 2, "D": 1}     # potential points by grade
MAX_CUSTOMERS, MAX_PINS = 50_000, 5_000


def prepare(customers, reps, pins=None):
    """Validate and complete the data. Optional columns: region, visits, potential (customers),
    region (reps), and the whole PIN file (built from the customers if missing)."""
    c, r = customers.copy(), reps.copy()
    for df in (c, r):
        df.columns = [str(x).strip().lower() for x in df.columns]
    for name, df, need in (("customers", c, CUSTOMER_COLS), ("reps", r, REP_COLS)):
        miss = [x for x in need if x not in df.columns]
        if miss:
            raise DataError(f"The {name} file is missing column(s): {', '.join(miss)}. "
                            f"Required columns: {', '.join(need)}.")
        if df.empty:
            raise DataError(f"The {name} file has no rows.")
    for col in ("customer_id", "current_owner", "city"):
        c[col] = c[col].astype(str).str.strip()
    for col in ("rep_id", "name", "city", "area_manager"):
        r[col] = r[col].astype(str).str.strip()
    c["grade"] = c["grade"].astype(str).str.strip().str.upper()
    bad = ~c.grade.isin(VISITS)
    if bad.any():
        raise DataError(f"{int(bad.sum())} customer(s) have a grade other than A, B, C or D "
                        f"(for example {c.loc[bad, 'grade'].iloc[0]!r}).")
    for df, cols in ((c, ["pin", "lat", "lon"]), (r, ["home_lat", "home_lon"])):
        for col in cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if df[col].isna().any():
                raise DataError(f"Column '{col}' has empty or non-numeric values.")
    for df, la, lo in ((c, "lat", "lon"), (r, "home_lat", "home_lon")):
        if not (df[la].between(-90, 90).all() and df[lo].between(-180, 180).all()):
            raise DataError("Latitude must be between -90 and 90 and longitude between -180 and 180. "
                            "Check the lat and lon columns are not swapped.")
    c["pin"] = c["pin"].astype(int)
    if c.customer_id.duplicated().any():
        raise DataError("customer_id must be unique: some IDs appear more than once.")
    if r.rep_id.duplicated().any():
        raise DataError("rep_id must be unique: some rep IDs appear more than once.")
    unknown = sorted(set(c.current_owner) - set(r.rep_id))
    if unknown:
        raise DataError(f"current_owner has rep IDs that are not in the reps file: {', '.join(unknown[:5])}.")
    no_rep = sorted(set(c.city) - set(r.city))
    if no_rep:
        raise DataError(f"No rep is based in: {', '.join(no_rep)}. Every customer city needs at least one rep "
                        "with the same city name.")
    if len(c) > MAX_CUSTOMERS:
        raise DataError(f"{len(c):,} customers is more than this demo handles ({MAX_CUSTOMERS:,}).")
    r["region"] = r["region"].astype(str).str.strip() if "region" in r.columns else r["city"]
    c["region"] = c["region"].astype(str).str.strip() if "region" in c.columns else c["city"]
    if "visits" not in c.columns:
        c["visits"] = c.grade.map(VISITS)
    if "potential" not in c.columns:
        c["potential"] = c.grade.map(POTENTIAL)

    if pins is None:
        p = c.groupby("pin").agg(region=("region", "first"), city=("city", "first"),
                                 lat=("lat", "mean"), lon=("lon", "mean")).reset_index()
    else:
        p = pins.copy()
        p.columns = [str(x).strip().lower() for x in p.columns]
        miss = [x for x in ("pin", "lat", "lon") if x not in p.columns]
        if miss:
            raise DataError(f"The PIN file is missing column(s): {', '.join(miss)}.")
        p["pin"] = pd.to_numeric(p["pin"], errors="coerce")
        if p[["pin", "lat", "lon"]].isna().any().any():
            raise DataError("The PIN file has empty or non-numeric pin, lat or lon values.")
        p["pin"] = p["pin"].astype(int)
        absent = sorted(set(c.pin) - set(p.pin))
        if absent:
            raise DataError(f"PIN codes used by customers but missing from the PIN file: "
                            f"{', '.join(map(str, absent[:5]))}.")
        info = c.groupby("pin").agg(region=("region", "first"), city=("city", "first")).reset_index()
        p = p[["pin", "lat", "lon"]].merge(info, on="pin", how="inner")
    if len(p) > MAX_PINS:
        raise DataError(f"{len(p):,} PIN codes is more than this demo handles ({MAX_PINS:,}).")
    return c.reset_index(drop=True), r.reset_index(drop=True), p.reset_index(drop=True)


# ---------- Step 6: baseline = every customer goes to the nearest rep ----------
def nearest_rep(customers, reps):
    d = haversine(customers.lat.values[:, None], customers.lon.values[:, None],
                  reps.home_lat.values[None, :], reps.home_lon.values[None, :])
    return reps.rep_id.values[d.argmin(axis=1)]


# ---------- Step 5: the four metrics ----------
def km_to_owner(customers, reps, owner_col):
    homes = reps.set_index("rep_id").loc[customers[owner_col]]
    return haversine(customers.lat.values, customers.lon.values,
                     homes.home_lat.values, homes.home_lon.values)


def per_rep_table(customers, reps, owner_col):
    """One row per rep (reps with no customers count as zero, not missing)."""
    g = customers.groupby(owner_col).agg(customers=("customer_id", "count"),
                                         workload=("visits", "sum"),
                                         potential=("potential", "sum"))
    return g.reindex(reps.rep_id).fillna(0)


def cv(series):
    """Coefficient of variation = spread / average. 0 = perfectly even."""
    return float(series.std(ddof=0) / series.mean())


def compute_metrics(customers, reps, owner_col, active=None):
    """active = list of rep_ids that count (default: all). Use it when a rep has left."""
    t = per_rep_table(customers, reps, owner_col)
    if active is not None:
        t = t.loc[active]
    km = km_to_owner(customers, reps, owner_col)
    return {
        "Workload spread (CV)": cv(t.workload),
        "Potential spread (CV)": cv(t.potential),
        "Avg travel per visit (km)": float((km * customers.visits).sum() / customers.visits.sum()),
        "Customers keeping their rep (%)": float((customers[owner_col] == customers.current_owner).mean() * 100),
    }


if __name__ == "__main__":
    c, r, _ = load()
    c["baseline_owner"] = nearest_rep(c, r)
    out = pd.DataFrame({"Current (before)": compute_metrics(c, r, "current_owner"),
                        "Nearest-rep baseline": compute_metrics(c, r, "baseline_owner")})
    print(out.round(2))
