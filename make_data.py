"""TerraBalance - Step 3: generate the synthetic market.

Run from the project root:  python make_data.py
Writes data/customers.csv, data/reps.csv, data/pins.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)  # fixed seed = same data every run
OUT = Path("data")
OUT.mkdir(exist_ok=True)

VISITS = {"A": 4, "B": 3, "C": 2, "D": 1}      # visits per month
POTENTIAL = {"A": 8, "B": 4, "C": 2, "D": 1}   # potential points
GRADES = ["A", "B", "C", "D"]

# 4 regions = 4 Area Managers, 6 reps each = 24 reps, 1500 customers.
# Centres are real city locations; the PINs around them are generated.
REGIONS = [
    dict(region="Mumbai", city="Mumbai-Pune", center=(19.076, 72.878), sigma=0.09,
         n_pins=55, n_cust=400, pin0=400001, am="Rohan Mehta"),
    dict(region="Pune", city="Mumbai-Pune", center=(18.520, 73.857), sigma=0.08,
         n_pins=45, n_cust=350, pin0=411001, am="Neha Kulkarni"),
    dict(region="Bengaluru North", city="Bengaluru", center=(13.040, 77.590), sigma=0.07,
         n_pins=40, n_cust=375, pin0=560001, am="Arjun Reddy"),
    dict(region="Bengaluru South", city="Bengaluru", center=(12.910, 77.600), sigma=0.07,
         n_pins=40, n_cust=375, pin0=560041, am="Kavya Nair"),
]
REPS_PER_REGION = 6

FIRST = ["Aarav", "Vivaan", "Aditya", "Ishaan", "Rahul", "Karan", "Siddharth", "Rohit",
         "Ananya", "Diya", "Priya", "Sneha", "Pooja", "Meera", "Isha", "Kriti",
         "Vikram", "Amit", "Suresh", "Manoj", "Deepa", "Lakshmi", "Farhan", "Zoya"]
LAST = ["Sharma", "Patil", "Iyer", "Gupta", "Joshi", "Rao", "Shah", "Naidu",
        "Desai", "Kapoor", "Menon", "Singh", "Bhat", "Pillai", "Khan", "Verma"]


def haversine(lat1, lon1, lat2, lon2):
    """Straight-line distance in km. Works on numpy arrays."""
    p = np.pi / 180
    a = (np.sin((lat2 - lat1) * p / 2) ** 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2)
    return 12742 * np.arcsin(np.sqrt(a))


pin_rows, rep_rows, cust_rows = [], [], []
rep_counter = 0

for reg in REGIONS:
    # ---- PINs: centroids scattered around the city centre ----
    n = reg["n_pins"]
    lat = reg["center"][0] + rng.normal(0, reg["sigma"], n)
    lon = reg["center"][1] + rng.normal(0, reg["sigma"], n)
    pins = reg["pin0"] + np.arange(n)
    affluence = rng.beta(2, 2, n)            # drives how many A/B customers a PIN has

    # uneven customer counts per PIN (a few dense PINs, many thin ones)
    w = rng.lognormal(0, 0.7, n)
    counts = rng.multinomial(reg["n_cust"], w / w.sum())

    # ---- Reps: homes sit in random PINs of the region ----
    home_idx = rng.choice(n, REPS_PER_REGION, replace=False)
    r_ids, r_lat, r_lon = [], [], []
    for i in home_idx:
        rep_counter += 1
        rid = f"R{rep_counter:02d}"
        name = f"{FIRST[rep_counter - 1]} {LAST[rng.integers(len(LAST))]}"
        hl = lat[i] + rng.normal(0, 0.01)
        ho = lon[i] + rng.normal(0, 0.01)
        rep_rows.append(dict(rep_id=rid, name=name, city=reg["city"], region=reg["region"],
                             home_lat=hl, home_lon=ho, area_manager=reg["am"]))
        r_ids.append(rid); r_lat.append(hl); r_lon.append(ho)
    r_lat, r_lon = np.array(r_lat), np.array(r_lon)

    # some reps historically hoarded accounts -> built-in imbalance
    rep_weight = rng.lognormal(0, 0.3, REPS_PER_REGION)

    # ---- Customers, PIN by PIN ----
    for j in range(n):
        d = haversine(lat[j], lon[j], r_lat, r_lon)      # km from this PIN to each rep
        # today's owner: usually a nearby rep, tilted by who hoards accounts
        p = np.exp(-d / 15) * rep_weight
        pin_owner = rng.choice(REPS_PER_REGION, p=p / p.sum())
        far = np.where(d > np.median(d))[0]              # the farther half of reps

        base = np.array([0.10, 0.25, 0.35, 0.30])
        tilt = np.exp(np.array([1.5, 0.5, -0.3, -1.2]) * (affluence[j] - 0.5) * 3)
        gp = base * tilt
        gp /= gp.sum()

        for _ in range(counts[j]):
            g = rng.choice(GRADES, p=gp)
            owner = pin_owner
            if rng.random() < 0.15:                      # 15% handed to a far-away rep
                owner = rng.choice(far)
            cust_rows.append(dict(
                pin=int(pins[j]), region=reg["region"], city=reg["city"],
                lat=lat[j] + rng.normal(0, 0.01), lon=lon[j] + rng.normal(0, 0.01),
                grade=g, visits=VISITS[g], potential=POTENTIAL[g],
                current_owner=r_ids[owner]))

    for j in range(n):
        pin_rows.append(dict(pin=int(pins[j]), region=reg["region"], city=reg["city"],
                             lat=lat[j], lon=lon[j]))

customers = pd.DataFrame(cust_rows)
customers.insert(0, "customer_id", [f"C{i + 1:04d}" for i in range(len(customers))])
reps = pd.DataFrame(rep_rows)
pins = pd.DataFrame(pin_rows)

# PIN roll-up (Step 4 unit table)
roll = customers.groupby("pin").agg(customers=("customer_id", "count"),
                                    workload=("visits", "sum"),
                                    potential=("potential", "sum")).reset_index()
pins = pins.merge(roll, on="pin", how="left").fillna({"customers": 0, "workload": 0, "potential": 0})

customers.to_csv(OUT / "customers.csv", index=False)
reps.to_csv(OUT / "reps.csv", index=False)
pins.to_csv(OUT / "pins.csv", index=False)

# ---------------- sanity report ----------------
c = customers.merge(reps[["rep_id", "home_lat", "home_lon"]],
                    left_on="current_owner", right_on="rep_id")
c["km"] = haversine(c.lat, c.lon, c.home_lat, c.home_lon)
per_rep = customers.groupby("current_owner").agg(
    customers=("customer_id", "count"), workload=("visits", "sum"), potential=("potential", "sum"))

print(f"Customers: {len(customers)}   Reps: {len(reps)}   PINs: {len(pins)}")
print(f"Workload CV : {per_rep.workload.std(ddof=0) / per_rep.workload.mean():.2f}")
print(f"Potential CV: {per_rep.potential.std(ddof=0) / per_rep.potential.mean():.2f}")
print(f"Avg customer-to-rep distance: {c.km.mean():.1f} km")
print(f"Customers > 15 km from their rep: {(c.km > 15).mean():.0%}")
print(f"Workload per rep: min {per_rep.workload.min()}, max {per_rep.workload.max()}, "
      f"mean {per_rep.workload.mean():.0f}")
print("Files written to data/")
