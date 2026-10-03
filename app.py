import colorsys
import hashlib
import io

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

from metrics import DataError, compute_metrics, km_to_owner, load, nearest_rep, per_rep_table, prepare
from optimiser import (DEFAULT_PRESET, PRESETS, RATING_WEIGHTS, TRAVEL_CAP_KM, Problem, Weights,
                       apply_assignment, explain_moves, majority_start, rate_reps, rollup_pins,
                       rollups, solve)

st.set_page_config(page_title="TerraBalance", layout="wide")
EXAMPLE_CUSTOMERS = ("customer_id,pin,lat,lon,grade,city,current_owner\n"
                     "C001,400001,19.0760,72.8777,A,Mumbai,R1\nC002,400002,19.0810,72.8850,C,Mumbai,R1\n"
                     "C003,400003,19.1000,72.8600,B,Mumbai,R2\n")
EXAMPLE_REPS = ("rep_id,name,city,home_lat,home_lon,area_manager\n"
                "R1,Asha Rao,Mumbai,19.0800,72.8800,Vikram Shah\nR2,Kabir Shah,Mumbai,19.1100,72.8600,Vikram Shah\n")
st.sidebar.header("Data")
source = st.sidebar.radio("Which data?", ["Built-in sample", "Upload my own files"],
                          help="The built-in sample is synthetic. You can upload your own CSV files instead.")
UP_C = UP_R = UP_P = None
if source == "Upload my own files":
    UP_C = st.sidebar.file_uploader("Customers (CSV)", type="csv")
    UP_R = st.sidebar.file_uploader("Reps (CSV)", type="csv")
    UP_P = st.sidebar.file_uploader("PIN codes (CSV, optional)", type="csv")
    st.sidebar.caption("Customers need: customer_id, pin, lat, lon, grade (A-D), city, current_owner. "
                       "Reps need: rep_id, name, city, home_lat, home_lon, area_manager. "
                       "Optional: region, and a PIN file (otherwise built from customers).")
    st.sidebar.download_button("Example customers file", EXAMPLE_CUSTOMERS, "customers_example.csv", "text/csv")
    st.sidebar.download_button("Example reps file", EXAMPLE_REPS, "reps_example.csv", "text/csv")
DATA_INTRO = ("The data is made up (synthetic) around Mumbai, Pune and Bengaluru, so it shows the method works, "
              "not what a real company would gain." if source == "Built-in sample"
              else "You are looking at your own uploaded data.")
DATA_LIMIT = ("**Synthetic data.** PIN centroids and customers are generated around real city centres. The results "
              "show the method works. They do not predict gains for a real company." if source == "Built-in sample"
              else "**Your data is used as given.** Results are only as good as the locations, grades and current "
                   "owners in your files, so check them before relying on the plan.")

st.title("TerraBalance")
st.markdown("##### Fair sales territories, designed by software")
st.info(
    "**The problem:** a company has field sales reps and many customers. When territories are drawn by hand, some reps "
    "get far too much work, others sit half-idle, and customers are visited by reps who live far away.\n\n"
    "**What this app does:** it groups customers by their postal PIN code and gives each PIN code to exactly one rep, "
    "so work and opportunity are shared fairly, drives are short, and as few customers as possible have to switch rep.\n\n"
    "**How to read it:** the four cards below compare the new plan with today. Then go through the numbered tabs, "
    "left to right. " + DATA_INTRO)
with st.expander("Plain-English glossary (open this if a word is unclear)"):
    st.markdown("""
- **PIN code**: the 6-digit postal code on an Indian address. We never split one between two reps.
- **Territory**: all the PIN codes (and so customers) that one rep looks after.
- **Visits per month**: how much work a customer needs. Grade A = 4 visits, B = 3, C = 2, D = 1.
- **Potential points**: how much sales opportunity a customer offers. Grade A = 8, B = 4, C = 2, D = 1.
- **Unevenness**: how different the reps' loads are from each other. 0 means perfectly equal. Lower is fairer.
- **Switching rep**: a customer who will be looked after by a different rep than today.
- **Today (before)**: the current alignment. **Nearest rep**: a simple method that gives each customer to the closest rep.
  **TerraBalance (after)**: the new plan from this app.
- **Area Manager / National Manager**: the levels above reps. Their totals are the sum of the reps below them.
- **Stars**: a 1 to 5 rating of how fair and workable a rep's territory is. It rates the territory, not the person.
""")


# ------------------------------------------------------------------ data + solver (cached)
@st.cache_resource(show_spinner="Preparing the data...")
def get_problem(cb, rb, pb):
    """cb, rb, pb = uploaded file bytes, or None to use the built-in sample."""
    if cb is None:
        c, r, p = load()
    else:
        c, r = pd.read_csv(io.BytesIO(cb)), pd.read_csv(io.BytesIO(rb))
        p = pd.read_csv(io.BytesIO(pb)) if pb else None
    c, r, p = prepare(c, r, p)
    c["baseline_owner"] = nearest_rep(c, r)
    pins, cnt = rollup_pins(c, r, p)
    return c, r, pins, Problem(c, r, pins, cnt)


if source == "Upload my own files" and not (UP_C and UP_R):
    st.info("Upload your customers file and reps file in the left panel to begin. "
            "The example files there show the format.")
    st.stop()
CB = UP_C.getvalue() if UP_C else None
RB = UP_R.getvalue() if UP_R else None
PB = UP_P.getvalue() if UP_P else None
DATA_ID = "builtin" if CB is None else hashlib.md5(CB + RB + (PB or b"")).hexdigest()[:12]
try:
    customers, reps, pins, prob = get_problem(CB, RB, PB)
except (DataError, pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as e:
    st.error(f"Could not use these files. {e}")
    st.stop()
CITIES = sorted(customers.city.unique())
AREA_OPTIONS = ["All cities"] + CITIES + sorted(set(customers.region.unique()) - set(CITIES))
MAJ_KEEP = 100 * prob.cnt[np.arange(len(prob.pins)), prob.cnt.argmax(axis=1)].sum() / prob.N
rep_name = reps.set_index("rep_id").name.to_dict()
rep_xy = reps.set_index("rep_id")[["home_lon", "home_lat"]]


@st.cache_data
def run_solve(wt, leaver=None, data_id=""):
    """wt = (travel, workload, potential, relationship). With a leaver: re-align from the plan."""
    w = Weights(*wt)
    a, start, hist = solve(prob, w)
    if leaver is None:
        return a, start, hist
    a2, _, hist2 = solve(prob, w, start=a, unavailable={leaver})
    return a2, a, hist2


# ------------------------------------------------------------------ sidebar: weights
st.sidebar.header("What should the plan care about most?")
PRESET_HELP = {"Stability first": "Change as few customers' reps as possible.",
               "Gentle": "A light touch: modest balancing.",
               "Balanced (recommended)": "A good mix: fair loads, shorter drives, most relationships kept.",
               "Balance first": "Make workloads as equal as possible, even if more customers switch rep.",
               "Travel first": "Shortest drives, even if many customers switch rep."}
preset = st.sidebar.selectbox("Priority setting", list(PRESETS), index=list(PRESETS).index(DEFAULT_PRESET))
st.sidebar.caption(PRESET_HELP[preset])
pw = PRESETS[preset]
if st.sidebar.checkbox("Set my own weights (advanced)"):
    wt = (st.sidebar.slider("Short drives", 0.0, 5.0, pw.travel, 0.5, key=f"t{preset}"),
          st.sidebar.slider("Equal workload", 0.0, 5.0, pw.workload, 0.5, key=f"w{preset}"),
          st.sidebar.slider("Equal opportunity", 0.0, 5.0, pw.potential, 0.5, key=f"p{preset}"),
          st.sidebar.slider("Keep customers with their rep", 0.0, 5.0, pw.relationship, 0.5, key=f"r{preset}"))
else:
    wt = (pw.travel, pw.workload, pw.potential, pw.relationship)
W = Weights(*wt)

assign, start, history = run_solve(wt, None, DATA_ID)
plan = apply_assignment(customers, prob, assign)           # adds column new_owner
OWNER_COL = {"Today (before)": "current_owner", "Simple method: nearest rep": "baseline_owner",
             "TerraBalance (after)": "new_owner"}

m_cur = compute_metrics(plan, reps, "current_owner")
m_near = compute_metrics(plan, reps, "baseline_owner")
m_new = compute_metrics(plan, reps, "new_owner")

DISPLAY = {"Workload spread (CV)": "Work unevenness",
           "Potential spread (CV)": "Opportunity unevenness",
           "Avg travel per visit (km)": "Avg drive per visit (km)",
           "Customers keeping their rep (%)": "Customers keeping their rep (%)"}
HELP = {"Work unevenness": "How different reps' monthly visit loads are. 0 = perfectly equal. Lower is better.",
        "Opportunity unevenness": "How different reps' sales potential is. 0 = perfectly equal. Lower is better.",
        "Avg drive per visit (km)": "Straight-line distance from the rep's home to the customer, averaged over visits. Lower is better.",
        "Customers keeping their rep (%)": "Share of customers who stay with the rep they have today. Higher means less disruption."}
st.subheader("Headline results: the new plan compared with today")
cols = st.columns(4)
for col, name in zip(cols, m_cur):
    lab = DISPLAY[name]
    col.metric(lab, f"{m_new[name]:.2f}" if "keeping" not in name and "km" not in name else f"{m_new[name]:.1f}",
               f"{m_new[name] - m_cur[name]:+.2f} vs today", help=HELP[lab],
               delta_color="inverse" if "keeping" not in name else "normal")
st.caption("Big number = new plan. Small number = change from today. Green means better. "
           "Hover the little question mark on a card for its meaning.")


# ------------------------------------------------------------------ map helpers
def make_palette(ids):
    out = {}
    for i, rid in enumerate(ids):
        r, g, b = colorsys.hsv_to_rgb((i * 0.618034) % 1, 0.75, 0.95)
        out[rid] = [int(r * 255), int(g * 255), int(b * 255)]
    return out


def convex_hull(points):
    pts = sorted(set(points))
    if len(pts) < 3:
        return []
    cross = lambda o, a, b: (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


PALETTE = make_palette(reps.rep_id.tolist())


def build_deck(df, area_name, owner_col, ref_col, rep_df, o):
    cu = df[(df.city == area_name) | (df.region == area_name)].copy()
    rp = rep_df[(rep_df.city == area_name) | (rep_df.region == area_name)].copy()
    cu["owner"] = cu[owner_col]
    cu["transfer"] = cu[owner_col] != cu[ref_col]
    cu["km"] = km_to_owner(cu, reps, owner_col)
    fid = o["focus"]
    in_focus = (cu.owner == fid) if fid else pd.Series(True, index=cu.index)
    fills = []
    for rid, t, f in zip(cu.owner, cu.transfer, in_focus):
        if not f:
            fills.append([150, 150, 150, 35])
        elif o["red"]:
            fills.append([220, 20, 20, 255] if t else [120, 120, 120, 90])
        else:
            fills.append(PALETTE[rid] + [230])
    cu["fill"] = fills
    cu["tip"] = ("Customer " + cu.customer_id + " | PIN " + cu.pin.astype(str) + " | grade " + cu.grade
                 + "\nRep in this view: " + cu.owner.map(rep_name)
                 + "\nRep today: " + cu.current_owner.map(rep_name))
    rp["fill"] = [PALETTE[r] + [255] for r in rp.rep_id]
    rp["tip"] = "Rep " + rp.name + " (" + rp.rep_id + ") | AM " + rp.area_manager

    layers = []
    if o["hulls"]:
        hulls = []
        for rid, grp in cu.groupby("owner"):
            if fid and rid != fid:
                continue
            poly = convex_hull(list(zip(grp.lon, grp.lat)))
            if poly:
                hulls.append({"polygon": poly, "fill": PALETTE[rid] + [22], "line": PALETTE[rid] + [170],
                              "tip": f"Territory of {rep_name[rid]}"})
        layers.append(pdk.Layer("PolygonLayer", hulls, get_polygon="polygon", get_fill_color="fill",
                                get_line_color="line", stroked=True, filled=True,
                                line_width_min_pixels=1, pickable=False))
    if o["lines"]:
        tr = cu[cu.transfer & (cu.km > o["min_km"]) & in_focus]
        ends = rep_xy.loc[tr.owner]
        lines = pd.DataFrame({"src": list(zip(tr.lon, tr.lat)), "dst": list(zip(ends.home_lon, ends.home_lat))})
        layers.append(pdk.Layer("LineLayer", lines, get_source_position="src", get_target_position="dst",
                                get_color=[200, 30, 30, 80], get_width=1, width_min_pixels=1))
    layers.append(pdk.Layer("ScatterplotLayer", cu, get_position=["lon", "lat"], get_fill_color="fill",
                            get_radius=80, radius_min_pixels=o["dot"], pickable=True))
    layers.append(pdk.Layer("ScatterplotLayer", rp, get_position=["home_lon", "home_lat"],
                            get_fill_color="fill", get_line_color=[255, 255, 255], stroked=True,
                            line_width_min_pixels=3, get_radius=500, radius_min_pixels=10, pickable=True))
    span = max(cu.lat.max() - cu.lat.min(), cu.lon.max() - cu.lon.min(), 0.05)
    zoom = float(np.clip(np.log2(1125 / span) - 0.9, 6, 12))
    state = pdk.ViewState(latitude=float(cu.lat.mean()), longitude=float(cu.lon.mean()), zoom=zoom)
    return (pdk.Deck(layers=layers, initial_view_state=state, map_style="light", tooltip={"text": "{tip}"}),
            len(cu), len(rp), int(cu.transfer.sum()))


def render_maps(df, owner_col, ref_col, area, o, rep_df=None, height=560):
    rep_df = reps if rep_df is None else rep_df
    areas = CITIES if area.startswith("All") else [area]
    tot = tr = 0
    for k in range(0, len(areas), 2):                      # two maps per row
        row = areas[k:k + 2]
        for slot, a in zip(st.columns(len(row)), row):
            deck, n_c, n_r, n_tr = build_deck(df, a, owner_col, ref_col, rep_df, o)
            tot += n_c
            tr += n_tr
            with slot:
                st.markdown(f"**{a}** - {n_c} customers, {n_r} reps")
                st.pydeck_chart(deck, height=height)
    return tot, tr


# ------------------------------------------------------------------ tabs
tabs = st.tabs(["1 The map", "2 Before vs after", "3 Each rep and stars", "4 Team totals",
                "5 Why a PIN moved", "6 If a rep leaves", "7 Limitations"])

# ---------------------------------------------------------------- Map
with tabs[0]:
    st.subheader("1. The map: which rep looks after which customers")
    st.markdown("Each dot is a customer, coloured by its rep. Big circles with a white rim are the reps' homes. "
                "Shaded areas are territories. **Switch between 'Today (before)' and 'TerraBalance (after)'** to see the "
                "tangled territories become compact ones. Hover any dot for details.")
    c1, c2, c3, c4 = st.columns([2, 2, 2, 3])
    view = c1.radio("Which plan to show", list(OWNER_COL), index=2)
    area = c2.selectbox("Area", AREA_OPTIONS)
    focus_label = c3.selectbox("Focus on one rep", ["None"] + [f"{n} ({i})" for i, n in rep_name.items()])
    red = c4.checkbox("Show customers who switch rep in red", False)
    hulls = c4.checkbox("Territory outlines", True)
    lines = c4.checkbox("Draw lines from switching customers to their rep", False)
    min_km = c4.slider("Only draw lines longer than (km)", 0, 40, 15, disabled=not lines)
    dot = c4.slider("Dot size", 1, 6, 2)
    opts = dict(focus=None if focus_label == "None" else focus_label.split("(")[-1].rstrip(")"),
                red=red, hulls=hulls, lines=lines, min_km=min_km, dot=dot)
    tot, tr = render_maps(plan, OWNER_COL[view], "current_owner", area, opts)
    st.caption(f"In this view, {tr} of {tot} customers ({tr / tot:.0%}) have a different rep from today. "
               "Tip: 'Focus on one rep' greys out everyone else so a single territory is easy to see.")

# ---------------------------------------------------------------- Compare
with tabs[1]:
    st.subheader("2. Before vs after: did it work?")
    st.markdown("Three ways of assigning customers, side by side. For the first three rows **lower is better**; "
                "for the last row higher means fewer customers have to switch rep.")
    comp = pd.DataFrame({"Today (before)": m_cur, "Simple method (nearest rep)": m_near,
                         f"TerraBalance (after)": m_new}).rename(index=DISPLAY)
    st.dataframe(comp.style.format("{:.2f}"), width="stretch")

    orphans = int(plan.new_owner.isna().sum())
    viol = prob.violations(assign)
    pct = lambda new, old: 100 * (1 - new / old)
    keys = list(m_cur)
    targets = pd.DataFrame([
        ["Workload spread at least 50% lower than current", f"{pct(m_new[keys[0]], m_cur[keys[0]]):.0f}% lower",
         pct(m_new[keys[0]], m_cur[keys[0]]) >= 50],
        ["Potential spread at least 40% lower than current", f"{pct(m_new[keys[1]], m_cur[keys[1]]):.0f}% lower",
         pct(m_new[keys[1]], m_cur[keys[1]]) >= 40],
        ["Travel no worse than 10% above nearest-rep",
         f"{m_new[keys[2]]:.1f} km vs limit {1.1 * m_near[keys[2]]:.1f} km", m_new[keys[2]] <= 1.1 * m_near[keys[2]]],
        ["85% or more customers keep their rep", f"{m_new[keys[3]]:.0f}%", m_new[keys[3]] >= 85],
        ["No orphan customers, no rule violations", f"{orphans} orphans, {viol} violations", orphans == 0 and viol == 0]],
        columns=["Target from the plan", "Result", "Met?"])
    targets["Met?"] = targets["Met?"].map({True: "Yes", False: "No"})
    st.markdown("**Did we hit the goals we set at the start? (honest scorecard)**")
    st.dataframe(targets, width="stretch", hide_index=True)
    st.info(f"Trade-off: against the current alignment, balance improves a lot and travel falls "
            f"{pct(m_new[keys[2]], m_cur[keys[2]]):.0f}%. Against nearest-rep, balance is far better but drives are "
            f"{m_new[keys[2]] / m_near[keys[2]] - 1:.0%} longer, and {100 - m_new[keys[3]]:.0f}% of customers change rep. "
            "Moving the weights slides along this trade-off (table below).")

    st.subheader("What happens if we change what matters most?")
    st.caption("Each row is a different priority setting. Moving down the list trades shorter drives against "
               "keeping customers with their rep.")
    rows = {}
    for name, p in PRESETS.items():
        a_, s_, h_ = run_solve((p.travel, p.workload, p.potential, p.relationship), None, DATA_ID)
        mm = compute_metrics(apply_assignment(customers, prob, a_), reps, "new_owner")
        mm["PINs moved"] = int((a_ != s_).sum())
        rows[name] = mm
    pt = pd.DataFrame(rows).T.rename(columns=DISPLAY).rename(columns={"PINs moved": "PIN codes moved"})
    st.dataframe(pt.style.format("{:.2f}", subset=list(DISPLAY.values())).format("{:.0f}", subset=["PIN codes moved"]),
                 width="stretch")

    left, right = st.columns(2)
    with left:
        st.markdown("**Optimiser progress: the score drops, then flattens (lower is better)**")
        st.line_chart(pd.DataFrame({"score": [float(x) for x in history]}, index=range(len(history))))
    with right:
        st.markdown("**Visits per month for each rep (taller bar = busier rep)**")
        st.bar_chart(pd.DataFrame({
            "Today": per_rep_table(plan, reps, "current_owner").workload,
            "Nearest rep": per_rep_table(plan, reps, "baseline_owner").workload,
            "TerraBalance": per_rep_table(plan, reps, "new_owner").workload}))

    with st.expander("Proof the scoring works: a bad plan scores worse than a good one"):
        near_a = np.where(prob.allowed(), prob.D, np.inf).argmin(axis=1)
        sc = {}
        for lab, arr in [("Today (each PIN with its main rep)", majority_start(prob)),
                         ("Nearest rep", near_a), ("TerraBalance", assign)]:
            tot_s, terms = prob.score(arr, W)
            sc[lab] = {"Total score (lower = better)": tot_s, **terms}
        st.dataframe(pd.DataFrame(sc).T.round(3).rename(columns={
            "travel": "Drive term", "workload_cv": "Work unevenness", "potential_cv": "Opportunity unevenness",
            "relationship": "Share switching rep"}), width="stretch")
        st.caption("Total score = (drive term) + (work unevenness) + (opportunity unevenness) + (share switching rep), "
                   "each multiplied by its weight. Hard rules: a PIN only goes to a rep in the same city, "
                   "and never to a rep who is unavailable.")

# ---------------------------------------------------------------- Reps and stars
with tabs[2]:
    st.subheader("3. Each rep: workload, fairness and a 1 to 5 star territory rating")
    st.markdown("One row per rep. **Alerts** warn about overloaded, underloaded, high or low opportunity, or very "
                "spread-out territories. **Stars** rate how fair and workable the territory is. Open 'How the star "
                "rating works' for the exact rules.")
    view_r = st.radio("Which plan to show", list(OWNER_COL), index=2, horizontal=True, key="rv")
    rt = rate_reps(plan, reps, OWNER_COL[view_r])
    d = rt.reset_index()[["rep_id", "name", "area_manager", "customers", "visits", "potential",
                          "avg_km_per_visit", "had_before", "kept", "gained", "lost", "flags", "rating", "stars"]]
    d.columns = ["Rep ID", "Name", "Area Manager", "Customers", "Visits per month", "Potential points",
                 "Avg drive per visit (km)", "Customers before", "Kept", "New to this rep", "Moved away",
                 "Alerts", "Rating (1-5)", "Stars"]
    st.dataframe(d.style.format({"Avg drive per visit (km)": "{:.1f}", "Rating (1-5)": "{:.1f}",
                                 "Customers": "{:.0f}", "Visits per month": "{:.0f}", "Potential points": "{:.0f}",
                                 "Customers before": "{:.0f}", "Kept": "{:.0f}", "New to this rep": "{:.0f}",
                                 "Moved away": "{:.0f}"}),
                 width="stretch", hide_index=True, height=450)
    st.caption(f"Average rating in this view: {rt.rating.mean():.2f} stars. "
               "Alerts: overloaded / underloaded = visits more than 20% above or below the team average; "
               "high / low potential = same rule on potential points; sprawling = average drive more than 1.5 times the typical rep.")
    with st.expander("How the star rating works"):
        st.markdown("A rep's territory is scored 0-100 and converted to 1-5 stars (1 + 4 x score / 100).")
        st.dataframe(pd.DataFrame({"Component": list(RATING_WEIGHTS), "Points": list(RATING_WEIGHTS.values()),
                                   "Full marks when": ["visits equal the team average (0 at 50% off)",
                                                       "potential equals the team average (0 at 50% off)",
                                                       f"drive per visit is 0 km (0 at {TRAVEL_CAP_KM:.0f} km or more)",
                                                       "the rep keeps all customers they had before",
                                                       "no new customers to win over"]}),
                     width="stretch", hide_index=True)
        st.caption("This rates the territory a rep is handed (fair, close, familiar), not their performance as a person. "
                   "In the Current view nobody has transfers, so the last two components are full marks by definition.")
    pick = st.selectbox("See how this rep's stars were earned", [f"{n} ({i})" for i, n in rep_name.items()])
    rid = pick.split("(")[-1].rstrip(")")
    sub = rt.loc[rid, [c for c in rt.columns if c.startswith("s_")]].astype(float)
    sub.index = [i[2:] for i in sub.index]
    st.bar_chart(sub * 100)
    st.caption(f"{rep_name[rid]}: {rt.loc[rid, 'rating']:.1f} stars. Bars show each component out of 100.")

# ---------------------------------------------------------------- Roll-ups
with tabs[3]:
    st.subheader("4. Team totals: rep, then Area Manager, then National")
    st.markdown("The same numbers at each level of the company. The checks at the bottom prove the levels add up.")
    view_u = st.radio("Which plan to show", list(OWNER_COL), index=2, horizontal=True, key="uv")
    am, nat, checks = rollups(rate_reps(plan, reps, OWNER_COL[view_u]))
    RN = {"reps": "Reps", "customers": "Customers", "visits": "Visits per month", "potential": "Potential points",
          "avg_km_per_visit": "Avg drive per visit (km)", "workload_cv": "Work unevenness",
          "customers_kept_%": "Customers keeping their rep (%)", "avg_stars": "Average stars"}
    am, nat = am.rename(columns=RN), nat.rename(columns=RN).rename(index={"National": "National total"})
    fmt = {"Reps": "{:.0f}", "Customers": "{:.0f}", "Visits per month": "{:.0f}", "Potential points": "{:.0f}",
           "Avg drive per visit (km)": "{:.1f}", "Work unevenness": "{:.2f}",
           "Customers keeping their rep (%)": "{:.0f}", "Average stars": "{:.2f}"}
    st.markdown("**Area Managers**")
    st.dataframe(am.style.format(fmt), width="stretch")
    st.markdown("**National Manager**")
    st.dataframe(nat.style.format(fmt), width="stretch")
    st.markdown("**Do the levels add up? (each should say yes)**  " + "  |  ".join(f"{k}: {'yes' if v else 'NO'}" for k, v in checks.items()))

# ---------------------------------------------------------------- Why it moved
with tabs[4]:
    st.subheader("5. Why did this PIN code change rep?")
    st.markdown("Pick any PIN code that changed hands and the app explains the decision in one sentence, using the numbers.")
    ex = explain_moves(prob, start, assign, customers)
    if ex.empty:
        st.write("No PIN moved with these weights.")
    else:
        pin_pick = st.selectbox("PIN code that changed rep", ex.pin.tolist(),
                                format_func=lambda p: f"{p} ({ex.set_index('pin').loc[p, 'customers']} customers)")
        st.success(ex.set_index("pin").loc[pin_pick, "reason"])
        info = pins.set_index("pin").loc[pin_pick]
        st.caption(f"Today: {info.customers} customers, owners {info.owner_mix}; majority owner {info.majority_owner} "
                   f"holds {info.majority_share:.0%}.")
        st.markdown(f"**All {len(ex)} PIN codes that changed rep**")
        st.dataframe(ex.drop(columns="reason").rename(columns={
            "pin": "PIN code", "customers": "Customers", "from_rep": "Old rep", "to_rep": "New rep",
            "km_before": "Drive before (km)", "km_after": "Drive after (km)",
            "from_load_before": "Old rep load before", "from_load_after": "Old rep load after",
            "to_load_before": "New rep load before", "to_load_after": "New rep load after",
            "already_theirs": "Customers already with new rep"}), width="stretch", hide_index=True)
        st.download_button("Download all reasons (CSV)", ex.to_csv(index=False), "pin_moves.csv", "text/csv")

# ---------------------------------------------------------------- Rep leaves
with tabs[5]:
    st.subheader("6. What if a rep leaves?")
    st.markdown("Choose a rep who becomes unavailable. Only nearby territories are re-drawn, so disruption stays small, "
                "and no customer is left without a rep.")
    city_size = reps.groupby("city").rep_id.transform("count")
    eligible = reps[city_size > 1]                      # a city's only rep cannot be covered by anyone
    if eligible.empty:
        st.info("Every city in this data has only one rep, so there is nobody to cover for a leaver.")
        st.stop()
    leaver_pick = st.selectbox("Rep who becomes unavailable", [f"{r.name} ({r.rep_id})" for r in eligible.itertuples()])
    leaver = leaver_pick.split("(")[-1].rstrip(")")
    a2, a_plan, _ = run_solve(wt, leaver, DATA_ID)
    after = apply_assignment(plan, prob, a2, col="after_owner")
    active = [r for r in reps.rep_id if r != leaver]
    m_after = compute_metrics(after, reps, "after_owner", active=active)
    m_plan = compute_metrics(after, reps, "new_owner")
    orph = int((after.after_owner == leaver).sum()) + int(after.after_owner.isna().sum())
    changed_cust = int((after.after_owner != after.new_owner).sum())
    cA, cB, cC = st.columns(3)
    cA.metric("Orphan customers", orph)
    cB.metric("PIN codes that moved", int((a2 != a_plan).sum()))
    cC.metric("Customers who get a new rep", changed_cust)
    st.dataframe(pd.DataFrame({"Plan (everyone available)": m_plan, "After the rep leaves": m_after})
                 .style.format("{:.2f}"), width="stretch")
    rt_b = rate_reps(after, reps, "new_owner")
    rt_a = rate_reps(after, reps, "after_owner", active=active)
    chg = pd.DataFrame({"name": rt_a.name, "visits before": rt_b.visits, "visits after": rt_a.visits,
                        "stars before": rt_b.rating, "stars after": rt_a.rating})
    chg["visit change"] = chg["visits after"] - chg["visits before"]
    chg = chg[(chg["visit change"] != 0)].sort_values("visit change")
    st.markdown(f"**Reps who absorb the work ({len(chg) - 1} others affected, out of 23)**")
    st.dataframe(chg.style.format("{:.0f}", subset=["visits before", "visits after", "visit change"])
                 .format("{:.1f}", subset=["stars before", "stars after"]), width="stretch")
    st.caption("Only PINs near the gap are expected to move. The remaining 22+ territories stay as planned.")
    st.markdown("**Map after the re-alignment (red = customers who moved vs the plan)**")
    city = reps.set_index("rep_id").loc[leaver, "city"]
    o2 = dict(focus=None, red=True, hulls=True, lines=False, min_km=15, dot=3)
    render_maps(after, "after_owner", "new_owner", city, o2, rep_df=reps[reps.rep_id != leaver], height=520)

# ---------------------------------------------------------------- Limitations
with tabs[6]:
    st.subheader("7. Limitations: what this demo cannot do (yet)")
    st.markdown("""
1. **Straight-line distance, not road time.** Hills, rivers and traffic can make a close PIN a long drive. A road
   travel-time matrix (for example from OSRM) for one city would be the next step.
2. **Whole-PIN assignment.** A PIN is never split, so one heavy PIN can leave some imbalance. It also caps how many
   relationships can be kept: even if every PIN stays with its majority owner, only about MAJKEEP% of customers keep their rep.
3. **Potential grades are static.** A, B, C and D ignore growth, churn and seasonality.
4. DATALIMIT
5. **Targets are not all met.** Balance targets are beaten by a wide margin. The relationship target (85%) and the travel
   target (within 10% of nearest-rep) are not, because they pull against balance. The weights show that trade-off openly.
6. **Star ratings rate territories, not people.** They reflect fairness, travel and disruption of a territory. They are
   not a performance review and should not be used as one.
7. **Local search is not guaranteed optimal.** Moving one PIN at a time can stop at a good but not perfect answer.
   There is no 120% capacity cap yet. Balance comes only from the score.
""".replace("MAJKEEP", f"{MAJ_KEEP:.0f}").replace("DATALIMIT", DATA_LIMIT))
