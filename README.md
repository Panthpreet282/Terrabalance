# TerraBalance

Fair sales territories for a field team. It groups customers by PIN code, gives each PIN code to exactly one rep,
and balances workload and sales opportunity while keeping drives short and as many customers as possible with
their current rep. Every move comes with a one-sentence reason.

The data is synthetic (generated around Mumbai, Pune and Bengaluru). The results show the method works. They do
not predict gains for a real company.

## Run it (needs Python 3.11 or newer)

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows.  On macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python make_data.py             # creates the data/ folder (same data every time)
streamlit run app.py            # opens http://localhost:8501
```

## What you should see

`python make_data.py` prints: 1500 customers, 24 reps, workload CV 0.37, potential CV 0.40.

The app's four headline cards show (default "Balanced" setting):

| | Today | TerraBalance |
|---|---|---|
| Work unevenness | 0.37 | 0.06 |
| Opportunity unevenness | 0.40 | 0.11 |
| Avg drive per visit (km) | 11.8 | 8.2 |
| Customers keeping their rep | 100% | 61% |

## Using your own data

In the app's left panel choose **Upload my own files** and upload two CSV files (a third is optional).

| File | Required columns | Optional |
|---|---|---|
| Customers | `customer_id`, `pin`, `lat`, `lon`, `grade` (A, B, C or D), `city`, `current_owner` | `region`, `visits`, `potential` |
| Reps | `rep_id`, `name`, `city`, `home_lat`, `home_lon`, `area_manager` | `region` |
| PIN codes | `pin`, `lat`, `lon` | built from the customers if you skip it |

Rules the app checks for you (it shows a plain message if something is wrong):
- `current_owner` must be a `rep_id` from the reps file.
- Every customer city needs at least one rep with the exact same city name.
- Latitude and longitude must be real values (not swapped).
- If `visits` and `potential` are missing they are worked out from the grade (visits 4/3/2/1, potential 8/4/2/1).
- Limits: 50,000 customers and 5,000 PIN codes.

The left panel has small example files to copy the format from.

## Files

| File | Purpose |
|---|---|
| `make_data.py` | Generates customers, reps and PIN codes into `data/` |
| `metrics.py` | Loads data, the nearest-rep baseline, the four metrics |
| `optimiser.py` | PIN roll-up, scoring, optimiser, explanations, star ratings |
| `app.py` | The Streamlit app (7 tabs) |

## Known limitations

Straight-line distance (not roads), whole PIN codes never split, fixed customer grades, synthetic data,
the 85% "keep their rep" goal is not met because it conflicts with fairness, and star ratings rate
territories, not people. The app's last tab lists them all.
