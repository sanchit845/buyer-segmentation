# 🏠 Buyer Segmentation & Investment Profiling

An unsupervised ML project that clusters 2,000 real-estate buyers into 4 segments using K-Means on engineered transaction features, and exposes the result as an interactive Streamlit dashboard.

Built as an internship project with **Unified Mentor** × **Parcl Co. Limited**.

---

## What it does

Takes 2,000 simulated clients and 10,000 simulated property transactions, merges them on `client_id`, engineers 8 features, runs K-Means + Hierarchical clustering, and ships a 5-tab dashboard (Overview, Investor Insights, Geographic, Segment Insights, Predict) plus a buyer-profile prediction form.

The 4 segments the model converges on:

| Segment | Profile |
|---|---|
| 🏠 First-Time Buyers | ~$252k, low satisfaction (1.7/5), high loan dependency |
| 🏢 Corporate Buyers | ~$260k, very high satisfaction (4.2/5), Apartment-only |
| 🌍 Global Investors | ~$379k, mid satisfaction, **Office-only** (100% commercial) |
| 💎 Luxury Investors | ~$486k, 1,599 sqft avg, highest investment score, no loan needed |

Full per-segment numbers are in `data/processed/cluster_summary.csv`.

---

## Project layout

```
buyer-segmentation/
├── dashboard/app.py              # Streamlit app (entry point)
├── data/raw/                     # clients.csv + properties.csv
├── data/processed/               # Outputs of each notebook
├── models/                       # Saved KMeans, scaler, cluster_mapping
├── notebooks/
│   ├── 01_eda.ipynb              # Clean + merge
│   ├── 02_preprocessing.ipynb    # Feature engineering + encoding
│   ├── 03_clustering.ipynb       # K-Means + Hierarchical + auto-labelling
│   └── 04_business_insights.ipynb  # Segment profiling + recommendations
├── main.py                       # Alternative launcher (optional)
└── requirements.txt
```

The four notebooks are meant to be run in order — each writes the CSVs / PKL files the next one consumes, and the dashboard needs the final outputs of `03_clustering.ipynb`.

---

## How to run it

```bash
git clone <repo-url>
cd buyer-segmentation
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
```

Then either run the notebooks in order (`jupyter notebook`) to regenerate everything from scratch, or skip straight to the dashboard if the processed data and models are already in the repo:

```bash
cd dashboard
streamlit run app.py
# → opens http://localhost:8501
```

## Method (the short version)

1. **Clean & merge** — drop duplicates, parse `sale_price` currency strings, drop properties with no linked client, compute `age` from `date_of_birth`. Merged frame: ~7,300 rows.
2. **Engineer features** — `price_per_sqft`, `investment_score` (weighted price + satisfaction), `loan_indicator` (0/1), `unit_type_encoded` (0/1). Label-encode the categoricals, standard-scale all 8 model features.
3. **Cluster** — sweep K from 2 to 8 with Elbow + Silhouette + Davies-Bouldin. Silhouette plateaus across K=3, 4, 5 (all within ~0.03). I went with **K=4** because it gave the cleanest business personas — K=3 had a vague "middle" segment, K=5 was splitting Corporate Buyers arbitrarily. Hierarchical (Ward linkage) on a sample confirms the same 4-cluster structure.
4. **Auto-label** — rank clusters by a composite of (price, investment_score, floor_area, satisfaction) → names like "Luxury Investors" come from the rank, not manual choice.

The interesting notebook is `03_clustering.ipynb` — that's where the silhouette comparison and the auto-labeller live.

---

## Dashboard

5 tabs, all interactive, with sidebar filters (segment / country / purpose / client type):

- **Overview** — segment pie + counts, purpose/client-type breakdowns
- **Investor Insights** — price-by-segment box, investment-score scatter, loan rates, referral mix
- **Geographic** — buyer counts and avg price by country, country×segment heatmap
- **Segment Insights** — per-segment KPIs and pies; switches to a deep-dive view when you select one segment
- **Predict** — type in a buyer's age, satisfaction, loan, price, area, unit type → predicted segment + match-likelihood bar across all 4

---

## Dependencies

```
pandas, numpy, scikit-learn, scipy
matplotlib, seaborn, plotly
streamlit, joblib
jupyter, ipykernel
```

Everything in `requirements.txt`. No unusual system deps.

---

## Known limitations

- The data is **synthetic** — `clients.csv` and `properties.csv` are simulated, not real Parcl transactions. Treat the segment counts and country distributions as illustrative.
- **No held-out validation.** K-Means was fit on the full set; the Predict tab's "match likelihood" is a softmax over centroid distance, not a calibrated probability.
- **Age and loan rate barely discriminate the segments** (age spread 55.1 → 57.6 yrs). The dashboard's Investor Insights scatter uses satisfaction × investment-score instead, which actually separate the clusters.
- **K=4 was a judgement call**, not a strong signal. Re-run `03_clustering.ipynb` to see the score table — K=3 is a defensible alternative.

Future work: held-out validation, DBSCAN for outlier detection, deploy the dashboard to Streamlit Community Cloud.

---

## Notes

The cleanest notebook is `02_preprocessing.ipynb` (feature engineering). The messiest is `04_business_insights.ipynb` — that one grew organically as I kept finding new things to slice, and the chart count is higher than it needs to be. I used Plotly throughout the dashboard because hover-tooltips and zoom are the actual killer feature for exploratory data work, and Matplotlib can't do them on a static export.

---

*Last updated: 2026-07-02*
