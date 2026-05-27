# 🏠 Buyer Segmentation & Investment Profiling
### Machine Learning Based Real Estate Market Intelligence
**Collaborating Organizations:** Unified Mentor × Parcl Co. Limited

---

## 📌 Problem Statement

Parcl currently lacks a data-driven understanding of different types of property buyers, their investment motivations, geographic differences in behaviour, and financing patterns. Without segmentation, all buyers are treated the same — leading to inefficient marketing, generic recommendations, and missed investment opportunities.

This project uses unsupervised machine learning (K-Means + Hierarchical Clustering) to discover hidden buyer segments and build an interactive intelligence dashboard.

---

## 🎯 Objectives

- Clean and merge client and property transaction data
- Engineer meaningful features from raw buyer and property attributes
- Encode and scale features for ML readiness
- Apply K-Means and Hierarchical Clustering to identify buyer segments
- Interpret and label clusters with business-meaningful names
- Build a Streamlit dashboard for interactive exploration and prediction

---

## 🗂️ Project Structure

```
BUYER-SEGMENTATION/
│
├── dashboard/
│   └── app.py                        # Streamlit web application
│
├── data/
│   ├── raw/
│   │   ├── clients.csv               # Raw client demographic data
│   │   └── properties.csv            # Raw property transaction data
│   └── processed/
│       ├── cleaned_dataset.csv       # Output of 01_eda.ipynb
│       ├── features_dataset.csv      # Output of 02_preprocessing.ipynb
│       ├── final_segmented_dataset.csv  # Output of 03_clustering.ipynb
│       ├── cluster_summary.csv       # Cluster stats for research paper
│       └── segment_insights.csv     # Business insights per segment
│
├── models/
│   ├── clustering_model.pkl          # Trained KMeans model
│   ├── scaler.pkl                    # Fitted StandardScaler
│   ├── cluster_mapping.pkl           # Auto-generated segment name map
│   ├── encoders.pkl                  # Label + OHE encoders
│   └── feature_columns.pkl          # Model feature name list
│
├── notebooks/
│   ├── 01_eda.ipynb                  # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb        # Feature Engineering & Encoding
│   ├── 03_clustering.ipynb           # K-Means + Hierarchical Clustering
│   └── 04_business_insights.ipynb   # Segment Profiling & Recommendations
│
├── reports/                          # Research paper and exports
├── main.py                           # Project entry point
├── requirements.txt                  # Python dependencies
└── README.md
```

---

## 📊 Dataset

### `clients.csv` — 2,000 records, 12 fields
| Field | Description |
|---|---|
| `client_id` | Unique client identifier |
| `client_type` | Individual / Company |
| `first_name`, `last_name` | Client name |
| `date_of_birth` | Used to compute age |
| `gender` | M / F |
| `country` | Country of residence (10 countries) |
| `region` | Geographic region (57 regions) |
| `acquisition_purpose` | Home / Investment |
| `satisfaction_score` | 1–5 rating |
| `loan_applied` | Yes / No |
| `referral_channel` | Website / Agency / Client |

### `properties.csv` — 10,000 records, 9 fields
| Field | Description |
|---|---|
| `listing_id` | Property listing ID |
| `unit_category` | Apartment / Office |
| `floor_area_sqft` | Property size in sq ft |
| `sale_price` | Transaction price (USD) |
| `transaction_date` | Date of transaction |
| `listing_status` | Sold / Available |
| `client_ref` | Links to `client_id` |

---

## 🧠 ML Methodology

### Step 1 — Data Cleaning (`01_eda.ipynb`)
- Remove duplicates from both datasets
- Parse mixed-format dates, clean `sale_price` currency strings
- Drop unlinked properties (`client_ref` null)
- Merge clients + properties on `client_id`
- Compute `age` from `date_of_birth`

### Step 2 — Feature Engineering & Encoding (`02_preprocessing.ipynb`)
- **Engineered:** `price_per_sqft`, `investment_score`, `loan_indicator`, `unit_type_encoded`
- **Label Encoding:** `client_type`, `gender`, `acquisition_purpose`
- **One-Hot Encoding:** `referral_channel`, `country`, `region`
- **Scaling:** StandardScaler on all 8 model features

### Step 3 — Clustering (`03_clustering.ipynb`)
- **Elbow Method** + **Silhouette Score** + **Davies-Bouldin Score** → optimal K=3
- **K-Means** (primary model, dashboard-integrated)
- **Hierarchical Clustering** (Ward linkage, dendrogram validation)
- **Auto-labelling:** clusters ranked by composite score (price + investment_score + price_per_sqft + floor_area + satisfaction) → no manual mapping

### Step 4 — Business Insights (`04_business_insights.ipynb`)
- Radar chart comparison across segments
- Geographic, financing, referral, and satisfaction analysis
- Actionable recommendations per segment

---

## 👥 Buyer Segments

| Segment | Profile |
|---|---|
| 🏠 **Budget Residential Buyers** | Lower price, smaller area, high loan dependency, Home purpose, Website channel |
| 🏢 **Mainstream Residential Buyers** | Mid-range properties, balanced Home/Investment mix, Agency channel effective |
| 💎 **Premium Investors** | High-value, large area, low loan rate, high satisfaction, diverse geography |

---

## 🚀 Setup & Run

### 1. Clone the repository
```bash
git clone <repo-url>
cd BUYER-SEGMENTATION
```

### 2. Create a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the notebooks in order
Open Jupyter and run each notebook sequentially:
```
notebooks/01_eda.ipynb
notebooks/02_preprocessing.ipynb
notebooks/03_clustering.ipynb
notebooks/04_business_insights.ipynb
```
This generates all processed CSVs and model PKL files.

### 5. Launch the dashboard
```bash
python main.py
```
Or directly:
```bash
streamlit run dashboard/app.py
```
Open your browser at **http://localhost:8501**

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `pandas`, `numpy` | Data manipulation |
| `matplotlib`, `seaborn` | Static visualisations |
| `scikit-learn` | KMeans, Hierarchical, Scaler, Metrics |
| `scipy` | Dendrogram / Linkage |
| `plotly` | Interactive dashboard charts |
| `streamlit` | Web dashboard framework |
| `joblib` | Model serialisation |
| `jupyter`, `ipykernel` | Notebook execution |

---

## 📋 Dashboard Features

| Tab | Content |
|---|---|
| 📊 Overview | Segment pie, bar charts, purpose & client type breakdown |
| 💹 Investor Insights | Price by segment, investment score scatter, loan rates, referral channel |
| 🌍 Geographic Analysis | Country counts, avg price by country, country×segment heatmap, top regions |
| 🔍 Segment Insights | Per-cluster KPIs, referral/purpose/country breakdowns |
| 🤖 Predict Buyer | Input buyer details → predicted segment + centroid distance chart |

**Filters:** Country · Region · Acquisition Purpose · Client Type · Buyer Segment

---

## 🏢 Collaborating Organizations
- **Unified Mentor**
- **Parcl Co. Limited**
