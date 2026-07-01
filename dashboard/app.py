import os
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Real Estate Buyer Intelligence",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# LOAD DATA & MODELS
# ─────────────────────────────────────────────

@st.cache_data
def load_data():
    return pd.read_csv("../data/processed/final_segmented_dataset.csv")

@st.cache_resource
def load_models():
    model         = joblib.load("../models/clustering_model.pkl")
    scaler        = joblib.load("../models/scaler.pkl")
    cluster_names = joblib.load("../models/cluster_mapping.pkl")
    return model, scaler, cluster_names

df                        = load_data()
model, scaler, CLUSTER_NAMES = load_models()

SEGMENT_ORDER = [
    'First-Time Buyers',
    'Corporate Buyers',
    'Global Investors',
    'Luxury Investors',
]

SEGMENT_COLORS = {
    'First-Time Buyers' : '#3498db',
    'Corporate Buyers'  : '#2ecc71',
    'Global Investors'  : '#f39c12',
    'Luxury Investors'  : '#e74c3c',
}

# ─────────────────────────────────────────────
# CUSTOM STYLING
# ─────────────────────────────────────────────

st.markdown("""
<style>
    section[data-testid="stSidebar"] { width: 280px !important; }
    div[data-testid="metric-container"] {
        background-color: #111827;
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #2d3748;
    }
    .segment-card {
        background-color: #1a1f2e;
        padding: 16px;
        border-radius: 10px;
        border-left: 4px solid;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

st.title("🏠 Real Estate Buyer Intelligence Dashboard")
st.caption("Machine Learning Based Buyer Segmentation & Investment Profiling — Parcl Co.")

# Data freshness: show when the source CSV was last modified
try:
    _csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "final_segmented_dataset.csv")
    _mtime = datetime.fromtimestamp(os.path.getmtime(_csv_path))
    st.caption(
        f"Data snapshot: {_mtime:%Y-%m-%d %H:%M}  ·  "
        f"{len(df):,} buyers  ·  4-segment KMeans model"
    )
except OSError:
    pass  # CSV not reachable from CWD — skip silently

st.markdown("---")

# ─────────────────────────────────────────────
# SIDEBAR — FILTERS
# ─────────────────────────────────────────────

st.sidebar.markdown("# 🎛 Dashboard Controls")
st.sidebar.markdown("---")

selected_segment = st.sidebar.selectbox(
    "Buyer Segment",
    ["All"] + SEGMENT_ORDER,
    help="Filter to one of the 4 ML-derived buyer segments.",
)

country_options = sorted(df["country"].dropna().unique().tolist())
selected_country = st.sidebar.selectbox(
    "Country",
    ["All"] + country_options,
    help="Filter by buyer's country of residence.",
)

purpose_options = sorted(df["acquisition_purpose"].dropna().unique().tolist())
selected_purpose = st.sidebar.selectbox(
    "Acquisition Purpose",
    ["All"] + purpose_options,
    help="Home (residential) vs Investment (rental / portfolio).",
)

client_type_options = sorted(df["client_type"].dropna().unique().tolist())
selected_client_type = st.sidebar.selectbox(
    "Client Type",
    ["All"] + client_type_options,
    help="Individual buyers vs Companies / corporate accounts.",
)

st.sidebar.markdown("---")
st.sidebar.caption(f"Total records loaded: {len(df):,}")

# ─────────────────────────────────────────────
# FILTERING
# ─────────────────────────────────────────────

filtered_df = df.copy()

if selected_segment    != "All":
    filtered_df = filtered_df[filtered_df["Buyer_Segment"]        == selected_segment]
if selected_country    != "All":
    filtered_df = filtered_df[filtered_df["country"]               == selected_country]
if selected_purpose    != "All":
    filtered_df = filtered_df[filtered_df["acquisition_purpose"]   == selected_purpose]
if selected_client_type != "All":
    filtered_df = filtered_df[filtered_df["client_type"]           == selected_client_type]

_filter_is_empty = False
if filtered_df.empty:
    st.warning("⚠️ No data matches the current filters. Please adjust your selections.")
    st.download_button(
        label="Download Full Dataset (un-filtered) as CSV",
        data=df.to_csv(index=False),
        file_name="buyer_segments_full.csv",
        mime="text/csv",
    )
    st.stop()

# ─────────────────────────────────────────────
# KPI CARDS  (skipped when filters are empty — they would show misleading "all" numbers)
# ─────────────────────────────────────────────

if not _filter_is_empty:
    def _delta(curr, base):
        """Format a +X.X% / -X.X% delta vs the un-filtered dataset, or None when unchanged."""
        if base == 0:
            return None
        pct = (curr / base - 1) * 100
        # When delta is effectively zero, return None to hide the indicator
        return None if abs(pct) < 0.05 else f"{pct:+.1f}%"

    base_n      = len(df)
    base_age    = df["age"].mean()
    base_price  = df["sale_price"].mean()
    base_invest = df["investment_score"].mean()
    base_sat    = df["satisfaction_score"].mean()

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

    with kpi1:
        st.metric("👥 Total Buyers",
                  f"{len(filtered_df):,}",
                  delta=_delta(len(filtered_df), base_n))
    with kpi2:
        st.metric("📈 Average Age",
                  f"{filtered_df['age'].mean():.1f} yrs",
                  delta=_delta(filtered_df['age'].mean(), base_age))
    with kpi3:
        st.metric("🏠 Avg Sale Price",
                  f"${filtered_df['sale_price'].mean():,.0f}",
                  delta=_delta(filtered_df['sale_price'].mean(), base_price))
    with kpi4:
        st.metric("💡 Avg Invest. Score",
                  f"{filtered_df['investment_score'].mean():.1f}",
                  delta=_delta(filtered_df['investment_score'].mean(), base_invest))
    with kpi5:
        st.metric("⭐ Avg Satisfaction",
                  f"{filtered_df['satisfaction_score'].mean():.2f}/5",
                  delta=_delta(filtered_df['satisfaction_score'].mean(), base_sat))

st.markdown("---")

# ─────────────────────────────────────────────
# TABS  (PRD: Overview · Investor Insights · Geographic Analysis · Segment Insights · Predict)
# ─────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "💹 Investor Insights",
    "🌍 Geographic Analysis",
    "🔍 Segment Insights",
    "🤖 Predict Buyer",
])

with tab1:
    st.subheader("Buyer Segmentation Overview")

    col1, col2 = st.columns(2)

    with col1:
        # Segment distribution pie
        if selected_segment == "All":
            chart_data = filtered_df["Buyer_Segment"].value_counts().reset_index()
            chart_data.columns = ["Segment", "Count"]
            fig_pie = px.pie(
                chart_data, names="Segment", values="Count",
                hole=0.5,
                color="Segment",
                color_discrete_map=SEGMENT_COLORS,
                category_orders={"Segment": SEGMENT_ORDER},
                title="Buyer Segment Distribution",
            )
            fig_pie.update_traces(textinfo="percent+label")
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            country_data = filtered_df["country"].value_counts().reset_index()
            country_data.columns = ["Country", "Count"]
            fig_pie = px.pie(
                country_data, names="Country", values="Count",
                hole=0.5, title=f"Country Mix — {selected_segment}",
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        # Segment count bar — always show all 4 segments for context,
        # independent of which segment the user has selected in the sidebar
        seg_counts = (
            df["Buyer_Segment"].value_counts()
            .reindex(SEGMENT_ORDER)
            .reset_index()
        )
        seg_counts.columns = ["Segment", "Count"]
        fig_bar = px.bar(
            seg_counts, x="Segment", y="Count",
            color="Segment",
            color_discrete_map=SEGMENT_COLORS,
            title="Segment Buyer Counts",
            text="Count",
        )
        fig_bar.update_traces(textposition="outside")
        fig_bar.update_layout(showlegend=False, height=420)
        st.plotly_chart(fig_bar, use_container_width=True)

    # Acquisition purpose & Client type breakdowns
    c1, c2 = st.columns(2)
    with c1:
        purpose_counts = filtered_df["acquisition_purpose"].value_counts().reset_index()
        purpose_counts.columns = ["Purpose", "Count"]
        fig_p = px.bar(purpose_counts, x="Purpose", y="Count",
                       color="Purpose", title="Acquisition Purpose",
                       color_discrete_sequence=["#3498db", "#e74c3c"])
        fig_p.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig_p, use_container_width=True)

    with c2:
        ct_counts = filtered_df["client_type"].value_counts().reset_index()
        ct_counts.columns = ["Type", "Count"]
        fig_ct = px.bar(ct_counts, x="Type", y="Count",
                        color="Type", title="Client Type",
                        color_discrete_sequence=["#9b59b6", "#f39c12"])
        fig_ct.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig_ct, use_container_width=True)

# ── TAB 2 · Investor Insights ────────────────

with tab2:
    st.subheader("Investor Behaviour Dashboard")

    c1, c2 = st.columns(2)

    with c1:
        # Sale price by segment box
        fig_box = px.box(
            filtered_df, x="Buyer_Segment", y="sale_price",
            color="Buyer_Segment",
            color_discrete_map=SEGMENT_COLORS,
            category_orders={"Buyer_Segment": SEGMENT_ORDER},
            title="Sale Price Distribution by Segment",
        )
        fig_box.update_layout(showlegend=False, height=380)
        st.plotly_chart(fig_box, use_container_width=True)

    with c2:
        # Investment score vs satisfaction — these are the two axes that
        # actually discriminate the 4 segments (age has near-zero spread:
        # 55.1 → 57.6 across segments, so an age-based scatter is uninformative)
        fig_sc = px.scatter(
            filtered_df,
            x="satisfaction_score", y="investment_score",
            color="Buyer_Segment",
            color_discrete_map=SEGMENT_COLORS,
            category_orders={"Buyer_Segment": SEGMENT_ORDER},
            size="sale_price",
            opacity=0.6,
            title="Investment Score vs Satisfaction (segment-defining axes)",
            labels={"investment_score": "Investment Score", "satisfaction_score": "Satisfaction"},
        )
        fig_sc.update_layout(height=380)
        st.plotly_chart(fig_sc, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        # Loan applied rate per segment
        loan_data = (
            filtered_df.groupby("Buyer_Segment")["loan_applied"]
            .apply(lambda x: (x == "Yes").mean() * 100)
            .reindex([s for s in SEGMENT_ORDER if s in filtered_df["Buyer_Segment"].unique()])
            .reset_index()
        )
        loan_data.columns = ["Segment", "Loan Rate (%)"]
        fig_loan = px.bar(
            loan_data, x="Segment", y="Loan Rate (%)",
            color="Segment", color_discrete_map=SEGMENT_COLORS,
            title="Loan Applied Rate by Segment (%)",
            text_auto=".1f",
        )
        fig_loan.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig_loan, use_container_width=True)

    with c4:
        # Referral channel breakdown
        ref_data = filtered_df["referral_channel"].value_counts().reset_index()
        ref_data.columns = ["Channel", "Count"]
        fig_ref = px.pie(
            ref_data, names="Channel", values="Count",
            title="Referral Channel Distribution",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        st.plotly_chart(fig_ref, use_container_width=True)

    # Sale price histogram
    fig_hist = px.histogram(
        filtered_df, x="sale_price",
        color="Buyer_Segment",
        color_discrete_map=SEGMENT_COLORS,
        category_orders={"Buyer_Segment": SEGMENT_ORDER},
        nbins=40, barmode="overlay", opacity=0.7,
        title="Sale Price Distribution by Segment",
    )
    st.plotly_chart(fig_hist, use_container_width=True)

# ── TAB 3 · Geographic Analysis ──────────────

with tab3:
    st.subheader("Geographic Buyer Analysis")

    c1, c2 = st.columns(2)

    with c1:
        # Buyers per country bar
        country_data = filtered_df["country"].value_counts().reset_index()
        country_data.columns = ["Country", "Buyer Count"]
        fig_country = px.bar(
            country_data, x="Country", y="Buyer Count",
            color="Buyer Count", color_continuous_scale="Blues",
            title="Buyer Count by Country",
            text="Buyer Count",
        )
        fig_country.update_traces(textposition="outside")
        fig_country.update_layout(height=400, coloraxis_showscale=False)
        st.plotly_chart(fig_country, use_container_width=True)

    with c2:
        # Avg sale price per country
        avg_price = (
            filtered_df.groupby("country")["sale_price"]
            .mean().sort_values(ascending=False).reset_index()
        )
        avg_price.columns = ["Country", "Avg Sale Price"]
        fig_ap = px.bar(
            avg_price, x="Country", y="Avg Sale Price",
            color="Avg Sale Price", color_continuous_scale="Oranges",
            title="Avg Sale Price by Country",
            text_auto=",.0f",
        )
        fig_ap.update_layout(height=400, coloraxis_showscale=False)
        st.plotly_chart(fig_ap, use_container_width=True)

    # Country × Segment heatmap
    country_seg = pd.crosstab(
        filtered_df["country"],
        filtered_df["Buyer_Segment"]
    ).reindex(
        columns=[s for s in SEGMENT_ORDER if s in filtered_df["Buyer_Segment"].unique()],
        fill_value=0
    )

    fig_heat = px.imshow(
        country_seg,
        color_continuous_scale="Blues",
        title="Country × Buyer Segment — Buyer Counts",
        text_auto=True,
        aspect="auto",
    )
    fig_heat.update_layout(height=420)
    st.plotly_chart(fig_heat, use_container_width=True)

# ── TAB 4 · Segment Insights ─────────────────

# Per-segment recommendation copy used in the deep-dive mode
SEGMENT_RECOMMENDATIONS = {
    "First-Time Buyers": "Focus on education and hand-holding — satisfaction is the lowest of all segments (1.74/5). Highlight financing options, walkthroughs, and post-purchase support. Website channel is dominant so invest in onboarding flows there.",
    "Corporate Buyers":  "High satisfaction (4.24/5) and Apartment-only. Maintain service quality and offer portfolio / multi-unit packages. They're a stable revenue base — prioritise retention and referrals.",
    "Global Investors":  "Office-only segment with mid-tier investment scores. Target with international-investor content, tax/regulatory guides, and remote-purchase tooling. Loan rate and investment% are average so upsell on premium services.",
    "Luxury Investors":   "Highest price (~$486k) and largest floor area (1,599 sqft). Premium portfolio buyers with no loan dependency. Lead with exclusive listings, white-glove service, and off-market opportunities.",
}

def _render_segment_comparison(seg, seg_df, color, total):
    """Compact 6-KPI + 3-chart card used in the 'All' comparison view."""
    st.markdown(f"""
    <div class="segment-card" style="border-left-color:{color}">
    <h4 style="color:{color}">{seg} &nbsp;·&nbsp; {len(seg_df):,} buyers ({len(seg_df)/total*100:.1f}%)</h4>
    </div>
    """, unsafe_allow_html=True)

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Avg Price",        f"${seg_df['sale_price'].mean():,.0f}")
    m2.metric("Avg Floor Area",   f"{seg_df['floor_area_sqft'].mean():,.0f} sqft")
    m3.metric("Avg Age",          f"{seg_df['age'].mean():.1f} yrs")
    m4.metric("Avg Satisfaction", f"{seg_df['satisfaction_score'].mean():.2f}/5")
    m5.metric("Loan Rate",        f"{(seg_df['loan_applied']=='Yes').mean()*100:.1f}%")
    m6.metric("Investment %",     f"{(seg_df['acquisition_purpose']=='Investment').mean()*100:.1f}%")

    c1, c2, c3 = st.columns(3)

    with c1:
        rc = seg_df["referral_channel"].value_counts().reset_index()
        rc.columns = ["Channel", "Count"]
        fig_rc = px.pie(rc, names="Channel", values="Count",
                        title="Referral Channel", hole=0.4,
                        color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_rc.update_layout(height=280, margin=dict(t=40, b=0, l=0, r=0))
        st.plotly_chart(fig_rc, use_container_width=True)

    with c2:
        pu = seg_df["acquisition_purpose"].value_counts().reset_index()
        pu.columns = ["Purpose", "Count"]
        fig_pu = px.pie(pu, names="Purpose", values="Count",
                        title="Acquisition Purpose", hole=0.4,
                        color_discrete_sequence=["#3498db","#e74c3c"])
        fig_pu.update_layout(height=280, margin=dict(t=40, b=0, l=0, r=0))
        st.plotly_chart(fig_pu, use_container_width=True)

    with c3:
        co = seg_df["country"].value_counts().head(5).reset_index()
        co.columns = ["Country", "Count"]
        fig_co = px.bar(co, x="Country", y="Count",
                        title="Top 5 Countries",
                        color_discrete_sequence=[color])
        fig_co.update_layout(height=280, margin=dict(t=40, b=0, l=0, r=0),
                             showlegend=False)
        st.plotly_chart(fig_co, use_container_width=True)

    st.markdown("---")

def _render_segment_deep_dive(seg, seg_df, color, total):
    """Single-segment view: larger header, 6 KPIs, 2×2 chart grid, recommendation."""
    st.markdown(f"""
    <div class="segment-card" style="border-left-color:{color}; padding:18px;">
        <h3 style="color:{color}; margin:0;">{seg}</h3>
        <div style="color:#9ca3af; margin-top:4px;">
            {len(seg_df):,} buyers &nbsp;·&nbsp; {len(seg_df)/total*100:.1f}% of filtered set
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 6 KPI strip
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Avg Price",        f"${seg_df['sale_price'].mean():,.0f}")
    m2.metric("Avg Floor Area",   f"{seg_df['floor_area_sqft'].mean():,.0f} sqft")
    m3.metric("Avg Age",          f"{seg_df['age'].mean():.1f} yrs")
    m4.metric("Avg Satisfaction", f"{seg_df['satisfaction_score'].mean():.2f}/5")
    m5.metric("Loan Rate",        f"{(seg_df['loan_applied']=='Yes').mean()*100:.1f}%")
    m6.metric("Investment %",     f"{(seg_df['acquisition_purpose']=='Investment').mean()*100:.1f}%")

    # 2x2 chart grid
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        rc = seg_df["referral_channel"].value_counts().reset_index()
        rc.columns = ["Channel", "Count"]
        fig_rc = px.pie(rc, names="Channel", values="Count",
                        title="Referral Channel", hole=0.4,
                        color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_rc.update_layout(height=320, margin=dict(t=40, b=0, l=0, r=0))
        st.plotly_chart(fig_rc, use_container_width=True)

    with r1c2:
        pu = seg_df["acquisition_purpose"].value_counts().reset_index()
        pu.columns = ["Purpose", "Count"]
        fig_pu = px.pie(pu, names="Purpose", values="Count",
                        title="Acquisition Purpose", hole=0.4,
                        color_discrete_sequence=["#3498db","#e74c3c"])
        fig_pu.update_layout(height=320, margin=dict(t=40, b=0, l=0, r=0))
        st.plotly_chart(fig_pu, use_container_width=True)

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        co = seg_df["country"].value_counts().head(8).reset_index()
        co.columns = ["Country", "Count"]
        fig_co = px.bar(co, x="Country", y="Count",
                        title="Top Countries",
                        color_discrete_sequence=[color])
        fig_co.update_layout(height=320, margin=dict(t=40, b=0, l=0, r=0),
                             showlegend=False)
        st.plotly_chart(fig_co, use_container_width=True)

    with r2c2:
        fig_age = px.histogram(seg_df, x="age", nbins=20,
                               title="Age Distribution",
                               color_discrete_sequence=[color])
        fig_age.update_layout(height=320, margin=dict(t=40, b=0, l=0, r=0),
                              showlegend=False, yaxis_title="Buyers")
        st.plotly_chart(fig_age, use_container_width=True)

    # Recommendation
    st.markdown("#### 🎯 Recommended Action")
    st.info(SEGMENT_RECOMMENDATIONS.get(seg, "No specific recommendation available for this segment."))


with tab4:
    st.subheader("Segment Insights Panel — Descriptive Statistics per Cluster")

    segments_present = [s for s in SEGMENT_ORDER if s in filtered_df["Buyer_Segment"].unique()]
    total = len(filtered_df) if len(filtered_df) > 0 else 1

    if len(segments_present) == 1:
        seg = segments_present[0]
        seg_df = filtered_df[filtered_df["Buyer_Segment"] == seg]
        color  = SEGMENT_COLORS[seg]
        _render_segment_deep_dive(seg, seg_df, color, total)
    else:
        for seg in segments_present:
            seg_df = filtered_df[filtered_df["Buyer_Segment"] == seg]
            color  = SEGMENT_COLORS[seg]
            _render_segment_comparison(seg, seg_df, color, total)

# ── TAB 5 · Predict Buyer ────────────────────

with tab5:
    st.subheader("🤖 Predict Buyer Segment")
    st.markdown("Enter a buyer's details below to predict which segment they belong to.")

    st.markdown("#### 👤 Buyer Details")
    t1, t2, t3 = st.columns(3)
    with t1:
        input_age = st.number_input("Age", min_value=18, max_value=100, value=35,
                                    key="pred_age",
                                    help="Buyer's age. E.g. 25, 35, 50")
    with t2:
        input_satisfaction = st.slider("Satisfaction Score", min_value=1, max_value=5, value=3,
                                       key="pred_satisfaction",
                                       help="1 = Very Low · 3 = Neutral · 5 = Excellent")
    with t3:
        input_loan = st.selectbox("Loan Applied", ["Yes", "No"],
                                   key="pred_loan",
                                   help="Whether the buyer applied for financing")

    st.markdown("#### 🏠 Property Details")
    t4, t5, t6 = st.columns(3)
    with t4:
        input_sale_price = st.number_input(
            "Property Price ($)", min_value=50000, value=300000, step=1000, format="%d",
            key="pred_price",
            help="Purchase value in USD",
        )
        st.caption(f"💰 {input_sale_price:,}")
    with t5:
        input_floor_area = st.number_input(
            "Floor Area (sqft)", min_value=400, value=1000, step=50,
            key="pred_area",
            help="Typical range: 400–2000 sqft",
        )
    with t6:
        input_unit_type = st.selectbox(
            "Unit Type", ["Apartment", "Office"],
            key="pred_unit",
            help="Apartment = residential · Office = commercial",
        )

    st.markdown("---")

    btn_col, reset_col = st.columns([4, 1])
    with btn_col:
        predict_clicked = st.button("🔍 Predict Segment", use_container_width=True)
    with reset_col:
        if st.button("↺ Reset", use_container_width=True, help="Reset all buyer inputs to defaults"):
            for k in ("pred_age", "pred_satisfaction", "pred_loan",
                      "pred_price", "pred_area", "pred_unit"):
                st.session_state.pop(k, None)
            st.rerun()

    if predict_clicked:
        with st.spinner("Analysing buyer profile..."):

            price_per_sqft   = input_sale_price / input_floor_area if input_floor_area > 0 else 0
            investment_score = (0.7 * (input_sale_price / df['sale_price'].max()) + 0.3 * (input_satisfaction / 5)) * 10

            input_data = pd.DataFrame([{
                "sale_price"       : input_sale_price,
                "floor_area_sqft"  : input_floor_area,
                "price_per_sqft"   : price_per_sqft,
                "investment_score" : investment_score,
                'loan_indicator'    : 1 if input_loan == "Yes" else 0,
                'satisfaction_score': input_satisfaction,
                'age'               : input_age,
                'unit_type_encoded' : 0 if input_unit_type == "Apartment" else 1,
            }])

            if hasattr(scaler, 'feature_names_in_'):
                input_data = input_data[scaler.feature_names_in_]
            cluster_id     = int(model.predict(scaler.transform(input_data))[0])
            predicted_seg  = CLUSTER_NAMES.get(cluster_id) or CLUSTER_NAMES.get(str(cluster_id), "Unknown")
            seg_color      = SEGMENT_COLORS.get(predicted_seg, "#888")

            if predicted_seg == "Unknown":
                st.warning(
                    f"Model returned cluster id {cluster_id}, which is not in "
                    f"models/cluster_mapping.pkl. Re-run notebook 03_clustering.ipynb "
                    f"to refresh the saved models."
                )

        st.markdown(f"""
        <div class="segment-card" style="border-left-color:{seg_color}; padding:20px;">
            <h3 style="color:{seg_color}">Predicted Segment: {predicted_seg}</h3>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Why this segment — derived features:**")
        r1, r2, r3 = st.columns(3)
        r1.metric("Price per sqft",   f"${price_per_sqft:.0f}")
        r2.metric("Investment Score",  f"{investment_score:.1f}")
        r3.metric("Unit Type",         input_unit_type)

        # Show how strongly the buyer's profile matches each segment centroid.
        # Convert raw scaled euclidean distance into softmax match-likelihood (%)
        # so the bars sum to 100 — the closest segment reads as the dominant match.
        st.markdown("**Match likelihood for each segment (higher = stronger match, sums to 100%):**")
        seg_means = df.groupby("Buyer_Segment")[[
            "sale_price", "floor_area_sqft", "price_per_sqft",
            "investment_score", "loan_indicator", "satisfaction_score",
            "age", "unit_type_encoded"
        ]].mean()

        if hasattr(scaler, 'feature_names_in_'):
            seg_means_aligned = seg_means.reindex(columns=scaler.feature_names_in_)
        else:
            seg_means_aligned = seg_means
        scaler_means = scaler.transform(seg_means_aligned)
        input_scaled = scaler.transform(input_data)

        distances = {}
        for i, seg in enumerate(seg_means.index):
            distances[seg] = float(np.linalg.norm(input_scaled - scaler_means[i]))

        dist_series = pd.Series(distances).sort_values()
        # Softmax with negative distances: exp(-d) — smaller d → larger share
        likelihood = np.exp(-dist_series.values)
        likelihood = 100 * likelihood / likelihood.sum()
        match_df = pd.DataFrame({
            "Segment": dist_series.index,
            "Match %": np.round(likelihood, 1),
        })

        fig_dist = px.bar(
            match_df, x="Segment", y="Match %",
            color="Segment", color_discrete_map=SEGMENT_COLORS,
            title="Match Likelihood by Segment (sums to 100%)",
            text="Match %",
        )
        fig_dist.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_dist.update_layout(showlegend=False, height=300, yaxis_ticksuffix="%")
        st.plotly_chart(fig_dist, use_container_width=True)

        st.info("Prediction generated using KMeans clustering model trained on 8 features.")

# ─────────────────────────────────────────────
# EXPORT
# ─────────────────────────────────────────────

st.markdown("---")
st.subheader("📥 Export")
st.download_button(
    label="Download Filtered Dataset as CSV",
    data=filtered_df.to_csv(index=False),
    file_name="filtered_buyer_segments.csv",
    mime="text/csv",
)
