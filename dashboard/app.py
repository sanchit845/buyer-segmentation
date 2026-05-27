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
    "Budget Residential Buyers",
    "Mainstream Residential Buyers",
    "Premium Investors",
]

SEGMENT_COLORS = {
    "Budget Residential Buyers"     : "#3498db",
    "Mainstream Residential Buyers" : "#2ecc71",
    "Premium Investors"             : "#e74c3c",
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
st.markdown("---")

# ─────────────────────────────────────────────
# SIDEBAR — FILTERS  (PRD: country, region, acquisition_purpose, client_type)
# ─────────────────────────────────────────────

st.sidebar.markdown("# 🎛 Dashboard Controls")
st.sidebar.markdown("---")

selected_segment = st.sidebar.selectbox(
    "Buyer Segment",
    ["All"] + SEGMENT_ORDER,
)

country_options = sorted(df["country"].dropna().unique().tolist())
selected_country = st.sidebar.selectbox(
    "Country",
    ["All"] + country_options,
)

region_options = sorted(df["region"].dropna().unique().tolist())
selected_region = st.sidebar.selectbox(
    "Region",
    ["All"] + region_options,
)

purpose_options = sorted(df["acquisition_purpose"].dropna().unique().tolist())
selected_purpose = st.sidebar.selectbox(
    "Acquisition Purpose",
    ["All"] + purpose_options,
)

client_type_options = sorted(df["client_type"].dropna().unique().tolist())
selected_client_type = st.sidebar.selectbox(
    "Client Type",
    ["All"] + client_type_options,
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
if selected_region     != "All":
    filtered_df = filtered_df[filtered_df["region"]                == selected_region]
if selected_purpose    != "All":
    filtered_df = filtered_df[filtered_df["acquisition_purpose"]   == selected_purpose]
if selected_client_type != "All":
    filtered_df = filtered_df[filtered_df["client_type"]           == selected_client_type]

if filtered_df.empty:
    st.warning("⚠️ No data matches the current filters. Please adjust your selections.")
    st.stop()

# ─────────────────────────────────────────────
# KPI CARDS
# ─────────────────────────────────────────────

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

with kpi1:
    st.metric("👥 Total Buyers", f"{len(filtered_df):,}")
with kpi2:
    st.metric("📈 Average Age", f"{filtered_df['age'].mean():.1f} yrs")
with kpi3:
    st.metric("🏠 Avg Sale Price", f"${filtered_df['sale_price'].mean():,.0f}")
with kpi4:
    st.metric("💡 Avg Invest. Score", f"{filtered_df['investment_score'].mean():.1f}")
with kpi5:
    st.metric("⭐ Avg Satisfaction", f"{filtered_df['satisfaction_score'].mean():.2f}/5")

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

# ── TAB 1 · Overview ─────────────────────────

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
        # Segment count bar
        seg_counts = (
            filtered_df["Buyer_Segment"].value_counts()
            .reindex([s for s in SEGMENT_ORDER if s in filtered_df["Buyer_Segment"].unique()])
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
        # Investment score vs age scatter
        fig_sc = px.scatter(
            filtered_df,
            x="age", y="investment_score",
            color="Buyer_Segment",
            color_discrete_map=SEGMENT_COLORS,
            category_orders={"Buyer_Segment": SEGMENT_ORDER},
            size="sale_price",
            opacity=0.6,
            title="Investment Score vs Age",
            labels={"investment_score": "Investment Score", "age": "Age"},
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

    # Top 15 regions
    top_regions = filtered_df["region"].value_counts().head(15).reset_index()
    top_regions.columns = ["Region", "Count"]
    fig_reg = px.bar(
        top_regions, x="Count", y="Region",
        orientation="h",
        color="Count", color_continuous_scale="Teal",
        title="Top 15 Regions by Buyer Count",
    )
    fig_reg.update_layout(height=500, coloraxis_showscale=False)
    st.plotly_chart(fig_reg, use_container_width=True)

# ── TAB 4 · Segment Insights ─────────────────

with tab4:
    st.subheader("Segment Insights Panel — Descriptive Statistics per Cluster")

    segments_present = [s for s in SEGMENT_ORDER if s in filtered_df["Buyer_Segment"].unique()]

    for seg in segments_present:
        seg_df = filtered_df[filtered_df["Buyer_Segment"] == seg]
        color  = SEGMENT_COLORS[seg]

        st.markdown(f"""
        <div class="segment-card" style="border-left-color:{color}">
        <h4 style="color:{color}">{seg} &nbsp;·&nbsp; {len(seg_df):,} buyers ({len(seg_df)/len(filtered_df)*100:.1f}%)</h4>
        </div>
        """, unsafe_allow_html=True)

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Avg Price",       f"${seg_df['sale_price'].mean():,.0f}")
        m2.metric("Avg Floor Area",  f"{seg_df['floor_area_sqft'].mean():,.0f} sqft")
        m3.metric("Avg Age",         f"{seg_df['age'].mean():.1f} yrs")
        m4.metric("Avg Satisfaction",f"{seg_df['satisfaction_score'].mean():.2f}/5")
        m5.metric("Loan Rate",       f"{(seg_df['loan_applied']=='Yes').mean()*100:.1f}%")
        m6.metric("Investment %",    f"{(seg_df['acquisition_purpose']=='Investment').mean()*100:.1f}%")

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

# ── TAB 5 · Predict Buyer ────────────────────

with tab5:
    st.subheader("🤖 Predict Buyer Segment")
    st.markdown("Enter a buyer's details below to predict which segment they belong to.")

    st.markdown("#### 👤 Buyer Details")
    t1, t2, t3 = st.columns(3)
    with t1:
        input_age = st.number_input("Age", min_value=18, max_value=100, value=35,
                                    help="Buyer's age. E.g. 25, 35, 50")
    with t2:
        input_satisfaction = st.slider("Satisfaction Score", min_value=1, max_value=5, value=3,
                                       help="1 = Very Low · 3 = Neutral · 5 = Excellent")
    with t3:
        input_loan = st.selectbox("Loan Applied", ["Yes", "No"],
                                   help="Whether the buyer applied for financing")

    st.markdown("#### 🏠 Property Details")
    t4, t5, t6 = st.columns(3)
    with t4:
        input_sale_price = st.number_input(
            "Property Price ($)", min_value=50000, value=300000, step=1000, format="%d",
            help="Purchase value in USD"
        )
        st.caption(f"💰 {input_sale_price:,}")
    with t5:
        input_floor_area = st.number_input(
            "Floor Area (sqft)", min_value=400, value=1000, step=50,
            help="Typical range: 400–2000 sqft"
        )
    with t6:
        input_unit_type = st.selectbox(
            "Unit Type", ["Apartment", "Office"],
            help="Apartment = residential · Office = commercial"
        )

    st.markdown("---")

    if st.button("🔍 Predict Segment", use_container_width=True):
        with st.spinner("Analysing buyer profile..."):

            price_per_sqft   = input_sale_price / input_floor_area
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

            input_data     = input_data[scaler.feature_names_in_]
            cluster_id     = int(model.predict(scaler.transform(input_data))[0])
            predicted_seg  = CLUSTER_NAMES[cluster_id]
            seg_color      = SEGMENT_COLORS.get(predicted_seg, "#888")

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

        # Show how close the buyer is to each segment mean
        st.markdown("**Distance to each segment centroid (lower = closer match):**")
        seg_means = df.groupby("Buyer_Segment")[[
            "sale_price", "floor_area_sqft", "price_per_sqft",
            "investment_score", "loan_indicator", "satisfaction_score",
            "age", "unit_type_encoded"
        ]].mean()

        scaler_means = scaler.transform(seg_means.reindex(columns=scaler.feature_names_in_))
        input_scaled = scaler.transform(input_data)

        distances = {}
        for i, seg in enumerate(seg_means.index):
            distances[seg] = float(np.linalg.norm(input_scaled - scaler_means[i]))

        dist_df = pd.DataFrame.from_dict(distances, orient="index", columns=["Distance"])
        dist_df = dist_df.sort_values("Distance")

        fig_dist = px.bar(
            dist_df.reset_index(), x="index", y="Distance",
            color="index", color_discrete_map=SEGMENT_COLORS,
            title="Euclidean Distance to Segment Centroid",
            labels={"index": "Segment"},
        )
        fig_dist.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig_dist, use_container_width=True)

        st.info("Prediction generated using KMeans clustering model trained on 8 features.")

# ─────────────────────────────────────────────
# BUYER PERSONAS REFERENCE
# ─────────────────────────────────────────────

st.markdown("---")
st.subheader("📋 Buyer Persona Reference")

p1, p2, p3 = st.columns(3)

with p1:
    st.info(
        "🏠 **Budget Residential Buyers**\n\n"
        "• Lower property value\n"
        "• Smaller floor area\n"
        "• High loan dependency\n"
        "• Cost-sensitive, primarily Home buyers\n"
        "• Mostly via Website channel"
    )
with p2:
    st.info(
        "🏢 **Mainstream Residential Buyers**\n\n"
        "• Mid-range properties\n"
        "• Balanced Apartment/Office mix\n"
        "• Average investment behaviour\n"
        "• Mix of Home & Investment purpose\n"
        "• Agency channel effective"
    )
with p3:
    st.info(
        "💎 **Premium Investors**\n\n"
        "• High-value properties\n"
        "• Large floor areas\n"
        "• High investment score\n"
        "• Low loan dependency\n"
        "• Diverse geographic origin\n"
        "• High satisfaction — strong referral potential"
    )

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
