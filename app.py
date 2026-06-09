import streamlit as st
import pandas as pd
import plotly.express as px

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Ecommerce Analytics Dashboard",
    page_icon="🛒",
    layout="wide"
)

# ==================================================
# LOAD DATA
# ==================================================

@st.cache_data
def load_data():
    return pd.read_csv(
        "data/cleaned_sales.csv"
    )

@st.cache_data
def load_rfm():
    return pd.read_csv(
        "data/rfm.csv"
    )

df = load_data()
rfm = load_rfm()

# ==================================================
# SIDEBAR FILTER
# ==================================================

st.sidebar.title("Filters")

selected_country = st.sidebar.selectbox(
    "Select Country",
    ["All"] + sorted(df["Country"].dropna().unique().tolist())
)

if selected_country != "All":
    df_filtered = df[df["Country"] == selected_country]
else:
    df_filtered = df.copy()

st.sidebar.markdown("---")

st.sidebar.info(
    """
    Ecommerce Analytics Project

    • RFM Segmentation

    • Churn Prediction

    • Product Analytics

    • Revenue Analysis
    """
)
# ==================================================
# DASHBOARD TITLE
# ==================================================

st.title("🛒 Ecommerce Analytics Dashboard")

st.markdown("""
Interactive dashboard for customer segmentation,
churn analysis, product analytics and revenue insights.
""")

# ==================================================
# KPI SECTION
# ==================================================

total_revenue = df_filtered["Revenue"].sum()
total_orders = df_filtered["InvoiceNo"].nunique()
total_customers = df_filtered["CustomerID"].nunique()
avg_order_value = total_revenue / total_orders

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Revenue", f"£{total_revenue:,.0f}")
col2.metric("Total Orders", f"{total_orders:,}")
col3.metric("Total Customers", f"{total_customers:,}")
col4.metric("Avg Order Value", f"£{avg_order_value:.2f}")

# ==================================================
# REVENUE BY COUNTRY
# ==================================================

st.markdown("---")
st.subheader("🌍 Revenue Analysis")

country_revenue = (
    df_filtered.groupby("Country")["Revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

fig = px.bar(
    country_revenue,
    x="Revenue",
    y="Country",
    orientation="h",
    title="Top 10 Countries by Revenue"
)

fig.update_layout(
    yaxis={'categoryorder': 'total ascending'}
)

st.plotly_chart(fig, use_container_width=True)

# ==================================================
# MONTHLY REVENUE TREND
# ==================================================

st.markdown("---")
st.subheader("📈 Revenue Trend")

df_filtered["InvoiceDate"] = pd.to_datetime(
    df_filtered["InvoiceDate"]
)

monthly_revenue = (
    df_filtered.groupby(
        df_filtered["InvoiceDate"].dt.to_period("M")
    )["Revenue"]
    .sum()
    .reset_index()
)

monthly_revenue["InvoiceDate"] = (
    monthly_revenue["InvoiceDate"]
    .astype(str)
)

fig2 = px.line(
    monthly_revenue,
    x="InvoiceDate",
    y="Revenue",
    title="Monthly Revenue Trend",
    markers=True
)

st.plotly_chart(fig2, use_container_width=True)

# ==================================================
# CUSTOMER SEGMENTATION
# ==================================================

st.markdown("---")
st.subheader("👥 Customer Segmentation")

segment_counts = (
    rfm["Segment"]
    .value_counts()
    .reset_index()
)

segment_counts.columns = ["Segment", "Count"]

segment_order = [
    "Champion",
    "Loyal Customer",
    "Regular Customer",
    "At Risk",
    "Lost Customer"
]

segment_counts["Segment"] = pd.Categorical(
    segment_counts["Segment"],
    categories=segment_order,
    ordered=True
)

segment_counts = segment_counts.sort_values(
    "Segment"
)

fig3 = px.bar(
    segment_counts,
    x="Segment",
    y="Count",
    title="Customer Segment Distribution"
)

st.plotly_chart(
    fig3,
    use_container_width=True
)

# ==================================================
# PRODUCT ANALYSIS
# ==================================================

st.markdown("---")
st.subheader("📦 Product Analysis")

top_products = (
    df_filtered.groupby("Description")["Revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

fig4 = px.bar(
    top_products,
    x="Revenue",
    y="Description",
    orientation="h",
    title="Top 10 Products by Revenue"
)

fig4.update_layout(
    yaxis={'categoryorder': 'total ascending'}
)

st.plotly_chart(
    fig4,
    use_container_width=True
)

# ==================================================
# CHURN ANALYSIS
# ==================================================

st.markdown("---")
st.subheader("⚠️ Churn Analysis")

total_rfm_customers = len(rfm)

churned_customers = rfm["Churn"].sum()

churn_rate = (
    churned_customers /
    total_rfm_customers
) * 100

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Customers",
    f"{total_rfm_customers:,}"
)

col2.metric(
    "Churned Customers",
    f"{int(churned_customers):,}"
)

col3.metric(
    "Churn Rate",
    f"{churn_rate:.2f}%"
)

feature_importance = pd.DataFrame({
    "Feature": [
        "Recency",
        "Frequency",
        "Monetary"
    ],
    "Importance": [
        0.883227,
        0.066616,
        0.050157
    ]
})

fig5 = px.bar(
    feature_importance,
    x="Feature",
    y="Importance",
    title="Random Forest Feature Importance"
)

st.plotly_chart(
    fig5,
    use_container_width=True
)