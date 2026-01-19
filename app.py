# ================== IMPORTS ==================
import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="Aadhaar Enrolment Intelligence",
    page_icon="🆔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================== PREMIUM UI STYLE ==================
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}
.block-container {
    padding-top: 1.5rem;
}
.glass {
    background: rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 20px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.35);
}
h1, h2, h3 {
    color: #ffffff;
}
.metric-card {
    background: linear-gradient(135deg,#1f4037,#99f2c8);
    padding: 20px;
    border-radius: 16px;
    text-align: center;
    color: black;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# ================== HEADER ==================
st.markdown("<h1>🆔 Aadhaar Enrolment Intelligence Dashboard</h1>", unsafe_allow_html=True)
st.caption("📊 Government-Scale Data Analytics | AI-Driven Forecasting | UIDAI")

# ================== DATA LOADING ==================
@st.cache_data(show_spinner=True)
def load_data():
    df = pd.read_csv("aadhaar_clean2_states (1).csv")

    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
    df.dropna(subset=["date"], inplace=True)
    df.fillna(0, inplace=True)

    df.rename(columns={
        "age_0_5": "Age 0–5",
        "age_5_17": "Age 5–17",
        "age_18_greater": "Age 18+"
    }, inplace=True)

    df["Total Enrolments"] = df[["Age 0–5", "Age 5–17", "Age 18+"]].sum(axis=1)
    return df

df = load_data()

# ================== SIDEBAR ==================
st.sidebar.markdown("## 🔍 Smart Filters")
state = st.sidebar.selectbox(
    "Select State",
    sorted(df["state"].unique()),
    placeholder="Choose a state"
)

filtered_df = df[df["state"] == state]

# ================== KPI METRICS ==================
st.markdown("### 📌 Key Performance Indicators")

k1, k2, k3 = st.columns(3)

k1.markdown(
    f"<div class='metric-card'>👥<br>Total Enrolments<br>{int(filtered_df['Total Enrolments'].sum()):,}</div>",
    unsafe_allow_html=True
)
k2.markdown(
    f"<div class='metric-card'>👶<br>Children (0–5)<br>{int(filtered_df['Age 0–5'].sum()):,}</div>",
    unsafe_allow_html=True
)
k3.markdown(
    f"<div class='metric-card'>🧑<br>Adults (18+)<br>{int(filtered_df['Age 18+'].sum()):,}</div>",
    unsafe_allow_html=True
)

# ================== TIME SERIES ==================
monthly = (
    filtered_df
    .set_index("date")
    .resample("M")
    .sum()["Total Enrolments"]
)

st.markdown("### 📈 Aadhaar Enrolment Trend (Monthly)")

trend_fig = px.line(
    x=monthly.index,
    y=monthly.values,
    labels={"x": "Month", "y": "Enrolments"},
    markers=True,
    template="plotly_dark"
)
trend_fig.update_traces(line=dict(width=3, color="#00f5d4"))
st.plotly_chart(trend_fig, use_container_width=True)

# ================== FORECASTING ==================
st.markdown("### 🔮 AI-Based Enrolment Forecast (Next 6 Months)")

forecast_used = "ARIMA Time-Series Model"

try:
    model = ARIMA(monthly, order=(1, 1, 1))
    model_fit = model.fit()
    forecast = model_fit.forecast(6)
except Exception:
    forecast_used = "Machine Learning Trend Projection"
    y = monthly.values
    X = np.arange(len(y)).reshape(-1, 1)

    lr = LinearRegression()
    lr.fit(X, y)

    future_X = np.arange(len(y), len(y) + 6).reshape(-1, 1)
    preds = lr.predict(future_X)

    forecast = pd.Series(
        preds,
        index=pd.date_range(monthly.index[-1], periods=7, freq="M")[1:]
    )

forecast_fig = go.Figure()

forecast_fig.add_trace(go.Scatter(
    x=monthly.index,
    y=monthly.values,
    mode="lines+markers",
    name="Historical"
))

forecast_fig.add_trace(go.Scatter(
    x=forecast.index,
    y=forecast.values,
    mode="lines+markers",
    name="Forecast",
    line=dict(dash="dash")
))

forecast_fig.update_layout(
    template="plotly_dark",
    title=f"Forecast Method: {forecast_used}"
)

st.plotly_chart(forecast_fig, use_container_width=True)

# ================== AGE DISTRIBUTION ==================
st.markdown("### 👨‍👩‍👧 Age-Wise Enrolment Distribution")

age_sum = filtered_df[["Age 0–5", "Age 5–17", "Age 18+"]].sum()

pie_fig = px.pie(
    values=age_sum.values,
    names=age_sum.index,
    hole=0.45,
    color_discrete_sequence=px.colors.sequential.Aggrnyl,
    template="plotly_dark"
)
st.plotly_chart(pie_fig, use_container_width=True)

# ================== DATA TABLE ==================
st.markdown("### 📄 Data Snapshot")
st.dataframe(filtered_df.head(200), use_container_width=True)

# ================== FOOTER ==================
st.caption(
    "🔐 Enterprise-grade dashboard built using Streamlit, Plotly, ARIMA & ML. "
    "Designed for policy planning, demographic insights & future forecasting."
)

