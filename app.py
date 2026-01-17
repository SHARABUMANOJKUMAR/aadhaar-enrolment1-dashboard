import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Aadhaar Enrolment Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Aadhaar Enrolment Trends Dashboard")
st.caption("Government Data Analytics | UIDAI Aadhaar Enrolment")

# ---------------- DATA LOADING (CACHED) ----------------
@st.cache_data(show_spinner=True)
def load_data():
    df = pd.read_csv("api_data_aadhar_enrolment_500000_1000000.csv")

    df['date'] = pd.to_datetime(df['date'], errors='coerce', dayfirst=True)
    df = df.dropna(subset=['date'])
    df.fillna(0, inplace=True)

    # Auto-detect age columns
    age_0_5 = next((c for c in df.columns if "0" in c and "5" in c), None)
    age_5_17 = next((c for c in df.columns if "5" in c and "17" in c), None)
    age_18_plus = next((c for c in df.columns if "18" in c), None)

    if not all([age_0_5, age_5_17, age_18_plus]):
        st.error("❌ Required age-wise columns not found")
        st.stop()

    df.rename(columns={
        age_0_5: "age_0_5",
        age_5_17: "age_5_17",
        age_18_plus: "age_18_plus"
    }, inplace=True)

    df["total"] = df[["age_0_5", "age_5_17", "age_18_plus"]].sum(axis=1)

    return df

df = load_data()

# ---------------- SIDEBAR FILTER ----------------
st.sidebar.header("🔎 Filters")
selected_state = st.sidebar.selectbox("Select State", sorted(df["state"].unique()))
filtered_df = df[df["state"] == selected_state]

# ---------------- KPIs ----------------
st.subheader(f"📍 State Selected: {selected_state}")

c1, c2, c3 = st.columns(3)
c1.metric("👥 Total Enrolments", f"{int(filtered_df['total'].sum()):,}")
c2.metric("👶 Children (0–5)", f"{int(filtered_df['age_0_5'].sum()):,}")
c3.metric("🧑 Adults (18+)", f"{int(filtered_df['age_18_plus'].sum()):,}")

st.divider()

# ---------------- TIME SERIES ----------------
monthly = (
    filtered_df
    .set_index("date")
    .resample("M")
    .sum()["total"]
)

st.subheader("📈 Monthly Aadhaar Enrolment Trend")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(monthly, label="Actual")
ax.grid(alpha=0.3)
ax.set_title("Monthly Enrolment Trend")
st.pyplot(fig, use_container_width=True)

# ---------------- 🔮 FORECASTING (ARIMA + ML FALLBACK) ----------------
st.subheader("🔮 Aadhaar Enrolment Forecast (Next 6 Months)")

forecast_generated = False

# 🔹 Try ARIMA first
if monthly.notna().sum() >= 6:
    try:
        arima_model = ARIMA(monthly, order=(1, 1, 1))
        arima_fit = arima_model.fit()
        forecast = arima_fit.forecast(steps=6)
        forecast_generated = True
        forecast_method = "ARIMA Time-Series Model"
    except:
        forecast_generated = False

# 🔹 ML FALLBACK: Linear Regression Trend
if not forecast_generated:
    y = monthly.values
    X = np.arange(len(y)).reshape(-1, 1)

    lr = LinearRegression()
    lr.fit(X, y)

    future_X = np.arange(len(y), len(y) + 6).reshape(-1, 1)
    forecast = lr.predict(future_X)

    forecast_index = pd.date_range(
        start=monthly.index[-1] + pd.offsets.MonthEnd(),
        periods=6,
        freq="M"
    )
    forecast = pd.Series(forecast, index=forecast_index)
    forecast_method = "Machine Learning Trend Projection (Linear Regression)"

# ---------------- FORECAST PLOT ----------------
fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(monthly, label="Historical Data")
ax2.plot(forecast, label="Forecast (Next 6 Months)", linestyle="--")
ax2.legend()
ax2.grid(alpha=0.3)
ax2.set_title(f"Forecast Method Used: {forecast_method}")
st.pyplot(fig2, use_container_width=True)

# ---------------- AGE DISTRIBUTION ----------------
st.subheader("👨‍👩‍👧 Age-wise Distribution")

age_sum = filtered_df[["age_0_5", "age_5_17", "age_18_plus"]].sum()

fig3, ax3 = plt.subplots(figsize=(5, 5))
ax3.pie(
    age_sum,
    labels=["0–5 Years", "5–17 Years", "18+ Years"],
    autopct="%1.1f%%",
    startangle=90
)
st.pyplot(fig3)

# ---------------- DATA PREVIEW ----------------
st.subheader("📄 Data Preview")
st.dataframe(filtered_df.head(100), use_container_width=True)

st.caption(
    "Forecasting uses ARIMA where data continuity exists; otherwise ML-based trend projection is applied. "
    "This ensures responsible and continuous decision support for government planning."
)
