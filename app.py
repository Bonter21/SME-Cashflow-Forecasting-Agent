import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="CashFlow Predictor", page_icon="💰", layout="wide")

COLORS = {
    "primary": "#228B22",
    "secondary": "#32CD32",
    "accent": "#90EE90",
    "background": "#FFFFFF",
    "text": "#1A1A1A",
    "light_bg": "#F0FFF0",
    "border": "#2E8B2E",
}


def apply_green_theme():
    st.markdown(
        f"""
    <style>
    .stApp {{
        background-color: {COLORS["background"]};
    }}
    .main-header {{
        background: linear-gradient(135deg, {COLORS["primary"]}, {COLORS["secondary"]});
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }}
    .main-header h1 {{
        color: white !important;
        margin: 0;
        font-size: 32px;
        font-weight: bold;
    }}
    .main-header p {{
        color: {COLORS["accent"]};
        margin: 5px 0 0 0;
    }}
    .stSidebar {{
        background-color: {COLORS["light_bg"]};
    }}
    .upload-zone {{
        border: 3px dashed {COLORS["primary"]};
        border-radius: 15px;
        padding: 40px;
        text-align: center;
        background-color: {COLORS["light_bg"]};
        margin: 20px 0;
    }}
    .metric-card {{
        background: white;
        border: 2px solid {COLORS["primary"]};
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(34, 139, 34, 0.2);
    }}
    .metric-card h3 {{
        color: {COLORS["primary"]};
        font-size: 14px;
        margin: 0;
    }}
    .metric-card h2 {{
        color: {COLORS["text"]};
        font-size: 28px;
        margin: 10px 0;
    }}
    .stButton>button {{
        background: linear-gradient(135deg, {COLORS["primary"]}, {COLORS["secondary"]});
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 25px;
    }}
    .stButton>button:hover {{
        background: linear-gradient(135deg, {COLORS["secondary"]}, {COLORS["primary"]});
    }}
    .insight-box {{
        background: {COLORS["light_bg"]};
        border-left: 4px solid {COLORS["primary"]};
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }}
    div[data-testid="stMetric"] {{
        background-color: {COLORS["light_bg"]};
        border: 1px solid {COLORS["accent"]};
        border-radius: 10px;
        padding: 15px;
    }}
    div[data-testid="stMetric"] label {{
        color: {COLORS["primary"]} !important;
    }}
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {{
        color: {COLORS["text"]} !important;
    }}
    </style>
    """,
        unsafe_allow_html=True,
    )


def detect_columns(df):
    date_cols = [
        "date",
        "date",
        "transaction_date",
        "trans_date",
        "timestamp",
        "datetime",
    ]
    amount_cols = ["amount", "balance", "value", "total", "sum", "net_amount"]
    income_cols = ["income", "revenue", "credit", " inflow", "sales"]
    expense_cols = ["expense", "expenses", "debit", " outflow", "cost"]

    df.columns = df.columns.str.lower().str.strip()

    date_col = next(
        (c for c in df.columns if any(d in c for d in date_cols)), df.columns[0]
    )
    amount_col = next(
        (c for c in df.columns if any(a in c for a in amount_cols)),
        df.columns[1] if len(df.columns) > 1 else df.columns[0],
    )

    return date_col, amount_col


def process_data(df):
    df = df.copy()
    df.columns = df.columns.str.lower().str.strip()

    date_col, amount_col = detect_columns(df)

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    df[amount_col] = pd.to_numeric(df[amount_col], errors="coerce")
    df = df.dropna(subset=[amount_col])

    df = df.sort_values(date_col)

    if df[amount_col].dtype in ["object"]:
        df[amount_col] = df[amount_col].str.replace(",", "").astype(float)

    df["running_balance"] = df[amount_col].cumsum()

    daily_df = (
        df.groupby(df[date_col].dt.date)
        .agg({amount_col: "sum", "running_balance": "last"})
        .reset_index()
    )
    daily_df.columns = ["date", "daily_change", "balance"]
    daily_df["date"] = pd.to_datetime(daily_df["date"])

    return df, daily_df, date_col, amount_col


def predict_cashflow(daily_df, days=30):
    if len(daily_df) < 7:
        return None

    daily_df = daily_df.copy()
    daily_df["day_num"] = (daily_df["date"] - daily_df["date"].min()).dt.days

    X = daily_df["day_num"].values.reshape(-1, 1)
    y = daily_df["daily_change"].values

    model = LinearRegression()
    model.fit(X, y)

    last_date = daily_df["date"].max()
    last_day = daily_df["day_num"].max()

    future_dates = [last_date + timedelta(days=i) for i in range(1, days + 1)]
    future_day_nums = [last_day + i for i in range(1, days + 1)]
    future_X = np.array(future_day_nums).reshape(-1, 1)
    predictions = model.predict(future_X)

    last_balance = daily_df["balance"].iloc[-1]
    predicted_balances = [
        last_balance + sum(predictions[: i + 1]) for i in range(len(predictions))
    ]

    return {
        "dates": future_dates,
        "predictions": predictions.tolist(),
        "balances": predicted_balances,
        "avg_daily": model.coef_[0],
        "trend": "up" if model.coef_[0] > 0 else "down",
    }


def create_chart(daily_df, predictions, dates_range):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=daily_df["date"],
            y=daily_df["balance"],
            mode="lines",
            name="Historical Balance",
            line=dict(color=COLORS["primary"], width=2),
            fill="tozeroy",
            fillcolor=f"rgba(34, 139, 34, 0.1)",
        )
    )

    if predictions:
        pred_dates = predictions["dates"]
        pred_balances = predictions["balances"]

        fig.add_trace(
            go.Scatter(
                x=pred_dates,
                y=pred_balances,
                mode="lines",
                name="Predicted Balance",
                line=dict(color=COLORS["secondary"], width=2, dash="dash"),
                fill="tozeroy",
                fillcolor=f"rgba(50, 205, 50, 0.1)",
            )
        )

    fig.update_layout(
        title="Cashflow Timeline",
        xaxis_title="Date",
        yaxis_title="Balance ($)",
        plot_bgcolor=COLORS["background"],
        paper_bgcolor=COLORS["background"],
        font=dict(color=COLORS["text"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
    )

    fig.update_xaxes(showgrid=True, gridcolor=COLORS["accent"])
    fig.update_yaxes(showgrid=True, gridcolor=COLORS["accent"])

    return fig


def create_trend_chart(daily_df):
    last_30 = daily_df.tail(30).copy()

    colors = [
        COLORS["primary"] if v >= 0 else "#DC143C" for v in last_30["daily_change"]
    ]

    fig = px.bar(
        x=last_30["date"],
        y=last_30["daily_change"],
        color=last_30["daily_change"],
        color_continuous_scale=["#DC143C", COLORS["primary"]],
        title="Daily Cashflow - Last 30 Days",
    )

    fig.update_layout(
        plot_bgcolor=COLORS["background"],
        paper_bgcolor=COLORS["background"],
        font=dict(color=COLORS["text"]),
        xaxis_title="Date",
        yaxis_title="Daily Change ($)",
        showlegend=False,
    )

    return fig


def main():
    apply_green_theme()

    st.markdown(
        """
    <div class="main-header">
        <h1>💰 CashFlow Predictor</h1>
        <p>AI-Powered Cashflow Forecasting for Small Businesses</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.sidebar.columns([1, 1])
    with col1:
        st.markdown("### 📤 Upload Data")
        st.markdown("Upload your Excel or CSV file to get started")

    uploaded_file = st.sidebar.file_uploader(
        "Choose a file",
        type=["csv", "xlsx", "xls"],
        help="Accepts .csv, .xlsx, .xls files",
    )

    prediction_days = st.sidebar.slider("Prediction Period", 30, 90, 30, step=30)

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.sidebar.success("✅ File uploaded successfully!")

            processed_df, daily_df, date_col, amount_col = process_data(df)

            current_balance = processed_df["running_balance"].iloc[-1]
            avg_daily = daily_df["daily_change"].mean()
            total_income = daily_df[daily_df["daily_change"] > 0]["daily_change"].sum()
            total_expense = abs(
                daily_df[daily_df["daily_change"] < 0]["daily_change"].sum()
            )

            st.markdown("### 📊 Dashboard Overview")

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Current Balance", f"${current_balance:,.2f}")
            with m2:
                st.metric("Avg Daily Change", f"${avg_daily:,.2f}")
            with m3:
                st.metric("Total Income", f"${total_income:,.2f}")
            with m4:
                st.metric("Total Expenses", f"${total_expense:,.2f}")

            predictions = predict_cashflow(daily_df, prediction_days)

            if predictions:
                future_balance = predictions["balances"][-1]
                min_future = min(predictions["balances"])
                max_future = max(predictions["balances"])

                st.markdown("### 🔮 Predictions")

                p1, p2, p3 = st.columns(3)
                with p1:
                    st.metric(
                        f"Predicted Balance ({prediction_days}d)",
                        f"${future_balance:,.2f}",
                    )
                with p2:
                    st.metric("Lowest Predicted", f"${min_future:,.2f}")
                with p3:
                    st.metric("Highest Predicted", f"${max_future:,.2f}")

                trend = predictions["trend"]
                trend_emoji = "📈" if trend == "up" else "📉"
                daily_rate = predictions["avg_daily"]

                st.markdown(
                    f"""
                <div class="insight-box">
                    <strong>{trend_emoji} Trend Analysis:</strong> 
                    Your cashflow is predicted to <strong>{trend.upper()}</strong> by 
                    ${abs(daily_rate):,.2f} per day on average.
                </div>
                """,
                    unsafe_allow_html=True,
                )

            st.markdown("### 📈 Cashflow Timeline")
            chart = create_chart(daily_df, predictions, prediction_days)
            st.plotly_chart(chart, use_container_width=True)

            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                trend_chart = create_trend_chart(daily_df)
                st.plotly_chart(trend_chart, use_container_width=True)

            with col_chart2:
                st.markdown("### 📋 Data Preview")
                st.dataframe(processed_df.tail(10), use_container_width=True)

            st.markdown("### 💾 Download Forecast")
            if predictions:
                forecast_df = pd.DataFrame(
                    {
                        "Date": predictions["dates"],
                        "Predicted Daily Change": predictions["predictions"],
                        "Predicted Balance": predictions["balances"],
                    }
                )
                csv = forecast_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Forecast CSV",
                    data=csv,
                    file_name="cashflow_forecast.csv",
                    mime="text/csv",
                )

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("Please ensure your file has date and amount columns.")
    else:
        st.markdown(
            """
        <div class="upload-zone">
            <h3 style="color: #228B22;">📁 Drop your Excel or CSV file here</h3>
            <p style="color: #666;">or click to browse</p>
            <p style="color: #999; font-size: 12px;">Supported formats: .csv, .xlsx, .xls</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown("### 📋 Sample Data Format")
        sample_data = pd.DataFrame(
            {
                "Date": [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-05",
                ],
                "Amount": [5000, -1200, 800, -2500, 1500],
            }
        )
        st.dataframe(sample_data, use_container_width=True)

        st.markdown("""
        ### 🔧 How It Works
        1. **Upload**: Drop your Excel or CSV file with transaction data
        2. **Analyze**: We automatically detect date and amount columns
        3. **Predict**: AI analyzes patterns and forecasts future cashflow
        4. **Decide**: Use insights to make informed business decisions
        """)

    st.markdown("---")
    st.markdown(
        f"""
    <div style="text-align: center; color: #228B22; padding: 20px;">
        <small>CashFlow Predictor | AI Agent for Small Businesses</small>
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
