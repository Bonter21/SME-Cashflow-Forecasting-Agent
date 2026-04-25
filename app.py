import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go

st.set_page_config(page_title="CashFlow Predictor", page_icon="money", layout="wide")

def apply_theme():
    st.markdown("""
    <style>
    .stApp { background-color: white; }
    .main-header { background: linear-gradient(135deg, #228B22, #32CD32); padding: 20px; border-radius: 10px; }
    .main-header h1 { color: white; font-size: 28px; }
    .stButton>button { background: #228B22; color: white; border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

def init_auth():
    if "user" not in st.session_state:
        st.session_state.user = None
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

def login_ui():
    st.markdown("<h1 style='color:#228B22;'>CashFlow Predictor Pro</h1>", unsafe_allow_html=True)
    st.markdown("### Welcome")
    
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if email and password:
            st.session_state.user = {"id": "demo", "email": email, "subscription_tier": "free"}
            st.session_state.logged_in = True
            st.success("Logged in!")
            st.rerun()
    
    if st.button("Demo Login"):
        st.session_state.user = {"id": "demo", "email": "demo@example.com", "subscription_tier": "free"}
        st.session_state.logged_in = True
        st.rerun()

def profile_ui():
    st.markdown("### Profile")
    user = st.session_state.user
    st.write("Email:", user.get('email', 'N/A') if user else 'N/A')
    if st.button("Logout"):
        st.session_state.user = None
        st.session_state.logged_in = False
        st.rerun()

def detect_columns(df):
    dc = ["date", "transaction_date", "timestamp", "datetime"]
    ac = ["amount", "balance", "value", "total"]
    df.columns = df.columns.str.lower().str.strip()
    dc = next((c for c in df.columns if any(d in c for d in dc)), df.columns[0])
    ac = next((c for c in df.columns if any(a in c for a in ac)), df.columns[1])
    return dc, ac

def process_data(df):
    df = df.copy()
    df.columns = df.columns.str.lower().str.strip()
    dc, ac = detect_columns(df)
    df[dc] = pd.to_datetime(df[dc], errors="coerce")
    df = df.dropna(subset=[dc])
    if df[ac].dtype == "object":
        df[ac] = df[ac].str.replace(",", "").astype(float)
    df[ac] = pd.to_numeric(df[ac], errors="coerce")
    df = df.dropna(subset=[ac])
    df = df.sort_values(dc)
    df["running"] = df[ac].cumsum()
    daily = df.groupby(df[dc].dt.date).agg({ac: "sum", "running": "last"}).reset_index()
    daily.columns = ["date", "change", "balance"]
    daily["date"] = pd.to_datetime(daily["date"])
    return df, daily

def predict_cashflow(df, days=30):
    if len(df) < 7:
        return None
    df = df.copy()
    df["dn"] = (df["date"] - df["date"].min()).dt.days
    X = df["dn"].values.reshape(-1, 1)
    y = df["change"].values
    model = LinearRegression()
    model.fit(X, y)
    ld = df["date"].max()
    fd = [ld + timedelta(days=i) for i in range(1, days+1)]
    fx = np.array([(ld - df["date"].min()).days + i for i in range(1, days+1)]).reshape(-1, 1)
    p = model.predict(fx)
    lb = df["balance"].iloc[-1]
    pb = [lb + sum(p[:i+1]) for i in range(len(p))]
    return {"dates": fd, "predictions": p.tolist(), "balances": pb, "trend": "up" if model.coef_[0] > 0 else "down"}

def main():
    init_auth()
    apply_theme()
    
    if not st.session_state.logged_in:
        login_ui()
        return
    
    with st.sidebar:
        st.write(f"**{st.session_state.user.get('email')}**")
        if st.button("Profile"):
            st.session_state.page = "profile"
        if st.button("Logout", use_container_width=True):
            st.session_state.user = None
            st.session_state.logged_in = False
            st.rerun()
    
    if st.session_state.get('page') == 'profile':
        profile_ui()
        return
    
    st.markdown("<div class='main-header'><h1>CashFlow Predictor Pro</h1></div>", unsafe_allow_html=True)
    
    f = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"])
    days = st.slider("Days", 30, 90, 30)
    
    if f:
        try:
            df = pd.read_csv(f) if f.name.endswith(".csv") else pd.read_excel(f)
            pdf, ddf = process_data(df)
            bal = pdf["running"].iloc[-1]
            
            st.metric("Balance", f"${bal:,.2f}")
            st.metric("Avg Daily", f"${ddf['change'].mean():,.2f}")
            
            pred = predict_cashflow(ddf, days)
            if pred:
                st.metric(f"Predicted ({days}d)", f"${pred['balances'][-1]:,.2f}")
                st.write("Trend:", pred['trend'].upper())
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ddf["date"], y=ddf["balance"], mode="lines", name="Balance"))
            if pred:
                fig.add_trace(go.Scatter(x=pred["dates"], y=pred["balances"], mode="lines", name="Predicted", line=dict(dash="dash")))
            st.plotly_chart(fig)
        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()