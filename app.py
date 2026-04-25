import streamlit as st
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import auth_utils

PRICING_TIERS = {
    'free': {'predictions': 10, 'pdf': False, 'arima': False, 'email': False},
    'pro': {'predictions': 100, 'pdf': True, 'arima': True, 'email': False, 'price': 9.99},
    'enterprise': {'predictions': -1, 'pdf': True, 'arima': True, 'email': True, 'price': 49}
}

SMTP_CONFIG = {
    'server': os.getenv('SMTP_SERVER', ''),
    'port': int(os.getenv('SMTP_PORT', 587) or 587),
    'sender': os.getenv('SENDER_EMAIL', ''),
    'password': os.getenv('SENDER_PASSWORD', ''),
    'reset_url': os.getenv('RESET_URL', 'https://your-app.streamlit.app')
}

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    STATSMODELS_AVAILABLE = True
except:
    STATSMODELS_AVAILABLE = False

st.set_page_config(page_title="CashFlow Predictor Pro", page_icon="💰", layout="wide", menu_items={
        'Get Help': 'https://streamlit.io',
        'Report a bug': 'https://github.com/Bonter21/SME-Cashflow-Forecasting-Agent/issues',
        'About': '# CashFlow Predictor Pro - Commercial Edition'
    })

st.markdown("""
    <meta name="mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <meta name="theme-color" content="#228B22">
""", unsafe_allow_html=True)

THEMES = {
    "light": {
        "primary": "#228B22",
        "secondary": "#32CD32",
        "accent": "#90EE90",
        "background": "#FFFFFF",
        "text": "#1A1A1A",
        "light_bg": "#F0FFF0",
        "border": "#2E8B2E",
        "chart_bg": "#FFFFFF",
    },
    "dark": {
        "primary": "#32CD32",
        "secondary": "#90EE90",
        "accent": "#228B22",
        "background": "#1A1A1A",
        "text": "#FFFFFF",
        "light_bg": "#2D2D2D",
        "border": "#32CD32",
        "chart_bg": "#2D2D2D",
    },
}

if "theme" not in st.session_state:
    st.session_state.theme = "light"
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

def get_colors():
    return THEMES[st.session_state.theme]

def init_auth_state():
    if "user" not in st.session_state:
        st.session_state.user = None
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "auth_mode" not in st.session_state:
        st.session_state.auth_mode = "login"

def login_ui():
    COLORS = get_colors()
    st.markdown(f"""
    <div style="text-align: center; padding: 40px 20px;">
        <h1 style="color: {COLORS['primary']};">💰 CashFlow Predictor Pro</h1>
        <p style="color: {COLORS['text']}; font-size: 18px;">AI-Powered Cashflow Forecasting</p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["🔐 Login", "📝 Sign Up"])
    
    with tab1:
        st.markdown("### Welcome Back")
        email = st.text_input("📧 Email", key="login_email")
        password = st.text_input("🔒 Password", type="password", key="login_password")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Login", use_container_width=True):
                if email and password:
                    success, result = auth_utils.verify_login(email, password)
                    if success:
                        st.session_state.user = result
                        st.session_state.logged_in = True
                        st.success("✅ Login successful!")
                        st.rerun()
                    else:
                        st.error(f"❌ {result}")
                else:
                    st.warning("Please enter email and password")
        
        with col2:
            if st.button("Forgot Password?", use_container_width=True):
                if email:
                    token, _ = auth_utils.create_reset_token(email)
                    if token and SMTP_CONFIG['server']:
                        smtp_cfg = SMTP_CONFIG.copy()
                        smtp_cfg['reset_url'] = SMTP_CONFIG['reset_url']
                        success, msg = auth_utils.send_reset_email(email, token, smtp_cfg)
                        if success:
                            st.success("📧 Check your email for reset link")
                        else:
                            st.info("💡 Password reset link generated (configure SMTP for email)")
                            st.code(f"Reset URL: {SMTP_CONFIG['reset_url']}?token={token}")
                    else:
                        st.info("💡 Enter your email and click to generate reset link")
                else:
                    st.warning("Enter your email first")
        
        if st.button("🔐 Demo Login (Try Free)", use_container_width=True):
            st.session_state.user = {"id": "demo", "email": "demo@example.com", "subscription_tier": "free", "predictions_used": 0}
            st.session_state.logged_in = True
            st.success("✅ Demo mode activated!")
            st.rerun()
    
    with tab2:
        st.markdown("### Create Account")
        new_email = st.text_input("📧 Email", key="signup_email")
        new_password = st.text_input("🔒 Password (min 8 chars)", type="password", key="signup_password")
        confirm_password = st.text_input("🔒 Confirm Password", type="password", key="signup_confirm")
        
        agree_terms = st.checkbox("I agree to the [Terms of Service](TOS.md) and [Privacy Policy](PRIVACY.md)")
        
        if st.button("Create Account", use_container_width=True):
            if not agree_terms:
                st.warning("Please agree to Terms of Service")
            elif new_password != confirm_password:
                st.error("❌ Passwords don't match")
            elif len(new_password) < 8:
                st.error("❌ Password must be at least 8 characters")
            else:
                success, result = auth_utils.create_user(new_email, new_password)
                if success:
                    st.success("✅ Account created! Please login.")
                    st.session_state.auth_mode = "login"
                    st.rerun()
                else:
                    st.error(f"❌ {result}")

def profile_ui():
    COLORS = get_colors()
    st.markdown("### 👤 My Profile")
    
    user = st.session_state.user
    user_id = user.get('id') if user else None
    stats = auth_utils.get_user_stats(user_id) if user_id and user_id != 'demo' else {"predictions_used": 0, "subscription_tier": "free", "total_predictions": 0}
    current_tier = stats.get('subscription_tier', 'free')
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📧 Email", user.get('email', 'N/A')[:25] if user else 'N/A')
    with col2:
        tier_display = current_tier.upper()
        if current_tier == 'pro':
            tier_display = '⭐ PRO'
        elif current_tier == 'enterprise':
            tier_display = '🏢 ENTERPRISE'
        st.metric("📦 Plan", tier_display)
    with col3:
        tier_limit = "∞" if current_tier == 'enterprise' else (100 if current_tier == 'pro' else 10)
        remaining = tier_limit - stats.get('predictions_used', 0)
        st.metric("📊 Remaining", f"{remaining}/{tier_limit}")
    
    st.markdown("### 📈 Upgrade Plan")
    
    tier_col1, tier_col2, tier_col3 = st.columns(3)
    with tier_col1:
        is_current = current_tier == 'free'
        border = f"border: 3px solid {COLORS['primary']};" if is_current else ""
        st.markdown(f"""
        <div style="{border} border-radius: 10px; padding: 20px; text-align: center;">
            <h3>🆓 Free</h3>
            <h2>$0</h2>
            <p>10 predictions/mo</p>
            <p>Basic features</p>
            {is_current and '<p>✅ Current</p>' or ''}
        </div>
        """, unsafe_allow_html=True)
    
    with tier_col2:
        is_current = current_tier == 'pro'
        border = f"border: 3px solid {COLORS['primary']};"
        st.markdown(f"""
        <div style="{border} border-radius: 10px; padding: 20px; text-align: center; background: {COLORS['light_bg']};">
            <h3>⭐ Pro</h3>
            <h2>$9.99/mo</h2>
            <p>100 predictions/mo</p>
            <p>PDF Reports ✅</p>
            <p>ARIMA Models ✅</p>
            {is_current and '<p>✅ Current</p>' or ''}
        </div>
        """, unsafe_allow_html=True)
        if not is_current and user_id and user_id != 'demo':
            if st.button("💳 Upgrade to Pro", use_container_width=True):
                try:
                    import billing_utils
                    url, error = billing_utils.create_checkout_session(user_id, 'pro')
                    if url:
                        st.markdown(f"[Click here to pay]({url})")
                    else:
                        st.warning("Stripe not configured. Please contact support.")
                except Exception as e:
                    st.warning("Stripe integration coming soon!")
    
    with tier_col3:
        is_current = current_tier == 'enterprise'
        border = f"border: 3px solid {COLORS['primary']};"
        st.markdown(f"""
        <div style="{border} border-radius: 10px; padding: 20px; text-align: center;">
            <h3>🏢 Enterprise</h3>
            <h2>$49/mo</h2>
            <p>Unlimited</p>
            <p>All Pro features</p>
            <p>API Access ✅</p>
            {is_current and '<p>✅ Current</p>' or ''}
        </div>
        """, unsafe_allow_html=True)
        if not is_current and user_id and user_id != 'demo':
            if st.button("💳 Contact Sales", use_container_width=True):
                st.info("📧 Email: sales@cashflowpredictor.com")
        </div>
        """, unsafe_allow_html=True)
        if st.button("Contact Sales", use_container_width=True):
            st.info("📧 Email: sales@example.com")
    
    st.markdown("### ⚙️ Account Settings")
    
    if st.button("📧 Change Password"):
        st.warning("Feature coming soon")
    
    if st.button("🗑️ Delete Account"):
        st.error("⚠️ This will permanently delete your account and all data.")
        if st.button("⚠️ Yes, Delete Permanently"):
            auth_utils.delete_user(user['id'])
            st.session_state.user = None
            st.session_state.logged_in = False
            st.success("Account deleted.")
            st.rerun()
    
    if st.button("🚪 Logout"):
        st.session_state.user = None
        st.session_state.logged_in = False
        st.rerun()

def history_ui():
    COLORS = get_colors()
    st.markdown(f"### 📜 Prediction History")
    
    user = st.session_state.user
    user_id = user.get('id') if user else None
    
    if not user_id or user_id == 'demo':
        st.warning("History is only available for logged-in users. Use Demo mode for testing.")
        if st.button("⬅️ Back to App"):
            st.session_state.page = None
            st.rerun()
        return
    
    with st.spinner("Loading history..."):
        history = auth_utils.get_prediction_history(user_id, limit=50)
    
    if not history:
        st.info("No predictions yet. Upload a file and make your first prediction!")
    else:
        st.markdown(f"**Total Predictions:** {len(history)}")
        
        history_df = pd.DataFrame(history)
        if 'created_at' in history_df.columns:
            history_df['created_at'] = pd.to_datetime(history_df['created_at']).dt.strftime('%Y-%m-%d %H:%M')
        
        cols = st.columns(3)
        with cols[0]:
            st.metric("Total", len(history))
        with cols[1]:
            status_counts = history_df['status'].value_counts().to_dict()
            st.metric("Excellent", status_counts.get('excellent', 0))
        with cols[2]:
            st.metric("Critical", status_counts.get('critical', 0))
        
        st.markdown("### 📊 Recent Predictions")
        display_df = history_df[['created_at', 'file_name', 'prediction_days', 'current_balance', 'predicted_balance', 'trend', 'status']].head(10)
        display_df['current_balance'] = display_df['current_balance'].apply(lambda x: f"${x:,.2f}" if x else "$0")
        display_df['predicted_balance'] = display_df['predicted_balance'].apply(lambda x: f"${x:,.2f}" if x else "$0")
        st.dataframe(display_df, use_container_width=True)
        
status_icons = {'excellent': 'Green', 'good': 'Green', 'stable': 'Yellow', 'warning': 'Orange', 'critical': 'Red', 'neutral': 'Gray'}
        for idx, row in enumerate(history[:5]):
            status = row.get('status', 'neutral')
            icon = status_icons.get(status, 'Gray')
            trend_icon = 'Up' if row.get('trend') == 'up' else 'Down'
            file_name = row.get('file_name', 'N/A')
            created = row.get('created_at', 'N/A')
            days = row.get('prediction_days', 30)
            conclusion = row.get('conclusion', 'N/A')
            st.markdown(f"""
            <div style="background: {COLORS['light_bg']}; border-left: 4px solid {COLORS['primary']}; padding: 15px; border-radius: 5px; margin: 10px 0;">
                <strong>{icon} {conclusion}</strong><br>
                File: {file_name} | Trend: {trend_icon}<br>
                Date: {created} | Days: {days}
            </div>
            """, unsafe_allow_html=True)
    
    if st.button("⬅️ Back to App"):
        st.session_state.page = None
        st.rerun()

def apply_theme():
    COLORS = get_colors()
    st.markdown(
        f"""
<style>
         /* Mobile Optimization */
         @media (max-width: 768px) {
             .stApp {
                 padding: 0 !important;
             }
             .main-header {
                 padding: 15px !important;
                 margin-bottom: 10px !important;
             }
             .main-header h1 {
                 font-size: 24px !important;
             }
             div[data-testid="stMetric"] {
                 padding: 10px !important;
             }
             .stButton>button {
                 padding: 8px 15px !important;
                 font-size: 14px !important;
             }
         }
         /* Mobile Touch Optimization */
         @media (hover: none) {
             button { 
                 min-height: 44px;
                 touch-action: manipulation;
             }
             .stDownloadButton > button {
                 min-height: 48px;
             }
         }
         .stApp {
             background-color: BACKGROUND;
             -webkit-tap-highlight-color: transparent;
         }
         .main-header {
             background: linear-gradient(135deg, PRIMARY, SECONDARY);
             padding: 20px;
             border-radius: 10px;
             margin-bottom: 20px;
         }
         .main-header h1 {
             color: white !important;
             margin: 0;
             font-size: 32px;
             font-weight: bold;
         }
         .main-header p {
             color: ACCENT;
             margin: 5px 0 0 0;
         }
         .stSidebar {
             background-color: LIGHT_BG;
             width: 280px !important;
         }
         .upload-zone {
             border: 3px dashed PRIMARY;
             border-radius: 15px;
             padding: 40px;
             text-align: center;
             background-color: LIGHT_BG;
             margin: 20px 0;
         }
         .insight-box {
             background: LIGHT_BG;
             border-left: 4px solid PRIMARY;
             padding: 15px;
             border-radius: 5px;
             margin: 10px 0;
         }
         div[data-testid="stMetric"] {
             background-color: LIGHT_BG;
             border: 1px solid ACCENT;
             border-radius: 10px;
             padding: 15px;
         }
         div[data-testid="stMetric"] label {
             color: PRIMARY !important;
         }
         div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
             color: TEXT !important;
         }
         .stButton>button {
             background: linear-gradient(135deg, PRIMARY, SECONDARY);
             color: white;
             border: none;
             border-radius: 8px;
             padding: 10px 25px;
         }
         .stButton>button:hover {
             background: linear-gradient(135deg, SECONDARY, PRIMARY);
         }
         .goal-box {
             background: rgba(PRIMARY, 0.1);
             border: 2px solid PRIMARY;
             border-radius: 10px;
             padding: 15px;
             margin: 10px 0;
         }
         .anomaly-alert {
             background: #FFF3CD;
             border-left: 4px solid #FFC107;
             padding: 10px 15px;
             border-radius: 5px;
             margin: 5px 0;
         }
         .loading-spinner {
             border: 3px solid LIGHT_BG;
             border-top: 3px solid PRIMARY;
             border-radius: 50%;
             width: 30px;
             height: 30px;
             animation: spin 1s linear infinite;
         }
         @keyframes spin {
             0% { transform: rotate(0deg); }
             100% { transform: rotate(360deg); }
         }
         /* Mobile Sticky Header */
         .st-emotion-cache-1w0r9p2 {
             position: sticky !important;
             top: 0 !important;
             z-index: 100 !important;
         }
         /* Mobile Charts */
         .js-plotly-plot {
             max-width: 100% !important;
         }
         </style>
        """.replace("BACKGROUND", COLORS["background"]).replace("PRIMARY", COLORS["primary"]).replace("SECONDARY", COLORS["secondary"]).replace("ACCENT", COLORS["accent"]).replace("LIGHT_BG", COLORS["light_bg"]).replace("TEXT", COLORS["text"]),
        unsafe_allow_html=True,
    )

def detect_columns(df):
    date_cols = ["date", "transaction_date", "trans_date", "timestamp", "datetime", "posted_date", "value_date"]
    amount_cols = ["amount", "balance", "value", "total", "sum", "net_amount", "transaction_amount"]
    
    df.columns = df.columns.str.lower().str.strip()
    
    date_col = next((c for c in df.columns if any(d in c for d in date_cols)), df.columns[0])
    amount_col = next((c for c in df.columns if any(a in c for a in amount_cols)), df.columns[1] if len(df.columns) > 1 else df.columns[0])
    
    return date_col, amount_col

def detect_date_format(dates):
    sample = dates.dropna().iloc[:5] if len(dates) > 5 else dates.dropna()
    formats = ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d", "%d-%m-%Y", "%m-%d-%Y"]
    for fmt in formats:
        try:
            pd.to_datetime(sample, format=fmt)
            return fmt
        except:
            continue
    return None

def process_data(df):
    with st.spinner("Processing data..."):
        df = df.copy()
        df.columns = df.columns.str.lower().str.strip()
        
        date_col, amount_col = detect_columns(df)
        
        date_format = detect_date_format(df[date_col])
        if date_format:
            df[date_col] = pd.to_datetime(df[date_col], format=date_format, errors="coerce")
        else:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        
        df = df.dropna(subset=[date_col])
        
        if df[amount_col].dtype == "object":
            df[amount_col] = df[amount_col].str.replace(",", "").str.replace("$", "").str.replace(" ", "")
            df[amount_col] = pd.to_numeric(df[amount_col], errors="coerce")
        else:
            df[amount_col] = pd.to_numeric(df[amount_col], errors="coerce")
        
        df = df.dropna(subset=[amount_col])
        df = df.sort_values(date_col)
        df = df.drop_duplicates(subset=[date_col], keep="last")
        
        df["running_balance"] = df[amount_col].cumsum()
        
        daily_df = df.groupby(df[date_col].dt.date).agg({amount_col: "sum", "running_balance": "last"}).reset_index()
        daily_df.columns = ["date", "daily_change", "balance"]
        daily_df["date"] = pd.to_datetime(daily_df["date"])
        
        return df, daily_df, date_col, amount_col

def detect_seasonality(daily_df):
    if len(daily_df) < 14:
        return {"detected": False, "pattern": "insufficient_data"}
    
    daily_df = daily_df.copy()
    daily_df["day_of_week"] = daily_df["date"].dt.dayofweek
    daily_df["day_name"] = daily_df["date"].dt.day_name()
    daily_df["week_of_year"] = daily_df["date"].dt.isocalendar().week
    daily_df["month"] = daily_df["date"].dt.month
    
    weekday_avg = daily_df.groupby("day_of_week")["daily_change"].mean()
    monthly_avg = daily_df.groupby("month")["daily_change"].mean()
    
    weekday_std = weekday_avg.std()
    monthly_std = monthly_avg.std()
    
    weekly_pattern = weekday_avg.idxmax()
    best_month = monthly_avg.idxmax()
    worst_month = monthly_avg.idxmin()
    
    pattern_type = "none"
    if weekday_std > abs(weekday_avg.mean()) * 0.3:
        pattern_type = "weekly"
    elif monthly_std > abs(monthly_avg.mean()) * 0.3:
        pattern_type = "monthly"
    
    return {
        "detected": pattern_type != "none",
        "pattern": pattern_type,
        "best_day": weekday_avg.idxmax(),
        "best_day_name": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][weekly_pattern],
        "worst_day": weekday_avg.idxmin(),
        "worst_day_name": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][weekday_avg.idxmin()],
        "best_month": best_month,
        "worst_month": worst_month,
        "weekday_averages": weekday_avg.to_dict(),
        "monthly_averages": monthly_avg.to_dict(),
    }

def detect_anomalies(daily_df, threshold=2.5):
    if len(daily_df) < 7:
        return []
    
    mean_val = daily_df["daily_change"].mean()
    std_val = daily_df["daily_change"].std()
    
    anomalies = []
    for idx, row in daily_df.iterrows():
        z_score = abs((row["daily_change"] - mean_val) / std_val) if std_val > 0 else 0
        if z_score > threshold:
            anomalies.append({
                "date": row["date"],
                "amount": row["daily_change"],
                "z_score": z_score,
                "type": "high" if row["daily_change"] > mean_val else "low"
            })
    
    return anomalies

def predict_arima(daily_df, days=30):
    if not STATSMODELS_AVAILABLE or len(daily_df) < 30:
        return None
    
    try:
        daily_df = daily_df.set_index('date')
        model = ARIMA(daily_df['daily_change'], order=(1, 1, 1))
        fitted = model.fit()
        forecast = fitted.forecast(steps=days)
        conf_int = fitted.get_forecast(steps=days).conf_int()
        
        last_balance = daily_df['balance'].iloc[-1]
        predicted_balances = [last_balance]
        current = last_balance
        for pred in forecast:
            current += pred
            predicted_balances.append(current)
        
        return {
            "predictions": forecast.tolist(),
            "balances": predicted_balances[1:],
            "confidence_upper": [b + u for b, u in zip(predicted_balances[1:], conf_int.iloc[:, 1].values)],
            "confidence_lower": [b - l for b, l in zip(predicted_balances[1:], conf_int.iloc[:, 0].values)],
            "avg_daily": forecast.mean(),
        }
    except:
        return None

def predict_sarima(daily_df, days=30):
    if not STATSMODELS_AVAILABLE or len(daily_df) < 60:
        return None
    
    try:
        daily_df = daily_df.set_index('date')
        model = SARIMAX(daily_df['daily_change'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
        fitted = model.fit(disp=False)
        forecast = fitted.forecast(steps=days)
        
        last_balance = daily_df['balance'].iloc[-1]
        predicted_balances = [last_balance]
        current = last_balance
        for pred in forecast:
            current += pred
            predicted_balances.append(current)
        
        return {
            "predictions": forecast.tolist(),
            "balances": predicted_balances[1:],
            "avg_daily": forecast.mean(),
        }
    except:
        return None

def send_email_alert(recipient, subject, body, attachment=None, attachment_name=None):
    smtp_server = SMTP_CONFIG['server']
    smtp_port = SMTP_CONFIG['port']
    sender_email = SMTP_CONFIG['sender']
    sender_password = SMTP_CONFIG['password']
    
    if not smtp_server or not sender_email:
        return False, "Email not configured. Set SMTP secrets in Streamlit settings."
    
    try:
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'html'))
        
        if attachment:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(attachment)
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f'attachment; filename={attachment_name}')
            msg.attach(part)
        
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
        
        return True, "Email sent successfully"
    except Exception as e:
        return False, str(e)

def create_budget_alerts(daily_df, predictions, budget_threshold=1000):
    alerts = []
    current_balance = daily_df['balance'].iloc[-1]
    avg_daily = daily_df['daily_change'].mean()
    std_daily = daily_df['daily_change'].std()
    
    if predictions:
        for i, (date, balance) in enumerate(zip(predictions['dates'], predictions['balances'])):
            if balance < budget_threshold:
                alerts.append({
                    'date': date,
                    'balance': balance,
                    'alert_type': 'low_balance',
                    'message': f"Balance (${balance:,.2f}) below threshold (${budget_threshold:,})"
                })
            
            daily_change = predictions['predictions'][i] if i < len(predictions['predictions']) else 0
            if daily_change < -abs(avg_daily) - 2 * std_daily:
                alerts.append({
                    'date': date,
                    'balance': balance,
                    'alert_type': 'high_expense',
                    'message': f"Unusual expense: ${abs(daily_change):,.2f}"
                })
    
    return alerts

def compare_files(file_list):
    if len(file_list) < 2:
        return None
    
    comparisons = []
    for i, (df, name) in enumerate(file_list):
        processed_df, daily_df, _, _ = process_data(df)
        current_balance = processed_df['running_balance'].iloc[-1]
        avg_daily = daily_df['daily_change'].mean()
        total_income = daily_df[daily_df['daily_change'] > 0]['daily_change'].sum()
        total_expense = abs(daily_df[daily_df['daily_change'] < 0]['daily_change'].sum())
        
        comparisons.append({
            'name': name,
            'current_balance': current_balance,
            'avg_daily': avg_daily,
            'total_income': total_income,
            'total_expense': total_expense,
            'data_points': len(daily_df),
            'daily_df': daily_df,
        })
    
    if len(comparisons) >= 2:
        max_balance = max(comparisons, key=lambda x: x['current_balance'])
        min_balance = min(comparisons, key=lambda x: x['current_balance'])
        best_performer = max_balance['name']
        worst_performer = min_balance['name']
        
        return {
            'files': comparisons,
            'max_balance': max_balance['current_balance'],
            'min_balance': min_balance['current_balance'],
            'best_performer': best_performer,
            'worst_performer': worst_performer,
            'balance_diff': max_balance['current_balance'] - min_balance['current_balance'],
        }
    
    return None

def predict_cashflow_ensemble(daily_df, days=30):
    if len(daily_df) < 7:
        return None
    
    with st.spinner("Running ensemble predictions..."):
        daily_df = daily_df.copy()
        daily_df["day_num"] = (daily_df["date"] - daily_df["date"].min()).dt.days
        daily_df["day_of_week"] = daily_df["date"].dt.dayofweek
        daily_df["day_of_month"] = daily_df["date"].dt.day
        daily_df["week_of_year"] = daily_df["date"].dt.isocalendar().week
        
        X = daily_df[["day_num", "day_of_week", "day_of_month", "week_of_year"]].values
        y = daily_df["daily_change"].values
        
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(alpha=1.0),
            "Random Forest": RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42),
        }
        
        predictions = {}
        scores = {}
        
        for name, model in models.items():
            model.fit(X, y)
            pred = model.predict(X)
            rmse = np.sqrt(mean_squared_error(y, pred))
            r2 = r2_score(y, pred)
            predictions[name] = pred.mean()
            scores[name] = {"rmse": rmse, "r2": r2}
        
        last_date = daily_df["date"].max()
        last_row = daily_df.iloc[-1]
        
        future_dates = [last_date + timedelta(days=i) for i in range(1, days + 1)]
        future_data = []
        for i, date in enumerate(future_dates):
            future_data.append([
                last_row["day_num"] + i + 1,
                date.dayofweek,
                date.day,
                date.isocalendar().week,
            ])
        
        future_X = np.array(future_data)
        
        ensemble_predictions = []
        confidence_ranges = []
        
        for name, model in models.items():
            model_preds = model.predict(future_X)
            ensemble_predictions.append(model_preds)
        
        ensemble_predictions = np.array(ensemble_predictions)
        mean_predictions = ensemble_predictions.mean(axis=0)
        std_predictions = ensemble_predictions.std(axis=0)
        
        last_balance = daily_df["balance"].iloc[-1]
        predicted_balances = [last_balance]
        current = last_balance
        for pred in mean_predictions:
            current += pred
            predicted_balances.append(current)
        
        predicted_balances = predicted_balances[1:]
        
        confidence_upper = []
        confidence_lower = []
        for i, (bal, std) in enumerate(zip(predicted_balances, std_predictions)):
            confidence_upper.append(bal + 1.96 * std * np.sqrt(i + 1))
            confidence_lower.append(bal - 1.96 * std * np.sqrt(i + 1))
        
        best_model = max(scores.keys(), key=lambda x: scores[x]["r2"])
        
        return {
            "dates": future_dates,
            "predictions": mean_predictions.tolist(),
            "balances": predicted_balances,
            "confidence_upper": confidence_upper,
            "confidence_lower": confidence_lower,
            "avg_daily": mean_predictions.mean(),
            "trend": "up" if mean_predictions.mean() > 0 else "down",
            "model_scores": scores,
            "best_model": best_model,
            "ensemble_std": std_predictions.tolist(),
        }

def create_advanced_chart(daily_df, predictions):
    COLORS = get_colors()
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.1, subplot_titles=("Cashflow Timeline", "Daily Changes"))
    
    fig.add_trace(go.Scatter(x=daily_df["date"], y=daily_df["balance"], mode="lines+markers", name="Historical Balance", line=dict(color=COLORS["primary"], width=2), marker=dict(size=4)), row=1, col=1)
    
    if predictions:
        pred_dates = predictions["dates"]
        fig.add_trace(go.Scatter(x=pred_dates, y=predictions["balances"], mode="lines", name="Predicted Balance", line=dict(color=COLORS["secondary"], width=2, dash="dash"), fill="tonexty", fillcolor=f"rgba(50, 205, 50, 0.1)"), row=1, col=1)
        fig.add_trace(go.Scatter(x=pred_dates, y=predictions["confidence_upper"], mode="lines", name="Upper Confidence", line=dict(color=COLORS["secondary"], width=1, dash="dot"), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=pred_dates, y=predictions["confidence_lower"], mode="lines", name="Lower Confidence", line=dict(color=COLORS["secondary"], width=1, dash="dot"), showlegend=False, fill="tonexty", fillcolor=f"rgba(50, 205, 50, 0.05)"), row=1, col=1)
    
    last_30 = daily_df.tail(30)
    colors = [COLORS["primary"] if v >= 0 else "#DC143C" for v in last_30["daily_change"]]
    fig.add_trace(go.Bar(x=last_30["date"], y=last_30["daily_change"], marker_color=colors, name="Daily Change"), row=2, col=1)
    
    fig.update_layout(height=600, showlegend=True, plot_bgcolor=COLORS["chart_bg"], paper_bgcolor=COLORS["chart_bg"], font=dict(color=COLORS["text"]), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_xaxes(showgrid=True, gridcolor=COLORS["light_bg"])
    fig.update_yaxes(showgrid=True, gridcolor=COLORS["light_bg"])
    
    return fig

def create_seasonality_chart(daily_df, seasonality):
    COLORS = get_colors()
    
    if seasonality["pattern"] == "weekly":
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        values = [seasonality["weekday_averages"].get(i, 0) for i in range(7)]
        title = "Weekly Pattern"
    else:
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        available_months = list(seasonality["monthly_averages"].keys())
        values = [seasonality["monthly_averages"].get(i, 0) for i in available_months]
        title = "Monthly Pattern"
        days = [months[i-1] if i <= 12 else str(i) for i in available_months]
    
    colors = [COLORS["primary"] if v >= 0 else "#DC143C" for v in values]
    fig = px.bar(x=days, y=values, color=values, color_continuous_scale=["#DC143C", COLORS["primary"]], title=title)
    fig.update_layout(plot_bgcolor=COLORS["chart_bg"], paper_bgcolor=COLORS["chart_bg"], font=dict(color=COLORS["text"]), showlegend=False, height=300)
    fig.update_yaxes(title="Average Daily Change ($)")
    
    return fig

def generate_conclusion(predictions, current_balance, daily_df, prediction_days, seasonality, anomalies):
    if not predictions:
        return {"status": "neutral", "title": "⚠️ Insufficient Data", "message": "Not enough historical data. Please upload more transactions (minimum 7 days).", "recommendations": ["Upload more historical transactions", "Ensure data covers at least one month"]}
    
    future_balance = predictions["balances"][-1]
    min_future = min(predictions["balances"])
    max_future = max(predictions["balances"])
    avg_daily = predictions["avg_daily"]
    trend = predictions["trend"]
    
    balance_change = future_balance - current_balance
    change_percent = (balance_change / abs(current_balance)) * 100 if current_balance != 0 else 0
    
    recent_volatility = daily_df["daily_change"].std()
    avg_daily_abs = abs(daily_df["daily_change"].mean())
    volatility_ratio = recent_volatility / avg_daily_abs if avg_daily_abs > 0 else 0
    
    positive_days = sum(1 for p in predictions["predictions"] if p > 0)
    negative_days = sum(1 for p in predictions["predictions"] if p < 0)
    
    recommendations = []
    
    if trend == "up" and balance_change > 0:
        if change_percent > 20:
            status = "excellent"
            title = "📈 Excellent Cashflow Outlook"
            message = f"Your cashflow is projected to GROW significantly over the next {prediction_days} days. Balance expected to increase by ${balance_change:,.2f} ({change_percent:.1f}%)."
            recommendations = ["Consider investing excess cash", "Build additional cash reserves", "Explore expansion opportunities"]
        else:
            status = "good"
            title = "📈 Positive Cashflow Trend"
            message = f"Your cashflow shows a healthy upward trend. Balance projected to grow by ${balance_change:,.2f} ({change_percent:.1f}%) over {prediction_days} days."
            recommendations = ["Continue current practices", "Monitor trends monthly", "Consider modest investments"]
    elif trend == "down" and balance_change < 0:
        if abs(change_percent) > 20 or min_future < 0:
            status = "critical"
            title = "📉 Critical Cashflow Alert"
            message = f"WARNING: Your cashflow is declining significantly. Balance may drop by ${abs(balance_change):,.2f} and could go as low as ${min_future:,.2f}. Immediate action required!"
            recommendations = ["URGENT: Review and reduce non-essential expenses", "Accelerate accounts receivable collection", "Negotiate extended payment terms with suppliers", "Consider short-term financing options"]
        else:
            status = "warning"
            title = "📉 Declining Cashflow"
            message = f"Your cashflow shows a downward trend. Balance expected to decrease by ${abs(balance_change):,.2f} ({abs(change_percent):.1f}%) over {prediction_days} days."
            recommendations = ["Review recent expenses for potential cuts", "Follow up on outstanding invoices", "Monitor cashflow weekly"]
    else:
        status = "stable"
        title = "➡️ Stable Cashflow"
        message = f"Your cashflow is expected to remain relatively stable over the next {prediction_days} days, with minimal change of ${balance_change:,.2f}."
        recommendations = ["Maintain current financial practices", "Document all transactions for better tracking", "Look for opportunities to increase revenue"]
    
    if volatility_ratio > 1.5:
        message += f"\n\n⚠️ High volatility detected. Predictions may vary significantly."
        recommendations.append("Build extra buffer for unexpected expenses")
    
    if seasonality["detected"]:
        message += f"\n\n�� Seasonality detected ({seasonality['pattern']}). Best day: {seasonality.get('best_day_name', 'N/A')}"
    
    if anomalies:
        message += f"\n\n⚠️ {len(anomalies)} unusual transactions detected."
        recommendations.append("Review flagged transactions")
    
    return {"status": status, "title": title, "message": message, "recommendations": recommendations, "balance_change": balance_change, "change_percent": change_percent, "min_future": min_future, "max_future": max_future}

def display_conclusion(conclusion):
    COLORS = get_colors()
    status_colors = {"excellent": "#228B22", "good": "#32CD32", "stable": "#90EE90", "warning": "#FFA500", "critical": "#DC143C", "neutral": "#808080"}
    
    color = status_colors.get(conclusion["status"], COLORS["primary"])
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {color}15, {color}25); border: 2px solid {color}; border-radius: 15px; padding: 25px; margin: 20px 0;">
        <h2 style="color: {color}; margin-top: 0;">{conclusion["title"]}</h2>
        <p style="font-size: 16px; line-height: 1.6;">{conclusion["message"]}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 💡 Recommendations")
    for i, rec in enumerate(conclusion["recommendations"], 1):
        st.markdown(f"""
        <div style="background: {COLORS["light_bg"]}; border-left: 4px solid {color}; padding: 12px 15px; margin: 8px 0; border-radius: 0 8px 8px 0;">
            <strong>{i}.</strong> {rec}
        </div>
        """, unsafe_allow_html=True)

def generate_pdf_report(predictions, current_balance, daily_df, prediction_days, seasonality, anomalies, conclusion):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=18, textColor=colors.HexColor('#228B22'), alignment=TA_CENTER, spaceAfter=20)
    heading_style = ParagraphStyle('CustomHeading', parent=styles['Heading2'], fontSize=14, textColor=colors.HexColor('#228B22'), spaceAfter=10)
    normal_style = ParagraphStyle('CustomNormal', parent=styles['Normal'], fontSize=10, spaceAfter=8)
    
    elements = []
    elements.append(Paragraph("CASHFLOW PREDICTION REPORT", title_style))
    elements.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
    elements.append(Spacer(1, 20))
    
    summary_data = [
        ['Metric', 'Value'],
        ['Current Balance', f"${current_balance:,.2f}"],
        ['Predicted Balance', f"${predictions['balances'][-1]:,.2f}"],
        ['95% Confidence Range', f"${predictions['confidence_lower'][-1]:,.2f} - ${predictions['confidence_upper'][-1]:,.2f}"],
        ['Daily Trend', f"${predictions['avg_daily']:,.2f}/day ({predictions['trend']})"],
        ['Best Model', predictions['best_model']],
    ]
    summary_table = Table(summary_data, colWidths=[2.5*inch, 3*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#228B22')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#90EE90')),
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 20))
    
    elements.append(Paragraph("AI CONCLUSION", heading_style))
    elements.append(Paragraph(f"<b>{conclusion['title']}</b>", normal_style))
    elements.append(Paragraph(conclusion['message'].replace('\n', '<br/>'), normal_style))
    elements.append(Spacer(1, 10))
    
    elements.append(Paragraph("Recommendations:", normal_style))
    for i, rec in enumerate(conclusion['recommendations'], 1):
        elements.append(Paragraph(f"{i}. {rec}", normal_style))
    
    if seasonality['detected']:
        elements.append(Spacer(1, 15))
        elements.append(Paragraph("Seasonality Insights:", heading_style))
        elements.append(Paragraph(f"Pattern: {seasonality['pattern'].title()}", normal_style))
        elements.append(Paragraph(f"Best Day: {seasonality.get('best_day_name', 'N/A')}", normal_style))
    
    if anomalies:
        elements.append(Spacer(1, 15))
        elements.append(Paragraph("Anomalies Detected:", heading_style))
        anomaly_data = [['Date', 'Amount', 'Z-Score']]
        for a in anomalies[:10]:
            anomaly_data.append([a['date'].strftime('%Y-%m-%d'), f"${a['amount']:,.2f}", f"{a['z_score']:.2f}"])
        anomaly_table = Table(anomaly_data, colWidths=[2*inch, 2*inch, 1.5*inch])
        anomaly_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#FFC107')),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ]))
        elements.append(anomaly_table)
    
    elements.append(Spacer(1, 30))
    elements.append(Paragraph("<i>This is an AI-generated prediction for reference only. Consult a financial advisor for important decisions.</i>", normal_style))
    
    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()

def main():
    init_auth_state()
    apply_theme()
    
    if not st.session_state.logged_in:
        login_ui()
        return
    
    with st.sidebar:
        st.markdown(f"### 👤 {st.session_state.user.get('email', 'User')}")
        
        st.markdown("### 🎨 Settings")
        if st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode):
            st.session_state.theme = "dark"
            st.session_state.dark_mode = True
        else:
            st.session_state.theme = "light"
            st.session_state.dark_mode = False
        st.markdown("---")
        
        st.markdown("### 📤 Upload Data")
        uploaded_files = st.file_uploader("Choose files (select multiple)", type=["csv", "xlsx", "xls"], accept_multiple_files=True)
        uploaded_file = st.session_state.get('uploaded_file', None)
        
        prediction_days = st.slider("Prediction Period", 30, 90, 30, step=30)
        
        st.markdown("---")
        st.markdown("### 🎯 Cash Goal")
        goal_enabled = st.checkbox("Set cash target")
        goal_amount = 0
        if goal_enabled:
            goal_amount = st.number_input("Target Balance ($)", value=0, step=1000)
        
        st.markdown("---")
        st.markdown("### 🔔 Budget Alerts")
        alert_enabled = st.checkbox("Enable budget alerts")
        alert_threshold = 0
        if alert_enabled:
            alert_threshold = st.number_input("Alert when balance < ($)", value=1000, step=500)
        
        st.markdown("---")
        st.markdown("### 📧 Email Notifications")
        email_enabled = st.checkbox("Enable email alerts")
        email_recipient = ""
        if email_enabled:
            email_recipient = st.text_input("Email address", value=st.session_state.user.get('email', ''))
            if st.button("Send Test Email"):
                if email_recipient:
                    success, message = send_email_alert(email_recipient, "Test from CashFlow Predictor", "<p>This is a test email from CashFlow Predictor Pro.</p>")
                    if success:
                        st.success(message)
                    else:
                        st.warning(f"Email not configured: {message}")
        
        with st.expander("⚙️ ARIMA Settings"):
            use_arima = st.checkbox("Enable ARIMA predictions")
            if use_arima and not STATSMODELS_AVAILABLE:
                st.warning("ARIMA requires statsmodels. Install: pip install statsmodels")
        
        if st.button("👤 Profile"):
            st.session_state.page = "profile"
        
        if st.button("📜 History"):
            st.session_state.page = "history"
        
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.user = None
            st.session_state.logged_in = False
            st.rerun()
        
        if st.button("🔄 Reset", use_container_width=True):
            st.cache_data.clear()
            st.session_state.clear()
            st.rerun()
    
    if st.session_state.get('page') == 'profile':
        profile_ui()
        return
    
    if st.session_state.get('page') == 'history':
        history_ui()
        return
    
    st.markdown("""
    <div class="main-header">
        <h1>💰 CashFlow Predictor Pro</h1>
        <p>AI-Powered Cashflow Forecasting with Ensemble & ARIMA Models</p>
    </div>
    """, unsafe_allow_html=True)
    
    if uploaded_files and len(uploaded_files) > 0:
        if len(uploaded_files) >= 2:
            st.markdown("### 📊 Multi-File Comparison")
            file_data = []
            for f in uploaded_files:
                if f.name.endswith(".csv"):
                    df_temp = pd.read_csv(f)
                else:
                    df_temp = pd.read_excel(f)
                file_data.append((df_temp, f.name))
            
            comparison = compare_files(file_data)
            
            if comparison:
                comp1, comp2 = st.columns(2)
                with comp1:
                    st.markdown(f"**🏆 Best: {comparison['best_performer']}**")
                    st.metric("Max Balance", f"${comparison['max_balance']:,.2f}")
                with comp2:
                    st.markdown(f"**⚠️ Needs Attention: {comparison['worst_performer']}**")
                    st.metric("Min Balance", f"${comparison['min_balance']:,.2f}")
                
                st.info(f"Difference: ${comparison['balance_diff']:,.2f}")
                
                st.markdown("### 📈 Comparison Chart")
                fig_comp = go.Figure()
                for comp in comparison['files']:
                    fig_comp.add_trace(go.Scatter(x=comp['daily_df']['date'], y=comp['daily_df']['balance'], mode='lines', name=comp['name']))
                fig_comp.update_layout(title="Balance Comparison", plot_bgcolor=get_colors()['chart_bg'], paper_bgcolor=get_colors()['chart_bg'], font=dict(color=get_colors()['text']))
                st.plotly_chart(fig_comp, use_container_width=True)
            
            uploaded_file = uploaded_files[0]
        else:
            uploaded_file = uploaded_files[0]
        
        try:
            with st.spinner("Loading file..."):
                if uploaded_file.name.endswith(".csv"):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
            
            st.sidebar.success("✅ File loaded successfully!")
            
            processed_df, daily_df, date_col, amount_col = process_data(df)
            
            current_balance = processed_df["running_balance"].iloc[-1]
            avg_daily = daily_df["daily_change"].mean()
            total_income = daily_df[daily_df["daily_change"] > 0]["daily_change"].sum()
            total_expense = abs(daily_df[daily_df["daily_change"] < 0]["daily_change"].sum())
            
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
            
            if goal_enabled and goal_amount > 0:
                progress = min(100, (current_balance / goal_amount) * 100)
                st.markdown(f"""
                <div class="goal-box">
                    <h4>🎯 Cash Goal Progress</h4>
                    <p>Target: ${goal_amount:,} | Current: ${current_balance:,.2f} | Progress: {progress:.1f}%</p>
                    <div style="background: #ddd; border-radius: 10px; height: 20px; width: 100%;">
                        <div style="background: linear-gradient(90deg, #228B22, #32CD32); border-radius: 10px; height: 20px; width: {progress:.1f}%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            seasonality = detect_seasonality(daily_df)
            anomalies = detect_anomalies(daily_df)
            
            predictions = predict_cashflow_ensemble(daily_df, prediction_days)
            
            if predictions:
                future_balance = predictions["balances"][-1]
                min_future = min(predictions["balances"])
                max_future = max(predictions["balances"])
                
                st.markdown("### 🔮 Predictions (Ensemble Model)")
                p1, p2, p3, p4 = st.columns(4)
                with p1:
                    st.metric(f"Predicted ({prediction_days}d)", f"${future_balance:,.2f}")
                with p2:
                    st.metric("Lowest (95% CI)", f"${predictions['confidence_lower'][-1]:,.2f}")
                with p3:
                    st.metric("Highest (95% CI)", f"${predictions['confidence_upper'][-1]:,.2f}")
                with p4:
                    st.metric("Best Model", predictions["best_model"][:15])
                
                trend = predictions["trend"]
                trend_emoji = "📈" if trend == "up" else "📉"
                
                st.markdown(f"""
                <div class="insight-box">
                    <strong>{trend_emoji} Trend:</strong> Cashflow trending <strong>{trend.upper()}</strong> by ${abs(predictions["avg_daily"]):,.2f}/day
                    | <strong>R²:</strong> {predictions["model_scores"][predictions["best_model"]]["r2"]:.3f}
                </div>
                """, unsafe_allow_html=True)
                
                if seasonality["detected"]:
                    with st.expander("📅 Seasonality Insights"):
                        st.plotly_chart(create_seasonality_chart(daily_df, seasonality), use_container_width=True)
                        st.info(f"Pattern: {seasonality['pattern'].title()} | Best: {seasonality.get('best_day_name', 'N/A')} | Worst: {seasonality.get('worst_day_name', 'N/A')}")
                
                if anomalies:
                    with st.expander(f"⚠️ Anomalies Detected ({len(anomalies)})"):
                        anomaly_df = pd.DataFrame(anomalies)
                        st.dataframe(anomaly_df, use_container_width=True)
                
                if alert_enabled and predictions:
                    budget_alerts = create_budget_alerts(daily_df, predictions, alert_threshold)
                    if budget_alerts:
                        with st.expander(f"🔔 Budget Alerts ({len(budget_alerts)})"):
                            for alert in budget_alerts[:5]:
                                st.warning(f"📅 {alert['date'].strftime('%Y-%m-%d')}: {alert['message']}")
                
                if 'use_arima' in dir() and use_arima and STATSMODELS_AVAILABLE:
                    with st.spinner("Running ARIMA model..."):
                        arima_pred = predict_arima(daily_df, prediction_days)
                        sarima_pred = predict_sarima(daily_df, prediction_days)
                    
                    if arima_pred:
                        with st.expander("📊 ARIMA Predictions"):
                            st.metric("ARIMA Predicted Balance", f"${arima_pred['balances'][-1]:,.2f}")
                            st.metric("ARIMA Avg Daily", f"${arima_pred['avg_daily']:,.2f}")
                    
                    if sarima_pred:
                        with st.expander("📊 SARIMA Seasonal Predictions"):
                            st.metric("SARIMA Predicted Balance", f"${sarima_pred['balances'][-1]:,.2f}")
                            st.metric("SARIMA Avg Daily", f"${sarima_pred['avg_daily']:,.2f}")
                
                conclusion = generate_conclusion(predictions, current_balance, daily_df, prediction_days, seasonality, anomalies)
                
                st.markdown("### 🎯 AI Conclusion")
                display_conclusion(conclusion)
                
                # Save prediction to history
                if predictions and st.session_state.get('logged_in') and st.session_state.get('user'):
                    user_id = st.session_state.user.get('id')
                    if user_id and user_id != 'demo':
                        prediction_data = {
                            'days': prediction_days,
                            'current_balance': current_balance,
                            'predicted_balance': future_balance,
                            'trend': predictions['trend']
                        }
                        auth_utils.save_prediction(user_id, uploaded_file.name if uploaded_file else 'N/A', prediction_data, conclusion)
                        st.toast("💾 Prediction saved to history")
            
            st.markdown("### 📈 Cashflow Analysis")
            st.plotly_chart(create_advanced_chart(daily_df, predictions), use_container_width=True)
            
            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                st.markdown("### 📋 Data Preview")
                st.dataframe(processed_df.tail(10), use_container_width=True)
            
            with col_chart2:
                st.markdown("### 📊 Model Performance")
                if predictions:
                    for model, scores in predictions["model_scores"].items():
                        st.markdown(f"**{model}**: R²={scores['r2']:.3f}, RMSE=${scores['rmse']:,.2f}")
            
            st.markdown("### 💾 Download Reports")
            col_download1, col_download2 = st.columns(2)
            
            if predictions:
                forecast_df = pd.DataFrame({
                    "Date": predictions["dates"],
                    "Predicted Daily Change": predictions["predictions"],
                    "Predicted Balance": predictions["balances"],
                    "Upper Bound (95% CI)": predictions["confidence_upper"],
                    "Lower Bound (95% CI)": predictions["confidence_lower"],
                })
                csv = forecast_df.to_csv(index=False)
                with col_download1:
                    st.download_button("📥 Download Forecast CSV", data=csv, file_name="cashflow_forecast.csv", mime="text/csv", use_container_width=True)
                
                report = f"""
==================================================
CASHFLOW PREDICTION REPORT - ENHANCED EDITION
==================================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Prediction Period: {prediction_days} days
==================================================

SUMMARY
-------
Current Balance: ${current_balance:,.2f}
Predicted Balance: ${predictions['balances'][-1]:,.2f}
95% Confidence Range: ${predictions['confidence_lower'][-1]:,.2f} - ${predictions['confidence_upper'][-1]:,.2f}
Daily Trend: ${predictions['avg_daily']:,.2f}/day ({predictions['trend']})

MODEL PERFORMANCE
--------------
Best Model: {predictions['best_model']}
"""
                for model, scores in predictions["model_scores"].items():
                    report += f"  {model}: R²={scores['r2']:.3f}\n"
                
                report += f"""
SEASONALITY
----------
Pattern Detected: {seasonality['pattern']}
Best Day: {seasonality.get('best_day_name', 'N/A')}
"""
                
                report += """
==================================================
AI CONCLUSION
==================================================
"""
                report += f"Status: {conclusion['title']}\n\n"
                report += f"Analysis:\n{conclusion['message']}\n\n"
                report += "Recommendations:\n"
                for i, rec in enumerate(conclusion['recommendations'], 1):
                    report += f"  {i}. {rec}\n"
                
                report += """
==================================================
DISCLAIMER
==================================================
This is an AI-generated prediction for reference only.
Consult a financial advisor for important decisions.
==================================================
"""
                
                with col_download2:
                    st.download_button("📄 Download Report", data=report, file_name="cashflow_report.txt", mime="text/plain", use_container_width=True)
                
                col_pdf = st.columns(1)[0]
                with col_pdf:
                    current_tier = stats.get('subscription_tier', 'free')
                    if current_tier in ['pro', 'enterprise']:
                        pdf_data = generate_pdf_report(predictions, current_balance, daily_df, prediction_days, seasonality, anomalies, conclusion)
                        st.download_button("📑 Download PDF Report", data=pdf_data, file_name="cashflow_report.pdf", mime="application/pdf", use_container_width=True)
                    else:
                        st.warning("📑 PDF Reports available for Pro subscribers")
                        if st.button("⭐ Upgrade for PDF"):
                            st.session_state.page = "profile"
                            st.rerun()
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("Ensure your file has date and amount columns.")
    else:
        st.markdown("""
        <div class="upload-zone">
            <h3 style="color: #228B22;">📁 Drop your Excel or CSV file here</h3>
            <p>or click to browse</p>
            <p style="color: #999; font-size: 12px;">Supported: .csv, .xlsx, .xls</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📋 Sample Data Format")
        sample_data = pd.DataFrame({"Date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"], "Amount": [5000, -1200, 800, -2500, 1500]})
        st.dataframe(sample_data, use_container_width=True)
        
        st.markdown("""
        ### 🔧 How It Works
        1. **Upload** your Excel/CSV file
        2. **Analyze** - Auto-detect columns, seasonality, anomalies
        3. **Predict** - Ensemble ML models with confidence intervals
        4. **Decide** - AI-powered recommendations
        """)
    
    st.markdown("---")
    st.markdown(f"<div style='text-align: center; color: {get_colors()['primary']}; padding: 20px;'><small>CashFlow Predictor Pro | Enhanced Edition</small></div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()