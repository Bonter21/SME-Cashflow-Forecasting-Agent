import streamlit as st

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")

st.markdown("""
<style>
    .dashboard-card {
        background: linear-gradient(135deg, #228B22, #32CD32);
        padding: 30px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 10px 0;
    }
    .quick-action {
        background: white;
        border: 2px solid #228B22;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin: 5px 0;
        cursor: pointer;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("# 📊 Quick Dashboard")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div class="dashboard-card">
        <h2>💰 Balance</h2>
        <h1>$0.00</h1>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="dashboard-card">
        <h2>📈 Trend</h2>
        <h1>Up</h1>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div class="dashboard-card">
        <h2>🔔 Alerts</h2>
        <h1>0</h1>
    </div>
    """, unsafe_allow_html=True)

st.markdown("### ⚡ Quick Actions")
qa1, qa2, qa3 = st.columns(3)
with qa1:
    st.markdown("""
    <div class="quick-action">
        <h3>📤 Upload</h3>
        <p>Add transactions</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Upload"):
        st.switch_page("app.py")

with qa2:
    st.markdown("""
    <div class="quick-action">
        <h3>📊 Predict</h3>
        <p>View forecast</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Predict"):
        st.switch_page("app.py")

with qa3:
    st.markdown("""
    <div class="quick-action">
        <h3>📄 Report</h3>
        <p>Download PDF</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Report"):
        st.switch_page("app.py")

st.markdown("### 📈 Recent Activity")
st.info("Upload a file to see your cashflow activity here.")

st.markdown("---")
st.markdown("[← Back to Main App](app.py)")