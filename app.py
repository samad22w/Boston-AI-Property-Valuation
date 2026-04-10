import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from fpdf import FPDF
import datetime
import time
from gtts import gTTS
import base64

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Boston AI Intelligence", layout="wide", page_icon="🏠", initial_sidebar_state="expanded")

# --- 2. SESSION STATE ---
if 'logged_in' not in st.session_state: st.session_state.logged_in = False
if 'theme' not in st.session_state: st.session_state.theme = 'Dark'
if 'history' not in st.session_state: st.session_state.history = []

# --- 3. THEME ENGINE ---
def set_theme():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        .stButton>button { border-radius: 8px; height: 50px; font-weight: bold; border: none; transition: all 0.3s ease; }
        [data-testid="stDownloadButton"] button { background-color: #238636 !important; color: white !important; border: none !important; }
    </style>
    """, unsafe_allow_html=True)

    if st.session_state.theme == 'Dark':
        st.markdown("""
        <style>
            .stApp { background-color: #0B0E13; color: #E2E8F0; }
            h1, h2, h3, h4, h5, p, label { color: #FFFFFF !important; }
            div[data-baseweb="select"] > div, div[data-baseweb="input"] > div { background-color: #161B22 !important; color: white !important; border-color: #30363D !important; }
            div[data-baseweb="popover"], div[data-baseweb="menu"] { background-color: #161B22 !important; }
            li[data-baseweb="option"] { color: white !important; }
            section[data-testid="stSidebar"] { background-color: #11161D; border-right: 1px solid #30363D; }
            div[data-testid="metric-container"] { background-color: #161B22; border: 1px solid #30363D; border-radius: 12px; padding: 15px; }
            div[data-testid="metric-container"] label { color: #94A3B8 !important; }
            div[data-testid="metric-container"] div[data-testid="stMetricValue"] { color: #5865F2 !important; }
            .stButton>button { background-color: #5865F2; color: white !important; }
            .footer { background-color: #0B0E13; color: #94A3B8 !important; border-top: 1px solid #30363D; }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
            .stApp { background-color: #FFFFFF; color: #1E293B; }
            h1, h2, h3, h4, h5, p, label { color: #1E293B !important; }
            div[data-baseweb="select"] > div { background-color: #FFFFFF !important; color: black !important; }
            section[data-testid="stSidebar"] { background-color: #F8FAFC; border-right: 1px solid #E2E8F0; }
            div[data-testid="metric-container"] { background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 12px; padding: 15px; }
            div[data-testid="metric-container"] label { color: #64748B !important; }
            div[data-testid="metric-container"] div[data-testid="stMetricValue"] { color: #5865F2 !important; }
            .stButton>button { background-color: #5865F2; color: white !important; }
            .footer { background-color: #FFFFFF; color: #64748B !important; border-top: 1px solid #E2E8F0; }
        </style>
        """, unsafe_allow_html=True)
set_theme()

# --- 4. LOGIN ---
def login_screen():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.title("🔐 App Login")
        st.info("Access Key: admin")
        password = st.text_input("Enter Access Key", type="password")
        if st.button("Done"):
            if password == "admin": st.session_state.logged_in = True; st.rerun()

if not st.session_state.logged_in: login_screen(); st.stop()

# --- 5. FUNCTIONS ---
def calculate_emi(p, r, t):
    r = r / (12 * 100); n = t * 12
    if r == 0: return p / n
    return (p * r * (1 + r)**n) / ((1 + r)**n - 1)

def create_pdf(price, rooms, status, emi_est, years, symbol, dis, client_name):
    class PDF(FPDF):
        def header(self):
            self.set_fill_color(28, 40, 51)
            self.rect(0, 0, 210, 50, 'F')
            self.set_y(15); self.set_font('Arial', 'B', 26); self.set_text_color(255, 255, 255)
            self.cell(0, 10, 'BOSTON INTELLIGENCE', 0, 1, 'C')
            self.set_font('Arial', 'I', 10); self.set_text_color(200, 200, 200)
            self.cell(0, 10, 'Boston AI predicter', 0, 1, 'C')
        def footer(self):
            self.set_y(-15); self.set_font('Arial', 'I', 8); self.set_text_color(128, 128, 128)
            self.cell(0, 10, f'Page {self.page_no()} | Developed by Engr. Samad', 0, 0, 'C')

    pdf = PDF(); pdf.add_page()
    pdf.set_y(55); pdf.set_font("Arial", size=10); pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, txt=f"Date: {datetime.datetime.now().strftime('%d-%b-%Y')} | ID: #AI-{datetime.datetime.now().strftime('%H%M')}", ln=1, align='R')
    
    if client_name:
        pdf.set_y(55); pdf.set_font("Arial", 'B', 10); pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 10, txt=f"Prepared For: {client_name}", ln=1, align='L')

    pdf.ln(5); pdf.set_fill_color(245, 247, 250); pdf.set_draw_color(200, 200, 200)
    pdf.rect(10, pdf.get_y(), 190, 40, 'FD')
    pdf.set_y(pdf.get_y() + 8); pdf.set_font("Arial", 'B', 12); pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, "ESTIMATED MARKET VALUE", 0, 1, 'C')
    pdf.set_font("Arial", 'B', 32); pdf.set_text_color(39, 174, 96)
    pdf.cell(0, 15, f"{symbol}{price:,.2f}", 0, 1, 'C'); pdf.ln(20)

    pdf.ln(10); pdf.set_font("Arial", 'B', 14); pdf.set_text_color(28, 40, 51)
    pdf.cell(0, 10, "PROPERTY SPECIFICATIONS", 0, 1, 'L')
    pdf.line(10, pdf.get_y(), 200, pdf.get_y()); pdf.ln(5)

    pdf.set_font("Arial", size=12); pdf.set_text_color(0, 0, 0)
    pdf.cell(95, 10, f"  Total Rooms: {rooms}", 1); pdf.cell(95, 10, f"  Distance to Hub: {dis} miles", 1, 1)
    pdf.cell(95, 10, f"  Neighborhood Rating: {status}%", 1); pdf.cell(95, 10, "  Condition: Excellent", 1, 1)

    pdf.ln(10); pdf.set_font("Arial", 'B', 14); pdf.set_text_color(28, 40, 51)
    pdf.cell(0, 10, "FINANCIAL PROJECTION", 0, 1, 'L')
    pdf.line(10, pdf.get_y(), 200, pdf.get_y()); pdf.ln(5)

    pdf.set_font("Arial", size=12); pdf.set_text_color(0, 0, 0)
    pdf.cell(95, 10, f"  Monthly EMI: {symbol}{emi_est:,.2f}", 1); pdf.cell(95, 10, f"  Down Payment: {symbol}{price*0.2:,.2f}", 1, 1)
    pdf.cell(95, 10, f"  Loan Tenure: {years} Years", 1); pdf.cell(95, 10, "  Interest Rate: 5.0%", 1, 1)

    pdf.ln(15); pdf.set_font("Arial", 'I', 9); pdf.set_text_color(150, 150, 150)
    pdf.multi_cell(0, 5, " Disclaimer: This valuation is generated by an Artificial Intelligence model based on historical data. Market conditions may vary. Please consult a real estate agent for final deals.")
    return pdf.output(dest='S').encode('latin-1')

def text_to_speech(text):
    tts = gTTS(text=text, lang='en'); filename = "prediction.mp3"; tts.save(filename); return filename

# --- 6. MODEL ---
@st.cache_resource
def train_model():
    url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
    df = pd.read_csv(url)
    X = df.drop('medv', axis=1); y = df['medv']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train); score = model.score(X_test, y_test)
    defaults = X.mean().to_dict()
    return model, defaults, columns, score

temp_df = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv")
columns = temp_df.drop('medv', axis=1).columns
model, defaults, cols, score = train_model()

# --- 7. SIDEBAR ---
with st.sidebar:
    st.markdown("### ⚙️ Controls")
    theme_mode = st.radio("Theme", ["Dark", "Light"], horizontal=True)
    if theme_mode != st.session_state.theme: st.session_state.theme = theme_mode; st.rerun()
    
    st.markdown("---")
    client_name = st.text_input("Client Name (For Report)", placeholder="e.g. Client Name")
    
    st.markdown("---")
    currency = st.selectbox("Currency", ["USD ($)", "PKR (Rs)", "INR (₹)"])
    exchange_rate = 1; symbol = "$"
    if currency == "PKR (Rs)": exchange_rate = 278.50; symbol = "Rs "
    elif currency == "INR (₹)": exchange_rate = 83.00; symbol = "₹ "

    loan_years = st.slider("Loan Duration Years", 5, 30, 20)
    st.markdown("<br><br>", unsafe_allow_html=True)
    if st.button("← Logout"): st.session_state.logged_in = False; st.rerun()

# --- 8. MAIN UI ---
st.markdown("# Boston AI ")
st.markdown("### Next-Gen Property Valuation System")
st.markdown("<br>", unsafe_allow_html=True)

c_cont = st.container()
with c_cont:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("##### 🏠 Specs")
        rooms = st.slider("Total Rooms (RM)", 1.0, 10.0, 6.0, 0.1)
        status = st.slider("Neighborhood Status (LSTAT%)", 0.0, 40.0, 10.0, 0.5)
    with col2:
        st.markdown("##### 📍 Location")
        dis = st.slider("Distance to Hub", 1.0, 10.0, 4.0, 0.5)
    with col3:
        st.markdown(" Tools")
        marla = st.number_input("Unit Converter", 0.0, 10.0, 5.0)
        st.caption(f"👉 **{marla * 272.25:,.0f}** Sq. Ft")
    
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Generate "):
        st.session_state.run_valuation = True
    else:
        st.session_state.run_valuation = False

# --- 9. RESULTS ---
if st.session_state.run_valuation:
    st.markdown("---")
    with st.spinner('Calculating...'):
        time.sleep(0.5)
        input_data = defaults.copy()
        input_data['rm'] = rooms; input_data['lstat'] = status; input_data['dis'] = dis
        final_df = pd.DataFrame([input_data], columns=cols)
        pred_usd = model.predict(final_df)[0] * 1000
        final_price = pred_usd * exchange_rate
        down_payment = final_price * 0.20
        emi = calculate_emi(final_price - down_payment, 5.0, loan_years)
        
        st.session_state.history.insert(0, {
            "Time": datetime.datetime.now().strftime("%H:%M:%S"),
            "Client": client_name if client_name else "Guest",
            "Rooms": rooms,
            "Valuation": f"{symbol}{final_price:,.0f}"
        })

    # Sidebar Msg
    if pred_usd > 40000: msg = "💎 Luxury Tier"
    elif pred_usd < 15000: msg = "⚠️ Value Tier"
    else: msg = "✅ Market Tier"
    st.sidebar.success(msg)

    # Metrics
    st.markdown("### 📊 Valuation Summary")
    m1, m2, m3 = st.columns(3)
    m1.metric("Asset Value", f"{symbol}{final_price:,.0f}", "91% Confidence")
    m2.metric("Down Payment", f"{symbol}{down_payment:,.0f}")
    m3.metric(f"Monthly EMI ({loan_years}Yr)", f"{symbol}{emi:,.0f}")

    # --- 🌍 GLOBAL CONTEXT (Updated Cities) ---
    st.markdown("---")
    st.markdown("### 🌍 Global Property Comparison")
    st.info(f"💡 For **{symbol}{final_price:,.0f}**, here is what you could get in other major cities:")

    # Row 1: Major Global Hubs
    c1, c2, c3 = st.columns(3)
    with c1:
        st.success("🇺🇸 **New York (USA)**")
        st.caption("Tiny Studio (Manhattan). High status, global hub.")
    with c2:
        st.warning("🇦🇪 **Dubai (UAE)**")
        st.caption("1-Bed Luxury Apt. Tax-free income & modern lifestyle.")
    with c3:
        st.error("🇹🇷 **Istanbul (Turkey)**")
        st.caption("3-Bed Duplex. Great for Tourism & Citizenship.")

    # Row 2: Regional Markets
    c4, c5, c6 = st.columns(3)
    with c4:
        st.info("🇮🇳 **Mumbai (India)**")
        st.caption("Small 1-Bed (Suburbs). High commercial value.")
    with c5:
        st.success("🇵🇰 **Islamabad (Pakistan)**")
        st.caption("1 Kanal House. Green, peaceful & secure.")
    with c6:
        st.warning("🇦🇫 **Kabul (Afghanistan)**")
        st.caption("Large Mansion. Low cost due to high risk.")

    # Graph
    st.markdown("---")
    st.markdown("### 📈 Growth Analysis")
    future_prices = pd.DataFrame({'Value': [final_price * (1.05 ** i) for i in range(11)]})
    st.area_chart(future_prices, color=["#8A91DA"] if st.session_state.theme=='Dark' else ["#5674B4"])

    # Map
    st.markdown("---")
    st.markdown("### 🗺️ Location")
    st.map(pd.DataFrame({'lat': [42.3601], 'lon': [-71.0589]}), zoom=11)

    # Downloads
    st.markdown("---")
    st.markdown("### 📥 Reports")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### 🔊 Audio")
        st.audio(open(text_to_speech(f"Value is {int(final_price)} {currency}"), "rb").read(), format="audio/mp3")
    with c2:
        st.markdown("##### 📄 PDF Report")
        pdf_bytes = create_pdf(final_price, rooms, status, emi, loan_years, symbol, dis, client_name)
        st.download_button("Download PDF", pdf_bytes, f"Report_{client_name if client_name else 'Valuation'}.pdf", "application/pdf")

# --- 10. HISTORY TABLE ---
if len(st.session_state.history) > 0:
    st.markdown("---")
    st.markdown("### 📜 Recent Search History")
    st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)

# Footer
st.markdown("""<div class="footer">Developed by <span style="color: #4299E1;">Engr. Samad</span></div>""", unsafe_allow_html=True)


