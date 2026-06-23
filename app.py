import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 1. Set halaman dan UI
st.set_page_config(page_title="Charis Business Simulator", page_icon="💠", layout="wide")

# Custom CSS Dashboard
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Nunito', sans-serif;
    }
    
    h1 {
        color: #1E3A8A;
        font-weight: 800 !important;
        font-size: 2.5rem !important;
        margin-bottom: 0px !important;
        padding-bottom: 5px !important;
    }
    
    p {
        color: #64748B;
    }
    
    /* Bagian: Card Metrik */
            
    [data-testid="stMetric"] {
        background-color: #FFFFFF;
        padding: 20px 25px;
        border-radius: 20px;
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.03); 
        border: 1px solid #F1F5F9;
    }
    
    [data-testid="stMetricLabel"] {
        color: #64748B !important;
        font-weight: 600;
        font-size: 1rem;
    }
    
    [data-testid="stMetricValue"] {
        color: #1E3A8A !important;
        font-weight: 800;
        font-size: 2.2rem;
    }
    
    [data-testid="stMetricDelta"] svg {
        display: none; 
    }
    
    [data-testid="stSidebar"] {
        border-right: 1px solid #E2E8F0;
    }
    
    .sidebar-title {
        color: #1E3A8A;
        font-weight: 800;
        font-size: 1.2rem;
        margin-bottom: 10px;
    }
    
    /* Bagian: SLider */
            
    .stSlider div[role="slider"] {
        background-color: #FFFFFF !important; 
        border: 2px solid #2F6AFA !important;
        border-radius: 10px !important; 
        width: 16px !important;
        height: 24px !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1) !important;
    }
    
    .stSlider div[role="slider"]:focus {
        box-shadow: 0 0 0 0.2rem rgba(47, 106, 250, 0.2) !important;
        outline: none !important;
    }
    
    .stSlider > div > div > div {
        height: 6px !important;
    }
    
    
    
    .st-emotion-cache-1c7y2kd {
        background-color: #2F6AFA;
        color: #FFFFFF;
        border-radius: 15px;
        border: none;
        padding: 15px;
        box-shadow: 0px 4px 15px rgba(47, 106, 250, 0.2);
    }
    .st-emotion-cache-1c7y2kd p {
        color: #FFFFFF; 
    }
    
    .st-emotion-cache-12101t4, .st-emotion-cache-1p1nwyz {
        border-radius: 15px;
        border: none;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.03);
    }
    </style>
    """, unsafe_allow_html=True)

#2. Logic Machine Learning (DIGITAL TWIN)
@st.cache_resource
def load_model_and_baseline():
    X_train = np.array([[5, 10], [10, 20], [15, 5], [20, 25], [25, 15]])
    y_train = np.array([50, 80, 110, 90, 150])
    
    model = LinearRegression().fit(X_train, y_train)
    
    baseline_input = np.array([[10, 10]])
    baseline_pred = model.predict(baseline_input)[0]
    
    return model, baseline_pred

model, baseline_pred = load_model_and_baseline()

#3. Simulasi WHAT-IF
def run_simulation(new_iklan, new_diskon):
    intervention_input = np.array([[new_iklan, new_diskon]])
    prediction = model.predict(intervention_input)[0]
    delta_y = prediction - baseline_pred
    return prediction, delta_y

#4. Build tampilan UI
st.markdown('<h1>💠 Dashboard Simulasi Ekonomi</h1>', unsafe_allow_html=True)
st.markdown("Mensimulasikan skenario *What-If* untuk estimasi operasional bisnis yang lebih baik.")
st.write("") # 

# Sidebar
st.sidebar.markdown('<div class="sidebar-title">🎛️ Filter Kebijakan</div>', unsafe_allow_html=True)
iklan_slider = st.sidebar.slider("Anggaran Iklan (Juta)", 0, 50, 10)
diskon_slider = st.sidebar.slider("Besaran Diskon (%)", 0, 50, 10)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tips:** Geser parameter di atas untuk melihat tren perubahan keuntungan pada grafik utama secara instan.")

# Engine
hasil_pred, delta = run_simulation(iklan_slider, diskon_slider)

# Hasil (Layouting)
col1, col2 = st.columns([1, 1.5])

with col1:
    st.metric("Prediksi Keuntungan Terkini", f"Rp {hasil_pred:.2f} Jt", f"{delta:.2f} Jt (vs Baseline)")

with col2:
    st.write("") # Spacer
    if delta > 0:
        st.success(f"**Tren Positif!**\n\nSkenario ini memproyeksikan tambahan keuntungan sebesar **Rp {delta:.2f} Juta** dibanding metrik bulan lalu.")
    elif delta < 0:
        st.error(f"**Tren Menurun!**\n\nSkenario ini memproyeksikan kerugian/penurunan sebesar **Rp {abs(delta):.2f} Juta** dari standar operasional.")
    else:
        st.warning("**Kondisi Stagnan**\n\nTidak ada perubahan signifikan dibanding metrik standar operasional.")

st.markdown("<br>", unsafe_allow_html=True)

#5. Visualisasi grafik Matplotlib
st.markdown('<h3 style="color: #1E3A8A; font-weight: 700; font-size: 1.2rem;">Perbandingan Tren Keuntungan</h3>', unsafe_allow_html=True)

# Setup Matplotlib
fig, ax = plt.subplots(figsize=(8, 4))

fig.patch.set_alpha(0.0)
ax.set_facecolor('#FFFFFF')

# Warna Batang
color_baseline = '#F1F5F9' 
color_skenario = '#2F6AFA' if delta >= 0 else '#EF4444'

bars = ax.bar(['Baseline\n(Bulan Lalu)', 'Skenario\n(Bulan Ini)'], [baseline_pred, hasil_pred], 
              color=[color_baseline, color_skenario], width=0.4, zorder=3)

for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 3, f'{yval:.1f} Jt', 
            ha='center', va='bottom', fontweight='bold', color='#334155', fontsize=11)

ax.set_ylim(0, max(baseline_pred, hasil_pred) * 1.3)

ax.yaxis.grid(True, linestyle='--', color='#E2E8F0', linewidth=1.5, zorder=0)
ax.xaxis.grid(False)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_color('#CBD5E1')

ax.tick_params(axis='both', which='both', length=0, colors='#64748B', labelsize=10)

st.pyplot(fig)