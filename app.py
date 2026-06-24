import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 1. Set halaman dan UI
st.set_page_config(page_title="Business Simulator", page_icon="💠", layout="wide")

# Custom CSS Dashboard
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Nunito', sans-serif;
    }
    
    .stApp {
        background-color: #EAEAEA; 
    }
    
    .block-container {
        padding-top: 2rem;
    }
    
    /* Layout Baru */
       
    /* 1. Banner Biru Lebar di Atas */
    .top-banner {
        background-color: #007BFF;
        color: white;
        padding: 30px 40px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .top-banner h1 {
        color: white !important;
        margin: 0;
        font-weight: 800;
        font-size: 2.2rem;
    }
    .top-banner p {
        color: #E2E8F0;
        margin: 5px 0 0 0;
        font-size: 1.1rem;
    }

    /* 2. Kartu Putih (Kiri Bawah & Sidebar) */
    .card-white {
        background-color: #FFFFFF;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        height: 100%;
    }
    
    /* 3. Kartu Biru Muda (Kanan Bawah) */
    .card-lightblue {
        background-color: #A4CAFB; /* Biru muda sesuai referensi */
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        height: 100%;
        color: #1E3A8A;
    }
    
    /* Text Styling dalam Kartu */
    .metric-title {
        font-size: 1rem;
        font-weight: 600;
        color: #64748B;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: #1E3A8A;
        line-height: 1.2;
    }
    .metric-delta {
        font-size: 0.95rem;
        font-weight: 700;
        margin-top: 5px;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #EAEAEA; /* Samain dengan background utama */
        border-right: none;
    }
    
    /* Bagian: SLider */
    .stSlider div[role="slider"] {
        background-color: #FFFFFF !important; 
        border: 2px solid #007BFF !important;
        border-radius: 10px !important; 
        width: 16px !important;
        height: 26px !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1) !important;
    }
    .stSlider div[role="slider"]:focus {
        box-shadow: 0 0 0 0.2rem rgba(0, 123, 255, 0.2) !important;
        outline: none !important;
    }
    .stSlider > div > div > div {
        height: 8px !important;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. Logic Machine Learning (DIGITAL TWIN)
@st.cache_resource
def load_model_and_baseline():
    X_train = np.array([[5, 10], [10, 20], [15, 5], [20, 25], [25, 15]])
    y_train = np.array([50, 80, 110, 90, 150])
    
    model = LinearRegression().fit(X_train, y_train)
    
    baseline_input = np.array([[10, 10]])
    baseline_pred = model.predict(baseline_input)[0]
    
    return model, baseline_pred

model, baseline_pred = load_model_and_baseline()

# 3. Simulasi WHAT-IF
def run_simulation(new_iklan, new_diskon):
    intervention_input = np.array([[new_iklan, new_diskon]])
    prediction = model.predict(intervention_input)[0]
    delta_y = prediction - baseline_pred
    return prediction, delta_y

# 4. Build tampilan UI

# SIDEBAR (Sesuai Layout: Kotak Putih - Slider - Kotak Putih)
st.sidebar.markdown("""
<div class="card-white" style="margin-bottom: 20px; padding: 15px;">
    <h3 style="color: #1E3A8A; margin:0; font-weight: 800;">🎛️ Filter Kebijakan</h3>
</div>
""", unsafe_allow_html=True)

iklan_slider = st.sidebar.slider("Anggaran Iklan (Juta)", 0, 50, 10)
diskon_slider = st.sidebar.slider("Besaran Diskon (%)", 0, 50, 10)

st.sidebar.markdown("""
<div class="card-white" style="margin-top: 20px; padding: 15px;">
    <p style="margin:0; font-size: 0.9rem; color: #64748B;">
    <b>💡 Tips:</b> Geser parameter di atas untuk melihat tren perubahan keuntungan secara instan.
    </p>
</div>
""", unsafe_allow_html=True)

# MAIN AREA

# 1. Banner Biru Lebar di atas
st.markdown("""
<div class="top-banner">
    <h1>Dashboard Simulasi Ekonomi</h1>
    <p>Mensimulasikan skenario What-If untuk estimasi operasional bisnis yang lebih baik.</p>
</div>
""", unsafe_allow_html=True)

# Engine Prediksi
hasil_pred, delta = run_simulation(iklan_slider, diskon_slider)

# 2. Dua Kotak di Bawah (Kiri Putih, Kanan Biru Muda)
col1, col2 = st.columns(2)

with col1:
    # Kotak Putih Kiri (Menampilkan Angka)
    delta_color = "#22C55E" if delta >= 0 else "#EF4444" 
    delta_symbol = "+" if delta >= 0 else ""
    
    st.markdown(f"""
    <div class="card-white">
        <div class="metric-title">Prediksi Keuntungan Terkini</div>
        <div class="metric-value">Rp {hasil_pred:.2f} Jt</div>
        <div class="metric-delta" style="color: {delta_color};">
            {delta_symbol}{delta:.2f} Jt (vs Baseline)
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    # Kotak Biru Muda Kanan (Menampilkan Status/Pesan)
    if delta > 0:
        status_teks = f"<b>📈 Tren Positif!</b><br><br>Skenario ini memproyeksikan tambahan keuntungan sebesar <b>Rp {delta:.2f} Juta</b> dibanding metrik bulan lalu."
    elif delta < 0:
        status_teks = f"<b>📉 Tren Menurun!</b><br><br>Skenario ini memproyeksikan penurunan sebesar <b>Rp {abs(delta):.2f} Juta</b> dari standar operasional."
    else:
        status_teks = "<b>Kondisi Stagnan</b><br><br>Tidak ada perubahan signifikan dibanding metrik standar operasional saat ini."
        
    st.markdown(f"""
    <div class="card-lightblue">
        <div style="font-size: 1.05rem; line-height: 1.5;">
            {status_teks}
        </div>
    </div>
    """, unsafe_allow_html=True)

st.write("")
st.write("")

# 5. Visualisasi grafik Matplotlib
st.markdown('<h3 style="color: #1E3A8A; font-weight: 800; font-size: 1.3rem; margin-left: 5px;">Perbandingan Tren Keuntungan</h3>', unsafe_allow_html=True)

fig, ax = plt.subplots(figsize=(10, 3.5))
fig.patch.set_alpha(0.0)
ax.set_facecolor('#EAEAEA')

color_baseline = '#CBD5E1' 
color_skenario = '#007BFF' if delta >= 0 else '#EF4444'

bars = ax.bar(['Baseline\n(Bulan Lalu)', 'Skenario\n(Bulan Ini)'], [baseline_pred, hasil_pred], 
              color=[color_baseline, color_skenario], width=0.3, zorder=3)

for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 3, f'{yval:.1f} Jt', 
            ha='center', va='bottom', fontweight='bold', color='#1E3A8A', fontsize=12)

ax.set_ylim(0, max(baseline_pred, hasil_pred) * 1.3)
ax.yaxis.grid(True, linestyle='--', color='#D1D5DB', linewidth=1.5, zorder=0)
ax.xaxis.grid(False)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_color('#9CA3AF')

ax.tick_params(axis='both', which='both', length=0, colors='#4B5563', labelsize=11)

st.pyplot(fig)