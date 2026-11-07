"""
app.py
------
Dashboard PIHPS — Prediksi Harga Pangan Jawa Barat

SETUP:
  pip install streamlit plotly pandas numpy holidays

JALANKAN:
  streamlit run app.py
"""

import subprocess
import sys
import numpy as np
import pandas as pd
import holidays as hol_lib
import plotly.graph_objects as go
import streamlit as st
from datetime import date, timedelta
from pathlib import Path

from database import Database
from forecaster import generate_forecast, generate_simulation_forecast, generate_weekly_forecast, generate_weekly_simulation_forecast

# =============================================================================
# KONFIGURASI HALAMAN
# =============================================================================

st.set_page_config(
    page_title="PanelHarga — Prediksi Harga Pangan Jawa Barat",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# STYLING
# =============================================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
#MainMenu, footer, { visibility: hidden; }
.stApp { background-color: #F7F6F2; }

/* Sidebar */
section[data-testid="stSidebar"] { background-color: #1C1C1E; }
section[data-testid="stSidebar"] .stButton button * {
    color: #1C1C1E !important;
    font-weight: 600;
}
section[data-testid="stSidebar"] label {
    color: #9A9A8E !important;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
/* Sidebar nav buttons */
.nav-btn {
    display: block;
    width: 100%;
    padding: 0.6rem 1rem;
    margin-bottom: 0.35rem;
    background: transparent;
    border: 1px solid #3A3A3C;
    border-radius: 8px;
    color: #E8E6DF;
    font-size: 0.88rem;
    font-family: 'Plus Jakarta Sans', sans-serif;
    cursor: pointer;
    text-align: left;
    transition: all 0.15s ease;
}
.nav-btn:hover  { background: #2C2C2E; border-color: #5A5A5C; }
.nav-btn.active { background: #F0C040; border-color: #F0C040; color: #1C1C1E !important; font-weight: 600; }

/*Warna Putih*/
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
            color: #FFFFFF !important;
            }

/* Page header */
.page-header { padding: 2rem 0 1.5rem 0; border-bottom: 1px solid #E0DED8; margin-bottom: 2rem; }
.page-header h1 { font-size: 1.6rem; font-weight: 700; color: #1C1C1E; margin: 0; letter-spacing: -0.02em; }
.page-header p  { font-size: 0.875rem; color: #6B6B63; margin: 0.25rem 0 0 0; }

/* Scorecard */
.scorecard { background:#FFF; border:1px solid #E8E6DF; border-radius:10px; padding:1rem 1.1rem; margin-bottom:0.5rem; position:relative; overflow:hidden; }
.scorecard-name  { font-size:0.72rem; color:#6B6B63; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.3rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.scorecard-price { font-family:'DM Mono',monospace; font-size:1.1rem; font-weight:500; color:#1C1C1E; }
.scorecard-trend { font-size:0.75rem; font-weight:600; margin-top:0.25rem; }
.trend-up   { color:#D94F3D; }
.trend-down { color:#2E7D52; }
.trend-flat { color:#6B6B63; }
.scorecard-bar  { position:absolute; bottom:0; left:0; right:0; height:3px; border-radius:0 0 10px 10px; }

/* Section label */
.section-label { font-size:0.7rem; font-weight:600; color:#9A9A8E; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.75rem; margin-top:2rem; }

/* Recommendation card */
.rec-card { background:#FFF; border:1px solid #E8E6DF; border-radius:10px; padding:1.1rem 1.2rem; margin-bottom:0.6rem; height:100%; }
.rec-role   { font-size:0.7rem; font-weight:700; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.4rem; }
.rec-action { font-size:0.95rem; font-weight:600; color:#1C1C1E; margin-bottom:0.3rem; }
.rec-detail { font-size:0.82rem; color:#6B6B63; line-height:1.5; }

/* Market banner */
.market-banner { background:#1C1C1E; color:#F7F6F2; border-radius:10px; padding:1rem 1.4rem; margin-bottom:1.5rem; font-size:0.9rem; line-height:1.6; }
.market-banner strong { color:#F0C040; }

/* Alert banners */
.alert-banner { background:#FFF3CD; border:1px solid #F0C040; border-radius:8px; padding:0.75rem 1rem; font-size:0.85rem; color:#856404; margin-bottom:1rem; }
.holiday-banner { background:#EEF2FF; border:1px solid #818CF8; border-radius:8px; padding:0.75rem 1rem; font-size:0.85rem; color:#3730A3; margin-bottom:1rem; }

/* Metric pill */
.metric-pill { display:inline-block; background:#F0EDE6; border-radius:20px; padding:0.2rem 0.7rem; font-family:'DM Mono',monospace; font-size:0.8rem; color:#1C1C1E; margin-right:0.4rem; }

/* Extreme weather badge */
.wx-badge { display:inline-block; border-radius:6px; padding:0.2rem 0.6rem; font-size:0.8rem; font-weight:600; margin-bottom:0.5rem; }
.wx-extreme { background:#FEE2E2; color:#991B1B; }
.wx-normal  { background:#DCFCE7; color:#166534; }

/* Table page */
.data-table-container { background:#FFF; border:1px solid #E8E6DF; border-radius:10px; padding:1rem; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CONSTANTS
# =============================================================================

# DB_PATH is anchored to this file's own folder (not the terminal's current
# working directory) so the app always reads the SAME pihps.db that the
# "Perbarui Sekarang" button's pipeline.py subprocess writes to. A bare
# relative path here resolves against wherever `streamlit run` happened to
# be launched from, which can silently point at a different (stale) file
# than the one pipeline.py updates — cache clearing can't fix that because
# the cache itself isn't the problem, the file being read is.
DB_PATH = str(Path(__file__).resolve().parent / 'pihps.db')

# Info siklus produksi untuk kotak "🌱 Siklus tanam" di halaman simulasi
# cuaca. HANYA komoditas dengan siklus domestik yang jelas dimasukkan —
# lihat catatan lengkap soal Bawang Putih & Minyak Goreng di CROP_LAGS
# (pipeline.py): keduanya sebagian besar impor / panen kontinu, sehingga
# kotak "siklus tanam" akan menyesatkan jika dipaksakan. Untuk keduanya,
# kotak info ini sengaja TIDAK ditampilkan (selectbox tetap berfungsi).
# 10 komoditas resmi saat ini (harus sama dengan ALL_KOMODITAS di
# pipeline.py). Dipakai untuk memfilter komoditas lama/usang yang mungkin
# masih tersimpan di database (mis. varian Beras/Cabai dari skema lama)
# supaya tidak muncul lagi di UI, tanpa perlu menghapus data historisnya
# dari database. Didefinisikan terpisah di sini (bukan import dari
# pipeline.py) supaya Streamlit app ini tidak ikut memuat TensorFlow saat
# start up.
TARGET_KOMODITAS = [
    'Beras', 'Daging Ayam', 'Daging Sapi', 'Telur Ayam',
    'Bawang Merah', 'Bawang Putih', 'Cabai Merah', 'Cabai Rawit',
    'Minyak Goreng', 'Gula Pasir',
]

CROP_CYCLE = {
    'Beras'        : ('90–120 hari', '2× setahun (Okt–Mar & Apr–Sep)'),
    'Bawang Merah' : ('60–75 hari',  'Sepanjang tahun, sensitif curah hujan tinggi'),
    'Cabai Merah'  : ('90–120 hari', 'Kritis di fase pembungaan 60–90 hari sebelum panen'),
    'Cabai Rawit'  : ('70–90 hari',  'Toleran kemarau, rentan genangan air'),
    'Daging Ayam'  : ('30–40 hari',  'Siklus ternak broiler (DOC→panen); harga juga mengikuti biaya pakan jagung'),
    'Telur Ayam'   : ('18–22 minggu', 'Ayam petelur matang bertelur; harga mengikuti lag biaya pakan'),
    'Daging Sapi'  : ('3–4+ bulan',  'Siklus penggemukan; sebagian pasokan impor (sapi bakalan)'),
    'Gula Pasir'   : ('10–12 bulan', 'Tebu, panen musim kemarau di Jawa; sebagian pasokan dari gula rafinasi impor'),
}

# Extreme weather scenario definitions
WEATHER_SCENARIOS = {
    # Deltas calibrated against real historical data (province-wide daily
    # mean for Jawa Barat, 2021-01-01 s/d 2026-07-08). The old deltas (e.g.
    # Gelombang Panas +6°C) all exceeded the most extreme single-day anomaly
    # this averaged series has ever recorded (+1.90°C) — meaning every
    # scenario was silently asking the model to extrapolate, just by
    # different amounts. These values are instead anchored to observed
    # anomaly percentiles (anomaly = daily value minus that month's normal).
    # Curah Hujan (ch) multipliers were left as round numbers — rainfall is
    # naturally highly right-skewed, so multiplying the mean toward zero for
    # dry scenarios, or up for wet ones, is already reasonably grounded
    # without needing anomaly-based calibration.
    'Normal'           : {'suhu': 0,     'ch': 1.0, 'rh': 0,   'label': 'Kondisi Normal',       'color': '#2E7D52'},
    # suhu: ~p95 anomaly (+0.78) | rh: mild dry, roughly p10 territory
    'Kemarau Ringan'   : {'suhu': +0.7,  'ch': 0.3, 'rh': -5,  'label': 'Kemarau Ringan',        'color': '#F59E0B'},
    # suhu: ~p99 anomaly (+1.17) | rh: ~p1 anomaly (-8.73)
    'Kemarau Ekstrem'  : {'suhu': +1.2,  'ch': 0.05, 'rh': -9, 'label': 'Kemarau Ekstrem 🔥',    'color': '#DC2626'},
    # suhu: ~p5 anomaly (-0.88) | rh: ~p95 anomaly (+4.39)
    'Hujan Lebat'      : {'suhu': -0.9,  'ch': 2.5, 'rh': +4,  'label': 'Hujan Lebat',           'color': '#2563EB'},
    # suhu: close to historical min anomaly (-1.58) | rh: near max anomaly (+8.65)
    'Banjir / Badai'   : {'suhu': -1.5,  'ch': 5.0, 'rh': +8,  'label': 'Banjir / Badai ⛈️',    'color': '#7C3AED'},
    # suhu/rh: mild sustained wet anomaly, similar magnitude to Hujan Lebat
    # since La Niña is a persistent pattern rather than a single-day extreme
    'La Niña (Basah)'  : {'suhu': -0.8,  'ch': 1.8, 'rh': +4,  'label': 'La Niña (Basah)',       'color': '#0891B2'},
    # suhu: ~p97 anomaly, sustained over weeks rather than a single day
    'El Niño (Kering)' : {'suhu': +1.0,  'ch': 0.2, 'rh': -9,  'label': 'El Niño (Kering)',      'color': '#EA580C'},
}

# =============================================================================
# HELPERS
# =============================================================================

@st.cache_resource
def get_db():
    return Database(DB_PATH)

@st.cache_data(ttl=300)
def load_commodities():
    # Filter ke 10 komoditas resmi, urut sesuai TARGET_KOMODITAS, dan
    # hanya sertakan yang benar-benar punya data (skip yang belum sempat
    # di-fetch sama sekali) — supaya varian lama (mis. "Beras Kualitas
    # Bawah I") tidak lagi tampil meski masih ada di database.
    available = set(get_db().get_commodities())
    return [k for k in TARGET_KOMODITAS if k in available]

@st.cache_data(ttl=300)
def load_prices_all():
    db = get_db()
    return {k: db.get_prices(k) for k in load_commodities()
            if not db.get_prices(k).empty}

@st.cache_data(ttl=300)
def load_predictions(komoditas: str):
    return get_db().get_predictions(komoditas)

@st.cache_data(ttl=300)
def load_metadata(komoditas: str):
    return get_db().get_model_metadata(komoditas)


def fmt_rp(value: float) -> str:
    if pd.isna(value): return "—"
    return f"Rp {value:,.2f}"


def get_trend(current: float, future: float, threshold: float = 0.02) -> str:
    if pd.isna(current) or pd.isna(future): return 'flat'
    change = (future - current) / current
    if change > threshold:  return 'up'
    if change < -threshold: return 'down'
    return 'flat'


def trend_icon(t): return {'up':'↑','down':'↓','flat':'→'}.get(t,'→')
def trend_label(t, pct):
    return {'up': f"Naik {abs(pct):.1f}%", 'down': f"Turun {abs(pct):.1f}%", 'flat': "Stabil"}.get(t,"Stabil")


import logging
log = logging.getLogger(__name__)

@st.cache_data(ttl=3600)   # cache 1 jam
def get_forecast(komoditas: str, n_days: int) -> pd.DataFrame:
    """
    Recursive LSTM multi-step forecast.
    Falls back to simple trend if model not found on disk.
    """
    db = get_db()
    try:
        df = generate_forecast(komoditas, n_days, db)
        if not df.empty:
            return df
    except Exception as e:
        log.warning(f"LSTM forecast failed for {komoditas}: {e}")

    # Fallback: simple trend extrapolation
    df_pred = db.get_predictions(komoditas)
    if df_pred.empty or len(df_pred) < 2:
        return pd.DataFrame()
    last_price = df_pred['Incr'].dropna().iloc[-1]
    last_date  = df_pred.index.max()
    daily_chg  = df_pred['Incr'].dropna().tail(30).pct_change().mean()
    dates, prices, d, p, count = [], [], last_date + timedelta(days=1), last_price, 0
    while count < n_days:
        if d.weekday() < 5:
            p = p * (1 + daily_chg)
            dates.append(d); prices.append(p); count += 1
        d += timedelta(days=1)
    arr = np.array(prices)
    unc = np.array([0.05 + 0.003 * i for i in range(n_days)])
    return pd.DataFrame({
        'Forecast': arr, 'Lower': arr*(1-1.645*unc),
        'Upper': arr*(1+1.645*unc), 'Sumber_Cuaca': ['Trend']*n_days,
    }, index=pd.DatetimeIndex(dates))


@st.cache_data(ttl=3600)   # cache 1 jam
def get_weekly_forecast(komoditas: str, n_weeks: int) -> pd.DataFrame:
    """
    Recursive weekly LSTM multi-step forecast — dipakai khusus fitur
    Simulasi Cuaca (lihat catatan di forecaster.py/pipeline.py dekat
    generate_weekly_forecast: rekursi mingguan jauh lebih stabil untuk
    horizon multi-minggu dibanding rekursi harian).
    Falls back to simple weekly trend jika model mingguan belum dilatih.
    """
    db = get_db()
    try:
        df = generate_weekly_forecast(komoditas, n_weeks, db)
        if not df.empty:
            return df
    except Exception as e:
        log.warning(f"Weekly LSTM forecast failed for {komoditas}: {e}")

    # Fallback: simple weekly trend extrapolation
    df_pred = db.get_predictions(komoditas)
    if df_pred.empty or len(df_pred) < 2:
        return pd.DataFrame()
    last_price = df_pred['Incr'].dropna().iloc[-1]
    last_date  = df_pred.index.max()
    weekly_chg = df_pred['Incr'].dropna().resample('W').mean().pct_change().mean()
    if pd.isna(weekly_chg):
        weekly_chg = 0.0
    dates  = [last_date + timedelta(weeks=i + 1) for i in range(n_weeks)]
    prices = [last_price * (1 + weekly_chg) ** (i + 1) for i in range(n_weeks)]
    arr = np.array(prices)
    unc = np.array([0.08 + 0.01 * i for i in range(n_weeks)])
    return pd.DataFrame({
        'Forecast': arr, 'Lower': arr*(1-1.645*unc),
        'Upper': arr*(1+1.645*unc), 'Sumber_Cuaca': ['Trend']*n_weeks,
    }, index=pd.DatetimeIndex(dates))


# Keep old name as alias for weather simulation page
def upcoming_holidays(days_ahead: int = 30) -> list[dict]:
    """Return list of upcoming Indonesian holidays within days_ahead."""
    today    = date.today()
    end_scan = today + timedelta(days=days_ahead)
    id_hols  = hol_lib.ID(years=[today.year, today.year + 1])
    upcoming = []
    for hdate, hname in sorted(id_hols.items()):
        if today <= hdate <= end_scan:
            days_away = (hdate - today).days
            upcoming.append({'date': hdate, 'name': hname, 'days_away': days_away})
    return upcoming


def holiday_warning_text(days_ahead: int = 30) -> str | None:
    """Return warning string if a major price-affecting holiday is approaching."""
    hols = upcoming_holidays(days_ahead)
    major = [h for h in hols if any(k in h['name'].lower() for k in
             ['idul fitri', 'idul adha', 'natal', 'imlek', 'eid'])]
    if not major: return None
    h = major[0]
    return f"📅 <strong>{h['name']}</strong> dalam <strong>{h['days_away']} hari</strong> ({h['date'].strftime('%d %b %Y')}) — harga cenderung naik menjelang hari besar."


def generate_recommendations(
    komoditas: str,
    current_price: float,
    forecast_df: pd.DataFrame,
    trend: str,
    pct_change: float,
    holiday_warning: str | None = None
) -> list[dict]:
    abs_pct   = abs(pct_change)
    price_rp  = fmt_rp(current_price)
    magnitude = "signifikan" if abs_pct > 5 else "moderat"
    n_days    = len(forecast_df)
    hw        = f" ⚠️ {holiday_warning}" if holiday_warning else ""

    if trend == 'up':
        recs = [
            {'role':'🌾 Petani',           'color':'#2E7D52',
             'action':'Tahan hasil panen jika memungkinkan',
             'detail':f'Harga diprediksi naik {abs_pct:.1f}% dalam {n_days} hari. Menunda penjualan berpotensi menghasilkan harga lebih baik.{hw}'},
            {'role':'🏪 Pedagang',          'color':'#1A5CA8',
             'action':'Tambah stok sekarang sebelum harga naik',
             'detail':f'Harga saat ini {price_rp}/kg. Window pembelian terbaik dalam 3–5 hari ke depan.{hw}'},
            {'role':'🛒 Pembeli / Konsumen','color':'#D94F3D',
             'action':'Beli kebutuhan sekarang',
             'detail':f'Harga diprediksi naik {abs_pct:.1f}%. Beli sebelum harga naik lebih tinggi.{hw}'},
            {'role':'🏛️ Pemerintah',        'color':'#7B5EA7',
             'action':'Pantau stok dan rantai distribusi',
             'detail':f'Kenaikan harga {magnitude} terdeteksi. Monitor pasokan dari daerah produksi dan antisipasi kelangkaan.{hw}'},
        ]
    elif trend == 'down':
        recs = [
            {'role':'🌾 Petani',           'color':'#D94F3D',
             'action':'Percepat penjualan hasil panen',
             'detail':f'Harga diprediksi turun {abs_pct:.1f}%. Jual sekarang di {price_rp}/kg sebelum harga turun lebih jauh.{hw}'},
            {'role':'🏪 Pedagang',          'color':'#1A5CA8',
             'action':'Kurangi volume pembelian sementara',
             'detail':f'Harga kemungkinan lebih rendah dalam beberapa hari. Tunda penambahan stok besar.{hw}'},
            {'role':'🛒 Pembeli / Konsumen','color':'#2E7D52',
             'action':'Tunggu beberapa hari untuk harga lebih baik',
             'detail':f'Harga diprediksi turun {abs_pct:.1f}% dalam {n_days} hari. Tunda pembelian besar jika tidak mendesak.{hw}'},
            {'role':'🏛️ Pemerintah',        'color':'#7B5EA7',
             'action':'Pastikan harga dasar petani terlindungi',
             'detail':f'Penurunan harga {magnitude} terdeteksi. Pastikan harga tidak jatuh di bawah harga pokok produksi.{hw}'},
        ]
    else:
        recs = [
            {'role':'🌾 Petani',           'color':'#6B6B63',
             'action':'Kondisi stabil — jual sesuai kebutuhan',
             'detail':f'Tidak ada tekanan mendesak untuk mempercepat atau menunda penjualan.{hw}'},
            {'role':'🏪 Pedagang',          'color':'#6B6B63',
             'action':'Pertahankan strategi stok saat ini',
             'detail':f'Tidak ada sinyal perubahan harga signifikan. Lanjutkan operasional normal.{hw}'},
            {'role':'🛒 Pembeli / Konsumen','color':'#6B6B63',
             'action':'Beli sesuai kebutuhan normal',
             'detail':f'Harga diperkirakan stabil — tidak perlu rush buy atau menunda.{hw}'},
            {'role':'🏛️ Pemerintah',        'color':'#6B6B63',
             'action':'Tidak ada intervensi mendesak',
             'detail':f'Harga dalam kondisi normal. Lanjutkan pemantauan rutin.{hw}'},
        ]
    return recs


def market_summary(prices_all: dict) -> str:
    rising = falling = stable = 0
    for df in prices_all.values():
        if len(df) < 7: continue
        recent = df['Harga (Rp)'].dropna().tail(7)
        if len(recent) < 2: continue
        chg = (recent.iloc[-1] - recent.iloc[0]) / recent.iloc[0]
        if chg > 0.02:   rising  += 1
        elif chg < -0.02: falling += 1
        else:             stable  += 1
    parts = []
    if rising:  parts.append(f"<strong>{rising} komoditas naik</strong>")
    if falling: parts.append(f"<strong>{falling} komoditas turun</strong>")
    if stable:  parts.append(f"{stable} komoditas stabil")
    return "Minggu ini: " + ", ".join(parts) + " berdasarkan data 7 hari terakhir." if parts else "Data pasar sedang diperbarui."


# =============================================================================
# NAVIGASI SIDEBAR
# =============================================================================

if 'page' not in st.session_state:
    st.session_state.page = 'Beranda'

with st.sidebar:
    st.markdown("### 🌾 PanelHarga")
    st.markdown(
        "<p style='font-size:0.75rem;color:#FFFFFF;margin-top:-0.5rem'>"
        "Prediksi Harga Pangan · Jawa Barat</p>",
        unsafe_allow_html=True
    )
    st.divider()

    for nav_label in ["Beranda", "Data Historis", "Simulasi Cuaca"]:
        is_active = st.session_state.page == nav_label
        css_class = "nav-btn active" if is_active else "nav-btn"
        if st.button(nav_label, key=f"nav_{nav_label}", use_container_width=True):
            st.session_state.page = nav_label
            st.rerun()

    st.divider()
    st.markdown(
        "<p style='font-size:0.7rem;color:#9A9A8E;text-transform:uppercase;"
        "letter-spacing:0.08em'>Update Data</p>",
        unsafe_allow_html=True
    )
    if st.button("🔄 Perbarui Sekarang", use_container_width=True):
        with st.spinner("Menjalankan pipeline untuk semua komoditas..."):
            pipeline_path = Path(__file__).resolve().parent / "pipeline.py"
            result = subprocess.run(
                [sys.executable, str(pipeline_path), "--no-retrain", "--all"],
                capture_output=True, text=True,
                cwd=str(pipeline_path.parent),
            )
            if result.returncode == 0:
                st.success("Data berhasil diperbarui")
                st.cache_data.clear()
                st.rerun()
            else:
                st.error("Gagal memperbarui data — lihat detail di bawah")
                with st.expander("Detail log pipeline", expanded=True):
                    st.code(result.stdout[-3000:] or "(tidak ada output)")
                    if result.stderr:
                        st.code(result.stderr[-2000:])

    st.divider()
    st.markdown(
        "<p style='font-size:0.7rem;color:#6B6B63'>"
        "Data: PIHPS Bank Indonesia<br>"
        "Cuaca: Open-Meteo <br>"
        "Model: Incremental LSTM</p>",
        unsafe_allow_html=True
    )

page = st.session_state.page


# =============================================================================
# PAGE 1: BERANDA
# =============================================================================

if page == "Beranda":
    commodities = load_commodities()
    prices_all  = load_prices_all()

    if not commodities:
        st.warning("⚠️ Belum ada data di database. Jalankan `python load_initial_data.py` lalu `python pipeline.py`.")
        st.stop()

    max_dates = [df.index.max() for df in prices_all.values() if not df.empty]
    last_updated = max(max_dates) if max_dates else date.today()

    st.markdown(
        f"<div class='page-header'><h1>Harga Pangan Hari Ini</h1>"
        f"<p>Jawa Barat · Diperbarui {last_updated.strftime('%d %B %Y')}</p></div>",
        unsafe_allow_html=True
    )

    # Holiday warning banner
    hw_text = holiday_warning_text(30)
    if hw_text:
        st.markdown(f"<div class='holiday-banner'>{hw_text}</div>", unsafe_allow_html=True)

    # Market summary
    st.markdown(f"<div class='market-banner'>{market_summary(prices_all)}</div>", unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # SCORECARDS
    # ------------------------------------------------------------------
    st.markdown("<div class='section-label'>Harga Komoditas</div>", unsafe_allow_html=True)

    bar_colors = {'up':'#D94F3D','down':'#2E7D52','flat':'#C8C5BE'}
    cols = st.columns(4)
    for i, k in enumerate(commodities):
        df = prices_all.get(k, pd.DataFrame())
        with cols[i % 4]:
            if df.empty or len(df) < 2:
                st.markdown(f"<div class='scorecard'><div class='scorecard-name'>{k}</div><div class='scorecard-price'>—</div></div>", unsafe_allow_html=True)
                continue
            recent   = df['Harga (Rp)'].dropna()
            current  = recent.iloc[-1]
            week_ago = recent.iloc[-7] if len(recent) >= 7 else recent.iloc[0]
            trend    = get_trend(week_ago, current)
            pct      = (current - week_ago) / week_ago * 100
            st.markdown(
                f"""<div class='scorecard'>
                <div class='scorecard-name'>{k}</div>
                <div class='scorecard-price'>{fmt_rp(current)}</div>
                <div class='scorecard-trend trend-{trend}'>{trend_icon(trend)} {trend_label(trend, pct)}</div>
                <div class='scorecard-bar' style='background:{bar_colors[trend]}'></div>
                </div>""", unsafe_allow_html=True
            )

   # ------------------------------------------------------------------
    # PREDIKSI — full width chart
    # ------------------------------------------------------------------
    st.markdown("<div class='section-label'>Prediksi Harga</div>", unsafe_allow_html=True)

    # Sedikit penyesuaian rasio kolom agar teks checkbox tidak terlalu sempit
    ctrl1, ctrl2, ctrl3 = st.columns([3, 1, 1.2]) 
    
    with ctrl1:
        selected = st.selectbox(
            "Komoditas",
            commodities,
            index=commodities.index('Cabai Merah') if 'Cabai Merah' in commodities else 0,
            label_visibility="hidden"  # KUNCI 1: Ubah dari "collapsed" ke "hidden"
        )
    with ctrl2:
        forecast_days = st.number_input(
            "Hari ke depan", min_value=1, max_value=366, value=14, step=1,
            help="Masukkan jumlah hari yang ingin diprediksi"
        )
    with ctrl3:
        # KUNCI 2: Tambahkan margin atas buatan agar sejajar dengan kotak input
        st.markdown("<div style='margin-top: 2.1rem;'></div>", unsafe_allow_html=True)
        show_all = st.checkbox("Tampilkan seluruh data", value=False)

    df_pred  = load_predictions(selected)
    df_price = prices_all.get(selected, pd.DataFrame())
    meta     = load_metadata(selected)

    if df_pred.empty:
        st.info(f"Belum ada prediksi untuk {selected}. Jalankan `python pipeline.py --komoditas \"{selected}\"`")
    else:
        df_forecast  = get_forecast(selected, int(forecast_days))
        last_actual  = df_pred['Aktual'].dropna().iloc[-1] if not df_pred['Aktual'].dropna().empty else None
        last_fc      = df_forecast['Forecast'].iloc[-1]    if not df_forecast.empty else None
        has_both     = last_actual is not None and last_fc is not None
        trend        = get_trend(last_actual, last_fc) if has_both else 'flat'
        pct_chg      = ((last_fc - last_actual) / last_actual * 100) if has_both else 0.0

        # Alert banner
        if abs(pct_chg) > 10:
            direction = "kenaikan" if pct_chg > 0 else "penurunan"
            st.markdown(
                f"<div class='alert-banner'>⚠️ Terdeteksi potensi <strong>{direction} signifikan "
                f"({abs(pct_chg):.1f}%)</strong> dalam {forecast_days} hari ke depan.</div>",
                unsafe_allow_html=True
            )

        # Metric pills
        if meta:
            st.markdown(
                f"<span class='metric-pill'>MAPE {meta['mape']:.2f}%</span>"
                f"<span class='metric-pill'>RMSE Rp {meta['rmse']:,.2f}</span>"
                f"<span class='metric-pill'>Fine-tune {meta['n_finetune']}×</span>",
                unsafe_allow_html=True
            )

        # Chart — full width, default last 60 days, toggle for all
        df_show = df_pred if show_all else df_pred.tail(60)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_show.index, y=df_show['Aktual'],
            name='Harga Aktual', line=dict(color='#1C1C1E', width=2),
            hovertemplate='%{x|%d %b %Y}<br>Aktual: Rp %{y:,.2f}<extra></extra>'
        ))
        fig.add_trace(go.Scatter(
            x=df_show.index, y=df_show['Incr'],
            name='Prediksi Model', line=dict(color='#2E7D52', width=2, dash='dot'),
            hovertemplate='%{x|%d %b %Y}<br>Prediksi: Rp %{y:,.2f}<extra></extra>'
        ))

        if not df_forecast.empty:
            bridge_x = [df_show.index[-1]] + list(df_forecast.index)
            bridge_y = [df_show['Incr'].iloc[-1]] + list(df_forecast['Forecast'])
            bridge_l = [df_show['Incr'].iloc[-1]] + list(df_forecast['Lower'])
            bridge_u = [df_show['Incr'].iloc[-1]] + list(df_forecast['Upper'])

            # Confidence band
            fig.add_trace(go.Scatter(
                x=list(bridge_x) + list(reversed(bridge_x)),
                y=list(bridge_u) + list(reversed(bridge_l)),
                fill='toself',
                fillcolor='rgba(240,192,64,0.12)',
                line=dict(color='rgba(0,0,0,0)'),
                name='Interval Kepercayaan 90%',
                showlegend=True,
                hoverinfo='skip',
            ))

            # Forecast line
            fig.add_trace(go.Scatter(
                x=bridge_x, y=bridge_y,
                name=f'Proyeksi {int(forecast_days)} Hari (LSTM)',
                line=dict(color='#F0C040', width=2.5, dash='dash'),
                hovertemplate='%{x|%d %b %Y}<br>Proyeksi: Rp %{y:,.2f}<extra></extra>'
            ))
            fig.add_vrect(
                x0=df_forecast.index[0], x1=df_forecast.index[-1],
                fillcolor='rgba(240,192,64,0.04)', layer='below', line_width=0
            )

            # Annotate where real forecast ends and seasonal begins
            if 'Sumber_Cuaca' in df_forecast.columns:
                cutoff_rows = df_forecast[df_forecast['Sumber_Cuaca'] == 'Seasonal']
                if not cutoff_rows.empty:
                    cutoff_date = cutoff_rows.index[0]
                    fig.add_vline(
                        x=cutoff_date, line_dash='dot',
                        line_color='#9A9A8E', line_width=1,
                    )
                    fig.add_annotation(
                        x=cutoff_date, y=1, yref='paper',
                        text='↑ Cuaca Seasonal', showarrow=False,
                        font=dict(size=9, color='#9A9A8E'),
                        xanchor='left', yanchor='top',
                    )

        fig.update_layout(
            height=380,
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0, font=dict(size=11)),
            xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(size=10, color='#6B6B63')),
            yaxis=dict(showgrid=True, gridcolor='#E8E6DF', zeroline=False,
                       tickfont=dict(size=10, color='#6B6B63'), tickprefix='Rp ', tickformat=',.0f'),
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)

        if int(forecast_days) > 14:
            st.caption("⚠️ Proyeksi lebih dari 14 hari bersifat indikatif — akurasi menurun seiring jarak prediksi.")

        # ------------------------------------------------------------------
        # REKOMENDASI — 2×2 grid below the chart
        # ------------------------------------------------------------------
        st.markdown("<div class='section-label'>Rekomendasi Tindakan</div>", unsafe_allow_html=True)

        hw_rec = holiday_warning_text(14)
        recs   = generate_recommendations(selected, last_actual, df_forecast, trend, pct_chg, hw_rec)

        row1, row2 = st.columns(2), st.columns(2)
        for idx, rec in enumerate(recs):
            col = row1[idx] if idx < 2 else row2[idx - 2]
            with col:
                st.markdown(
                    f"""<div class='rec-card'>
                    <div class='rec-role' style='color:{rec["color"]}'>{rec["role"]}</div>
                    <div class='rec-action'>{rec["action"]}</div>
                    <div class='rec-detail'>{rec["detail"]}</div>
                    </div>""", unsafe_allow_html=True
                )


# =============================================================================
# PAGE 2: DATA HISTORIS
# =============================================================================

elif page == "Data Historis":
    st.markdown(
        "<div class='page-header'><h1>Data Historis</h1>"
        "<p>Harga aktual dan hasil prediksi model per komoditas</p></div>",
        unsafe_allow_html=True
    )

    commodities = load_commodities()
    if not commodities:
        st.warning("Belum ada data di database.")
        st.stop()

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1: selected_h = st.selectbox("Komoditas", commodities, key='hist_kom')
    with col2: date_from  = st.date_input("Dari",    value=date(2025, 1, 1), key='hist_from')
    with col3: date_to    = st.date_input("Sampai",  value=date.today(),     key='hist_to')

    db       = get_db()
    df_price = db.get_prices(selected_h, start=date_from.isoformat(), end=date_to.isoformat())
    df_pred  = db.get_predictions(selected_h, start=date_from.isoformat(), end=date_to.isoformat())

    if df_price.empty and df_pred.empty:
        st.info(f"Tidak ada data untuk {selected_h} pada rentang tanggal tersebut.")
        st.stop()

    # Build table — only Tanggal, Harga Aktual, Prediksi Model (no error columns)
    df_hist = pd.DataFrame()
    if not df_price.empty:
        df_hist['Harga Aktual (Rp)'] = df_price['Harga (Rp)'].round(2)
    # if not df_pred.empty:
    #     df_hist['Prediksi Model (Rp)'] = df_pred['Incr'].round(2)
    #     if 'Aktual' in df_pred.columns and df_pred['Aktual'].notna().any():
    #         df_hist['Harga Aktual (Rp)'] = df_pred['Aktual'].round(2)

    if not df_hist.empty:
        df_hist = df_hist.sort_index(ascending=False)
        df_hist.index.name = "Tanggal"

        m1, m2, m3 = st.columns(3)
        with m1: st.metric("Total Baris", f"{len(df_hist):,}")
        with m2:
            if 'Harga Aktual (Rp)' in df_hist.columns:
                st.metric("Rata-rata Harga", f"Rp {df_hist['Harga Aktual (Rp)'].mean():,.2f}")
        with m3:
            meta = load_metadata(selected_h)
            if meta and meta.get('mape'):
                st.metric("MAPE Model", f"{meta['mape']:.2f}%")

        st.divider()
        st.markdown("<div class='section-label'>Grafik Pergerakan Harga</div>", unsafe_allow_html=True)
        
        # Sort index ascending so the chart plots left-to-right (past to present)
        df_chart = df_hist.sort_index(ascending=True)
        
        fig_hist = go.Figure()
        
        if 'Harga Aktual (Rp)' in df_chart.columns:
            fig_hist.add_trace(go.Scatter(
                x=df_chart.index, 
                y=df_chart['Harga Aktual (Rp)'],
                name='Harga Aktual', 
                line=dict(color='#1C1C1E', width=2),
                hovertemplate='%{x|%d %b %Y}<br>Harga: Rp %{y:,.2f}<extra></extra>'
            ))
            
        # Optional: If you ever uncomment the 'Prediksi Model' columns in the table, it will plot them too
        if 'Prediksi Model (Rp)' in df_chart.columns:
            fig_hist.add_trace(go.Scatter(
                x=df_chart.index, 
                y=df_chart['Prediksi Model (Rp)'],
                name='Prediksi Model', 
                line=dict(color='#2E7D52', width=2, dash='dot'),
                hovertemplate='%{x|%d %b %Y}<br>Prediksi: Rp %{y:,.2f}<extra></extra>'
            ))

        fig_hist.update_layout(
            height=380,
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0, font=dict(size=11)),
            xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(size=10, color='#6B6B63')),
            yaxis=dict(showgrid=True, gridcolor='#E8E6DF', zeroline=False,
                       tickfont=dict(size=10, color='#6B6B63'), tickprefix='Rp ', tickformat=',.0f'),
            hovermode='x unified'
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        # st.markdown("<div class='data-table-container'>", unsafe_allow_html=True)
        st.divider()
        st.dataframe(
            df_hist, 
            use_container_width=True, 
            height=450,
            column_config={
                "Tanggal": st.column_config.DateColumn(
                    "Tanggal",
                    format="DD MMM YYYY" # Streamlit's frontend format
                )
            }
        )
        st.markdown("</div>", unsafe_allow_html=True)

        csv = df_hist.to_csv().encode('utf-8')
        st.download_button(
            "⬇️ Download CSV", csv,
            file_name=f"harga_{selected_h.replace(' ','_')}_{date_from}_{date_to}.csv",
            mime='text/csv'
        )


# =============================================================================
# PAGE 3: SIMULASI CUACA
# =============================================================================

elif page == "Simulasi Cuaca":
    st.markdown(
        "<div class='page-header'><h1>Simulasi Cuaca Ekstrem</h1>"
        "<p>Lihat dampak skenario cuaca ekstrem terhadap harga komoditas melalui model LSTM</p></div>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<div class='alert-banner'>"
        "⚠️ <strong>Cara kerja:</strong> Cuaca ekstrem yang dipilih diinjeksikan langsung "
        "sebagai input ke model LSTM selama durasi yang ditentukan. Model kemudian "
        "memprediksi harga secara rekursif — dampak pada harga muncul secara alami "
        "melalui pola yang dipelajari model, termasuk lag siklus tanam (30–90 hari). "
        "Setelah periode cuaca ekstrem berakhir, model kembali menggunakan cuaca normal."
        "</div>", unsafe_allow_html=True
    )

    commodities = load_commodities()
    if not commodities:
        st.warning("Belum ada data di database.")
        st.stop()

    # ------------------------------------------------------------------
    # CONTROLS (Top Layout)
    # ------------------------------------------------------------------
    
    # Membagi area kontrol menjadi dua kolom di atas (kiri & kanan)
    top_left, top_right = st.columns(2)

    with top_left:
        selected_s = st.selectbox("Komoditas", commodities, key='sim_kom')

        # Crop cycle info
        if selected_s in CROP_CYCLE:
            cycle, notes = CROP_CYCLE[selected_s]
            st.markdown(
                f"<div style='background:#F0EDE6;border-radius:8px;padding:0.75rem 1rem;"
                f"font-size:0.82rem;color:#1C1C1E;margin-bottom:1rem;line-height:1.6'>"
                f"🌱 <strong>Siklus tanam:</strong> {cycle}<br>"
                f"<span style='color:#6B6B63'>{notes}</span><br>"
                f"<span style='color:#9A9A8E;font-size:0.75rem'>"
                f"💡 Dampak cuaca baru terlihat setelah lag siklus tanam</span>"
                f"</div>",
                unsafe_allow_html=True
            )

    with top_right:
        st.markdown("**Skenario Cuaca**")
        scenario_name = st.selectbox(
            "Pilih skenario", list(WEATHER_SCENARIOS.keys()),
            index=0, key='sim_scenario'
        )
        scenario  = WEATHER_SCENARIOS[scenario_name]
        is_normal = scenario_name == 'Normal'

        # Show badge
        badge_cls = "wx-normal" if is_normal else "wx-extreme"
        st.markdown(
            f"<span class='wx-badge {badge_cls}'>{scenario['label']}</span>",
            unsafe_allow_html=True
        )

        # Scenario parameter preview
        db_s       = get_db()
        df_weather = db_s.get_weather()
        if not df_weather.empty:
            avg_suhu = df_weather['Suhu (C)'].mean()
            avg_ch   = df_weather['Curah Hujan (mm)'].mean()
            avg_rh   = df_weather['Kelembapan (%)'].mean()
            abs_suhu = avg_suhu + scenario['suhu']
            abs_ch   = avg_ch   * scenario['ch']
            abs_rh   = float(np.clip(avg_rh + scenario['rh'], 0, 100))
            st.markdown(
                f"<div style='background:#F7F6F2;border:1px solid #E8E6DF;"
                f"border-radius:8px;padding:0.7rem 1rem;font-size:0.8rem;"
                f"color:#1C1C1E;margin-top:0.5rem;margin-bottom:1.5rem'>"
                f"🌡️ Suhu: <strong>{abs_suhu:.1f}°C</strong> "
                f"({scenario['suhu']:+.0f}°C dari rata-rata {avg_suhu:.1f}°C)<br>"
                f"🌧️ Curah Hujan: <strong>{abs_ch:.1f} mm/hari</strong> "
                f"({scenario['ch']:.1f}× dari rata-rata {avg_ch:.1f} mm)<br>"
                f"💧 Kelembapan: <strong>{abs_rh:.1f}%</strong> "
                f"({scenario['rh']:+.0f}% dari rata-rata {avg_rh:.1f}%)"
                f"</div>",
                unsafe_allow_html=True
            )

        st.markdown("**Durasi & Proyeksi**")
        st.caption(
            "🔁 Simulasi memakai model MINGGUAN (bukan harian) — rekursi jauh "
            "lebih sedikit langkah untuk horizon yang sama, jadi hasilnya lebih "
            "stabil untuk skenario multi-minggu."
        )
        c1, c2 = st.columns(2)
        with c1:
            override_dur = st.number_input(
                "Lama cuaca ekstrem (minggu)",
                min_value=1, step=1, value=2,
                help="Berapa minggu cuaca ekstrem ini berlangsung",
                disabled=is_normal
            )
        with c2:
            sim_days = st.number_input(
                "Total proyeksi (minggu)",
                min_value=int(override_dur) + 2 if not is_normal else 4,
                step=1, value=max(int(override_dur) + 2 if not is_normal else 4, 4),
                help="Harus lebih panjang dari durasi cuaca agar dampaknya terlihat"
            )

        if not is_normal and int(sim_days) <= int(override_dur):
            st.warning("Total proyeksi harus lebih panjang dari durasi cuaca ekstrem agar dampaknya terlihat.")

    # Tombol run berada di luar kolom agar memanjang penuh di bawah form
    st.markdown("<br>", unsafe_allow_html=True)
    run_sim = st.button(
        "▶ Jalankan Simulasi", use_container_width=True,
        type="primary"
    )
    
    st.divider()

    # ------------------------------------------------------------------
    # SIMULATION OUTPUT
    # ------------------------------------------------------------------
    df_pred_s = load_predictions(selected_s)

    if df_pred_s.empty:
        st.info(
            f"Belum ada prediksi untuk {selected_s}. "
            f"Jalankan `python pipeline.py --komoditas \"{selected_s}\"` dulu."
        )
        st.stop()

    # Always show baseline forecast (MINGGUAN)
    df_base_fc = get_weekly_forecast(selected_s, int(sim_days))
    df_sim_fc  = None

    # Key describing the exact params the cached result belongs to
    sim_key = (selected_s, scenario_name, int(override_dur), int(sim_days))

    if run_sim:
        with st.spinner(f"Menjalankan simulasi mingguan {scenario['label']} selama {int(override_dur)} minggu..."):
            try:
                df_sim_fc = generate_weekly_simulation_forecast(
                    komoditas                = selected_s,
                    n_weeks                  = int(sim_days),
                    db                       = db_s,
                    scenario                 = scenario,
                    override_duration_weeks  = int(override_dur),
                    df_weather_hist          = df_weather,
                )
                st.session_state['sim_result'] = {'key': sim_key, 'df': df_sim_fc}
            except Exception as e:
                st.error(f"Simulasi gagal: {e}")
                df_sim_fc = None
                st.session_state.pop('sim_result', None)
    else:
        # Reuse the last simulation only if it still matches the current
        # commodity/scenario/duration/projection selections. This keeps the
        # result visible across unrelated reruns (e.g. expanding the
        # correlation panel) but clears it once the user changes a setting
        # without re-running, so a stale result is never shown.
        cached = st.session_state.get('sim_result')
        if cached and cached['key'] == sim_key:
            df_sim_fc = cached['df']

    if df_sim_fc is not None and not df_sim_fc.empty and not df_base_fc.empty:
        # Summary card
        base_end = df_base_fc['Forecast'].iloc[-1]
        sim_end  = df_sim_fc['Forecast'].iloc[-1]
        diff     = sim_end - base_end
        pct_diff = diff / base_end * 100
        direction = "lebih tinggi" if diff > 0 else "lebih rendah"
        color_bg  = "#FEE2E2" if diff > 0 else "#DCFCE7"
        color_tx  = "#991B1B" if diff > 0 else "#166534"

        # Find when the maximum deviation occurs
        dev = (df_sim_fc['Forecast'] - df_base_fc['Forecast']).abs()
        peak_date = dev.idxmax()
        peak_diff = df_sim_fc.loc[peak_date, 'Forecast'] - df_base_fc.loc[peak_date, 'Forecast']
        peak_pct  = peak_diff / df_base_fc.loc[peak_date, 'Forecast'] * 100

        st.markdown(
            f"<div style='background:{color_bg};border-radius:8px;"
            f"padding:0.9rem 1.1rem;font-size:0.88rem;color:{color_tx};"
            f"margin-bottom:1rem;line-height:1.7'>"
            f"<strong>Hasil Simulasi: {scenario['label']} selama {int(override_dur)} minggu</strong><br>"
            f"Dampak terbesar: <strong>Rp {abs(peak_diff):,.2f} {direction} "
            f"({abs(peak_pct):.1f}%)</strong> pada {peak_date.strftime('%d %b %Y')}<br>"
            f"Pada akhir proyeksi ({int(sim_days)} minggu): "
            f"<strong>Rp {abs(diff):,.2f} {direction} ({abs(pct_diff):.1f}%)</strong>"
            f"</div>",
            unsafe_allow_html=True
        )

    # ------------------------------------------------------------------
    # CHART
    # ------------------------------------------------------------------
    fig_s     = go.Figure()
    df_hist_s = df_pred_s.tail(30)

    # Historical actual prices
    fig_s.add_trace(go.Scatter(
        x=df_hist_s.index, y=df_hist_s['Aktual'],
        name='Harga Aktual', line=dict(color='#1C1C1E', width=2),
        hovertemplate='%{x|%d %b %Y}<br>Aktual: Rp %{y:,.2f}<extra></extra>'
    ))

    # Baseline forecast (normal weather)
    if not df_base_fc.empty:
        anchor_x = df_hist_s.index[-1]
        anchor_y = df_hist_s['Incr'].iloc[-1]
        bx = [anchor_x] + list(df_base_fc.index)
        by = [anchor_y] + list(df_base_fc['Forecast'])
        bl = [anchor_y] + list(df_base_fc['Lower'])
        bu = [anchor_y] + list(df_base_fc['Upper'])

        # Baseline confidence band
        fig_s.add_trace(go.Scatter(
            x=list(bx) + list(reversed(bx)),
            y=list(bu) + list(reversed(bl)),
            fill='toself', fillcolor='rgba(46,125,82,0.08)',
            line=dict(color='rgba(0,0,0,0)'),
            name='CI Normal 90%', showlegend=True, hoverinfo='skip'
        ))
        fig_s.add_trace(go.Scatter(
            x=bx, y=by, name='Proyeksi Normal',
            line=dict(color='#2E7D52', width=2, dash='dot'),
            hovertemplate='%{x|%d %b %Y}<br>Normal: Rp %{y:,.2f}<extra></extra>'
        ))

    # Simulation forecast
    if df_sim_fc is not None and not df_sim_fc.empty:
        sx = [df_hist_s.index[-1]] + list(df_sim_fc.index)
        sy = [df_hist_s['Incr'].iloc[-1]] + list(df_sim_fc['Forecast'])
        sl = [df_hist_s['Incr'].iloc[-1]] + list(df_sim_fc['Lower'])
        su = [df_hist_s['Incr'].iloc[-1]] + list(df_sim_fc['Upper'])

        # Simulation confidence band
        fig_s.add_trace(go.Scatter(
            x=list(sx) + list(reversed(sx)),
            y=list(su) + list(reversed(sl)),
            fill='toself', fillcolor='rgba(217,69,61,0.08)',
            line=dict(color='rgba(0,0,0,0)'),
            name=f'CI {scenario["label"]} 90%', showlegend=True, hoverinfo='skip'
        ))
        fig_s.add_trace(go.Scatter(
            x=sx, y=sy, name=f'Skenario: {scenario["label"]}',
            line=dict(color='#D94F3D', width=2.5, dash='dash'),
            hovertemplate='%{x|%d %b %Y}<br>Skenario: Rp %{y:,.2f}<extra></extra>'
        ))

        # Shade the override period (when extreme weather is active)
        if not df_sim_fc.empty:
            override_end_date = df_sim_fc.index[min(int(override_dur)-1, len(df_sim_fc)-1)]
            fig_s.add_vrect(
                x0=df_sim_fc.index[0], x1=override_end_date,
                fillcolor='rgba(217,69,61,0.07)', layer='below', line_width=0,
                annotation_text=f"⛈️ {scenario['label']}",
                annotation_position="top left",
                annotation_font=dict(size=10, color='#D94F3D'),
            )
            if override_end_date < df_sim_fc.index[-1]:
                fig_s.add_vline(
                    x=override_end_date,
                    line_dash='dot', line_color='#D94F3D', line_width=1.5,
                )
                fig_s.add_annotation(
                    x=override_end_date, y=1, yref='paper',
                    text='Cuaca normal kembali →',
                    showarrow=False, font=dict(size=9, color='#D94F3D'),
                    xanchor='left', yanchor='top', xshift=5,
                )

    fig_s.update_layout(
        height=420,
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            orientation='h', yanchor='bottom', y=1.02,
            xanchor='left', x=0, font=dict(size=10)
        ),
        xaxis=dict(
            showgrid=False, zeroline=False,
            tickfont=dict(size=10, color='#6B6B63')
        ),
        yaxis=dict(
            showgrid=True, gridcolor='#E8E6DF', zeroline=False,
            tickfont=dict(size=10, color='#6B6B63'),
            tickprefix='Rp ', tickformat=',.0f'
        ),
        hovermode='x unified'
    )
    st.plotly_chart(fig_s, use_container_width=True)

    st.caption(
        "Garis merah = proyeksi skenario cuaca ekstrem · "
        "Area merah muda = periode cuaca ekstrem aktif · "
        "Garis hijau = proyeksi normal sebagai pembanding"
    )

    # # Correlation table
    with st.expander("📊 Lihat korelasi historis cuaca–harga"):
        df_prices_info = db_s.get_prices(selected_s)
        if not df_prices_info.empty and not df_weather.empty:
            comb = df_prices_info.join(df_weather, how='inner').dropna()
            if len(comb) > 10:
                corr_df = pd.DataFrame({
                    'Variabel': [
                        'Suhu (°C) — sama hari',
                        'Suhu (°C) — lag 14 hari',
                        'Suhu (°C) — lag 30 hari',
                        'Suhu (°C) — lag 60 hari',
                        'Curah Hujan (mm) — sama hari',
                        'Curah Hujan — lag 14 hari',
                        'Curah Hujan — lag 30 hari',
                        'Curah Hujan — lag 60 hari',
                        'Kelembapan (%)'
                    ],
                    'Korelasi dengan Harga': [
                        comb['Harga (Rp)'].corr(comb['Suhu (C)']),
                        comb['Harga (Rp)'].corr(comb['Suhu (C)'].shift(14).fillna(0)),
                        comb['Harga (Rp)'].corr(comb['Suhu (C)'].shift(30).fillna(0)),
                        comb['Harga (Rp)'].corr(comb['Suhu (C)'].shift(60).fillna(0)),
                        comb['Harga (Rp)'].corr(comb['Curah Hujan (mm)']),
                        comb['Harga (Rp)'].corr(comb['Curah Hujan (mm)'].shift(14).fillna(0)),
                        comb['Harga (Rp)'].corr(comb['Curah Hujan (mm)'].shift(30).fillna(0)),
                        comb['Harga (Rp)'].corr(comb['Curah Hujan (mm)'].shift(60).fillna(0)),
                        comb['Harga (Rp)'].corr(comb['Kelembapan (%)']),
                    ]
                }).set_index('Variabel').round(3)

                # Color code: stronger correlation = more important
                st.dataframe(corr_df, use_container_width=True)
                st.caption(
                    "Nilai mendekati +1 atau −1 = korelasi kuat. "
                    "Lag yang menunjukkan korelasi terkuat mencerminkan "
                    "berapa hari setelah hujan/kemarau harga biasanya berubah."
                )