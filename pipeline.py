"""
pipeline.py
-----------
Orkestrator utama — menghubungkan semua komponen:
  1. Weather fetcher  (Open-Meteo)
  2. Price loader     (Excel untuk sekarang, nanti PIHPS scraper)
  3. Preprocessing    (merge, interpolasi, lag features)
  4. LSTM model       (base training + incremental fine-tune)
  5. Database         (simpan semua hasil)

PENGGUNAAN:
  # Jalankan pipeline penuh (training + update)
  python pipeline.py

  # Hanya update data & prediksi, skip re-training base model
  python pipeline.py --no-retrain

  # Jalankan untuk komoditas tertentu saja
  python pipeline.py --komoditas "Cabai Merah"

  # Jalankan untuk semua komoditas
  python pipeline.py --all

  # Cek status tanpa menjalankan apapun
  python pipeline.py --status

REQUIREMENT:
  pip install tensorflow pandas numpy scikit-learn openpyxl requests
"""

import os
import sys
# Add this before importing TensorFlow to disable the oneDNN warning
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# (Optional) Add this to hide ALL TensorFlow INFO and WARNING logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging
import argparse
import pickle
import numpy as np
import pandas as pd
import holidays as hol_lib
from datetime import date, timedelta
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential, clone_model, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import random
import warnings
warnings.filterwarnings('ignore')

from database import Database
import weather_fetcher
from pihps_scraper import fetch_all_komoditas as pihps_fetch_all, fetch_and_update as pihps_fetch_one
from holiday_features import compute_holiday_features, HOLIDAY_FEATURE_COLS as _HF_COLS  # dipakai khusus oleh pipeline mingguan (lihat bagian bawah)

# ---------------------------------------------------------------------------
# KUNCI RANDOMNESS (UNTUK REPRODUKTIBILITAS PIPELINE)
# ---------------------------------------------------------------------------
SEED = 123
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__name__)

# =============================================================================
# KONFIGURASI
# =============================================================================

# File sumber harga (sementara pakai Excel, nanti ganti pihps_scraper)
HARGA_FILE = 'Tabel Harga Berdasarkan Daerah (1).xlsx'

# Database
DB_PATH = 'pihps.db'

# Folder untuk menyimpan model yang sudah dilatih
MODEL_DIR = Path('models')
MODEL_DIR.mkdir(exist_ok=True)

# Komoditas default
DEFAULT_KOMODITAS = 'Cabai Merah'

# Semua komoditas yang tersedia (10 komoditas dasar — tanpa turunan/varian,
# selaras dengan pihps_scraper.KOMODITAS_MAP yang sudah disederhanakan)
ALL_KOMODITAS = [
    'Beras', 'Daging Ayam', 'Daging Sapi', 'Telur Ayam',
    'Bawang Merah', 'Bawang Putih', 'Cabai Merah', 'Cabai Rawit',
    'Minyak Goreng', 'Gula Pasir',
]

# Lag curah hujan per komoditas (hari) — mencerminkan siklus produksi/tanam
# masing-masing, HANYA untuk komoditas dengan keterkaitan curah hujan
# domestik yang jelas. Komoditas yang tidak dicantumkan otomatis memakai
# default [14, 30] (lihat CROP_LAGS.get() di bawah) alih-alih dipaksakan
# siklus tanam yang tidak relevan.
#
# Beras        : siklus 90-120 hari, musiman kuat -> lag 30, 60, 90...
# Bawang Merah : siklus 60-75 hari, sensitif hujan -> lag 14, 45, 75
# Cabai Merah  : siklus 90-120 hari, kritis di pembungaan -> lag 30, 60, 90
# Cabai Rawit  : siklus 70-90 hari, lebih cepat -> lag 14, 45, 75
# Daging Ayam  : siklus ternak broiler pendek (~30-40 hari DOC->panen) +
#                lag biaya pakan (jagung) -> lag 30, 45, 60
# Telur Ayam   : ayam petelur matang ~18-22 minggu + lag biaya pakan
#                -> lag 30, 60, 90
# Daging Sapi  : siklus penggemukan lebih panjang (3-4+ bulan), ketersediaan
#                pakan/rumput terkait hujan -> lag 90, 120, 180
# Gula Pasir   : tebu, siklus ~10-12 bulan, panen musim kemarau di Jawa
#                -> lag 90, 180, 270
#
# SENGAJA TIDAK dimasukkan (pakai default [14, 30]):
# Bawang Putih : ~90%+ pasokan impor (terutama Tiongkok) — hubungan dengan
#                curah hujan domestik lemah, siklus tanam lokal tidak
#                representatif untuk harga.
# Minyak Goreng: dari CPO, panen sawit kontinu sepanjang tahun (tidak
#                musiman tajam), harga lebih didorong pasar CPO
#                global/pungutan ekspor/kebijakan DMO daripada curah hujan
#                domestik jangka pendek.
CROP_LAGS = {
    'Beras'        : [30, 60, 90, 120, 150, 180],
    'Bawang Merah' : [14, 45, 75, 120],
    'Cabai Merah'  : [30, 60, 90, 120, 150, 180],
    'Cabai Rawit'  : [14, 45, 75, 120, 150],
    'Daging Ayam'  : [30, 45, 60],
    'Telur Ayam'   : [30, 60, 90],
    'Daging Sapi'  : [90, 120, 180],
    'Gula Pasir'   : [90, 180, 270],
}

# Split data
BASE_START = '2021-01-01'
BASE_END   = '2025-12-31'
INCR_START = '2026-01-01'

# Hyperparameter model
WINDOW_SIZE          = 30
FINETUNE_LR          = 0.0001
REPLAY_BUFFER_DAYS   = 30
REPLAY_BASE_SAMPLES  = 60
FINETUNE_EPOCHS      = 5
TRAIN_EPOCHS         = 100
BATCH_SIZE           = 32

# Soft weight blending (Polyak averaging) after each fine-tune cycle:
#   new_weights = ALPHA * finetuned_weights + (1 - ALPHA) * base_weights
# Each cycle only partially commits to what that cycle's small, partly-random
# replay batch learned, and pulls back toward the known-stable base model the
# rest of the way. This directly caps how far the incremental model's weights
# can drift from the base over many repeated fine-tune cycles — the drift
# mechanism believed responsible for the incremental model's instability
# under sustained/recursive weather simulation, without adding any extra
# training compute (it's a weighted average of already-computed weights).
FINETUNE_BLEND_ALPHA = 0.25

BASE_FEATURE_COLS = [
    'Harga', 'Suhu (C)', 'Kelembapan (%)', 'Curah Hujan (mm)',
    'CH_30d_sum', 'consecutive_dry_days'
]


BASE_HOLIDAY_COLS = [
    'is_idul_fitri', 'is_idul_adha', 'is_christmas',
    'is_new_year', 'is_ramadhan',
]
HOLIDAY_FEATURE_COLS = BASE_HOLIDAY_COLS + [f'{c}_H7' for c in BASE_HOLIDAY_COLS]


def get_feature_cols(komoditas: str) -> list[str]:
    """
    Return daftar feature kolom sesuai komoditas.
    Lag curah hujan disesuaikan dengan siklus tanam masing-masing komoditas.
    Fitur hari raya (12 kolom) ditambahkan untuk semua komoditas.
    """
    lags = CROP_LAGS.get(komoditas, [14, 30])
    cols = BASE_FEATURE_COLS + [f'CH_lag{d}' for d in lags]+ HOLIDAY_FEATURE_COLS
    return cols


# =============================================================================
# MODEL MINGGUAN — khusus untuk fitur Simulasi Cuaca
# =============================================================================
# Model harian yang direkursi 30x (satu hari per langkah) untuk simulasi
# multi-minggu terbukti tidak stabil — setiap langkah menambah sedikit galat
# yang terus terkompensasi lewat feedback loop, dan setelah cukup banyak
# langkah bisa lari jauh dari kewajaran (lihat riwayat debugging simulasi
# cuaca). Satu cara mengurangi itu tanpa mengubah arsitektur: kurangi JUMLAH
# langkah rekursif yang dibutuhkan untuk mencakup horizon yang sama — sebuah
# simulasi 4 minggu butuh ~4 langkah mingguan, dibanding ~28 langkah harian.
#
# Model ini SENGAJA dibuat tanpa fine-tuning inkremental (yang juga terbukti
# jadi sumber ketidakstabilan lain) — cukup dilatih ulang penuh (base-only)
# setiap kali datanya di-refresh, karena di level mingguan jumlah baris jauh
# lebih sedikit sehingga retrain penuh jadi murah untuk dijalankan berkala.

WEEKLY_WINDOW_SIZE = 12   # ~1 kuartal riwayat mingguan sebagai window LSTM

# Lag mingguan diturunkan langsung dari CROP_LAGS harian (dibagi 7, dibulatkan,
# minimum 1 minggu, deduplikasi) supaya siklus tanam yang sama tetap
# tercermin, hanya dalam satuan minggu.
CROP_LAGS_WEEKS = {
    k: sorted(set(max(1, round(d / 7)) for d in v))
    for k, v in CROP_LAGS.items()
}

WEEKLY_BASE_FEATURE_COLS = [
    'Harga', 'Suhu (C)', 'Kelembapan (%)', 'Curah Hujan (mm)',
    'CH_4wk_sum', 'consecutive_dry_weeks',
]


def get_weekly_feature_cols(komoditas: str) -> list[str]:
    """Analog get_feature_cols(), tapi untuk model mingguan."""
    lags = CROP_LAGS_WEEKS.get(komoditas, [2, 4])
    cols = WEEKLY_BASE_FEATURE_COLS + [f'CH_lag{d}w' for d in lags]
    return cols

# =============================================================================
# TAHAP 1: UPDATE DATA CUACA
# =============================================================================

def update_weather(db: Database):
    """
    Cek tanggal cuaca terakhir di database, fetch hanya yang baru
    dari Open-Meteo, simpan ke database.
    """
    log.info("--- Update data cuaca ---")

    last = db.get_last_weather_date()
    start = (last + timedelta(days=1)) if last else date(2021, 1, 1)
    end   = date.today()

    if start > end:
        log.info(f"Cuaca sudah up-to-date (s/d {last})")
        return

    log.info(f"Fetch cuaca: {start} s/d {end}")
    df_new = weather_fetcher.fetch_and_update(
        start_date=start,
        end_date=end,
        save_csv=False,   # sudah disimpan ke database, tidak perlu CSV terpisah
    )

    if df_new.empty:
        log.warning("Tidak ada data cuaca baru")
        return

    # Hanya simpan yang benar-benar baru
    df_to_save = df_new[df_new.index >= pd.Timestamp(start)]
    db.upsert_weather(df_to_save)
    log.info(f"Cuaca: {len(df_to_save)} hari baru disimpan")


# =============================================================================
# TAHAP 2: UPDATE DATA HARGA
# =============================================================================

def update_prices(db: Database, komoditas: str):
    """
    Update harga dari PIHPS API.
    Jika PIHPS gagal (site down / error), fallback ke Excel.
    Hanya fetch tanggal yang belum ada di database.
    """
    log.info(f"--- Update harga [{komoditas}] ---")

    last = db.get_last_price_date(komoditas)

    # --- Coba PIHPS API dulu ---
    try:
        df_new = pihps_fetch_one(
            komoditas  = komoditas,
            start_date = date(2021, 1, 1),
            last_date  = last
        )
        if not df_new.empty:
            db.upsert_prices(df_new, komoditas, sumber='pihps')
            log.info(f"Harga [{komoditas}]: {len(df_new)} baris baru dari PIHPS")
            return
        else:
            log.info(f"Harga [{komoditas}] sudah up-to-date via PIHPS")
            return

    except Exception as e:
        log.warning(f"PIHPS gagal untuk [{komoditas}]: {e} — fallback ke Excel")

    # --- Fallback: Excel ---
    if not Path(HARGA_FILE).exists():
        log.error(f"Excel fallback tidak ditemukan: {HARGA_FILE}")
        return

    df_harga = pd.read_excel(HARGA_FILE)
    df_k     = df_harga[df_harga['Komoditas (Rp)'] == komoditas].copy()

    if df_k.empty:
        log.warning(f"Komoditas '{komoditas}' tidak ditemukan di Excel")
        return

    df_long = df_k.melt(
        id_vars=['No', 'Komoditas (Rp)'],
        var_name='Tanggal',
        value_name='Harga'
    )
    df_long['Tanggal'] = pd.to_datetime(
        df_long['Tanggal'], format='%d/ %m/ %Y', errors='coerce'
    )
    df_long['Harga'] = (
        df_long['Harga']
        .replace({',': '', '-': np.nan}, regex=True)
        .astype(float)
    )
    df_long = (
        df_long[['Tanggal', 'Harga']]
        .dropna(subset=['Tanggal', 'Harga'])
        .set_index('Tanggal')
        .groupby(level=0).mean()
        .sort_index()
        .rename(columns={'Harga': 'Harga (Rp)'})
    )

    df_new = df_long[df_long.index > pd.Timestamp(last)] \
             if last else df_long

    if df_new.empty:
        log.info(f"Harga [{komoditas}] sudah up-to-date via Excel")
        return

    db.upsert_prices(df_new, komoditas, sumber='excel')
    log.info(f"Harga [{komoditas}]: {len(df_new)} baris baru dari Excel (fallback)")


# =============================================================================
# TAHAP 3: PREPROCESSING
# =============================================================================

def preprocess(db: Database, komoditas: str) -> tuple:
    """
    Ambil data dari database, merge, interpolasi, tambah lag features
    sesuai siklus tanam komoditas.

    Return:
        df_gabung       : DataFrame lengkap (semua tanggal)
        real_trade_days : index tanggal perdagangan nyata
        base_data       : slice 2021-2024
        incr_data       : slice 2025+
        feature_cols    : list kolom fitur untuk komoditas ini
    """
    log.info(f"--- Preprocessing [{komoditas}] ---")

    df_harga = db.get_prices(komoditas)
    df_cuaca = db.get_weather()

    if df_harga.empty:
        raise ValueError(f"Tidak ada data harga untuk '{komoditas}' di database")
    if df_cuaca.empty:
        raise ValueError("Tidak ada data cuaca di database")

    df_harga = df_harga.rename(columns={'Harga (Rp)': 'Harga'})

    # Simpan hari perdagangan asli sebelum interpolasi
    real_trade_days = df_harga[df_harga['Harga'].notna()].index

    # Tambah lag curah hujan sesuai siklus tanam komoditas
    lags = CROP_LAGS.get(komoditas, [14, 30])
    for lag in lags:
        df_cuaca[f'CH_lag{lag}'] = df_cuaca['Curah Hujan (mm)'].shift(lag)

    # 1. Rolling 30-day rainfall sum
    df_cuaca['CH_30d_sum'] = df_cuaca['Curah Hujan (mm)'].rolling(window=30, min_periods=1).sum()
    
    # 2. Consecutive Dry Days (< 1mm rainfall)
    is_dry = (df_cuaca['Curah Hujan (mm)'] < 1.0).astype(int)
    df_cuaca['consecutive_dry_days'] = is_dry.groupby((is_dry != is_dry.shift()).cumsum()).cumsum()

    # Join & resample harian
    df_gabung = df_harga.join(df_cuaca, how='outer')
    df_gabung = df_gabung.resample('D').asfreq()
    df_gabung.interpolate(method='linear', inplace=True)
    df_gabung.bfill(inplace=True)

    # ------------------------------------------------------------------
    # Fitur hari raya — inline (tidak perlu file terpisah)
    # ------------------------------------------------------------------
    years_in_data = list(range(
        df_gabung.index.min().year,
        df_gabung.index.max().year + 2
    ))
    id_holidays = hol_lib.ID(years=years_in_data)

    df_gabung['_hname'] = [
        str(id_holidays.get(d.date(), '')).lower()
        for d in df_gabung.index
    ]

    df_gabung['is_idul_fitri'] = df_gabung['_hname'].str.contains(
        'idul fitri|eid al-fitr', regex=True).astype(int)
    df_gabung['is_idul_adha']  = df_gabung['_hname'].str.contains(
        'idul adha|eid al-adha',  regex=True).astype(int)
    df_gabung['is_christmas']  = df_gabung['_hname'].str.contains(
        'natal|christmas',        regex=True).astype(int)
    df_gabung['is_new_year']   = (
        (df_gabung.index.month == 1) & (df_gabung.index.day == 1)
    ).astype(int)

    # Ramadhan — generate tanggal awal Ramadhan via hijridate
    try:
        from hijridate import Hijri
        ramadhan_starts = []
        min_h = min(years_in_data) - 580
        max_h = max(years_in_data) - 578
        for h_year in range(min_h, max_h + 3):
            try:
                g = Hijri(h_year, 9, 1).to_gregorian()
                if g.year in years_in_data:
                    ramadhan_starts.append(
                        pd.Timestamp(g.year, g.month, g.day)
                    )
            except Exception:
                continue
    except ImportError:
        log.warning("hijridate tidak terinstall — is_ramadhan akan bernilai 0. "
                    "Jalankan: pip install hijridate")
        ramadhan_starts = []

    is_ramadhan = np.zeros(len(df_gabung), dtype=int)
    for i, d in enumerate(df_gabung.index):
        for ram_start in sorted(ramadhan_starts):
            if ram_start <= d:
                # Cari Idul Fitri terdekat setelah ram_start
                idul_fitri = None
                for cd in pd.date_range(ram_start, ram_start + pd.Timedelta(days=45)):
                    nm = str(id_holidays.get(cd.date(), '')).lower()
                    if 'idul fitri' in nm or 'eid al-fitr' in nm:
                        idul_fitri = cd
                        break
                if idul_fitri and ram_start <= d < idul_fitri:
                    is_ramadhan[i] = 1
                    break
    df_gabung['is_ramadhan'] = is_ramadhan
    df_gabung.drop(columns=['_hname'], inplace=True)

    # Efek H-7: flag 1 selama 7 hari sebelum setiap hari raya (vectorized)
    for col in BASE_HOLIDAY_COLS:
        df_gabung[f'{col}_H7'] = (
            df_gabung[col]
            .rolling(window=8, min_periods=1)
            .max()
            .shift(-7)
            .fillna(0)
            .astype(int) - df_gabung[col]
        ).clip(lower=0)

    # Ambil kolom yang sesuai untuk komoditas ini
    feature_cols = get_feature_cols(komoditas)
    df_gabung    = df_gabung[feature_cols]

    log.info(
        f"Data gabungan: {len(df_gabung)} baris "
        f"({df_gabung.index.min().date()} s/d {df_gabung.index.max().date()})\n"
        f"  Features: {feature_cols}"
    )

    base_data = df_gabung.loc[BASE_START:BASE_END]
    incr_data = df_gabung.loc[INCR_START:]

    if base_data.empty:
        raise ValueError(f"Tidak ada data base ({BASE_START} - {BASE_END})")
    if incr_data.empty:
        raise ValueError(f"Tidak ada data incremental (mulai {INCR_START})")

    log.info(f"Base: {len(base_data)} hari | Incremental: {len(incr_data)} hari")

    return df_gabung, real_trade_days, base_data, incr_data, feature_cols


# =============================================================================
# PREPROCESSING MINGGUAN — untuk model simulasi cuaca
# =============================================================================

def preprocess_weekly(db: Database, komoditas: str) -> tuple:
    """
    Analog preprocess(), tapi agregasi ke MINGGUAN sebelum menghitung fitur
    lag/musiman/hari raya. Dipakai khusus oleh model simulasi cuaca — lihat
    catatan panjang di dekat get_weekly_feature_cols().

    Agregasi:
        Harga, Suhu, Kelembapan : rata-rata mingguan
        Curah Hujan             : JUMLAH mingguan (curah hujan per minggu
                                   lebih bermakna secara fisik daripada
                                   rata-rata harian yang di-rata-rata lagi)

    Return: (df_gabung_weekly, base_data, incr_data, feature_cols)
    Tidak mengembalikan real_trade_days — evaluasi model mingguan memakai
    semua baris minggu (tidak ada konsep "hari libur pasar" di level minggu).
    """
    log.info(f"--- Preprocessing MINGGUAN [{komoditas}] ---")

    df_harga = db.get_prices(komoditas)
    df_cuaca = db.get_weather()

    if df_harga.empty:
        raise ValueError(f"Tidak ada data harga untuk '{komoditas}' di database")
    if df_cuaca.empty:
        raise ValueError("Tidak ada data cuaca di database")

    df_harga = df_harga.rename(columns={'Harga (Rp)': 'Harga'})

    # Join harian dulu (harga + cuaca), interpolasi gap harian SEBELUM
    # diagregasi ke mingguan — supaya minggu dengan beberapa hari libur
    # pasar tidak kehilangan sinyal cuaca hari-hari di sekitarnya.
    df_daily = df_harga.join(df_cuaca, how='outer')
    df_daily = df_daily.resample('D').asfreq()
    df_daily[['Harga', 'Suhu (C)', 'Kelembapan (%)']] = (
        df_daily[['Harga', 'Suhu (C)', 'Kelembapan (%)']].interpolate(method='linear')
    )
    df_daily['Curah Hujan (mm)'] = df_daily['Curah Hujan (mm)'].fillna(0.0)
    df_daily.bfill(inplace=True)

    # Agregasi ke minggu (Senin sebagai awal minggu)
    df_w = df_daily.resample('W-MON', label='left', closed='left').agg({
        'Harga'            : 'mean',
        'Suhu (C)'         : 'mean',
        'Kelembapan (%)'   : 'mean',
        'Curah Hujan (mm)' : 'sum',
    })
    df_w.dropna(subset=['Harga'], inplace=True)

    # Lag mingguan sesuai siklus tanam (dalam minggu)
    lags = CROP_LAGS_WEEKS.get(komoditas, [2, 4])
    for lag in lags:
        df_w[f'CH_lag{lag}w'] = df_w['Curah Hujan (mm)'].shift(lag)

    # Analog CH_30d_sum / consecutive_dry_days, tapi mingguan:
    # ~30 hari = ~4 minggu, dan ambang "kering" 1mm/hari -> ~7mm/minggu
    df_w['CH_4wk_sum'] = df_w['Curah Hujan (mm)'].rolling(window=4, min_periods=1).sum()
    is_dry_week = (df_w['Curah Hujan (mm)'] < 7.0).astype(int)
    df_w['consecutive_dry_weeks'] = (
        is_dry_week.groupby((is_dry_week != is_dry_week.shift()).cumsum()).cumsum()
    )

    # Fitur hari raya — pakai modul bersama holiday_features.py (lihat
    # docstring modul tsb: satu-satunya implementasi, dipakai apa adanya
    # di sini karena ini kode BARU, bukan mengganti logika inline yang
    # sudah ada di preprocess()/forecaster.py)
    df_hol = compute_holiday_features(df_w.index)
    # Untuk minggu (bukan hari), "H-7" dari flag harian yang di-reindex ke
    # index mingguan berarti "flag aktif jika ADA hari dalam minggu itu yang
    # H-7" — cukup ambil max per minggu dari komputasi harian aslinya.
    df_hol_daily = compute_holiday_features(df_daily.index)
    df_hol_weekly = df_hol_daily.resample('W-MON', label='left', closed='left').max()
    df_w = df_w.join(df_hol_weekly.reindex(df_w.index).fillna(0).astype(int))

    df_w.interpolate(method='linear', inplace=True)
    df_w.bfill(inplace=True)
    df_w.fillna(0, inplace=True)

    feature_cols = get_weekly_feature_cols(komoditas)
    df_w = df_w[feature_cols]

    log.info(
        f"Data mingguan: {len(df_w)} minggu "
        f"({df_w.index.min().date()} s/d {df_w.index.max().date()})\n"
        f"  Features: {feature_cols}"
    )

    base_data = df_w.loc[BASE_START:BASE_END]
    incr_data = df_w.loc[INCR_START:]

    if len(base_data) < WEEKLY_WINDOW_SIZE * 2:
        raise ValueError(
            f"Data mingguan base terlalu sedikit ({len(base_data)} minggu) "
            f"untuk window {WEEKLY_WINDOW_SIZE} — perlu setidaknya "
            f"{WEEKLY_WINDOW_SIZE * 2} minggu."
        )

    log.info(f"Base: {len(base_data)} minggu | Incremental: {len(incr_data)} minggu")

    return df_w, base_data, incr_data, feature_cols



def create_sliding_window(data: np.ndarray, window_size: int):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size, :])
        y.append(data[i + window_size, 0])
    return np.array(X), np.array(y)


def prepare_windows(base_data, incr_data, scaler):
    """Normalisasi dan buat sliding windows untuk base dan incremental data."""
    base_scaled = scaler.fit_transform(base_data)
    incr_scaled = scaler.transform(incr_data)

    X_base, y_base = create_sliding_window(base_scaled, WINDOW_SIZE)

    # Stream: sambung 30 hari terakhir base agar hari pertama incr bisa diprediksi
    stream_input    = np.vstack((base_scaled[-WINDOW_SIZE:], incr_scaled))
    X_stream, y_stream = create_sliding_window(stream_input, WINDOW_SIZE)

    # Split base: 80% train, 20% val
    split   = int(len(X_base) * 0.8)
    X_train = X_base[:split];  y_train = y_base[:split]
    X_val   = X_base[split:];  y_val   = y_base[split:]

    return X_train, y_train, X_val, y_val, X_base, y_base, X_stream, y_stream


# =============================================================================
# TAHAP 5: BUILD & TRAIN BASE MODEL
# =============================================================================

def build_model(input_shape):
    model = Sequential([
        LSTM(64, activation='tanh', return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32, activation='tanh'),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model


def train_base_model(X_train, y_train, X_val, y_val, komoditas: str):
    """
    Latih base model. Simpan ke disk agar tidak perlu training ulang
    setiap pipeline dijalankan.
    """
    model_path  = MODEL_DIR / f'base_{komoditas.replace(" ", "_")}.keras'
    scaler_path = MODEL_DIR / f'scaler_{komoditas.replace(" ", "_")}.pkl'

    log.info(f"Training base model [{komoditas}]...")

    model = build_model((X_train.shape[1], X_train.shape[2]))
    early_stop = EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=True
    )
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=TRAIN_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=1
    )

    model.save(model_path)
    log.info(f"Base model disimpan: {model_path}")

    return model, history


def load_or_train_base(
    X_train, y_train, X_val, y_val,
    komoditas: str,
    force_retrain: bool = False
) -> tuple:
    """
    Load base model dari disk jika ada, train baru jika belum ada atau force.
    """
    model_path = MODEL_DIR / f'base_{komoditas.replace(" ", "_")}.keras'

    if model_path.exists() and not force_retrain:
        log.info(f"Load base model dari disk: {model_path}")
        model = load_model(model_path)
        return model, None
    else:
        return train_base_model(X_train, y_train, X_val, y_val, komoditas)


# =============================================================================
# MODEL MINGGUAN — training & pipeline standalone
# =============================================================================

def train_weekly_model(X_train, y_train, X_val, y_val, komoditas: str):
    """
    Latih model mingguan. TIDAK ada fine-tuning inkremental untuk model ini
    (lihat catatan di get_weekly_feature_cols()) — cukup retrain penuh
    setiap kali dijalankan, karena jumlah baris mingguan jauh lebih sedikit
    sehingga full retrain tetap murah untuk dijalankan berkala.
    """
    model_path = MODEL_DIR / f'weekly_{komoditas.replace(" ", "_")}.keras'
    log.info(f"Training model MINGGUAN [{komoditas}]...")

    model = build_model((X_train.shape[1], X_train.shape[2]))
    early_stop = EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=True
    )
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=TRAIN_EPOCHS,
        batch_size=min(BATCH_SIZE, max(4, len(X_train) // 4)),  # data mingguan jauh lebih sedikit
        callbacks=[early_stop],
        verbose=1
    )
    model.save(model_path)
    log.info(f"Model mingguan disimpan: {model_path}")
    return model


def run_weekly_pipeline(komoditas: str, db: Database, force_retrain: bool = True) -> bool:
    """
    Pipeline khusus model mingguan untuk fitur Simulasi Cuaca. Terpisah dari
    run_pipeline() (harian) karena tidak butuh tahap incremental sama sekali.

    force_retrain default True karena tujuannya memang "retrain penuh
    berkala" — beda dari model harian yang defaultnya menghindari retrain
    kecuali diminta.
    """
    log.info(f"\n{'='*60}")
    log.info(f"PIPELINE MINGGUAN: {komoditas}")
    log.info(f"{'='*60}")

    try:
        df_w, base_data, incr_data, feature_cols = preprocess_weekly(db, komoditas)

        # Model mingguan dilatih atas SELURUH data (base + incremental
        # digabung) — tidak ada replay/fine-tune terpisah untuk data
        # terbaru, jadi tidak perlu menyisihkan porsi "incremental".
        full_data = pd.concat([base_data, incr_data]).sort_index()
        full_data = full_data[~full_data.index.duplicated(keep='last')]

        scaler = MinMaxScaler()
        full_scaled = scaler.fit_transform(full_data)
        X_all, y_all = create_sliding_window(full_scaled, WEEKLY_WINDOW_SIZE)

        if len(X_all) < 20:
            raise ValueError(
                f"Hanya {len(X_all)} sequence mingguan tersedia — terlalu "
                "sedikit untuk training yang wajar."
            )

        split   = int(len(X_all) * 0.85)
        X_train, y_train = X_all[:split], y_all[:split]
        X_val,   y_val   = X_all[split:], y_all[split:]

        model_path = MODEL_DIR / f'weekly_{komoditas.replace(" ", "_")}.keras'
        if model_path.exists() and not force_retrain:
            log.info(f"Load model mingguan dari disk: {model_path}")
        else:
            train_weekly_model(X_train, y_train, X_val, y_val, komoditas)

        scaler_path = MODEL_DIR / f'weekly_scaler_{komoditas.replace(" ", "_")}.pkl'
        with open(scaler_path, 'wb') as f:
            pickle.dump({'scaler': scaler, 'feature_cols': feature_cols}, f)

        log.info(f"Pipeline mingguan [{komoditas}] selesai ✓")
        return True

    except Exception as e:
        log.error(f"Pipeline mingguan [{komoditas}] GAGAL: {e}", exc_info=True)
        return False


# =============================================================================
# TAHAP 6: INCREMENTAL LEARNING
# =============================================================================

def run_incremental(
    base_model,
    X_base, y_base,
    X_stream, y_stream
) -> tuple[list, int]:
    """
    Jalankan incremental learning dengan replay buffer.
    Return (predictions_scaled, finetune_count)
    """
    # Clone model agar base model tidak berubah
    incr_model = clone_model(base_model)
    incr_model.set_weights(base_model.get_weights())
    incr_model.compile(optimizer=Adam(learning_rate=FINETUNE_LR), loss='mse')

    predictions   = []
    buffer_X      = []
    buffer_y      = []
    finetune_count = 0

    for i in range(len(X_stream)):
        # Predict
        pred = incr_model.predict(X_stream[i:i+1], verbose=0)[0][0]
        predictions.append(pred)

        # Akumulasi buffer
        buffer_X.append(X_stream[i])
        buffer_y.append(y_stream[i])

        # Fine-tune setiap buffer penuh
        if len(buffer_X) >= REPLAY_BUFFER_DAYS:
            buf_X = np.array(buffer_X)
            buf_y = np.array(buffer_y)

            # Replay dari base data
            replay_idx = np.random.choice(
                len(X_base), size=REPLAY_BASE_SAMPLES, replace=False
            )
            X_combined = np.vstack([buf_X, X_base[replay_idx]])
            y_combined = np.concatenate([buf_y, y_base[replay_idx]])

            # Shuffle
            idx = np.random.permutation(len(X_combined))
            incr_model.fit(
                X_combined[idx], y_combined[idx],
                epochs=FINETUNE_EPOCHS,
                batch_size=BATCH_SIZE,
                verbose=0
            )

            # Soft weight blending (Polyak averaging) — pull the freshly
            # fine-tuned weights back toward the base model's weights rather
            # than accepting them outright. Caps cumulative drift across many
            # fine-tune cycles without any extra training compute.
            finetuned_weights = incr_model.get_weights()
            base_weights      = base_model.get_weights()
            blended_weights = [
                FINETUNE_BLEND_ALPHA * fw + (1 - FINETUNE_BLEND_ALPHA) * bw
                for fw, bw in zip(finetuned_weights, base_weights)
            ]
            incr_model.set_weights(blended_weights)

            finetune_count += 1
            log.info(
                f"  Fine-tune #{finetune_count} setelah hari ke-{i+1} "
                f"({len(buf_X)} baru + {REPLAY_BASE_SAMPLES} replay, "
                f"blend alpha={FINETUNE_BLEND_ALPHA})"
            )
            buffer_X, buffer_y = [], []

    # Simpan model incremental terakhir
    return predictions, finetune_count, incr_model


# =============================================================================
# TAHAP 7: EVALUASI
# =============================================================================

def evaluate(
    predictions_scaled: list,
    y_stream: np.ndarray,
    incr_data,
    real_trade_days,
    scaler: MinMaxScaler,
    komoditas: str,
    feature_cols: list
) -> tuple:
    """
    Denormalisasi dan hitung metrik hanya pada hari perdagangan nyata.
    Return (results_df, rmse, mape, mae)
    """
    n_feat = len(feature_cols)   # pakai jumlah fitur aktual komoditas ini

    def denorm(vals):
        dummy = np.zeros((len(vals), n_feat))
        dummy[:, 0] = vals
        return scaler.inverse_transform(dummy)[:, 0]

    preds   = denorm(np.array(predictions_scaled))
    actuals = denorm(y_stream)

    pred_dates = incr_data.index[:len(actuals)]
    results    = pd.DataFrame(
        {'Aktual': actuals, 'Incr': preds},
        index=pred_dates
    )

    # Evaluasi hanya pada hari perdagangan nyata
    eval_days    = real_trade_days[
        (real_trade_days >= pd.Timestamp(INCR_START)) &
        (real_trade_days <= pred_dates.max())
    ]
    results_eval = results[results.index.isin(eval_days)]

    if results_eval.empty:
        log.warning("Tidak ada hari perdagangan nyata untuk evaluasi")
        return results, None, None, None

    rmse = np.sqrt(mean_squared_error(
        results_eval['Aktual'], results_eval['Incr']
    ))
    mape = mean_absolute_percentage_error(
        results_eval['Aktual'], results_eval['Incr']
    ) * 100
    mae  = np.mean(np.abs(results_eval['Aktual'] - results_eval['Incr']))

    log.info(
        f"Evaluasi [{komoditas}] pada {len(results_eval)} hari perdagangan:\n"
        f"  RMSE : Rp {rmse:,.2f}\n"
        f"  MAE  : Rp {mae:,.2f}\n"
        f"  MAPE : {mape:.2f}%"
    )

    return results, rmse, mape, mae


# =============================================================================
# TAHAP 8: SIMPAN KE DATABASE
# =============================================================================

def save_results(
    db: Database,
    komoditas: str,
    results: pd.DataFrame,
    scaler: MinMaxScaler,
    rmse: float, mape: float, mae: float,
    finetune_count: int,
    feature_cols: list
):
    """Simpan prediksi dan metadata ke database."""
    df_to_save = results[['Aktual', 'Incr']].copy()
    db.upsert_predictions(df_to_save, komoditas)

    # Simpan scaler dan feature_cols bersama dalam satu file
    scaler_path = MODEL_DIR / f'scaler_{komoditas.replace(" ", "_")}.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump({'scaler': scaler, 'feature_cols': feature_cols}, f)

    db.update_model_metadata(
        komoditas     = komoditas,
        last_trained  = date.today().isoformat(),
        training_end  = BASE_END,
        window_size   = WINDOW_SIZE,
        rmse          = float(rmse) if rmse else None,
        mape          = float(mape) if mape else None,
        mae           = float(mae)  if mae  else None,
        n_finetune    = finetune_count,
    )
    log.info(f"Hasil [{komoditas}] disimpan ke database")


# =============================================================================
# PIPELINE PENUH UNTUK SATU KOMODITAS
# =============================================================================

def run_pipeline(
    komoditas: str,
    db: Database,
    force_retrain: bool = False
):
    """
    Jalankan pipeline lengkap untuk satu komoditas:
    update data -> preprocess -> train/load model -> incremental -> simpan
    """
    log.info(f"\n{'='*60}")
    log.info(f"PIPELINE: {komoditas}")
    log.info(f"{'='*60}")

    try:
        # 1. Update harga dari Excel
        update_prices(db, komoditas)

        # 2. Preprocessing (commodity-specific lags & seasonal features)
        df_gabung, real_trade_days, base_data, incr_data, feature_cols = preprocess(
            db, komoditas
        )

        # 3. Normalisasi & sliding windows
        scaler = MinMaxScaler()
        (X_train, y_train, X_val, y_val,
         X_base, y_base, X_stream, y_stream) = prepare_windows(
            base_data, incr_data, scaler
        )

        # 4. Load atau train base model
        base_model, _ = load_or_train_base(
            X_train, y_train, X_val, y_val,
            komoditas, force_retrain
        )

        # 5. Incremental learning
        log.info(f"Incremental learning [{komoditas}]...")
        predictions, finetune_count, incr_model = run_incremental(
            base_model, X_base, y_base, X_stream, y_stream
        )

        # Simpan model incremental
        incr_path = MODEL_DIR / f'incr_{komoditas.replace(" ", "_")}.keras'
        incr_model.save(incr_path)

        # 6. Evaluasi
        results, rmse, mape, mae = evaluate(
            predictions, y_stream, incr_data,
            real_trade_days, scaler, komoditas,
            feature_cols   # pass commodity-specific feature list
        )

        # 7. Simpan ke database
        save_results(
            db, komoditas, results, scaler,
            rmse, mape, mae, finetune_count,
            feature_cols
        )

        log.info(f"Pipeline [{komoditas}] selesai ✓")
        return True

    except Exception as e:
        log.error(f"Pipeline [{komoditas}] GAGAL: {e}", exc_info=True)
        return False


# =============================================================================
# STATUS
# =============================================================================

def print_status(db: Database):
    """Tampilkan status pipeline saat ini."""
    print()
    db.summary()

    print("\nModel files:")
    for komoditas in ALL_KOMODITAS:
        name       = komoditas.replace(' ', '_')
        base_exist = (MODEL_DIR / f'base_{name}.keras').exists()
        incr_exist = (MODEL_DIR / f'incr_{name}.keras').exists()
        meta       = db.get_model_metadata(komoditas)
        trained    = meta['last_trained'] if meta else 'belum'
        print(
            f"  {komoditas:<35} "
            f"base: {'✓' if base_exist else '✗'}  "
            f"incr: {'✓' if incr_exist else '✗'}  "
            f"trained: {trained}"
        )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pipeline LSTM Incremental')
    parser.add_argument(
        '--komoditas', default=DEFAULT_KOMODITAS,
        help=f'Nama komoditas (default: {DEFAULT_KOMODITAS})'
    )
    parser.add_argument(
        '--all', action='store_true',
        help='Jalankan untuk semua komoditas'
    )
    parser.add_argument(
        '--no-retrain', action='store_true',
        help='Skip training ulang base model (pakai yang sudah ada)'
    )
    parser.add_argument(
        '--skip-weather', action='store_true',
        help='Skip update data cuaca'
    )
    parser.add_argument(
        '--status', action='store_true',
        help='Tampilkan status pipeline tanpa menjalankan apapun'
    )
    parser.add_argument(
        '--db', default=DB_PATH,
        help=f'Path database (default: {DB_PATH})'
    )
    parser.add_argument(
        '--weekly', action='store_true',
        help='Latih/perbarui model MINGGUAN (khusus fitur Simulasi Cuaca) '
             'alih-alih pipeline harian biasa'
    )
    args = parser.parse_args()

    with Database(args.db) as db:

        if args.status:
            print_status(db)

        else:
            # Update cuaca dulu (sekali, untuk semua komoditas)
            if not args.skip_weather:
                update_weather(db)

            # Tentukan komoditas yang akan dijalankan
            targets = ALL_KOMODITAS if args.all else [args.komoditas]
            force_retrain = not args.no_retrain

            results_summary = []
            for k in targets:
                if args.weekly:
                    success = run_weekly_pipeline(
                        komoditas=k,
                        db=db,
                        force_retrain=force_retrain
                    )
                else:
                    success = run_pipeline(
                        komoditas=k,
                        db=db,
                        force_retrain=force_retrain
                    )
                results_summary.append((k, '✓' if success else '✗'))

            # Ringkasan akhir
            print(f"\n{'='*60}")
            print(f"RINGKASAN PIPELINE {'MINGGUAN' if args.weekly else ''}")
            print(f"{'='*60}")
            for k, status in results_summary:
                print(f"  {status}  {k}")
            print(f"{'='*60}")

            # run_pipeline() catches its own exceptions per commodity and
            # returns False rather than raising — which previously left the
            # process exiting 0 even when EVERY commodity failed (e.g. prices
            # updated fine in step 1, then preprocessing/training/evaluation
            # threw and predictions/metadata never got saved). Callers like
            # app.py's subprocess only check the exit code, so a silent
            # failure here used to show up as "Data berhasil diperbarui" in
            # the UI even though nothing downstream of prices actually
            # updated. Surface it properly.
            if any(status == '✗' for _, status in results_summary):
                sys.exit(1)