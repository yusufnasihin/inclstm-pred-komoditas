"""
weather_fetcher.py
------------------
Mengambil data cuaca otomatis dari Open-Meteo API untuk wilayah Jawa Barat.

Fitur utama:
  - Satu API call per update (bukan satu call per kota)
  - Grid 4x4 = 16 titik tersebar merata di seluruh Jawa Barat
  - Rata-rata spasial dari semua titik grid -> 1 nilai per hari per variabel
  - Incremental: hanya download hari yang belum ada
  - Otomatis pilih endpoint archive vs recent sesuai tanggal
  - Cache JSON lokal agar tidak re-download jika script dijalankan ulang

SETUP:
  pip install requests pandas openpyxl

PENGGUNAAN dari kode lain:
  from weather_fetcher import fetch_and_update
  df_cuaca = fetch_and_update()          # update ke hari ini
  df_cuaca = fetch_and_update(force=True) # download ulang semua

PENGGUNAAN dari terminal:
  python weather_fetcher.py              # incremental update
  python weather_fetcher.py --start 2021-01-01
  python weather_fetcher.py --force
  python weather_fetcher.py --dry-run
"""

import os
import json
import time
import logging
import requests
import numpy as np
import pandas as pd
from datetime import date, timedelta
from pathlib import Path

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

# Bounding box Jawa Barat
# Ubah nilai ini jika wilayah studimu berbeda
BBOX = {
    'lat_min': -8.5,    # selatan
    'lat_max': -5.8,    # utara
    'lon_min': 105.1,   # barat
    'lon_max': 109.3,   # timur
}

# Ukuran grid — 4x4 = 16 titik, cukup representatif untuk Jawa Barat
# Naikkan ke 5 atau 6 jika butuh resolusi lebih tinggi (max 1000 titik)
GRID_SIZE = 4

# Variabel harian yang diminta dari Open-Meteo
DAILY_VARIABLES = [
    'temperature_2m_mean',        # Suhu rata-rata (°C)
    'relative_humidity_2m_mean',  # Kelembapan relatif (%)
    'precipitation_sum',          # Total curah hujan (mm)
]

# Timezone
TIMEZONE = 'Asia/Jakarta'

# Output & cache
OUTPUT_CSV = 'data_cuaca_openmeteo.csv'
CACHE_DIR  = Path('weather_cache')
CACHE_DIR.mkdir(exist_ok=True)

# Open-Meteo archive tersedia hingga ~5 hari lalu
ARCHIVE_LAG_DAYS = 5

# Request settings
MAX_RETRIES  = 3
RETRY_DELAY  = 5    # detik antar retry
# Maksimum hari per satu request (hindari response terlalu besar)
# Open-Meteo menghitung >14 hari sebagai multiple calls,
# tapi tetap diizinkan — kita batasi 90 hari per chunk untuk keamanan
CHUNK_DAYS   = 90


# =============================================================================
# GENERATE GRID KOORDINAT
# =============================================================================

def _build_grid() -> tuple[list[float], list[float]]:
    """
    Buat grid merata GRID_SIZE x GRID_SIZE di dalam BBOX.
    Return (lat_list, lon_list) — panjang keduanya = GRID_SIZE²
    """
    lats = np.linspace(BBOX['lat_max'], BBOX['lat_min'], GRID_SIZE)
    lons = np.linspace(BBOX['lon_min'], BBOX['lon_max'], GRID_SIZE)
    grid_lons, grid_lats = np.meshgrid(lons, lats)
    lat_list = grid_lats.flatten().round(4).tolist()
    lon_list = grid_lons.flatten().round(4).tolist()
    return lat_list, lon_list


# =============================================================================
# HTTP REQUEST HELPER
# =============================================================================

def _get(url: str, params: dict) -> list | dict:
    """GET dengan retry otomatis. Return parsed JSON."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            log.warning(f"Request gagal (attempt {attempt}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
            else:
                raise


# =============================================================================
# FETCH SATU CHUNK TANGGAL (SEMUA GRID DALAM SATU CALL)
# =============================================================================

def _fetch_chunk(
    lat_list: list[float],
    lon_list: list[float],
    start: date,
    end: date,
    use_archive: bool
) -> pd.DataFrame:
    """
    Fetch satu rentang tanggal untuk semua titik grid dalam SATU API call.
    Return DataFrame harian dengan kolom Suhu, Kelembapan, CurahHujan
    (sudah dirata-rata dari semua titik grid).
    """
    endpoint = (
        'https://archive-api.open-meteo.com/v1/archive'
        if use_archive else
        'https://api.open-meteo.com/v1/forecast'
    )

    params = {
        'latitude'  : ','.join(str(x) for x in lat_list),
        'longitude' : ','.join(str(x) for x in lon_list),
        'daily'     : ','.join(DAILY_VARIABLES),
        'timezone'  : TIMEZONE,
    }

    if use_archive:
        params['start_date'] = start.isoformat()
        params['end_date']   = end.isoformat()
    else:
        # Forecast API: gunakan past_days
        past_days = (date.today() - start).days
        params['past_days']     = min(past_days, 92)
        params['forecast_days'] = 1

    log.info(
        f"  Fetching {'archive' if use_archive else 'recent'}: "
        f"{start} s/d {end} | {len(lat_list)} titik grid | endpoint: {endpoint}"
    )

    raw = _get(endpoint, params)

    # Response bisa berupa list (multi-lokasi) atau dict (single lokasi)
    if isinstance(raw, dict):
        raw = [raw]

    return _parse_multi_location(raw, start, end)


def _parse_multi_location(
    responses: list[dict],
    start: date,
    end: date
) -> pd.DataFrame:
    """
    Parse array response Open-Meteo (satu dict per lokasi grid).
    Rata-rata semua lokasi -> DataFrame harian.
    """
    all_dfs = []

    for loc_data in responses:
        daily = loc_data.get('daily', {})
        if not daily or 'time' not in daily:
            continue

        n = len(daily['time'])
        df = pd.DataFrame({
            'Tanggal'          : pd.to_datetime(daily['time']),
            'Suhu (C)'         : daily.get('temperature_2m_mean',       [np.nan] * n),
            'Kelembapan (%)'   : daily.get('relative_humidity_2m_mean', [np.nan] * n),
            'Curah Hujan (mm)' : daily.get('precipitation_sum',         [np.nan] * n),
        }).set_index('Tanggal')

        all_dfs.append(df)

    if not all_dfs:
        log.warning("Tidak ada data yang bisa di-parse dari response.")
        return pd.DataFrame()

    # Stack semua lokasi dan ambil rata-rata per hari
    stacked  = pd.concat(all_dfs)
    df_mean  = stacked.groupby(stacked.index).mean()
    df_mean.index.name = 'Tanggal'

    # Trim ke range yang diminta
    df_mean = df_mean[
        (df_mean.index >= pd.Timestamp(start)) &
        (df_mean.index <= pd.Timestamp(end))
    ]

    log.info(
        f"  Parsed {len(all_dfs)} lokasi -> "
        f"{len(df_mean)} hari ({df_mean.index.min().date()} s/d {df_mean.index.max().date()})"
    )
    return df_mean


# =============================================================================
# FETCH RENTANG PENUH (DENGAN CHUNKING & CACHE)
# =============================================================================

def _fetch_range(start: date, end: date) -> pd.DataFrame:
    """
    Fetch semua data antara start–end.
    Otomatis split archive vs recent, dan chunk per CHUNK_DAYS hari.
    Hasil tiap chunk di-cache sebagai JSON.
    """
    lat_list, lon_list = _build_grid()
    log.info(
        f"Grid: {GRID_SIZE}x{GRID_SIZE} = {len(lat_list)} titik | "
        f"Bbox: lat [{BBOX['lat_min']}, {BBOX['lat_max']}] "
        f"lon [{BBOX['lon_min']}, {BBOX['lon_max']}]"
    )

    archive_cutoff = date.today() - timedelta(days=ARCHIVE_LAG_DAYS)
    frames = []

    # Bagi range menjadi chunk-chunk
    # PENTING: satu chunk tidak boleh melewati archive_cutoff, karena archive
    # endpoint tidak punya data ~5 hari terakhir dan recent endpoint tidak
    # selalu punya data jauh ke belakang. Jika dibiarkan straddling, hari-hari
    # terbaru bisa hilang dan hanya keisi hasil interpolasi.
    cursor = start
    while cursor <= end:
        use_archive = cursor <= archive_cutoff
        if use_archive:
            chunk_end = min(cursor + timedelta(days=CHUNK_DAYS - 1), end, archive_cutoff)
        else:
            chunk_end = min(cursor + timedelta(days=CHUNK_DAYS - 1), end)

        # Cek cache
        cache_key  = f"{cursor}_{chunk_end}_{'arc' if use_archive else 'rec'}"
        cache_file = CACHE_DIR / f"cache_{cache_key}.json"

        if cache_file.exists():
            log.info(f"  Cache hit: {cursor} s/d {chunk_end}")
            with open(cache_file) as f:
                cached = json.load(f)
            df_chunk = pd.DataFrame(cached)
            df_chunk.index = pd.to_datetime(df_chunk.index)
            df_chunk.index.name = 'Tanggal'
        else:
            df_chunk = _fetch_chunk(
                lat_list, lon_list,
                start=cursor,
                end=chunk_end,
                use_archive=use_archive
            )
            if not df_chunk.empty:
                # Simpan ke cache
                with open(cache_file, 'w') as f:
                    # Create a temporary copy to convert Timestamp index to strings
                    cache_df = df_chunk.copy()
                    cache_df.index = cache_df.index.astype(str)
                    json.dump(cache_df.to_dict(), f)

        if not df_chunk.empty:
            frames.append(df_chunk)

        cursor = chunk_end + timedelta(days=1)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames)
    combined = combined[~combined.index.duplicated(keep='last')].sort_index()
    return combined


# =============================================================================
# LOAD / VALIDATE / SAVE
# =============================================================================

def load_existing(csv_path: str = OUTPUT_CSV) -> pd.DataFrame:
    """Baca CSV existing. Return DataFrame kosong jika belum ada."""
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, parse_dates=['Tanggal'], index_col='Tanggal')
        log.info(
            f"Data existing: {len(df)} baris "
            f"({df.index.min().date()} s/d {df.index.max().date()})"
        )
        return df
    log.info("Belum ada data existing — akan download dari awal.")
    return pd.DataFrame()


def _validate(df: pd.DataFrame):
    """Cek anomali dasar sebelum simpan."""
    issues = []
    for col, pct in (df.isna().mean() * 100).items():
        if pct > 5:
            issues.append(f"  {col}: {pct:.1f}% NaN")
    if (df['Suhu (C)'] < 15).any() or (df['Suhu (C)'] > 42).any():
        issues.append("  Suhu ada yang di luar range 15–42°C")
    if (df['Kelembapan (%)'] < 20).any() or (df['Kelembapan (%)'] > 100).any():
        issues.append("  Kelembapan ada yang di luar range 20–100%")
    if (df['Curah Hujan (mm)'] < 0).any():
        issues.append("  Curah hujan ada nilai negatif")
    if issues:
        log.warning("Peringatan validasi:\n" + "\n".join(issues))
    else:
        log.info("Validasi: OK")


# =============================================================================
# FUNGSI UTAMA
# =============================================================================

def fetch_and_update(
    start_date: date = date(2021, 1, 1),
    end_date: date   = None,
    force: bool      = False,
    csv_path: str    = OUTPUT_CSV,
    save_csv: bool   = True,
) -> pd.DataFrame:
    """
    Update data cuaca secara incremental. Panggil ini dari pipeline.py / app.py.

    Args:
        start_date : tanggal mulai jika download dari awal
        end_date   : tanggal akhir (default: hari ini)
        force      : True = abaikan existing & cache, download ulang semua
        csv_path   : path file CSV output
        save_csv   : True = simpan juga ke csv_path. Set False jika caller
                     (mis. pipeline2.py) sudah menyimpan hasilnya sendiri ke
                     database, agar tidak ada dua sumber data yang bisa drift.

    Return:
        pd.DataFrame — kolom: Suhu (C), Kelembapan (%), Curah Hujan (mm)
    """
    if end_date is None:
        end_date = date.today()

    # Hapus cache jika force
    if force and CACHE_DIR.exists():
        for f in CACHE_DIR.glob('cache_*.json'):
            f.unlink()
        log.info("Cache dihapus (--force)")

    existing = pd.DataFrame() if force else load_existing(csv_path)

    # Tentukan dari mana mulai fetch
    if not existing.empty:
        fetch_start = existing.index.max().date() + timedelta(days=1)
        if fetch_start > end_date:
            log.info("Data sudah up-to-date, tidak perlu fetch.")
            return existing
        log.info(f"Incremental update: {fetch_start} s/d {end_date}")
    else:
        fetch_start = start_date
        log.info(f"Download penuh: {fetch_start} s/d {end_date}")

    # Fetch
    new_data = _fetch_range(start=fetch_start, end=end_date)

    if new_data.empty:
        log.warning("Tidak ada data baru — kembalikan data existing.")
        return existing

    # Gabung dengan existing
    if not existing.empty:
        combined = pd.concat([existing, new_data])
        combined = combined[~combined.index.duplicated(keep='last')].sort_index()
    else:
        combined = new_data.sort_index()

    # Isi gap kecil (hari yang missing di response)
    combined = combined.resample('D').asfreq()
    combined.interpolate(method='linear', inplace=True)
    combined.bfill(inplace=True)

    _validate(combined)

    if save_csv:
        combined.to_csv(csv_path)
        log.info(
            f"Disimpan: {csv_path} | "
            f"{len(combined)} baris | "
            f"{combined.index.min().date()} s/d {combined.index.max().date()}"
        )
    return combined


def print_summary(df: pd.DataFrame):
    print("\n" + "=" * 55)
    print("RINGKASAN DATA CUACA (Open-Meteo)")
    print(f"Grid: {GRID_SIZE}×{GRID_SIZE} = {GRID_SIZE**2} titik | Jawa Barat")
    print("=" * 55)
    print(f"  Rentang    : {df.index.min().date()} s/d {df.index.max().date()}")
    print(f"  Total hari : {len(df)}")
    print(f"  NaN        : {df.isna().sum().to_dict()}")
    print()
    print(df.describe().round(2).to_string())
    print("=" * 55)


# =============================================================================
# JALANKAN LANGSUNG
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Open-Meteo Weather Fetcher — Jawa Barat Grid')
    parser.add_argument('--start',   default='2021-01-01',
                        help='Tanggal mulai (YYYY-MM-DD)')
    parser.add_argument('--end',     default=None,
                        help='Tanggal akhir (YYYY-MM-DD), default: hari ini')
    parser.add_argument('--force',   action='store_true',
                        help='Hapus cache dan download ulang semua')
    parser.add_argument('--dry-run', action='store_true',
                        help='Cek saja apa yang perlu diupdate tanpa fetch')
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end   = date.fromisoformat(args.end) if args.end else date.today()

    if args.dry_run:
        existing = load_existing()
        lat_list, lon_list = _build_grid()
        print(f"\nGrid       : {GRID_SIZE}×{GRID_SIZE} = {len(lat_list)} titik")
        print(f"Bbox       : lat [{BBOX['lat_min']}, {BBOX['lat_max']}] "
              f"lon [{BBOX['lon_min']}, {BBOX['lon_max']}]")
        if existing.empty:
            days = (end - start).days + 1
            chunks = -(-days // CHUNK_DAYS)   # ceiling division
            print(f"Status     : belum ada data")
            print(f"Perlu fetch: {start} s/d {end} ({days} hari, ~{chunks} API call)")
        else:
            last        = existing.index.max().date()
            fetch_from  = last + timedelta(days=1)
            days_missing = (end - fetch_from).days + 1
            chunks = max(1, -(-days_missing // CHUNK_DAYS))
            print(f"Data s/d   : {last}")
            print(f"Perlu fetch: {fetch_from} s/d {end} ({days_missing} hari, ~{chunks} API call)")
    else:
        df = fetch_and_update(
            start_date=start,
            end_date=end,
            force=args.force
        )
        print_summary(df)
