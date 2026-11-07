"""
pihps_scraper.py
----------------
Mengambil data harga pangan dari PIHPS Bank Indonesia via API endpoint resmi.

Endpoint ditemukan oleh: pihpsscraper2 (reverse-engineered dari AJAX call)
URL: https://www.bi.go.id/hargapangan/WebSite/TabelHarga/GetGridDataDaerah

Tidak memerlukan Selenium — murni requests ke API JSON.

SETUP:
  pip install requests pandas

PENGGUNAAN dari kode lain:
  from pihps_scraper import fetch_and_update
  df = fetch_and_update('Cabai Merah')

PENGGUNAAN dari terminal:
  python pihps_scraper.py
  python pihps_scraper.py --komoditas "Cabai Merah"
  python pihps_scraper.py --all
  python pihps_scraper.py --start 2025-01-01 --end 2025-12-31
  python pihps_scraper.py --dry-run
"""

import time
import logging
import requests
import pandas as pd
import numpy as np
from datetime import date, timedelta
from pathlib import Path

log = logging.getLogger(__name__)

# =============================================================================
# KONFIGURASI API
# =============================================================================

BASE_URL = (
    'https://www.bi.go.id/hargapangan/WebSite/TabelHarga/GetGridDataDaerah'
)

HEADERS = {
    'User-Agent'      : (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/149.0.0.0 Safari/537.36'
    ),
    'Accept'          : 'application/json, text/javascript, */*; q=0.01',
    'Accept-Language' : 'id-ID,id;q=0.9,en-US;q=0.8',
    'X-Requested-With': 'XMLHttpRequest',
    'Referer'         : (
        'https://www.bi.go.id/hargapangan/TabelHarga/PasarTradisionalDaerah'
    ),
}

# Parameter tetap (sesuai scraper asli)
# province_id 12   = Jawa Barat
# price_type_id 1  = Harga Konsumen
# tipe_laporan 1   = Harian
# comcat_id        = kategori 10 komoditas:
#   cat_1  = Beras
#   cat_2  = Daging Ayam
#   cat_3  = Daging Sapi
#   cat_4  = Telur Ayam
#   cat_5  = Bawang Merah
#   cat_6  = Bawang Putih
#   cat_7  = Cabai Merah
#   cat_8  = Cabai Rawit
#   cat_9  = Minyak Goreng
#   cat_10 = Gula Pasir
FIXED_PARAMS = {
    'price_type_id': '1',
    'comcat_id'    : 'cat_1,cat_2,cat_3,cat_4,cat_5,cat_6,cat_7,cat_8,cat_9,cat_10',
    'province_id'  : '12',
    'regency_id'   : '',
    'market_id'    : '',
    'tipe_laporan' : '1',
}

# Maks hari per request — jangan terlalu besar agar respons cepat
CHUNK_DAYS  = 30
MAX_RETRIES = 3
RETRY_DELAY = 5   # detik
REQUEST_DELAY = 1 # detik antar request agar tidak dianggap spam

# =============================================================================
# MAPPING NAMA KOMODITAS PIHPS -> DATABASE
# Sesuaikan jika nama di response API berbeda dari nama di database
# =============================================================================

KOMODITAS_MAP = {
    'Beras'       : 'Beras',        # cat_1
    'Daging Ayam' : 'Daging Ayam',  # cat_2
    'Daging Sapi' : 'Daging Sapi',  # cat_3
    'Telur Ayam'  : 'Telur Ayam',   # cat_4
    'Bawang Merah': 'Bawang Merah', # cat_5
    'Bawang Putih': 'Bawang Putih', # cat_6
    'Cabai Merah' : 'Cabai Merah',  # cat_7
    'Cabai Rawit' : 'Cabai Rawit',  # cat_8
    'Minyak Goreng': 'Minyak Goreng', # cat_9
    'Gula Pasir'  : 'Gula Pasir',   # cat_10
}

# Semua nama komoditas yang ada di database
ALL_KOMODITAS = list(KOMODITAS_MAP.values())


# =============================================================================
# HTTP REQUEST HELPER
# =============================================================================

def _get(params: dict) -> list | dict | None:
    """
    GET ke PIHPS API dengan retry otomatis.
    Return parsed JSON atau None jika semua retry gagal.
    """
    session = requests.Session()

    # Hit halaman utama dulu untuk dapat cookies
    try:
        session.get(
            'https://www.bi.go.id/hargapangan/TabelHarga/PasarTradisionalDaerah',
            headers={'User-Agent': HEADERS['User-Agent']},
            timeout=15
        )
    except Exception:
        pass  # Lanjut meski gagal dapat cookie

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(
                BASE_URL,
                params=params,
                headers=HEADERS,
                timeout=30
            )
            if r.status_code == 200:
                return r.json()
            else:
                log.warning(
                    f"HTTP {r.status_code} (attempt {attempt}/{MAX_RETRIES})"
                )
        except requests.exceptions.RequestException as e:
            log.warning(f"Request error (attempt {attempt}/{MAX_RETRIES}): {e}")

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY)

    log.error("Semua retry gagal")
    return None


# =============================================================================
# PARSE RESPONSE
# =============================================================================

def _parse_response(data: list | dict) -> pd.DataFrame:
    """
    Konversi response JSON PIHPS ke DataFrame standar:
        Tanggal | komoditas | Harga (Rp)

    Response PIHPS format wide:
    [
        {'no': 1, 'name': 'Cabai Merah', 'level': '...', '01/06/2026': 55000, '02/06/2026': 56000},
        ...
    ]
    Kolom tanggal berbentuk 'dd/mm/yyyy' sebagai header.
    """
    # Normalisasi ke list
    if isinstance(data, dict):
        rows = data.get('data', data.get('Data', []))
    elif isinstance(data, list):
        rows = data
    else:
        log.warning(f"Format response tidak dikenal: {type(data)}")
        return pd.DataFrame()

    if not rows:
        log.warning("Response kosong")
        return pd.DataFrame()

    sample = rows[0] if rows else {}
    keys   = list(sample.keys())
    log.debug(f"Field names di response: {keys}")

    # Deteksi kolom nama komoditas
    name_field = _find_field(keys, [
        'name', 'nm_komoditi', 'commodity', 'komoditas', 'nama', 'nm_commodity'
    ])

    if not name_field:
        log.error(f"Tidak bisa deteksi field nama. Keys: {keys}")
        return pd.DataFrame()

    # Deteksi kolom tanggal — format dd/mm/yyyy sebagai header
    import re
    date_cols = [
        k for k in keys
        if re.match(r'\d{1,2}/\d{1,2}/\d{4}', str(k))
    ]

    if not date_cols:
        log.error(f"Tidak ada kolom tanggal ditemukan. Keys: {keys}")
        return pd.DataFrame()

    log.debug(f"Kolom tanggal ditemukan: {date_cols}")

    # Pivot dari wide ke long
    records = []
    for row in rows:
        komoditas_raw = str(row.get(name_field, '')).strip()
        if not komoditas_raw:
            continue

        # Map nama ke database
        db_name = KOMODITAS_MAP.get(komoditas_raw)
        if db_name is None:
            # Coba partial match
            db_name = next(
                (v for k, v in KOMODITAS_MAP.items()
                 if k.lower() in komoditas_raw.lower() or
                 komoditas_raw.lower() in k.lower()),
                None
            )
        if db_name is None:
            log.debug(f"Komoditas tidak dikenal: '{komoditas_raw}'")
            continue

        for date_col in date_cols:
            try:
                tanggal = pd.to_datetime(date_col, format='%d/%m/%Y', errors='coerce')
                if pd.isna(tanggal):
                    continue

                harga_raw = row.get(date_col)
                harga = pd.to_numeric(
                    str(harga_raw).replace(',', '').replace('-', '').strip(),
                    errors='coerce'
                )

                if pd.isna(harga) or harga <= 0:
                    continue

                records.append({
                    'Tanggal'   : tanggal,
                    'komoditas' : db_name,
                    'Harga (Rp)': harga,
                })
            except Exception as e:
                log.debug(f"Skip kolom {date_col}: {e}")
                continue

    if not records:
        log.warning("Tidak ada record valid setelah parsing")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df['Tanggal'] = pd.to_datetime(df['Tanggal'])

    # Rata-rata per tanggal+komoditas (ada banyak pasar per hari)
    df = (
        df.groupby(['Tanggal', 'komoditas'])['Harga (Rp)']
        .mean()
        .reset_index()
    )

    log.debug(f"Parsed {len(df)} baris untuk {df['komoditas'].nunique()} komoditas")
    return df


def _find_field(keys: list, candidates: list) -> str | None:
    """Cari nama field dari list kandidat secara case-insensitive."""
    keys_lower = {k.lower(): k for k in keys}
    for c in candidates:
        if c.lower() in keys_lower:
            return keys_lower[c.lower()]
    return None


# =============================================================================
# FETCH SATU RENTANG TANGGAL (SEMUA KOMODITAS SEKALIGUS)
# =============================================================================

def _fetch_chunk(start: date, end: date) -> pd.DataFrame:
    """
    Fetch satu chunk tanggal — semua komoditas dalam satu request.
    Return DataFrame dengan kolom: Tanggal, komoditas, Harga (Rp)
    """
    params = {
        **FIXED_PARAMS,
        'start_date': start.strftime('%Y-%m-%d'),
        'end_date'  : end.strftime('%Y-%m-%d'),
    }

    log.info(f"  Fetching {start} s/d {end}...")
    data = _get(params)

    if data is None:
        return pd.DataFrame()

    df = _parse_response(data)
    log.info(
        f"  → {len(df)} baris | "
        f"{df['komoditas'].nunique() if not df.empty else 0} komoditas"
    )
    return df


# =============================================================================
# FUNGSI UTAMA — fetch satu komoditas
# =============================================================================

def fetch_and_update(
    komoditas: str,
    start_date: date      = date(2021, 1, 1),
    end_date: date        = None,
    last_date: date       = None,
) -> pd.DataFrame:
    """
    Fetch data harga untuk satu komoditas dari PIHPS.
    Dipanggil oleh pipeline.py untuk setiap komoditas.

    Args:
        komoditas  : nama komoditas sesuai database
        start_date : tanggal mulai jika data kosong
        end_date   : tanggal akhir (default: kemarin)
        last_date  : tanggal terakhir yang sudah ada di database
                     (jika ada, fetch hanya dari last_date+1)

    Return:
        pd.DataFrame — index Tanggal, kolom Harga (Rp)
        Kosong jika tidak ada data baru.
    """
    if end_date is None:
        end_date = date.today() - timedelta(days=1)

    fetch_start = (last_date + timedelta(days=1)) if last_date else start_date

    if fetch_start > end_date:
        log.info(f"[{komoditas}] Sudah up-to-date (s/d {last_date})")
        return pd.DataFrame()

    log.info(f"[{komoditas}] Fetch: {fetch_start} s/d {end_date}")

    all_frames = []
    cursor = fetch_start

    while cursor <= end_date:
        chunk_end = min(cursor + timedelta(days=CHUNK_DAYS - 1), end_date)

        df_chunk = _fetch_chunk(cursor, chunk_end)

        if not df_chunk.empty:
            # Filter hanya komoditas yang diminta
            df_k = df_chunk[df_chunk['komoditas'] == komoditas].copy()
            if not df_k.empty:
                df_k = (
                    df_k[['Tanggal', 'Harga (Rp)']]
                    .set_index('Tanggal')
                    .sort_index()
                )
                all_frames.append(df_k)

        cursor = chunk_end + timedelta(days=1)
        if cursor <= end_date:
            time.sleep(REQUEST_DELAY)

    if not all_frames:
        log.warning(f"[{komoditas}] Tidak ada data baru")
        return pd.DataFrame()

    df_result = pd.concat(all_frames)
    df_result = df_result[~df_result.index.duplicated(keep='last')].sort_index()
    log.info(f"[{komoditas}] Total: {len(df_result)} baris baru")
    return df_result


# =============================================================================
# FETCH SEMUA KOMODITAS SEKALIGUS (lebih efisien — 1 request per chunk)
# =============================================================================

def fetch_all_komoditas(
    start_date: date = date(2021, 1, 1),
    end_date: date   = None,
    last_dates: dict = None,
) -> dict[str, pd.DataFrame]:
    """
    Fetch semua komoditas dalam satu set request.
    Jauh lebih efisien daripada memanggil fetch_and_update per komoditas.

    Args:
        start_date : tanggal mulai default
        end_date   : tanggal akhir (default: kemarin)
        last_dates : dict {komoditas: last_date} dari database
                     untuk incremental update per komoditas

    Return:
        dict {komoditas: DataFrame}
    """
    if end_date is None:
        end_date = date.today() - timedelta(days=1)
    if last_dates is None:
        last_dates = {}

    # Tentukan fetch_start global (paling awal dari semua komoditas)
    fetch_start = end_date
    for k in ALL_KOMODITAS:
        last = last_dates.get(k)
        s    = (last + timedelta(days=1)) if last else start_date
        if s < fetch_start:
            fetch_start = s

    if fetch_start > end_date:
        log.info("Semua komoditas sudah up-to-date")
        return {}

    log.info(f"Fetch semua komoditas: {fetch_start} s/d {end_date}")

    # Kumpulkan semua data dalam satu set chunk requests
    all_rows = []
    cursor = fetch_start
    while cursor <= end_date:
        chunk_end = min(cursor + timedelta(days=CHUNK_DAYS - 1), end_date)
        df_chunk  = _fetch_chunk(cursor, chunk_end)
        if not df_chunk.empty:
            all_rows.append(df_chunk)
        cursor = chunk_end + timedelta(days=1)
        if cursor <= end_date:
            time.sleep(REQUEST_DELAY)

    if not all_rows:
        log.warning("Tidak ada data yang berhasil diambil")
        return {}

    df_all = pd.concat(all_rows)
    df_all = df_all[~df_all.duplicated(['Tanggal', 'komoditas'], keep='last')]

    # Split per komoditas dan filter hanya baris baru
    results = {}
    for komoditas in ALL_KOMODITAS:
        df_k = df_all[df_all['komoditas'] == komoditas].copy()
        if df_k.empty:
            continue

        last = last_dates.get(komoditas)
        if last:
            df_k = df_k[df_k['Tanggal'] > pd.Timestamp(last)]

        if df_k.empty:
            continue

        df_k = (
            df_k[['Tanggal', 'Harga (Rp)']]
            .set_index('Tanggal')
            .sort_index()
        )
        results[komoditas] = df_k
        log.info(f"  {komoditas:<35} {len(df_k):>4} baris baru")

    return results


# =============================================================================
# JALANKAN LANGSUNG
# =============================================================================

if __name__ == '__main__':
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )

    parser = argparse.ArgumentParser(description='PIHPS BI API Scraper')
    parser.add_argument('--komoditas', default=None,
                        help='Nama komoditas spesifik')
    parser.add_argument('--all',    action='store_true',
                        help='Fetch semua komoditas sekaligus')
    parser.add_argument('--start',  default='2021-01-01',
                        help='Tanggal mulai (YYYY-MM-DD)')
    parser.add_argument('--end',    default=None,
                        help='Tanggal akhir (YYYY-MM-DD), default: kemarin')
    parser.add_argument('--dry-run', action='store_true',
                        help='Cek koneksi API tanpa simpan data')
    parser.add_argument('--comcat', default=None,
                        help='Override comcat_id, misal "cat_1" untuk test satu kategori saja')
    args = parser.parse_args()

    if args.comcat:
        FIXED_PARAMS['comcat_id'] = args.comcat

    start = date.fromisoformat(args.start)
    end   = date.fromisoformat(args.end) if args.end \
            else date.today() - timedelta(days=1)

    if args.dry_run:
        # Test koneksi dengan 3 hari data saja
        test_end = min(start + timedelta(days=2), end)
        print(f"Dry run: test fetch {start} s/d {test_end}...")

        # Also show raw response to debug field names
        params = {
            **FIXED_PARAMS,
            'start_date': start.strftime('%Y-%m-%d'),
            'end_date'  : test_end.strftime('%Y-%m-%d'),
        }
        raw = _get(params)
        print(f"\nRaw response — type: {type(raw)}")
        if raw is None:
            print("Raw response is None — request/HTTP gagal (lihat log WARNING/ERROR di atas).")
        else:
            rows = raw if isinstance(raw, list) else raw.get('data', raw.get('Data', []))
            print(f"Raw response — jumlah baris: {len(rows) if hasattr(rows, '__len__') else 'N/A'}")
            if not isinstance(raw, list):
                print(f"Raw response — top-level keys: {list(raw.keys())}")
            if rows:
                print(f"Raw response — keys per baris: {list(rows[0].keys())}")
                print(f"Raw response — first 2 rows:")
                import json as _json
                print(_json.dumps(rows[:2], indent=2, ensure_ascii=False))
            else:
                print("Raw response — rows kosong. Isi raw mentah (dipotong):")
                import json as _json
                raw_str = _json.dumps(raw, indent=2, ensure_ascii=False)
                print(raw_str[:2000])

        df_test = _fetch_chunk(start, test_end)
        if df_test.empty:
            print("\n❌ Parsing gagal — lihat raw response di atas untuk debug mapping")
        else:
            print(f"\n✓ Berhasil! {len(df_test)} baris, "
                  f"{df_test['komoditas'].nunique()} komoditas")
            print(f"\nKomoditas yang berhasil di-map:")
            for k in sorted(df_test['komoditas'].unique()):
                n = len(df_test[df_test['komoditas'] == k])
                print(f"  ✓ {k:<35} ({n} baris)")

    elif args.all:
        results = fetch_all_komoditas(start_date=start, end_date=end)
        print(f"\nHasil fetch semua komoditas:")
        for k, df in results.items():
            print(f"  {k:<35} {len(df):>4} baris")

    else:
        komoditas = args.komoditas or 'Cabai Merah'
        df = fetch_and_update(komoditas, start_date=start, end_date=end)
        if df.empty:
            print(f"Tidak ada data untuk {komoditas}")
        else:
            print(f"\n{komoditas}: {len(df)} baris")
            print(df.head(10).to_string())