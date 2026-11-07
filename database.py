"""
database.py
-----------
Modul tunggal untuk semua operasi database SQLite.
Dipanggil oleh pipeline.py, weather_fetcher.py, pihps_scraper.py, dan app.py.

Skema tabel:
  - weather        : data cuaca harian (Open-Meteo)
  - prices         : data harga harian per komoditas (PIHPS / Excel)
  - predictions    : hasil prediksi incremental LSTM per komoditas
  - model_metadata : info training terakhir per komoditas

PENGGUNAAN:
  from database import Database

  db = Database()                   # buka/buat pihps.db
  db = Database('custom_path.db')   # path custom

  # Cuaca
  db.upsert_weather(df)
  df = db.get_weather()
  df = db.get_weather('2024-01-01', '2024-12-31')
  last = db.get_last_weather_date()

  # Harga
  db.upsert_prices(df, 'Cabai Merah')
  df = db.get_prices('Cabai Merah')
  df = db.get_prices('Cabai Merah', '2024-01-01', '2024-12-31')
  last = db.get_last_price_date('Cabai Merah')
  commodities = db.get_commodities()

  # Prediksi
  db.upsert_predictions(df, 'Cabai Merah')
  df = db.get_predictions('Cabai Merah')
  last = db.get_last_prediction_date('Cabai Merah')

  # Metadata model
  db.update_model_metadata('Cabai Merah', rmse=2500, mape=3.2, mae=1800, ...)
  meta = db.get_model_metadata('Cabai Merah')

  db.close()

TIDAK PERLU install apapun — sqlite3 sudah built-in di Python.
"""

import sqlite3
import logging
import threading
import functools
import pandas as pd
from datetime import date, datetime
from pathlib import Path
from contextlib import contextmanager
from typing import Optional

log = logging.getLogger(__name__)


def _locked(method):
    """
    Serialize access to the shared sqlite3 connection.

    When Database is used behind Streamlit's @st.cache_resource, ONE
    Database instance (and its ONE sqlite3.Connection) is shared across
    every session/rerun/concurrent request on the server.
    check_same_thread=False only disables Python's own guard that would
    otherwise raise an explicit error for cross-thread use — it does NOT
    make the connection safe for genuinely simultaneous access from
    multiple threads. Under real concurrency (e.g. two browser tabs/
    reruns hitting the app at the same moment), that can corrupt the
    connection's internal state and surface as confusing, low-level
    errors (e.g. "pop from empty list" from deep inside sqlite3/pandas
    internals) that have nothing to do with the query itself.

    This decorator forces every public method to run one at a time.
    """
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return wrapper

# =============================================================================
# SCHEMA SQL
# =============================================================================

_SCHEMA = """
-- Data cuaca harian dari Open-Meteo
CREATE TABLE IF NOT EXISTS weather (
    tanggal          TEXT PRIMARY KEY,   -- format: YYYY-MM-DD
    suhu_c           REAL,               -- Suhu rata-rata (°C)
    kelembapan_pct   REAL,               -- Kelembapan relatif (%)
    curah_hujan_mm   REAL,               -- Total curah hujan (mm)
    updated_at       TEXT
);

-- Data harga harian per komoditas dari PIHPS / Excel
CREATE TABLE IF NOT EXISTS prices (
    tanggal          TEXT NOT NULL,      -- format: YYYY-MM-DD
    komoditas        TEXT NOT NULL,
    harga_rp         REAL,               -- harga dalam ribuan rupiah
    sumber           TEXT DEFAULT 'excel', -- 'excel' / 'pihps' / 'manual'
    updated_at       TEXT,
    PRIMARY KEY (tanggal, komoditas)
);

-- Hasil prediksi incremental LSTM per komoditas
CREATE TABLE IF NOT EXISTS predictions (
    tanggal          TEXT NOT NULL,
    komoditas        TEXT NOT NULL,
    harga_aktual     REAL,               -- harga nyata (null jika belum diketahui)
    pred_incremental REAL,               -- prediksi incremental LSTM
    updated_at       TEXT,
    PRIMARY KEY (tanggal, komoditas)
);

-- Metadata model terakhir per komoditas
CREATE TABLE IF NOT EXISTS model_metadata (
    komoditas        TEXT PRIMARY KEY,
    last_trained     TEXT,               -- tanggal training terakhir
    training_end     TEXT,               -- data s/d tanggal berapa
    window_size      INTEGER,
    rmse             REAL,
    mape             REAL,
    mae              REAL,
    n_finetune       INTEGER,            -- berapa kali fine-tune dilakukan
    updated_at       TEXT
);

-- Index untuk query cepat
CREATE INDEX IF NOT EXISTS idx_prices_komoditas
    ON prices (komoditas, tanggal);
CREATE INDEX IF NOT EXISTS idx_predictions_komoditas
    ON predictions (komoditas, tanggal);
"""


# =============================================================================
# KELAS DATABASE
# =============================================================================

class Database:
    """
    Handler SQLite untuk seluruh pipeline.
    Gunakan sebagai context manager untuk auto-close:

        with Database() as db:
            df = db.get_weather()
    """

    def __init__(self, path: str = 'pihps.db'):
        self.path = path
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.Lock()
        self._connect()
        self._init_schema()
        log.info(f"Database siap: {Path(path).resolve()}")

    # ------------------------------------------------------------------
    # Koneksi
    # ------------------------------------------------------------------

    def _connect(self):
        self._conn = sqlite3.connect(
            self.path,
            check_same_thread=False,
            timeout=10
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute('PRAGMA journal_mode=WAL')
        self._conn.execute('PRAGMA foreign_keys=ON')

    def _init_schema(self):
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    @contextmanager
    def _transaction(self):
        """Context manager untuk transaksi atomik."""
        try:
            yield self._conn
            self._conn.commit()
        except Exception as e:
            self._conn.rollback()
            log.error(f"Transaksi gagal, rollback: {e}")
            raise

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    @staticmethod
    def _now() -> str:
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    @staticmethod
    def _to_date_str(d) -> str:
        if isinstance(d, str):
            return d[:10]
        if isinstance(d, (date, datetime, pd.Timestamp)):
            return d.strftime('%Y-%m-%d')
        return str(d)[:10]

    # ------------------------------------------------------------------
    # WEATHER
    # ------------------------------------------------------------------

    @_locked
    def upsert_weather(self, df: pd.DataFrame):
        """
        Simpan/update data cuaca.
        df harus punya index Tanggal dan kolom:
            Suhu (C), Kelembapan (%), Curah Hujan (mm)
        """
        if df.empty:
            return

        now  = self._now()
        rows = [
            (
                self._to_date_str(tanggal),
                _safe_float(row.get('Suhu (C)')),
                _safe_float(row.get('Kelembapan (%)')),
                _safe_float(row.get('Curah Hujan (mm)')),
                now
            )
            for tanggal, row in df.iterrows()
        ]

        with self._transaction():
            self._conn.executemany(
                """INSERT INTO weather
                   (tanggal, suhu_c, kelembapan_pct, curah_hujan_mm, updated_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(tanggal) DO UPDATE SET
                       suhu_c         = excluded.suhu_c,
                       kelembapan_pct = excluded.kelembapan_pct,
                       curah_hujan_mm = excluded.curah_hujan_mm,
                       updated_at     = excluded.updated_at
                """,
                rows
            )
        log.info(f"Weather: {len(rows)} baris di-upsert")

    @_locked
    def get_weather(
        self,
        start: Optional[str] = None,
        end: Optional[str]   = None
    ) -> pd.DataFrame:
        """Ambil data cuaca. Return DataFrame dengan index Tanggal."""
        query, params = "SELECT * FROM weather", []
        if start and end:
            query += " WHERE tanggal BETWEEN ? AND ?"
            params = [start, end]
        elif start:
            query += " WHERE tanggal >= ?"
            params = [start]
        elif end:
            query += " WHERE tanggal <= ?"
            params = [end]
        query += " ORDER BY tanggal"

        df = pd.read_sql_query(query, self._conn, params=params,
                               parse_dates=['tanggal'])
        if df.empty:
            return df

        df = (df.set_index('tanggal')
                .rename(columns={
                    'suhu_c'         : 'Suhu (C)',
                    'kelembapan_pct' : 'Kelembapan (%)',
                    'curah_hujan_mm' : 'Curah Hujan (mm)',
                }))
        df.index.name = 'Tanggal'
        return df[['Suhu (C)', 'Kelembapan (%)', 'Curah Hujan (mm)']]

    @_locked
    def get_last_weather_date(self) -> Optional[date]:
        """Tanggal terbaru di tabel weather. None jika kosong."""
        row = self._conn.execute(
            "SELECT MAX(tanggal) as last FROM weather"
        ).fetchone()
        return date.fromisoformat(row['last']) if row and row['last'] else None

    @_locked
    def get_weather_count(self) -> int:
        return self._conn.execute(
            "SELECT COUNT(*) FROM weather"
        ).fetchone()[0]

    # ------------------------------------------------------------------
    # PRICES
    # ------------------------------------------------------------------

    @_locked
    def upsert_prices(
        self,
        df: pd.DataFrame,
        komoditas: str,
        sumber: str = 'excel'
    ):
        """
        Simpan/update data harga untuk satu komoditas.
        df harus punya index Tanggal dan kolom harga
        (nama kolom: 'Harga', 'Harga (Rp)', atau 'harga_rp').
        """
        if df.empty:
            return

        # Cari kolom harga secara fleksibel
        harga_col = next(
            (c for c in ['Harga', 'Harga (Rp)', 'harga_rp', 'harga']
             if c in df.columns),
            df.columns[0]
        )

        now  = self._now()
        rows = [
            (
                self._to_date_str(tanggal),
                komoditas,
                _safe_float(row[harga_col]),
                sumber,
                now
            )
            for tanggal, row in df.iterrows()
        ]

        with self._transaction():
            self._conn.executemany(
                """INSERT INTO prices
                   (tanggal, komoditas, harga_rp, sumber, updated_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(tanggal, komoditas) DO UPDATE SET
                       harga_rp   = excluded.harga_rp,
                       sumber     = excluded.sumber,
                       updated_at = excluded.updated_at
                """,
                rows
            )
        log.info(f"Prices [{komoditas}]: {len(rows)} baris di-upsert (sumber: {sumber})")

    @_locked
    def get_prices(
        self,
        komoditas: str,
        start: Optional[str] = None,
        end: Optional[str]   = None
    ) -> pd.DataFrame:
        """Ambil harga satu komoditas. Return DataFrame index Tanggal, kolom Harga (Rp)."""
        query  = "SELECT tanggal, harga_rp FROM prices WHERE komoditas = ?"
        params = [komoditas]
        if start:
            query += " AND tanggal >= ?"
            params.append(start)
        if end:
            query += " AND tanggal <= ?"
            params.append(end)
        query += " ORDER BY tanggal"

        df = pd.read_sql_query(query, self._conn, params=params,
                               parse_dates=['tanggal'])
        if df.empty:
            return df

        df = df.set_index('tanggal').rename(columns={'harga_rp': 'Harga (Rp)'})
        df.index.name = 'Tanggal'
        return df

    @_locked
    def get_last_price_date(self, komoditas: str) -> Optional[date]:
        """Tanggal terbaru di tabel prices untuk komoditas tertentu."""
        row = self._conn.execute(
            "SELECT MAX(tanggal) as last FROM prices WHERE komoditas = ?",
            (komoditas,)
        ).fetchone()
        return date.fromisoformat(row['last']) if row and row['last'] else None

    @_locked
    def get_commodities(self) -> list[str]:
        """List semua komoditas yang ada di database."""
        rows = self._conn.execute(
            "SELECT DISTINCT komoditas FROM prices ORDER BY komoditas"
        ).fetchall()
        return [r['komoditas'] for r in rows]

    @_locked
    def get_price_count(self, komoditas: str) -> int:
        return self._conn.execute(
            "SELECT COUNT(*) FROM prices WHERE komoditas = ?",
            (komoditas,)
        ).fetchone()[0]

    # ------------------------------------------------------------------
    # PREDICTIONS
    # ------------------------------------------------------------------

    @_locked
    def upsert_predictions(self, df: pd.DataFrame, komoditas: str):
        """
        Simpan hasil prediksi incremental LSTM.
        df harus punya index Tanggal dan kolom: Aktual, Incr
        """
        if df.empty:
            return

        now  = self._now()
        rows = [
            (
                self._to_date_str(tanggal),
                komoditas,
                _safe_float(row.get('Aktual')),
                _safe_float(row.get('Incr')),
                now
            )
            for tanggal, row in df.iterrows()
        ]

        with self._transaction():
            self._conn.executemany(
                """INSERT INTO predictions
                   (tanggal, komoditas, harga_aktual, pred_incremental, updated_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(tanggal, komoditas) DO UPDATE SET
                       harga_aktual     = excluded.harga_aktual,
                       pred_incremental = excluded.pred_incremental,
                       updated_at       = excluded.updated_at
                """,
                rows
            )
        log.info(f"Predictions [{komoditas}]: {len(rows)} baris di-upsert")

    @_locked
    def get_predictions(
        self,
        komoditas: str,
        start: Optional[str] = None,
        end: Optional[str]   = None
    ) -> pd.DataFrame:
        """
        Ambil prediksi incremental untuk satu komoditas.
        Return DataFrame dengan kolom: Aktual, Incr.
        """
        query  = """SELECT tanggal, harga_aktual, pred_incremental
                    FROM predictions WHERE komoditas = ?"""
        params = [komoditas]
        if start:
            query += " AND tanggal >= ?"
            params.append(start)
        if end:
            query += " AND tanggal <= ?"
            params.append(end)
        query += " ORDER BY tanggal"

        df = pd.read_sql_query(query, self._conn, params=params,
                               parse_dates=['tanggal'])
        if df.empty:
            return df

        df = (df.set_index('tanggal')
                .rename(columns={
                    'harga_aktual'     : 'Aktual',
                    'pred_incremental' : 'Incr',
                }))
        df.index.name = 'Tanggal'
        return df

    @_locked
    def get_last_prediction_date(self, komoditas: str) -> Optional[date]:
        row = self._conn.execute(
            "SELECT MAX(tanggal) as last FROM predictions WHERE komoditas = ?",
            (komoditas,)
        ).fetchone()
        return date.fromisoformat(row['last']) if row and row['last'] else None

    # ------------------------------------------------------------------
    # MODEL METADATA
    # ------------------------------------------------------------------

    @_locked
    def update_model_metadata(
        self,
        komoditas: str,
        last_trained: Optional[str]  = None,
        training_end: Optional[str]  = None,
        window_size: Optional[int]   = None,
        rmse: Optional[float]        = None,
        mape: Optional[float]        = None,
        mae: Optional[float]         = None,
        n_finetune: Optional[int]    = None,
    ):
        """Simpan/update metadata model incremental untuk satu komoditas."""
        now = self._now()
        with self._transaction():
            self._conn.execute(
                """INSERT INTO model_metadata
                   (komoditas, last_trained, training_end, window_size,
                    rmse, mape, mae, n_finetune, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(komoditas) DO UPDATE SET
                       last_trained = COALESCE(excluded.last_trained, last_trained),
                       training_end = COALESCE(excluded.training_end, training_end),
                       window_size  = COALESCE(excluded.window_size,  window_size),
                       rmse         = COALESCE(excluded.rmse,         rmse),
                       mape         = COALESCE(excluded.mape,         mape),
                       mae          = COALESCE(excluded.mae,          mae),
                       n_finetune   = COALESCE(excluded.n_finetune,   n_finetune),
                       updated_at   = excluded.updated_at
                """,
                (
                    komoditas, last_trained, training_end, window_size,
                    rmse, mape, mae, n_finetune, now
                )
            )
        log.info(f"Model metadata [{komoditas}] diupdate")

    @_locked
    def get_model_metadata(self, komoditas: str) -> Optional[dict]:
        """
        Ambil metadata model untuk satu komoditas.
        Return dict atau None jika belum pernah dilatih.
        """
        row = self._conn.execute(
            "SELECT * FROM model_metadata WHERE komoditas = ?",
            (komoditas,)
        ).fetchone()
        return dict(row) if row else None

    @_locked
    def get_all_metadata(self) -> pd.DataFrame:
        """Semua metadata model dalam satu DataFrame."""
        return pd.read_sql_query(
            "SELECT * FROM model_metadata ORDER BY komoditas",
            self._conn
        )

    # ------------------------------------------------------------------
    # DIAGNOSTIK
    # ------------------------------------------------------------------

    @_locked
    def summary(self):
        """Cetak ringkasan isi database ke terminal."""
        weather_count = self.get_weather_count()
        last_weather  = self.get_last_weather_date()
        commodities   = self.get_commodities()

        print("\n" + "=" * 60)
        print(f"DATABASE: {Path(self.path).resolve()}")
        print("=" * 60)
        print(f"  Weather  : {weather_count} baris | terakhir: {last_weather}")
        print(f"  Komoditas: {len(commodities)}")
        for k in commodities:
            count     = self.get_price_count(k)
            last      = self.get_last_price_date(k)
            last_pred = self.get_last_prediction_date(k)
            meta      = self.get_model_metadata(k)
            mape      = f"{meta['mape']:.2f}%" if meta and meta['mape'] else "belum dilatih"
            print(
                f"    {k:<30} harga: {count:>4} | "
                f"s/d {last} | prediksi s/d {last_pred} | MAPE: {mape}"
            )
        print("=" * 60)


# =============================================================================
# HELPER INTERNAL
# =============================================================================

def _safe_float(value) -> Optional[float]:
    """Konversi nilai ke float, return None jika tidak bisa."""
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


# =============================================================================
# JALANKAN LANGSUNG (test / diagnostik)
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Database diagnostik')
    parser.add_argument('--db',      default='pihps.db',
                        help='Path ke file database')
    parser.add_argument('--summary', action='store_true',
                        help='Tampilkan ringkasan isi database')
    parser.add_argument('--test',    action='store_true',
                        help='Jalankan tes insert/query sederhana')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s')

    with Database(args.db) as db:
        if args.summary:
            db.summary()

        elif args.test:
            print("Menjalankan tes database...")

            # Test weather
            df_w = pd.DataFrame({
                'Suhu (C)'         : [24.5, 25.1, 23.8],
                'Kelembapan (%)'   : [85.2, 87.0, 89.3],
                'Curah Hujan (mm)' : [102.5, 0.0, 55.3],
            }, index=pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03']))
            df_w.index.name = 'Tanggal'
            db.upsert_weather(df_w)
            result_w = db.get_weather('2024-01-01', '2024-01-03')
            print(f"  Weather insert+query: {len(result_w)} baris ✓")

            # Test prices
            df_p = pd.DataFrame({
                'Harga (Rp)': [55.0, 57.5, 56.2],
            }, index=pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03']))
            df_p.index.name = 'Tanggal'
            db.upsert_prices(df_p, 'Cabai Merah', sumber='test')
            result_p = db.get_prices('Cabai Merah', '2024-01-01', '2024-01-03')
            print(f"  Prices insert+query: {len(result_p)} baris ✓")

            # Test predictions
            df_pred = pd.DataFrame({
                'Aktual' : [55.0, 57.5, 56.2],
                'Incr'   : [54.8, 57.1, 56.0],
            }, index=pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03']))
            df_pred.index.name = 'Tanggal'
            db.upsert_predictions(df_pred, 'Cabai Merah')
            result_pred = db.get_predictions('Cabai Merah')
            print(f"  Predictions insert+query: {len(result_pred)} baris ✓")

            # Test metadata
            db.update_model_metadata(
                'Cabai Merah',
                last_trained='2024-01-03',
                training_end='2024-12-31',
                window_size=30,
                rmse=2300.0, mape=2.9, mae=1600.0,
                n_finetune=8
            )
            meta = db.get_model_metadata('Cabai Merah')
            print(f"  Metadata insert+query: MAPE = {meta['mape']}% ✓")

            print("\nSemua tes berhasil!")
            db.summary()

        else:
            db.summary()