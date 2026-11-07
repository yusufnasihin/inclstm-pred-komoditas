"""
forecaster.py
-------------
Recursive multi-step forecasting untuk model LSTM incremental.

Cara kerja:
  1. Load model (.keras) dan scaler+feature_cols (.pkl) dari folder models/
  2. Ambil window 30 hari terakhir dari database sebagai input awal
  3. Untuk setiap hari ke depan:
       a. Prediksi harga menggunakan LSTM
       b. Build input hari berikutnya:
            - Harga   : prediksi hari ini (recursive)
            - Cuaca   : Open-Meteo forecast (hari 1-16), seasonal mean (hari 17+)
            - Holiday : dari kalender (deterministic, 100% akurat)
            - Lag     : geser dari window sebelumnya
  4. Return DataFrame prediksi dengan confidence envelope

PENGGUNAAN dari app.py:
  from forecaster import generate_forecast
  df_fc = generate_forecast('Cabai Merah', n_days=30, db=db)
  # df_fc columns: date, forecast, lower, upper

PENGGUNAAN langsung:
  python forecaster.py --komoditas "Cabai Merah" --days 30
"""

import pickle
import logging
import requests
import warnings
import numpy as np
import pandas as pd
from datetime import date, timedelta
from pathlib import Path

warnings.filterwarnings('ignore')
log = logging.getLogger(__name__)

# =============================================================================
# KONFIGURASI
# =============================================================================

MODEL_DIR   = Path('models')
WINDOW_SIZE = 30
TIMEZONE    = 'Asia/Jakarta'

# Open-Meteo forecast grid — sama dengan weather_fetcher.py
# Rata-rata dari beberapa titik di Jawa Barat
FORECAST_LOCATIONS = [
    {'lat': -6.9175, 'lon': 107.6191},  # Bandung
    {'lat': -6.5971, 'lon': 106.8060},  # Bogor
    {'lat': -6.7063, 'lon': 108.5570},  # Cirebon
]

BASE_HOLIDAY_COLS = [
    'is_idul_fitri', 'is_idul_adha', 'is_christmas',
    'is_new_year', 'is_ramadhan',
]
HOLIDAY_FEATURE_COLS = BASE_HOLIDAY_COLS + [f'{c}_H7' for c in BASE_HOLIDAY_COLS]

# Lag curah hujan per komoditas — HARUS identik dengan CROP_LAGS di
# pipeline.py (dipakai untuk merekonstruksi feature_cols saat forecasting).
# Komoditas yang tidak dicantumkan otomatis memakai default [14, 30] —
# lihat catatan lengkap soal Bawang Putih & Minyak Goreng di pipeline.py.
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


# =============================================================================
# STEP 1: LOAD MODEL & SCALER
# =============================================================================

def _load_model_and_scaler(komoditas: str, model_variant: str = 'incr'):
    """
    Load Keras model dan scaler dari disk.

    Args:
        model_variant: 'incr' (default, current behavior) loads the
                        incrementally fine-tuned model (incr_*.keras).
                       'base' loads the original fully-trained base model
                        (base_*.keras) before any replay-buffer fine-tuning
                        — useful for isolating whether an issue comes from
                        the base model itself or from incremental fine-tuning.
                        Both variants share the same scaler.

    Return (model, scaler, feature_cols) atau raise jika tidak ada.
    """
    if model_variant not in ('incr', 'base'):
        raise ValueError(f"model_variant harus 'incr' atau 'base', dapat: {model_variant!r}")

    name         = komoditas.replace(' ', '_')
    model_path   = MODEL_DIR / f'{model_variant}_{name}.keras'
    scaler_path  = MODEL_DIR / f'scaler_{name}.pkl'

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model tidak ditemukan: {model_path}\n"
            f"Jalankan: python pipeline.py --komoditas \"{komoditas}\" dulu."
        )
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler tidak ditemukan: {scaler_path}")

    # Import di sini agar tidak wajib install TF untuk import modul
    from tensorflow.keras.models import load_model
    model = load_model(model_path, compile=False)

    with open(scaler_path, 'rb') as f:
        saved = pickle.load(f)

    # Handle dua format: dict baru {'scaler':..,'feature_cols':..} atau scaler langsung
    if isinstance(saved, dict):
        scaler       = saved['scaler']
        feature_cols = saved['feature_cols']
    else:
        scaler       = saved
        # Reconstruct feature_cols dari CROP_LAGS
        lags         = CROP_LAGS.get(komoditas, [14, 30])
        feature_cols = (
            ['Harga', 'Suhu (C)', 'Kelembapan (%)', 'Curah Hujan (mm)']
            + [f'CH_lag{d}' for d in lags]
            + [f'Suhu_lag{d}' for d in lags]
            + HOLIDAY_FEATURE_COLS
        )

    log.info(f"Model loaded: {model_path.name} | features: {len(feature_cols)}")
    return model, scaler, feature_cols


# =============================================================================
# STEP 2: FETCH WEATHER FORECAST (Open-Meteo, days 1–16)
# =============================================================================

def _fetch_weather_forecast(n_days: int) -> pd.DataFrame:
    """
    Fetch weather forecast dari Open-Meteo untuk n_days ke depan.
    Return DataFrame index=date, cols=[Suhu (C), Kelembapan (%), Curah Hujan (mm)]
    Kosong jika API tidak bisa dijangkau.
    """
    frames = []
    for loc in FORECAST_LOCATIONS:
        try:
            r = requests.get(
                'https://api.open-meteo.com/v1/forecast',
                params={
                    'latitude'      : loc['lat'],
                    'longitude'     : loc['lon'],
                    'daily'         : 'temperature_2m_mean,relative_humidity_2m_mean,precipitation_sum',
                    'timezone'      : TIMEZONE,
                    'forecast_days' : min(n_days, 16),
                },
                timeout=10
            )
            if r.status_code != 200:
                continue
            d = r.json()['daily']
            df = pd.DataFrame({
                'Suhu (C)'         : d['temperature_2m_mean'],
                'Kelembapan (%)'   : d['relative_humidity_2m_mean'],
                'Curah Hujan (mm)' : d['precipitation_sum'],
            }, index=pd.to_datetime(d['time']))
            frames.append(df)
        except Exception as e:
            log.warning(f"Weather forecast fetch failed for {loc}: {e}")
            continue

    if not frames:
        return pd.DataFrame()

    # Spatial average across locations
    combined = pd.concat(frames).groupby(level=0).mean()
    combined.index.name = 'date'
    return combined


# =============================================================================
# STEP 3: SEASONAL WEATHER MEANS (fallback for days 17+)
# =============================================================================

def _seasonal_weather(df_weather_hist: pd.DataFrame, future_dates: list) -> pd.DataFrame:
    """
    Untuk setiap tanggal masa depan, hitung rata-rata cuaca historis
    berdasarkan bulan yang sama di semua tahun sebelumnya.
    Ini mencerminkan pola musiman tanpa mengklaim tahu cuaca aktual.
    """
    if df_weather_hist.empty:
        # Fallback ke nilai tropis rata-rata jika tidak ada data historis
        return pd.DataFrame({
            'Suhu (C)'         : [27.0] * len(future_dates),
            'Kelembapan (%)'   : [80.0] * len(future_dates),
            'Curah Hujan (mm)' : [5.0]  * len(future_dates),
        }, index=pd.DatetimeIndex(future_dates))

    records = []
    overall_mean = df_weather_hist.mean()  # global fallback if a month's data is unusable
    for d in future_dates:
        month_data = df_weather_hist[df_weather_hist.index.month == d.month]
        if month_data.empty:
            month_data = df_weather_hist
        row = {
            'Suhu (C)'         : month_data['Suhu (C)'].mean(),
            'Kelembapan (%)'   : month_data['Kelembapan (%)'].mean(),
            'Curah Hujan (mm)' : month_data['Curah Hujan (mm)'].mean(),
        }
        # A month can be non-empty but have zero valid (non-NaN) readings
        # for a specific column, in which case .mean() silently returns NaN
        # rather than raising — fall back to the overall historical mean
        # for just that column instead of letting NaN through.
        for col, val in row.items():
            if pd.isna(val):
                row[col] = overall_mean.get(col, {'Suhu (C)': 27.0, 'Kelembapan (%)': 80.0, 'Curah Hujan (mm)': 5.0}[col])
        records.append(row)
    return pd.DataFrame(records, index=pd.DatetimeIndex(future_dates))


# =============================================================================
# STEP 4: BUILD HOLIDAY FEATURES FOR FUTURE DATES (deterministic)
# =============================================================================

def _build_holiday_features(future_dates: list) -> pd.DataFrame:
    """
    Generate holiday features for future dates.
    These are 100% deterministic — we know exactly when holidays fall.
    """
    try:
        import holidays as hol_lib
        from hijridate import Hijri
    except ImportError:
        log.warning("holidays or hijridate not installed — holiday features will be zero")
        n = len(future_dates)
        return pd.DataFrame(
            np.zeros((n, len(HOLIDAY_FEATURE_COLS))),
            columns=HOLIDAY_FEATURE_COLS,
            index=pd.DatetimeIndex(future_dates)
        )

    years = list(set(d.year for d in future_dates))
    # Also include adjacent years for H-7 window
    years += [min(years) - 1, max(years) + 1]
    years = sorted(set(years))

    id_holidays = hol_lib.ID(years=years)

    # Build a range wide enough for H-7 rolling window
    all_dates = pd.date_range(
        min(future_dates) - timedelta(days=10),
        max(future_dates) + timedelta(days=1),
        freq='D'
    )
    df = pd.DataFrame(index=all_dates)

    df['_hname'] = [str(id_holidays.get(d.date(), '')).lower() for d in df.index]

    df['is_idul_fitri'] = df['_hname'].str.contains('idul fitri|eid al-fitr',    regex=True).astype(int)
    df['is_idul_adha']  = df['_hname'].str.contains('idul adha|eid al-adha',     regex=True).astype(int)
    df['is_christmas']  = df['_hname'].str.contains('natal|christmas',            regex=True).astype(int)
    df['is_new_year']   = ((df.index.month == 1) & (df.index.day == 1)).astype(int)

    # Ramadhan
    ramadhan_starts = []
    min_h = min(years) - 580
    max_h = max(years) - 578
    for h_year in range(min_h, max_h + 3):
        try:
            g = Hijri(h_year, 9, 1).to_gregorian()
            if g.year in years:
                ramadhan_starts.append(pd.Timestamp(g.year, g.month, g.day))
        except Exception:
            continue

    is_ramadhan = np.zeros(len(df), dtype=int)
    for i, d in enumerate(df.index):
        for ram_start in sorted(ramadhan_starts):
            if ram_start <= d:
                idul_fitri = None
                for cd in pd.date_range(ram_start, ram_start + pd.Timedelta(days=45)):
                    nm = str(id_holidays.get(cd.date(), '')).lower()
                    if 'idul fitri' in nm or 'eid al-fitr' in nm:
                        idul_fitri = cd
                        break
                if idul_fitri and ram_start <= d < idul_fitri:
                    is_ramadhan[i] = 1
                    break
    df['is_ramadhan'] = is_ramadhan
    df.drop(columns=['_hname'], inplace=True)

    # H-7 rolling window
    for col in BASE_HOLIDAY_COLS:
        df[f'{col}_H7'] = (
            df[col].rolling(window=8, min_periods=1).max()
            .shift(-7).fillna(0).astype(int) - df[col]
        ).clip(lower=0)

    # Trim to exact future dates
    df = df.loc[df.index.isin(pd.DatetimeIndex(future_dates))]
    return df[HOLIDAY_FEATURE_COLS]


# =============================================================================
# STEP 5: BUILD WEATHER FEATURE ARRAY FOR FUTURE DATES
# =============================================================================

def _build_weather_features(
    future_dates: list,
    df_weather_hist: pd.DataFrame
) -> pd.DataFrame:
    """
    Combine historical weather (for past dates), Open-Meteo forecast (days 0-16),
    and seasonal means (days 17+).
    """
    today_ts = pd.Timestamp(date.today())
    n_real_forecast = 16

    past_dates     = [d for d in future_dates if pd.Timestamp(d) < today_ts]
    real_fc_dates   = [d for d in future_dates if 0 <= (pd.Timestamp(d) - today_ts).days <= n_real_forecast]
    seasonal_dates  = [d for d in future_dates if (pd.Timestamp(d) - today_ts).days > n_real_forecast]

    frames = []

    # 1. Historical weather for past dates
    if past_dates:
        if not df_weather_hist.empty:
            df_past = df_weather_hist.reindex(pd.DatetimeIndex(past_dates))
            if df_past.isna().any().any():
                df_past_seasonal = _seasonal_weather(df_weather_hist, past_dates)
                df_past = df_past.fillna(df_past_seasonal)
            frames.append(df_past)
        else:
            seasonal_dates.extend(past_dates)

    # 2. Real forecast for near-future dates (0-16 days out)
    if real_fc_dates:
        df_real = _fetch_weather_forecast(len(real_fc_dates))
        if not df_real.empty:
            df_real = df_real.reindex(pd.DatetimeIndex(real_fc_dates))
            df_real = df_real.interpolate(method='linear').bfill().ffill()
            # Same safety net as the historical branch above: if the fetched
            # forecast's dates didn't actually line up with real_fc_dates
            # (API returned a different range, timezone mismatch, etc.),
            # reindex can leave entire rows/columns NaN that interpolate/
            # bfill/ffill can't rescue (nothing valid to fill FROM). Without
            # this, that NaN silently enters the recursive window and
            # poisons every prediction from the next day onward.
            if df_real.isna().any().any():
                log.warning(
                    f"Forecast Open-Meteo untuk {real_fc_dates[0]}..{real_fc_dates[-1]} "
                    "masih ada NaN setelah fill — kemungkinan tanggal tidak selaras "
                    "dengan hasil fetch. Fallback ke rata-rata musiman untuk sisanya."
                )
                df_real_seasonal = _seasonal_weather(df_weather_hist, real_fc_dates)
                df_real = df_real.fillna(df_real_seasonal)
            frames.append(df_real)
        else:
            log.warning("Open-Meteo forecast unavailable — using seasonal means")
            seasonal_dates.extend(real_fc_dates)

    # 3. Seasonal fallback for far-future dates (>16 days out)
    if seasonal_dates:
        df_seasonal = _seasonal_weather(df_weather_hist, seasonal_dates)
        frames.append(df_seasonal)

    if not frames:
        return _seasonal_weather(df_weather_hist, future_dates)

    result = pd.concat(frames).sort_index()
    result = result.loc[~result.index.duplicated(keep='first')]
    return result


# =============================================================================
# STEP 6: ASSEMBLE FULL FEATURE ROW FOR A FUTURE DATE
# =============================================================================

def _safe_weather_val(row: pd.Series, key: str, default: float) -> float:
    """
    Like row.get(key, default), but ALSO falls back to `default` if the key
    exists but its value is NaN — plain .get() only catches a missing key,
    not an existing-but-NaN one, and a single NaN weather value here
    poisons the recursive forecast window from the following day onward.
    """
    val = row.get(key, default)
    return default if pd.isna(val) else val


def _build_feature_row(
    d: pd.Timestamp,
    predicted_price: float,
    weather_row: pd.Series,
    holiday_row: pd.Series,
    rain_history: list,   
    suhu_history: list,   
    current_cdd: int,          # NEW: tracked state
    current_30d_sum: float,    # NEW: tracked state
    feature_cols: list,
    scaler,
    komoditas: str,
) -> np.ndarray:
    """
    Build a single feature row for date d in the same order as feature_cols.
    Returns scaled 1D array of length n_features.
    """
    lags         = CROP_LAGS.get(komoditas, [14, 30])
    col_idx      = {c: i for i, c in enumerate(feature_cols)}

    row = np.zeros(len(feature_cols))

    # Harga (unscaled → will be scaled together at the end)
    # We set index 0 to predicted price — will inverse/forward scale below
    row[col_idx['Harga']] = predicted_price

    # Weather
    row[col_idx['Suhu (C)']]          = _safe_weather_val(weather_row, 'Suhu (C)',         27.0)
    row[col_idx['Kelembapan (%)']]     = _safe_weather_val(weather_row, 'Kelembapan (%)',   80.0)
    row[col_idx['Curah Hujan (mm)']]   = _safe_weather_val(weather_row, 'Curah Hujan (mm)', 5.0)
    if 'consecutive_dry_days' in col_idx:
        row[col_idx['consecutive_dry_days']] = current_cdd
    if 'CH_30d_sum' in col_idx:
        row[col_idx['CH_30d_sum']] = current_30d_sum
    # Rainfall AND temperature lags — pull from raw history buffers kept long
    # enough to cover the largest configured lag (up to 90 days).
    # NOTE: previously CH_lag looked back into the 30-row scaled model
    # window, which meant CH_lag60/90 could never actually reach more than
    # 30 days during recursive forecasting — silently wrong for rice and
    # long-cycle chilies. Using dedicated raw history buffers fixes this.
    # Suhu_lag MUST be filled here — if feature_cols includes it (i.e. the
    # model was trained with it) but this function doesn't set it, it stays
    # at 0.0 by default, which is wildly outside anything the scaler was fit
    # on (real temperatures are ~24-32°C) and produces nonsensical output
    # from the very first prediction.
    for lag in lags:
        ch_key   = f'CH_lag{lag}'
        suhu_key = f'Suhu_lag{lag}'
        if ch_key in col_idx:
            row[col_idx[ch_key]] = (
                rain_history[-lag] if len(rain_history) >= lag else 0.0
            )
        if suhu_key in col_idx:
            row[col_idx[suhu_key]] = (
                suhu_history[-lag] if len(suhu_history) >= lag else weather_row.get('Suhu (C)', 27.0)
            )

    # Holiday features (already 0/1, no scaling needed)
    for hcol in HOLIDAY_FEATURE_COLS:
        if hcol in col_idx:
            row[col_idx[hcol]] = holiday_row.get(hcol, 0)

    # Scale the entire row using the fitted scaler
    row_scaled = scaler.transform(row.reshape(1, -1))[0]
    return row_scaled


# =============================================================================
# MAIN: GENERATE FORECAST
# =============================================================================

def generate_forecast(
    komoditas: str,
    n_days: int,
    db,
    confidence: float = 0.90,
    weather_override: dict = None,
    override_duration: int = None,
    trace: bool = False,
    model_variant: str = 'incr',
) -> pd.DataFrame:
    """
    Generate n_days recursive multi-step forecast for a commodity.

    Args:
        komoditas         : commodity name matching database
        n_days            : number of days to forecast
        db                : Database instance
        confidence        : confidence interval width (default 90%)
        weather_override  : dict with extreme weather values to inject, e.g:
                            {
                              'Suhu (C)': 35.0,          # absolute value
                              'Curah Hujan (mm)': 150.0,  # absolute value
                              'Kelembapan (%)': 95.0,     # absolute value
                            }
                            None = use real/seasonal weather (normal forecast)
        override_duration : how many days from start to apply weather_override.
                            After this, reverts to normal weather.
                            None = apply for full n_days.
        trace              : if True, add extra diagnostic columns to the
                            result — the actual Suhu/Kelembapan/Curah Hujan
                            value used as model input each day, and the raw
                            scaled model output. Off by default; existing
                            callers are unaffected.
        model_variant      : 'incr' (default, unchanged behavior) or 'base'.
                            See _load_model_and_scaler docstring.

    Return:
        pd.DataFrame with columns:
            Tanggal | Forecast | Lower | Upper | Sumber_Cuaca
            (+ Suhu_Used | Kelembapan_Used | CurahHujan_Used | Pred_Scaled if trace=True)
        Sumber_Cuaca values: 'Forecast', 'Seasonal', 'Simulasi'
        Empty DataFrame if model not found.
    """
    # -- Load model & scaler --
    try:
        model, scaler, feature_cols = _load_model_and_scaler(komoditas, model_variant)
    except FileNotFoundError as e:
        log.error(str(e))
        return pd.DataFrame()

    n_feat = len(feature_cols)

    # -- Get last WINDOW_SIZE days of actual data from DB --
    df_prices  = db.get_prices(komoditas)
    df_weather = db.get_weather()

    if df_prices.empty:
        log.error(f"Tidak ada data harga untuk {komoditas}")
        return pd.DataFrame()

    # Merge prices and weather into feature DataFrame
    df_hist = df_prices.rename(columns={'Harga (Rp)': 'Harga'})
    if not df_weather.empty:
        df_hist = df_hist.join(df_weather, how='left')

    # Add lags to historical data (rainfall AND temperature — live pipeline2.py
    # now trains with Suhu_lag* features, so this must compute them too)
    lags = CROP_LAGS.get(komoditas, [14, 30])
    for lag in lags:
        df_hist[f'CH_lag{lag}']   = df_hist['Curah Hujan (mm)'].shift(lag)
        df_hist[f'Suhu_lag{lag}'] = df_hist['Suhu (C)'].shift(lag)

    # --- NEW: Initialize drought features for history ---
    df_hist['CH_30d_sum'] = df_hist['Curah Hujan (mm)'].rolling(window=30, min_periods=1).sum()
    is_dry_hist = (df_hist['Curah Hujan (mm)'] < 1.0).astype(int)
    df_hist['consecutive_dry_days'] = is_dry_hist.groupby((is_dry_hist != is_dry_hist.shift()).cumsum()).cumsum()

    # Add holiday features to historical
    try:
        import holidays as hol_lib
        from hijridate import Hijri

        years_hist    = list(range(df_hist.index.min().year, df_hist.index.max().year + 2))
        id_hols       = hol_lib.ID(years=years_hist)
        df_hist['_hname'] = [str(id_hols.get(d.date(), '')).lower() for d in df_hist.index]
        df_hist['is_idul_fitri'] = df_hist['_hname'].str.contains('idul fitri|eid al-fitr', regex=True).astype(int)
        df_hist['is_idul_adha']  = df_hist['_hname'].str.contains('idul adha|eid al-adha',  regex=True).astype(int)
        df_hist['is_christmas']  = df_hist['_hname'].str.contains('natal|christmas',         regex=True).astype(int)
        df_hist['is_new_year']   = ((df_hist.index.month == 1) & (df_hist.index.day == 1)).astype(int)

        # Ramadhan
        ramadhan_starts = []
        min_h = min(years_hist) - 580
        max_h = max(years_hist) - 578
        for h_year in range(min_h, max_h + 3):
            try:
                g = Hijri(h_year, 9, 1).to_gregorian()
                if g.year in years_hist:
                    ramadhan_starts.append(pd.Timestamp(g.year, g.month, g.day))
            except Exception:
                continue
        is_ramadhan = np.zeros(len(df_hist), dtype=int)
        for i, d in enumerate(df_hist.index):
            for rs in sorted(ramadhan_starts):
                if rs <= d:
                    idul_fitri = None
                    for cd in pd.date_range(rs, rs + pd.Timedelta(days=45)):
                        nm = str(id_hols.get(cd.date(), '')).lower()
                        if 'idul fitri' in nm or 'eid al-fitr' in nm:
                            idul_fitri = cd; break
                    if idul_fitri and rs <= d < idul_fitri:
                        is_ramadhan[i] = 1; break
        df_hist['is_ramadhan'] = is_ramadhan
        df_hist.drop(columns=['_hname'], inplace=True)

        for col in BASE_HOLIDAY_COLS:
            df_hist[f'{col}_H7'] = (
                df_hist[col].rolling(window=8, min_periods=1).max()
                .shift(-7).fillna(0).astype(int) - df_hist[col]
            ).clip(lower=0)

    except Exception as e:
        log.warning(f"Holiday features failed: {e}")
        for hcol in HOLIDAY_FEATURE_COLS:
            df_hist[hcol] = 0

    # Fill missing columns with 0
    for col in feature_cols:
        if col not in df_hist.columns:
            df_hist[col] = 0

    df_hist = df_hist[feature_cols].copy()
    df_hist.interpolate(method='linear', inplace=True)
    df_hist.bfill(inplace=True)
    df_hist.fillna(0, inplace=True)

    # Scale historical data
    hist_scaled = scaler.transform(df_hist.values)

    # Take last WINDOW_SIZE rows as the initial input window
    if len(hist_scaled) < WINDOW_SIZE:
        log.error(f"Tidak cukup data historis untuk window size {WINDOW_SIZE}")
        return pd.DataFrame()

    window = hist_scaled[-WINDOW_SIZE:].copy()   # shape: (30, n_feat)

    # Raw (unscaled) rainfall AND temperature history, kept separately from
    # the scaled model window so CH_lag/Suhu_lag features can look back as
    # far as CROP_LAGS requires (up to 90 days) instead of being capped at
    # WINDOW_SIZE.
    rain_history = list(df_hist['Curah Hujan (mm)'].fillna(0.0).values)
    suhu_history = list(df_hist['Suhu (C)'].ffill().fillna(27.0).values)

    # -- Build future dates (calendar days, skip nothing — price on weekends = last known) --
    last_date    = df_hist.index.max()
    future_dates = [last_date + timedelta(days=i+1) for i in range(n_days)]

    # -- Get weather features for all future dates --
    df_wx_future  = _build_weather_features(future_dates, df_weather)

    # -- Get holiday features for all future dates (deterministic) --
    df_hol_future = _build_holiday_features(future_dates)

    # -- Track which dates use real vs seasonal weather --
    wx_sources = []
    cutoff     = pd.Timestamp(date.today() + timedelta(days=16))

    # -- Recursive multi-step prediction --
    forecasts    = []
    current_price = df_hist['Harga'].dropna().iloc[-1]   # last known price in original scale
    all_preds_scaled = []
    trace_suhu, trace_kelembapan, trace_curah_hujan = [], [], []

    override_days = override_duration if override_duration is not None else n_days

    # Weather scenarios previously injected a perfectly FLAT value (identical
    # Suhu/Kelembapan/Curah Hujan every single day of the override period).
    # Real weather always wobbles day to day; nothing in training ever holds
    # dead flat for weeks. That artificial constancy is destabilizing to the
    # recursive forecast independent of the override's magnitude.
    #
    # Fix: add realistic day-to-day noise around the override's mean value,
    # scaled to each variable's own historical day-to-day variability.
    # Seeded deterministically per (komoditas, scenario, duration) so
    # repeated calls with the same parameters are reproducible.
    weather_noise_std = {
        col: float(df_hist[col].diff().std()) if col in df_hist.columns else 0.0
        for col in ['Suhu (C)', 'Kelembapan (%)', 'Curah Hujan (mm)']
    }
    if weather_override is not None:
        seed_material = (komoditas, tuple(sorted(weather_override.items())), override_days)
        wx_rng = np.random.default_rng(abs(hash(seed_material)) % (2**32))
    else:
        wx_rng = None

    # --- NEW: Track initial states before entering the loop ---
    last_cdd = df_hist['consecutive_dry_days'].iloc[-1] if not df_hist.empty else 0

    for i, future_d in enumerate(future_dates):
        fd_ts = pd.Timestamp(future_d)

        # Weather row for this date
        wx_row = df_wx_future.loc[fd_ts] if fd_ts in df_wx_future.index \
                 else pd.Series({'Suhu (C)': 27.0, 'Kelembapan (%)': 80.0, 'Curah Hujan (mm)': 5.0})

        # Apply weather override if active for this day
        is_override_active = weather_override is not None and i < override_days
        if is_override_active:
            wx_row = wx_row.copy()
            for key, val in weather_override.items():
                if key in wx_row.index:
                    # Suppress noise for rainfall if the scenario dictates extreme drought
                    if key == 'Curah Hujan (mm)' and val < 1.0:
                        noise = 0.0
                    else:
                        noise = wx_rng.normal(0, weather_noise_std.get(key, 0.0))
                        
                    new_val = val + noise
                    
                    if key == 'Curah Hujan (mm)':
                        new_val = max(new_val, 0.0)
                    elif key == 'Kelembapan (%)':
                        new_val = float(np.clip(new_val, 0, 100))
                        
                    wx_row[key] = new_val
            wx_source = 'Simulasi'
        else:
            wx_source = 'Forecast' if fd_ts <= cutoff else 'Seasonal'

        hol_row = df_hol_future.loc[fd_ts] if fd_ts in df_hol_future.index \
                  else pd.Series({c: 0 for c in HOLIDAY_FEATURE_COLS})

        # --- NEW: Update tracked drought states based on today's weather ---
        current_rain = float(wx_row.get('Curah Hujan (mm)', 5.0))
        if current_rain < 1.0:
            last_cdd += 1
        else:
            last_cdd = 0
            
        # Compute derived weather features BEFORE appending so lags are correct
        current_30d_sum = sum(rain_history[-30:]) if len(rain_history) >= 30 else sum(rain_history)
        # -------------------------------------------------------------------

        # Build scaled input row (rain_history and suhu_history do NOT yet
        # contain today — the lag lookups want purely historical values)
        row_scaled = _build_feature_row(
            fd_ts, current_price, wx_row, hol_row,
            rain_history, suhu_history,  # history up to but not including today
            last_cdd, current_30d_sum,
            feature_cols, scaler, komoditas
        )

        # -- DIAGNOSTIC: catch NaN at the exact moment it enters the window --
        # Once ANY NaN enters row_scaled, it gets appended into the sliding
        # window and poisons every prediction from the NEXT day onward
        # forever (this day's own prediction is unaffected, since it still
        # uses the window BEFORE this row is appended — which is exactly
        # why "day 1 looks fine, day 2+ is all NaN" is the classic symptom
        # of this failure mode). This check identifies precisely which
        # source column was NaN and on which day, instead of only seeing
        # the downstream all-NaN result several steps later.
        if np.isnan(row_scaled).any():
            bad_cols = [feature_cols[j] for j in np.where(np.isnan(row_scaled))[0]]
            log.error(
                f"[{komoditas}] NaN terdeteksi di hari ke-{i+1} ({fd_ts.date()}), "
                f"sumber cuaca='{wx_source}'. Kolom NaN: {bad_cols}. "
                f"wx_row mentah: {dict(wx_row)}. "
                "Ini akan meracuni SEMUA prediksi mulai hari berikutnya — "
                "cek kolom di atas pada df_wx_future / df_hist untuk tanggal ini."
            )

        # NOW append today's values so future days can lag back to them
        rain_history.append(float(_safe_weather_val(wx_row, 'Curah Hujan (mm)', 5.0)))
        suhu_history.append(float(_safe_weather_val(wx_row, 'Suhu (C)', 27.0)))

        if trace:
            trace_suhu.append(wx_row.get('Suhu (C)'))
            trace_kelembapan.append(wx_row.get('Kelembapan (%)'))
            trace_curah_hujan.append(wx_row.get('Curah Hujan (mm)'))

        # Predict using current window
        X_input   = window.reshape(1, WINDOW_SIZE, n_feat)
        pred_scaled = model.predict(X_input, verbose=0)[0][0]

        # Denormalize prediction
        dummy             = np.zeros((1, n_feat))
        dummy[0, 0]       = pred_scaled
        current_price     = scaler.inverse_transform(dummy)[0, 0]

        forecasts.append(current_price)
        all_preds_scaled.append(pred_scaled)
        wx_sources.append(wx_source)

        # Update row_scaled with actual predicted price
        row_scaled[0] = pred_scaled

        # Slide window forward
        window = np.vstack([window[1:], row_scaled])

    # -- Confidence envelope --
    # Based on MAPE from model metadata, widening over time
    forecasts_arr = np.array(forecasts)

    # Use a conservative 5% base uncertainty, growing by 0.3% per day
    z = 1.645 if confidence == 0.90 else 1.96   # 90% or 95% CI
    base_err    = 0.05
    growth_rate = 0.003
    uncertainty = np.array([base_err + growth_rate * i for i in range(n_days)])
    lower = forecasts_arr * (1 - z * uncertainty)
    upper = forecasts_arr * (1 + z * uncertainty)

    result = pd.DataFrame({
        'Forecast'     : forecasts_arr,
        'Lower'        : lower,
        'Upper'        : upper,
        'Sumber_Cuaca' : wx_sources,
    }, index=pd.DatetimeIndex(future_dates))
    result.index.name = 'Tanggal'

    if trace:
        result['Suhu_Used']        = trace_suhu
        result['Kelembapan_Used']  = trace_kelembapan
        result['CurahHujan_Used']  = trace_curah_hujan
        result['Pred_Scaled']      = all_preds_scaled

    log.info(
        f"Forecast [{komoditas}] {n_days} hari: "
        f"Rp {forecasts_arr[0]:,.2f} → Rp {forecasts_arr[-1]:,.2f}"
    )
    return result


def generate_simulation_forecast(
    komoditas: str,
    n_days: int,
    db,
    scenario: dict,
    override_duration: int,
    df_weather_hist: pd.DataFrame,
    trace: bool = False,
    model_variant: str = 'incr',
) -> pd.DataFrame:
    """
    Convenience wrapper for the weather simulation page.

    Args:
        scenario          : dict from WEATHER_SCENARIOS with keys:
                            suhu (delta °C), ch (multiplier), rh (delta %)
        override_duration : how many days the extreme weather lasts
        df_weather_hist   : historical weather DataFrame to compute absolute values
        trace              : passthrough to generate_forecast — see its docstring
        model_variant      : 'incr' (default) or 'base' — see generate_forecast

    Return:
        Same as generate_forecast — Forecast, Lower, Upper, Sumber_Cuaca
    """
    # A true no-op scenario (all deltas/multiplier at identity) should behave
    # EXACTLY like the main page's forecast — real/seasonal day-to-day
    # weather, not a flat override. Previously "Normal" still built and
    # applied a weather_override dict (just with zero deltas), which meant
    # every day got pinned to the single overall historical average instead
    # of the real, naturally-varying forecast — producing a different result
    # from the main page even though nothing was supposed to be different.
    is_noop = (
        scenario.get('suhu', 0) == 0 and
        scenario.get('ch', 1.0) == 1.0 and
        scenario.get('rh', 0) == 0
    )
    if is_noop:
        log.info(
            f"Simulation [{komoditas}]: {scenario.get('label', 'custom')} — "
            "no delta, using real/seasonal weather (same as main page forecast)"
        )
        return generate_forecast(komoditas=komoditas, n_days=n_days, db=db, trace=trace, model_variant=model_variant)

    # Convert scenario deltas/multipliers to absolute values
    # using historical averages as the baseline
    if df_weather_hist.empty:
        avg_suhu = 27.0
        avg_ch   = 5.0
        avg_rh   = 80.0
    else:
        avg_suhu = df_weather_hist['Suhu (C)'].mean()
        avg_ch   = df_weather_hist['Curah Hujan (mm)'].mean()
        avg_rh   = df_weather_hist['Kelembapan (%)'].mean()

    weather_override = {
        'Suhu (C)'         : avg_suhu + scenario['suhu'],
        'Curah Hujan (mm)' : avg_ch   * scenario['ch'],
        'Kelembapan (%)'   : float(np.clip(avg_rh + scenario['rh'], 0, 100)),
    }

    log.info(
        f"Simulation [{komoditas}]: {scenario.get('label','custom')} "
        f"for {override_duration} days | "
        f"Suhu={weather_override['Suhu (C)']:.1f}°C, "
        f"CH={weather_override['Curah Hujan (mm)']:.1f}mm, "
        f"RH={weather_override['Kelembapan (%)']:.1f}%"
    )

    return generate_forecast(
        komoditas         = komoditas,
        n_days            = n_days,
        db                = db,
        weather_override  = weather_override,
        override_duration = override_duration,
        trace             = trace,
        model_variant     = model_variant,
    )


# =============================================================================
# MODEL MINGGUAN — khusus fitur Simulasi Cuaca
# =============================================================================
# Lihat catatan panjang di pipeline.py dekat get_weekly_feature_cols() untuk
# alasan model ini ada: rekursi harian (28-30 langkah untuk simulasi
# sebulan) terbukti tidak stabil untuk skenario cuaca berkelanjutan. Model
# mingguan mencakup horizon yang sama dalam ~4 langkah rekursif saja.
#
# Model ini tidak memakai fine-tuning inkremental (base-only, retrain penuh
# berkala) — jadi tidak ada model_variant 'incr' vs 'base' di sini, hanya
# satu file: weekly_{komoditas}.keras / weekly_scaler_{komoditas}.pkl.

WEEKLY_WINDOW_SIZE = 12


def _load_weekly_model_and_scaler(komoditas: str):
    """Load model mingguan dari disk. Return (model, scaler, feature_cols)."""
    name        = komoditas.replace(' ', '_')
    model_path  = MODEL_DIR / f'weekly_{name}.keras'
    scaler_path = MODEL_DIR / f'weekly_scaler_{name}.pkl'

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model mingguan tidak ditemukan: {model_path}\n"
            f"Jalankan: python pipeline.py --komoditas \"{komoditas}\" --weekly dulu."
        )
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler mingguan tidak ditemukan: {scaler_path}")

    from tensorflow.keras.models import load_model
    model = load_model(model_path, compile=False)

    with open(scaler_path, 'rb') as f:
        saved = pickle.load(f)
    scaler       = saved['scaler']
    feature_cols = saved['feature_cols']

    log.info(f"Model mingguan loaded: {model_path.name} | features: {len(feature_cols)}")
    return model, scaler, feature_cols


def _weekly_crop_lags(komoditas: str) -> list:
    """Sama seperti CROP_LAGS_WEEKS di pipeline.py — diturunkan dari CROP_LAGS harian."""
    daily_lags = CROP_LAGS.get(komoditas, [14, 30])
    return sorted(set(max(1, round(d / 7)) for d in daily_lags))


def _build_weekly_feature_row(
    week_start: pd.Timestamp,
    predicted_price: float,
    weather_row: pd.Series,
    holiday_row: pd.Series,
    rain_history: list,   # RAW (unscaled) curah hujan MINGGUAN, kronologis
    feature_cols: list,
    scaler,
    komoditas: str,
) -> np.ndarray:
    """Analog _build_feature_row() harian, tapi untuk satu langkah mingguan."""
    lags    = _weekly_crop_lags(komoditas)
    col_idx = {c: i for i, c in enumerate(feature_cols)}
    row     = np.zeros(len(feature_cols))

    row[col_idx['Harga']] = predicted_price
    row[col_idx['Suhu (C)']]        = weather_row.get('Suhu (C)', 27.0)
    row[col_idx['Kelembapan (%)']]  = weather_row.get('Kelembapan (%)', 80.0)
    row[col_idx['Curah Hujan (mm)']] = weather_row.get('Curah Hujan (mm)', 35.0)  # weekly SUM, bukan harian

    for lag in lags:
        key = f'CH_lag{lag}w'
        if key in col_idx:
            row[col_idx[key]] = rain_history[-lag] if len(rain_history) >= lag else 0.0

    if 'CH_4wk_sum' in col_idx:
        recent4 = rain_history[-4:] if len(rain_history) >= 1 else []
        row[col_idx['CH_4wk_sum']] = sum(recent4) if recent4 else 0.0

    if 'consecutive_dry_weeks' in col_idx:
        streak = 0
        for v in reversed(rain_history):
            if v < 7.0:
                streak += 1
            else:
                break
        row[col_idx['consecutive_dry_weeks']] = streak

    if 'bulan_sin' in col_idx:
        row[col_idx['bulan_sin']] = np.sin(2 * np.pi * week_start.month / 12)
        row[col_idx['bulan_cos']] = np.cos(2 * np.pi * week_start.month / 12)

    for hcol in HOLIDAY_FEATURE_COLS:
        if hcol in col_idx:
            row[col_idx[hcol]] = holiday_row.get(hcol, 0)

    return scaler.transform(row.reshape(1, -1))[0]


def generate_weekly_forecast(
    komoditas: str,
    n_weeks: int,
    db,
    weather_override: dict = None,
    override_duration: int = None,
    trace: bool = False,
) -> pd.DataFrame:
    """
    Forecast rekursif MINGGUAN — dipakai khusus oleh fitur Simulasi Cuaca
    karena jauh lebih sedikit langkah rekursif dibanding forecast harian
    untuk horizon yang sama (~4 langkah untuk 30 hari, bukan ~30 langkah).

    weather_override di sini memakai UNIT MINGGUAN:
        'Suhu (C)'         : rata-rata mingguan (sama seperti sebelumnya)
        'Kelembapan (%)'   : rata-rata mingguan (sama seperti sebelumnya)
        'Curah Hujan (mm)' : TOTAL mingguan (bukan harian!) — kalikan 7 jika
                              mengonversi dari angka harian skenario lama.
    override_duration dihitung dalam MINGGU, bukan hari.

    Return: DataFrame index=awal minggu, kolom Forecast/Lower/Upper/Sumber_Cuaca
            (+ Suhu_Used/Kelembapan_Used/CurahHujan_Used/Pred_Scaled jika trace=True)
    """
    try:
        model, scaler, feature_cols = _load_weekly_model_and_scaler(komoditas)
    except FileNotFoundError as e:
        log.error(str(e))
        return pd.DataFrame()

    n_feat = len(feature_cols)

    df_harga = db.get_prices(komoditas)
    df_cuaca = db.get_weather()
    if df_harga.empty or df_cuaca.empty:
        log.error(f"Data tidak cukup untuk forecast mingguan [{komoditas}]")
        return pd.DataFrame()

    df_harga = df_harga.rename(columns={'Harga (Rp)': 'Harga'})
    df_daily = df_harga.join(df_cuaca, how='outer').resample('D').asfreq()
    df_daily[['Harga', 'Suhu (C)', 'Kelembapan (%)']] = (
        df_daily[['Harga', 'Suhu (C)', 'Kelembapan (%)']].interpolate(method='linear')
    )
    df_daily['Curah Hujan (mm)'] = df_daily['Curah Hujan (mm)'].fillna(0.0)
    df_daily.bfill(inplace=True)

    df_w = df_daily.resample('W-MON', label='left', closed='left').agg({
        'Harga': 'mean', 'Suhu (C)': 'mean', 'Kelembapan (%)': 'mean', 'Curah Hujan (mm)': 'sum',
    })
    df_w.dropna(subset=['Harga'], inplace=True)

    lags = _weekly_crop_lags(komoditas)
    for lag in lags:
        df_w[f'CH_lag{lag}w'] = df_w['Curah Hujan (mm)'].shift(lag)
    df_w['CH_4wk_sum'] = df_w['Curah Hujan (mm)'].rolling(window=4, min_periods=1).sum()
    is_dry = (df_w['Curah Hujan (mm)'] < 7.0).astype(int)
    df_w['consecutive_dry_weeks'] = is_dry.groupby((is_dry != is_dry.shift()).cumsum()).cumsum()

    try:
        from holiday_features import compute_holiday_features
        df_hol_daily  = compute_holiday_features(df_daily.index)
        df_hol_weekly = df_hol_daily.resample('W-MON', label='left', closed='left').max()
        df_w = df_w.join(df_hol_weekly.reindex(df_w.index).fillna(0).astype(int))
    except Exception as e:
        log.warning(f"Fitur hari raya mingguan gagal: {e}")
        for hcol in HOLIDAY_FEATURE_COLS:
            df_w[hcol] = 0

    for col in feature_cols:
        if col not in df_w.columns:
            df_w[col] = 0
    df_w = df_w[feature_cols].copy()
    df_w.interpolate(method='linear', inplace=True)
    df_w.bfill(inplace=True)
    df_w.fillna(0, inplace=True)

    if len(df_w) < WEEKLY_WINDOW_SIZE:
        log.error(f"Data mingguan historis tidak cukup ({len(df_w)} < {WEEKLY_WINDOW_SIZE})")
        return pd.DataFrame()

    hist_scaled = scaler.transform(df_w.values)
    window = hist_scaled[-WEEKLY_WINDOW_SIZE:].copy()
    rain_history = list(df_w['Curah Hujan (mm)'].fillna(0.0).values)

    last_week_start = df_w.index.max()
    future_weeks = [last_week_start + timedelta(weeks=i + 1) for i in range(n_weeks)]
    override_weeks = override_duration if override_duration is not None else n_weeks

    avg_suhu = df_cuaca['Suhu (C)'].mean()
    avg_rh   = df_cuaca['Kelembapan (%)'].mean()

    weather_noise_std = {
        'Suhu (C)'        : float(df_w['Suhu (C)'].diff().std()) if len(df_w) > 1 else 0.0,
        'Kelembapan (%)'  : float(df_w['Kelembapan (%)'].diff().std()) if len(df_w) > 1 else 0.0,
        'Curah Hujan (mm)': float(df_w['Curah Hujan (mm)'].diff().std()) if len(df_w) > 1 else 0.0,
    }
    if weather_override is not None:
        seed_material = (komoditas, tuple(sorted(weather_override.items())), override_weeks, 'weekly')
        wx_rng = np.random.default_rng(abs(hash(seed_material)) % (2**32))
    else:
        wx_rng = None

    forecasts, all_preds_scaled = [], []
    trace_suhu, trace_rh, trace_ch = [], [], []
    current_price = df_w['Harga'].dropna().iloc[-1]

    for i, wk_start in enumerate(future_weeks):
        is_active = weather_override is not None and i < override_weeks
        if is_active:
            wx_row = pd.Series({'Suhu (C)': avg_suhu, 'Kelembapan (%)': avg_rh, 'Curah Hujan (mm)': rain_history[-1] if rain_history else 35.0})
            for key, val in weather_override.items():
                if key in wx_row.index:
                    noise   = wx_rng.normal(0, weather_noise_std.get(key, 0.0))
                    new_val = val + noise
                    if key == 'Curah Hujan (mm)':
                        new_val = max(new_val, 0.0)
                    elif key == 'Kelembapan (%)':
                        new_val = float(np.clip(new_val, 0, 100))
                    wx_row[key] = new_val
            wx_source = 'Simulasi'
        else:
            # Di luar periode override: pakai rata-rata musiman minggu tsb
            # dari data historis (pendekatan sederhana — model mingguan
            # tidak mengambil forecast Open-Meteo real-time seperti harian,
            # karena horizon mingguan biasanya jauh melampaui jangkauan
            # forecast cuaca beberapa hari ke depan).
            month = wk_start.month
            hist_month = df_daily[df_daily.index.month == month]
            wx_row = pd.Series({
                'Suhu (C)'        : hist_month['Suhu (C)'].mean() if not hist_month.empty else avg_suhu,
                'Kelembapan (%)'  : hist_month['Kelembapan (%)'].mean() if not hist_month.empty else avg_rh,
                'Curah Hujan (mm)': hist_month['Curah Hujan (mm)'].mean() * 7 if not hist_month.empty else 35.0,
            })
            wx_source = 'Seasonal'

        hol_row = pd.Series({c: 0 for c in HOLIDAY_FEATURE_COLS})  # H-7 mingguan disederhanakan jadi 0 di luar riwayat

        row_scaled = _build_weekly_feature_row(
            wk_start, current_price, wx_row, hol_row,
            rain_history, feature_cols, scaler, komoditas
        )
        rain_history.append(float(_safe_weather_val(wx_row, 'Curah Hujan (mm)', 35.0)))

        if trace:
            trace_suhu.append(wx_row.get('Suhu (C)'))
            trace_rh.append(wx_row.get('Kelembapan (%)'))
            trace_ch.append(wx_row.get('Curah Hujan (mm)'))

        X_input = window.reshape(1, WEEKLY_WINDOW_SIZE, n_feat)
        pred_scaled = model.predict(X_input, verbose=0)[0][0]

        dummy = np.zeros((1, n_feat))
        dummy[0, 0] = pred_scaled
        current_price = scaler.inverse_transform(dummy)[0, 0]

        forecasts.append(current_price)
        all_preds_scaled.append(pred_scaled)
        row_scaled[0] = pred_scaled
        window = np.vstack([window[1:], row_scaled])

    forecasts_arr = np.array(forecasts)
    margin = forecasts_arr * 0.08  # pita ketidakpastian lebih lebar per-langkah karena tiap langkah = 1 minggu penuh
    result = pd.DataFrame({
        'Forecast'    : forecasts_arr,
        'Lower'       : forecasts_arr - margin,
        'Upper'       : forecasts_arr + margin,
        'Sumber_Cuaca': ['Simulasi' if (weather_override is not None and i < override_weeks) else 'Seasonal' for i in range(n_weeks)],
    }, index=pd.DatetimeIndex(future_weeks))
    result.index.name = 'MingguMulai'

    if trace:
        result['Suhu_Used']       = trace_suhu
        result['Kelembapan_Used'] = trace_rh
        result['CurahHujan_Used'] = trace_ch
        result['Pred_Scaled']     = all_preds_scaled

    log.info(
        f"Forecast mingguan [{komoditas}] {n_weeks} minggu: "
        f"Rp {forecasts_arr[0]:,.2f} → Rp {forecasts_arr[-1]:,.2f}"
    )
    return result


def generate_weekly_simulation_forecast(
    komoditas: str,
    n_weeks: int,
    db,
    scenario: dict,
    override_duration_weeks: int,
    df_weather_hist: pd.DataFrame,
    trace: bool = False,
) -> pd.DataFrame:
    """
    Wrapper skenario cuaca untuk model MINGGUAN — analog
    generate_simulation_forecast() harian. `scenario` memakai format
    WEATHER_SCENARIOS yang sama (suhu delta, ch multiplier, rh delta),
    dikonversi ke unit mingguan di sini.
    """
    is_noop = (
        scenario.get('suhu', 0) == 0 and
        scenario.get('ch', 1.0) == 1.0 and
        scenario.get('rh', 0) == 0
    )
    if is_noop:
        return generate_weekly_forecast(komoditas, n_weeks, db, trace=trace)

    if df_weather_hist.empty:
        avg_suhu, avg_ch_daily, avg_rh = 27.0, 5.0, 80.0
    else:
        avg_suhu     = df_weather_hist['Suhu (C)'].mean()
        avg_ch_daily = df_weather_hist['Curah Hujan (mm)'].mean()
        avg_rh       = df_weather_hist['Kelembapan (%)'].mean()

    abs_suhu = avg_suhu + scenario['suhu']
    # ch multiplier diterapkan ke rata-rata HARIAN lalu dikali 7 untuk dapat
    # total MINGGUAN, supaya magnitudo tetap konsisten dengan skenario harian
    abs_ch_weekly = max(avg_ch_daily * scenario['ch'] * 7, 0.0)
    abs_rh   = float(np.clip(avg_rh + scenario['rh'], 0, 100))

    weather_override = {
        'Suhu (C)'         : abs_suhu,
        'Curah Hujan (mm)' : abs_ch_weekly,
        'Kelembapan (%)'   : abs_rh,
    }

    log.info(
        f"Simulasi mingguan [{komoditas}]: {scenario.get('label', 'custom')} "
        f"untuk {override_duration_weeks} minggu | "
        f"Suhu={abs_suhu:.1f}°C, CH={abs_ch_weekly:.1f}mm/minggu, RH={abs_rh:.1f}%"
    )

    return generate_weekly_forecast(
        komoditas         = komoditas,
        n_weeks           = n_weeks,
        db                = db,
        weather_override  = weather_override,
        override_duration = override_duration_weeks,
        trace             = trace,
    )


# =============================================================================
# JALANKAN LANGSUNG
# =============================================================================

if __name__ == '__main__':
    import argparse, sys
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s')

    parser = argparse.ArgumentParser(description='Generate price forecast')
    parser.add_argument('--komoditas', default='Cabai Merah')
    parser.add_argument('--days',      type=int, default=30)
    parser.add_argument('--db',        default='pihps.db')
    args = parser.parse_args()

    from database import Database
    with Database(args.db) as db:
        df = generate_forecast(args.komoditas, args.days, db)

    if df.empty:
        print("Forecast gagal — cek log di atas")
        sys.exit(1)

    print(f"\nForecast {args.komoditas} — {args.days} hari ke depan:\n")
    print(f"{'Tanggal':<15} {'Forecast':>12} {'Lower':>12} {'Upper':>12} {'Cuaca'}")
    print("-" * 58)
    for d, row in df.iterrows():
        print(
            f"{str(d.date()):<15} "
            f"Rp {row['Forecast']:>9,.2f} "
            f"Rp {row['Lower']:>9,.2f} "
            f"Rp {row['Upper']:>9,.2f} "
            f"{row['Sumber_Cuaca']}"
        )