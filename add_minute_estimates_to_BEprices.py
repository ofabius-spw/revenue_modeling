import pandas as pd
import pytz

# Timezone used in dataset 1 (handles CET/CEST daylight saving)
tz_local = pytz.timezone('Europe/Brussels')

# --- Load Dataset 1: Per-minute estimates ---
df_min = pd.read_csv('BE_minute_imbprice_est_trialperiod.csv', sep=';', skipinitialspace=True)

# Parse ISO 8601 timestamps with timezones directly and convert to UTC
df_min['Datetime'] = pd.to_datetime(df_min['Datetime'], utc=True)
df_min['Quarter hour'] = pd.to_datetime(df_min['Quarter hour'], utc=True)

# Drop invalid rows (if any timestamp couldn't be localized)
df_min = df_min.dropna(subset=['Datetime', 'Quarter hour'])

# Keep only necessary columns
df_min = df_min[['Datetime', 'Quarter hour', 'Imbalance Price']]

# Calculate minute offset from PTU start
df_min['minutes_after_ptu_start'] = (df_min['Datetime'] - df_min['Quarter hour']).dt.total_seconds() // 60
df_min['minutes_after_ptu_start'] = df_min['minutes_after_ptu_start'].astype(int)

# Pivot: 1 column per per-minute forecast
df_min_pivot = df_min.pivot_table(
    index='Quarter hour',
    columns='minutes_after_ptu_start',
    values='Imbalance Price'
)
df_min_pivot.columns = [f"Estimate_{int(c)}min_after" for c in df_min_pivot.columns]
df_min_pivot = df_min_pivot.reset_index()

# --- Load Dataset 2: Settled PTU prices ---
df_ptu = pd.read_csv('BE_merged_imbalance_with_forecast.csv', sep=';', skipinitialspace=True)

# Parse 'Forecasted QH (utc)' as timezone-aware UTC
df_ptu['Forecasted QH (utc)'] = pd.to_datetime(df_ptu['Forecasted QH (utc)'], utc=True)

# Merge estimates into PTU-level dataset
df_merged = df_ptu.merge(df_min_pivot, left_on='Forecasted QH (utc)', right_on='Quarter hour', how='left')

# Drop 'Quarter hour' if not needed
df_merged = df_merged.drop(columns=['Quarter hour'])

# Output result
df_merged.to_csv('ptu_with_estimates.csv', index=False)
