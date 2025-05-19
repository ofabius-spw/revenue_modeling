import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Load merged dataframe
df = pd.read_csv('ptu_with_estimates.csv')

# Target variables
true_col = "- Imbalance Price [EUR/MWh] - SCA|BE"
forecast_col = "Imbalance price forecast (EUR)"
reference_col = "Estimate_14min_after"

# Initialize dictionaries
r2_vs_true = {}
r2_vs_14min = {}

# Forecast made before PTU (t = -1)
if forecast_col in df.columns:
    valid = df[[true_col, forecast_col]].dropna()
    r2_vs_true[-1] = r2_score(valid[true_col], valid[forecast_col])
    
    valid_14 = df[[reference_col, forecast_col]].dropna()
    r2_vs_14min[-1] = r2_score(valid_14[reference_col], valid_14[forecast_col])

# Minute-by-minute estimates (t = 0 to 14)
for t in range(15):
    col = f"Estimate_{t}min_after"
    if col in df.columns:
        valid = df[[true_col, col]].dropna()
        if not valid.empty:
            r2_vs_true[t] = r2_score(valid[true_col], valid[col])

        valid_14 = df[[reference_col, col]].dropna()
        if not valid_14.empty:
            r2_vs_14min[t] = r2_score(valid_14[reference_col], valid_14[col])

# Prepare plot
plt.figure(figsize=(10, 6))
plt.plot(list(r2_vs_true.keys()), list(r2_vs_true.values()), marker='o', label='R² vs ENTSO-E Imbalance Price')
plt.plot(list(r2_vs_14min.keys()), list(r2_vs_14min.values()), marker='o', label='R² vs 14-min Estimate')

# Labels and aesthetics
plt.title('Correlation of Estimates with ENTSO-E Imbalance Price and 14-min Estimate')
plt.xlabel('Minutes after PTU Start (t)')
plt.ylabel('R² Score')
plt.xticks(range(-1, 15))  # from -1 to 14
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
