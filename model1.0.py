# %%
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt

#%%
equity = pd.read_excel("Datapython.xlsx", sheet_name="LOG_RETURN_EQUITY")      
market = pd.read_excel("Datapython.xlsx", sheet_name="LOG_RETURN_S&PTSX")  
rf = pd.read_excel("Datapython.xlsx", sheet_name="LOG_RETURN_REPO")    
events = pd.read_excel("Datapython.xlsx", sheet_name="DATE_EVENT")  

#%%
equity_long = equity.melt(
    id_vars="Dates",        
    var_name="ticker",      
    value_name="Ri"         
)

print("Format long des rendements actions :")
display(equity_long.head())

#%%
equity_long["Dates"] = pd.to_datetime(equity_long["Dates"])
market["Dates"] = pd.to_datetime(market["Dates"])
rf["Dates"] = pd.to_datetime(rf["Dates"])
events["Date"] = pd.to_datetime(events["Date"])

market = market.rename(columns={"S&PTSX": "Rm"})
rf = rf.rename(columns={"CAONREPO Index": "Rf"})

data = equity_long.merge(market, on="Dates", how="left").merge(rf, on="Dates", how="left")

print("Données fusionnées (titre + marché + repo) :")
display(data.head())

# %%
def compute_beta_for_row(row):
    """
    Calcule le bêta (modèle de marché / Modèle 1)
    pour un ticker et une date d'événement donnés.
    """
    ticker = row["Ticker"]
    event_date = row["Date"]

    d = data[data["ticker"] == ticker].copy()

    # Fenêtre d'estimation : [-250, -30] 
    mask = (d["Dates"] >= event_date - pd.Timedelta(days=250)) & \
           (d["Dates"] <= event_date - pd.Timedelta(days=30))
    window = d.loc[mask].dropna(subset=["Ri", "Rm", "Rf"])

    if len(window) < 30:
        return np.nan

    window["excess_i"] = window["Ri"] - window["Rf"]  
    window["excess_m"] = window["Rm"] - window["Rf"]   

    X = sm.add_constant(window["excess_m"])
    y = window["excess_i"]

    model = sm.OLS(y, X).fit()

    return model.params["excess_m"]  

# %%
events["beta_modele1"] = events.apply(compute_beta_for_row, axis=1)

print("Événements avec leur bêta Modèle 1 :")
display(events[["Ticker", "Date", "beta_modele1"]])

# %%
def compute_ar_modele1_for_row(row):
    """
    Calcule les rendements anormaux (AR) sur [-5, +5] jours ouvrables
    selon le Modèle 1 (CAPM / modèle de marché) :
        E[Ri] = Rf + beta * (Rm - Rf)
    """
    ticker = row["Ticker"]
    event_date = row["Date"]
    beta = row["beta_modele1"]

    if pd.isna(beta):
        return pd.Series([np.nan]*11, index=[f"AR_{i}" for i in range(-5, 6)])

    d = data[data["ticker"] == ticker].copy()
    if d.empty:
        return pd.Series([np.nan]*11, index=[f"AR_{i}" for i in range(-5, 6)])

    d = d.sort_values("Dates").reset_index(drop=True)

    mask_event = d["Dates"] == event_date
    if not mask_event.any():
        return pd.Series([np.nan]*11, index=[f"AR_{i}" for i in range(-5, 6)])

    event_idx = mask_event[mask_event].index[0]

    start_idx = event_idx - 5
    end_idx   = event_idx + 5

    window = d.iloc[max(start_idx, 0): end_idx+1].copy()

    window = window.reset_index(drop=True)
    rel_idx = list(range(start_idx, end_idx+1))

    if len(window) != len(rel_idx):
        return pd.Series([np.nan]*11, index=[f"AR_{i}" for i in range(-5, 6)])

    window["excess_m"] = window["Rm"] - window["Rf"]
    expected = window["Rf"] + beta * window["excess_m"]

    AR = window["Ri"] - expected

    AR.index = [f"AR_{i}" for i in range(-5, 6)]
    return AR

ar_m1 = events.apply(compute_ar_modele1_for_row, axis=1)
events_m1 = pd.concat([events, ar_m1], axis=1)

display(events_m1.head())

# %%
ar_cols = [f"AR_{i}" for i in range(-5, 6)] 
events_m1["CAR_-5_+5"] = events_m1[ar_cols].sum(axis=1)

display(events_m1[["Ticker", "Date", "CAR_-5_+5"]])

# %%
car_sector_mean_m1 = events_m1.groupby("Industry")["CAR_-5_+5"].mean()

print("CAR moyen par secteur (-5 à +5 jours) – Modèle 1 :")
display(car_sector_mean_m1)

# %%
results_m1 = {}

for sector in car_sector_mean_m1.index:
    vals = events_m1.loc[
        events_m1["Industry"] == sector, "CAR_-5_+5"
    ].dropna()

    if len(vals) < 2:
        results_m1[sector] = {
            "mean_CAR": vals.mean() if len(vals) > 0 else np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
            "n": len(vals)
        }
        continue

    t_stat, p_value = stats.ttest_1samp(vals, 0)

    results_m1[sector] = {
        "mean_CAR": vals.mean(),
        "t_stat": t_stat,
        "p_value": p_value,
        "n": len(vals)
    }

car_sector_test_m1 = pd.DataFrame(results_m1).T

print("Test : le CAR moyen par secteur est-il ≠ 0 ? (Modèle 1)")
display(car_sector_test_m1)

# %%
ar_cols = [f"AR_{i}" for i in range(-5, 6)]

sectors = sorted(events_m1["Industry"].unique())

print("Secteurs trouvés :", sectors)

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
axes = axes.ravel()

x_vals = list(range(-5, 6))

for ax, sector in zip(axes, sectors):

    df_sec = events_m1[events_m1["Industry"] == sector]

    mean_ar = df_sec[ar_cols].mean()

    ax.plot(x_vals, mean_ar.values, marker="o")
    ax.axhline(0, linewidth=1)
    ax.set_title(sector)
    ax.set_xlabel("Jour relatif à l'événement")
    ax.set_ylabel("AR moyen (Modèle 1)")
    ax.grid(alpha=0.3)
    ax.set_xticks(x_vals)

plt.suptitle("Rendements anormaux moyens par secteur\nFenêtre [-5 ; +5] – Modèle 1 (CAPM / marché)",
             fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
