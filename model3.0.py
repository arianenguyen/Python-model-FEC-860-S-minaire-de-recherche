#%%
from scipy import stats
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy import stats

# %%
file = "Datapython.xlsx"

equity = pd.read_excel(file, sheet_name="LOG_RETURN_EQUITY")
events = pd.read_excel(file, sheet_name="DATE_EVENT")
fama5 = pd.read_excel(file, sheet_name="FAMA5")

#%%
equity_long = equity.melt(
    id_vars="Dates",     
    var_name="ticker",  
    value_name="Ri"    
)

print("Format long des rendements actions avec index :")
display(equity_long.head(20))

#%%
equity_long["Dates"] = pd.to_datetime(equity_long["Dates"])
events["Date"] = pd.to_datetime(events["Date"])
fama5["DATE"]= pd.to_datetime(fama5["DATE"])

#%%
fama5 = fama5.rename(columns={
    "Mkt-RF": "MKT_RF", 
    "SMB": "SMB",
    "HML": "HML",
    "RWL": "RMW",   
    "CWA": "CMA",  
    "RF": "Rf"     
})

print("FAMA5 après renommage :")
display(fama5.head())

#%%
data_ff5 = equity_long.merge(
    fama5[["DATE", "MKT_RF", "SMB", "HML", "RMW", "CMA", "Rf"]],
    left_on="Dates", right_on="DATE",
    how="left"
)

print("Données fusionnées pour le Modèle 3 (Fama-French 5 facteurs) :")
display(data_ff5.head())

#%%
def compute_betas_fama5_for_row(row):
    """
    Modèle 5 facteurs (Fama-French) :
    Ri - Rf = alpha
              + beta_MKT * MKT_RF
              + beta_SMB * SMB
              + beta_HML * HML
              + beta_RMW * RMW
              + beta_CMA * CMA

    Fenêtre d'estimation : [-250, -30] jours de bourse avant la date d'événement.
    """
    ticker = row["Ticker"]
    event_date = row["Date"]

    d = data_ff5[data_ff5["ticker"] == ticker].copy()
    if d.empty:
        return pd.Series([np.nan]*6,
                         index=["alpha_ff5", "beta_mkt_ff5", "beta_smb_ff5",
                                "beta_hml_ff5", "beta_rmw_ff5", "beta_cma_ff5"])

    d = d.sort_values("Dates").reset_index(drop=True)

    div_decla = d["Dates"] == event_date

    div_decla = d["Dates"] == event_date
    if not div_decla.any():
        return pd.Series([np.nan]*6,
                         index=["alpha_ff5", "beta_mkt_ff5", "beta_smb_ff5",
                                "beta_hml_ff5", "beta_rmw_ff5", "beta_cma_ff5"])

    event_idx = div_decla[div_decla].index[0]

    start_idx = max(0, event_idx - 250)
    end_idx   = event_idx - 30

    window = d.iloc[start_idx:end_idx + 1].dropna(
        subset=["Ri", "MKT_RF", "SMB", "HML", "RMW", "CMA", "Rf"]
    )

    if len(window) < 30:
        return pd.Series([np.nan]*6,
                         index=["alpha_ff5", "beta_mkt_ff5", "beta_smb_ff5",
                                "beta_hml_ff5", "beta_rmw_ff5", "beta_cma_ff5"])

    window = window.copy()
    window["excess_i_ff5"] = window["Ri"] - window["Rf"]

    X = window[["MKT_RF", "SMB", "HML", "RMW", "CMA"]]
    X = sm.add_constant(X)  # alpha
    y = window["excess_i_ff5"]

    model = sm.OLS(y, X).fit()

    return pd.Series(
        [model.params["const"],
         model.params["MKT_RF"],
         model.params["SMB"],
         model.params["HML"],
         model.params["RMW"],
         model.params["CMA"]],
        index=["alpha_ff5", "beta_mkt_ff5", "beta_smb_ff5",
               "beta_hml_ff5", "beta_rmw_ff5", "beta_cma_ff5"]
    )
#%%
betas_ff5 = events.apply(compute_betas_fama5_for_row, axis=1)
events_ff5 = pd.concat([events, betas_ff5], axis=1)

print("Événements avec bêtas Fama-French (Modèle 3) :")
display(events_ff5[["Ticker", "Date",
                    "alpha_ff5", "beta_mkt_ff5", "beta_smb_ff5",
                    "beta_hml_ff5", "beta_rmw_ff5", "beta_cma_ff5"]])

# %%
def compute_ar_ff5_for_row(row):
    """
    Calcule les rendements anormaux (AR) sur [-5, +5] jours ouvrables
    selon le modèle Fama-French 5 facteurs.
    """
    ticker = row["Ticker"]
    event_date = row["Date"]

    alpha    = row["alpha_ff5"]
    beta_mkt = row["beta_mkt_ff5"]
    beta_smb = row["beta_smb_ff5"]
    beta_hml = row["beta_hml_ff5"]
    beta_rmw = row["beta_rmw_ff5"]
    beta_cma = row["beta_cma_ff5"]

    if pd.isna(alpha):
        return pd.Series([np.nan]*11,
                         index=[f"AR_{i}" for i in range(-5, 6)])

    d = data_ff5[data_ff5["ticker"] == ticker].copy()
    if d.empty:
        return pd.Series([np.nan]*11,
                         index=[f"AR_{i}" for i in range(-5, 6)])

    d = d.sort_values("Dates").reset_index(drop=True)

    mask_event = d["Dates"] == event_date
    if not mask_event.any():
        return pd.Series([np.nan]*11,
                         index=[f"AR_{i}" for i in range(-5, 6)])

    event_idx = mask_event[mask_event].index[0]

    start_idx = event_idx - 5
    end_idx   = event_idx + 5

    window = d.iloc[max(start_idx, 0): end_idx+1]

    window = window.reset_index(drop=True)
    rel_idx = list(range(start_idx, end_idx+1))

    if len(window) != len(rel_idx):
        return pd.Series([np.nan]*11,
                         index=[f"AR_{i}" for i in range(-5, 6)])

    expected = (
        window["Rf"]
        + alpha
        + beta_mkt * window["MKT_RF"]
        + beta_smb * window["SMB"]
        + beta_hml * window["HML"]
        + beta_rmw * window["RMW"]
        + beta_cma * window["CMA"]
    )

    AR = window["Ri"] - expected

    AR.index = [f"AR_{i}" for i in range(-5, 6)]
    return AR

ar_ff5 = events_ff5.apply(compute_ar_ff5_for_row, axis=1)
events_with_ar_ff5 = pd.concat([events_ff5, ar_ff5], axis=1)

display(events_with_ar_ff5.head())

#%%
ar_cols = [f"AR_{i}" for i in range(-5, 6)]  
events_with_ar_ff5["CAR_-5_+5"] = events_with_ar_ff5[ar_cols].sum(axis=1)

display(events_with_ar_ff5[["Ticker", "Date", "CAR_-5_+5"]])

#%%
car_sector_mean_ff5 = events_with_ar_ff5.groupby("Industry")["CAR_-5_+5"].mean()

print("CAR moyen par secteur (-5 à +5 jours) – Modèle FF5 :")
display(car_sector_mean_ff5)

#%%

results_ff5 = {}

for sector in car_sector_mean_ff5.index:
    vals = events_with_ar_ff5.loc[
        events_with_ar_ff5["Industry"] == sector, "CAR_-5_+5"
    ].dropna()

    if len(vals) < 2:
        results_ff5[sector] = {
            "mean_CAR": vals.mean() if len(vals) > 0 else np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
            "n": len(vals)
        }
        continue

    t_stat, p_value = stats.ttest_1samp(vals, 0)

    results_ff5[sector] = {
        "mean_CAR": vals.mean(),
        "t_stat": t_stat,
        "p_value": p_value,
        "n": len(vals)
    }

car_sector_test_ff5 = pd.DataFrame(results_ff5).T

print("Test : le CAR moyen par secteur est-il ≠ 0 ? (Modèle FF5)")
display(car_sector_test_ff5)

#%%
ar_cols = [f"AR_{i}" for i in range(-5, 6)]

sectors = sorted(events_with_ar_ff5["Industry"].unique())

print("Secteurs trouvés :", sectors)

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
axes = axes.ravel()

x_vals = list(range(-5, 6))

for ax, sector in zip(axes, sectors):
    df_sec = events_with_ar_ff5[events_with_ar_ff5["Industry"] == sector]

    mean_ar = df_sec[ar_cols].mean()

    ax.plot(x_vals, mean_ar.values, marker="o")
    ax.axhline(0, linewidth=1)
    ax.set_title(sector)
    ax.set_xlabel("Jour relatif à l'événement")
    ax.set_ylabel("AR moyen (FF5)")
    ax.grid(alpha=0.3)
    ax.set_xticks(x_vals)

plt.suptitle("Rendements anormaux moyens par secteur\nFenêtre [-5 ; +5] – Modèle Fama-French 5 facteurs",
             fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# %%
