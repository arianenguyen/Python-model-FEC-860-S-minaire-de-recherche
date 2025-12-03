#%%
from scipy import stats
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# %%
file = "Datapython.xlsx"

equity = pd.read_excel(file, sheet_name="LOG_RETURN_EQUITY")
events = pd.read_excel(file, sheet_name="DATE_EVENT")
fama3 = pd.read_excel(file, sheet_name="FAMA3")

#%%
equity_long = equity.melt(
    id_vars="Dates",     
    var_name="ticker",   
    value_name="Ri"    
)

print("Format long des rendements actions :")
display(equity_long.head())

#%%
equity["Dates"] = pd.to_datetime(equity["Dates"])
events["Date"] = pd.to_datetime(events["Date"])
fama3["DATE"] = pd.to_datetime(fama3["DATE"])

#%%
fama3 = fama3.rename(columns={
    "Mkt-Rf": "MKT_RF", 
    "SMB": "SMB",
    "HML": "HML",
    "RF": "Rf"         
})

#%%
data_ff3 = equity_long.merge(
    fama3[["DATE", "MKT_RF", "SMB", "HML", "Rf"]],
    left_on="Dates", right_on="DATE",
    how="left"
)

print("Données fusionnées pour le Modèle 2 (Fama-French 3 facteurs) :")
display(data_ff3.head())

# %%
def compute_betas_fama3_for_row(row):
    ticker = row["Ticker"]
    event_date = row["Date"]

    cols = ["alpha_ff3", "beta_mkt_ff3", "beta_smb_ff3", "beta_hml_ff3"]

    d = data_ff3[data_ff3["ticker"] == ticker].copy()
    if d.empty:
        return pd.Series([np.nan]*4, index=cols)

    d = d.sort_values("Dates").reset_index(drop=True)

    div_decla = d["Dates"] == event_date
    if not div_decla.any():
        return pd.Series([np.nan]*4, index=cols)

    event_idx = div_decla[div_decla].index[0]

    start_idx = max(0, event_idx - 250)
    end_idx   = event_idx - 30

    if end_idx <= start_idx:
        return pd.Series([np.nan]*4, index=cols)

    window = d.iloc[start_idx:end_idx + 1].dropna(
        subset=["Ri", "MKT_RF", "SMB", "HML", "Rf"]
    )

    if len(window) < 30:
        return pd.Series([np.nan]*4, index=cols)

    window = window.copy()
    window["excess_i_ff3"] = window["Ri"] - window["Rf"]

    X = window[["MKT_RF", "SMB", "HML"]]
    X = sm.add_constant(X)
    y = window["excess_i_ff3"]

    model = sm.OLS(y, X).fit()

    return pd.Series(
        [model.params["const"],
         model.params["MKT_RF"],
         model.params["SMB"],
         model.params["HML"]],
        index=cols,
    )

#%%
betas_ff3 = events.apply(compute_betas_fama3_for_row, axis=1)
events_ff3 = pd.concat([events, betas_ff3], axis=1)

print("Événements avec bêtas Fama-French (Modèle 2) :")
display(events_ff3[["Ticker", "Date", "alpha_ff3", "beta_mkt_ff3", "beta_smb_ff3", "beta_hml_ff3"]])

#%%
def compute_ar_ff3_for_row(row):
    """
    Calcule les rendements anormaux (AR) sur [-5, +5] jours ouvrables
    selon le modèle Fama-French 3 facteurs.
    """
    ticker = row["Ticker"]
    event_date = row["Date"]

    alpha = row["alpha_ff3"]
    beta_mkt = row["beta_mkt_ff3"]
    beta_smb = row["beta_smb_ff3"]
    beta_hml = row["beta_hml_ff3"]

    if pd.isna(alpha):
        return pd.Series([np.nan]*11,
            index=[f"AR_{i}" for i in range(-5,6)]
        )

    d = data_ff3[data_ff3["ticker"] == ticker].copy()

    if d.empty:
        return pd.Series([np.nan]*11, index=[f"AR_{i}" for i in range(-5,6)])

    d = d.sort_values("Dates").reset_index(drop=True)

    mask_event = d["Dates"] == event_date
    if not mask_event.any():
        return pd.Series([np.nan]*11, index=[f"AR_{i}" for i in range(-5,6)])

    event_idx = mask_event[mask_event].index[0]

    start_idx = event_idx - 5
    end_idx = event_idx + 5

    window = d.iloc[max(start_idx,0): end_idx+1]

    window = window.reset_index(drop=True)
    rel_idx = list(range(start_idx, end_idx+1))

    if len(window) != len(rel_idx):
        return pd.Series([np.nan]*11, index=[f"AR_{i}" for i in range(-5,6)])

    expected = (
        window["Rf"]
        + alpha
        + beta_mkt * window["MKT_RF"]
        + beta_smb * window["SMB"]
        + beta_hml * window["HML"]
    )

    AR = window["Ri"] - expected

    AR.index = [f"AR_{i}" for i in range(-5,6)]
    return AR

#%%
ar_ff3 = events_ff3.apply(compute_ar_ff3_for_row, axis=1)
events_with_ar_ff3 = pd.concat([events_ff3, ar_ff3], axis=1)

display(events_with_ar_ff3.head())


# %%
ar_cols = [f"AR_{i}" for i in range(-5, 6)] 
events_with_ar_ff3["CAR_-5_+5"] = events_with_ar_ff3[ar_cols].sum(axis=1)

display(events_with_ar_ff3[["Ticker", "Date", "CAR_-5_+5"]])

# %%
car_sector_mean = events_with_ar_ff3.groupby("Industry")["CAR_-5_+5"].mean()

print("CAR moyen par secteur (-5 à +5 jours) :")
display(car_sector_mean)

# %%
results = {}

for sector in car_sector_mean.index:

    vals = events_with_ar_ff3.loc[
        events_with_ar_ff3["Industry"] == sector, "CAR_-5_+5"
    ].dropna()

    t_stat, p_value = stats.ttest_1samp(vals, 0)

    results[sector] = {
        "mean_CAR": vals.mean(),
        "t_stat": t_stat,
        "p_value": p_value,
        "n": len(vals)
    }

car_sector_test = pd.DataFrame(results).T

print("Test : le CAR moyen par secteur est-il ≠ 0 ?")
display(car_sector_test)

# %%
ar_cols = [f"AR_{i}" for i in range(-5, 6)]
sectors = sorted(events_with_ar_ff3["Industry"].unique())

print("Secteurs trouvés :", sectors)

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
axes = axes.ravel()

x_vals = list(range(-5, 6)) 

for ax, sector in zip(axes, sectors):
    df_sec = events_with_ar_ff3[events_with_ar_ff3["Industry"] == sector]

    mean_ar = df_sec[ar_cols].mean()

    ax.plot(x_vals, mean_ar.values, marker="o")
    ax.axhline(0, linewidth=1)
    ax.set_title(sector)
    ax.set_xlabel("Jour relatif à l'événement")
    ax.set_ylabel("AR moyen")
    ax.grid(alpha=0.3)
    ax.set_xticks(x_vals)

plt.suptitle("Rendements anormaux moyens par secteur\nFenêtre [-5 ; +5]", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()