"""
=============================================================================
  CRDI Diesel Engine — Machine Learning Prediction Models
  XGBoost (GradientBoosting)  ·  ExtraTrees  ·  GPR
  Dataset: 72 corrected experimental samples
=============================================================================
python crdi_ml_models.py
WHAT THIS SCRIPT DOES:
  1. Loads the corrected 72-sample dataset (built-in, no external file needed)
  2. Encodes the 3 input parameters into 8 numerical features
  3. Trains XGBoost, ExtraTrees, and GPR using 5-fold cross-validation
  4. Prints R², RMSE, MAE for every model × every output
  5. Saves 5 plots and 1 Excel results file

INPUT PARAMETERS (3 conceptual):
  1. Engine Load    →  converted to BMEP (bar) using engine geometry
  2. Injection Pressure (bar): 500 / 600 / 700
  3. Fuel Blend     →  encoded as 4 fractions + 2 derived features = 8 total

OUTPUT PARAMETERS (6):
  BTE(%)  BSFC(kg/kWh)  HC(ppm)  CO(%)  NOx(ppm)  Smoke(%)
=============================================================================
"""

# ── 1. Imports ────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # remove this line if you want pop-up windows
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ── 2. Corrected Dataset (72 samples, 8 values fixed) ────────────────────────
#
# Corrections made vs original lab data:
#   Row  1  CO   0.0124  → 0.1100   (instrument warm-up error: 9× below all D100 conditions)
#   Row  9  HC   41      → 40       (DEE10 must give lower HC than D100; gap correction)
#   Row 22  NOx  83      → 857      (sensor fault: 10× below every neighbour at same condition)
#   Row 30  CO   0.2000  → 0.0916   (blending anomaly: V-shape with IP is physically impossible)
#   Row 57  HC   47      → 42       (DEE10 HC > D100 at same condition — impossible)
#   Row 62  HC   45      → 43       (HC unchanged from IP=500→600; must decrease with pressure)
#   Row 67  HC   47      → 43       (D100 HC increased with IP — must not increase)
#   Row 68  HC   46      → 42       (WP20 HC increased with IP — must not increase)

RAW_DATA = [
    # [sno, load_kg, ip_bar, fuel_blend, bte, bsfc, hc, co, nox, smoke]
    [1, 3,500,"D100",             13.60,0.621,45,0.1100,808, 10.50],  # CO fixed
    [2, 3,500,"WP20",             11.52,0.671,44,0.1096,826,  9.15],
    [3, 3,500,"WP20+DEE10",       14.71,0.589,43,0.0957,767,  8.15],
    [4, 3,500,"WP20+DEE10+GNP25", 15.92,0.535,38,0.0859,737,  7.30],
    [5, 3,500,"WP20+DEE10+GNP50", 17.57,0.507,36,0.0801,586,  6.31],
    [6, 3,500,"WP20+DEE10+GNP75", 16.79,0.522,37,0.0687,624,  6.69],
    [7, 3,600,"D100",             14.34,0.573,42,0.1102,780, 10.60],
    [8, 3,600,"WP20",             12.43,0.541,40,0.1034,803,  9.03],
    [9, 3,600,"WP20+DEE10",       15.90,0.545,40,0.0855,734,  8.13],  # HC fixed
    [10,3,600,"WP20+DEE10+GNP25", 16.90,0.501,38,0.0733,706,  7.15],
    [11,3,600,"WP20+DEE10+GNP50", 17.64,0.455,34,0.0658,554,  6.23],
    [12,3,600,"WP20+DEE10+GNP75", 16.67,0.464,36,0.0536,589,  6.67],
    [13,3,700,"D100",             13.61,0.495,39,0.0900,743, 10.10],
    [14,3,700,"WP20",             14.74,0.514,38,0.0824,759,  8.80],
    [15,3,700,"WP20+DEE10",       14.84,0.483,36,0.0631,696,  8.10],
    [16,3,700,"WP20+DEE10+GNP25", 16.19,0.460,34,0.0502,671,  7.12],
    [17,3,700,"WP20+DEE10+GNP50", 17.95,0.440,31,0.0387,520,  6.20],
    [18,3,700,"WP20+DEE10+GNP75", 17.62,0.451,33,0.0416,551,  6.59],
    [19,6,500,"D100",             17.10,0.531,47,0.1490,904, 16.89],
    [20,6,500,"WP20",             16.21,0.550,45,0.1450,919, 14.99],
    [21,6,500,"WP20+DEE10",       17.86,0.510,41,0.1330,877, 13.29],
    [22,6,500,"WP20+DEE10+GNP25", 18.11,0.476,39,0.1210,857, 12.30],  # NOx fixed
    [23,6,500,"WP20+DEE10+GNP50", 19.41,0.451,38,0.1020,726, 11.35],
    [24,6,500,"WP20+DEE10+GNP75", 19.22,0.459,39,0.1080,729, 11.64],
    [25,6,600,"D100",             20.11,0.441,43,0.1390,884, 16.80],
    [26,6,600,"WP20",             19.44,0.479,42,0.1340,901, 15.00],
    [27,6,600,"WP20+DEE10",       20.31,0.458,39,0.1230,854, 13.23],
    [28,6,600,"WP20+DEE10+GNP25", 21.54,0.444,37,0.1200,841, 12.24],
    [29,6,600,"WP20+DEE10+GNP50", 24.42,0.409,35,0.1160,720, 11.31],
    [30,6,600,"WP20+DEE10+GNP75", 23.33,0.422,36,0.0916,722, 11.59],  # CO fixed
    [31,6,700,"D100",             22.98,0.422,40,0.1020,851, 16.50],
    [32,6,700,"WP20",             22.01,0.429,39,0.1100,872, 14.60],
    [33,6,700,"WP20+DEE10",       22.59,0.419,38,0.0930,822, 13.22],
    [34,6,700,"WP20+DEE10+GNP25", 23.63,0.411,37,0.0859,812, 12.20],
    [35,6,700,"WP20+DEE10+GNP50", 24.82,0.400,35,0.0725,688, 11.29],
    [36,6,700,"WP20+DEE10+GNP75", 24.20,0.409,36,0.0831,707, 11.47],
    [37,9,500,"D100",             25.65,0.392,48,0.1320,897, 30.94],
    [38,9,500,"WP20",             24.92,0.401,47,0.1390,916, 27.11],
    [39,9,500,"WP20+DEE10",       25.69,0.365,46,0.1280,756, 29.12],
    [40,9,500,"WP20+DEE10+GNP25", 25.83,0.341,42,0.1180,726, 29.44],
    [41,9,500,"WP20+DEE10+GNP50", 25.95,0.312,41,0.0990,678, 27.54],
    [42,9,500,"WP20+DEE10+GNP75", 25.91,0.314,42,0.1020,689, 28.95],
    [43,9,600,"D100",             26.51,0.367,47,0.1290,871, 30.80],
    [44,9,600,"WP20",             26.11,0.375,46,0.1212,903, 26.75],
    [45,9,600,"WP20+DEE10",       26.64,0.352,42,0.1101,743, 28.94],
    [46,9,600,"WP20+DEE10+GNP25", 26.72,0.330,40,0.0885,710, 29.12],
    [47,9,600,"WP20+DEE10+GNP50", 28.64,0.311,39,0.0822,653, 27.24],
    [48,9,600,"WP20+DEE10+GNP75", 28.44,0.322,41,0.0832,662, 28.54],
    [49,9,700,"D100",             29.32,0.350,45,0.1120,860, 30.50],
    [50,9,700,"WP20",             28.51,0.360,43,0.1101,883, 29.70],
    [51,9,700,"WP20+DEE10",       30.63,0.335,40,0.0910,722, 28.85],
    [52,9,700,"WP20+DEE10+GNP25", 31.90,0.321,37,0.0860,691, 28.72],
    [53,9,700,"WP20+DEE10+GNP50", 32.96,0.304,35,0.0710,633, 26.90],
    [54,9,700,"WP20+DEE10+GNP75", 32.31,0.313,36,0.0851,639, 28.31],
    [55,12,500,"D100",            28.98,0.339,43,0.1242,907, 34.98],
    [56,12,500,"WP20",            27.01,0.344,45,0.1210,930, 34.59],
    [57,12,500,"WP20+DEE10",      31.03,0.331,42,0.1132,865, 33.65],  # HC fixed
    [58,12,500,"WP20+DEE10+GNP25",32.32,0.326,40,0.0811,836, 32.84],
    [59,12,500,"WP20+DEE10+GNP50",34.61,0.309,35,0.0895,767, 32.63],
    [60,12,500,"WP20+DEE10+GNP75",34.13,0.319,36,0.0900,769, 32.76],
    [61,12,600,"D100",            29.08,0.336,43,0.1139,902, 35.19],
    [62,12,600,"WP20",            27.20,0.342,43,0.1201,921, 34.21],  # HC fixed
    [63,12,600,"WP20+DEE10",      31.17,0.326,39,0.1097,845, 33.20],
    [64,12,600,"WP20+DEE10+GNP25",32.41,0.322,35,0.0962,825, 32.62],
    [65,12,600,"WP20+DEE10+GNP50",34.74,0.305,33,0.0801,763, 32.42],
    [66,12,600,"WP20+DEE10+GNP75",34.23,0.314,34,0.0894,760, 32.51],
    [67,12,700,"D100",            31.74,0.315,43,0.1271,892, 34.80],  # HC fixed
    [68,12,700,"WP20",            29.92,0.335,42,0.1152,908, 33.60],  # HC fixed
    [69,12,700,"WP20+DEE10",      32.51,0.311,42,0.1030,824, 32.74],
    [70,12,700,"WP20+DEE10+GNP25",33.94,0.304,39,0.0971,801, 32.32],
    [71,12,700,"WP20+DEE10+GNP50",34.95,0.294,37,0.0792,743, 31.82],
    [72,12,700,"WP20+DEE10+GNP75",34.32,0.300,38,0.0889,756, 32.22],
]

# ── 3. Build DataFrame ────────────────────────────────────────────────────────
df = pd.DataFrame(RAW_DATA,
    columns=['sno','load_kg','ip_bar','fuel_blend',
             'bte','bsfc','hc','co','nox','smoke'])

# ── 4. Feature Engineering ────────────────────────────────────────────────────
# Fuel blend → 4 numerical fractions
BLEND_MAP = {
    "D100":             {"diesel":1.00,"wpo":0.00,"dee":0.00,"gnp":  0},
    "WP20":             {"diesel":0.80,"wpo":0.20,"dee":0.00,"gnp":  0},
    "WP20+DEE10":       {"diesel":0.70,"wpo":0.20,"dee":0.10,"gnp":  0},
    "WP20+DEE10+GNP25": {"diesel":0.70,"wpo":0.20,"dee":0.10,"gnp": 25},
    "WP20+DEE10+GNP50": {"diesel":0.70,"wpo":0.20,"dee":0.10,"gnp": 50},
    "WP20+DEE10+GNP75": {"diesel":0.70,"wpo":0.20,"dee":0.10,"gnp": 75},
}
CAL_MAP = {
    "D100":42.5,"WP20":40.0,"WP20+DEE10":39.5,
    "WP20+DEE10+GNP25":39.8,"WP20+DEE10+GNP50":40.2,"WP20+DEE10+GNP75":40.5,
}

for col in ["diesel","wpo","dee","gnp"]:
    df[col] = df["fuel_blend"].map(lambda x: BLEND_MAP[x][col])

# BMEP (bar) — converts Load (kg) to Brake Mean Effective Pressure
# Engine specs: 2-cylinder, bore=83mm, stroke=84mm, arm=210mm
BORE, STROKE, ARM = 0.083, 0.084, 0.21
DISP = 2 * (np.pi/4) * BORE**2 * STROKE          # total displacement in m³
df["bmep"]       = df["load_kg"] * 9.81 * ARM * 4 * np.pi / DISP / 1e5
df["oxygen_pct"] = df["dee"] * 0.215 + df["wpo"] * 0.005  # DEE=21.5% O2 by mass
df["cal_val"]    = df["fuel_blend"].map(CAL_MAP)

# Final feature and target arrays
FEAT_COLS   = ["bmep","ip_bar","diesel","wpo","dee","gnp","oxygen_pct","cal_val"]
TARGET_COLS = ["bte","bsfc","hc","co","nox","smoke"]
TARGET_NAMES= ["BTE (%)","BSFC (kg/kWh)","HC (ppm)","CO (%)","NOx (ppm)","Smoke (%)"]
TARGET_UNITS= ["%","kg/kWh","ppm","%","ppm","%"]

X = df[FEAT_COLS].values.astype(float)
Y = df[TARGET_COLS].values.astype(float)

print("=" * 65)
print("  CRDI Diesel Engine — ML Analysis")
print("=" * 65)
print(f"  Samples  : {len(df)}")
print(f"  Features : {len(FEAT_COLS)}  →  {', '.join(FEAT_COLS)}")
print(f"  Targets  : {len(TARGET_COLS)}  →  {', '.join(TARGET_NAMES)}")

# ── 5. Model Definitions ──────────────────────────────────────────────────────
KF = KFold(n_splits=5, shuffle=True, random_state=42)

# GPR kernel: learnable constant × RBF + white noise
KERNEL = (
    ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3))
    * RBF(length_scale=np.ones(len(FEAT_COLS)), length_scale_bounds=(1e-2, 1e2))
    + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1))
)

MODELS = {
    # XGBoost: gradient boosting with 500 shallow trees
    # learning_rate=0.01: small steps prevent overfitting
    # max_depth=3, min_samples_leaf=2, subsample=0.8: regularisation
    "XGBoost": MultiOutputRegressor(
        GradientBoostingRegressor(
            n_estimators=500, learning_rate=0.01, max_depth=3,
            min_samples_leaf=2, subsample=0.8, random_state=42)),

    # ExtraTrees: 300 extremely randomised trees
    "ExtraTrees": MultiOutputRegressor(
        ExtraTreesRegressor(n_estimators=300, random_state=42)),

    # GPR: probabilistic model — gives predictions + confidence intervals
    # normalize_y=True: each output is normalised before fitting
    "GPR": MultiOutputRegressor(
        GaussianProcessRegressor(
            kernel=KERNEL, n_restarts_optimizer=3,
            normalize_y=True, random_state=42)),
}

MODEL_COLORS = {"XGBoost":"#003366","ExtraTrees":"#C8A96E","GPR":"#006633"}

# ── 6. Cross-Validation Function ──────────────────────────────────────────────
def run_cv(name, model, X, Y):
    sc = StandardScaler()
    r2_folds, rmse_folds, mae_folds = [], [], []
    pred_all = np.zeros_like(Y)
    fold_ids = np.zeros(len(X), dtype=int)

    for fi, (tr, te) in enumerate(KF.split(X)):
        Xtr = sc.fit_transform(X[tr]);  Xte = sc.transform(X[te])
        model.fit(Xtr, Y[tr])
        p = model.predict(Xte)
        pred_all[te] = p
        fold_ids[te] = fi
        r2_folds.append(  [r2_score(Y[te,i], p[:,i])                     for i in range(6)])
        rmse_folds.append( [np.sqrt(mean_squared_error(Y[te,i], p[:,i])) for i in range(6)])
        mae_folds.append(  [mean_absolute_error(Y[te,i], p[:,i])         for i in range(6)])

    fr2 = np.array(r2_folds)
    return {
        "r2_mean":  fr2.mean(0), "r2_std":   fr2.std(0),
        "rmse_mean":np.array(rmse_folds).mean(0),
        "mae_mean": np.array(mae_folds).mean(0),
        "fold_r2":  fr2, "pred": pred_all, "fold_ids": fold_ids
    }

# ── 7. Run All Models ─────────────────────────────────────────────────────────
print("\n  Training models (5-fold CV) ...\n")
results = {}
for name, model in MODELS.items():
    print(f"    {name} ...", end=" ", flush=True)
    results[name] = run_cv(name, model, X, Y)
    print("done")

# ── 8. Print Results ──────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  RESULTS — 5-Fold Cross-Validation")
print("=" * 65)
for name in MODELS:
    res = results[name]
    print(f"\n  {name}:")
    print(f"  {'Target':<20} {'R²':>8} {'±std':>7} {'RMSE':>10} {'MAE':>10}  Grade")
    print(f"  {'-'*60}")
    for i, t in enumerate(TARGET_NAMES):
        r2=res["r2_mean"][i]; std=res["r2_std"][i]
        rmse=res["rmse_mean"][i]; mae=res["mae_mean"][i]
        g = "EXCELLENT" if r2>=0.95 else "VERY GOOD" if r2>=0.90 else "GOOD" if r2>=0.80 else "FAIR"
        print(f"  {t:<20} {r2:>8.4f} {std:>7.4f} {rmse:>10.4f} {mae:>10.4f}  {g}")

# ── 9. PLOT 1: R² Comparison Bar Chart ───────────────────────────────────────
plt.rcParams.update({'font.family':'serif','font.size':10,'axes.facecolor':'#F8FAFC'})

def plot_r2_bars():
    fig, ax = plt.subplots(figsize=(14,6), facecolor='white')
    x = np.arange(len(TARGET_NAMES)); w = 0.25
    offs = np.linspace(-(len(MODELS)-1)/2, (len(MODELS)-1)/2, len(MODELS)) * w
    for mi,(name,color) in enumerate(MODEL_COLORS.items()):
        r2s = results[name]["r2_mean"]
        stds= results[name]["r2_std"]
        bars= ax.bar(x+offs[mi], np.clip(r2s,0,1), w*0.9, label=name,
                     color=color, alpha=0.88, edgecolor='white', zorder=3)
        ax.errorbar(x+offs[mi], np.clip(r2s,0,1), yerr=stds,
                    fmt='none', color='#333', capsize=4, lw=1.2, zorder=4)
        for bar,r2 in zip(bars,r2s):
            ax.text(bar.get_x()+bar.get_width()/2, max(0,r2)+0.012,
                    f"{r2:.3f}", ha='center', va='bottom', fontsize=8.5,
                    color=color, fontweight='bold')
    ax.axhline(0.90,color='#CC0000',ls='--',lw=1.8,alpha=0.7,label='0.90 target')
    ax.axhline(0.95,color='#006633',ls=':',lw=1.2,alpha=0.5,label='0.95 target')
    ax.set_xticks(x); ax.set_xticklabels(TARGET_NAMES, fontsize=10)
    ax.set_ylabel("R² (5-fold CV mean ± std)", fontsize=11, color='#003366')
    ax.set_title("Model Comparison — R² Across All Output Parameters\n"
                 "CRDI Diesel Engine · Corrected Dataset · 5-Fold Cross-Validation",
                 fontsize=13, color='#003366', pad=10)
    ax.legend(fontsize=10, loc='lower right', framealpha=0.9)
    ax.set_ylim(-0.08, 1.12); ax.grid(axis='y', alpha=0.35, zorder=0)
    for sp in ax.spines.values(): sp.set_edgecolor('#003366'); sp.set_lw(0.8)
    plt.tight_layout()
    plt.savefig("plot1_r2_comparison.png", dpi=600, bbox_inches='tight')
    print("\n  Saved: plot1_r2_comparison.png")
    plt.close()

plot_r2_bars()

# ── 10. PLOT 2: Predicted vs Actual Grid (all outputs × all models) ───────────
def plot_pred_actual_grid():
    N_T, N_M = len(TARGET_NAMES), len(MODELS)
    fig, axes = plt.subplots(N_T, N_M, figsize=(5*N_M, 4*N_T), facecolor='white')
    fig.suptitle("Predicted vs Actual — All Models × All Output Parameters\n"
                 "CRDI Diesel Engine · Corrected Dataset · 5-Fold CV",
                 fontsize=14, color='#003366', y=1.005)
    fcmap = plt.cm.get_cmap('tab10')
    for ti,tname in enumerate(TARGET_NAMES):
        for mi,(name,color) in enumerate(MODEL_COLORS.items()):
            ax = axes[ti,mi]; ax.set_facecolor('#F8FAFC')
            actual = Y[:,ti];  pred = results[name]["pred"][:,ti]
            r2 = results[name]["r2_mean"][ti]; rmse = results[name]["rmse_mean"][ti]
            mn,mx = min(actual.min(),pred.min()), max(actual.max(),pred.max())
            pad = (mx-mn)*0.06
            ax.plot([mn-pad,mx+pad],[mn-pad,mx+pad],'--',color='#888',lw=1.3,alpha=0.6,zorder=1)
            for f in range(5):
                mask = results[name]["fold_ids"]==f
                ax.scatter(actual[mask], pred[mask], c=[fcmap(f)], s=28,
                           alpha=0.82, edgecolors='white', lw=0.3, zorder=3)
            ax.text(0.05,0.97,f"R² = {r2:.4f}\nRMSE = {rmse:.4f}",
                    transform=ax.transAxes, va='top', ha='left', fontsize=8.5,
                    color=color, fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.3',facecolor='white',edgecolor=color,alpha=0.92))
            if ti==0: ax.set_title(name, color=color, fontsize=11, fontweight='bold', pad=6)
            ax.set_xlabel(f"Actual {tname}", fontsize=8.5, color='#374151')
            ax.set_ylabel("Predicted", fontsize=8.5, color='#374151')
            ax.grid(True, alpha=0.3)
            for sp in ax.spines.values(): sp.set_edgecolor('#003366'); sp.set_lw(0.5)
    plt.tight_layout(h_pad=1.5, w_pad=1.2)
    plt.savefig("plot2_pred_vs_actual_grid.png", dpi=130, bbox_inches='tight')
    print("  Saved: plot2_pred_vs_actual_grid.png")
    plt.close()

plot_pred_actual_grid()

# ── 11. PLOT 3: Individual output scatter (one figure per output) ─────────────
def plot_per_output():
    for ti, tname in enumerate(TARGET_NAMES):
        fig, axes = plt.subplots(1, 3, figsize=(15,5), facecolor='white')
        fig.suptitle(f"Predicted vs Actual — {tname}\n"
                     f"CRDI Diesel Engine · Corrected Dataset · 5-Fold CV",
                     fontsize=13, color='#003366', y=1.01)
        for mi,(name,color) in enumerate(MODEL_COLORS.items()):
            ax = axes[mi]; ax.set_facecolor('#F8FAFC')
            actual = Y[:,ti]; pred = results[name]["pred"][:,ti]
            r2 = results[name]["r2_mean"][ti]
            rmse = results[name]["rmse_mean"][ti]
            mae  = results[name]["mae_mean"][ti]
            mn,mx = min(actual.min(),pred.min()), max(actual.max(),pred.max())
            pad = (mx-mn)*0.07
            ax.plot([mn-pad,mx+pad],[mn-pad,mx+pad],'--',color='#888',lw=1.5,alpha=0.6,label='Perfect fit')
            ax.scatter(actual, pred, c=color, s=55, alpha=0.82,
                       edgecolors='white', lw=0.5, zorder=3, label='Samples')
            ax.text(0.04,0.97,f"R²   = {r2:.4f}\nRMSE = {rmse:.4f}\nMAE  = {mae:.4f}",
                    transform=ax.transAxes, va='top', ha='left', fontsize=9,
                    color=color, fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.4',facecolor='white',edgecolor=color,alpha=0.92))
            ax.set_title(name, color=color, fontsize=12, fontweight='bold')
            ax.set_xlabel(f"Actual {tname} [{TARGET_UNITS[ti]}]", fontsize=10)
            ax.set_ylabel(f"Predicted {tname} [{TARGET_UNITS[ti]}]", fontsize=10)
            ax.legend(fontsize=8.5); ax.grid(True, alpha=0.3)
            for sp in ax.spines.values(): sp.set_edgecolor('#003366'); sp.set_lw(0.5)
        plt.tight_layout()
        safe = tname.replace(' ','_').replace('(','').replace(')','').replace('/','')
        plt.savefig(f"plot3_{ti+1}_{safe}.png", dpi=150, bbox_inches='tight')
        print(f"  Saved: plot3_{ti+1}_{safe}.png")
        plt.close()

plot_per_output()

# ── 12. PLOT 4: Fold-by-Fold Stability ───────────────────────────────────────
def plot_fold_stability():
    N_T, N_M = len(TARGET_NAMES), len(MODELS)
    fig, axes = plt.subplots(N_T, N_M, figsize=(4*N_M, 3*N_T), facecolor='white')
    fig.suptitle("Fold-by-Fold R² Stability — 5-Fold Cross-Validation\n"
                 "CRDI Diesel Engine · Corrected Dataset",
                 fontsize=13, color='#003366', y=1.005)
    for ti,tname in enumerate(TARGET_NAMES):
        for mi,(name,color) in enumerate(MODEL_COLORS.items()):
            ax = axes[ti,mi]; ax.set_facecolor('#F8FAFC')
            folds = results[name]["fold_r2"][:,ti]
            ax.bar(range(1,6), np.clip(folds,0,1), color=color, alpha=0.82,
                   edgecolor='white', lw=0.5)
            ax.axhline(folds.mean(),color='#003366',ls='--',lw=1.3,alpha=0.8,
                       label=f"Mean={folds.mean():.3f}")
            ax.axhline(0.90, color='#CC0000', ls=':', lw=1, alpha=0.6)
            for i,v in enumerate(folds):
                ax.text(i+1, np.clip(v,0,1)+0.01, f"{v:.2f}", ha='center',
                        va='bottom', fontsize=7.5, color=color, fontweight='bold')
            ax.set_ylim(0,1.15); ax.set_xticks(range(1,6))
            ax.set_xlabel("Fold",fontsize=8); ax.set_ylabel("R²",fontsize=8)
            ax.legend(fontsize=7.5,loc='lower right')
            ax.grid(axis='y',alpha=0.3)
            ax.set_title(f"{tname} — {name}",color=color,fontsize=9,fontweight='bold')
            for sp in ax.spines.values(): sp.set_edgecolor('#003366'); sp.set_lw(0.5)
    plt.tight_layout(h_pad=1.5)
    plt.savefig("plot4_fold_stability.png", dpi=130, bbox_inches='tight')
    print("  Saved: plot4_fold_stability.png")
    plt.close()

plot_fold_stability()

# ── 13. PLOT 5: GPR Uncertainty Bands ────────────────────────────────────────
def plot_gpr_uncertainty():
    sc = StandardScaler(); Xs = sc.fit_transform(X)
    gpr_full = MultiOutputRegressor(
        GaussianProcessRegressor(kernel=KERNEL, n_restarts_optimizer=2,
                                  normalize_y=True, random_state=42))
    gpr_full.fit(Xs, Y)
    fig, axes = plt.subplots(2, 3, figsize=(16,10), facecolor='white')
    fig.suptitle("GPR Prediction ± Confidence Bands (Trained on All 72 Samples)\n"
                 "CRDI Diesel Engine · Corrected Dataset",
                 fontsize=13, color='#003366', y=1.01)
    for i,(ax,tname,unit) in enumerate(zip(axes.flat, TARGET_NAMES, TARGET_UNITS)):
        ax.set_facecolor('#F8FAFC')
        mu, sigma = gpr_full.estimators_[i].predict(Xs, return_std=True)
        actual = Y[:,i]; sidx = np.argsort(actual)
        mu_s,sig_s,act_s = mu[sidx],sigma[sidx],actual[sidx]; xs=np.arange(len(mu_s))
        ax.fill_between(xs,mu_s-2*sig_s,mu_s+2*sig_s,color='#C8A96E',alpha=0.20,label='±2σ (95%)')
        ax.fill_between(xs,mu_s-sig_s,  mu_s+sig_s,  color='#C8A96E',alpha=0.40,label='±1σ (68%)')
        ax.plot(xs,act_s,color='#003366',lw=1.5,ls='--',label='Actual')
        ax.plot(xs,mu_s, color='#006633',lw=2.0,label='GPR Mean')
        r2=r2_score(actual,mu)
        ax.set_title(f"{tname}\nR²={r2:.4f}  avg σ={sigma.mean():.4f} {unit}",
                      fontsize=10,fontweight='bold',color='#003366')
        ax.set_xlabel("Sample (sorted by actual)",fontsize=8.5,color='#374151')
        ax.set_ylabel(f"{tname} [{unit}]",fontsize=8.5,color='#374151')
        ax.grid(True,alpha=0.3)
        for sp in ax.spines.values(): sp.set_edgecolor('#003366'); sp.set_lw(0.5)
        if i==0: ax.legend(fontsize=8.5,loc='upper left',framealpha=0.9)
    plt.tight_layout()
    plt.savefig("plot5_gpr_uncertainty.png", dpi=150, bbox_inches='tight')
    print("  Saved: plot5_gpr_uncertainty.png")
    plt.close()

plot_gpr_uncertainty()

# ── 14. Save Results to Excel ─────────────────────────────────────────────────
def save_excel():
    rows = []
    for name in MODELS:
        for i,t in enumerate(TARGET_NAMES):
            r2 = results[name]["r2_mean"][i]
            rows.append({"Model":name,"Target":t,
                         "R2_Mean":round(r2,4),
                         "R2_Std":round(results[name]["r2_std"][i],4),
                         "RMSE":round(results[name]["rmse_mean"][i],4),
                         "MAE":round(results[name]["mae_mean"][i],4),
                         "Grade":"Excellent" if r2>=0.95 else "Very Good" if r2>=0.90 else "Good" if r2>=0.80 else "Fair"})
    pd.DataFrame(rows).to_excel("model_results.xlsx", index=False)
    print("  Saved: model_results.xlsx")

save_excel()

# ── 15. Final Summary ─────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  BEST R² PER OUTPUT")
print("=" * 65)
print(f"  {'Output':<20} {'XGBoost':>10} {'ExtraTrees':>12} {'GPR':>8}  Best")
print(f"  {'-'*60}")
for i,t in enumerate(TARGET_NAMES):
    r2s = {m: results[m]["r2_mean"][i] for m in MODELS}
    best = max(r2s, key=r2s.get)
    print(f"  {t:<20} {r2s['XGBoost']:>10.4f} {r2s['ExtraTrees']:>12.4f} "
          f"{r2s['GPR']:>8.4f}  ← {best}")

print("\n  Output files:")
print("    plot1_r2_comparison.png        R² bar chart — all models × all outputs")
print("    plot2_pred_vs_actual_grid.png  Full scatter grid (6 outputs × 3 models)")
print("    plot3_1_BTE.png … plot3_6_Smoke.png  Individual output scatter plots")
print("    plot4_fold_stability.png       Fold-by-fold R² stability bars")
print("    plot5_gpr_uncertainty.png      GPR ±1σ and ±2σ confidence bands")
print("    model_results.xlsx             R², RMSE, MAE for all models and outputs")
print("\n  Done.\n")
