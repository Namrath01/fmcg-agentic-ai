"""
Generate output charts as PNG files for GitHub.
Reads from outputs/*.csv and saves to outputs/charts/.
"""
import sys
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT  = Path("outputs")
CHARTS = OUT / "charts"
CHARTS.mkdir(exist_ok=True)

BLUE   = "#1a73e8"
GREEN  = "#34a853"
ORANGE = "#f9ab00"
RED    = "#ea4335"
PURPLE = "#7b2ff7"
GREY   = "#e8eaed"
BG     = "#ffffff"
TEXT   = "#1a1a2e"

def save(fig, name):
    path = CHARTS / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  saved {name}")

# ── 1. Demand Forecast — top 6 SKUs ──────────────────────────────────────
df_fc = pd.read_csv(OUT / "01_demand_forecast.csv", parse_dates=["Date"])
top_skus = df_fc.groupby("SkuId")["ForecastQty"].sum().nlargest(6).index.tolist()

fig, axes = plt.subplots(2, 3, figsize=(15, 7), facecolor=BG)
fig.suptitle("Demand Intelligence — 30-Day Forecast (Top 6 SKUs)", fontsize=14, color=TEXT, fontweight="bold", y=1.01)
for ax, sku in zip(axes.flat, top_skus):
    d = df_fc[df_fc["SkuId"] == sku].sort_values("Date")
    ax.fill_between(d["Date"], d["CI_Lower"], d["CI_Upper"], alpha=0.18, color=BLUE)
    ax.plot(d["Date"], d["ForecastQty"], color=BLUE, linewidth=2)
    ax.set_title(sku, fontsize=10, color=TEXT)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=30, labelsize=7)
    ax.tick_params(axis="y", labelsize=8)
    ax.spines[["top","right"]].set_visible(False)
    ax.set_facecolor(BG)
plt.tight_layout()
save(fig, "01_demand_forecast.png")

# ── 2. SKU Velocity Distribution ─────────────────────────────────────────
df_pack = pd.read_csv(OUT / "02_pack_recommendations.csv")
vc_counts = df_pack["VelocityClass"].value_counts()
colours = {"fast_mover": GREEN, "medium_mover": BLUE, "slow_mover": ORANGE}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor=BG)
fig.suptitle("Pack Size Optimisation — SKU Velocity Classification", fontsize=13, color=TEXT, fontweight="bold")

wedge_colours = [colours.get(k, GREY) for k in vc_counts.index]
ax1.pie(vc_counts.values, labels=[k.replace("_", " ").title() for k in vc_counts.index],
        colors=wedge_colours, autopct="%1.0f%%", startangle=90,
        textprops={"color": TEXT, "fontsize": 11})
ax1.set_title("Velocity Distribution", color=TEXT)

pack_types = df_pack["PackConfig"].value_counts()
bars = ax2.barh(pack_types.index, pack_types.values, color=[BLUE, GREEN, ORANGE][:len(pack_types)], height=0.5)
ax2.set_xlabel("Number of SKUs", color=TEXT)
ax2.set_title("Pack Configuration", color=TEXT)
ax2.spines[["top","right"]].set_visible(False)
ax2.set_facecolor(BG)
ax2.tick_params(colors=TEXT)
for bar in bars:
    ax2.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
             str(int(bar.get_width())), va="center", fontsize=10, color=TEXT)
plt.tight_layout()
save(fig, "02_pack_velocity.png")

# ── 3. Financial Impact — Before vs After ────────────────────────────────
df_fin = pd.read_csv(OUT / "04_financial_impact.csv")
top10 = df_fin.nlargest(10, "After_AnnualRevenue")

fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG)
fig.suptitle("Financial Impact — Revenue & PBT Before vs After", fontsize=13, color=TEXT, fontweight="bold")

x = np.arange(len(top10))
w = 0.38
for ax, (before_col, after_col, title, ylabel) in zip(axes, [
    ("Before_AnnualRevenue", "After_AnnualRevenue", "Annual Revenue (£)", "Revenue (£)"),
    ("Before_PBT",           "After_PBT",           "Profit Before Tax (£)", "PBT (£)"),
]):
    b1 = ax.bar(x - w/2, top10[before_col], w, label="Before", color=GREY, edgecolor="#ccc")
    b2 = ax.bar(x + w/2, top10[after_col],  w, label="After",  color=GREEN, edgecolor="#27ae60")
    ax.set_title(title, color=TEXT, fontsize=11)
    ax.set_ylabel(ylabel, color=TEXT)
    ax.set_xticks(x)
    ax.set_xticklabels(top10["SkuId"], rotation=45, ha="right", fontsize=8)
    ax.legend(fontsize=9)
    ax.spines[["top","right"]].set_visible(False)
    ax.set_facecolor(BG)
    ax.tick_params(colors=TEXT)
plt.tight_layout()
save(fig, "03_financial_impact.png")

# ── 4. KPI Benchmark Gauges ──────────────────────────────────────────────
df_bench = pd.read_csv(OUT / "11_benchmark_comparison.csv")

fig, axes = plt.subplots(1, 5, figsize=(16, 4), facecolor=BG)
fig.suptitle("KPI Benchmark Comparison — Achieved vs Target Range", fontsize=13, color=TEXT, fontweight="bold")

for ax, (_, row) in zip(axes, df_bench.iterrows()):
    achieved = float(row["Achieved"])
    lo, hi   = float(row["Target_Min"]), float(row["Target_Max"])
    on_target = lo <= achieved <= hi

    ax.barh(["Target Range"], [hi - lo], left=lo, height=0.4, color=GREY, edgecolor="#ccc")
    ax.scatter([achieved], ["Target Range"], color=GREEN if on_target else RED,
               s=200, zorder=5, marker="D")
    ax.axvline(achieved, color=GREEN if on_target else RED, linestyle="--", alpha=0.5)

    label = row["Metric"].replace(" %", "%")
    ax.set_title(label, fontsize=9, color=TEXT, wrap=True)
    ax.set_xlim(max(0, lo - 5), hi + 5)
    ax.text(achieved, 0.55, f"{achieved:.1f}%", ha="center", fontsize=10,
            color=GREEN if on_target else RED, fontweight="bold",
            transform=ax.get_xaxis_transform())
    ax.spines[["top","right","left"]].set_visible(False)
    ax.set_yticks([])
    ax.set_facecolor(BG)
    ax.tick_params(colors=TEXT)
plt.tight_layout()
save(fig, "04_benchmark_kpis.png")

# ── 5. Production Planning — Plant Utilisation ───────────────────────────
df_util = pd.read_csv(OUT / "07_plant_utilisation.csv")
df_util["PlantId"] = df_util["PlantId"].astype(str)
df_util = df_util.sort_values("TargetTotalDailyProd", ascending=True).tail(15)

fig, ax = plt.subplots(figsize=(12, 6), facecolor=BG)
fig.suptitle("Production Planning — Target Daily Production by Plant", fontsize=13, color=TEXT, fontweight="bold")

bar_colours = [GREEN if float(r["CurrentUtilisation"]) < 0.82 else ORANGE
               for _, r in df_util.iterrows()]
bars = ax.barh(df_util["PlantId"], df_util["TargetTotalDailyProd"],
               color=bar_colours, edgecolor="none", height=0.6)
ax.set_xlabel("Target Daily Production (units)", color=TEXT)
ax.set_ylabel("Plant ID", color=TEXT)
ax.spines[["top","right"]].set_visible(False)
ax.set_facecolor(BG)
ax.tick_params(colors=TEXT)
for bar in bars:
    ax.text(bar.get_width() + 20, bar.get_y() + bar.get_height()/2,
            f"{bar.get_width():,.0f}", va="center", fontsize=8, color=TEXT)
legend_patches = [
    mpatches.Patch(color=GREEN,  label="Normal utilisation"),
    mpatches.Patch(color=ORANGE, label="High utilisation (>82%)"),
]
ax.legend(handles=legend_patches, fontsize=9)
plt.tight_layout()
save(fig, "05_plant_utilisation.png")

# ── 6. Dispatch — Lead Time Reduction ────────────────────────────────────
df_disp = pd.read_csv(OUT / "08_dispatch_plan.csv")
sku_lt = df_disp.groupby("SkuId").agg(
    CurrentLeadTimeDays=("CurrentLeadTimeDays","mean"),
    OptimisedLeadTimeDays=("OptimisedLeadTimeDays","mean"),
).reset_index().sort_values("CurrentLeadTimeDays", ascending=False).head(20)

fig, ax = plt.subplots(figsize=(13, 6), facecolor=BG)
fig.suptitle("Dispatch Optimisation — Lead Time Before vs After (Top 20 SKUs)", fontsize=13, color=TEXT, fontweight="bold")

x = np.arange(len(sku_lt))
w = 0.38
ax.bar(x - w/2, sku_lt["CurrentLeadTimeDays"],   w, label="Before", color=ORANGE, edgecolor="none")
ax.bar(x + w/2, sku_lt["OptimisedLeadTimeDays"], w, label="After",  color=GREEN,  edgecolor="none")
ax.set_xticks(x)
ax.set_xticklabels(sku_lt["SkuId"], rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Lead Time (days)", color=TEXT)
ax.legend(fontsize=10)
ax.spines[["top","right"]].set_visible(False)
ax.set_facecolor(BG)
ax.tick_params(colors=TEXT)
avg_red = df_disp["LeadTimeReduction_Pct"].mean()
ax.text(0.98, 0.97, f"Avg reduction: {avg_red:.1f}%", transform=ax.transAxes,
        ha="right", va="top", fontsize=11, color=GREEN, fontweight="bold")
plt.tight_layout()
save(fig, "06_dispatch_lead_time.png")

print("\nAll 6 charts saved to outputs/charts/")
