"""Supplementary figures: composition profile of the dataset_v3 TEST split.

Reads only the aggregate CSVs in data/ (exported from the VM profiles by
scripts/dataset_v3/report_profile.py); no chip-level data leaves the VM.
Palette, rcParams and helpers are the census set's (SUPP_dataset_v3_census),
so the two figure sets read as one system.
Run: conda run -n eda python make_figures.py
"""
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE / "data"
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

# palette (dataviz reference instance, light mode; validated 3-slot categorical)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"   # slots 1-3
C_WATER, C_LAND, C_MIXED = BLUE, ORANGE, AQUA
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
       "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
INK, INK2, MUTED, GRID, BASE, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#ffffff"

mpl.rcParams.update({
    "font.family": "sans-serif", "font.size": 8, "axes.labelsize": 8,
    "axes.titlesize": 9, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.edgecolor": BASE, "axes.linewidth": 0.8, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.facecolor": SURF, "axes.facecolor": SURF,
    "legend.frameon": False, "legend.fontsize": 7.5,
    "pdf.fonttype": 42, "savefig.dpi": 300,
})


def style(ax, grid_axis="y"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=grid_axis)
    ax.grid(axis="x" if grid_axis == "y" else "y", visible=False)


def panel_label(ax, s):
    ax.text(-0.08, 1.06, s, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="bottom", ha="right", color=INK)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", facecolor=SURF)
    plt.close(fig)
    print("wrote", name)


scal = pd.read_csv(DATA / "summary_scalars.csv").iloc[0]
N = int(scal.chips)

# ============================ FIG T1 — geography =============================
geo = pd.read_csv(DATA / "pair_geo.csv")
other = geo[geo.split_v3 != "test"]
test = geo[geo.split_v3 == "test"]

fig = plt.figure(figsize=(9.5, 4.6), constrained_layout=True)
gs = fig.add_gridspec(2, 3, height_ratios=[1.65, 1])
axm = fig.add_subplot(gs[0, :])
axs = fig.add_subplot(gs[1, 0]); axla = fig.add_subplot(gs[1, 1]); axsz = fig.add_subplot(gs[1, 2])

axm.scatter(other.lon, other.lat, s=np.clip(other.n_chips / 90, 1.0, 18),
            color=BASE, alpha=0.55, linewidths=0, label="train + val")
axm.scatter(test.lon, test.lat, s=np.clip(test.n_chips / 60, 2.5, 34),
            color=BLUE, alpha=0.85, linewidths=0, label="test")
axm.set_xlim(-180, 180); axm.set_ylim(-62, 72)
axm.set_xlabel("longitude (°)"); axm.set_ylabel("latitude (°)")
axm.set_xticks(range(-180, 181, 60)); axm.set_yticks(range(-60, 61, 30))
axm.legend(loc="lower left", markerscale=1.6, borderaxespad=0.3)
axm.text(0.99, 0.04, "one dot per pair · area ∝ chips", transform=axm.transAxes,
         fontsize=7.5, color=INK2, ha="right")
style(axm, grid_axis="y"); axm.grid(axis="x", visible=True)
panel_label(axm, "a")

# (b) pairs and chips per split
ax = axs
counts = geo.groupby("split_v3").agg(pairs=("pair_name", "size"),
                                     chips=("n_chips", "sum"))
counts = counts.reindex(["train", "val", "test"])
x = np.arange(3)
ax.bar(x, counts.chips / 1000, color=[BASE, BASE, BLUE], width=0.6)
for i, (p, c) in enumerate(zip(counts.pairs, counts.chips)):
    ax.text(i, c / 1000, f"{p} pairs", ha="center", va="bottom", fontsize=7, color=INK2)
ax.set_xticks(x); ax.set_xticklabels(counts.index)
ax.set_ylabel("chips (thousands)"); ax.set_ylim(0, 1050)
style(ax); panel_label(ax, "b")

# (c) latitude band, test vs corpus share
ax = axla
geo["latband"] = (geo.lat // 10 * 10).astype(int)
lb = geo.groupby(["latband", "split_v3"]).n_chips.sum().unstack(fill_value=0)
lb["corpus"] = lb.sum(axis=1)
share_test = 100 * lb["test"] / lb["test"].sum()
share_corpus = 100 * lb["corpus"] / lb["corpus"].sum()
ax.barh(lb.index + 5, share_corpus, height=8.6, color=BASE, label="corpus")
ax.plot(share_test, lb.index + 5, color=BLUE, lw=1.8, marker="o", ms=3.2,
        label="test")
ax.set_ylabel("latitude band (°)"); ax.set_xlabel("share of chips (%)")
ax.set_yticks(np.arange(-60, 70, 30))
ax.set_xlim(0, 26)
ax.legend(loc="lower right", borderaxespad=0.3)
style(ax, grid_axis="x"); panel_label(ax, "c")

# (d) chips per pair distribution
ax = axsz
ps = pd.read_csv(DATA / "pair_stability.csv")
ax.hist(ps.n_chips, bins=np.arange(0, 1700, 50), color=BLUE)
med = ps.n_chips.median()
ax.axvline(med, color=INK2, lw=0.9, ls="--")
ax.text(med + 40, ax.get_ylim()[1] * 0.94, f"median {med:.0f}", fontsize=7,
        color=INK2, va="top")
ax.set_xlabel("chips per pair (test)"); ax.set_ylabel("pairs")
style(ax); panel_label(ax, "d")
save(fig, "fig_T1_geography")

# ============================ FIG T2 — composition ===========================
wh = pd.read_csv(DATA / "ws_hist.csv")
wc = pd.read_csv(DATA / "ws_classes.csv")
fig, axes = plt.subplots(1, 3, figsize=(9.5, 2.7), constrained_layout=True)

# (a) test water-share distribution with the corpus outline overlaid
ax = axes[0]


def rebin(group, n_bins=50):
    g = wh[wh.group == group]
    bins = g[g.kind == "bin"].copy()
    bins["b"] = np.minimum((bins.left * n_bins).astype(int), n_bins - 1)
    h = bins.groupby("b")["count"].sum().reindex(range(n_bins), fill_value=0)
    e0 = int(g[g.kind == "exact0"]["count"].iloc[0])
    e1 = int(g[g.kind == "exact1"]["count"].iloc[0])
    return h.values, e0, e1


h_test, e0_t, e1_t = rebin("test")
h_corp, e0_c, e1_c = rebin("corpus")
n_corp = h_corp.sum() + e0_c + e1_c
centers = (np.arange(50) + 0.5) / 50
colors = [C_LAND if c <= 0.01 else C_WATER if c >= 0.99 else C_MIXED for c in centers]
ax.bar(centers, 100 * h_test / N, width=0.9 / 50, color=colors, edgecolor="none")
ax.bar([-0.05], [100 * e0_t / N], width=0.9 / 50, color=C_LAND)
ax.bar([1.05], [100 * e1_t / N], width=0.9 / 50, color=C_WATER)
# corpus outline, same normalization -> shapes are comparable
outline_x = np.concatenate([[-0.05], centers, [1.05]])
outline_y = 100 * np.concatenate([[e0_c], h_corp, [e1_c]]) / n_corp
ax.step(outline_x, outline_y, where="mid", color=INK2, lw=1.0, label="corpus")
ax.set_yscale("log"); ax.set_ylim(2e-3, 60)
ax.set_xticks([-0.05, .25, .5, .75, 1.05])
ax.set_xticklabels(["=0", "0.25", "0.50", "0.75", "=1"])
handles = [mpl.patches.Patch(color=c, label=l) for c, l in
           [(C_LAND, "pure land"), (C_MIXED, "mixed"), (C_WATER, "pure water")]]
handles.append(mpl.lines.Line2D([], [], color=INK2, lw=1.0, label="corpus"))
ax.legend(handles=handles, loc="upper center", ncol=1, handlelength=1.2,
          borderaxespad=0.2)
ax.set_xlabel("water share  $n_w/(n_w+n_l)$"); ax.set_ylabel("% of split (log)")
style(ax); panel_label(ax, "a")

# (b) class shares at t = 0 / 1 / 5 %, test vs corpus
ax = axes[1]
order = ["pure_land", "mixed", "pure_water"]
cols = {"pure_land": C_LAND, "mixed": C_MIXED, "pure_water": C_WATER}
ts = [0.0, 0.01, 0.05]
width = 0.36
for k, (grp, offset, alpha) in enumerate([("corpus", -width / 2, 0.45),
                                          ("test", width / 2, 1.0)]):
    sub = wc[wc.group == grp].set_index("t").reindex(ts)
    bottom = np.zeros(3)
    for cls in order:
        v = 100 * sub[cls].values
        ax.bar(np.arange(3) + offset, v, bottom=bottom, width=width,
               color=cols[cls], alpha=alpha, edgecolor=SURF, linewidth=1.2)
        if grp == "test":
            for i in range(3):
                ax.text(i + offset, bottom[i] + v[i] / 2, f"{v[i]:.0f}%",
                        ha="center", va="center", fontsize=7,
                        color="white" if cls != "mixed" else INK)
        bottom += v
ax.set_xticks(range(3)); ax.set_xticklabels(["t = 0%", "t = 1%", "t = 5%"])
ax.set_ylabel("share of split (%)"); ax.set_ylim(0, 108)
ax.set_yticks(range(0, 101, 20))
ax.text(0.5, 1.03, "each pair: corpus (pale, left) · test (solid, labelled)",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=7, color=INK2)
style(ax); panel_label(ax, "b")

# (c) representativeness: TVD and W1
ax = axes[2]
tv = pd.read_csv(DATA / "tvd.csv").sort_values("vs_corpus")
y = np.arange(len(tv))
ax.barh(y, tv.vs_corpus, height=0.6, color=BLUE)
for i, v in enumerate(tv.vs_corpus):
    ax.text(v + 0.002, i, f"{v:.4f}", va="center", fontsize=6.8, color=INK2)
ax.set_yticks(y); ax.set_yticklabels(tv.axis)
ax.set_xlim(0, 0.108)
ax.set_xlabel("total variation distance, test vs corpus")
style(ax, grid_axis="x"); panel_label(ax, "c")
save(fig, "fig_T2_composition")

# ============================ FIG T3 — strata ================================
st = pd.read_csv(DATA / "strata.csv")
st[["gcl", "clim"]] = st.stratum_coarse.str.split("_", expand=True)
st["gcl"] = st.gcl.astype(int)
gname = {0: "artificial", 1: "biogenic", 2: "sandy", 3: "muddy", 4: "rocky", 5: "estuary"}
clims = ["A", "B", "C", "D", "E"]
gcls = [2, 4, 0, 1, 3, 5]  # by corpus share, as in the census figure
chips = np.full((6, 5), np.nan); pairs = np.full((6, 5), np.nan)
absent_in_test = np.zeros((6, 5), dtype=bool)
for _, r in st.iterrows():
    i, j = gcls.index(r.gcl), clims.index(r.clim)
    if r.chips > 0:
        chips[i, j] = r.chips; pairs[i, j] = r.pairs
    else:
        absent_in_test[i, j] = True

fig, ax = plt.subplots(figsize=(5.6, 3.5), constrained_layout=True)
cmap = mpl.colors.LinearSegmentedColormap.from_list("seq", SEQ)
norm = mpl.colors.LogNorm(vmin=400, vmax=np.nanmax(chips))
im = ax.pcolormesh(np.ma.masked_invalid(chips), cmap=cmap, norm=norm,
                   edgecolors=SURF, linewidth=2)
for i in range(6):
    for j in range(5):
        if absent_in_test[i, j]:
            ax.text(j + 0.5, i + 0.5, "no test\nchips", ha="center", va="center",
                    fontsize=6.8, color=ORANGE, style="italic", fontweight="bold")
        elif np.isnan(chips[i, j]):
            ax.text(j + 0.5, i + 0.5, "absent", ha="center", va="center",
                    fontsize=7, color=MUTED, style="italic")
        else:
            dark = chips[i, j] > 8000
            k = f"{chips[i,j]/1000:.1f}k" if chips[i, j] >= 10000 else f"{chips[i,j]:,.0f}"
            pw = "pair" if pairs[i, j] == 1 else "pairs"
            ax.text(j + 0.5, i + 0.5, f"{k}\n{pairs[i,j]:.0f} {pw}", ha="center",
                    va="center", fontsize=7, color="white" if dark else INK)
ax.set_xticks(np.arange(5) + 0.5)
ax.set_xticklabels(["A tropical", "B arid", "C temperate", "D cold", "E polar"])
ax.set_yticks(np.arange(6) + 0.5)
ax.set_yticklabels([gname[g] for g in gcls])
ax.invert_yaxis()
ax.set_xlabel("Köppen–Geiger broad climate"); ax.set_ylabel("GCL_FCS30 coastline type")
ax.grid(False)
for sp in ax.spines.values():
    sp.set_visible(False)
cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
cb.set_label("test chips (log scale)", fontsize=7.5)
cb.outline.set_visible(False)
save(fig, "fig_T3_strata")

# ============================ FIG T4 — pair-macro stability ==================
fig, axes = plt.subplots(1, 3, figsize=(9.5, 2.7), constrained_layout=True)

# (a) mixed chips per pair, with the <5 flag
ax = axes[0]
ax.hist(ps.n_mixed, bins=np.arange(0, 720, 20), color=BLUE)
ax.axvline(5, color=ORANGE, lw=1.2)
n_fragile = int(scal.fragile_pairs)
ax.text(0.97, 0.94,
        f"{n_fragile} of {int(scal.pairs)} pairs have <5 mixed chips\n"
        f"({int(scal.no_mixed_pairs)} have none) — orange line at 5",
        transform=ax.transAxes, ha="right", va="top", fontsize=7, color=ORANGE)
ax.set_xlabel("mixed chips per pair  ·  t = 1%"); ax.set_ylabel("pairs")
style(ax); panel_label(ax, "a")

# (b) Lorenz curve with Gini / Kish
ax = axes[1]
s = np.sort(ps.n_chips.values)
lor = np.concatenate([[0], np.cumsum(s) / s.sum()])
x = np.linspace(0, 1, len(lor))
ax.plot(x * 100, lor * 100, color=BLUE, lw=2)
ax.plot([0, 100], [0, 100], color=BASE, lw=0.8, ls=":")
ax.text(4, 88, f"Gini {scal.gini_chips_over_pairs:.2f}\n"
               f"Kish n$_e$ {scal.kish_pairs:,.0f} of {int(scal.pairs)} pairs",
        fontsize=7.5, color=INK2, va="top")
ax.set_xlabel("pairs, smallest first (%)"); ax.set_ylabel("cumulative chip mass (%)")
ax.set_xlim(0, 100); ax.set_ylim(0, 100)
style(ax); panel_label(ax, "b")

# (c) mixed share vs pair size — where the fragile pairs sit
ax = axes[2]
frag = ps.n_mixed < 5
ax.scatter(ps.n_chips[~frag], 100 * ps.mixed_share[~frag], s=9, color=BLUE,
           alpha=0.55, linewidths=0, label="≥5 mixed chips")
ax.scatter(ps.n_chips[frag], 100 * ps.mixed_share[frag], s=22, color=ORANGE,
           alpha=0.95, linewidths=0, label="<5 mixed chips")
ax.set_xscale("log")
ax.set_xlabel("chips per pair (log)"); ax.set_ylabel("mixed chips (%)  ·  t = 1%")
ax.legend(loc="upper left", borderaxespad=0.2, markerscale=1.3)
style(ax); panel_label(ax, "c")
save(fig, "fig_T4_stability")

# ============================ FIG T5 — radiometry ============================
rad = pd.read_csv(DATA / "radiometry_hist.csv")
w1 = pd.read_csv(DATA / "w1.csv").set_index("axis")
fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.8), constrained_layout=True)
for ax, band, label, letter in [(axes[0], "vv", "VV", "a"), (axes[1], "vh", "VH", "b")]:
    for grp, color, lw, ls in [("train", BASE, 1.6, "-"), ("test", BLUE, 1.8, "-")]:
        g = rad[(rad.band == band) & (rad.group == grp)].sort_values("left")
        dens = 100 * g["count"].values / g["count"].sum()
        ax.stairs(dens, np.append(g.left.values, g.right.values[-1]),
                  color=color, lw=lw, ls=ls, label=grp)
    d = w1.loc[f"{label} chip-mean", "vs_train"]
    ax.text(0.03, 0.94, f"$W_1$ test vs train\n{d:.3f} dB",
            transform=ax.transAxes, fontsize=7.5, color=INK2, va="top")
    ax.set_xlabel(f"{label} chip-mean backscatter (dB)")
    ax.set_ylabel("% of split")
    ax.set_xlim(-38, 0)
    ax.legend(loc="upper right", borderaxespad=0.2)
    style(ax); panel_label(ax, letter)
save(fig, "fig_T5_radiometry")

# ============================ FIG T6 — tide & time ===========================
tl = pd.read_csv(DATA / "tide_level_hist.csv")
ph = pd.read_csv(DATA / "phase.csv")
yy = pd.read_csv(DATA / "year.csv")
mo = pd.read_csv(DATA / "month.csv")
fig, axes = plt.subplots(1, 4, figsize=(11.0, 2.6), constrained_layout=True)

ax = axes[0]
for grp, color, lw in [("corpus", BASE, 1.6), ("test", BLUE, 1.8)]:
    g = tl[tl.group == grp].sort_values("left")
    dens = 100 * g["count"].values / g["count"].sum()
    ax.stairs(dens, np.append(g.left.values, g.right.values[-1]), color=color,
              lw=lw, label=grp)
ax.set_xlim(-300, 300)
ax.set_xlabel("tide level at S1 (cm rel. MSL)"); ax.set_ylabel("% of split")
ax.text(0.03, 0.94, f"$W_1$ vs corpus\n{w1.loc['tide level at S1','vs_corpus']:.1f} cm",
        transform=ax.transAxes, fontsize=7.5, color=INK2, va="top")
ax.legend(loc="upper right", borderaxespad=0.2)
style(ax); panel_label(ax, "a")

ax = axes[1]
order = ["ebb", "flood", "high", "low", "none"]
pv = ph.pivot(index="phase", columns="group", values="share").reindex(order)
x = np.arange(len(order))
ax.bar(x - 0.19, 100 * pv["corpus"], width=0.38, color=BASE, label="corpus")
ax.bar(x + 0.19, 100 * pv["test"], width=0.38, color=BLUE, label="test")
ax.set_xticks(x); ax.set_xticklabels(order)
ax.set_xlabel("tidal phase at S1"); ax.set_ylabel("share of split (%)")
ax.legend(loc="upper right", borderaxespad=0.2)
style(ax); panel_label(ax, "b")

ax = axes[2]
yt = yy[yy.group == "test"].set_index("year")["chips"]
yc = yy[yy.group == "corpus"].set_index("year")["chips"]
years = sorted(yc.index)
ax.bar(years, 100 * yc.reindex(years).fillna(0) / yc.sum(), width=0.72,
       color=BASE, label="corpus")
ax.plot(years, 100 * yt.reindex(years).fillna(0) / yt.sum(), color=BLUE, lw=1.8,
        marker="o", ms=3.4, label="test")
ax.set_xlabel("year (S1 acquisition)"); ax.set_ylabel("share of split (%)")
ax.legend(loc="upper left", borderaxespad=0.2)
style(ax); panel_label(ax, "c")

ax = axes[3]
mt = mo[mo.group == "test"].set_index("month")["chips"]
mc = mo[mo.group == "corpus"].set_index("month")["chips"]
months = list(range(1, 13))
ax.bar(months, 100 * mc.reindex(months).fillna(0) / mc.sum(), width=0.72,
       color=BASE, label="corpus")
ax.plot(months, 100 * mt.reindex(months).fillna(0) / mt.sum(), color=BLUE, lw=1.8,
        marker="o", ms=3.4, label="test")
ax.set_xticks(months); ax.set_xlabel("month"); ax.set_ylabel("share of split (%)")
ax.legend(loc="upper left", borderaxespad=0.2)
style(ax); panel_label(ax, "d")
save(fig, "fig_T6_tide_time")

print("all figures written to", OUT)
