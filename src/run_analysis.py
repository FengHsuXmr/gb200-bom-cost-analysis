"""執行全部分析，輸出圖表 (outputs/charts) 與結果 (outputs/results.json、*.csv)。

用法：
    python -m src.run_analysis
    python -m src.run_analysis --asp 3200000   # 改整櫃售價
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from . import model

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
CHARTS = OUT / "charts"

# 中文字型：優先使用 Noto Sans CJK TC，找不到時退回系統預設
for name in ("Noto Sans CJK TC", "Noto Sans TC", "Microsoft JhengHei", "PingFang TC", "Heiti TC"):
    if any(name in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = name
        break
plt.rcParams["axes.unicode_minus"] = False

INK = "#1f2933"
MUTED = "#7b8794"
ACCENT = "#2f6fdf"
ACCENT_2 = "#e07a2f"
GRID = "#e4e7eb"


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fmt_k(v):
    return f"${v/1e6:.2f}M" if abs(v) >= 1e6 else f"${v/1e3:,.0f}K"


def chart_subsystem(rc: model.RackCost):
    s = rc.by_subsystem().iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    colors = [ACCENT if i == len(s) - 1 else "#9bb8ef" for i in range(len(s))]
    ax.barh(s.index, s["cost"], color=colors)
    for i, (c, sh) in enumerate(zip(s["cost"], s["share_of_cost"])):
        ax.text(c, i, f"  {fmt_k(c)} ({sh:.1%})", va="center", fontsize=9, color=INK)
    ax.set_xscale("log")
    ax.set_xlabel("單櫃成本 (USD，對數刻度)", color=MUTED)
    ax.set_title("GB200 NVL72 子系統成本：運算晶片占絕大多數", loc="left", color=INK, fontsize=12)
    _style(ax)
    ax.set_xlim(right=s["cost"].max() * 8)
    fig.tight_layout()
    fig.savefig(CHARTS / "01_subsystem_cost.png", dpi=160)
    plt.close(fig)


def chart_non_nvidia(rc: model.RackCost):
    bom = rc.bom[rc.bom["vendor_type"] != "NVIDIA"]
    s = bom.groupby("subsystem")["ext_base"].sum().sort_values()
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.barh(s.index, s.values, color=ACCENT_2)
    for i, v in enumerate(s.values):
        ax.text(v, i, f"  {fmt_k(v)} ({v/s.sum():.0%})", va="center", fontsize=9, color=INK)
    ax.set_title(f"非 NVIDIA 料件 {fmt_k(s.sum())}：成本工程真正能動的範圍", loc="left", color=INK, fontsize=12)
    ax.set_xlabel("USD", color=MUTED)
    _style(ax)
    ax.set_xlim(right=s.max() * 1.35)
    fig.tight_layout()
    fig.savefig(CHARTS / "02_non_nvidia_bom.png", dpi=160)
    plt.close(fig)


def chart_tornado(tor):
    t = tor.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    y = range(len(t))
    ax.barh(y, t["delta_low"], color="#7fb685", label="單價取 low")
    ax.barh(y, t["delta_high"], color="#d9685f", label="單價取 high")
    ax.set_yticks(list(y))
    ax.set_yticklabels([f"{i} {n[:22]}" for i, n in zip(t["item_id"], t["item"])], fontsize=8.5)
    ax.axvline(0, color=INK, linewidth=0.8)
    ax.set_xlabel("相對 base 的總成本變化 (USD)", color=MUTED)
    ax.set_title("敏感度分析：前 10 大影響料號", loc="left", color=INK, fontsize=12)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    _style(ax)
    fig.tight_layout()
    fig.savefig(CHARTS / "03_tornado.png", dpi=160)
    plt.close(fig)


def chart_monte_carlo(mc):
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.hist(mc["samples_margin"] * 100, bins=60, color="#9bb8ef")
    for p, lab in ((mc["p10_margin"], "P10"), (mc["p50_margin"], "P50"), (mc["p90_margin"], "P90")):
        ax.axvline(p * 100, color=ACCENT if lab == "P50" else MUTED, linestyle="--", linewidth=1)
        ax.text(p * 100, ax.get_ylim()[1] * 0.92, f" {lab} {p:.1%}", fontsize=9, color=INK)
    ax.axvline(0, color="#d9685f", linewidth=1)
    ax.set_xlabel("ODM 毛利率 (%)", color=MUTED)
    ax.set_title("蒙地卡羅模擬 (20,000 次)：售價固定 300 萬美元時的 ODM 毛利率分佈", loc="left", color=INK, fontsize=11)
    _style(ax)
    ax.grid(axis="y", color=GRID)
    fig.tight_layout()
    fig.savefig(CHARTS / "04_monte_carlo_margin.png", dpi=160)
    plt.close(fig)


def chart_cost_down(cd, base_margin):
    d = cd.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    color_map = {"商務": ACCENT, "規格": ACCENT_2, "製程": "#7fb685", "設計": "#a77fd1"}
    ax.barh(d["name"], d["saving_usd"], color=[color_map[c] for c in d["category"]])
    for i, (v, pt) in enumerate(zip(d["saving_usd"], d["margin_uplift_pt"])):
        ax.text(v, i, f"  {fmt_k(v)} (+{pt:.2f}pt)", va="center", fontsize=9, color=INK)
    total = cd["saving_usd"].sum()
    ax.set_title(f"降本方案：合計可省 {fmt_k(total)}，ODM 毛利率 {base_margin:.1%} → "
                 f"{base_margin + total/model.DEFAULT_RACK_ASP:.1%}", loc="left", color=INK, fontsize=11)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in color_map.values()]
    ax.legend(handles, color_map.keys(), frameon=False, fontsize=9, loc="lower right")
    ax.set_xlabel("單櫃節省 (USD)", color=MUTED)
    _style(ax)
    ax.set_xlim(right=d["saving_usd"].max() * 1.45)
    fig.tight_layout()
    fig.savefig(CHARTS / "05_cost_down.png", dpi=160)
    plt.close(fig)


def chart_hbm(hbm):
    fig, ax1 = plt.subplots(figsize=(8, 3.8))
    ax1.plot(hbm["hbm_usd_per_gb"], hbm["b200_cost"], marker="o", color=ACCENT)
    ax1.set_xlabel("HBM3E 價格 (USD/GB)", color=MUTED)
    ax1.set_ylabel("B200 製造成本 (USD)", color=ACCENT)
    ax2 = ax1.twinx()
    ax2.plot(hbm["hbm_usd_per_gb"], hbm["hbm_share"] * 100, marker="s", color=ACCENT_2)
    ax2.set_ylabel("HBM 占晶片成本 (%)", color=ACCENT_2)
    ax1.set_title("HBM 漲價對 B200 製造成本的影響", loc="left", color=INK, fontsize=12)
    _style(ax1)
    ax2.spines[["top"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(CHARTS / "06_hbm_sensitivity.png", dpi=160)
    plt.close(fig)


def main(rack_asp: float = model.DEFAULT_RACK_ASP):
    CHARTS.mkdir(parents=True, exist_ok=True)
    bom = model.load_bom()
    rc = model.RackCost(bom, rack_asp=rack_asp)
    rc_low = model.RackCost(bom, rack_asp=rack_asp, case="low")
    rc_high = model.RackCost(bom, rack_asp=rack_asp, case="high")
    tor = model.tornado(bom, rack_asp)
    mc = model.monte_carlo(bom, rack_asp)
    cd = model.cost_down_table(bom, rack_asp)
    sc = model.should_cost_summary(model.load_should_cost())
    hbm = model.hbm_price_sensitivity(model.load_should_cost())
    tco = model.tco_per_gpu_hour(rack_capex=rack_asp)

    chart_subsystem(rc)
    chart_non_nvidia(rc)
    chart_tornado(tor)
    chart_monte_carlo(mc)
    chart_cost_down(cd, rc.odm_gross_margin)
    chart_hbm(hbm)

    rc.by_subsystem().to_csv(OUT / "subsystem_breakdown.csv", encoding="utf-8-sig")
    rc.pareto().to_csv(OUT / "pareto.csv", index=False, encoding="utf-8-sig")
    tor.to_csv(OUT / "tornado.csv", index=False, encoding="utf-8-sig")
    cd.to_csv(OUT / "cost_down.csv", index=False, encoding="utf-8-sig")
    hbm.to_csv(OUT / "hbm_sensitivity.csv", index=False, encoding="utf-8-sig")

    all_levers = model.apply_levers(bom, model.DEFAULT_LEVERS)
    after_total = float(all_levers["ext_base"].sum())
    results = {
        "rack_asp": rack_asp,
        "total_cost": {"low": rc_low.total_cost, "base": rc.total_cost, "high": rc_high.total_cost},
        "odm_gross_margin": {"low_cost": rc_low.odm_gross_margin, "base": rc.odm_gross_margin,
                             "high_cost": rc_high.odm_gross_margin},
        "odm_gross_profit_base": rc.odm_gross_profit,
        "cost_per_gpu_base": rc.cost_per_gpu,
        "non_nvidia_bom_base": rc.non_nvidia_bom(),
        "nvidia_share_of_asp": float(rc.by_vendor_type().loc["NVIDIA", "share_of_asp"]),
        "subsystem": rc.by_subsystem().reset_index().to_dict(orient="records"),
        "monte_carlo": {k: v for k, v in mc.items() if not k.startswith("samples")},
        "cost_down_total_saving": rc.total_cost - after_total,
        "cost_down_margin_after": (rack_asp - after_total) / rack_asp,
        "b200_should_cost": sc.attrs["total"],
        "b200_chip_gross_margin": sc.attrs["chip_gross_margin"],
        "tco": tco,
    }
    (OUT / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    print(f"整櫃 BOM 成本 (base)：${rc.total_cost:,.0f}  (low ${rc_low.total_cost:,.0f} / high ${rc_high.total_cost:,.0f})")
    print(f"ODM 毛利：${rc.odm_gross_profit:,.0f}  毛利率 {rc.odm_gross_margin:.2%}")
    print(f"非 NVIDIA 料件：${rc.non_nvidia_bom():,.0f}")
    print(f"蒙地卡羅 P10/P50/P90 毛利率：{mc['p10_margin']:.2%} / {mc['p50_margin']:.2%} / {mc['p90_margin']:.2%}")
    print(f"毛利率 <3% 機率：{mc['prob_margin_below_3pct']:.1%}；虧損機率：{mc['prob_loss']:.1%}")
    print(f"降本方案合計：${results['cost_down_total_saving']:,.0f} → 毛利率 {results['cost_down_margin_after']:.2%}")
    print(f"B200 should-cost：${sc.attrs['total']:,.0f}，晶片毛利率 {sc.attrs['chip_gross_margin']:.1%}")
    print(f"TCO：${tco['cost_per_gpu_hour']:.2f} / GPU-hr (利用率 80%)")
    print(rc.by_subsystem().to_string())
    print(cd[["lever_id", "name", "saving_usd", "margin_uplift_pt"]].to_string())
    print(tor[["item_id", "delta_low", "delta_high"]].to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--asp", type=float, default=model.DEFAULT_RACK_ASP, help="整櫃售價 USD")
    main(p.parse_args().asp)
