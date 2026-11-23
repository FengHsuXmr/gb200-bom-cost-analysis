"""GB200 NVL72 機櫃 BOM 成本模型。

所有金額單位為美元（USD），以單一機櫃（1 rack）為計算單位。
單價為依公開研調／媒體報導推估的模型值，詳見 data/bom_gb200_nvl72.csv 的 source_note 欄位。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# 整櫃售價（ODM 賣給雲端客戶）：大摩 / HSBC / Barron's 皆估約 300 萬美元
DEFAULT_RACK_ASP = 3_000_000
GPUS_PER_RACK = 72


def load_bom(path: Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(path or DATA_DIR / "bom_gb200_nvl72.csv")
    for col in ("low", "base", "high"):
        df[f"ext_{col}"] = df["qty_per_rack"] * df[f"unit_cost_{col}"]
    return df


def load_should_cost(path: Path | None = None) -> pd.DataFrame:
    return pd.read_csv(path or DATA_DIR / "b200_should_cost.csv")


@dataclass
class RackCost:
    bom: pd.DataFrame
    rack_asp: float = DEFAULT_RACK_ASP
    case: str = "base"

    @property
    def total_cost(self) -> float:
        return float(self.bom[f"ext_{self.case}"].sum())

    @property
    def odm_gross_profit(self) -> float:
        return self.rack_asp - self.total_cost

    @property
    def odm_gross_margin(self) -> float:
        return self.odm_gross_profit / self.rack_asp

    @property
    def cost_per_gpu(self) -> float:
        return self.total_cost / GPUS_PER_RACK

    def by_subsystem(self) -> pd.DataFrame:
        col = f"ext_{self.case}"
        out = (
            self.bom.groupby("subsystem", sort=False)[col]
            .sum()
            .sort_values(ascending=False)
            .rename("cost")
            .to_frame()
        )
        out["share_of_cost"] = out["cost"] / self.total_cost
        out["share_of_asp"] = out["cost"] / self.rack_asp
        return out

    def by_vendor_type(self) -> pd.DataFrame:
        col = f"ext_{self.case}"
        out = self.bom.groupby("vendor_type")[col].sum().rename("cost").to_frame()
        out.loc["ODM 毛利"] = self.odm_gross_profit
        out["share_of_asp"] = out["cost"] / self.rack_asp
        return out.sort_values("cost", ascending=False)

    def pareto(self) -> pd.DataFrame:
        col = f"ext_{self.case}"
        out = self.bom[["item_id", "subsystem", "item", col]].sort_values(col, ascending=False)
        out = out.rename(columns={col: "cost"}).reset_index(drop=True)
        out["cum_share"] = out["cost"].cumsum() / self.total_cost
        return out

    def non_nvidia_bom(self) -> float:
        col = f"ext_{self.case}"
        return float(self.bom.loc[self.bom["vendor_type"] != "NVIDIA", col].sum())


# ---------------------------------------------------------------- 敏感度分析
def tornado(bom: pd.DataFrame, rack_asp: float = DEFAULT_RACK_ASP, top_n: int = 10) -> pd.DataFrame:
    """一次只動一個料號（low / high），其他維持 base，看總成本與 ODM 毛利率的變化。"""
    base_total = bom["ext_base"].sum()
    rows = []
    for _, r in bom.iterrows():
        low_total = base_total - r["ext_base"] + r["ext_low"]
        high_total = base_total - r["ext_base"] + r["ext_high"]
        rows.append(
            {
                "item_id": r["item_id"],
                "item": r["item"],
                "delta_low": low_total - base_total,
                "delta_high": high_total - base_total,
                "margin_at_low": (rack_asp - low_total) / rack_asp,
                "margin_at_high": (rack_asp - high_total) / rack_asp,
                "swing": high_total - low_total,
            }
        )
    out = pd.DataFrame(rows).sort_values("swing", ascending=False)
    return out.head(top_n).reset_index(drop=True)


def monte_carlo(
    bom: pd.DataFrame, rack_asp: float = DEFAULT_RACK_ASP, n: int = 20_000, seed: int = 42
) -> dict:
    """每個料號單價以三角分佈 (low, base, high) 抽樣，估計總成本與毛利率分佈。"""
    rng = np.random.default_rng(seed)
    totals = np.zeros(n)
    for _, r in bom.iterrows():
        lo, mode, hi = r["unit_cost_low"], r["unit_cost_base"], r["unit_cost_high"]
        if hi == lo:
            draws = np.full(n, mode)
        else:
            draws = rng.triangular(lo, mode, hi, n)
        totals += draws * r["qty_per_rack"]
    margins = (rack_asp - totals) / rack_asp
    return {
        "samples_total": totals,
        "samples_margin": margins,
        "p10_total": float(np.percentile(totals, 10)),
        "p50_total": float(np.percentile(totals, 50)),
        "p90_total": float(np.percentile(totals, 90)),
        "p10_margin": float(np.percentile(margins, 10)),
        "p50_margin": float(np.percentile(margins, 50)),
        "p90_margin": float(np.percentile(margins, 90)),
        "prob_margin_below_3pct": float((margins < 0.03).mean()),
        "prob_loss": float((margins < 0).mean()),
    }


# ---------------------------------------------------------------- 降本方案
@dataclass
class CostDownLever:
    lever_id: str
    name: str
    category: str  # 商務 / 設計 / 規格 / 製程
    items: list[str]
    # 每個料號的調整方式：("unit_pct", -0.12) 代表單價 -12%；("qty", 36) 代表數量改為 36
    adjust: tuple[str, float]
    feasibility: str  # 高 / 中 / 低
    risk: str
    owner: str = field(default="")


DEFAULT_LEVERS: list[CostDownLever] = [
    CostDownLever("L1", "CDU 改由機房 in-row CDU 提供", "規格", ["K04"], ("unit_abs", 0),
                  "中", "需客戶機房有設施水；改變交付範圍", "PM / 客戶端"),
    CostDownLever("L2", "冷板模組導入第二供應商議價", "商務", ["K01", "K02"], ("unit_pct", -0.12),
                  "高", "第二家需通過 NVIDIA 認證、漏液測試", "採購 / 熱流"),
    CostDownLever("L3", "Power Shelf 雙供應商 (台達電+光寶科)", "商務", ["P01"], ("unit_pct", -0.10),
                  "高", "韌體與 PMC 相容性驗證", "採購 / 電源"),
    CostDownLever("L4", "機構件 ODM 自製 (鈑金/機櫃/滑軌)", "製程", ["T02", "M01", "M02"], ("unit_pct", -0.15),
                  "中", "需自有鈑金產能，前期投資", "機構 / 製造"),
    CostDownLever("L5", "本地 SSD 由 4 顆降為 2 顆 / tray", "規格", ["T01"], ("qty", 36),
                  "中", "需與客戶確認儲存需求", "PM / 客戶端"),
    CostDownLever("L6", "整櫃測試流程優化 (縮短燒機時間)", "製程", ["I01"], ("unit_pct", -0.20),
                  "中", "早期失效 (infant mortality) 檢出率下降", "測試工程"),
    CostDownLever("L7", "Busbar/線材設計共用化 (GB200→GB300)", "設計", ["P03", "M04"], ("unit_pct", -0.15),
                  "高", "需同步兩代平台設計規範", "RD / 機構"),
]


def apply_levers(bom: pd.DataFrame, levers: list[CostDownLever]) -> pd.DataFrame:
    new = bom.copy()
    new["unit_cost_base"] = new["unit_cost_base"].astype(float)
    new["qty_per_rack"] = new["qty_per_rack"].astype(float)
    for lv in levers:
        mask = new["item_id"].isin(lv.items)
        kind, val = lv.adjust
        if kind == "unit_pct":
            new.loc[mask, "unit_cost_base"] = new.loc[mask, "unit_cost_base"] * (1 + val)
        elif kind == "unit_abs":
            new.loc[mask, "unit_cost_base"] = val
        elif kind == "qty":
            new.loc[mask, "qty_per_rack"] = val
        else:
            raise ValueError(f"unknown adjust kind: {kind}")
    new["ext_base"] = new["qty_per_rack"] * new["unit_cost_base"]
    return new


def cost_down_table(bom: pd.DataFrame, rack_asp: float = DEFAULT_RACK_ASP,
                    levers: list[CostDownLever] | None = None) -> pd.DataFrame:
    levers = levers or DEFAULT_LEVERS
    base_total = bom["ext_base"].sum()
    rows = []
    for lv in levers:
        saving = base_total - apply_levers(bom, [lv])["ext_base"].sum()
        rows.append({
            "lever_id": lv.lever_id, "name": lv.name, "category": lv.category,
            "saving_usd": saving, "margin_uplift_pt": saving / rack_asp * 100,
            "feasibility": lv.feasibility, "risk": lv.risk, "owner": lv.owner,
        })
    return pd.DataFrame(rows).sort_values("saving_usd", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------- B200 晶片 should-cost
def should_cost_summary(sc: pd.DataFrame, b200_asp: float = 40_000) -> pd.DataFrame:
    total = sc["cost_base"].sum()
    out = sc[["component", "cost_base"]].copy()
    out["share"] = out["cost_base"] / total
    out.attrs["total"] = total
    out.attrs["chip_gross_margin"] = 1 - total / b200_asp
    return out


def hbm_price_sensitivity(sc: pd.DataFrame, b200_asp: float = 40_000,
                          prices_per_gb=(12, 15, 18, 21, 24, 30), hbm_gb: int = 192) -> pd.DataFrame:
    """HBM 每 GB 價格變動對 B200 製造成本與晶片毛利率的影響。"""
    non_hbm = sc.loc[~sc["component"].str.contains("HBM"), "cost_base"].sum()
    rows = []
    for p in prices_per_gb:
        hbm = p * hbm_gb
        cost = non_hbm + hbm
        rows.append({"hbm_usd_per_gb": p, "hbm_cost": hbm, "b200_cost": cost,
                     "hbm_share": hbm / cost, "chip_gross_margin": 1 - cost / b200_asp})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 客戶端 TCO
def tco_per_gpu_hour(rack_capex: float = DEFAULT_RACK_ASP, network_attach: float = 300_000,
                     rack_kw: float = 135, colo_usd_per_kw_month: float = 196,
                     years: int = 5, utilization: float = 0.8) -> dict:
    """簡化 TCO：直線折舊 + 機房託管 (含電力)。colo 單價參考北美批發託管行情。"""
    capex = rack_capex + network_attach
    annual_dep = capex / years
    annual_colo = rack_kw * colo_usd_per_kw_month * 12
    gpu_hours = GPUS_PER_RACK * 24 * 365 * utilization
    return {
        "capex": capex,
        "annual_depreciation": annual_dep,
        "annual_colo": annual_colo,
        "annual_total": annual_dep + annual_colo,
        "utilized_gpu_hours": gpu_hours,
        "cost_per_gpu_hour": (annual_dep + annual_colo) / gpu_hours,
        "capex_share": annual_dep / (annual_dep + annual_colo),
    }
