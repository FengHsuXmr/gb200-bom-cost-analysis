import pytest

from src import model


@pytest.fixture(scope="module")
def bom():
    return model.load_bom()


def test_bom_ranges_are_ordered(bom):
    assert (bom["unit_cost_low"] <= bom["unit_cost_base"]).all()
    assert (bom["unit_cost_base"] <= bom["unit_cost_high"]).all()


def test_item_ids_unique(bom):
    assert bom["item_id"].is_unique


def test_total_cost_is_sum_of_lines(bom):
    rc = model.RackCost(bom)
    assert rc.total_cost == pytest.approx((bom["qty_per_rack"] * bom["unit_cost_base"]).sum())
    assert rc.by_subsystem()["share_of_cost"].sum() == pytest.approx(1.0)


def test_base_margin_is_mid_single_digit(bom):
    # 研調普遍認為 GB200 機櫃 ODM 毛利率落在個位數
    m = model.RackCost(bom).odm_gross_margin
    assert 0.0 < m < 0.10


def test_compute_share_matches_public_estimate(bom):
    # 大摩估 GPU 約占整櫃售價 72%；HSBC 估超級晶片合計 216~252 萬美元
    rc = model.RackCost(bom)
    chip = bom.loc[bom["item_id"] == "C01", "ext_base"].iloc[0]
    assert 0.65 <= chip / rc.rack_asp <= 0.80


def test_levers_reduce_cost(bom):
    cd = model.cost_down_table(bom)
    assert (cd["saving_usd"] > 0).all()
    after = model.apply_levers(bom, model.DEFAULT_LEVERS)["ext_base"].sum()
    assert after < bom["ext_base"].sum()


def test_monte_carlo_within_bounds(bom):
    mc = model.monte_carlo(bom, n=2000)
    assert bom["ext_low"].sum() <= mc["p10_total"] <= mc["p50_total"] <= mc["p90_total"] <= bom["ext_high"].sum()


def test_should_cost_close_to_epoch():
    sc = model.should_cost_summary(model.load_should_cost())
    assert sc.attrs["total"] == pytest.approx(6400, rel=0.05)
