"""Kostnadsannualisering, budtrappans geometri och terminalkurvans egenskaper."""
import math

import pandas as pd
import pytest

from nordpsa.network.costs import annualized_cost, crf
from nordpsa.wv import terminal_curve as tc


def test_crf_matches_annuity_formula():
    r, L = 0.06, 60
    assert crf(L, r) == pytest.approx(r * (1 + r) ** L / ((1 + r) ** L - 1))
    # CRF × nuvärdet av L årliga betalningar = 1
    assert crf(L, r) * sum((1 + r) ** -t for t in range(1, L + 1)) == pytest.approx(1.0)


def test_annualized_cost_is_eur_per_mw_year():
    # 7 EUR/W = 7e6 EUR/MW; (CRF + FOM) × overnight
    assert annualized_cost(7.0, 60, 0.06, 0.02) == pytest.approx(7e6 * (crf(60, 0.06) + 0.02))


@pytest.mark.parametrize("K, W", [(3, 36), (5, 30), (5, 34.6)])
def test_bid_ladder_offsets_are_centred_and_sd_matches_formula(K, W):
    offs = [W * ((k + 0.5) / K - 0.5) for k in range(K)]
    assert sum(offs) == pytest.approx(0.0)
    assert offs == sorted(offs)                                  # stigande: billigaste först
    assert max(offs) - min(offs) == pytest.approx(W * (K - 1) / K)   # spannet ≠ bredden
    sd = math.sqrt(sum(o * o for o in offs) / K)
    assert sd == pytest.approx(W * math.sqrt((1 - 1 / K ** 2) / 12))


def test_production_curves_load():
    params, anchor = tc.load_params(None)
    assert set(params) == {"SE-N", "SE-S", "NO-N", "NO-S", "FI"}
    assert all(p.b_mean == 4.0 and p.b_amp == 0.0 for p in params.values())
    _, anchor_exp = tc.load_params("config/terminal_curves/terminal_curve_2040_gemini_v12_exp73.yaml")
    assert set(anchor_exp.values()) == {73.0}


def test_seasonal_level_has_annual_mean_one():
    params, _ = tc.load_params(None)
    for p in params.values():
        assert sum(tc.a_factor(w, p) for w in range(1, 53)) / 52 == pytest.approx(1.0, abs=1e-12)


def test_segment_profile_is_non_increasing_and_mid_normalised():
    prof = tc.segment_profile(4.0, segments=5, norm="mid", x_ref=0.5)
    assert all(a >= b for a, b in zip(prof, prof[1:]))
    assert prof[2] == pytest.approx(1.0)                         # mittsegmentet ligger på x_ref
    assert tc.segment_profile(0.0, segments=4, norm="mid") == [1.0] * 4   # b = 0 → platt


def test_expansion_mc_equals_anchor_times_level_along_normal_path():
    params, _ = tc.load_params(None)
    sn = pd.date_range("2024-01-01", periods=52 * 7, freq="D")
    anchor = {z: 73.0 for z in params}
    mc = tc.hydro_mc_from_curve(sn, list(params), params, anchor)
    for z, s in mc.items():
        expect = [73.0 * tc.a_factor(tc.week_of(t), params[z]) for t in sn]
        assert s.tolist() == pytest.approx(expect)
