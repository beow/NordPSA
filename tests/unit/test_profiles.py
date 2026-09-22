"""Tidsserieprofiler: parametrisk tillrinning, RoR-högfrekvens, kärnkraftstillgänglighet,
CF-höjning."""
import numpy as np
import pandas as pd
import pytest

from nordpsa.inputs import boost_capfac
from nordpsa.profiles.hydro_inflow import add_ror_hifreq, inflow_timeseries
from nordpsa.profiles.nuclear_availability import availability_timeseries

PARAMS = {"A": 3000.0, "mu": 135.0, "sigma": 20.0, "B": 500.0, "phi": 183.0, "C": 800.0}


def test_inflow_hits_annual_targets_and_peaks_in_spring():
    ts = pd.date_range("2023-01-01", "2024-12-31 23:00", freq="3h")
    s = inflow_timeseries(PARAMS, ts, target_annual_twh={2023: 10.0, 2024: 12.0})
    for year, twh in ((2023, 10.0), (2024, 12.0)):
        assert s[s.index.year == year].sum() * 3 / 1e6 == pytest.approx(twh)
    assert (s >= 0).all()
    assert 110 <= s[s.index.year == 2023].idxmax().dayofyear <= 160   # vårfloden


def test_ror_hifreq_keeps_weekly_energy_and_capacity():
    idx = pd.date_range("2024-01-01", periods=24 * 7 * 8, freq="h")
    ror = pd.Series(np.repeat(np.linspace(300, 900, 8), 24 * 7), index=idx)
    p_nom = float(ror.max())
    out = add_ror_hifreq(ror, p_nom, sigma=0.22, tau_days=3.5, seed=7)
    wk = idx.to_period("W-SUN")
    pd.testing.assert_series_equal(out.groupby(wk).sum(), ror.groupby(wk).sum(), rtol=1e-9)
    assert out.max() <= p_nom + 1e-6 and out.min() >= 0.0
    assert out.nunique() > 100                                   # veckotrappan är bruten
    assert add_ror_hifreq(ror, p_nom, sigma=0.0) is ror           # sigma 0 = orört
    pd.testing.assert_series_equal(out, add_ror_hifreq(ror, p_nom, 0.22, 3.5, seed=7))


def test_nuclear_availability_is_deterministic_and_near_target():
    ts = pd.date_range("2023-01-01", "2025-12-31 23:00", freq="h")
    params = {"target_cf": 0.85}
    a = availability_timeseries(params, ts, 6, seed=101)
    b = availability_timeseries(params, ts, 6, seed=101)
    pd.testing.assert_series_equal(a, b)
    assert not a.equals(availability_timeseries(params, ts, 6, seed=102))
    assert a.between(0, 1).all()
    assert a.mean() == pytest.approx(0.85, abs=0.05)


def test_boost_capfac_raises_mean_and_keeps_envelope():
    rng = np.random.default_rng(0)
    cf = pd.DataFrame({"SE-N_wind_onshore": rng.beta(2, 4, 5000) * 0.8,
                       "SE-N_solar": rng.uniform(0, 0.6, 5000)})
    out = boost_capfac(cf, 0.30, "wind_onshore")
    col = "SE-N_wind_onshore"
    assert out[col].mean() == pytest.approx(cf[col].mean() * 1.30, rel=1e-6)
    assert out[col].max() == pytest.approx(cf[col].max())         # märkeffekten orörd
    pd.testing.assert_series_equal(out["SE-N_solar"], cf["SE-N_solar"])   # andra kolumner orörda
    assert boost_capfac(cf, 0.0, "wind_onshore") is cf
