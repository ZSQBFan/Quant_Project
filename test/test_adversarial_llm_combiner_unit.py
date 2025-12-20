import numpy as np
import pandas as pd
import pandas.testing as pdt

from factors.pipeline.combiners.rolling.ai_combiner import AdversarialLLMCombiner


def _make_combiner(factor_names):
    return AdversarialLLMCombiner(
        factor_names=factor_names,
        rolling_window_days=30,
        allow_negative_weights=True,
    )


def test_combine_handles_non_numeric_payload_and_falls_back_to_equal_weight():
    combiner = _make_combiner(["f1", "f2"])
    daily_factors = pd.DataFrame(
        {"f1": [1.0, 2.0], "f2": [3.0, 4.0]}, index=["assetA", "assetB"]
    )

    payload = {"f1": "invalid", "f2": {"nested": 1}, "f3": np.nan}

    result = combiner._combine_factors_for_day(payload, daily_factors)

    expected_weights = pd.Series([0.5, 0.5], index=["f1", "f2"], dtype=float)
    expected = (daily_factors * expected_weights).sum(axis=1)

    pdt.assert_series_equal(result, expected)


def test_combine_supports_single_asset_series_slice():
    combiner = _make_combiner(["f_single"])
    daily_series = pd.Series({"f_single": 2.0}, name="AssetX")

    result = combiner._combine_factors_for_day({"f_single": 1.5}, daily_series)

    expected = pd.Series([2.0], index=pd.Index(["AssetX"], name="asset"))
    pdt.assert_series_equal(result, expected)
