import numpy as np
import pandas as pd

from factors.analysis import metrics


def test_calculate_rank_ic_series_all_nan_returns_empty_series():
    index = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2024-01-01"), "AssetA"),
            (pd.Timestamp("2024-01-01"), "AssetB"),
        ],
        names=["date", "asset"],
    )

    factor_data = pd.DataFrame(
        {
            "factor_value": [np.nan, np.nan],
            "forward_return_5d": [np.nan, np.nan],
        },
        index=index,
    )

    result = metrics.calculate_rank_ic_series(factor_data, period=5)

    assert isinstance(result, pd.Series)
    assert result.empty
    assert result.name == "rank_ic_5d"
