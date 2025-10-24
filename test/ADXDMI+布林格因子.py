def calculate_adx_dmi_factor(df: pd.DataFrame,
                             period: int = 14,
                             trend_threshold: int = 20) -> pd.Series:
    """
    计算 ADX/DMI 趋势因子。
    结合了趋势强度(ADX)和方向(DMI)，当ADX低于阈值时认为无趋势（因子值为0）。
    """
    if ta is None: raise ImportError("pandas-ta 未安装")
    dmi = ta.adx(df['high'], df['low'], df['close'], length=period)

    adx = dmi[f'ADX_{period}']
    plus_di = dmi[f'DMP_{period}']
    minus_di = dmi[f'DMN_{period}']

    direction_strength = (plus_di - minus_di) / (plus_di + minus_di).replace(
        0, np.nan)
    trend_strength_weight = adx / 100.0

    # 核心逻辑：只有当ADX大于阈值时，信号才有效
    factor_series = pd.Series(np.where(
        adx > trend_threshold, direction_strength * trend_strength_weight,
        0.0),
                              index=df.index)

    return factor_series


def calculate_bollinger_bands_factor(df: pd.DataFrame,
                                     period: int = 20,
                                     devfactor: float = 2.0) -> pd.Series:
    """
    计算布林带因子（已更新为动态列名查找，更稳健）。
    因子值表示价格偏离中轨的归一化程度，符号取反以符合“低买高卖”。
    因子值为正表示价格在中轨之下，值越大表明离下轨越近（买入信号）。
    """
    if ta is None: raise ImportError("pandas-ta 未安装")
    bbands = ta.bbands(df['close'], length=period, std=devfactor)

    # 健壮性检查：如果数据不足无法计算，则返回一个填充为0的Series
    if bbands is None or bbands.empty:
        return pd.Series(0, index=df.index)

    # 【【核心修复】】不再硬编码列名，而是通过关键字动态查找
    # 这样做可以适应不同 pandas-ta 版本生成的不同列名格式
    try:
        # 查找包含 'BBM' (布林中轨) 的列名
        mid_band_col = [col for col in bbands.columns if 'BBM' in col][0]
        # 查找包含 'BBU' (布林上轨) 的列名
        top_band_col = [col for col in bbands.columns if 'BBU' in col][0]
    except IndexError:
        # 如果没有找到对应的列，则打印错误并返回，防止程序崩溃
        print(f"错误：无法在 {bbands.columns.tolist()} 中找到包含 'BBM' 或 'BBU' 的列")
        return pd.Series(0, index=df.index)

    mid_band = bbands[mid_band_col]
    top_band = bbands[top_band_col]

    band_width = top_band - mid_band
    # 防止除零错误
    safe_band_width = band_width.replace(0, np.nan)

    return -(df['close'] - mid_band) / safe_band_width
