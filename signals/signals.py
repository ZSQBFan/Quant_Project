# signals.py (已重构的完整文件)
import backtrader as bt
import numpy as np

# 移除了对 pandas 和 scipy.stats 的依赖，因为标准化逻辑已外包

# --- 如何添加新的因子（信号） ---
#
# 要在本框架中添加一个新的因子，请遵循以下步骤：
#
# 1. **创建新类**:
#    - 创建一个继承自 `BaseSignal` 的新 Python 类。
#    - 例如: `class MyNewSignal(BaseSignal):`
#
# 2. **定义参数 (可选)**:
#    - 在类中定义一个 `params` 元组，用于存放该因子所需的所有参数及其默认值。
#    - 格式为 `params = (('param_name_1', default_value_1), ('param_name_2', default_value_2))`。
#
# 3. **实现 `__init__` 方法**:
#    - 构造函数必须接收 `strategy`, `data`, 和 `**kwargs`。
#    - 第一行必须调用父类的构造函数: `super().__init__(strategy, data, **kwargs)`。
#    - 使用 `bt.indicators` 或自定义计算来创建技术指标。
#    - 将最终计算出的原始因子值（它本身必须是一个 Backtrader 的 indicator line 对象）赋给 `self.raw_indicator`。
#    - **重要**: 此处只计算原始值，不要进行任何标准化或归一化操作。
#
# 4. **实现 `get_raw_indicator` 方法**:
#    - 这是 `BaseSignal` 要求的必须实现的方法。
#    - 该方法非常简单，只需返回在 `__init__` 中创建的原始指标即可。
#    - 代码应为: `def get_raw_indicator(self): return self.raw_indicator`
#
# 5. **添加文档字符串**:
#    - 在类定义下方添加清晰的文档字符串，说明因子的计算逻辑、信号值的含义（例如，正值代表什么，负值代表什么）。
#
# 遵循以上模式，每个因子类都会成为一个独立的、可复用的计算单元，并能与系统的其他部分（如标准化、组合）无缝集成。
# --- 均值回归信号 ---


class BaseSignal:
    """
    信号基类 (已重构)。
    
    此版本的基类职责非常纯粹：
    1. 接收策略和特定的数据源 (data feed)。
    2. 初始化子类中定义的原始技术指标。
    3. 提供一个统一的 get_signal() 方法，用于在当前时间点获取指标的原始数值。
    
    所有标准化逻辑都已移除，并转移到 factor_standardizer.py 模块中。
    """
    # 移除了所有与标准化相关的参数
    params = ()

    def __init__(self, strategy, data, **kwargs):
        """
        构造函数接收一个特定的 data feed，确保信号计算与特定股票绑定。
        """
        self.strategy = strategy
        self.data = data
        self.dataclose = self.data.close
        self._setup_params(**kwargs)

    def _setup_params(self, **kwargs):
        all_params = dict(self.params)
        all_params.update(kwargs)
        for key, value in all_params.items():
            setattr(self, key, value)

    def get_raw_indicator(self):
        """
        获取原始的 Backtrader 指标实例。
        每个子类都必须实现此方法，返回在 __init__ 中创建的指标。
        """
        raise NotImplementedError("每个信号子类都必须实现 get_raw_indicator 方法")

    def get_signal(self):
        """
        从此信号获取当前的原始数值。
        
        该方法不再进行任何标准化计算。它直接从原始指标中提取
        当前时间点的值。如果指标还没有足够的计算周期，则返回 NaN。
        """
        raw_indicator = self.get_raw_indicator()
        if len(raw_indicator) > 0:
            return raw_indicator[0]
        return np.nan  # 返回 NaN，由后续的决策逻辑（如 dropna）统一处理


# --- 均值回归信号 ---


class RSISignal(BaseSignal):
    """
    RSI 均值回归信号。
    原始信号值 = 50 - RSI。当 RSI > 50 (超买区) 时信号为负，
    当 RSI < 50 (超卖区) 时信号为正，符合“低买高卖”的逻辑。
    """
    params = (('rsi_period', 14), )

    def __init__(self, strategy, data, **kwargs):
        super().__init__(strategy, data, **kwargs)
        self.rsi = bt.indicators.RelativeStrengthIndex(self.data,
                                                       period=self.rsi_period)
        self.raw_indicator = 50 - self.rsi

    def get_raw_indicator(self):
        return self.raw_indicator


class KDJSignal(BaseSignal):
    """
    KDJ 随机指标信号。
    原始信号值 = K值 - D值。当 K 线上穿 D 线时，信号由负转正，产生买入信号。
    反之，当 K 线下穿 D 线时，信号由正转负，产生卖出信号。
    """
    params = (('period', 9), ('period_dfast', 3), ('period_dslow', 3))

    def __init__(self, strategy, data, **kwargs):
        super().__init__(strategy, data, **kwargs)
        self.kdj = bt.indicators.Stochastic(self.data,
                                            period=self.period,
                                            period_dfast=self.period_dfast,
                                            period_dslow=self.period_dslow)
        self.raw_indicator = self.kdj.lines.percK - self.kdj.lines.percD

    def get_raw_indicator(self):
        return self.raw_indicator


class BollingerBandsSignal(BaseSignal):
    """
    布林带均值回归信号。
    原始信号值被归一化处理，表示价格偏离中轨的程度。
    值为负表示价格在中轨之上，值越大表明离上轨越近（卖出信号越强）。
    值为正表示价格在中轨之下，值越大表明离下轨越近（买入信号越强）。
    """
    params = (('period', 20), ('devfactor', 2.0))

    def __init__(self, strategy, data, **kwargs):
        super().__init__(strategy, data, **kwargs)
        self.b_band = bt.indicators.BollingerBands(self.data,
                                                   period=self.period,
                                                   devfactor=self.devfactor)
        band_width = self.b_band.lines.top - self.b_band.lines.mid
        # 防止布林带宽度为0导致除零错误
        safe_band_width = bt.indicators.Max(band_width, 0.00001)
        # 计算价格与中轨的偏离程度，并用带宽进行归一化，符号取反以符合“低买高卖”
        self.raw_indicator = -(self.dataclose -
                               self.b_band.lines.mid) / safe_band_width

    def get_raw_indicator(self):
        return self.raw_indicator


# --- 趋势跟踪信号 ---


class MovingAverageCrossSignal(BaseSignal):
    """
    均线交叉趋势信号。
    原始信号值 = (短期均线 - 长期均线) / 长期均线 * 100。
    当短期均线上穿长期均线（金叉）时，信号值为正，表示上升趋势。
    当短期均线下穿长期均线（死叉）时，信号值为负，表示下降趋势。
    """
    params = (('short_ma', 14), ('long_ma', 30))

    def __init__(self, strategy, data, **kwargs):
        super().__init__(strategy, data, **kwargs)
        self.short_ma = bt.indicators.SimpleMovingAverage(self.data,
                                                          period=self.short_ma)
        self.long_ma = bt.indicators.SimpleMovingAverage(self.data,
                                                         period=self.long_ma)
        self.raw_indicator = (self.short_ma -
                              self.long_ma) / self.long_ma * 100

    def get_raw_indicator(self):
        return self.raw_indicator


class MACDSignal(BaseSignal):
    """
    MACD 趋势信号。
    原始信号值 = MACD值 / 收盘价 * 100，进行了价格归一化。
    MACD值为正且变大，表示多头趋势增强。
    MACD值为负且变小，表示空头趋势增强。
    """
    params = (('fast_ema', 12), ('slow_ema', 26), ('signal_ema', 9))

    def __init__(self, strategy, data, **kwargs):
        super().__init__(strategy, data, **kwargs)
        self.macd_indicator = bt.indicators.MACD(self.data.close,
                                                 period_me1=self.fast_ema,
                                                 period_me2=self.slow_ema,
                                                 period_signal=self.signal_ema)
        self.raw_indicator = self.macd_indicator.macd / self.dataclose * 100

    def get_raw_indicator(self):
        return self.raw_indicator


class ADXDMISignal(BaseSignal):
    """
    ADX/DMI 趋势强度与方向信号。
    当 ADX > 阈值时，认为存在趋势。
    信号值结合了趋势强度 (ADX) 和趋势方向 (+DI vs -DI)。
    信号值为正表示上升趋势，值为负表示下降趋势，绝对值大小代表趋势强度。
    """
    params = (('period', 14), ('trend_threshold', 20))

    def __init__(self, strategy, data, **kwargs):
        super().__init__(strategy, data, **kwargs)
        self.dmi_indicator = bt.indicators.DirectionalMovementIndex(
            self.data, period=self.period)
        # 使用 CustomIndicator 封装复杂的信号计算逻辑
        self.raw_indicator = bt.indicators.CustomIndicator(
            self.dmi_indicator,
            self,
            func=self._calculate_raw_adx_signal,
            plot=False)

    @staticmethod
    def _calculate_raw_adx_signal(dmi_indicator, self_ref):
        adx = dmi_indicator.adx[0]
        plus_di = dmi_indicator.plus_di[0]
        minus_di = dmi_indicator.minus_di[0]
        # 如果ADX小于阈值，认为无明显趋势，信号为0
        if adx < self_ref.trend_threshold:
            return 0.0
        # 计算方向强度
        if (plus_di + minus_di) == 0:
            direction_strength = 0.0
        else:
            direction_strength = (plus_di - minus_di) / (plus_di + minus_di)
        # 用趋势强度(ADX)对方向强度进行加权
        trend_strength_weight = adx / 100.0
        return direction_strength * trend_strength_weight

    def get_raw_indicator(self):
        return self.raw_indicator


class MomentumSignal(BaseSignal):
    """
    动量信号 (趋势跟踪)。
    原始信号值 = 价格在过去 N 期内的变化百分比 (Rate of Change)。
    正值表示上涨动量，负值表示下跌动量。
    """
    params = (('period', 20), )

    def __init__(self, strategy, data, **kwargs):
        super().__init__(strategy, data, **kwargs)
        self.roc = bt.indicators.RateOfChange100(self.data.close,
                                                 period=self.period)

    def get_raw_indicator(self):
        return self.roc


# --- 其他信号 ---


class VolumeSpikeSignal(BaseSignal):
    """
    成交量突增信号。
    原始信号值 = (当前成交量 - 均线成交量) / 均线成交量。
    信号值表示当前成交量相比于近期平均水平的偏离程度。
    正值表示放量，负值表示缩量。
    """
    params = (('period', 20), )

    def __init__(self, strategy, data, **kwargs):
        super().__init__(strategy, data, **kwargs)
        self.ma_vol = bt.indicators.SimpleMovingAverage(self.data.volume,
                                                        period=self.period)
        # 防止均线成交量为0导致除零错误
        safe_ma_vol = bt.indicators.Max(self.ma_vol, 1.0)
        self.raw_indicator = (self.data.volume - self.ma_vol) / safe_ma_vol

    def get_raw_indicator(self):
        return self.raw_indicator
