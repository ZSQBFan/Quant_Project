"""
调仓日历生成器

根据不同的调仓策略生成调仓日期列表。
支持：每日、每周、每月、每N个交易日调仓。
"""

import pandas as pd
from typing import List, Optional, Literal
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class RebalanceCalendar:
    """
    调仓日历生成器

    根据配置的调仓频率和策略，自动生成调仓日期列表。
    """

    def __init__(
        self,
        start_date: str,
        end_date: str,
        frequency: Literal["daily", "weekly", "monthly", "interval"] = "monthly",
        weekday: Optional[int] = None,
        monthday: Optional[int] = None,
        interval_days: Optional[int] = None,
        trading_calendar: Optional[pd.DatetimeIndex] = None
    ):
        """
        初始化调仓日历生成器

        Args:
            start_date: 回测开始日期 (YYYY-MM-DD)
            end_date: 回测结束日期 (YYYY-MM-DD)
            frequency: 调仓频率
                - "daily": 每日调仓
                - "weekly": 每周调仓
                - "monthly": 每月调仓
                - "interval": 每隔N个交易日调仓
            weekday: 每周调仓时，指定周几 (0=周一, 4=周五)
            monthday: 每月调仓时，指定每月几号 (1-31)
                      如果该月没有该日期，使用该月最后一天
            interval_days: 间隔调仓时，间隔的交易日数
            trading_calendar: 交易日历（可选），如果提供则只使用交易日
        """
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.frequency = frequency
        self.weekday = weekday
        self.monthday = monthday
        self.interval_days = interval_days
        self.trading_calendar = trading_calendar

        # 验证参数
        self._validate_params()

    def _validate_params(self):
        """验证参数有效性"""
        if self.start_date > self.end_date:
            raise ValueError(f"开始日期 {self.start_date} 不能晚于结束日期 {self.end_date}")

        if self.frequency == "weekly" and self.weekday is None:
            raise ValueError("周调仓模式下必须指定 weekday (0-6)")

        if self.frequency == "weekly" and not (0 <= self.weekday <= 6):
            raise ValueError(f"weekday 必须在 0-6 之间，当前值: {self.weekday}")

        if self.frequency == "monthly" and self.monthday is None:
            self.monthday = 1  # 默认每月第一天
            logger.info("月调仓模式未指定 monthday，使用默认值 1 (每月第一天)")

        if self.frequency == "monthly" and not (1 <= self.monthday <= 31):
            raise ValueError(f"monthday 必须在 1-31 之间，当前值: {self.monthday}")

        if self.frequency == "interval" and self.interval_days is None:
            raise ValueError("间隔调仓模式下必须指定 interval_days")

        if self.frequency == "interval" and self.interval_days < 1:
            raise ValueError(f"interval_days 必须大于等于 1，当前值: {self.interval_days}")

    def generate(self) -> List[str]:
        """
        生成调仓日期列表

        Returns:
            调仓日期列表，格式为 YYYY-MM-DD
        """
        if self.frequency == "daily":
            return self._generate_daily()
        elif self.frequency == "weekly":
            return self._generate_weekly()
        elif self.frequency == "monthly":
            return self._generate_monthly()
        elif self.frequency == "interval":
            return self._generate_interval()
        else:
            raise ValueError(f"不支持的调仓频率: {self.frequency}")

    def _generate_daily(self) -> List[str]:
        """生成每日调仓日期"""
        if self.trading_calendar is not None:
            # 使用交易日历
            dates = self.trading_calendar[
                (self.trading_calendar >= self.start_date) &
                (self.trading_calendar <= self.end_date)
            ]
        else:
            # 使用工作日（周一到周五）
            dates = pd.bdate_range(start=self.start_date, end=self.end_date)

        result = [d.strftime('%Y-%m-%d') for d in dates]
        logger.info(f"生成每日调仓日历: {len(result)} 个交易日")
        return result

    def _generate_weekly(self) -> List[str]:
        """生成每周调仓日期"""
        if self.trading_calendar is not None:
            # 使用交易日历
            all_dates = self.trading_calendar[
                (self.trading_calendar >= self.start_date) &
                (self.trading_calendar <= self.end_date)
            ]
        else:
            # 使用工作日
            all_dates = pd.bdate_range(start=self.start_date, end=self.end_date)

        # 筛选指定的星期几
        # weekday: 0=周一, 6=周日
        dates = all_dates[all_dates.weekday == self.weekday]

        result = [d.strftime('%Y-%m-%d') for d in dates]
        weekday_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
        logger.info(f"生成每周调仓日历 ({weekday_names[self.weekday]}): {len(result)} 个调仓日")
        return result

    def _generate_monthly(self) -> List[str]:
        """生成每月调仓日期"""
        dates = []
        current = self.start_date.replace(day=1)  # 从当月第一天开始

        while current <= self.end_date:
            # 计算该月的目标日期
            year, month = current.year, current.month

            # 获取该月最后一天
            if month == 12:
                next_month = current.replace(year=year + 1, month=1, day=1)
            else:
                next_month = current.replace(month=month + 1, day=1)
            last_day = (next_month - timedelta(days=1)).day

            # 如果指定的日期大于该月天数，使用该月最后一天
            target_day = min(self.monthday, last_day)
            target_date = current.replace(day=target_day)

            # 检查是否在范围内
            if self.start_date <= target_date <= self.end_date:
                # 如果提供了交易日历，找到最近的交易日
                if self.trading_calendar is not None:
                    valid_dates = self.trading_calendar[self.trading_calendar >= target_date]
                    if len(valid_dates) > 0:
                        target_date = valid_dates[0]
                # 如果是周末，调整到下一个工作日
                elif target_date.weekday() >= 5:
                    # 5=周六, 6=周日
                    days_to_add = 7 - target_date.weekday()
                    target_date = target_date + timedelta(days=days_to_add)

                # 再次检查是否在范围内（调整后可能超出）
                if target_date <= self.end_date:
                    dates.append(target_date)

            # 移动到下个月
            if month == 12:
                current = current.replace(year=year + 1, month=1, day=1)
            else:
                current = current.replace(month=month + 1, day=1)

        result = [d.strftime('%Y-%m-%d') for d in dates]
        logger.info(f"生成每月调仓日历 (每月{self.monthday}号): {len(result)} 个调仓日")
        return result

    def _generate_interval(self) -> List[str]:
        """生成间隔N个交易日的调仓日期"""
        if self.trading_calendar is not None:
            # 使用交易日历
            all_dates = self.trading_calendar[
                (self.trading_calendar >= self.start_date) &
                (self.trading_calendar <= self.end_date)
            ]
        else:
            # 使用工作日
            all_dates = pd.bdate_range(start=self.start_date, end=self.end_date)

        if len(all_dates) == 0:
            logger.warning("没有找到任何交易日")
            return []

        # 从第一个交易日开始，每隔interval_days个交易日选一个
        dates = all_dates[::self.interval_days]

        result = [d.strftime('%Y-%m-%d') for d in dates]
        logger.info(f"生成间隔调仓日历 (每{self.interval_days}个交易日): {len(result)} 个调仓日")
        return result


def create_rebalance_calendar(
    start_date: str,
    end_date: str,
    config: dict,
    trading_calendar: Optional[pd.DatetimeIndex] = None
) -> List[str]:
    """
    根据配置创建调仓日历

    Args:
        start_date: 回测开始日期
        end_date: 回测结束日期
        config: 调仓配置字典，格式：
            {
                "frequency": "monthly",  # daily/weekly/monthly/interval
                "weekday": 0,           # 周调仓时使用
                "monthday": 1,          # 月调仓时使用
                "interval_days": 5      # 间隔调仓时使用
            }
        trading_calendar: 交易日历（可选）

    Returns:
        调仓日期列表
    """
    frequency = config.get("frequency", "monthly")
    weekday = config.get("weekday")
    monthday = config.get("monthday", 1)
    interval_days = config.get("interval_days")

    calendar = RebalanceCalendar(
        start_date=start_date,
        end_date=end_date,
        frequency=frequency,
        weekday=weekday,
        monthday=monthday,
        interval_days=interval_days,
        trading_calendar=trading_calendar
    )

    return calendar.generate()
