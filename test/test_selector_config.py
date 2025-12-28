"""
测试选股器配置读取

验证配置加载器是否正确读取选股器配置
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import load_config


def test_selector_config_loading():
    """测试选股器配置加载"""
    print("=" * 60)
    print("测试选股器配置加载")
    print("=" * 60)

    # 加载配置
    config_loader = load_config()
    backtest_config = config_loader.load_backtest()

    print(f"\n配置加载成功！")
    print(f"选股器类型: {backtest_config.selector}")
    print(f"选股器参数: {backtest_config.selector_params}")

    # 验证
    assert backtest_config.selector is not None, "选股器类型不应为空"
    assert isinstance(backtest_config.selector_params, dict), "选股器参数应该是字典"

    # 根据不同的选股器类型验证参数
    if backtest_config.selector == 'TopN':
        assert 'top_n' in backtest_config.selector_params, "TopN 选股器应该有 top_n 参数"
        print(f"✅ TopN 选股器配置正确: top_n={backtest_config.selector_params['top_n']}")

    elif backtest_config.selector == 'IndustryNeutral':
        assert 'total_stocks' in backtest_config.selector_params, "IndustryNeutral 选股器应该有 total_stocks 参数"
        assert 'strategy' in backtest_config.selector_params, "IndustryNeutral 选股器应该有 strategy 参数"
        print(f"✅ IndustryNeutral 选股器配置正确:")
        print(f"   total_stocks={backtest_config.selector_params['total_stocks']}")
        print(f"   strategy={backtest_config.selector_params['strategy']}")

    print("\n" + "=" * 60)
    print("✅ 测试通过！配置加载逻辑正确")
    print("=" * 60)
    print("\n配置层级说明:")
    print("1. configs/backtest/config.yaml - 选择使用哪个选股器 (selector: TopN)")
    print("2. configs/backtest/pipeline/selectors.yaml - 配置各选股器的参数")
    print("\n切换选股器方法:")
    print("在 configs/backtest/config.yaml 中修改:")
    print("  selector: TopN  # 或 IndustryNeutral")


if __name__ == "__main__":
    test_selector_config_loading()
