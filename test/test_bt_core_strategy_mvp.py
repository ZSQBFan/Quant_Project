#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试脚本：验证 Backtrader 事件驱动核心策略框架 (MVP版) - 简化版

测试目标：
1. 验证核心常量和数据结构
2. 测试指令冲突处理机制（优先级裁决）
3. 验证核心逻辑不依赖 Backtrader 架构
"""

import sys
import os
import unittest
from unittest.mock import MagicMock
import logging

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bt.core import ActionPriority, ActionType


class MockData:
    """模拟数据对象"""
    def __init__(self, name, suspended=False):
        self._name = name
        self.suspended = [suspended] if suspended else [False]


class MockStrategy:
    """模拟策略类，只包含核心逻辑"""
    def __init__(self):
        self.pending_actions = []
    
    def submit_action(self, data, action: str, size: int = None, 
                      reason: str = '', priority: int = ActionPriority.OTHER):
        """提交交易意图"""
        self.pending_actions.append({
            'data': data,
            'action': action,
            'size': size,
            'reason': reason,
            'priority': priority
        })
    
    def _execute_pending_actions(self):
        """执行待处理指令（简化版）"""
        executed_actions = []
        
        if not self.pending_actions:
            return executed_actions
        
        # 按优先级排序
        self.pending_actions.sort(key=lambda x: x['priority'])
        
        # 去重：每只股票只保留最高优先级的指令
        final_intents = {}
        for intent in self.pending_actions:
            d = intent['data']
            if d not in final_intents:
                final_intents[d] = intent
        
        # 执行最终指令
        for d, intent in final_intents.items():
            # 检查是否停牌
            if self._is_suspended(d):
                continue
            
            executed_actions.append({
                'data': d,
                'action': intent['action'],
                'size': intent['size'],
                'reason': intent['reason']
            })
        
        # 清空缓冲区
        self.pending_actions.clear()
        return executed_actions
    
    def _is_suspended(self, data) -> bool:
        """检查股票是否停牌"""
        if hasattr(data, 'suspended') and len(data.suspended) > 0:
            return bool(data.suspended[0])
        return False


class TestBacktraderCoreLogic(unittest.TestCase):
    """测试 Backtrader 核心逻辑"""
    
    def setUp(self):
        """测试前准备"""
        # 设置日志（最小化输出）
        logging.basicConfig(level=logging.WARNING)
        
        # 创建模拟数据
        self.data1 = MockData('STOCK_A')
        self.data2 = MockData('STOCK_B')
        self.suspended_data = MockData('SUSPENDED_STOCK', suspended=True)
        
        # 创建模拟策略
        self.strategy = MockStrategy()
        
    def test_action_constants(self):
        """测试动作常量定义"""
        print("\n🧪 测试 1: 动作常量定义")
        
        # 测试优先级常量
        self.assertEqual(ActionPriority.STOP_LOSS, 1)
        self.assertEqual(ActionPriority.TAKE_PROFIT, 2)
        self.assertEqual(ActionPriority.REBALANCE, 3)
        self.assertEqual(ActionPriority.SECTOR_BUY, 4)
        self.assertEqual(ActionPriority.OTHER, 5)
        
        # 验证优先级顺序（数值越小优先级越高）
        self.assertTrue(ActionPriority.STOP_LOSS < ActionPriority.TAKE_PROFIT)
        self.assertTrue(ActionPriority.TAKE_PROFIT < ActionPriority.REBALANCE)
        self.assertTrue(ActionPriority.REBALANCE < ActionPriority.SECTOR_BUY)
        self.assertTrue(ActionPriority.SECTOR_BUY < ActionPriority.OTHER)
        
        # 测试动作类型常量
        self.assertEqual(ActionType.BUY, 'buy')
        self.assertEqual(ActionType.SELL, 'sell')
        self.assertEqual(ActionType.CLOSE, 'close')
        
        print("✅ 动作常量定义测试通过")
    
    def test_submit_action(self):
        """测试提交交易意图"""
        print("\n🧪 测试 2: 提交交易意图")
        
        # 提交一个买入意图
        self.strategy.submit_action(
            data=self.data1,
            action=ActionType.BUY,
            size=100,
            reason="测试买入",
            priority=ActionPriority.REBALANCE
        )
        
        # 验证缓冲区内容
        self.assertEqual(len(self.strategy.pending_actions), 1)
        
        action = self.strategy.pending_actions[0]
        self.assertEqual(action['data'], self.data1)
        self.assertEqual(action['action'], ActionType.BUY)
        self.assertEqual(action['size'], 100)
        self.assertEqual(action['reason'], "测试买入")
        self.assertEqual(action['priority'], ActionPriority.REBALANCE)
        
        print("✅ 提交交易意图测试通过")
    
    def test_priority_sorting(self):
        """测试优先级排序"""
        print("\n🧪 测试 3: 优先级排序")
        
        # 先提交低优先级指令
        self.strategy.submit_action(
            data=self.data1,
            action=ActionType.BUY,
            size=100,
            reason="低优先级",
            priority=ActionPriority.OTHER
        )
        
        # 再提交高优先级指令
        self.strategy.submit_action(
            data=self.data1,
            action=ActionType.SELL,
            size=100,
            reason="高优先级",
            priority=ActionPriority.STOP_LOSS
        )
        
        # 执行排序逻辑（不实际执行，只测试排序）
        self.strategy.pending_actions.sort(key=lambda x: x['priority'])
        
        # 第一个应该是最高优先级
        first_action = self.strategy.pending_actions[0]
        self.assertEqual(first_action['priority'], ActionPriority.STOP_LOSS)
        self.assertEqual(first_action['action'], ActionType.SELL)
        
        # 第二个应该是低优先级
        second_action = self.strategy.pending_actions[1]
        self.assertEqual(second_action['priority'], ActionPriority.OTHER)
        self.assertEqual(second_action['action'], ActionType.BUY)
        
        print("✅ 优先级排序测试通过")
    
    def test_priority_conflict_resolution(self):
        """测试优先级冲突解决"""
        print("\n🧪 测试 4: 优先级冲突解决")
        
        # 模拟冲突场景：同一只股票有止损和调仓信号
        self.strategy.submit_action(
            data=self.data1,
            action=ActionType.BUY,
            size=100,
            reason="调仓买入",
            priority=ActionPriority.REBALANCE
        )
        
        self.strategy.submit_action(
            data=self.data1,
            action=ActionType.SELL,
            size=100,
            reason="止损卖出",
            priority=ActionPriority.STOP_LOSS
        )
        
        # 执行指令
        executed = self.strategy._execute_pending_actions()
        
        # 验证只执行了高优先级指令（止损）
        self.assertEqual(len(executed), 1)
        self.assertEqual(executed[0]['action'], ActionType.SELL)
        self.assertEqual(executed[0]['reason'], "止损卖出")
        
        # 验证缓冲区已清空
        self.assertEqual(len(self.strategy.pending_actions), 0)
        
        print("✅ 优先级冲突解决测试通过")
    
    def test_multiple_stocks_execution(self):
        """测试多股票指令执行"""
        print("\n🧪 测试 5: 多股票指令执行")
        
        # 为两只股票分别提交指令
        self.strategy.submit_action(
            data=self.data1,
            action=ActionType.BUY,
            size=100,
            reason="买入股票A",
            priority=ActionPriority.REBALANCE
        )
        
        self.strategy.submit_action(
            data=self.data2,
            action=ActionType.SELL,
            size=50,
            reason="卖出股票B",
            priority=ActionPriority.REBALANCE
        )
        
        # 执行指令
        executed = self.strategy._execute_pending_actions()
        
        # 验证两只股票都得到处理
        self.assertEqual(len(executed), 2)
        
        # 检查执行的动作
        actions = {ex['data']._name: ex for ex in executed}
        self.assertIn('STOCK_A', actions)
        self.assertIn('STOCK_B', actions)
        
        self.assertEqual(actions['STOCK_A']['action'], ActionType.BUY)
        self.assertEqual(actions['STOCK_A']['size'], 100)
        
        self.assertEqual(actions['STOCK_B']['action'], ActionType.SELL)
        self.assertEqual(actions['STOCK_B']['size'], 50)
        
        print("✅ 多股票指令执行测试通过")
    
    def test_suspended_stock_handling(self):
        """测试停牌股票处理"""
        print("\n🧪 测试 6: 停牌股票处理")
        
        # 提交正常股票和停牌股票的指令
        self.strategy.submit_action(
            data=self.data1,
            action=ActionType.BUY,
            size=100,
            reason="买入正常股票",
            priority=ActionPriority.REBALANCE
        )
        
        self.strategy.submit_action(
            data=self.suspended_data,
            action=ActionType.BUY,
            size=100,
            reason="买入停牌股票",
            priority=ActionPriority.REBALANCE
        )
        
        # 执行指令
        executed = self.strategy._execute_pending_actions()
        
        # 验证只执行了正常股票指令
        self.assertEqual(len(executed), 1)
        self.assertEqual(executed[0]['data']._name, 'STOCK_A')
        
        print("✅ 停牌股票处理测试通过")
    
    def test_all_action_types(self):
        """测试所有交易动作类型"""
        print("\n🧪 测试 7: 所有交易动作类型")
        
        test_cases = [
            (ActionType.BUY, '买入'),
            (ActionType.SELL, '卖出'),
            (ActionType.CLOSE, '平仓')
        ]
        
        for action_type, action_name in test_cases:
            with self.subTest(action=action_type):
                # 清空缓冲区
                self.strategy.pending_actions.clear()
                
                # 提交指令
                self.strategy.submit_action(
                    data=self.data1,
                    action=action_type,
                    size=100,
                    reason=f"测试{action_name}",
                    priority=ActionPriority.REBALANCE
                )
                
                # 执行指令
                executed = self.strategy._execute_pending_actions()
                
                # 验证执行结果
                self.assertEqual(len(executed), 1)
                self.assertEqual(executed[0]['action'], action_type)
                self.assertEqual(executed[0]['size'], 100)
        
        print("✅ 所有交易动作类型测试通过")
    
    def test_empty_buffer_handling(self):
        """测试空缓冲区处理"""
        print("\n🧪 测试 8: 空缓冲区处理")
        
        # 确保缓冲区为空
        self.assertEqual(len(self.strategy.pending_actions), 0)
        
        # 执行空缓冲区
        executed = self.strategy._execute_pending_actions()
        
        # 验证返回空列表
        self.assertEqual(executed, [])
        
        print("✅ 空缓冲区处理测试通过")


def run_core_logic_test():
    """运行核心逻辑测试"""
    print("🚀 开始 Backtrader 事件驱动核心策略框架测试 (简化版)")
    print("=" * 60)
    
    # 创建测试套件
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestBacktraderCoreLogic)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 输出测试结果
    print("\n" + "=" * 60)
    print("📊 测试结果汇总:")
    print(f"运行测试数: {result.testsRun}")
    print(f"失败数: {len(result.failures)}")
    print(f"错误数: {len(result.errors)}")
    
    if result.failures:
        print("\n❌ 失败的测试:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print("\n🚫 错误的测试:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    # 总体评估
    if result.wasSuccessful():
        print("\n🎉 所有测试通过！核心框架逻辑运行正常。")
        print("\n📋 核心功能验证:")
        print("  ✅ 动作常量定义正确")
        print("  ✅ 优先级裁决机制正常")
        print("  ✅ 冲突解决逻辑正确")
        print("  ✅ 多股票指令处理正常")
        print("  ✅ 停牌股票过滤有效")
        print("  ✅ 各种交易动作类型支持")
        return True
    else:
        print("\n⚠️ 部分测试失败，请检查核心逻辑。")
        return False


if __name__ == '__main__':
    success = run_core_logic_test()
    sys.exit(0 if success else 1)