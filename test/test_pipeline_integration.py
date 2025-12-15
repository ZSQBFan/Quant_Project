import pytest
import logging
import math
import sys
import os

# Add project root to path to ensure imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bt.pipeline import (
    TopNSelector,
    EqualWeightAllocator,
    FullPositionManager,
    create_top_n_selector,
    create_equal_weight_allocator,
    create_full_position_manager
)
from logger.logger_config import setup_logging

# Setup logging for tests
setup_logging(log_prefix='pipeline_test')
logger = logging.getLogger(__name__)

class MockData:
    """
    Mock Backtrader DataFeed for testing.
    Simulates the behavior of backtrader data lines (access via [0]).
    """
    def __init__(self, name, signal=None, suspended=False, price=100.0, length=100):
        self._name = name
        self.price = price
        self._len = length
        
        # Simulate lines with list-like access (index 0 is current)
        self.suspended = [suspended]
        self.combined_signal = [signal] if signal is not None else [float('nan')]
        self.close = [price]

    def __len__(self):
        return self._len
    
    def __repr__(self):
        signal_val = self.combined_signal[0]
        return f"MockData(name={self._name}, signal={signal_val}, suspended={self.suspended[0]})"

# --- Fixtures ---

@pytest.fixture
def normal_market_data():
    """Create a set of normal market data with varying signals"""
    return [
        MockData("StockA", signal=0.9, suspended=False),
        MockData("StockB", signal=0.8, suspended=False),
        MockData("StockC", signal=0.7, suspended=False),
        MockData("StockD", signal=0.6, suspended=False),
        MockData("StockE", signal=0.5, suspended=False),
    ]

@pytest.fixture
def mixed_market_data():
    """Create a mixed set of data including suspended and NaN signals"""
    return [
        MockData("StockHigh", signal=0.95, suspended=False),      # Should be selected
        MockData("StockSuspended", signal=0.99, suspended=True),  # High signal but suspended
        MockData("StockNaN", signal=float('nan'), suspended=False), # NaN signal
        MockData("StockMid", signal=0.5, suspended=False),        # Should be selected
        MockData("StockLow", signal=0.1, suspended=False),        # Low signal
        MockData("StockNoData", signal=0.0, length=0),            # No data length
    ]

@pytest.fixture
def pipeline_components():
    """Return initialized pipeline components"""
    return {
        'selector': create_top_n_selector(top_n=2),
        'allocator': create_equal_weight_allocator(),
        'capital_manager': create_full_position_manager(utilization_ratio=0.95)
    }

# --- Component Tests ---

class TestSelector:
    def test_top_n_selection(self, normal_market_data):
        """Test basic Top N selection logic"""
        selector = TopNSelector(top_n=3)
        selected = selector.select(normal_market_data)
        
        assert len(selected) == 3
        # Verify order (descending signal)
        assert selected[0]._name == "StockA"
        assert selected[1]._name == "StockB"
        assert selected[2]._name == "StockC"

    def test_filtering_logic(self, mixed_market_data):
        """Test filtering of suspended, NaN, and empty data"""
        selector = TopNSelector(top_n=5)
        selected = selector.select(mixed_market_data)
        
        selected_names = [d._name for d in selected]
        
        # Should contain valid stocks
        assert "StockHigh" in selected_names
        assert "StockMid" in selected_names
        assert "StockLow" in selected_names
        
        # Should NOT contain invalid stocks
        assert "StockSuspended" not in selected_names
        assert "StockNaN" not in selected_names
        assert "StockNoData" not in selected_names
        
        # Verify count
        assert len(selected) == 3

    def test_empty_input(self):
        """Test behavior with empty input list"""
        selector = TopNSelector(top_n=10)
        selected = selector.select([])
        assert selected == []

class TestAllocator:
    def test_equal_weight_allocation(self, normal_market_data):
        """Test equal weight allocation"""
        allocator = EqualWeightAllocator()
        # Select top 2 manually
        selected = normal_market_data[:2] 
        weights = allocator.allocate(selected)
        
        assert len(weights) == 2
        for stock, weight in weights.items():
            assert weight == 0.5

    def test_single_stock_allocation(self, normal_market_data):
        """Test allocation for a single stock"""
        allocator = EqualWeightAllocator()
        selected = [normal_market_data[0]]
        weights = allocator.allocate(selected)
        
        assert len(weights) == 1
        assert list(weights.values())[0] == 1.0

    def test_empty_allocation(self):
        """Test allocation for empty list"""
        allocator = EqualWeightAllocator()
        weights = allocator.allocate([])
        assert weights == {}

class TestCapitalManager:
    def test_full_position_allocation(self):
        """Test capital allocation calculation"""
        manager = FullPositionManager(utilization_ratio=0.90)
        total_value = 100000.0
        allocation = manager.get_allocation(total_value)
        
        assert allocation == 90000.0
        assert manager.get_reserved_ratio() == pytest.approx(0.10)

    def test_invalid_utilization(self):
        """Test invalid utilization ratio initialization"""
        with pytest.raises(ValueError):
            FullPositionManager(utilization_ratio=1.5)
        with pytest.raises(ValueError):
            FullPositionManager(utilization_ratio=-0.1)

    def test_allocation_boundaries(self):
        """Test boundary conditions for allocation"""
        manager = FullPositionManager(utilization_ratio=0.95)
        
        # Zero value (should raise ValueError as per implementation)
        with pytest.raises(ValueError, match="总价值必须大于0"):
            manager.get_allocation(0)
        
        # Negative value
        with pytest.raises(ValueError, match="总价值必须大于0"):
            manager.get_allocation(-1000)

# --- Integration Tests ---

class TestPipelineIntegration:
    def test_full_pipeline_flow(self, mixed_market_data, pipeline_components):
        """
        Test the complete pipeline flow:
        Data -> Selector -> Allocator -> Capital Manager -> Final Target Values
        """
        selector = pipeline_components['selector']
        allocator = pipeline_components['allocator']
        capital_manager = pipeline_components['capital_manager']
        
        # 1. Selection
        selected_stocks = selector.select(mixed_market_data)
        assert len(selected_stocks) == 2  # Top 2 from 3 valid stocks
        assert selected_stocks[0]._name == "StockHigh"
        
        # 2. Allocation
        weights = allocator.allocate(selected_stocks)
        assert len(weights) == 2
        assert sum(weights.values()) == pytest.approx(1.0)
        
        # 3. Capital Management
        portfolio_value = 1_000_000.0
        trade_capital = capital_manager.get_allocation(portfolio_value)
        assert trade_capital == 950_000.0  # 95% utilization
        
        # 4. Final Target Calculation (Simulation of Strategy Logic)
        target_values = {}
        for stock, weight in weights.items():
            target_value = trade_capital * weight
            target_values[stock._name] = target_value
            
            # Verify calculation
            expected_value = 950_000.0 * 0.5  # Equal weight (1/2)
            assert target_value == expected_value
            
        logger.info(f"Pipeline Integration Test Complete. Targets: {target_values}")

    def test_pipeline_no_valid_stocks(self, pipeline_components):
        """Test pipeline behavior when no stocks meet criteria"""
        # Create data where all are invalid
        bad_data = [
            MockData("Suspended", suspended=True),
            MockData("NaN", signal=float('nan')),
        ]
        
        selector = pipeline_components['selector']
        allocator = pipeline_components['allocator']
        
        # 1. Selection
        selected = selector.select(bad_data)
        assert len(selected) == 0
        
        # 2. Allocation
        weights = allocator.allocate(selected)
        assert weights == {}
        
        # 3. Verify downstream handling
        if not weights:
            logger.info("Pipeline correctly handled no valid stocks (empty weights)")
        else:
            pytest.fail("Pipeline generated weights for invalid stocks")

    def test_pipeline_single_stock_case(self, pipeline_components):
        """Test pipeline with only one valid stock available"""
        single_data = [
            MockData("LoneStar", signal=0.8),
            MockData("Suspended", suspended=True)
        ]
        
        selector = pipeline_components['selector']
        allocator = pipeline_components['allocator']
        capital_manager = pipeline_components['capital_manager']
        
        # Select
        selected = selector.select(single_data)
        assert len(selected) == 1
        assert selected[0]._name == "LoneStar"
        
        # Allocate
        weights = allocator.allocate(selected)
        assert weights[selected[0]] == 1.0
        
        # Capital
        cap = capital_manager.get_allocation(10000)
        target_value = cap * weights[selected[0]]
        
        assert target_value == 9500.0  # 10000 * 0.95 * 1.0

if __name__ == "__main__":
    # Allow running this script directly
    sys.exit(pytest.main(["-v", __file__]))