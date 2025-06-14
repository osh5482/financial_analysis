"""
데이터 수집기 테스트 패키지
"""

from .test_stock_collector import test_stock_collector_functionality
from .test_etf_collector import test_etf_collector_functionality
from .test_index_collector import test_index_collector_functionality

__all__ = [
    "test_stock_collector_functionality",
    "test_etf_collector_functionality",
    "test_index_collector_functionality",
]
