"""
금융 포트폴리오 분석 시스템 메인 패키지
"""

# 버전 정보
__version__ = "0.1.0"
__author__ = "osh5482"
__email__ = ""
__description__ = "파이썬을 활용한 자동화된 금융 포트폴리오 분석 시스템"

# 로거 초기화
from src.utils.logger import setup_logging

# 프로젝트 전체에서 사용할 기본 로거 설정
logger = setup_logging()
logger.info(f"금융 포트폴리오 분석 시스템 v{__version__} 초기화 완료")
