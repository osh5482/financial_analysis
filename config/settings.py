"""
데이터 수집 설정 파일
- FinanceDataReader를 이용한 데이터 수집 대상 종목 및 설정
- 한국(KRX), 미국(NASDAQ, NYSE) 주식, ETF, 지수, 환율 수집
"""

from pathlib import Path
from datetime import datetime, timedelta

# 프로젝트 루트 디렉토리
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data"

# 데이터 저장 경로 설정
DATA_PATHS = {
    "raw": DATA_ROOT / "raw",
    "processed": DATA_ROOT / "processed",
    "stocks": DATA_ROOT / "raw" / "stocks",
    "etfs": DATA_ROOT / "raw" / "etfs",
    "indices": DATA_ROOT / "raw" / "indices",
    "exchange_rates": DATA_ROOT / "raw" / "exchange_rates",
    "commodities": DATA_ROOT / "raw" / "commodities",
}

# 데이터 수집 기본 설정
DEFAULT_CONFIG = {
    "start_date": "2020-01-01",  # 기본 수집 시작일
    "end_date": None,  # None이면 현재 날짜까지
    "max_retries": 3,  # API 호출 실패시 재시도 횟수
    "retry_delay": 1,  # 재시도 간격 (초)
    "batch_size": 100,  # 배치 처리 크기
    "rate_limit_delay": 0.1,  # API 호출 간격 (초)
}

# 수집 대상 거래소별 주식 리스트 설정
STOCK_EXCHANGES = {
    "KRX": {
        "symbol": "KRX",
        "description": "한국거래소 (KOSPI, KOSDAQ, KONEX 포함)",
        "enabled": True,
        "include_etf": True,  # ETF 포함 여부
        "file_prefix": "krx_stocks",
    },
    "NASDAQ": {
        "symbol": "NASDAQ",
        "description": "나스닥 거래소",
        "enabled": True,
        "include_etf": True,
        "file_prefix": "nasdaq_stocks",
    },
    "NYSE": {
        "symbol": "NYSE",
        "description": "뉴욕증권거래소",
        "enabled": True,
        "include_etf": True,
        "file_prefix": "nyse_stocks",
    },
    "SP500": {
        "symbol": "S&P500",
        "description": "S&P 500 구성 종목",
        "enabled": True,
        "include_etf": False,  # S&P500은 지수 구성종목이므로 ETF 별도 수집
        "file_prefix": "sp500_stocks",
    },
}

# ETF 수집 설정 (국가별)
ETF_MARKETS = {
    "KR": {
        "symbol": "KR",
        "description": "한국 ETF",
        "enabled": True,
        "file_prefix": "kr_etfs",
    },
    "US": {
        "symbol": "US",
        "description": "미국 ETF",
        "enabled": True,
        "file_prefix": "us_etfs",
    },
}

# 주요 지수 수집 설정
MAJOR_INDICES = {
    # 한국 지수
    "KOSPI": {
        "symbol": "KS11",
        "name": "코스피 지수",
        "description": "한국종합주가지수",
        "market": "KR",
        "enabled": True,
    },
    "KOSDAQ": {
        "symbol": "KQ11",
        "name": "코스닥 지수",
        "description": "코스닥종합지수",
        "market": "KR",
        "enabled": True,
    },
    "KOSPI200": {
        "symbol": "KS200",
        "name": "코스피 200",
        "description": "코스피 200 지수",
        "market": "KR",
        "enabled": True,
    },
    # 미국 지수
    "DOW": {
        "symbol": "DJI",
        "name": "다우존스 지수",
        "description": "다우존스 산업평균지수",
        "market": "US",
        "enabled": True,
    },
    "NASDAQ_INDEX": {
        "symbol": "IXIC",
        "name": "나스닥 지수",
        "description": "나스닥 종합지수",
        "market": "US",
        "enabled": True,
    },
    "SP500_INDEX": {
        "symbol": "US500",
        "name": "S&P 500 지수",
        "description": "S&P 500 지수",
        "market": "US",
        "enabled": True,
    },
}

# 환율 수집 설정
EXCHANGE_RATES = {
    "USDKRW": {
        "symbol": "USD/KRW",
        "name": "달러원 환율",
        "description": "미국 달러 대 한국 원 환율",
        "enabled": True,
    },
    "EURKRW": {
        "symbol": "EUR/KRW",
        "name": "유로원 환율",
        "description": "유로 대 한국 원 환율",
        "enabled": False,  # 필요시 활성화
    },
    "JPYKRW": {
        "symbol": "JPY/KRW",
        "name": "엔원 환율",
        "description": "일본 엔 대 한국 원 환율",
        "enabled": False,  # 필요시 활성화
    },
}

# 원자재 수집 설정
COMMODITIES = {
    # 귀금속
    "GOLD": {
        "symbol": "GOLDAMGBD228NLBM",
        "name": "금 가격",
        "description": "런던 불리언 마켓 금 가격 (USD)",
        "category": "precious_metals",
        "data_source": "FRED",
        "enabled": True,
    },
    # 에너지
    "WTI_CRUDE": {
        "symbol": "POILWTIUSDM",
        "name": "WTI 원유",
        "description": "서부 텍사스 중질유 가격 (USD/배럴)",
        "category": "energy",
        "data_source": "FRED",
        "enabled": True,
    },
    "BRENT_CRUDE": {
        "symbol": "POILBREUSDM",
        "name": "브렌트 원유",
        "description": "북해 브렌트 원유 가격 (USD/배럴)",
        "category": "energy",
        "data_source": "FRED",
        "enabled": True,
    },
    "DUBAI_CRUDE": {
        "symbol": "POILDUBUSDM",
        "name": "두바이 원유",
        "description": "두바이 원유 가격 (USD/배럴)",
        "category": "energy",
        "data_source": "FRED",
        "enabled": False,  # 한국 수입에 중요하지만 필요시 활성화
    },
    "NATURAL_GAS": {
        "symbol": "NG",
        "name": "천연가스",
        "description": "천연가스 선물 가격 (NYMEX)",
        "category": "energy",
        "data_source": "FDR",
        "enabled": False,  # 필요시 활성화
    },
    # 원자재 지수
    "DJP": {
        "symbol": "DJP",
        "name": "DJ 원자재 지수 ETF",
        "description": "DJ-UBS 원자재 지수 추종 ETF",
        "category": "commodity_index",
        "data_source": "FDR",
        "enabled": False,  # 필요시 활성화
    },
    "PDBC": {
        "symbol": "PDBC",
        "name": "Invesco 원자재 ETF",
        "description": "Invesco DB 원자재 인덱스 추종 ETF",
        "category": "commodity_index",
        "data_source": "FDR",
        "enabled": False,  # 필요시 활성화
    },
}

# 데이터 검증 설정
VALIDATION_CONFIG = {
    "min_data_points": 100,  # 최소 데이터 포인트 수
    "max_missing_ratio": 0.1,  # 최대 결측치 비율 (10%)
    "price_range_multiplier": 10,  # 가격 이상치 검출 배수
    "volume_zero_threshold": 0.05,  # 거래량 0인 날짜 비율 임계값
}

# 파일 저장 설정
FILE_CONFIG = {
    "csv_encoding": "utf-8-sig",  # CSV 파일 인코딩 (엑셀 호환)
    "parquet_compression": "snappy",  # Parquet 압축 방식
    "date_format": "%Y-%m-%d",  # 날짜 포맷
    "float_precision": 4,  # 소수점 자릿수
}

# 로깅 필터 설정 (데이터 수집 전용)
LOGGING_FILTERS = {
    "data_collection_modules": [
        "data_collector.base_collector",
        "data_collector.stock_collector",
        "data_collector.index_collector",
        "data_collector.currency_collector",
        "data_collector.commodity_collector",
    ]
}


# 활성화된 수집 대상 필터링 함수들
def get_enabled_stock_exchanges() -> dict[str, dict]:
    """활성화된 주식 거래소 설정 반환"""
    return {k: v for k, v in STOCK_EXCHANGES.items() if v.get("enabled", False)}


def get_enabled_etf_markets() -> dict[str, dict]:
    """활성화된 ETF 시장 설정 반환"""
    return {k: v for k, v in ETF_MARKETS.items() if v.get("enabled", False)}


def get_enabled_indices() -> dict[str, dict]:
    """활성화된 지수 설정 반환"""
    return {k: v for k, v in MAJOR_INDICES.items() if v.get("enabled", False)}


def get_enabled_exchange_rates() -> dict[str, dict]:
    """활성화된 환율 설정 반환"""
    return {k: v for k, v in EXCHANGE_RATES.items() if v.get("enabled", False)}


def get_enabled_commodities() -> dict[str, dict]:
    """활성화된 원자재 설정 반환"""
    return {k: v for k, v in COMMODITIES.items() if v.get("enabled", False)}


def get_commodities_by_category(category: str) -> dict[str, dict]:
    """카테고리별 원자재 설정 반환"""
    return {
        k: v
        for k, v in COMMODITIES.items()
        if v.get("category") == category and v.get("enabled", False)
    }


def get_default_date_range() -> tuple[str, str]:
    """기본 날짜 범위 반환 (시작일, 종료일)"""
    start_date = DEFAULT_CONFIG["start_date"]
    end_date = DEFAULT_CONFIG["end_date"] or datetime.now().strftime("%Y-%m-%d")
    return start_date, end_date


def create_data_directories() -> None:
    """데이터 저장 디렉토리 생성"""
    for path in DATA_PATHS.values():
        path.mkdir(parents=True, exist_ok=True)


# 설정 요약 출력 함수
def print_collection_summary() -> None:
    """수집 설정 요약 출력"""
    print("=" * 60)
    print("데이터 수집 설정 요약")
    print("=" * 60)

    print(f"\n📊 주식 거래소 ({len(get_enabled_stock_exchanges())}개):")
    for name, config in get_enabled_stock_exchanges().items():
        print(f"  - {name}: {config['description']}")

    print(f"\n📈 ETF 시장 ({len(get_enabled_etf_markets())}개):")
    for name, config in get_enabled_etf_markets().items():
        print(f"  - {name}: {config['description']}")

    print(f"\n📉 주요 지수 ({len(get_enabled_indices())}개):")
    for name, config in get_enabled_indices().items():
        print(f"  - {config['name']} ({config['symbol']})")

    print(f"\n💱 환율 ({len(get_enabled_exchange_rates())}개):")
    for name, config in get_enabled_exchange_rates().items():
        print(f"  - {config['name']} ({config['symbol']})")

    print(f"\n🥇 원자재 ({len(get_enabled_commodities())}개):")
    for name, config in get_enabled_commodities().items():
        print(f"  - {config['name']} ({config['symbol']}) [{config['category']}]")

    start_date, end_date = get_default_date_range()
    print(f"\n📅 수집 기간: {start_date} ~ {end_date}")
    print("=" * 60)


# 모듈 임포트시 디렉토리 자동 생성
if __name__ == "__main__":
    create_data_directories()
    print_collection_summary()
else:
    create_data_directories()
