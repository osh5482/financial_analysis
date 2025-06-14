"""
설정 파일 테스트 스크립트
"""

from config.settings import (
    get_enabled_stock_exchanges,
    get_enabled_etf_markets,
    get_enabled_indices,
    get_enabled_exchange_rates,
    get_enabled_commodities,
    get_commodities_by_category,
    get_default_date_range,
    print_collection_summary,
    DATA_PATHS,
    DEFAULT_CONFIG,
)


def test_settings():
    """설정 파일 테스트"""
    print("설정 파일 테스트 시작\n")

    # 설정 요약 출력
    print_collection_summary()

    # 개별 설정 확인
    print("\n" + "=" * 50)
    print("상세 설정 확인")
    print("=" * 50)

    # 주식 거래소 설정
    stock_exchanges = get_enabled_stock_exchanges()
    print(f"\n활성화된 주식 거래소: {len(stock_exchanges)}개")
    for name, config in stock_exchanges.items():
        print(f"  {name}: {config['symbol']} - {config['description']}")

    # ETF 시장 설정
    etf_markets = get_enabled_etf_markets()
    print(f"\n활성화된 ETF 시장: {len(etf_markets)}개")
    for name, config in etf_markets.items():
        print(f"  {name}: {config['symbol']} - {config['description']}")

    # 지수 설정
    indices = get_enabled_indices()
    print(f"\n활성화된 지수: {len(indices)}개")
    for name, config in indices.items():
        print(f"  {name}: {config['symbol']} - {config['name']}")

    # 환율 설정
    exchange_rates = get_enabled_exchange_rates()
    print(f"\n활성화된 환율: {len(exchange_rates)}개")
    for name, config in exchange_rates.items():
        print(f"  {name}: {config['symbol']} - {config['name']}")

    # 원자재 설정
    commodities = get_enabled_commodities()
    print(f"\n활성화된 원자재: {len(commodities)}개")
    for name, config in commodities.items():
        print(f"  {name}: {config['symbol']} - {config['name']} [{config['category']}]")

    # 카테고리별 원자재
    print(f"\n카테고리별 원자재:")
    categories = ["precious_metals", "energy", "commodity_index"]
    for category in categories:
        category_commodities = get_commodities_by_category(category)
        print(f"  {category}: {len(category_commodities)}개")
        for name, config in category_commodities.items():
            print(f"    - {config['name']} ({config['symbol']})")

    # 날짜 범위
    start_date, end_date = get_default_date_range()
    print(f"\n기본 수집 기간: {start_date} ~ {end_date}")

    # 데이터 경로 확인
    print(f"\n데이터 저장 경로:")
    for name, path in DATA_PATHS.items():
        print(f"  {name}: {path}")
        print(f"    존재 여부: {path.exists()}")

    # 기본 설정 확인
    print(f"\n기본 설정:")
    for key, value in DEFAULT_CONFIG.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    test_settings()
