"""
ETFCollector 테스트 스크립트
"""

import pandas as pd
from datetime import datetime, timedelta

from src.data_collector.etf_collector import ETFCollector
from src.utils.logger import setup_logging

# 로거 초기화 (개발 모드)
logger = setup_logging(development_mode=True)


def create_test_etf_collector(market: str = "KR", months_back: int = 6) -> ETFCollector:
    """
    테스트용 ETF 수집기 생성

    Args:
        market: 시장 (KR, US)
        months_back: 수집할 과거 개월 수

    Returns:
        테스트용 ETFCollector
    """
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=months_back * 30)).strftime(
        "%Y-%m-%d"
    )

    collector = ETFCollector(
        market=market,
        start_date=start_date,
        end_date=end_date,
        include_leveraged=True,
        include_inverse=True,
        batch_size=5,  # 테스트용 작은 배치
        max_retries=2,
        rate_limit_delay=0.1,
    )

    # 테스트용 검증 기준 완화
    collector.min_data_points = 20

    return collector


def test_kr_etf_list_collection():
    """한국 ETF 리스트 수집 테스트"""
    logger.info("=" * 60)
    logger.info("한국 ETF 리스트 수집 테스트")
    logger.info("=" * 60)

    collector = create_test_etf_collector("KR", months_back=3)

    # ETF 리스트 수집
    symbols = collector.get_symbol_list()
    logger.info(f"✅ 한국 ETF 개수: {len(symbols)}개")

    if symbols:
        logger.info(f"📋 주요 ETF 리스트:")
        for i, symbol in enumerate(symbols[:10]):  # 상위 10개만 출력
            etf_info = collector.get_etf_info(symbol)
            if etf_info:
                logger.info(
                    f"   {i+1:2d}. {symbol}: {etf_info.get('Name', 'N/A')} ({etf_info.get('Category', 'N/A')})"
                )
            else:
                logger.info(f"   {i+1:2d}. {symbol}: 정보 없음")

    return symbols


def test_us_etf_list_collection():
    """미국 ETF 리스트 수집 테스트"""
    logger.info("=" * 60)
    logger.info("미국 ETF 리스트 수집 테스트")
    logger.info("=" * 60)

    collector = create_test_etf_collector("US", months_back=3)

    # ETF 리스트 수집
    symbols = collector.get_symbol_list()
    logger.info(f"✅ 미국 ETF 개수: {len(symbols)}개")

    if symbols:
        logger.info(f"📋 주요 ETF 리스트:")
        for i, symbol in enumerate(symbols[:10]):  # 상위 10개만 출력
            etf_info = collector.get_etf_info(symbol)
            if etf_info:
                logger.info(
                    f"   {i+1:2d}. {symbol}: {etf_info.get('Name', 'N/A')} ({etf_info.get('Category', 'N/A')})"
                )
            else:
                logger.info(f"   {i+1:2d}. {symbol}: 정보 없음")

    return symbols


def test_kr_etf_data_collection():
    """한국 ETF 데이터 수집 테스트"""
    logger.info("=" * 60)
    logger.info("한국 ETF 데이터 수집 테스트")
    logger.info("=" * 60)

    collector = create_test_etf_collector("KR", months_back=6)

    # 주요 한국 ETF들로 테스트
    test_symbols = ["069500", "102110", "251340", "143850", "132030"]  # 다양한 카테고리

    logger.info(f"📊 테스트 대상 ETF: {test_symbols}")

    collected_data = collector.run_collection(test_symbols)

    # 결과 분석
    successful_symbols = [s for s, data in collected_data.items() if data is not None]
    logger.info(f"✅ 수집 성공: {len(successful_symbols)}/{len(test_symbols)}개")

    # 상세 결과 출력
    for symbol in test_symbols:
        data = collected_data.get(symbol)
        etf_info = collector.get_etf_info(symbol)
        etf_name = etf_info.get("Name", "Unknown") if etf_info else "Unknown"

        if data is not None:
            logger.info(f"✅ {symbol} ({etf_name}): {len(data)}건 수집 성공")
            logger.info(
                f"   기간: {data.index.min().date()} ~ {data.index.max().date()}"
            )

            if "Close" in data.columns:
                close_price = data["Close"].iloc[-1]
                logger.info(f"   최종 NAV: {close_price:,.0f}원")

                # 수익률 계산
                total_return = ((close_price / data["Close"].iloc[0]) - 1) * 100
                logger.info(f"   기간 수익률: {total_return:.2f}%")

            # ETF 특화 지표 확인
            if "Volatility_20" in data.columns:
                avg_vol = data["Volatility_20"].mean()
                logger.info(f"   평균 변동성: {avg_vol:.4f}")
        else:
            logger.warning(f"❌ {symbol} ({etf_name}): 수집 실패")


def test_us_etf_data_collection():
    """미국 ETF 데이터 수집 테스트"""
    logger.info("=" * 60)
    logger.info("미국 ETF 데이터 수집 테스트")
    logger.info("=" * 60)

    collector = create_test_etf_collector("US", months_back=3)

    # 주요 미국 ETF들로 테스트
    test_symbols = ["SPY", "QQQ", "VTI", "GLD"]  # 대표적인 ETF들

    logger.info(f"📊 테스트 대상 ETF: {test_symbols}")

    collected_data = collector.run_collection(test_symbols)

    # 결과 분석
    successful_symbols = [s for s, data in collected_data.items() if data is not None]
    logger.info(f"✅ 수집 성공: {len(successful_symbols)}/{len(test_symbols)}개")

    # 상세 결과 출력
    for symbol in test_symbols:
        data = collected_data.get(symbol)
        etf_info = collector.get_etf_info(symbol)
        etf_name = etf_info.get("Name", "Unknown") if etf_info else "Unknown"

        if data is not None:
            logger.info(f"✅ {symbol} ({etf_name}): {len(data)}건 수집 성공")
            logger.info(
                f"   기간: {data.index.min().date()} ~ {data.index.max().date()}"
            )

            if "Close" in data.columns:
                close_price = data["Close"].iloc[-1]
                logger.info(f"   최종 NAV: ${close_price:.2f}")

                # 수익률 계산
                total_return = ((close_price / data["Close"].iloc[0]) - 1) * 100
                logger.info(f"   기간 수익률: {total_return:.2f}%")
        else:
            logger.warning(f"❌ {symbol} ({etf_name}): 수집 실패")


def test_etf_filtering_functionality():
    """ETF 필터링 기능 테스트"""
    logger.info("=" * 60)
    logger.info("ETF 필터링 기능 테스트")
    logger.info("=" * 60)

    # 모든 ETF 포함
    collector_all = ETFCollector(
        market="KR",
        include_leveraged=True,
        include_inverse=True,
        start_date="2024-01-01",
        end_date="2024-12-31",
    )

    # 레버리지/인버스 제외
    collector_filtered = ETFCollector(
        market="KR",
        include_leveraged=False,
        include_inverse=False,
        start_date="2024-01-01",
        end_date="2024-12-31",
    )

    symbols_all = collector_all.get_symbol_list()
    symbols_filtered = collector_filtered.get_symbol_list()

    logger.info(f"📊 전체 ETF: {len(symbols_all)}개")
    logger.info(f"📊 필터링 후: {len(symbols_filtered)}개")
    logger.info(f"📊 제외된 ETF: {len(symbols_all) - len(symbols_filtered)}개")

    # 제외된 ETF들 확인
    excluded_symbols = set(symbols_all) - set(symbols_filtered)
    if excluded_symbols:
        logger.info(f"🚫 제외된 ETF들:")
        for symbol in list(excluded_symbols)[:5]:  # 상위 5개만
            etf_info = collector_all.get_etf_info(symbol)
            if etf_info:
                logger.info(f"   {symbol}: {etf_info.get('Name', 'N/A')}")


def test_etf_category_analysis():
    """ETF 카테고리별 분석"""
    logger.info("=" * 60)
    logger.info("ETF 카테고리별 분석")
    logger.info("=" * 60)

    collector = create_test_etf_collector("KR", months_back=6)

    # 카테고리별 대표 ETF
    category_etfs = {
        "Index": ["069500", "102110"],  # KODEX 200, TIGER 200
        "Sector": ["091160", "091170"],  # 반도체, 은행
        "International": ["143850", "133690"],  # 미국나스닥100, 미국S&P500
        "Commodity": ["132030"],  # 골드선물
        "Inverse": ["114800"],  # 인버스
    }

    category_results = {}

    for category, symbols in category_etfs.items():
        logger.info(f"📈 {category} 카테고리 분석:")

        category_performance = []

        for symbol in symbols:
            performance = collector.analyze_etf_performance(symbol)

            if performance:
                etf_info = collector.get_etf_info(symbol)
                etf_name = etf_info.get("Name", "Unknown") if etf_info else "Unknown"

                logger.info(f"   {symbol} ({etf_name}):")
                logger.info(f"     수익률: {performance.get('total_return', 0):.2f}%")
                logger.info(f"     변동성: {performance.get('volatility', 0):.2f}%")
                logger.info(f"     최대낙폭: {performance.get('max_drawdown', 0):.2f}%")

                category_performance.append(performance)

        if category_performance:
            # 카테고리 평균 성과
            avg_return = sum(
                p.get("total_return", 0) for p in category_performance
            ) / len(category_performance)
            avg_volatility = sum(
                p.get("volatility", 0) for p in category_performance
            ) / len(category_performance)

            category_results[category] = {
                "avg_return": avg_return,
                "avg_volatility": avg_volatility,
                "count": len(category_performance),
            }

            logger.info(
                f"   📊 {category} 평균: 수익률 {avg_return:.2f}%, 변동성 {avg_volatility:.2f}%\n"
            )

    # 카테고리별 요약
    if category_results:
        logger.info("🏆 카테고리별 요약:")

        best_return_cat = max(
            category_results.items(), key=lambda x: x[1]["avg_return"]
        )
        lowest_vol_cat = min(
            category_results.items(), key=lambda x: x[1]["avg_volatility"]
        )

        logger.info(
            f"   최고 수익률: {best_return_cat[0]} ({best_return_cat[1]['avg_return']:.2f}%)"
        )
        logger.info(
            f"   최저 변동성: {lowest_vol_cat[0]} ({lowest_vol_cat[1]['avg_volatility']:.2f}%)"
        )


def test_etf_comprehensive():
    """ETF 종합 성능 테스트"""
    logger.info("=" * 60)
    logger.info("ETF 종합 성능 테스트")
    logger.info("=" * 60)

    # 한국과 미국 주요 ETF 각 1개씩 상세 분석
    test_cases = [
        {"market": "KR", "symbol": "069500", "name": "KODEX 200"},
        {"market": "US", "symbol": "SPY", "name": "SPDR S&P 500"},
    ]

    for case in test_cases:
        logger.info(f"🔍 {case['name']} ({case['symbol']}) 종합 분석:")

        collector = ETFCollector(
            market=case["market"], start_date="2024-01-01", end_date="2024-12-31"
        )

        # 데이터 수집
        etf_data = collector.collect_single_etf(case["symbol"])

        if etf_data is not None:
            logger.info(f"   ✅ 데이터 수집 성공: {len(etf_data)}일")

            # 기본 통계
            if "Close" in etf_data.columns:
                nav = etf_data["Close"]
                logger.info(f"   💰 NAV: {nav.iloc[0]:.2f} → {nav.iloc[-1]:.2f}")
                logger.info(f"   📈 수익률: {((nav.iloc[-1]/nav.iloc[0])-1)*100:.2f}%")

            # ETF 특화 지표
            if "Daily_Return" in etf_data.columns:
                returns = etf_data["Daily_Return"].dropna()
                logger.info(
                    f"   📊 일일수익률: 평균 {returns.mean():.4f}, 표준편차 {returns.std():.4f}"
                )

            if "Volume_Ratio" in etf_data.columns:
                vol_ratio = etf_data["Volume_Ratio"].mean()
                logger.info(f"   📊 거래량 비율: {vol_ratio:.2f}")

            # 성과 분석
            performance = collector.analyze_etf_performance(case["symbol"])
            if performance:
                logger.info(
                    f"   🏆 최대낙폭: {performance.get('max_drawdown', 0):.2f}%"
                )
                logger.info(
                    f"   🏆 상승일 비율: {performance.get('positive_days_ratio', 0):.1f}%"
                )

            logger.info("")
        else:
            logger.warning(f"   ❌ {case['name']} 데이터 수집 실패")


if __name__ == "__main__":
    logger.info("ETFCollector 테스트 시작")

    try:
        # ETF 리스트 수집 테스트
        test_kr_etf_list_collection()
        test_us_etf_list_collection()

        # 데이터 수집 테스트
        test_kr_etf_data_collection()
        test_us_etf_data_collection()

        # 고급 기능 테스트
        test_etf_filtering_functionality()
        test_etf_category_analysis()
        test_etf_comprehensive()

        logger.info("=" * 60)
        logger.info("모든 ETF 테스트 완료!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"ETF 테스트 중 오류 발생: {str(e)}")
        import traceback

        logger.error(f"상세 오류:\n{traceback.format_exc()}")
