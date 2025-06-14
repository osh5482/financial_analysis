"""
StockCollector 테스트 스크립트
"""

import pandas as pd
from datetime import datetime, timedelta

from src.data_collector.stock_collector import StockCollector
from src.utils.logger import setup_logging

# 로거 초기화 (개발 모드)
logger = setup_logging(development_mode=True)


def create_test_collector(
    exchange: str = "KRX", months_back: int = 6
) -> StockCollector:
    """
    테스트용 수집기 생성 (더 긴 기간 + 느슨한 검증)

    Args:
        exchange: 거래소
        months_back: 수집할 과거 개월 수

    Returns:
        테스트용 StockCollector
    """
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=months_back * 30)).strftime(
        "%Y-%m-%d"
    )

    collector = StockCollector(
        exchange=exchange,
        start_date=start_date,
        end_date=end_date,
        batch_size=5,  # 테스트용 작은 배치
        max_retries=2,
        rate_limit_delay=0.1,
    )

    # 테스트용 검증 기준 완화
    collector.min_data_points = 20  # 100 → 20으로 완화

    return collector


def test_krx_stock_collector():
    """한국 주식 수집기 테스트"""
    logger.info("=" * 60)
    logger.info("한국 주식 수집기 테스트 시작")
    logger.info("=" * 60)

    # 6개월 데이터로 테스트
    collector = create_test_collector("KRX", months_back=6)

    # 주식 리스트 조회
    symbols = collector.get_symbol_list()
    logger.info(f"KRX 주식 개수: {len(symbols)}")

    if symbols:
        # 유명한 대형주들로 테스트
        famous_stocks = [
            "005930",
            "000660",
            "035420",
            "005380",
            "068270",
        ]  # 삼성전자, SK하이닉스, NAVER, 현대차, 셀트리온
        test_symbols = [s for s in famous_stocks if s in symbols]

        if not test_symbols:
            test_symbols = symbols[:5]

        logger.info(f"테스트 대상: {test_symbols}")

        # 임시로 검증 기준 완화
        original_min_points = collector.validate_dataframe.__func__.__defaults__

        # 데이터 수집
        collected_data = collector.run_collection(test_symbols)

        # 결과 분석
        successful_symbols = [
            s for s, data in collected_data.items() if data is not None
        ]
        logger.info(f"수집 성공: {len(successful_symbols)}개")

        # 상세 결과 출력
        for symbol in test_symbols:
            data = collected_data.get(symbol)
            if data is not None:
                logger.info(f"✅ {symbol}: {len(data)}건 수집 성공")
                logger.info(
                    f"   기간: {data.index.min().date()} ~ {data.index.max().date()}"
                )
                if "Close" in data.columns:
                    logger.info(f"   최종가: {data['Close'].iloc[-1]:,.0f}원")
            else:
                logger.warning(f"❌ {symbol}: 수집 실패")


def test_individual_stock_detailed():
    """개별 주식 상세 테스트"""
    logger.info("=" * 60)
    logger.info("개별 주식 상세 테스트")
    logger.info("=" * 60)

    # 1년 데이터로 삼성전자 테스트
    collector = StockCollector(
        exchange="KRX", start_date="2024-01-01", end_date="2024-12-31"
    )

    symbol = "005930"  # 삼성전자
    logger.info(f"{symbol} 데이터 수집 및 분석...")

    # 데이터 수집
    stock_data = collector.collect_single_stock(symbol)

    if stock_data is not None:
        logger.info(f"✅ {symbol} 수집 성공!")

        # 기본 정보
        logger.info(f"📊 기본 정보:")
        logger.info(
            f"   데이터 기간: {stock_data.index.min().date()} ~ {stock_data.index.max().date()}"
        )
        logger.info(f"   총 거래일: {len(stock_data)}일")
        logger.info(f"   컬럼: {list(stock_data.columns)}")

        # 가격 통계
        if "Close" in stock_data.columns:
            close_prices = stock_data["Close"]
            logger.info(f"💰 가격 통계:")
            logger.info(f"   최고가: {close_prices.max():,.0f}원")
            logger.info(f"   최저가: {close_prices.min():,.0f}원")
            logger.info(f"   평균가: {close_prices.mean():,.0f}원")
            logger.info(f"   최종가: {close_prices.iloc[-1]:,.0f}원")
            logger.info(
                f"   연간 수익률: {((close_prices.iloc[-1] / close_prices.iloc[0]) - 1) * 100:.2f}%"
            )

        # 거래량 통계
        if "Volume" in stock_data.columns:
            volumes = stock_data["Volume"]
            logger.info(f"📈 거래량 통계:")
            logger.info(f"   평균 거래량: {volumes.mean():,.0f}주")
            logger.info(f"   최대 거래량: {volumes.max():,.0f}주")
            logger.info(f"   거래량 0인 날: {(volumes == 0).sum()}일")

        # 변동성 분석
        if "Close" in stock_data.columns:
            daily_returns = stock_data["Close"].pct_change().dropna()
            logger.info(f"📊 변동성 분석:")
            logger.info(f"   일일 수익률 평균: {daily_returns.mean():.4f}")
            logger.info(f"   일일 변동성: {daily_returns.std():.4f}")
            logger.info(f"   연환산 변동성: {daily_returns.std() * (252**0.5):.2%}")

        # 최근 5일 데이터 출력
        logger.info(f"📅 최근 5일 데이터:")
        logger.info(f"\n{stock_data.tail().to_string()}")

        # 주식 정보 조회
        stock_info = collector.get_stock_info(symbol)
        if stock_info:
            logger.info(f"ℹ️ 주식 정보:")
            logger.info(f"   종목명: {stock_info.get('Name', 'N/A')}")
            logger.info(f"   시장: {stock_info.get('Market', 'N/A')}")
            logger.info(f"   시가총액: {stock_info.get('Marcap', 0):,.0f}원")
    else:
        logger.error(f"❌ {symbol} 수집 실패")


def test_data_quality_check():
    """데이터 품질 검증 테스트"""
    logger.info("=" * 60)
    logger.info("데이터 품질 검증 테스트")
    logger.info("=" * 60)

    collector = create_test_collector("KRX", months_back=3)

    # 여러 종목으로 품질 검증
    test_symbols = ["005930", "000660", "035420"]  # 대형주

    for symbol in test_symbols:
        logger.info(f"🔍 {symbol} 품질 검증 중...")

        data = collector.collect_single_stock(symbol)

        if data is not None:
            # 가격 일관성 검증
            price_columns = ["Open", "High", "Low", "Close"]
            existing_cols = [col for col in price_columns if col in data.columns]

            if len(existing_cols) >= 4:
                # High >= Low 검증
                high_low_check = (data["High"] >= data["Low"]).all()
                # High >= Open, Close 검증
                high_open_check = (data["High"] >= data["Open"]).all()
                high_close_check = (data["High"] >= data["Close"]).all()
                # Low <= Open, Close 검증
                low_open_check = (data["Low"] <= data["Open"]).all()
                low_close_check = (data["Low"] <= data["Close"]).all()

                logger.info(f"   ✅ High >= Low: {high_low_check}")
                logger.info(f"   ✅ High >= Open: {high_open_check}")
                logger.info(f"   ✅ High >= Close: {high_close_check}")
                logger.info(f"   ✅ Low <= Open: {low_open_check}")
                logger.info(f"   ✅ Low <= Close: {low_close_check}")

                # 음수 가격 검증
                negative_prices = any((data[col] <= 0).any() for col in existing_cols)
                logger.info(f"   ✅ 음수 가격 없음: {not negative_prices}")

            # 거래량 검증
            if "Volume" in data.columns:
                zero_volume_ratio = (data["Volume"] == 0).sum() / len(data)
                logger.info(f"   📊 거래량 0 비율: {zero_volume_ratio:.2%}")

            # 결측치 검증
            missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
            logger.info(f"   📊 결측치 비율: {missing_ratio:.2%}")

            logger.info(f"   ✅ {symbol} 품질 검증 완료\n")
        else:
            logger.warning(f"   ❌ {symbol} 데이터 없음\n")


def test_us_stocks_working():
    """미국 주식 작동 테스트 (간단한 버전)"""
    logger.info("=" * 60)
    logger.info("미국 주식 작동 테스트")
    logger.info("=" * 60)

    # 3개월 데이터로 테스트
    collector = create_test_collector("SP500", months_back=3)

    # 유명한 주식 1개만 테스트
    test_symbol = "AAPL"  # 애플

    logger.info(f"{test_symbol} 데이터 수집 중...")

    data = collector.collect_single_stock(test_symbol)

    if data is not None:
        logger.info(f"✅ {test_symbol} 수집 성공!")
        logger.info(f"   기간: {data.index.min().date()} ~ {data.index.max().date()}")
        logger.info(f"   데이터 건수: {len(data)}")
        if "Close" in data.columns:
            logger.info(f"   최종가: ${data['Close'].iloc[-1]:.2f}")
    else:
        logger.warning(f"❌ {test_symbol} 수집 실패")


if __name__ == "__main__":
    logger.info("StockCollector 테스트 시작")

    try:
        # 개별 테스트 실행
        test_individual_stock_detailed()
        test_data_quality_check()
        test_krx_stock_collector()
        test_us_stocks_working()

        logger.info("=" * 60)
        logger.info("모든 StockCollector 테스트 완료!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"테스트 중 오류 발생: {str(e)}")
        import traceback

        logger.error(f"상세 오류:\n{traceback.format_exc()}")
