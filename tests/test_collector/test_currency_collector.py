"""
CurrencyCollector 테스트 스크립트
"""

import pandas as pd
from datetime import datetime, timedelta

from src.data_collector.currency_collector import CurrencyCollector
from src.utils.logger import setup_logging

# 로거 초기화 (개발 모드)
logger = setup_logging(development_mode=True)


def create_test_currency_collector(
    base_currency: str = "KRW", months_back: int = 6
) -> CurrencyCollector:
    """
    테스트용 환율 수집기 생성

    Args:
        base_currency: 기준 통화 (KRW, USD 등)
        months_back: 수집할 과거 개월 수

    Returns:
        테스트용 CurrencyCollector
    """
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=months_back * 30)).strftime(
        "%Y-%m-%d"
    )

    collector = CurrencyCollector(
        base_currency=base_currency,
        start_date=start_date,
        end_date=end_date,
        include_minor_pairs=True,
        batch_size=5,  # 테스트용 작은 배치
        max_retries=2,
        rate_limit_delay=0.1,
    )

    # 테스트용 검증 기준 완화
    collector.min_data_points = 20

    return collector


def test_currency_list_collection():
    """환율 리스트 수집 테스트"""
    logger.info("=" * 60)
    logger.info("환율 리스트 수집 테스트")
    logger.info("=" * 60)

    collector = create_test_currency_collector("KRW", months_back=3)

    # 통화쌍 리스트 수집
    symbols = collector.get_symbol_list()
    logger.info(f"✅ 수집 대상 통화쌍: {len(symbols)}개")

    if symbols:
        logger.info(f"📋 주요 통화쌍 리스트:")
        for i, symbol in enumerate(symbols):
            currency_info = collector.get_currency_info(symbol)
            if currency_info:
                name = currency_info.get("name", currency_info.get("Name", "Unknown"))
                pair_type = currency_info.get(
                    "type", currency_info.get("Type", "Major")
                )
                logger.info(f"   {i+1:2d}. {symbol}: {name} ({pair_type})")
            else:
                logger.info(f"   {i+1:2d}. {symbol}: 정보 없음")

    return symbols


def test_individual_currency_detailed():
    """개별 통화쌍 상세 테스트"""
    logger.info("=" * 60)
    logger.info("개별 통화쌍 상세 테스트")
    logger.info("=" * 60)

    # 3개월 데이터로 테스트
    collector = create_test_currency_collector("KRW", months_back=3)

    # 주요 통화쌍 테스트
    test_symbols = ["USD/KRW", "USDKRW=X", "EUR/KRW", "EURKRW=X", "JPY/KRW", "JPYKRW=X"]

    for symbol in test_symbols:
        logger.info(f"\n{symbol} 상세 테스트")
        logger.info("-" * 40)

        # 개별 데이터 수집
        currency_data = collector.collect_single_currency(symbol)

        if currency_data is not None and not currency_data.empty:
            logger.info(f"   ✅ 수집 성공: {len(currency_data)}개 레코드")
            logger.info(
                f"   📅 기간: {currency_data.index.min().date()} ~ {currency_data.index.max().date()}"
            )

            # 컬럼 정보
            logger.info(f"   📊 컬럼: {list(currency_data.columns)}")

            # 기본 통계
            if "Close" in currency_data.columns:
                close_data = currency_data["Close"]
                logger.info(f"   💱 현재 환율: {close_data.iloc[-1]:.4f}")
                logger.info(f"   📈 최고값: {close_data.max():.4f}")
                logger.info(f"   📉 최저값: {close_data.min():.4f}")
                logger.info(f"   📊 평균값: {close_data.mean():.4f}")

            # 변동성 정보 (있는 경우)
            if "Volatility_20" in currency_data.columns:
                vol_data = currency_data["Volatility_20"].dropna()
                if not vol_data.empty:
                    logger.info(f"   📊 최근 변동성: {vol_data.iloc[-1]:.4f}")

            # 일일 수익률 정보 (있는 경우)
            if "Daily_Return" in currency_data.columns:
                return_data = currency_data["Daily_Return"].dropna()
                if not return_data.empty:
                    logger.info(f"   📊 평균 일일 변동: {return_data.mean():.4f}")
                    logger.info(f"   📊 일일 변동 표준편차: {return_data.std():.4f}")

            # 결측치 확인
            missing_data = currency_data.isnull().sum()
            total_missing = missing_data.sum()
            if total_missing > 0:
                logger.info(f"   ⚠️ 결측치: {total_missing}개")
                for col, count in missing_data.items():
                    if count > 0:
                        logger.info(f"     - {col}: {count}개")
            else:
                logger.info(f"   ✅ 결측치 없음")

        else:
            logger.warning(f"   ❌ {symbol} 데이터 수집 실패")
            # 실패한 경우 다른 심볼 형태로 재시도
            if "/" in symbol:
                alternative_symbol = symbol.replace("/", "") + "=X"
                logger.info(f"   🔄 대안 심볼로 재시도: {alternative_symbol}")
                alt_data = collector.collect_single_currency(alternative_symbol)
                if alt_data is not None:
                    logger.info(
                        f"   ✅ 대안 심볼로 수집 성공: {len(alt_data)}개 레코드"
                    )


def test_currency_data_quality():
    """환율 데이터 품질 검증 테스트"""
    logger.info("=" * 60)
    logger.info("환율 데이터 품질 검증 테스트")
    logger.info("=" * 60)

    collector = create_test_currency_collector("KRW", months_back=6)

    # 주요 통화쌍으로 품질 테스트
    test_symbols = ["USDKRW=X", "EURKRW=X"]

    for symbol in test_symbols:
        logger.info(f"\n{symbol} 품질 검증")
        logger.info("-" * 40)

        data = collector.collect_single_currency(symbol)

        if data is not None and not data.empty:
            logger.info(f"   ✅ 기본 수집 성공: {len(data)}개 레코드")

            # 환율 범위 검증 (양수 확인)
            if "Close" in data.columns:
                positive_rates = (data["Close"] > 0).all()
                logger.info(f"   ✅ 모든 환율이 양수: {positive_rates}")

                # 환율 변동성 검증 (너무 극단적인 변화 확인)
                if len(data) > 1:
                    daily_changes = data["Close"].pct_change().dropna()
                    extreme_changes = (daily_changes.abs() > 0.1).sum()  # 10% 이상 변동
                    logger.info(f"   📊 극단적 변동 (10% 이상): {extreme_changes}일")

                    if daily_changes.std() > 0:
                        logger.info(
                            f"   📊 일일 변동률 표준편차: {daily_changes.std():.4f}"
                        )

            # OHLC 데이터 일관성 검증 (있는 경우)
            ohlc_cols = ["Open", "High", "Low", "Close"]
            existing_ohlc = [col for col in ohlc_cols if col in data.columns]

            if len(existing_ohlc) >= 4:
                logger.info(f"   📊 OHLC 데이터 일관성 검증:")

                # High >= Low 검증
                high_low_check = (data["High"] >= data["Low"]).all()
                logger.info(f"   ✅ High >= Low: {high_low_check}")

                # High >= Open, Close 검증
                high_open_check = (data["High"] >= data["Open"]).all()
                high_close_check = (data["High"] >= data["Close"]).all()
                logger.info(f"   ✅ High >= Open: {high_open_check}")
                logger.info(f"   ✅ High >= Close: {high_close_check}")

                # Low <= Open, Close 검증
                low_open_check = (data["Low"] <= data["Open"]).all()
                low_close_check = (data["Low"] <= data["Close"]).all()
                logger.info(f"   ✅ Low <= Open: {low_open_check}")
                logger.info(f"   ✅ Low <= Close: {low_close_check}")

            # 연속성 검증 (날짜 간격)
            if isinstance(data.index, pd.DatetimeIndex) and len(data) > 1:
                date_gaps = data.index.to_series().diff().dropna()
                avg_gap = date_gaps.mean()
                max_gap = date_gaps.max()
                logger.info(f"   📅 평균 날짜 간격: {avg_gap}")
                logger.info(f"   📅 최대 날짜 간격: {max_gap}")

            # 결측치 비율
            missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
            logger.info(f"   📊 결측치 비율: {missing_ratio:.2%}")

            logger.info(f"   ✅ {symbol} 품질 검증 완료\n")
        else:
            logger.warning(f"   ❌ {symbol} 데이터 없음\n")


def test_currency_batch_collection():
    """환율 배치 수집 테스트"""
    logger.info("=" * 60)
    logger.info("환율 배치 수집 테스트")
    logger.info("=" * 60)

    # 3개월 데이터로 테스트
    collector = create_test_currency_collector("KRW", months_back=3)

    # 주요 통화쌍들로 배치 테스트
    test_symbols = ["USDKRW=X", "EURKRW=X", "JPYKRW=X", "GBPKRW=X", "CHDKRW=X"]

    logger.info(f"배치 수집 대상: {test_symbols}")

    # 배치 수집 실행
    collected_data = collector.collect_data(test_symbols)

    # 결과 분석
    successful_symbols = [s for s, data in collected_data.items() if data is not None]
    failed_symbols = [s for s in test_symbols if s not in successful_symbols]

    logger.info(f"✅ 수집 성공: {len(successful_symbols)}개")
    logger.info(f"❌ 수집 실패: {len(failed_symbols)}개")

    if successful_symbols:
        logger.info(f"\n수집 성공 통화쌍:")
        for symbol in successful_symbols:
            data = collected_data[symbol]
            logger.info(f"  - {symbol}: {len(data)}개 레코드")

    if failed_symbols:
        logger.info(f"\n수집 실패 통화쌍:")
        for symbol in failed_symbols:
            logger.info(f"  - {symbol}")

    return collected_data


def test_currency_performance_analysis():
    """환율 성과 분석 테스트"""
    logger.info("=" * 60)
    logger.info("환율 성과 분석 테스트")
    logger.info("=" * 60)

    collector = create_test_currency_collector("KRW", months_back=6)

    # USD/KRW 성과 분석
    test_symbol = "USDKRW=X"
    logger.info(f"{test_symbol} 성과 분석")

    analysis_result = collector.analyze_currency_performance(test_symbol)

    if analysis_result:
        logger.info(f"✅ 성과 분석 성공")
        logger.info(f"   심볼: {analysis_result.get('symbol')}")
        logger.info(f"   분석 기간: {analysis_result.get('period')}")
        logger.info(f"   총 거래일: {analysis_result.get('total_days')}")

        # 수익률 정보
        if "total_return" in analysis_result:
            logger.info(f"   총 수익률: {analysis_result['total_return']:.2%}")

        if "annual_return" in analysis_result:
            logger.info(f"   연환산 수익률: {analysis_result['annual_return']:.2%}")

        if "volatility" in analysis_result:
            logger.info(f"   변동성: {analysis_result['volatility']:.4f}")

        # 트렌드 정보
        if "trend_direction" in analysis_result:
            logger.info(f"   트렌드: {analysis_result['trend_direction']}")

        if "currency_strength" in analysis_result:
            strength = analysis_result["currency_strength"]
            if strength is not None:
                logger.info(f"   통화 강도: {strength:.4f}")

    else:
        logger.warning(f"❌ {test_symbol} 성과 분석 실패")


def test_currency_correlation_analysis():
    """통화쌍 상관관계 분석 테스트"""
    logger.info("=" * 60)
    logger.info("통화쌍 상관관계 분석 테스트")
    logger.info("=" * 60)

    collector = create_test_currency_collector("KRW", months_back=6)

    # 주요 통화쌍들의 상관관계 분석
    test_symbols = ["USDKRW=X", "EURKRW=X", "JPYKRW=X"]
    logger.info(f"상관관계 분석 대상: {test_symbols}")

    correlation_matrix = collector.analyze_currency_correlation(test_symbols)

    if correlation_matrix is not None:
        logger.info(f"✅ 상관관계 분석 성공")
        logger.info(f"   분석 통화쌍: {len(correlation_matrix.columns)}개")
        logger.info(f"\n상관관계 매트릭스:")

        # 상관관계 매트릭스 출력
        for i, row_symbol in enumerate(correlation_matrix.index):
            correlations = []
            for j, col_symbol in enumerate(correlation_matrix.columns):
                if i <= j:  # 상삼각행렬만 출력
                    corr_value = correlation_matrix.iloc[i, j]
                    if pd.notna(corr_value):
                        correlations.append(f"{col_symbol}: {corr_value:.3f}")

            if correlations:
                logger.info(f"   {row_symbol}: {' | '.join(correlations)}")

    else:
        logger.warning(f"❌ 상관관계 분석 실패")


def test_hedge_analysis():
    """환헤지 분석 테스트"""
    logger.info("=" * 60)
    logger.info("환헤지 분석 테스트")
    logger.info("=" * 60)

    collector = create_test_currency_collector("KRW", months_back=3)

    # 100만원을 달러로 투자하는 경우의 환헤지 분석
    base_amount = 1_000_000  # 100만원
    target_currency = "USD"
    hedge_ratio = 0.5  # 50% 헤지

    logger.info(f"환헤지 분석 조건:")
    logger.info(f"  - 투자 금액: {base_amount:,}원")
    logger.info(f"  - 대상 통화: {target_currency}")
    logger.info(f"  - 헤지 비율: {hedge_ratio:.0%}")

    hedge_result = collector.get_hedge_analysis(
        base_amount, target_currency, hedge_ratio
    )

    if hedge_result:
        logger.info(f"✅ 환헤지 분석 성공")
        logger.info(f"   현재 환율: {hedge_result.get('current_rate', 0):.2f}")
        logger.info(f"   환전 금액: ${hedge_result.get('converted_amount', 0):.2f}")
        logger.info(f"   헤지 금액: {hedge_result.get('hedged_amount', 0):,.0f}원")
        logger.info(f"   무헤지 금액: {hedge_result.get('unhedged_amount', 0):,.0f}원")

        # 리스크 정보
        if "monthly_volatility" in hedge_result:
            logger.info(f"   월간 변동성: {hedge_result['monthly_volatility']:.2f}%")

        if "value_at_risk_95" in hedge_result:
            var_amount = hedge_result["value_at_risk_95"]
            logger.info(f"   월간 VaR (95%): {var_amount:,.0f}원")

    else:
        logger.warning(f"❌ 환헤지 분석 실패")


if __name__ == "__main__":
    logger.info("CurrencyCollector 테스트 시작")

    try:
        # 개별 테스트 실행
        test_currency_list_collection()
        test_individual_currency_detailed()
        test_currency_data_quality()
        test_currency_batch_collection()
        test_currency_performance_analysis()
        test_currency_correlation_analysis()
        test_hedge_analysis()

        logger.info("=" * 60)
        logger.info("모든 CurrencyCollector 테스트 완료!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"테스트 중 오류 발생: {str(e)}")
        import traceback

        logger.error(f"상세 오류:\n{traceback.format_exc()}")
