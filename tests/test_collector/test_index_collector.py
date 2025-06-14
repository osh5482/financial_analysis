"""
IndexCollector 테스트 스크립트
"""

import pandas as pd
from datetime import datetime, timedelta

from src.data_collector.index_collector import IndexCollector
from src.utils.logger import setup_logging

# 로거 초기화 (개발 모드)
logger = setup_logging(development_mode=True)


def create_test_index_collector(
    markets: list[str] = None, months_back: int = 6
) -> IndexCollector:
    """
    테스트용 지수 수집기 생성

    Args:
        markets: 수집할 시장 리스트
        months_back: 수집할 과거 개월 수

    Returns:
        테스트용 IndexCollector
    """
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=months_back * 30)).strftime(
        "%Y-%m-%d"
    )

    collector = IndexCollector(
        markets=markets or ["KR", "US"],
        start_date=start_date,
        end_date=end_date,
        batch_size=5,  # 테스트용 작은 배치
        max_retries=2,
        rate_limit_delay=0.1,
    )

    # 테스트용 검증 기준 완화
    collector.min_data_points = 20

    return collector


def test_index_list_collection():
    """지수 리스트 수집 테스트"""
    logger.info("=" * 60)
    logger.info("지수 리스트 수집 테스트")
    logger.info("=" * 60)

    collector = create_test_index_collector(["KR", "US"], months_back=3)

    # 지수 리스트 수집
    symbols = collector.get_symbol_list()
    logger.info(f"✅ 전체 지수 개수: {len(symbols)}개")

    if symbols:
        logger.info(f"📋 주요 지수 리스트:")
        for i, symbol in enumerate(symbols):
            index_info = collector.get_index_info(symbol)
            if index_info:
                name = index_info.get("Name", "Unknown")
                market = index_info.get("Market", "Unknown")
                logger.info(f"   {i+1:2d}. {symbol}: {name} ({market})")
            else:
                logger.info(f"   {i+1:2d}. {symbol}: 정보 없음")

    return symbols


def test_kr_indices_collection():
    """한국 지수 데이터 수집 테스트"""
    logger.info("=" * 60)
    logger.info("한국 지수 데이터 수집 테스트")
    logger.info("=" * 60)

    collector = create_test_index_collector(["KR"], months_back=6)

    # 주요 한국 지수들
    test_symbols = ["KS11", "KQ11", "KS200"]  # KOSPI, KOSDAQ, KOSPI200

    logger.info(f"📊 테스트 대상 지수: {test_symbols}")

    collected_data = collector.run_collection(test_symbols)

    # 결과 분석
    successful_symbols = [s for s, data in collected_data.items() if data is not None]
    logger.info(f"✅ 수집 성공: {len(successful_symbols)}/{len(test_symbols)}개")

    # 상세 결과 출력
    for symbol in test_symbols:
        data = collected_data.get(symbol)
        index_info = collector.get_index_info(symbol)
        index_name = index_info.get("Name", "Unknown") if index_info else "Unknown"

        if data is not None:
            logger.info(f"✅ {symbol} ({index_name}): {len(data)}건 수집 성공")
            logger.info(
                f"   기간: {data.index.min().date()} ~ {data.index.max().date()}"
            )

            if "Close" in data.columns:
                close_level = data["Close"].iloc[-1]
                logger.info(f"   최종 레벨: {close_level:,.2f}")

                # 수익률 계산
                total_return = ((close_level / data["Close"].iloc[0]) - 1) * 100
                logger.info(f"   기간 수익률: {total_return:.2f}%")

            # 지수 특화 지표 확인
            if "Annual_Volatility" in data.columns:
                annual_vol = (
                    data["Annual_Volatility"].dropna().iloc[-1]
                    if not data["Annual_Volatility"].dropna().empty
                    else None
                )
                if annual_vol:
                    logger.info(f"   연환산 변동성: {annual_vol:.2%}")

            if "RSI" in data.columns:
                current_rsi = (
                    data["RSI"].dropna().iloc[-1]
                    if not data["RSI"].dropna().empty
                    else None
                )
                if current_rsi:
                    logger.info(f"   현재 RSI: {current_rsi:.1f}")
        else:
            logger.warning(f"❌ {symbol} ({index_name}): 수집 실패")


def test_us_indices_collection():
    """미국 지수 데이터 수집 테스트"""
    logger.info("=" * 60)
    logger.info("미국 지수 데이터 수집 테스트")
    logger.info("=" * 60)

    collector = create_test_index_collector(["US"], months_back=3)

    # 주요 미국 지수들
    test_symbols = ["DJI", "IXIC", "US500"]  # 다우존스, 나스닥, S&P500

    logger.info(f"📊 테스트 대상 지수: {test_symbols}")

    collected_data = collector.run_collection(test_symbols)

    # 결과 분석
    successful_symbols = [s for s, data in collected_data.items() if data is not None]
    logger.info(f"✅ 수집 성공: {len(successful_symbols)}/{len(test_symbols)}개")

    # 상세 결과 출력
    for symbol in test_symbols:
        data = collected_data.get(symbol)
        index_info = collector.get_index_info(symbol)
        index_name = index_info.get("Name", "Unknown") if index_info else "Unknown"

        if data is not None:
            logger.info(f"✅ {symbol} ({index_name}): {len(data)}건 수집 성공")
            logger.info(
                f"   기간: {data.index.min().date()} ~ {data.index.max().date()}"
            )

            if "Close" in data.columns:
                close_level = data["Close"].iloc[-1]
                logger.info(f"   최종 레벨: {close_level:,.2f}")

                # 수익률 계산
                total_return = ((close_level / data["Close"].iloc[0]) - 1) * 100
                logger.info(f"   기간 수익률: {total_return:.2f}%")
        else:
            logger.warning(f"❌ {symbol} ({index_name}): 수집 실패")


def test_individual_index_analysis():
    """개별 지수 상세 분석 테스트"""
    logger.info("=" * 60)
    logger.info("개별 지수 상세 분석 테스트")
    logger.info("=" * 60)

    # KOSPI 지수 상세 분석
    collector = IndexCollector(
        markets=["KR"], start_date="2024-01-01", end_date="2024-12-31"
    )

    symbol = "KS11"  # KOSPI
    logger.info(f"{symbol} 지수 상세 분석...")

    # 개별 수집
    index_data = collector.collect_single_index(symbol)

    if index_data is not None:
        logger.info(f"✅ {symbol} 수집 성공!")

        # 기본 정보
        logger.info(f"📊 기본 정보:")
        logger.info(
            f"   데이터 기간: {index_data.index.min().date()} ~ {index_data.index.max().date()}"
        )
        logger.info(f"   총 거래일: {len(index_data)}일")
        logger.info(f"   컬럼: {list(index_data.columns)}")

        # 지수 레벨 통계
        if "Close" in index_data.columns:
            levels = index_data["Close"]
            logger.info(f"💹 지수 레벨 통계:")
            logger.info(f"   최고점: {levels.max():,.2f}")
            logger.info(f"   최저점: {levels.min():,.2f}")
            logger.info(f"   평균: {levels.mean():,.2f}")
            logger.info(f"   최종: {levels.iloc[-1]:,.2f}")
            logger.info(
                f"   연간 수익률: {((levels.iloc[-1] / levels.iloc[0]) - 1) * 100:.2f}%"
            )

        # 변동성 분석
        if "Daily_Return" in index_data.columns:
            daily_returns = index_data["Daily_Return"].dropna()
            logger.info(f"📊 변동성 분석:")
            logger.info(f"   일일 수익률 평균: {daily_returns.mean():.4f}")
            logger.info(f"   일일 변동성: {daily_returns.std():.4f}")
            logger.info(f"   연환산 변동성: {daily_returns.std() * (252**0.5):.2%}")

            # 극값 분석
            logger.info(f"   최대 일일 상승: {daily_returns.max():.2%}")
            logger.info(f"   최대 일일 하락: {daily_returns.min():.2%}")

        # 기술적 지표
        if "RSI" in index_data.columns:
            current_rsi = (
                index_data["RSI"].dropna().iloc[-1]
                if not index_data["RSI"].dropna().empty
                else None
            )
            if current_rsi:
                logger.info(f"📈 기술적 지표:")
                logger.info(f"   현재 RSI: {current_rsi:.1f}")

                # RSI 해석
                if current_rsi > 70:
                    rsi_signal = "과매수"
                elif current_rsi < 30:
                    rsi_signal = "과매도"
                else:
                    rsi_signal = "중립"
                logger.info(f"   RSI 신호: {rsi_signal}")

        # 최대 낙폭
        if "Drawdown" in index_data.columns:
            max_dd = index_data["Drawdown"].min()
            logger.info(f"📉 리스크 지표:")
            logger.info(f"   최대 낙폭(MDD): {max_dd:.2f}%")

        # 성과 분석 실행
        performance = collector.analyze_index_performance(symbol)
        if performance:
            logger.info(f"🏆 종합 성과 분석:")
            logger.info(f"   총 수익률: {performance.get('total_return', 0):.2f}%")
            logger.info(f"   변동성: {performance.get('volatility', 0):.2f}%")
            logger.info(
                f"   상승일 비율: {performance.get('positive_days_ratio', 0):.1f}%"
            )

            # 트렌드 분석
            short_trend = performance.get("short_term_trend", "Unknown")
            long_trend = performance.get("long_term_trend", "Unknown")
            logger.info(f"   단기 트렌드: {short_trend}")
            logger.info(f"   장기 트렌드: {long_trend}")
    else:
        logger.error(f"❌ {symbol} 수집 실패")


def test_market_correlation():
    """지수 간 상관관계 분석 테스트"""
    logger.info("=" * 60)
    logger.info("지수 간 상관관계 분석 테스트")
    logger.info("=" * 60)

    collector = create_test_index_collector(["KR", "US"], months_back=6)

    # 주요 지수들로 상관관계 분석
    test_symbols = ["KS11", "KQ11", "DJI", "IXIC"]  # KOSPI, KOSDAQ, 다우존스, 나스닥

    logger.info(f"📊 상관관계 분석 대상: {test_symbols}")

    correlation_matrix = collector.analyze_market_correlation(test_symbols)

    if correlation_matrix is not None:
        logger.info(f"✅ 상관관계 분석 완료")
        logger.info(f"📈 상관관계 매트릭스:")

        # 상관관계 매트릭스 출력
        logger.info(f"\n{correlation_matrix.round(3).to_string()}")

        # 주요 상관관계 분석
        logger.info(f"\n🔍 주요 상관관계:")

        # 한국 지수 간 상관관계
        if "KS11" in correlation_matrix.index and "KQ11" in correlation_matrix.index:
            kospi_kosdaq_corr = correlation_matrix.loc["KS11", "KQ11"]
            logger.info(f"   KOSPI-KOSDAQ: {kospi_kosdaq_corr:.3f}")

        # 미국 지수 간 상관관계
        if "DJI" in correlation_matrix.index and "IXIC" in correlation_matrix.index:
            dow_nasdaq_corr = correlation_matrix.loc["DJI", "IXIC"]
            logger.info(f"   다우존스-나스닥: {dow_nasdaq_corr:.3f}")

        # 한미 시장 간 상관관계
        if "KS11" in correlation_matrix.index and "DJI" in correlation_matrix.index:
            kospi_dow_corr = correlation_matrix.loc["KS11", "DJI"]
            logger.info(f"   KOSPI-다우존스: {kospi_dow_corr:.3f}")

        # 가장 높은/낮은 상관관계 찾기
        corr_values = []
        for i in range(len(correlation_matrix.index)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                idx1, idx2 = correlation_matrix.index[i], correlation_matrix.columns[j]
                corr_val = correlation_matrix.loc[idx1, idx2]
                corr_values.append((idx1, idx2, corr_val))

        if corr_values:
            highest_corr = max(corr_values, key=lambda x: x[2])
            lowest_corr = min(corr_values, key=lambda x: x[2])

            logger.info(f"\n📊 극값 상관관계:")
            logger.info(
                f"   최고 상관관계: {highest_corr[0]}-{highest_corr[1]} ({highest_corr[2]:.3f})"
            )
            logger.info(
                f"   최저 상관관계: {lowest_corr[0]}-{lowest_corr[1]} ({lowest_corr[2]:.3f})"
            )
    else:
        logger.warning(f"❌ 상관관계 분석 실패")


def test_market_summary():
    """시장별 요약 분석 테스트"""
    logger.info("=" * 60)
    logger.info("시장별 요약 분석 테스트")
    logger.info("=" * 60)

    collector = create_test_index_collector(["KR", "US"], months_back=6)

    # 한국 시장 요약
    logger.info(f"🇰🇷 한국 시장 요약:")
    kr_summary = collector.get_market_summary("KR")

    if kr_summary:
        logger.info(f"   지수 개수: {kr_summary.get('indices_count', 0)}개")
        logger.info(f"   평균 수익률: {kr_summary.get('average_return', 0):.2f}%")
        logger.info(f"   평균 변동성: {kr_summary.get('average_volatility', 0):.2f}%")

        indices_data = kr_summary.get("indices_data", {})
        for idx_name, data in indices_data.items():
            logger.info(
                f"     {data['name']}: {data['return']:.2f}% (변동성: {data['volatility']:.2f}%)"
            )

    # 미국 시장 요약
    logger.info(f"\n🇺🇸 미국 시장 요약:")
    us_summary = collector.get_market_summary("US")

    if us_summary:
        logger.info(f"   지수 개수: {us_summary.get('indices_count', 0)}개")
        logger.info(f"   평균 수익률: {us_summary.get('average_return', 0):.2f}%")
        logger.info(f"   평균 변동성: {us_summary.get('average_volatility', 0):.2f}%")

        indices_data = us_summary.get("indices_data", {})
        for idx_name, data in indices_data.items():
            logger.info(
                f"     {data['name']}: {data['return']:.2f}% (변동성: {data['volatility']:.2f}%)"
            )

    # 시장 간 비교
    if kr_summary and us_summary:
        logger.info(f"\n📊 시장 간 비교:")
        kr_return = kr_summary.get("average_return", 0)
        us_return = us_summary.get("average_return", 0)

        if kr_return > us_return:
            better_market = "한국"
            return_diff = kr_return - us_return
        else:
            better_market = "미국"
            return_diff = us_return - kr_return

        logger.info(f"   우수 시장: {better_market} (+{return_diff:.2f}%p)")

        kr_vol = kr_summary.get("average_volatility", 0)
        us_vol = us_summary.get("average_volatility", 0)

        if kr_vol < us_vol:
            stable_market = "한국"
            vol_diff = us_vol - kr_vol
        else:
            stable_market = "미국"
            vol_diff = kr_vol - us_vol

        logger.info(f"   안정 시장: {stable_market} (-{vol_diff:.2f}%p 변동성)")


def test_index_data_quality():
    """지수 데이터 품질 검증 테스트"""
    logger.info("=" * 60)
    logger.info("지수 데이터 품질 검증 테스트")
    logger.info("=" * 60)

    collector = create_test_index_collector(["KR"], months_back=3)

    # 여러 지수로 품질 검증
    test_symbols = ["KS11", "KQ11", "KS200"]  # KOSPI, KOSDAQ, KOSPI200

    for symbol in test_symbols:
        logger.info(f"🔍 {symbol} 품질 검증 중...")

        data = collector.collect_single_index(symbol)

        if data is not None:
            # 지수 레벨 일관성 검증
            if "Close" in data.columns:
                # 음수 레벨 검증
                negative_levels = (data["Close"] <= 0).sum()
                logger.info(f"   ✅ 음수 레벨 없음: {negative_levels == 0}")

                # 지수 연속성 검증
                daily_changes = data["Close"].pct_change().dropna()
                extreme_changes = (abs(daily_changes) > 0.1).sum()  # 10% 이상 변동
                logger.info(f"   📊 극단적 변동: {extreme_changes}건")

            # 기술적 지표 검증
            if "RSI" in data.columns:
                rsi_valid = ((data["RSI"] >= 0) & (data["RSI"] <= 100)).all()
                logger.info(f"   ✅ RSI 범위 유효: {rsi_valid}")

            # 이동평균 순서 검증 (단기 < 장기는 아닐 수 있음)
            if "MA_20" in data.columns and "MA_60" in data.columns:
                ma_crossovers = (
                    (data["MA_20"] > data["MA_60"]).astype(int).diff() != 0
                ).sum()
                logger.info(f"   📊 이동평균 교차: {ma_crossovers}회")

            # 변동성 검증
            if "Volatility_20" in data.columns:
                avg_vol = data["Volatility_20"].mean()
                logger.info(f"   📊 평균 변동성: {avg_vol:.4f}")

            logger.info(f"   ✅ {symbol} 품질 검증 완료\n")
        else:
            logger.warning(f"   ❌ {symbol} 데이터 없음\n")


if __name__ == "__main__":
    logger.info("IndexCollector 테스트 시작")

    try:
        # 각 테스트 실행
        test_index_list_collection()
        test_individual_index_analysis()
        test_kr_indices_collection()
        test_us_indices_collection()
        test_market_correlation()
        test_market_summary()
        test_index_data_quality()

        logger.info("=" * 60)
        logger.info("모든 IndexCollector 테스트 완료!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Index 테스트 중 오류 발생: {str(e)}")
        import traceback

        logger.error(f"상세 오류:\n{traceback.format_exc()}")
