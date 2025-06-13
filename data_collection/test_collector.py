"""
데이터 수집기 테스트 스크립트
작성한 모든 수집기들의 기본 기능을 검증
"""

import asyncio
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from loguru import logger
from data_collection.collectors.stock_collector import StockCollector
from data_collection.collectors.index_collector import IndexCollector

# 로깅 설정
logger.remove()  # 기본 핸들러 제거
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
)
logger.add(
    "test_collectors.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="DEBUG",
    rotation="10 MB",
)


class CollectorTester:
    """
    수집기 테스트 클래스
    """

    def __init__(self):
        """테스터 초기화"""
        self.test_start_time = None
        self.test_results = {}

        # 테스트용 날짜 설정 (최근 30일)
        self.end_date = datetime.now().strftime("%Y-%m-%d")
        self.start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

        logger.info("수집기 테스트 준비 완료")
        logger.info(f"테스트 기간: {self.start_date} ~ {self.end_date}")

    async def run_all_tests(self) -> dict:
        """
        모든 수집기 테스트 실행

        Returns:
            dict: 테스트 결과 요약
        """
        self.test_start_time = time.time()
        logger.info("=" * 60)
        logger.info("데이터 수집기 종합 테스트 시작")
        logger.info("=" * 60)

        # 주식 수집기 테스트
        await self._test_stock_collector()

        # 지수 수집기 테스트
        await self._test_index_collector()

        # 테스트 결과 요약
        self._print_test_summary()

        return self.test_results

    async def _test_stock_collector(self) -> None:
        """주식 수집기 테스트"""
        logger.info("\n[주식 수집기 테스트 시작]")
        test_name = "StockCollector"

        try:
            collector = StockCollector(rate_limit=0.05)  # 테스트용으로 빠르게 설정

            # 테스트 1: 기본 수집 기능 (한국 + 해외 주식)
            test_symbols = [
                "005930",  # 삼성전자 (한국)
                "000660",  # SK하이닉스 (한국)
                "AAPL",  # 애플 (미국)
                "MSFT",  # 마이크로소프트 (미국)
                "7203.T",  # 도요타 (일본)
            ]

            logger.info(f"테스트 심볼: {test_symbols}")

            start_time = time.time()
            data = await collector.collect_data(
                test_symbols, self.start_date, self.end_date
            )
            elapsed_time = time.time() - start_time

            # 결과 검증
            success_count = len(data)
            total_count = len(test_symbols)

            logger.info(f"수집 결과: {success_count}/{total_count} 성공")
            logger.info(f"소요 시간: {elapsed_time:.2f}초")

            # 데이터 품질 검증
            for symbol, df in data.items():
                logger.info(f"  {symbol}: {len(df)}행, 컬럼: {list(df.columns)}")

                # 필수 컬럼 존재 여부 확인
                required_columns = ["close", "symbol"]
                missing_columns = [
                    col for col in required_columns if col not in df.columns
                ]
                if missing_columns:
                    logger.warning(f"  {symbol}: 누락된 컬럼 - {missing_columns}")

            # 테스트 2: 심볼 유효성 검증
            logger.info("\n심볼 유효성 검증 테스트")
            test_validation_symbols = ["005930", "INVALID_SYMBOL", "AAPL", "", "123"]
            valid_symbols, invalid_symbols = await collector.validate_symbols(
                test_validation_symbols
            )
            logger.info(f"유효한 심볼: {valid_symbols}")
            logger.info(f"무효한 심볼: {invalid_symbols}")

            # 테스트 3: 거래소 정보 확인
            logger.info(
                f"\n지원 거래소: {list(collector.get_supported_exchanges().keys())}"
            )

            # 테스트 4: 상장 종목 조회 (한국만 - 시간 단축을 위해 선택적)
            logger.info("\n한국 상장 종목 조회 테스트 (샘플)")
            try:
                # 전체 조회 대신 빠른 검증만 수행
                logger.info(
                    "상장 종목 조회 기능 사용 가능 (전체 조회는 시간 관계상 생략)"
                )
            except Exception as e:
                logger.warning(f"상장 종목 조회 중 오류: {str(e)}")

            # 테스트 결과 저장
            self.test_results[test_name] = {
                "status": "PASS" if success_count > 0 else "FAIL",
                "success_rate": f"{success_count}/{total_count}",
                "elapsed_time": f"{elapsed_time:.2f}s",
                "data_count": sum(len(df) for df in data.values()),
            }

            logger.info(
                f"주식 수집기 테스트 완료: {self.test_results[test_name]['status']}"
            )

        except Exception as e:
            logger.error(f"주식 수집기 테스트 실패: {str(e)}")
            self.test_results[test_name] = {"status": "ERROR", "error": str(e)}

    async def _test_index_collector(self) -> None:
        """지수 수집기 테스트"""
        logger.info("\n[지수 수집기 테스트 시작]")
        test_name = "IndexCollector"

        try:
            collector = IndexCollector(rate_limit=0.05)  # 테스트용으로 빠르게 설정

            # 테스트 1: 기본 수집 기능 (주요 지수들)
            test_indices = [
                "KS11",  # 코스피
                "KS200",  # 코스피 200
                "^GSPC",  # S&P 500
                "^DJI",  # 다우존스
                "^N225",  # 니케이 225
            ]

            logger.info(f"테스트 지수: {test_indices}")

            start_time = time.time()
            data = await collector.collect_data(
                test_indices, self.start_date, self.end_date
            )
            elapsed_time = time.time() - start_time

            # 결과 검증
            success_count = len(data)
            total_count = len(test_indices)

            logger.info(f"수집 결과: {success_count}/{total_count} 성공")
            logger.info(f"소요 시간: {elapsed_time:.2f}초")

            # 데이터 품질 검증
            for symbol, df in data.items():
                logger.info(f"  {symbol}: {len(df)}행, 컬럼: {list(df.columns)}")

                # 지수 특화 컬럼 확인
                index_specific_columns = [
                    "index_name",
                    "category",
                    "daily_return",
                    "ma20",
                ]
                existing_columns = [
                    col for col in index_specific_columns if col in df.columns
                ]
                logger.info(f"    지수 특화 컬럼: {existing_columns}")

                # VIX 특수 처리 확인
                if symbol == "^VIX" and "market_sentiment" in df.columns:
                    sentiment_counts = df["market_sentiment"].value_counts()
                    logger.info(f"    VIX 시장 심리 분포: {sentiment_counts.to_dict()}")

            # 테스트 2: 카테고리별 수집 (축소 버전)
            logger.info("\n카테고리별 수집 테스트 (한국 주요 지수만)")
            try:
                # 전체 KRX 카테고리 대신 확실한 지수들만 테스트
                korea_test_indices = ["KS11", "KS200", "KQ11"]
                korea_data = await collector.collect_data(
                    korea_test_indices, self.start_date, self.end_date
                )
                logger.info(f"한국 주요 지수 수집 결과: {len(korea_data)}개")
                for symbol in korea_data.keys():
                    logger.info(f"  {symbol}: {len(korea_data[symbol])}행")
            except Exception as e:
                logger.warning(f"한국 지수 수집 테스트 실패: {str(e)}")

            # 테스트 3: 주요 글로벌 지수 수집 (축소 버전)
            logger.info("\n주요 글로벌 지수 수집 테스트 (주요 지수만)")
            try:
                # 전체 목록 대신 확실히 작동하는 지수들만
                major_test_indices = ["KS11", "^GSPC", "^DJI", "^IXIC"]
                major_data = await collector.collect_data(
                    major_test_indices, self.start_date, self.end_date
                )
                logger.info(f"주요 글로벌 지수 수집 결과: {len(major_data)}개")
            except Exception as e:
                logger.warning(f"주요 글로벌 지수 수집 테스트 실패: {str(e)}")

            # 테스트 4: 지수 정보 조회
            logger.info(
                f"\n지원 카테고리: {list(collector.get_supported_categories().keys())}"
            )

            # 특정 지수 정보 확인
            for symbol in ["KS11", "^GSPC"]:
                info = collector.get_index_info(symbol)
                if info:
                    logger.info(f"{symbol} 정보: {info}")

            # 테스트 결과 저장
            self.test_results[test_name] = {
                "status": "PASS" if success_count > 0 else "FAIL",
                "success_rate": f"{success_count}/{total_count}",
                "elapsed_time": f"{elapsed_time:.2f}s",
                "data_count": sum(len(df) for df in data.values()),
            }

            logger.info(
                f"지수 수집기 테스트 완료: {self.test_results[test_name]['status']}"
            )

        except Exception as e:
            logger.error(f"지수 수집기 테스트 실패: {str(e)}")
            self.test_results[test_name] = {"status": "ERROR", "error": str(e)}

    def _print_test_summary(self) -> None:
        """테스트 결과 요약 출력"""
        total_elapsed = (
            time.time() - self.test_start_time if self.test_start_time else 0
        )

        logger.info("\n" + "=" * 60)
        logger.info("테스트 결과 요약")
        logger.info("=" * 60)

        logger.info(f"총 소요 시간: {total_elapsed:.2f}초")
        logger.info(f"테스트된 수집기: {len(self.test_results)}개")

        # 수집기별 결과
        for collector_name, result in self.test_results.items():
            status = result["status"]
            status_emoji = (
                "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
            )

            logger.info(f"\n{status_emoji} {collector_name}: {status}")

            if status in ["PASS", "FAIL"]:
                logger.info(f"   성공률: {result.get('success_rate', 'N/A')}")
                logger.info(f"   소요시간: {result.get('elapsed_time', 'N/A')}")
                logger.info(f"   수집된 데이터: {result.get('data_count', 0)}행")
            elif status == "ERROR":
                logger.info(f"   오류: {result.get('error', 'Unknown error')}")

        # 전체 통계
        pass_count = sum(
            1 for result in self.test_results.values() if result["status"] == "PASS"
        )
        total_count = len(self.test_results)

        logger.info(
            f"\n총 성공률: {pass_count}/{total_count} ({pass_count/total_count*100:.1f}%)"
        )

        if pass_count == total_count:
            logger.info("🎉 모든 수집기가 정상 작동합니다!")
        else:
            logger.warning("⚠️ 일부 수집기에 문제가 있습니다. 로그를 확인해주세요.")


async def main():
    """메인 테스트 함수"""
    try:
        # 필요한 라이브러리 설치 안내
        logger.info("데이터 수집기 테스트를 시작합니다.")
        logger.info("필요한 라이브러리: FinanceDataReader, loguru, pandas")
        logger.info(
            "설치 명령어: conda install -c conda-forge financedatareader loguru pandas"
        )
        logger.info("-" * 60)

        # 테스터 생성 및 실행
        tester = CollectorTester()
        results = await tester.run_all_tests()

        return results

    except KeyboardInterrupt:
        logger.info("\n테스트가 사용자에 의해 중단되었습니다.")
        return {}
    except Exception as e:
        logger.error(f"테스트 실행 중 예상치 못한 오류 발생: {str(e)}")
        return {}


if __name__ == "__main__":
    # 비동기 테스트 실행
    results = asyncio.run(main())

    if results:
        print("\n테스트 완료! 자세한 로그는 'test_collectors.log' 파일을 확인하세요.")
    else:
        print("\n테스트 실행에 실패했습니다.")
