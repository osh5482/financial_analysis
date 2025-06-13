"""
기본 데이터 수집기 추상 클래스
모든 금융 데이터 수집기가 상속받아야 하는 베이스 클래스
"""

import asyncio
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
import pandas as pd
import FinanceDataReader as fdr
from loguru import logger


class BaseCollector(ABC):
    """
    모든 금융 데이터 수집기의 기본 클래스
    비동기 방식으로 데이터 수집을 수행하며, 공통 기능을 제공
    """

    def __init__(self, name: str, rate_limit: float = 0.1):
        """
        기본 수집기 초기화

        Args:
            name: 수집기 이름 (로깅 용도)
            rate_limit: API 호출 간 최소 대기 시간 (초)
        """
        self.name = name
        self.rate_limit = rate_limit
        self.session_start_time = None
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0

        logger.info(f"{self.name} 수집기 초기화 완료 (Rate Limit: {rate_limit}초)")

    async def collect_data(
        self,
        symbols: list[str],
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, pd.DataFrame]:
        """
        비동기 방식으로 여러 심볼의 데이터를 수집

        Args:
            symbols: 수집할 심볼 리스트
            start_date: 시작 날짜 (YYYY-MM-DD 형식)
            end_date: 종료 날짜 (YYYY-MM-DD 형식)

        Returns:
            dict: {심볼: DataFrame} 형태의 수집된 데이터
        """
        self.session_start_time = time.time()
        self.total_requests = len(symbols)
        self.successful_requests = 0
        self.failed_requests = 0

        logger.info(f"{self.name} 데이터 수집 시작 - 대상: {len(symbols)}개 심볼")

        # 기본 날짜 설정 (지정되지 않은 경우 최근 1년)
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

        logger.info(f"수집 기간: {start_date} ~ {end_date}")

        # 세마포어를 사용하여 동시 요청 수 제한 (최대 5개)
        semaphore = asyncio.Semaphore(5)

        # 비동기 작업 생성
        tasks = [
            self._collect_single_symbol(semaphore, symbol, start_date, end_date)
            for symbol in symbols
        ]

        # 모든 작업 실행
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 결과 정리
        collected_data = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.error(f"{symbol} 수집 실패: {str(result)}")
                self.failed_requests += 1
            elif result is not None and not result.empty:
                collected_data[symbol] = result
                self.successful_requests += 1
                logger.debug(f"{symbol} 수집 성공: {len(result)}행")
            else:
                logger.warning(f"{symbol} 수집 결과 없음")
                self.failed_requests += 1

        self._log_session_summary()
        return collected_data

    async def _collect_single_symbol(
        self, semaphore: asyncio.Semaphore, symbol: str, start_date: str, end_date: str
    ) -> pd.DataFrame | None:
        """
        단일 심볼의 데이터를 비동기로 수집

        Args:
            semaphore: 동시 요청 수 제한용 세마포어
            symbol: 수집할 심볼
            start_date: 시작 날짜
            end_date: 종료 날짜

        Returns:
            DataFrame: 수집된 데이터 또는 None
        """
        async with semaphore:
            try:
                # Rate limiting 적용
                await asyncio.sleep(self.rate_limit)

                # 실제 데이터 수집 (구체적인 구현은 각 수집기에서 오버라이드)
                data = await self._fetch_data(symbol, start_date, end_date)

                if data is not None and not data.empty:
                    # 데이터 후처리
                    processed_data = self._process_raw_data(data, symbol)
                    logger.debug(f"{symbol} 원시 데이터 처리 완료")
                    return processed_data
                else:
                    logger.warning(f"{symbol} 데이터 없음")
                    return None

            except Exception as e:
                logger.error(f"{symbol} 수집 중 오류 발생: {str(e)}")
                raise e

    @abstractmethod
    async def _fetch_data(
        self, symbol: str, start_date: str, end_date: str
    ) -> pd.DataFrame | None:
        """
        실제 데이터 수집 로직 (각 수집기에서 구현 필요)

        Args:
            symbol: 수집할 심볼
            start_date: 시작 날짜
            end_date: 종료 날짜

        Returns:
            DataFrame: 수집된 원시 데이터
        """
        pass

    def _process_raw_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        수집된 원시 데이터를 후처리

        Args:
            data: 원시 데이터
            symbol: 심볼명

        Returns:
            DataFrame: 처리된 데이터
        """
        processed_data = data.copy()

        # 공통 컬럼명 표준화
        column_mapping = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
            "Adj Close": "adj_close",
        }

        # 컬럼명 변경 (존재하는 컬럼만)
        for old_name, new_name in column_mapping.items():
            if old_name in processed_data.columns:
                processed_data = processed_data.rename(columns={old_name: new_name})

        # 심볼 컬럼 추가
        processed_data["symbol"] = symbol

        # 날짜 인덱스를 컬럼으로 변환
        if processed_data.index.name == "Date" or isinstance(
            processed_data.index, pd.DatetimeIndex
        ):
            processed_data = processed_data.reset_index()
            if "Date" in processed_data.columns:
                processed_data = processed_data.rename(columns={"Date": "date"})
            elif processed_data.columns[0] == "index":
                processed_data = processed_data.rename(columns={"index": "date"})

        # 날짜 컬럼 타입 확인 및 변환
        if "date" in processed_data.columns:
            processed_data["date"] = pd.to_datetime(processed_data["date"])

        # 결측치 확인 및 로깅 (계산된 지표의 결측치는 DEBUG 레벨로)
        null_counts = processed_data.isnull().sum()
        if null_counts.sum() > 0:
            # 원본 데이터 컬럼의 결측치만 WARNING으로 로깅
            original_columns = ["open", "high", "low", "close", "volume"]
            critical_nulls = {
                col: count
                for col, count in null_counts.items()
                if col in original_columns and count > 0
            }

            if critical_nulls:
                logger.warning(f"{symbol} 원본 데이터 결측치 발견: {critical_nulls}")

            # 계산된 지표의 결측치는 DEBUG 레벨로
            calculated_nulls = {
                col: count
                for col, count in null_counts.items()
                if col not in original_columns and count > 0
            }
            if calculated_nulls:
                logger.debug(f"{symbol} 계산된 지표 결측치 (정상): {calculated_nulls}")

        # 데이터 정렬 (날짜 기준)
        if "date" in processed_data.columns:
            processed_data = processed_data.sort_values("date")

        logger.debug(
            f"{symbol} 데이터 후처리 완료: {len(processed_data)}행, {len(processed_data.columns)}열"
        )
        return processed_data

    def _log_session_summary(self) -> None:
        """
        수집 세션 요약 정보를 로깅
        """
        if self.session_start_time:
            elapsed_time = time.time() - self.session_start_time
            success_rate = (
                (self.successful_requests / self.total_requests * 100)
                if self.total_requests > 0
                else 0
            )

            logger.info(f"{self.name} 수집 세션 완료")
            logger.info(f"총 소요시간: {elapsed_time:.2f}초")
            logger.info(f"총 요청: {self.total_requests}개")
            logger.info(f"성공: {self.successful_requests}개")
            logger.info(f"실패: {self.failed_requests}개")
            logger.info(f"성공률: {success_rate:.1f}%")

            if self.successful_requests > 0:
                avg_time_per_request = elapsed_time / self.total_requests
                logger.info(f"평균 처리시간: {avg_time_per_request:.2f}초/건")

    async def validate_symbols(self, symbols: list[str]) -> tuple[list[str], list[str]]:
        """
        심볼 유효성 검증

        Args:
            symbols: 검증할 심볼 리스트

        Returns:
            tuple: (유효한 심볼 리스트, 무효한 심볼 리스트)
        """
        logger.info(f"{len(symbols)}개 심볼 유효성 검증 시작")

        valid_symbols = []
        invalid_symbols = []

        # 간단한 형식 검증 (각 수집기에서 오버라이드 가능)
        for symbol in symbols:
            if await self._is_valid_symbol(symbol):
                valid_symbols.append(symbol)
            else:
                invalid_symbols.append(symbol)

        logger.info(
            f"유효한 심볼: {len(valid_symbols)}개, 무효한 심볼: {len(invalid_symbols)}개"
        )

        if invalid_symbols:
            logger.warning(f"무효한 심볼들: {invalid_symbols}")

        return valid_symbols, invalid_symbols

    async def _is_valid_symbol(self, symbol: str) -> bool:
        """
        개별 심볼의 유효성 검증 (기본 구현, 각 수집기에서 오버라이드 가능)

        Args:
            symbol: 검증할 심볼

        Returns:
            bool: 유효성 여부
        """
        # 기본적인 형식 검증
        if not symbol or not isinstance(symbol, str):
            return False

        # 빈 문자열이나 공백만 있는 경우
        if not symbol.strip():
            return False

        # 기본적으로 모든 문자열을 유효하다고 가정 (각 수집기에서 구체적 검증)
        return True

    def get_collection_stats(self) -> dict[str, int | float]:
        """
        현재 세션의 수집 통계 반환

        Returns:
            dict: 수집 통계 정보
        """
        elapsed_time = (
            time.time() - self.session_start_time if self.session_start_time else 0
        )
        success_rate = (
            (self.successful_requests / self.total_requests * 100)
            if self.total_requests > 0
            else 0
        )

        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": round(success_rate, 2),
            "elapsed_time": round(elapsed_time, 2),
        }
