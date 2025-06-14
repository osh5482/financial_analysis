"""
기본 데이터 수집기 추상 클래스
- 모든 개별 수집기가 상속받을 공통 기능 정의
- 에러 핸들링, 재시도 로직, 데이터 검증, 성능 모니터링 포함
"""

import time
import pandas as pd
from abc import ABC, abstractmethod
from pathlib import Path
from datetime import datetime, timedelta
from loguru import logger

from config.settings import (
    DEFAULT_CONFIG,
    VALIDATION_CONFIG,
    FILE_CONFIG,
    DATA_PATHS,
)


class BaseCollector(ABC):
    """
    모든 데이터 수집기의 기본 클래스

    공통 기능:
    - API 호출 재시도 로직
    - 데이터 검증 및 정제
    - 파일 저장/로드
    - 로깅 및 성능 모니터링
    """

    def __init__(
        self,
        name: str,
        start_date: str | None = None,
        end_date: str | None = None,
        max_retries: int | None = None,
        retry_delay: float | None = None,
        rate_limit_delay: float | None = None,
    ) -> None:
        """
        BaseCollector 초기화

        Args:
            name: 수집기 이름 (로깅 및 식별용)
            start_date: 수집 시작일 (YYYY-MM-DD)
            end_date: 수집 종료일 (YYYY-MM-DD)
            max_retries: 최대 재시도 횟수
            retry_delay: 재시도 간격 (초)
            rate_limit_delay: API 호출 제한 간격 (초)
        """
        self.name = name
        self.start_date = start_date or DEFAULT_CONFIG["start_date"]
        self.end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        self.max_retries = max_retries or DEFAULT_CONFIG["max_retries"]
        self.retry_delay = retry_delay or DEFAULT_CONFIG["retry_delay"]
        self.rate_limit_delay = rate_limit_delay or DEFAULT_CONFIG["rate_limit_delay"]

        # 성능 측정 변수
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_processing_time = 0.0

        # 로거 설정
        self.logger = logger.bind(collector=self.name)
        self.logger.info(f"{self.name} 수집기 초기화 완료")
        self.logger.info(f"수집 기간: {self.start_date} ~ {self.end_date}")

    @abstractmethod
    def collect_data(self, symbols: list[str]) -> dict[str, pd.DataFrame]:
        """
        데이터 수집 메인 메서드 (하위 클래스에서 구현 필수)

        Args:
            symbols: 수집할 심볼 리스트

        Returns:
            심볼별 데이터프레임 딕셔너리
        """
        pass

    @abstractmethod
    def get_symbol_list(self) -> list[str]:
        """
        수집 대상 심볼 리스트 반환 (하위 클래스에서 구현 필수)

        Returns:
            심볼 리스트
        """
        pass

    def _retry_request(self, func: callable, *args, **kwargs) -> any:
        """
        API 요청 재시도 로직

        Args:
            func: 실행할 함수
            *args: 함수 인자
            **kwargs: 함수 키워드 인자

        Returns:
            함수 실행 결과

        Raises:
            Exception: 최대 재시도 횟수 초과시
        """
        self.total_requests += 1

        for attempt in range(self.max_retries + 1):
            try:
                start_time = time.time()
                result = func(*args, **kwargs)

                # 성공시 통계 업데이트
                processing_time = time.time() - start_time
                self.total_processing_time += processing_time
                self.successful_requests += 1

                if attempt > 0:
                    self.logger.info(f"재시도 {attempt}회 후 성공")

                # API 호출 제한 준수
                if self.rate_limit_delay > 0:
                    time.sleep(self.rate_limit_delay)

                return result

            except Exception as e:
                if attempt < self.max_retries:
                    self.logger.warning(
                        f"요청 실패 (시도 {attempt + 1}/{self.max_retries + 1}): {str(e)}"
                    )
                    time.sleep(self.retry_delay * (attempt + 1))  # 지수적 백오프
                else:
                    self.failed_requests += 1
                    self.logger.error(f"최대 재시도 횟수 초과: {str(e)}")
                    raise

    def validate_dataframe(
        self,
        df: pd.DataFrame,
        symbol: str,
        required_columns: list[str] | None = None,
        min_data_points: int | None = None,
    ) -> bool:
        """
        DataFrame 데이터 검증

        Args:
            df: 검증할 DataFrame
            symbol: 심볼명 (로깅용)
            required_columns: 필수 컬럼 리스트
            min_data_points: 최소 데이터 포인트 (None시 기본값 사용)

        Returns:
            검증 통과 여부
        """
        if df is None or df.empty:
            self.logger.warning(f"{symbol}: 빈 데이터프레임")
            return False

        # 최소 데이터 포인트 검증 (동적 조정 가능)
        min_points = (
            min_data_points
            if min_data_points is not None
            else VALIDATION_CONFIG["min_data_points"]
        )
        if len(df) < min_points:
            self.logger.warning(
                f"{symbol}: 데이터 포인트 부족 ({len(df)} < {min_points})"
            )
            return False

        # 필수 컬럼 검증
        if required_columns:
            missing_columns = set(required_columns) - set(df.columns)
            if missing_columns:
                self.logger.error(f"{symbol}: 필수 컬럼 누락 {list(missing_columns)}")
                return False

        # 결측치 비율 검증
        max_missing_ratio = VALIDATION_CONFIG["max_missing_ratio"]
        missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))

        if missing_ratio > max_missing_ratio:
            self.logger.warning(
                f"{symbol}: 높은 결측치 비율 ({missing_ratio:.2%} > {max_missing_ratio:.2%})"
            )
            return False

        # 날짜 인덱스 검증 (있는 경우)
        if hasattr(df.index, "dtype") and "datetime" in str(df.index.dtype):
            # 중복 날짜 검증
            if df.index.duplicated().any():
                self.logger.warning(f"{symbol}: 중복 날짜 발견")
                df = df[~df.index.duplicated(keep="last")]

        self.logger.info(
            f"{symbol}: 검증 통과 - 행수={len(df)}, 열수={len(df.columns)}, "
            f"결측치={missing_ratio:.2%}"
        )
        return True

    def clean_dataframe(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        DataFrame 데이터 정제

        Args:
            df: 정제할 DataFrame
            symbol: 심볼명 (로깅용)

        Returns:
            정제된 DataFrame
        """
        original_length = len(df)

        # 중복 제거 (날짜 기준)
        if hasattr(df.index, "dtype") and "datetime" in str(df.index.dtype):
            df = df[~df.index.duplicated(keep="last")]

        # 극값 제거 (가격 데이터가 있는 경우)
        price_columns = ["Open", "High", "Low", "Close", "open", "high", "low", "close"]
        existing_price_cols = [col for col in price_columns if col in df.columns]

        if existing_price_cols:
            multiplier = VALIDATION_CONFIG["price_range_multiplier"]
            for col in existing_price_cols:
                if col in df.columns:
                    median_price = df[col].median()
                    lower_bound = median_price / multiplier
                    upper_bound = median_price * multiplier

                    outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
                    if outliers.any():
                        self.logger.warning(
                            f"{symbol}: {col} 극값 {outliers.sum()}개 제거"
                        )
                        df = df[~outliers]

        # 정렬 (날짜순)
        if hasattr(df.index, "dtype") and "datetime" in str(df.index.dtype):
            df = df.sort_index()

        cleaned_length = len(df)
        if cleaned_length != original_length:
            self.logger.info(
                f"{symbol}: 데이터 정제 완료 ({original_length} → {cleaned_length})"
            )

        return df

    def save_to_file(
        self, df: pd.DataFrame, file_path: Path, file_format: str = "csv"
    ) -> bool:
        """
        DataFrame을 파일로 저장

        Args:
            df: 저장할 DataFrame
            file_path: 저장 경로
            file_format: 파일 형식 ("csv" 또는 "parquet")

        Returns:
            저장 성공 여부
        """
        try:
            # 디렉토리 생성
            file_path.parent.mkdir(parents=True, exist_ok=True)

            if file_format.lower() == "csv":
                df.to_csv(
                    file_path,
                    encoding=FILE_CONFIG["csv_encoding"],
                    float_format=f"%.{FILE_CONFIG['float_precision']}f",
                )
            elif file_format.lower() == "parquet":
                df.to_parquet(file_path, compression=FILE_CONFIG["parquet_compression"])
            else:
                raise ValueError(f"지원하지 않는 파일 형식: {file_format}")

            self.logger.info(f"파일 저장 완료: {file_path}")
            return True

        except Exception as e:
            self.logger.error(f"파일 저장 실패 {file_path}: {str(e)}")
            return False

    def load_from_file(
        self, file_path: Path, file_format: str = "csv"
    ) -> pd.DataFrame | None:
        """
        파일에서 DataFrame 로드

        Args:
            file_path: 로드할 파일 경로
            file_format: 파일 형식 ("csv" 또는 "parquet")

        Returns:
            로드된 DataFrame 또는 None
        """
        try:
            if not file_path.exists():
                self.logger.warning(f"파일이 존재하지 않음: {file_path}")
                return None

            if file_format.lower() == "csv":
                df = pd.read_csv(
                    file_path, index_col=0, encoding=FILE_CONFIG["csv_encoding"]
                )
            elif file_format.lower() == "parquet":
                df = pd.read_parquet(file_path)
            else:
                raise ValueError(f"지원하지 않는 파일 형식: {file_format}")

            # 날짜 인덱스 변환 (필요한 경우)
            if "date" in df.index.name.lower() if df.index.name else False:
                df.index = pd.to_datetime(df.index)

            self.logger.info(f"파일 로드 완료: {file_path} (행수: {len(df)})")
            return df

        except Exception as e:
            self.logger.error(f"파일 로드 실패 {file_path}: {str(e)}")
            return None

    def get_performance_stats(self) -> dict[str, any]:
        """
        수집기 성능 통계 반환

        Returns:
            성능 통계 딕셔너리
        """
        success_rate = (
            self.successful_requests / self.total_requests * 100
            if self.total_requests > 0
            else 0
        )

        avg_processing_time = (
            self.total_processing_time / self.successful_requests
            if self.successful_requests > 0
            else 0
        )

        return {
            "collector_name": self.name,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": f"{success_rate:.2f}%",
            "total_processing_time": f"{self.total_processing_time:.2f}초",
            "average_processing_time": f"{avg_processing_time:.4f}초",
        }

    def log_performance_summary(self) -> None:
        """성능 통계 로그 출력"""
        stats = self.get_performance_stats()

        self.logger.info("=" * 50)
        self.logger.info(f"{self.name} 수집기 성능 요약")
        self.logger.info("=" * 50)

        for key, value in stats.items():
            if key != "collector_name":
                self.logger.info(f"{key}: {value}")

        self.logger.info("=" * 50)

    def run_collection(
        self, symbols: list[str] | None = None
    ) -> dict[str, pd.DataFrame]:
        """
        전체 수집 프로세스 실행

        Args:
            symbols: 수집할 심볼 리스트 (None시 모든 심볼)

        Returns:
            수집된 데이터 딕셔너리
        """
        start_time = time.time()

        # 심볼 리스트 확정
        if symbols is None:
            symbols = self.get_symbol_list()

        self.logger.info(f"데이터 수집 시작: {len(symbols)}개 심볼")

        try:
            # 데이터 수집 실행
            collected_data = self.collect_data(symbols)

            # 수집 완료 로그
            total_time = time.time() - start_time
            successful_symbols = len(
                [k for k, v in collected_data.items() if v is not None]
            )

            self.logger.info(
                f"데이터 수집 완료: {successful_symbols}/{len(symbols)}개 성공, "
                f"총 소요시간: {total_time:.2f}초"
            )

            # 성능 요약 출력
            self.log_performance_summary()

            return collected_data

        except Exception as e:
            total_time = time.time() - start_time
            self.logger.error(
                f"데이터 수집 실패: {str(e)}, 소요시간: {total_time:.2f}초"
            )
            raise
