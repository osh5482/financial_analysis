"""
주식 데이터 수집기
- FinanceDataReader를 활용한 주식 데이터 수집
- 한국(KRX), 미국(NASDAQ, NYSE, S&P500) 주식 지원
- 거래량 0 검증, 가격 일관성 검증 포함
"""

import pandas as pd
import FinanceDataReader as fdr
from pathlib import Path
from datetime import datetime, timedelta

from .base_collector import BaseCollector
from config.settings import (
    STOCK_EXCHANGES,
    VALIDATION_CONFIG,
    DATA_PATHS,
    get_enabled_stock_exchanges,
)


class StockCollector(BaseCollector):
    """
    주식 데이터 수집기

    기능:
    - 거래소별 주식 리스트 자동 수집
    - 개별 주식 가격 데이터 수집
    - 주식 특화 데이터 검증 (거래량, 가격 일관성)
    - 배치 처리 및 증분 업데이트 지원
    """

    def __init__(
        self,
        exchange: str = "KRX",
        start_date: str | None = None,
        end_date: str | None = None,
        include_delisted: bool = False,
        batch_size: int | None = None,
        **kwargs,
    ) -> None:
        """
        StockCollector 초기화

        Args:
            exchange: 거래소 코드 (KRX, NASDAQ, NYSE, SP500)
            start_date: 수집 시작일
            end_date: 수집 종료일
            include_delisted: 상장폐지 종목 포함 여부
            batch_size: 배치 처리 크기
            **kwargs: BaseCollector 추가 인자
        """
        super().__init__(f"StockCollector_{exchange}", start_date, end_date, **kwargs)

        self.exchange = exchange.upper()
        self.include_delisted = include_delisted
        self.batch_size = batch_size or 50

        # 거래소 설정 검증
        if self.exchange not in STOCK_EXCHANGES:
            raise ValueError(f"지원하지 않는 거래소: {exchange}")

        self.exchange_config = STOCK_EXCHANGES[self.exchange]

        # 데이터 저장 경로 설정
        self.data_path = DATA_PATHS["stocks"] / self.exchange.lower()
        self.symbol_list_path = (
            self.data_path / f"{self.exchange_config['file_prefix']}_list.csv"
        )

        self.logger.info(
            f"거래소: {self.exchange} ({self.exchange_config['description']})"
        )
        self.logger.info(f"배치 크기: {self.batch_size}")

    def get_symbol_list(self) -> list[str]:
        """
        거래소별 주식 심볼 리스트 반환

        Returns:
            주식 심볼 리스트
        """
        try:
            self.logger.info(f"{self.exchange} 주식 리스트 수집 시작")

            # FinanceDataReader를 통한 주식 리스트 수집
            if self.exchange == "KRX":
                # 한국 전체 (KOSPI + KOSDAQ + KONEX)
                symbol_df = fdr.StockListing(self.exchange)

            elif self.exchange in ["NASDAQ", "NYSE"]:
                # 미국 거래소별
                symbol_df = fdr.StockListing(self.exchange)

            elif self.exchange == "SP500":
                # S&P 500 구성종목
                symbol_df = fdr.StockListing("S&P500")

            else:
                raise ValueError(f"지원하지 않는 거래소: {self.exchange}")

            # 데이터 검증
            if symbol_df is None or symbol_df.empty:
                self.logger.error(f"{self.exchange} 주식 리스트 수집 실패")
                return []

            # 상장폐지 종목 필터링
            if not self.include_delisted and "Market" in symbol_df.columns:
                # 상장폐지된 종목 제외
                symbol_df = symbol_df[symbol_df["Market"].notna()]

            # 심볼 리스트 추출
            if "Symbol" in symbol_df.columns:
                symbols = symbol_df["Symbol"].tolist()
            elif "Code" in symbol_df.columns:
                symbols = symbol_df["Code"].tolist()
            else:
                # 인덱스가 심볼인 경우
                symbols = symbol_df.index.tolist()

            # 심볼 정제 (문자열 변환 및 공백 제거)
            symbols = [str(symbol).strip() for symbol in symbols if pd.notna(symbol)]

            # 주식 리스트 저장
            self._save_symbol_list(symbol_df)

            self.logger.info(f"{self.exchange} 주식 리스트 수집 완료: {len(symbols)}개")
            return symbols

        except Exception as e:
            self.logger.error(f"{self.exchange} 주식 리스트 수집 실패: {str(e)}")

            # 기존 저장된 리스트 사용 시도
            return self._load_cached_symbol_list()

    def _save_symbol_list(self, symbol_df: pd.DataFrame) -> None:
        """주식 리스트를 파일로 저장"""
        try:
            # 타임스탬프 추가
            symbol_df["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 파일 저장
            self.save_to_file(symbol_df, self.symbol_list_path, "csv")
            self.logger.info(f"주식 리스트 저장 완료: {self.symbol_list_path}")

        except Exception as e:
            self.logger.error(f"주식 리스트 저장 실패: {str(e)}")

    def _load_cached_symbol_list(self) -> list[str]:
        """캐시된 주식 리스트 로드"""
        try:
            symbol_df = self.load_from_file(self.symbol_list_path, "csv")

            if symbol_df is not None and not symbol_df.empty:
                if "Symbol" in symbol_df.columns:
                    symbols = symbol_df["Symbol"].tolist()
                elif "Code" in symbol_df.columns:
                    symbols = symbol_df["Code"].tolist()
                else:
                    symbols = symbol_df.index.tolist()

                symbols = [
                    str(symbol).strip() for symbol in symbols if pd.notna(symbol)
                ]

                self.logger.info(f"캐시된 주식 리스트 로드: {len(symbols)}개")
                return symbols

        except Exception as e:
            self.logger.error(f"캐시된 주식 리스트 로드 실패: {str(e)}")

        return []

    def collect_single_stock(self, symbol: str) -> pd.DataFrame | None:
        """
        개별 주식 데이터 수집

        Args:
            symbol: 주식 심볼

        Returns:
            주식 가격 데이터 DataFrame 또는 None
        """
        try:
            # FinanceDataReader로 데이터 수집
            data = fdr.DataReader(symbol, self.start_date, self.end_date)

            if data is None or data.empty:
                self.logger.warning(f"{symbol}: 데이터 없음")
                return None

            # 인덱스를 날짜로 변환 (필요한 경우)
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)

            # 컬럼명 표준화 (첫 글자 대문자)
            column_mapping = {
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
                "change": "Change",
            }

            for old_col, new_col in column_mapping.items():
                if old_col in data.columns:
                    data.rename(columns={old_col: new_col}, inplace=True)

            # 주식 특화 데이터 검증
            if not self._validate_stock_data(data, symbol):
                return None

            # 데이터 정제
            data = self.clean_dataframe(data, symbol)

            # 기본 검증 (테스트용 완화된 기준 적용)
            required_columns = ["Open", "High", "Low", "Close", "Volume"]
            min_points = getattr(self, "min_data_points", None)  # 테스트용 완화 기준
            if not self.validate_dataframe(data, symbol, required_columns, min_points):
                return None

            self.logger.debug(f"{symbol}: 데이터 수집 성공 ({len(data)}개 레코드)")
            return data

        except Exception as e:
            self.logger.error(f"{symbol}: 데이터 수집 실패 - {str(e)}")
            return None

    def _validate_stock_data(self, df: pd.DataFrame, symbol: str) -> bool:
        """
        주식 특화 데이터 검증

        Args:
            df: 검증할 DataFrame
            symbol: 주식 심볼

        Returns:
            검증 통과 여부
        """
        try:
            # 가격 컬럼 존재 확인
            price_columns = ["Open", "High", "Low", "Close"]
            missing_price_cols = [col for col in price_columns if col not in df.columns]

            if missing_price_cols:
                self.logger.error(f"{symbol}: 필수 가격 컬럼 누락 {missing_price_cols}")
                return False

            # 가격 일관성 검증 (High >= Low, High >= Open/Close, Low <= Open/Close)
            invalid_prices = (
                (df["High"] < df["Low"])
                | (df["High"] < df["Open"])
                | (df["High"] < df["Close"])
                | (df["Low"] > df["Open"])
                | (df["Low"] > df["Close"])
            )

            if invalid_prices.any():
                invalid_count = invalid_prices.sum()
                self.logger.warning(f"{symbol}: 가격 일관성 오류 {invalid_count}건")

                # 전체의 5% 이상이면 데이터 품질 문제로 판단
                if invalid_count / len(df) > 0.05:
                    return False

            # 거래량 검증
            if "Volume" in df.columns:
                zero_volume_ratio = (df["Volume"] == 0).sum() / len(df)
                threshold = VALIDATION_CONFIG["volume_zero_threshold"]

                if zero_volume_ratio > threshold:
                    self.logger.warning(
                        f"{symbol}: 높은 거래량 0 비율 ({zero_volume_ratio:.2%} > {threshold:.2%})"
                    )
                    return False

            # 음수 가격 확인
            for col in price_columns:
                if (df[col] <= 0).any():
                    self.logger.error(f"{symbol}: {col}에 음수 또는 0 가격 존재")
                    return False

            return True

        except Exception as e:
            self.logger.error(f"{symbol}: 주식 데이터 검증 중 오류 - {str(e)}")
            return False

    def collect_data(self, symbols: list[str]) -> dict[str, pd.DataFrame]:
        """
        여러 주식 데이터 배치 수집

        Args:
            symbols: 수집할 주식 심볼 리스트

        Returns:
            심볼별 데이터 딕셔너리
        """
        collected_data = {}
        total_symbols = len(symbols)

        self.logger.info(f"주식 데이터 배치 수집 시작: {total_symbols}개")

        # 배치별로 처리
        for i in range(0, total_symbols, self.batch_size):
            batch_symbols = symbols[i : i + self.batch_size]
            batch_num = i // self.batch_size + 1
            total_batches = (total_symbols + self.batch_size - 1) // self.batch_size

            self.logger.info(
                f"배치 {batch_num}/{total_batches} 처리 중: {len(batch_symbols)}개 심볼"
            )

            # 배치 내 각 심볼 처리
            for j, symbol in enumerate(batch_symbols):
                self.logger.debug(f"처리 중: {symbol} ({i+j+1}/{total_symbols})")

                # 재시도 로직과 함께 데이터 수집
                stock_data = self._retry_request(self.collect_single_stock, symbol)

                if stock_data is not None and not stock_data.empty:
                    collected_data[symbol] = stock_data

                    # 개별 파일로 저장 (옵션)
                    if (
                        hasattr(self, "save_individual_files")
                        and self.save_individual_files
                    ):
                        file_path = self.data_path / f"{symbol}.csv"
                        self.save_to_file(stock_data, file_path, "csv")
                else:
                    collected_data[symbol] = None

            # 배치 완료 로그
            successful_in_batch = sum(
                1 for symbol in batch_symbols if collected_data.get(symbol) is not None
            )
            self.logger.info(
                f"배치 {batch_num} 완료: {successful_in_batch}/{len(batch_symbols)}개 성공"
            )

        successful_total = len([v for v in collected_data.values() if v is not None])
        self.logger.info(f"전체 수집 완료: {successful_total}/{total_symbols}개 성공")

        return collected_data

    def get_stock_info(self, symbol: str) -> dict[str, any] | None:
        """
        개별 주식 정보 조회

        Args:
            symbol: 주식 심볼

        Returns:
            주식 정보 딕셔너리 또는 None
        """
        try:
            # 저장된 주식 리스트에서 정보 조회
            symbol_df = self.load_from_file(self.symbol_list_path, "csv")

            if symbol_df is not None:
                if "Symbol" in symbol_df.columns:
                    stock_info = symbol_df[symbol_df["Symbol"] == symbol]
                elif "Code" in symbol_df.columns:
                    stock_info = symbol_df[symbol_df["Code"] == symbol]
                else:
                    stock_info = symbol_df[symbol_df.index == symbol]

                if not stock_info.empty:
                    return stock_info.iloc[0].to_dict()

            self.logger.warning(f"{symbol}: 주식 정보 없음")
            return None

        except Exception as e:
            self.logger.error(f"{symbol}: 주식 정보 조회 실패 - {str(e)}")
            return None

    def update_data(
        self, symbols: list[str] | None = None, days_back: int = 7
    ) -> dict[str, pd.DataFrame]:
        """
        증분 데이터 업데이트 (최근 N일)

        Args:
            symbols: 업데이트할 심볼 리스트 (None시 전체)
            days_back: 업데이트할 과거 일수

        Returns:
            업데이트된 데이터 딕셔너리
        """
        # 업데이트 기간 설정
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

        # 임시로 수집 기간 변경
        original_start = self.start_date
        original_end = self.end_date

        self.start_date = start_date
        self.end_date = end_date

        try:
            if symbols is None:
                symbols = self.get_symbol_list()

            self.logger.info(
                f"증분 업데이트 시작: {len(symbols)}개 심볼, {days_back}일간"
            )

            updated_data = self.collect_data(symbols)

            self.logger.info("증분 업데이트 완료")
            return updated_data

        finally:
            # 원래 설정 복원
            self.start_date = original_start
            self.end_date = original_end
