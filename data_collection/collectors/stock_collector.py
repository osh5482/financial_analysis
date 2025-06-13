"""
주식 데이터 수집기
한국 주식 및 해외 주식 데이터를 비동기로 수집
"""

import asyncio
import re
import time
from datetime import datetime
import pandas as pd
import FinanceDataReader as fdr
from loguru import logger

from .base_collector import BaseCollector


class StockCollector(BaseCollector):
    """
    주식 데이터 수집기
    한국 주식(KOSPI, KOSDAQ)과 해외 주식(NASDAQ, NYSE, TSE 등)을 지원
    """

    def __init__(self, rate_limit: float = 0.1):
        """
        주식 수집기 초기화

        Args:
            rate_limit: API 호출 간 최소 대기 시간 (초)
        """
        super().__init__("Stock Collector", rate_limit)

        # 지원하는 거래소별 심볼 패턴 정의
        self.exchange_patterns = {
            "KRX": r"^[0-9]{6}$",  # 한국: 6자리 숫자 (예: 005930)
            "NASDAQ": r"^[A-Z]{1,5}$",  # 나스닥: 1-5자리 대문자 (예: AAPL, MSFT)
            "NYSE": r"^[A-Z]{1,4}$",  # 뉴욕증권거래소: 1-4자리 대문자 (예: IBM, KO)
            "TSE": r"^[0-9]{4}\.T$",  # 도쿄증권거래소: 4자리숫자.T (예: 7203.T)
            "SSE": r"^[0-9]{6}\.SS$",  # 상하이증권거래소: 6자리숫자.SS (예: 000001.SS)
            "SZSE": r"^[0-9]{6}\.SZ$",  # 선전증권거래소: 6자리숫자.SZ (예: 000002.SZ)
            "LSE": r"^[A-Z]{2,4}\.L$",  # 런던증권거래소: 2-4자리대문자.L (예: LLOY.L)
            "ASX": r"^[A-Z]{3}\.AX$",  # 호주증권거래소: 3자리대문자.AX (예: CBA.AX)
        }

        # 거래소별 데이터 소스 매핑
        self.exchange_sources = {
            "KRX": "krx",
            "NASDAQ": "yahoo",
            "NYSE": "yahoo",
            "TSE": "yahoo",
            "SSE": "yahoo",
            "SZSE": "yahoo",
            "LSE": "yahoo",
            "ASX": "yahoo",
        }

        logger.info(
            f"주식 수집기 초기화 완료 - 지원 거래소: {list(self.exchange_patterns.keys())}"
        )

    async def _fetch_data(
        self, symbol: str, start_date: str, end_date: str
    ) -> pd.DataFrame | None:
        """
        개별 주식의 데이터를 비동기로 수집

        Args:
            symbol: 주식 심볼 (예: '005930', 'AAPL', '7203.T')
            start_date: 시작 날짜 (YYYY-MM-DD)
            end_date: 종료 날짜 (YYYY-MM-DD)

        Returns:
            DataFrame: 수집된 주식 데이터 또는 None
        """
        try:
            # 심볼의 거래소 식별
            exchange = self._identify_exchange(symbol)
            if not exchange:
                logger.warning(f"{symbol}: 지원하지 않는 심볼 형식")
                return None

            data_source = self.exchange_sources[exchange]

            logger.debug(
                f"{symbol} 데이터 수집 시작 (거래소: {exchange}, 소스: {data_source})"
            )

            # 비동기 실행을 위해 별도 스레드에서 실행
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(
                None,
                self._fetch_stock_data_sync,
                symbol,
                start_date,
                end_date,
                data_source,
            )

            if data is not None and not data.empty:
                # 거래소 정보 추가
                data = data.copy()
                data["exchange"] = exchange
                data["data_source"] = data_source

                logger.debug(f"{symbol} 수집 완료: {len(data)}행 (거래소: {exchange})")
                return data
            else:
                logger.warning(f"{symbol}: 데이터 없음 (거래소: {exchange})")
                return None

        except Exception as e:
            logger.error(f"{symbol} 수집 실패: {str(e)}")
            raise e

    def _fetch_stock_data_sync(
        self, symbol: str, start_date: str, end_date: str, data_source: str
    ) -> pd.DataFrame | None:
        """
        동기 방식으로 주식 데이터 수집 (executor에서 실행됨)

        Args:
            symbol: 주식 심볼
            start_date: 시작 날짜
            end_date: 종료 날짜
            data_source: 데이터 소스 ('krx' 또는 'yahoo')

        Returns:
            DataFrame: 수집된 데이터 또는 None
        """
        try:
            if data_source == "krx":
                # 한국 주식의 경우 KRX 데이터 사용 (새로운 형식)
                data = fdr.DataReader(symbol, start_date, end_date)
            else:
                # 해외 주식의 경우 Yahoo Finance 데이터 사용 (새로운 형식)
                if symbol.endswith(".T"):
                    # 일본 주식
                    data = fdr.DataReader(f"TSE:{symbol[:-2]}", start_date, end_date)
                elif symbol.endswith(".SS"):
                    # 상하이 증권거래소
                    data = fdr.DataReader(f"SSE:{symbol[:-3]}", start_date, end_date)
                elif symbol.endswith(".SZ"):
                    # 선전 증권거래소
                    data = fdr.DataReader(f"SZSE:{symbol[:-3]}", start_date, end_date)
                elif symbol.endswith(".L"):
                    # 런던 증권거래소
                    data = fdr.DataReader(f"LSE:{symbol[:-2]}", start_date, end_date)
                elif symbol.endswith(".AX"):
                    # 호주 증권거래소
                    data = fdr.DataReader(f"ASX:{symbol[:-3]}", start_date, end_date)
                else:
                    # 미국 주식 (NASDAQ, NYSE)
                    try:
                        data = fdr.DataReader(f"NASDAQ:{symbol}", start_date, end_date)
                    except:
                        # NASDAQ에서 실패하면 NYSE 시도
                        data = fdr.DataReader(f"NYSE:{symbol}", start_date, end_date)

            # 데이터 유효성 검증
            if data is None or data.empty:
                return None

            # 기본적인 데이터 정제
            data = self._clean_stock_data(data, symbol)

            return data

        except Exception as e:
            logger.debug(f"{symbol} 동기 수집 중 오류: {str(e)}")
            return None

    def _clean_stock_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        주식 데이터 정제 및 표준화

        Args:
            data: 원시 주식 데이터
            symbol: 주식 심볼

        Returns:
            DataFrame: 정제된 데이터
        """
        cleaned_data = data.copy()

        # 숫자형 컬럼들의 결측치를 0으로 처리하지 않고 forward fill 적용
        numeric_columns = ["Open", "High", "Low", "Close", "Volume"]
        for col in numeric_columns:
            if col in cleaned_data.columns:
                # 극단적인 이상치 제거 (가격이 0이거나 음수인 경우)
                if col in ["Open", "High", "Low", "Close"]:
                    cleaned_data.loc[cleaned_data[col] <= 0, col] = None

                # Volume이 음수인 경우 0으로 처리
                if col == "Volume":
                    cleaned_data.loc[cleaned_data[col] < 0, col] = 0

        # 가격 데이터의 논리적 일관성 검증
        if all(col in cleaned_data.columns for col in ["Open", "High", "Low", "Close"]):
            # High가 Low보다 작은 경우 수정
            invalid_high_low = cleaned_data["High"] < cleaned_data["Low"]
            if invalid_high_low.any():
                logger.warning(
                    f"{symbol}: {invalid_high_low.sum()}개 행에서 High < Low 발견, 수정 중"
                )
                # High와 Low 값을 교체
                cleaned_data.loc[invalid_high_low, ["High", "Low"]] = cleaned_data.loc[
                    invalid_high_low, ["Low", "High"]
                ].values

            # 시가/종가가 고가/저가 범위를 벗어나는 경우 수정
            for price_col in ["Open", "Close"]:
                if price_col in cleaned_data.columns:
                    # 고가보다 높은 경우
                    invalid_high = cleaned_data[price_col] > cleaned_data["High"]
                    if invalid_high.any():
                        logger.warning(
                            f"{symbol}: {invalid_high.sum()}개 행에서 {price_col} > High 발견"
                        )
                        cleaned_data.loc[invalid_high, "High"] = cleaned_data.loc[
                            invalid_high, price_col
                        ]

                    # 저가보다 낮은 경우
                    invalid_low = cleaned_data[price_col] < cleaned_data["Low"]
                    if invalid_low.any():
                        logger.warning(
                            f"{symbol}: {invalid_low.sum()}개 행에서 {price_col} < Low 발견"
                        )
                        cleaned_data.loc[invalid_low, "Low"] = cleaned_data.loc[
                            invalid_low, price_col
                        ]

        # 수익률 계산 추가
        if "Close" in cleaned_data.columns:
            cleaned_data["daily_return"] = cleaned_data["Close"].pct_change()
            cleaned_data["daily_return_pct"] = cleaned_data["daily_return"] * 100

        # 거래량 이동평균 추가 (5일, 20일)
        if "Volume" in cleaned_data.columns:
            cleaned_data["volume_ma5"] = cleaned_data["Volume"].rolling(window=5).mean()
            cleaned_data["volume_ma20"] = (
                cleaned_data["Volume"].rolling(window=20).mean()
            )

        return cleaned_data

    def _identify_exchange(self, symbol: str) -> str | None:
        """
        심볼을 기반으로 거래소 식별

        Args:
            symbol: 주식 심볼

        Returns:
            str: 거래소 코드 또는 None
        """
        symbol = symbol.upper().strip()

        for exchange, pattern in self.exchange_patterns.items():
            if re.match(pattern, symbol):
                return exchange

        return None

    async def _is_valid_symbol(self, symbol: str) -> bool:
        """
        주식 심볼의 유효성 검증

        Args:
            symbol: 검증할 심볼

        Returns:
            bool: 유효성 여부
        """
        # 기본 검증
        if not await super()._is_valid_symbol(symbol):
            return False

        # 거래소 패턴 매칭 검증
        exchange = self._identify_exchange(symbol)
        if not exchange:
            logger.debug(f"{symbol}: 지원하지 않는 심볼 형식")
            return False

        return True

    async def get_available_stocks(self, exchange: str = "KRX") -> pd.DataFrame | None:
        """
        특정 거래소의 상장 종목 리스트 조회

        Args:
            exchange: 거래소 코드 ('KRX', 'NASDAQ', 'NYSE' 등)

        Returns:
            DataFrame: 상장 종목 정보 또는 None
        """
        try:
            logger.info(f"{exchange} 거래소 상장 종목 리스트 조회 시작")

            loop = asyncio.get_event_loop()

            if exchange == "KRX":
                # 한국 거래소 전체 종목
                kospi_stocks = await loop.run_in_executor(
                    None, fdr.StockListing, "KOSPI"
                )
                kosdaq_stocks = await loop.run_in_executor(
                    None, fdr.StockListing, "KOSDAQ"
                )

                # KOSPI와 KOSDAQ 데이터 합치기
                all_stocks = pd.concat([kospi_stocks, kosdaq_stocks], ignore_index=True)
                all_stocks["Exchange"] = all_stocks["Market"].apply(
                    lambda x: "KOSPI" if x == "KOSPI" else "KOSDAQ"
                )

            elif exchange == "NASDAQ":
                all_stocks = await loop.run_in_executor(
                    None, fdr.StockListing, "NASDAQ"
                )
                all_stocks["Exchange"] = "NASDAQ"

            elif exchange == "NYSE":
                all_stocks = await loop.run_in_executor(None, fdr.StockListing, "NYSE")
                all_stocks["Exchange"] = "NYSE"

            else:
                logger.warning(f"{exchange}: 지원하지 않는 거래소")
                return None

            if all_stocks is not None and not all_stocks.empty:
                logger.info(f"{exchange} 종목 조회 완료: {len(all_stocks)}개 종목")
                return all_stocks
            else:
                logger.warning(f"{exchange} 종목 조회 결과 없음")
                return None

        except Exception as e:
            logger.error(f"{exchange} 종목 조회 실패: {str(e)}")
            return None

    async def collect_by_market_cap(
        self,
        exchange: str = "KRX",
        min_market_cap: int = 1000,
        max_count: int = 100,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, pd.DataFrame]:
        """
        시가총액 기준으로 상위 종목들의 데이터 수집

        Args:
            exchange: 거래소 코드
            min_market_cap: 최소 시가총액 (억원)
            max_count: 최대 수집 종목 수
            start_date: 시작 날짜
            end_date: 종료 날짜

        Returns:
            dict: 수집된 주식 데이터
        """
        try:
            # 상장 종목 리스트 조회
            stock_list = await self.get_available_stocks(exchange)
            if stock_list is None or stock_list.empty:
                logger.error(f"{exchange} 종목 리스트 조회 실패")
                return {}

            # 시가총액 기준 필터링 및 정렬
            if "Marcap" in stock_list.columns:
                # 시가총액이 최소 기준 이상인 종목 필터링
                filtered_stocks = stock_list[
                    stock_list["Marcap"] >= min_market_cap * 100000000
                ]  # 억원 → 원
                # 시가총액 기준 내림차순 정렬
                filtered_stocks = filtered_stocks.sort_values("Marcap", ascending=False)
            else:
                # 시가총액 정보가 없으면 전체 리스트 사용
                filtered_stocks = stock_list
                logger.warning(f"{exchange}: 시가총액 정보 없음, 전체 종목 대상")

            # 상위 종목 선택
            top_stocks = filtered_stocks.head(max_count)
            symbols = (
                top_stocks["Code"].tolist()
                if "Code" in top_stocks.columns
                else top_stocks["Symbol"].tolist()
            )

            logger.info(
                f"{exchange} 시가총액 상위 {len(symbols)}개 종목 데이터 수집 시작"
            )

            # 데이터 수집
            return await self.collect_data(symbols, start_date, end_date)

        except Exception as e:
            logger.error(f"시가총액 기준 수집 실패: {str(e)}")
            return {}

    def get_supported_exchanges(self) -> dict[str, str]:
        """
        지원하는 거래소 목록 반환

        Returns:
            dict: {거래소코드: 설명} 형태의 거래소 정보
        """
        exchange_descriptions = {
            "KRX": "한국거래소 (KOSPI, KOSDAQ)",
            "NASDAQ": "나스닥",
            "NYSE": "뉴욕증권거래소",
            "TSE": "도쿄증권거래소",
            "SSE": "상하이증권거래소",
            "SZSE": "선전증권거래소",
            "LSE": "런던증권거래소",
            "ASX": "호주증권거래소",
        }

        return exchange_descriptions
