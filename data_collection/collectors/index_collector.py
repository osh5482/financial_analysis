"""
지수 데이터 수집기
전세계 주요 거래소의 대표 지수 데이터를 비동기로 수집
"""

import asyncio
import re
from datetime import datetime
import pandas as pd
import FinanceDataReader as fdr
from loguru import logger

from .base_collector import BaseCollector


class IndexCollector(BaseCollector):
    """
    지수 데이터 수집기
    전세계 주요 거래소의 대표 지수들을 수집
    """

    def __init__(self, rate_limit: float = 0.1):
        """
        지수 수집기 초기화

        Args:
            rate_limit: API 호출 간 최소 대기 시간 (초)
        """
        super().__init__("Index Collector", rate_limit)

        # 거래소별 주요 지수 정의
        self.index_definitions = {
            # 한국 거래소 지수
            "KRX": {
                "KOSPI": {"symbol": "KS11", "name": "코스피 지수"},
                "KOSPI200": {"symbol": "KS200", "name": "코스피 200"},
                "KOSDAQ": {"symbol": "KQ11", "name": "코스닥 지수"},
                "KOSPI_VALUE": {"symbol": "KS11V", "name": "코스피 밸류"},
                "KOSPI_GROWTH": {"symbol": "KS11G", "name": "코스피 성장"},
                "KOSDAQ_STAR": {"symbol": "KQS", "name": "코스닥 스타"},
            },
            # 미국 거래소 지수
            "US": {
                "SPX": {"symbol": "^GSPC", "name": "S&P 500"},
                "DJI": {"symbol": "^DJI", "name": "다우존스 산업평균"},
                "NASDAQ": {"symbol": "^IXIC", "name": "나스닥 종합지수"},
                "NASDAQ100": {"symbol": "^NDX", "name": "나스닥 100"},
                "RUSSELL2000": {"symbol": "^RUT", "name": "러셀 2000"},
                "VIX": {"symbol": "^VIX", "name": "변동성 지수"},
                "SPX_GROWTH": {"symbol": "^SP500-45", "name": "S&P 500 성장주"},
                "SPX_VALUE": {"symbol": "^SP500-40", "name": "S&P 500 가치주"},
            },
            # 일본 거래소 지수
            "JAPAN": {
                "NIKKEI225": {"symbol": "^N225", "name": "니케이 225"},
                "TOPIX": {"symbol": "^TPX", "name": "도쿄증권거래소 지수"},
                "NIKKEI300": {"symbol": "^N300", "name": "니케이 300"},
                "MOTHERS": {"symbol": "^MOTHERS", "name": "마더스 지수"},
                "JASDAQ": {"symbol": "^JASDAQ", "name": "JASDAQ 지수"},
            },
            # 중국 거래소 지수
            "CHINA": {
                "SSE_COMPOSITE": {"symbol": "000001.SS", "name": "상하이 종합지수"},
                "SZSE_COMPOSITE": {"symbol": "399001.SZ", "name": "선전 종합지수"},
                "CSI300": {"symbol": "000300.SS", "name": "CSI 300"},
                "CSI500": {"symbol": "000905.SS", "name": "CSI 500"},
                "CHINEXT": {"symbol": "399006.SZ", "name": "창업판 지수"},
                "HANG_SENG": {"symbol": "^HSI", "name": "항셍지수"},
            },
            # 유럽 거래소 지수
            "EUROPE": {
                "FTSE100": {"symbol": "^FTSE", "name": "FTSE 100"},
                "DAX": {"symbol": "^GDAXI", "name": "DAX 30"},
                "CAC40": {"symbol": "^FCHI", "name": "CAC 40"},
                "EUROSTOXX50": {"symbol": "^STOXX50E", "name": "유로스톡스 50"},
                "FTSE250": {"symbol": "^FTMC", "name": "FTSE 250"},
                "IBEX35": {"symbol": "^IBEX", "name": "IBEX 35"},
                "AEX": {"symbol": "^AEX", "name": "AEX 지수"},
            },
            # 아시아-태평양 지수
            "ASIA_PACIFIC": {
                "ASX200": {"symbol": "^AXJO", "name": "ASX 200"},
                "ASX_ALL_ORDS": {"symbol": "^AORD", "name": "All Ordinaries"},
                "SENSEX": {"symbol": "^BSESN", "name": "봄베이 센섹스"},
                "NIFTY50": {"symbol": "^NSEI", "name": "니프티 50"},
                "STRAIT_TIMES": {"symbol": "^STI", "name": "스트레이츠 타임즈"},
                "KLCI": {"symbol": "^KLSE", "name": "쿠알라룸푸르 종합지수"},
                "SET": {"symbol": "^SET.BK", "name": "태국 SET 지수"},
            },
            # 신흥시장 및 기타
            "EMERGING": {
                "BOVESPA": {"symbol": "^BVSP", "name": "브라질 보베스파"},
                "MERVAL": {"symbol": "^MERV", "name": "아르헨티나 메르발"},
                "MICEX": {"symbol": "IMOEX.ME", "name": "러시아 MOEX"},
                "TAIEX": {"symbol": "^TWII", "name": "대만 가권지수"},
                "JSE": {"symbol": "J203.JO", "name": "남아공 JSE"},
                "EGX30": {"symbol": "^CASE30", "name": "이집트 EGX 30"},
            },
        }

        # 지수 카테고리별 데이터 소스 매핑
        self.category_sources = {
            "KRX": "krx",
            "US": "yahoo",
            "JAPAN": "yahoo",
            "CHINA": "yahoo",
            "EUROPE": "yahoo",
            "ASIA_PACIFIC": "yahoo",
            "EMERGING": "yahoo",
        }

        # 전체 지수 심볼 매핑 생성
        self.all_indices = {}
        for category, indices in self.index_definitions.items():
            for index_key, index_info in indices.items():
                self.all_indices[index_info["symbol"]] = {
                    "name": index_info["name"],
                    "category": category,
                    "key": index_key,
                }

        logger.info(f"지수 수집기 초기화 완료 - 총 {len(self.all_indices)}개 지수 지원")
        logger.info(f"지원 카테고리: {list(self.index_definitions.keys())}")

    async def _fetch_data(
        self, symbol: str, start_date: str, end_date: str
    ) -> pd.DataFrame | None:
        """
        개별 지수의 데이터를 비동기로 수집

        Args:
            symbol: 지수 심볼 (예: 'KS11', '^GSPC', '^N225')
            start_date: 시작 날짜 (YYYY-MM-DD)
            end_date: 종료 날짜 (YYYY-MM-DD)

        Returns:
            DataFrame: 수집된 지수 데이터 또는 None
        """
        try:
            # 심볼의 카테고리 및 정보 확인
            index_info = self.all_indices.get(symbol)
            if not index_info:
                logger.warning(f"{symbol}: 지원하지 않는 지수 심볼")
                return None

            category = index_info["category"]
            data_source = self.category_sources[category]

            logger.debug(
                f"{symbol} 지수 데이터 수집 시작 (카테고리: {category}, 소스: {data_source})"
            )

            # 비동기 실행을 위해 별도 스레드에서 실행
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(
                None,
                self._fetch_index_data_sync,
                symbol,
                start_date,
                end_date,
                data_source,
                category,
            )

            if data is not None and not data.empty:
                # 지수 정보 추가
                data = data.copy()
                data["index_name"] = index_info["name"]
                data["category"] = category
                data["index_key"] = index_info["key"]
                data["data_source"] = data_source

                logger.debug(
                    f"{symbol} 수집 완료: {len(data)}행 ({index_info['name']})"
                )
                return data
            else:
                logger.warning(f"{symbol}: 데이터 없음 ({index_info['name']})")
                return None

        except Exception as e:
            logger.error(f"{symbol} 수집 실패: {str(e)}")
            raise e

    def _fetch_index_data_sync(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        data_source: str,
        category: str,
    ) -> pd.DataFrame | None:
        """
        동기 방식으로 지수 데이터 수집 (executor에서 실행됨)

        Args:
            symbol: 지수 심볼
            start_date: 시작 날짜
            end_date: 종료 날짜
            data_source: 데이터 소스
            category: 지수 카테고리

        Returns:
            DataFrame: 수집된 데이터 또는 None
        """
        try:
            if data_source == "krx":
                # 한국 지수의 경우 KRX 데이터 사용
                data = fdr.DataReader(symbol, start_date, end_date)
            else:
                # 해외 지수의 경우 새로운 형식 사용
                if category == "US":
                    data = fdr.DataReader(symbol, start_date, end_date)
                elif category == "JAPAN":
                    data = fdr.DataReader(symbol, start_date, end_date)
                elif category == "CHINA":
                    if symbol.endswith(".SS"):
                        data = fdr.DataReader(
                            f"SSE:{symbol[:-3]}", start_date, end_date
                        )
                    elif symbol.endswith(".SZ"):
                        data = fdr.DataReader(
                            f"SZSE:{symbol[:-3]}", start_date, end_date
                        )
                    else:
                        data = fdr.DataReader(symbol, start_date, end_date)
                else:
                    # 기타 해외 지수
                    data = fdr.DataReader(symbol, start_date, end_date)

            # 데이터 유효성 검증
            if data is None or data.empty:
                return None

            # 지수 데이터 정제
            data = self._clean_index_data(data, symbol, category)

            return data

        except Exception as e:
            logger.debug(f"{symbol} 동기 수집 중 오류: {str(e)}")
            return None

    def _clean_index_data(
        self, data: pd.DataFrame, symbol: str, category: str
    ) -> pd.DataFrame:
        """
        지수 데이터 정제 및 표준화

        Args:
            data: 원시 지수 데이터
            symbol: 지수 심볼
            category: 지수 카테고리

        Returns:
            DataFrame: 정제된 데이터
        """
        cleaned_data = data.copy()

        # 지수는 보통 거래량이 없거나 의미가 다르므로 별도 처리
        if "Volume" in cleaned_data.columns:
            # VIX 같은 특수 지수는 거래량이 의미있을 수 있음
            if symbol == "^VIX":
                # VIX는 거래량 대신 변동성 수치로 해석
                cleaned_data["volatility_volume"] = cleaned_data["Volume"]
            else:
                # 일반 지수는 거래량을 제거하거나 별도 컬럼으로 처리
                cleaned_data["index_volume"] = cleaned_data["Volume"]

        # 지수 특성에 맞는 추가 계산
        if "Close" in cleaned_data.columns:
            # 지수 일일 변화율
            cleaned_data["daily_return"] = cleaned_data["Close"].pct_change()
            cleaned_data["daily_return_pct"] = cleaned_data["daily_return"] * 100

            # 지수 이동평균 (5일, 20일, 60일, 120일)
            for window in [5, 20, 60, 120]:
                cleaned_data[f"ma{window}"] = (
                    cleaned_data["Close"].rolling(window=window).mean()
                )

            # 지수 변동성 (20일 롤링)
            cleaned_data["volatility_20d"] = (
                cleaned_data["daily_return"].rolling(window=20).std() * (252**0.5) * 100
            )

            # 최고점/최저점 대비 현재 위치 (52주 기준)
            if len(cleaned_data) >= 252:
                cleaned_data["high_52w"] = (
                    cleaned_data["Close"].rolling(window=252).max()
                )
                cleaned_data["low_52w"] = (
                    cleaned_data["Close"].rolling(window=252).min()
                )
                cleaned_data["position_52w"] = (
                    (cleaned_data["Close"] - cleaned_data["low_52w"])
                    / (cleaned_data["high_52w"] - cleaned_data["low_52w"])
                    * 100
                )

        # 카테고리별 특수 처리
        if category == "KRX":
            # 한국 지수의 경우 원화 단위 명시
            cleaned_data["currency"] = "KRW"
        elif category == "US":
            cleaned_data["currency"] = "USD"
            # VIX의 경우 특별한 해석 추가
            if symbol == "^VIX":
                if "Close" in cleaned_data.columns:
                    # VIX 수준에 따른 시장 심리 분류
                    cleaned_data["market_sentiment"] = cleaned_data["Close"].apply(
                        self._classify_vix_level
                    )
        elif category == "JAPAN":
            cleaned_data["currency"] = "JPY"
        elif category == "CHINA":
            cleaned_data["currency"] = "CNY"
        elif category == "EUROPE":
            cleaned_data["currency"] = "EUR"
        else:
            cleaned_data["currency"] = "USD"  # 기본값

        # 음수나 0 값 검증 (지수는 일반적으로 양수여야 함)
        if "Close" in cleaned_data.columns:
            invalid_values = cleaned_data["Close"] <= 0
            if invalid_values.any():
                logger.warning(
                    f"{symbol}: {invalid_values.sum()}개 행에서 비정상적인 지수 값 발견"
                )
                # 이전 값으로 forward fill
                cleaned_data.loc[invalid_values, "Close"] = None
                cleaned_data["Close"] = cleaned_data["Close"].fillna(method="ffill")

        return cleaned_data

    def _classify_vix_level(self, vix_value: float) -> str:
        """
        VIX 수준에 따른 시장 심리 분류

        Args:
            vix_value: VIX 지수 값

        Returns:
            str: 시장 심리 분류
        """
        if vix_value < 20:
            return "Low_Volatility"
        elif vix_value < 30:
            return "Normal_Volatility"
        elif vix_value < 40:
            return "High_Volatility"
        else:
            return "Extreme_Volatility"

    async def _is_valid_symbol(self, symbol: str) -> bool:
        """
        지수 심볼의 유효성 검증

        Args:
            symbol: 검증할 심볼

        Returns:
            bool: 유효성 여부
        """
        # 기본 검증
        if not await super()._is_valid_symbol(symbol):
            return False

        # 지원하는 지수 목록에 있는지 확인
        if symbol not in self.all_indices:
            logger.debug(f"{symbol}: 지원하지 않는 지수 심볼")
            return False

        return True

    async def collect_by_category(
        self, category: str, start_date: str | None = None, end_date: str | None = None
    ) -> dict[str, pd.DataFrame]:
        """
        특정 카테고리의 모든 지수 데이터 수집

        Args:
            category: 지수 카테고리 ('KRX', 'US', 'JAPAN' 등)
            start_date: 시작 날짜
            end_date: 종료 날짜

        Returns:
            dict: 수집된 지수 데이터
        """
        if category not in self.index_definitions:
            logger.error(f"지원하지 않는 카테고리: {category}")
            return {}

        symbols = [info["symbol"] for info in self.index_definitions[category].values()]

        logger.info(f"{category} 카테고리 지수 수집 시작: {len(symbols)}개")

        return await self.collect_data(symbols, start_date, end_date)

    async def collect_major_global_indices(
        self, start_date: str | None = None, end_date: str | None = None
    ) -> dict[str, pd.DataFrame]:
        """
        전세계 주요 지수들을 선별하여 수집

        Args:
            start_date: 시작 날짜
            end_date: 종료 날짜

        Returns:
            dict: 수집된 주요 지수 데이터
        """
        # 각 카테고리별 대표 지수 선정
        major_indices = [
            "KS11",  # 코스피
            "KS200",  # 코스피 200
            "KQ11",  # 코스닥
            "^GSPC",  # S&P 500
            "^DJI",  # 다우존스
            "^IXIC",  # 나스닥
            "^N225",  # 니케이 225
            "^HSI",  # 항셍
            "^FTSE",  # FTSE 100
            "^GDAXI",  # DAX
            "^FCHI",  # CAC 40
            "^AXJO",  # ASX 200
        ]

        logger.info(f"전세계 주요 지수 수집 시작: {len(major_indices)}개")

        return await self.collect_data(major_indices, start_date, end_date)

    def get_available_indices(self, category: str | None = None) -> dict[str, dict]:
        """
        사용 가능한 지수 목록 반환

        Args:
            category: 특정 카테고리만 반환 (None이면 전체)

        Returns:
            dict: 지수 정보
        """
        if category:
            if category in self.index_definitions:
                return self.index_definitions[category]
            else:
                logger.warning(f"존재하지 않는 카테고리: {category}")
                return {}
        else:
            return self.index_definitions

    def get_index_info(self, symbol: str) -> dict | None:
        """
        특정 지수의 상세 정보 반환

        Args:
            symbol: 지수 심볼

        Returns:
            dict: 지수 정보 또는 None
        """
        return self.all_indices.get(symbol)

    def get_supported_categories(self) -> dict[str, str]:
        """
        지원하는 지수 카테고리 목록 반환

        Returns:
            dict: {카테고리: 설명} 형태의 카테고리 정보
        """
        category_descriptions = {
            "KRX": "한국 거래소 지수 (코스피, 코스닥 등)",
            "US": "미국 지수 (S&P 500, 다우존스, 나스닥 등)",
            "JAPAN": "일본 지수 (니케이 225, TOPIX 등)",
            "CHINA": "중국/홍콩 지수 (상하이종합, 항셍 등)",
            "EUROPE": "유럽 지수 (FTSE 100, DAX, CAC 40 등)",
            "ASIA_PACIFIC": "아시아-태평양 지수 (ASX 200, 센섹스 등)",
            "EMERGING": "신흥시장 지수 (보베스파, 메르발 등)",
        }

        return category_descriptions
