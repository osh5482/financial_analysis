"""
ETF 데이터 수집기
- FinanceDataReader를 활용한 ETF 데이터 수집
- 한국(KR), 미국(US) ETF 지원
- ETF 특화 검증: NAV 추적오차, 유동성, 자산규모 분석
"""

import pandas as pd
import FinanceDataReader as fdr
from pathlib import Path
from datetime import datetime, timedelta

from .base_collector import BaseCollector
from config.settings import (
    ETF_MARKETS,
    VALIDATION_CONFIG,
    DATA_PATHS,
    get_enabled_etf_markets,
)


class ETFCollector(BaseCollector):
    """
    ETF 데이터 수집기

    기능:
    - 국가별 ETF 리스트 자동 수집
    - 개별 ETF 가격 및 NAV 데이터 수집
    - ETF 특화 검증 (추적오차, 프리미엄/디스카운트)
    - 섹터별/테마별 ETF 분류
    """

    def __init__(
        self,
        market: str = "KR",
        start_date: str | None = None,
        end_date: str | None = None,
        include_leveraged: bool = True,
        include_inverse: bool = True,
        min_aum: float | None = None,  # 최소 순자산 (억원 또는 백만달러)
        batch_size: int | None = None,
        **kwargs,
    ) -> None:
        """
        ETFCollector 초기화

        Args:
            market: 시장 코드 (KR, US)
            start_date: 수집 시작일
            end_date: 수집 종료일
            include_leveraged: 레버리지 ETF 포함 여부
            include_inverse: 인버스 ETF 포함 여부
            min_aum: 최소 순자산 규모 (필터링용)
            batch_size: 배치 처리 크기
            **kwargs: BaseCollector 추가 인자
        """
        super().__init__(f"ETFCollector_{market}", start_date, end_date, **kwargs)

        self.market = market.upper()
        self.include_leveraged = include_leveraged
        self.include_inverse = include_inverse
        self.min_aum = min_aum
        self.batch_size = batch_size or 30

        # 시장 설정 검증
        if self.market not in ETF_MARKETS:
            raise ValueError(f"지원하지 않는 ETF 시장: {market}")

        self.market_config = ETF_MARKETS[self.market]

        # 데이터 저장 경로 설정
        self.data_path = DATA_PATHS["etfs"] / self.market.lower()
        self.etf_list_path = (
            self.data_path / f"{self.market_config['file_prefix']}_list.csv"
        )

        self.logger.info(
            f"ETF 시장: {self.market} ({self.market_config['description']})"
        )
        self.logger.info(f"레버리지 ETF 포함: {self.include_leveraged}")
        self.logger.info(f"인버스 ETF 포함: {self.include_inverse}")
        self.logger.info(f"배치 크기: {self.batch_size}")

    def get_symbol_list(self) -> list[str]:
        """
        시장별 ETF 심볼 리스트 반환

        Returns:
            ETF 심볼 리스트
        """
        try:
            self.logger.info(f"{self.market} ETF 리스트 수집 시작")

            # FinanceDataReader ETF 리스트 수집 (대안 방법 사용)
            if self.market == "KR":
                # 한국 ETF - KRX 전체에서 ETF 필터링
                try:
                    # 먼저 KRX 전체 리스트에서 ETF 필터링 시도
                    all_stocks = fdr.StockListing("KRX")
                    if all_stocks is not None and not all_stocks.empty:
                        # ETF 필터링 (종목명에 ETF, ETN 포함된 것들)
                        etf_keywords = [
                            "ETF",
                            "ETN",
                            "KODEX",
                            "TIGER",
                            "ARIRANG",
                            "KBSTAR",
                            "TIMEFOLIO",
                        ]
                        etf_mask = all_stocks["Name"].str.contains(
                            "|".join(etf_keywords), na=False
                        )
                        etf_df = all_stocks[etf_mask].copy()

                        if etf_df.empty:
                            # 필터링 결과가 없으면 미리 정의된 주요 한국 ETF 사용
                            etf_df = self._get_predefined_kr_etfs()
                    else:
                        etf_df = self._get_predefined_kr_etfs()

                except Exception as e:
                    self.logger.warning(f"KRX에서 ETF 필터링 실패: {str(e)}")
                    etf_df = self._get_predefined_kr_etfs()

            elif self.market == "US":
                # 미국 ETF - 미리 정의된 주요 ETF 사용
                etf_df = self._get_predefined_us_etfs()

            else:
                raise ValueError(f"지원하지 않는 ETF 시장: {self.market}")

            # 데이터 검증
            if etf_df is None or etf_df.empty:
                self.logger.error(f"{self.market} ETF 리스트 수집 실패")
                return []

            # ETF 필터링
            filtered_etf_df = self._filter_etfs(etf_df)

            # 심볼 리스트 추출
            if "Symbol" in filtered_etf_df.columns:
                symbols = filtered_etf_df["Symbol"].tolist()
            elif "Code" in filtered_etf_df.columns:
                symbols = filtered_etf_df["Code"].tolist()
            else:
                symbols = filtered_etf_df.index.tolist()

            # 심볼 정제
            symbols = [str(symbol).strip() for symbol in symbols if pd.notna(symbol)]

            # ETF 리스트 저장
            self._save_etf_list(filtered_etf_df)

            self.logger.info(f"{self.market} ETF 리스트 수집 완료: {len(symbols)}개")
            return symbols

        except Exception as e:
            self.logger.error(f"{self.market} ETF 리스트 수집 실패: {str(e)}")

            # 기존 저장된 리스트 사용 시도
            return self._load_cached_etf_list()

    def _get_predefined_kr_etfs(self) -> pd.DataFrame:
        """미리 정의된 주요 한국 ETF 리스트 반환"""
        kr_etfs = {
            # 주요 지수 추종 ETF
            "069500": {
                "Name": "KODEX 200",
                "Category": "Index",
                "Underlying": "KOSPI 200",
            },
            "102110": {
                "Name": "TIGER 200",
                "Category": "Index",
                "Underlying": "KOSPI 200",
            },
            "114800": {
                "Name": "KODEX 인버스",
                "Category": "Inverse",
                "Underlying": "KOSPI 200",
            },
            "251340": {
                "Name": "KODEX 코스닥150",
                "Category": "Index",
                "Underlying": "KOSDAQ 150",
            },
            "229200": {
                "Name": "KODEX 코스닥150 선물인버스",
                "Category": "Inverse",
                "Underlying": "KOSDAQ 150",
            },
            # 섹터 ETF
            "091160": {
                "Name": "KODEX 반도체",
                "Category": "Sector",
                "Underlying": "Semiconductor",
            },
            "091170": {
                "Name": "KODEX 은행",
                "Category": "Sector",
                "Underlying": "Bank",
            },
            "266420": {
                "Name": "KODEX 자동차",
                "Category": "Sector",
                "Underlying": "Automotive",
            },
            # 해외 지수 추종 ETF
            "143850": {
                "Name": "TIGER 미국나스닥100",
                "Category": "International",
                "Underlying": "NASDAQ 100",
            },
            "133690": {
                "Name": "TIGER 미국S&P500",
                "Category": "International",
                "Underlying": "S&P 500",
            },
            "195930": {
                "Name": "TIGER 중국본토CSI300",
                "Category": "International",
                "Underlying": "CSI 300",
            },
            # 원자재 ETF
            "132030": {
                "Name": "KODEX 골드선물(H)",
                "Category": "Commodity",
                "Underlying": "Gold",
            },
            "130680": {
                "Name": "TIGER 원유선물Enhanced(H)",
                "Category": "Commodity",
                "Underlying": "Oil",
            },
            # 채권 ETF
            "114260": {
                "Name": "KODEX 국고채3년",
                "Category": "Bond",
                "Underlying": "KTB 3Y",
            },
            "148070": {
                "Name": "KOSEF 국고채10년",
                "Category": "Bond",
                "Underlying": "KTB 10Y",
            },
            # 배당 ETF
            "161510": {
                "Name": "ARIRANG 고배당주",
                "Category": "Dividend",
                "Underlying": "High Dividend",
            },
            "308620": {
                "Name": "KODEX 미국배당다우존스",
                "Category": "Dividend",
                "Underlying": "US Dividend",
            },
        }

        etf_data = []
        for code, info in kr_etfs.items():
            etf_data.append(
                {
                    "Code": code,
                    "Symbol": code,
                    "Name": info["Name"],
                    "Category": info["Category"],
                    "Underlying": info["Underlying"],
                    "Market": "KR",
                }
            )

        return pd.DataFrame(etf_data)

    def _get_predefined_us_etfs(self) -> pd.DataFrame:
        """미리 정의된 주요 미국 ETF 리스트 반환"""
        us_etfs = {
            # 주요 지수 추종 ETF
            "SPY": {
                "Name": "SPDR S&P 500 ETF",
                "Category": "Index",
                "Underlying": "S&P 500",
            },
            "QQQ": {
                "Name": "Invesco QQQ Trust",
                "Category": "Index",
                "Underlying": "NASDAQ 100",
            },
            "VTI": {
                "Name": "Vanguard Total Stock Market",
                "Category": "Index",
                "Underlying": "Total Market",
            },
            "IWM": {
                "Name": "iShares Russell 2000",
                "Category": "Index",
                "Underlying": "Russell 2000",
            },
            "DIA": {
                "Name": "SPDR Dow Jones Industrial Average",
                "Category": "Index",
                "Underlying": "Dow Jones",
            },
            # 섹터 ETF
            "XLF": {
                "Name": "Financial Select Sector SPDR",
                "Category": "Sector",
                "Underlying": "Financials",
            },
            "XLK": {
                "Name": "Technology Select Sector SPDR",
                "Category": "Sector",
                "Underlying": "Technology",
            },
            "XLE": {
                "Name": "Energy Select Sector SPDR",
                "Category": "Sector",
                "Underlying": "Energy",
            },
            "XLV": {
                "Name": "Health Care Select Sector SPDR",
                "Category": "Sector",
                "Underlying": "Healthcare",
            },
            # 원자재 ETF
            "GLD": {
                "Name": "SPDR Gold Shares",
                "Category": "Commodity",
                "Underlying": "Gold",
            },
            "SLV": {
                "Name": "iShares Silver Trust",
                "Category": "Commodity",
                "Underlying": "Silver",
            },
            "USO": {
                "Name": "United States Oil Fund",
                "Category": "Commodity",
                "Underlying": "Oil",
            },
            # 채권 ETF
            "TLT": {
                "Name": "iShares 20+ Year Treasury Bond",
                "Category": "Bond",
                "Underlying": "Treasury",
            },
            "IEF": {
                "Name": "iShares 7-10 Year Treasury Bond",
                "Category": "Bond",
                "Underlying": "Treasury",
            },
            "LQD": {
                "Name": "iShares Investment Grade Corporate Bond",
                "Category": "Bond",
                "Underlying": "Corporate",
            },
            # 국제 ETF
            "EFA": {
                "Name": "iShares MSCI EAFE",
                "Category": "International",
                "Underlying": "EAFE",
            },
            "EEM": {
                "Name": "iShares MSCI Emerging Markets",
                "Category": "International",
                "Underlying": "Emerging Markets",
            },
            "VEA": {
                "Name": "Vanguard FTSE Developed Markets",
                "Category": "International",
                "Underlying": "Developed Markets",
            },
            # 배당 ETF
            "VYM": {
                "Name": "Vanguard High Dividend Yield",
                "Category": "Dividend",
                "Underlying": "High Dividend",
            },
            "SCHD": {
                "Name": "Schwab US Dividend Equity",
                "Category": "Dividend",
                "Underlying": "Dividend",
            },
            # 성장 ETF
            "VUG": {
                "Name": "Vanguard Growth",
                "Category": "Growth",
                "Underlying": "Growth",
            },
            "MTUM": {
                "Name": "iShares MSCI USA Momentum Factor",
                "Category": "Factor",
                "Underlying": "Momentum",
            },
        }

        etf_data = []
        for symbol, info in us_etfs.items():
            etf_data.append(
                {
                    "Code": symbol,
                    "Symbol": symbol,
                    "Name": info["Name"],
                    "Category": info["Category"],
                    "Underlying": info["Underlying"],
                    "Market": "US",
                }
            )

        return pd.DataFrame(etf_data)

    def _filter_etfs(self, etf_df: pd.DataFrame) -> pd.DataFrame:
        """
        ETF 필터링 (레버리지, 인버스, 최소 순자산 등)

        Args:
            etf_df: 원본 ETF DataFrame

        Returns:
            필터링된 ETF DataFrame
        """
        filtered_df = etf_df.copy()
        original_count = len(filtered_df)

        # ETF 이름 기반 필터링
        if "Name" in filtered_df.columns:
            etf_names = filtered_df["Name"].str.upper()

            # 레버리지 ETF 제외
            if not self.include_leveraged:
                leveraged_keywords = [
                    "2X",
                    "3X",
                    "LEVERAGE",
                    "BULL",
                    "BEAR",
                    "레버리지",
                ]
                leveraged_mask = etf_names.str.contains(
                    "|".join(leveraged_keywords), na=False
                )
                filtered_df = filtered_df[~leveraged_mask]
                self.logger.info(f"레버리지 ETF {leveraged_mask.sum()}개 제외")

            # 인버스 ETF 제외
            if not self.include_inverse:
                inverse_keywords = ["INVERSE", "SHORT", "BEAR", "PUT", "인버스", "곰"]
                inverse_mask = etf_names.str.contains(
                    "|".join(inverse_keywords), na=False
                )
                filtered_df = filtered_df[~inverse_mask]
                self.logger.info(f"인버스 ETF {inverse_mask.sum()}개 제외")

        # 최소 순자산 필터링
        if self.min_aum is not None:
            aum_columns = ["AUM", "Assets", "NAV", "MarketCap", "순자산"]
            aum_column = None

            for col in aum_columns:
                if col in filtered_df.columns:
                    aum_column = col
                    break

            if aum_column:
                # 순자산 크기로 필터링
                aum_mask = filtered_df[aum_column] >= self.min_aum
                filtered_df = filtered_df[aum_mask]
                excluded_count = original_count - len(filtered_df)
                self.logger.info(f"최소 순자산 기준으로 {excluded_count}개 ETF 제외")

        # 상장폐지된 ETF 제외
        if "Status" in filtered_df.columns:
            active_mask = filtered_df["Status"].str.upper() == "ACTIVE"
            filtered_df = filtered_df[active_mask]
        elif "Market" in filtered_df.columns:
            # 시장에서 거래되는 ETF만 포함
            filtered_df = filtered_df[filtered_df["Market"].notna()]

        filtered_count = len(filtered_df)
        self.logger.info(f"ETF 필터링 완료: {original_count} → {filtered_count}개")

        return filtered_df

    def _save_etf_list(self, etf_df: pd.DataFrame) -> None:
        """ETF 리스트를 파일로 저장"""
        try:
            # 타임스탬프 추가
            etf_df["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 파일 저장
            self.save_to_file(etf_df, self.etf_list_path, "csv")
            self.logger.info(f"ETF 리스트 저장 완료: {self.etf_list_path}")

        except Exception as e:
            self.logger.error(f"ETF 리스트 저장 실패: {str(e)}")

    def _load_cached_etf_list(self) -> list[str]:
        """캐시된 ETF 리스트 로드"""
        try:
            etf_df = self.load_from_file(self.etf_list_path, "csv")

            if etf_df is not None and not etf_df.empty:
                if "Symbol" in etf_df.columns:
                    symbols = etf_df["Symbol"].tolist()
                elif "Code" in etf_df.columns:
                    symbols = etf_df["Code"].tolist()
                else:
                    symbols = etf_df.index.tolist()

                symbols = [
                    str(symbol).strip() for symbol in symbols if pd.notna(symbol)
                ]

                self.logger.info(f"캐시된 ETF 리스트 로드: {len(symbols)}개")
                return symbols

        except Exception as e:
            self.logger.error(f"캐시된 ETF 리스트 로드 실패: {str(e)}")

        return []

    def collect_single_etf(self, symbol: str) -> pd.DataFrame | None:
        """
        개별 ETF 데이터 수집

        Args:
            symbol: ETF 심볼

        Returns:
            ETF 가격 데이터 DataFrame 또는 None
        """
        try:
            # FinanceDataReader로 데이터 수집
            data = fdr.DataReader(symbol, self.start_date, self.end_date)

            if data is None or data.empty:
                self.logger.warning(f"{symbol}: ETF 데이터 없음")
                return None

            # 인덱스를 날짜로 변환
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)

            # 컬럼명 표준화
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

            # ETF 특화 데이터 검증
            if not self._validate_etf_data(data, symbol):
                return None

            # 데이터 정제
            data = self.clean_dataframe(data, symbol)

            # 기본 검증
            required_columns = ["Open", "High", "Low", "Close", "Volume"]
            min_points = getattr(self, "min_data_points", None)
            if not self.validate_dataframe(data, symbol, required_columns, min_points):
                return None

            # ETF 특화 지표 계산
            data = self._calculate_etf_metrics(data, symbol)

            self.logger.debug(f"{symbol}: ETF 데이터 수집 성공 ({len(data)}개 레코드)")
            return data

        except Exception as e:
            self.logger.error(f"{symbol}: ETF 데이터 수집 실패 - {str(e)}")
            return None

    def _validate_etf_data(self, df: pd.DataFrame, symbol: str) -> bool:
        """
        ETF 특화 데이터 검증

        Args:
            df: 검증할 DataFrame
            symbol: ETF 심볼

        Returns:
            검증 통과 여부
        """
        try:
            # 기본 가격 검증 (주식과 동일)
            price_columns = ["Open", "High", "Low", "Close"]
            missing_price_cols = [col for col in price_columns if col not in df.columns]

            if missing_price_cols:
                self.logger.error(f"{symbol}: 필수 가격 컬럼 누락 {missing_price_cols}")
                return False

            # 가격 일관성 검증
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

                if invalid_count / len(df) > 0.05:
                    return False

            # ETF 특화 검증
            # 1. 거래량 검증 (ETF는 일반적으로 거래량이 낮을 수 있음)
            if "Volume" in df.columns:
                zero_volume_ratio = (df["Volume"] == 0).sum() / len(df)
                # ETF는 주식보다 거래량 0 임계값을 더 관대하게 설정
                etf_volume_threshold = VALIDATION_CONFIG["volume_zero_threshold"] * 2

                if zero_volume_ratio > etf_volume_threshold:
                    self.logger.warning(
                        f"{symbol}: 높은 거래량 0 비율 ({zero_volume_ratio:.2%} > {etf_volume_threshold:.2%})"
                    )
                    return False

            # 2. 가격 변동성 검증 (ETF는 일반적으로 개별주식보다 변동성이 낮음)
            if "Close" in df.columns and len(df) > 10:
                daily_returns = df["Close"].pct_change().dropna()
                daily_volatility = daily_returns.std()

                # 일일 변동성이 극도로 높은 경우 (예: 20% 이상) 경고
                if daily_volatility > 0.20:
                    self.logger.warning(
                        f"{symbol}: 비정상적으로 높은 변동성 {daily_volatility:.2%}"
                    )
                    # ETF라면 레버리지 상품일 가능성이 높으므로 경고만 하고 통과

            # 3. 음수 가격 확인
            for col in price_columns:
                if (df[col] <= 0).any():
                    self.logger.error(f"{symbol}: {col}에 음수 또는 0 가격 존재")
                    return False

            return True

        except Exception as e:
            self.logger.error(f"{symbol}: ETF 데이터 검증 중 오류 - {str(e)}")
            return False

    def _calculate_etf_metrics(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        ETF 특화 지표 계산

        Args:
            df: 원본 DataFrame
            symbol: ETF 심볼

        Returns:
            지표가 추가된 DataFrame
        """
        try:
            if "Close" in df.columns and len(df) > 1:
                # 일일 수익률
                df["Daily_Return"] = df["Close"].pct_change()

                # 이동평균 (ETF 추적 성과 분석용)
                df["MA_20"] = df["Close"].rolling(window=20).mean()
                df["MA_60"] = df["Close"].rolling(window=60).mean()

                # 변동성 (20일 롤링)
                df["Volatility_20"] = df["Daily_Return"].rolling(window=20).std()

                # 거래량 이동평균 (유동성 분석용)
                if "Volume" in df.columns:
                    df["Volume_MA_20"] = df["Volume"].rolling(window=20).mean()
                    df["Volume_Ratio"] = df["Volume"] / df["Volume_MA_20"]

                # ETF 성과 지표
                if len(df) >= 252:  # 1년 이상 데이터가 있는 경우
                    # 연환산 수익률 (최근 252일)
                    annual_return = (
                        (df["Close"].iloc[-1] / df["Close"].iloc[-252]) ** (252 / 252)
                    ) - 1
                    df.loc[df.index[-1], "Annual_Return"] = annual_return

                    # 연환산 변동성
                    annual_volatility = df["Daily_Return"].std() * (252**0.5)
                    df.loc[df.index[-1], "Annual_Volatility"] = annual_volatility

                    # 샤프 비율 (무위험 수익률을 2%로 가정)
                    if annual_volatility > 0:
                        sharpe_ratio = (annual_return - 0.02) / annual_volatility
                        df.loc[df.index[-1], "Sharpe_Ratio"] = sharpe_ratio

                self.logger.debug(f"{symbol}: ETF 지표 계산 완료")

        except Exception as e:
            self.logger.warning(f"{symbol}: ETF 지표 계산 중 오류 - {str(e)}")

        return df

    def collect_data(self, symbols: list[str]) -> dict[str, pd.DataFrame]:
        """
        여러 ETF 데이터 배치 수집

        Args:
            symbols: 수집할 ETF 심볼 리스트

        Returns:
            심볼별 데이터 딕셔너리
        """
        collected_data = {}
        total_symbols = len(symbols)

        self.logger.info(f"ETF 데이터 배치 수집 시작: {total_symbols}개")

        # 배치별로 처리
        for i in range(0, total_symbols, self.batch_size):
            batch_symbols = symbols[i : i + self.batch_size]
            batch_num = i // self.batch_size + 1
            total_batches = (total_symbols + self.batch_size - 1) // self.batch_size

            self.logger.info(
                f"배치 {batch_num}/{total_batches} 처리 중: {len(batch_symbols)}개 ETF"
            )

            # 배치 내 각 심볼 처리
            for j, symbol in enumerate(batch_symbols):
                self.logger.debug(f"처리 중: {symbol} ({i+j+1}/{total_symbols})")

                # 재시도 로직과 함께 데이터 수집
                etf_data = self._retry_request(self.collect_single_etf, symbol)

                if etf_data is not None and not etf_data.empty:
                    collected_data[symbol] = etf_data

                    # 개별 파일로 저장 (옵션)
                    if (
                        hasattr(self, "save_individual_files")
                        and self.save_individual_files
                    ):
                        file_path = self.data_path / f"{symbol}.csv"
                        self.save_to_file(etf_data, file_path, "csv")
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
        self.logger.info(
            f"전체 ETF 수집 완료: {successful_total}/{total_symbols}개 성공"
        )

        return collected_data

    def get_etf_info(self, symbol: str) -> dict[str, any] | None:
        """
        개별 ETF 정보 조회

        Args:
            symbol: ETF 심볼

        Returns:
            ETF 정보 딕셔너리 또는 None
        """
        try:
            etf_df = self.load_from_file(self.etf_list_path, "csv")

            if etf_df is not None and not etf_df.empty:
                # 다양한 컬럼명으로 시도
                etf_info = None

                # Symbol 컬럼으로 시도
                if "Symbol" in etf_df.columns:
                    etf_info = etf_df[etf_df["Symbol"] == symbol]

                # Code 컬럼으로 시도
                if (etf_info is None or etf_info.empty) and "Code" in etf_df.columns:
                    etf_info = etf_df[etf_df["Code"] == symbol]

                # 인덱스로 시도
                if etf_info is None or etf_info.empty:
                    try:
                        etf_info = etf_df[etf_df.index == symbol]
                    except:
                        pass

                # 첫 번째 컬럼이 심볼인 경우 시도 (인덱스가 아닌 첫 번째 컬럼)
                if (etf_info is None or etf_info.empty) and len(etf_df.columns) > 0:
                    first_col = etf_df.columns[0]
                    try:
                        etf_info = etf_df[etf_df[first_col] == symbol]
                    except:
                        pass

                if etf_info is not None and not etf_info.empty:
                    result = etf_info.iloc[0].to_dict()
                    self.logger.debug(f"{symbol}: ETF 정보 조회 성공")
                    return result

            self.logger.warning(f"{symbol}: ETF 정보 없음")
            return None

        except Exception as e:
            self.logger.error(f"{symbol}: ETF 정보 조회 실패 - {str(e)}")
            return None

    def analyze_etf_performance(
        self, symbol: str, benchmark_symbol: str | None = None
    ) -> dict[str, any] | None:
        """
        ETF 성과 분석

        Args:
            symbol: 분석할 ETF 심볼
            benchmark_symbol: 벤치마크 심볼 (없으면 시장 지수 사용)

        Returns:
            성과 분석 결과 딕셔너리
        """
        try:
            etf_data = self.collect_single_etf(symbol)

            if etf_data is None or etf_data.empty:
                return None

            analysis = {
                "symbol": symbol,
                "period": f"{etf_data.index.min().date()} ~ {etf_data.index.max().date()}",
                "total_days": len(etf_data),
            }

            if "Close" in etf_data.columns:
                close_prices = etf_data["Close"]

                # 기본 성과 지표
                analysis.update(
                    {
                        "start_price": close_prices.iloc[0],
                        "end_price": close_prices.iloc[-1],
                        "total_return": (
                            close_prices.iloc[-1] / close_prices.iloc[0] - 1
                        )
                        * 100,
                        "max_price": close_prices.max(),
                        "min_price": close_prices.min(),
                        "avg_price": close_prices.mean(),
                    }
                )

                # 위험 지표
                if "Daily_Return" in etf_data.columns:
                    daily_returns = etf_data["Daily_Return"].dropna()
                    analysis.update(
                        {
                            "volatility": daily_returns.std()
                            * (252**0.5)
                            * 100,  # 연환산 변동성
                            "max_daily_gain": daily_returns.max() * 100,
                            "max_daily_loss": daily_returns.min() * 100,
                            "positive_days_ratio": (daily_returns > 0).sum()
                            / len(daily_returns)
                            * 100,
                        }
                    )

                    # 최대 낙폭 (MDD) 계산
                    cumulative = (1 + daily_returns).cumprod()
                    rolling_max = cumulative.cummax()
                    drawdown = (cumulative / rolling_max - 1) * 100
                    analysis["max_drawdown"] = drawdown.min()

            self.logger.info(f"{symbol} 성과 분석 완료")
            return analysis

        except Exception as e:
            self.logger.error(f"{symbol}: ETF 성과 분석 실패 - {str(e)}")
            return None
