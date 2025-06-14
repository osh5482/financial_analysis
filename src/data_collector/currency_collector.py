"""
환율 데이터 수집기
- FinanceDataReader를 활용한 환율 데이터 수집
- 주요 통화쌍 (USD/KRW, EUR/KRW, JPY/KRW 등) 지원
- 환율 특화 검증: 변동성, 스프레드, 시장 시간 분석
"""

import pandas as pd
import FinanceDataReader as fdr
from pathlib import Path
from datetime import datetime, timedelta

from .base_collector import BaseCollector
from config.settings import (
    EXCHANGE_RATES,
    VALIDATION_CONFIG,
    DATA_PATHS,
)


class CurrencyCollector(BaseCollector):
    """
    환율 데이터 수집기

    기능:
    - 주요 통화쌍 환율 데이터 자동 수집
    - 환율 변동성 및 트렌드 분석
    - 중앙은행 정책금리와의 상관관계 분석
    - 해외투자 포트폴리오 환헤지 분석 지원
    """

    def __init__(
        self,
        base_currency: str = "KRW",
        start_date: str | None = None,
        end_date: str | None = None,
        include_minor_pairs: bool = False,
        batch_size: int | None = None,
        **kwargs,
    ) -> None:
        """
        CurrencyCollector 초기화

        Args:
            base_currency: 기준 통화 (KRW, USD 등)
            start_date: 수집 시작일
            end_date: 수집 종료일
            include_minor_pairs: 마이너 통화쌍 포함 여부
            batch_size: 배치 처리 크기
            **kwargs: BaseCollector 추가 인자
        """
        super().__init__(
            f"CurrencyCollector_{base_currency}", start_date, end_date, **kwargs
        )

        self.base_currency = base_currency.upper()
        self.include_minor_pairs = include_minor_pairs
        self.batch_size = batch_size or 10

        # 데이터 저장 경로 설정
        self.data_path = DATA_PATHS["exchange_rates"]
        self.currency_list_path = (
            self.data_path / f"{self.base_currency.lower()}_pairs_list.csv"
        )

        # 수집 대상 통화쌍 설정
        self.target_pairs = self._get_target_currency_pairs()

        self.logger.info(f"기준 통화: {self.base_currency}")
        self.logger.info(f"수집 대상 통화쌍: {len(self.target_pairs)}개")
        self.logger.info(f"마이너 통화쌍 포함: {self.include_minor_pairs}")
        self.logger.info(f"배치 크기: {self.batch_size}")

    def _get_target_currency_pairs(self) -> dict[str, dict]:
        """수집 대상 통화쌍 필터링"""
        target_pairs = {}

        # settings.py의 EXCHANGE_RATES에서 기본 통화쌍 가져오기
        for pair_id, config in EXCHANGE_RATES.items():
            if config.get("enabled", True):
                target_pairs[pair_id] = config

        # 기준 통화별 추가 통화쌍
        additional_pairs = self._get_additional_currency_pairs()
        for pair_id, config in additional_pairs.items():
            if pair_id not in target_pairs:
                target_pairs[pair_id] = config

        return target_pairs

    def _get_additional_currency_pairs(self) -> dict[str, dict]:
        """기준 통화별 추가 통화쌍 정의"""
        additional = {}

        if self.base_currency == "KRW":
            # 한국원 기준 주요 통화쌍
            additional.update(
                {
                    "GBPKRW": {
                        "symbol": "GBPKRW=X",
                        "name": "파운드원 환율",
                        "description": "영국 파운드 대 한국 원 환율",
                        "enabled": self.include_minor_pairs,
                    },
                    "CHFKRW": {
                        "symbol": "CHFKRW=X",
                        "name": "프랑원 환율",
                        "description": "스위스 프랑 대 한국 원 환율",
                        "enabled": self.include_minor_pairs,
                    },
                    "CNYKRW": {
                        "symbol": "CNYKRW=X",
                        "name": "위안원 환율",
                        "description": "중국 위안 대 한국 원 환율",
                        "enabled": self.include_minor_pairs,
                    },
                    "HKDKRW": {
                        "symbol": "HKDKRW=X",
                        "name": "홍콩달러원 환율",
                        "description": "홍콩 달러 대 한국 원 환율",
                        "enabled": self.include_minor_pairs,
                    },
                }
            )

        elif self.base_currency == "USD":
            # 미국달러 기준 주요 통화쌍
            additional.update(
                {
                    "EURUSD": {
                        "symbol": "EURUSD=X",
                        "name": "유로달러 환율",
                        "description": "유로 대 미국 달러 환율",
                        "enabled": True,
                    },
                    "GBPUSD": {
                        "symbol": "GBPUSD=X",
                        "name": "파운드달러 환율",
                        "description": "영국 파운드 대 미국 달러 환율",
                        "enabled": True,
                    },
                    "JPYUSD": {
                        "symbol": "USDJPY=X",
                        "name": "달러엔 환율",
                        "description": "미국 달러 대 일본 엔 환율",
                        "enabled": True,
                    },
                    "AUDUSD": {
                        "symbol": "AUDUSD=X",
                        "name": "호주달러 환율",
                        "description": "호주 달러 대 미국 달러 환율",
                        "enabled": self.include_minor_pairs,
                    },
                    "CADUSD": {
                        "symbol": "USDCAD=X",
                        "name": "달러캐나다달러 환율",
                        "description": "미국 달러 대 캐나다 달러 환율",
                        "enabled": self.include_minor_pairs,
                    },
                }
            )

        return additional

    def get_symbol_list(self) -> list[str]:
        """
        통화쌍 심볼 리스트 반환

        Returns:
            통화쌍 심볼 리스트
        """
        try:
            self.logger.info(f"{self.base_currency} 기준 통화쌍 리스트 수집 시작")

            symbols = []
            currency_data = []

            for pair_id, config in self.target_pairs.items():
                if config.get("enabled", True):
                    symbol = config["symbol"]
                    symbols.append(symbol)

                    # 통화쌍 정보 저장
                    currency_data.append(
                        {
                            "Pair_ID": pair_id,
                            "Symbol": symbol,
                            "Name": config["name"],
                            "Description": config["description"],
                            "Base_Currency": self.base_currency,
                            "Enabled": config.get("enabled", True),
                            "last_updated": datetime.now().strftime(
                                "%Y-%m-%d %H:%M:%S"
                            ),
                        }
                    )

            # 통화쌍 리스트 DataFrame 생성 및 저장
            currencies_df = pd.DataFrame(currency_data)
            self._save_currency_list(currencies_df)

            self.logger.info(f"통화쌍 리스트 수집 완료: {len(symbols)}개")
            return symbols

        except Exception as e:
            self.logger.error(f"통화쌍 리스트 수집 실패: {str(e)}")
            return self._load_cached_currency_list()

    def _save_currency_list(self, currencies_df: pd.DataFrame) -> None:
        """통화쌍 리스트를 파일로 저장"""
        try:
            self.save_to_file(currencies_df, self.currency_list_path, "csv")
            self.logger.info(f"통화쌍 리스트 저장 완료: {self.currency_list_path}")

        except Exception as e:
            self.logger.error(f"통화쌍 리스트 저장 실패: {str(e)}")

    def _load_cached_currency_list(self) -> list[str]:
        """캐시된 통화쌍 리스트 로드"""
        try:
            currencies_df = self.load_from_file(self.currency_list_path, "csv")

            if currencies_df is not None and not currencies_df.empty:
                symbols = currencies_df["Symbol"].tolist()
                symbols = [
                    str(symbol).strip() for symbol in symbols if pd.notna(symbol)
                ]

                self.logger.info(f"캐시된 통화쌍 리스트 로드: {len(symbols)}개")
                return symbols

        except Exception as e:
            self.logger.error(f"캐시된 통화쌍 리스트 로드 실패: {str(e)}")

        return []

    def collect_single_currency(self, symbol: str) -> pd.DataFrame | None:
        """
        개별 통화쌍 데이터 수집

        Args:
            symbol: 통화쌍 심볼

        Returns:
            환율 데이터 DataFrame 또는 None
        """
        try:
            # FinanceDataReader로 데이터 수집
            data = fdr.DataReader(symbol, self.start_date, self.end_date)

            if data is None or data.empty:
                self.logger.warning(f"{symbol}: 환율 데이터 없음")
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

            # 환율 특화 데이터 검증
            if not self._validate_currency_data(data, symbol):
                return None

            # 데이터 정제
            data = self.clean_dataframe(data, symbol)

            # 기본 검증 (환율은 Close만 필수)
            required_columns = ["Close"]
            min_points = getattr(self, "min_data_points", None)
            if not self.validate_dataframe(data, symbol, required_columns, min_points):
                return None

            # 환율 특화 지표 계산
            data = self._calculate_currency_metrics(data, symbol)

            self.logger.debug(f"{symbol}: 환율 데이터 수집 성공 ({len(data)}개 레코드)")
            return data

        except Exception as e:
            self.logger.error(f"{symbol}: 환율 데이터 수집 실패 - {str(e)}")
            return None

    def _validate_currency_data(self, df: pd.DataFrame, symbol: str) -> bool:
        """
        환율 특화 데이터 검증

        Args:
            df: 검증할 DataFrame
            symbol: 통화쌍 심볼

        Returns:
            검증 통과 여부
        """
        try:
            # Close 환율 필수 확인
            if "Close" not in df.columns:
                self.logger.error(f"{symbol}: Close 컬럼 누락")
                return False

            # 환율 양수 검증 (음수 환율 불가)
            if (df["Close"] <= 0).any():
                self.logger.error(f"{symbol}: 음수 또는 0 환율 존재")
                return False

            # OHLC가 있는 경우 일관성 검증
            if all(col in df.columns for col in ["Open", "High", "Low", "Close"]):
                invalid_rates = (
                    (df["High"] < df["Low"])
                    | (df["High"] < df["Open"])
                    | (df["High"] < df["Close"])
                    | (df["Low"] > df["Open"])
                    | (df["Low"] > df["Close"])
                )

                if invalid_rates.any():
                    invalid_count = invalid_rates.sum()
                    self.logger.warning(f"{symbol}: 환율 일관성 오류 {invalid_count}건")

                    if invalid_count / len(df) > 0.05:
                        return False

            # 환율 변동성 검증 (극단적 변동 체크)
            if "Close" in df.columns and len(df) > 1:
                daily_changes = df["Close"].pct_change().dropna()

                # 일일 변동이 20% 이상인 경우 (환율로는 매우 이례적)
                extreme_changes = (abs(daily_changes) > 0.2).sum()
                if extreme_changes > 0:
                    self.logger.warning(
                        f"{symbol}: 극단적 환율 변동 {extreme_changes}건"
                    )

                    # 신흥국 통화나 특수한 경우 제외하고 경고
                    if extreme_changes > 2:
                        self.logger.warning(f"{symbol}: 환율 데이터 품질 의심")

            # 환율 갭 검증 (주말/공휴일 갭 체크)
            if "Close" in df.columns and len(df) > 1:
                # 전일 대비 갭이 10% 이상인 경우
                daily_gaps = abs(df["Close"].pct_change())
                large_gaps = (daily_gaps > 0.1).sum()

                if large_gaps > 3:  # 연간 3회 이상은 이상
                    self.logger.warning(f"{symbol}: 큰 환율 갭 {large_gaps}건")

            # 환율 레벨 합리성 검증 (상식적 범위)
            if "Close" in df.columns:
                avg_rate = df["Close"].mean()

                # USD/KRW의 경우 상식적 범위 체크
                if "USD" in symbol and "KRW" in symbol:
                    if avg_rate < 500 or avg_rate > 2000:  # 500원~2000원 범위
                        self.logger.warning(
                            f"{symbol}: 비정상적 환율 레벨 {avg_rate:.2f}"
                        )

                # EUR/USD의 경우
                elif "EUR" in symbol and "USD" in symbol:
                    if avg_rate < 0.5 or avg_rate > 2.0:  # 0.5~2.0 범위
                        self.logger.warning(
                            f"{symbol}: 비정상적 환율 레벨 {avg_rate:.4f}"
                        )

            return True

        except Exception as e:
            self.logger.error(f"{symbol}: 환율 데이터 검증 중 오류 - {str(e)}")
            return False

    def _calculate_currency_metrics(
        self, df: pd.DataFrame, symbol: str
    ) -> pd.DataFrame:
        """
        환율 특화 지표 계산

        Args:
            df: 원본 DataFrame
            symbol: 통화쌍 심볼

        Returns:
            지표가 추가된 DataFrame
        """
        try:
            if "Close" in df.columns and len(df) > 1:
                # 일일 변화율 (환율 수익률)
                df["Daily_Return"] = df["Close"].pct_change()

                # 일일 변화량 (절대값)
                df["Daily_Change"] = df["Close"].diff()

                # 이동평균 (환율 트렌드 분석용)
                df["MA_5"] = df["Close"].rolling(window=5).mean()
                df["MA_20"] = df["Close"].rolling(window=20).mean()
                df["MA_60"] = df["Close"].rolling(window=60).mean()

                # 환율 변동성 지표
                df["Volatility_5"] = df["Daily_Return"].rolling(window=5).std()
                df["Volatility_20"] = df["Daily_Return"].rolling(window=20).std()
                df["Volatility_60"] = df["Daily_Return"].rolling(window=60).std()

                # 볼린저 밴드 (20일 기준)
                if len(df) >= 20:
                    bb_std = df["Close"].rolling(window=20).std()
                    df["BB_Upper"] = df["MA_20"] + (bb_std * 2)
                    df["BB_Lower"] = df["MA_20"] - (bb_std * 2)
                    df["BB_Position"] = (df["Close"] - df["BB_Lower"]) / (
                        df["BB_Upper"] - df["BB_Lower"]
                    )

                # RSI (14일 기준)
                if len(df) >= 14:
                    delta = df["Close"].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    df["RSI"] = 100 - (100 / (1 + rs))

                # 환율 강세/약세 지표
                if "MA_20" in df.columns and "MA_60" in df.columns:
                    df["Currency_Strength"] = (
                        df["Close"] / df["MA_60"] - 1
                    ) * 100  # 60일 평균 대비 %
                    df["Trend_Signal"] = (df["MA_20"] > df["MA_60"]).astype(
                        int
                    )  # 상승 추세 신호

                # 변동성 확장/축소 지표
                if "Volatility_20" in df.columns and "Volatility_60" in df.columns:
                    df["Vol_Ratio"] = (
                        df["Volatility_20"] / df["Volatility_60"]
                    )  # 단기/장기 변동성 비율

                # 연환산 변동성 (환헤지 전략용)
                if "Daily_Return" in df.columns:
                    df["Annual_Volatility"] = df["Daily_Return"].rolling(
                        window=252
                    ).std() * (252**0.5)

                # 환율 성과 지표 (연간화)
                if len(df) >= 252:  # 1년 이상 데이터
                    # 연환산 수익률
                    annual_return = (
                        (df["Close"].iloc[-1] / df["Close"].iloc[-252]) ** (252 / 252)
                    ) - 1
                    df.loc[df.index[-1], "Annual_Return"] = annual_return

                    # 샤프 비율 (환율은 무위험 수익률 0% 가정)
                    annual_vol = df["Daily_Return"].std() * (252**0.5)
                    if annual_vol > 0:
                        sharpe_ratio = annual_return / annual_vol
                        df.loc[df.index[-1], "Sharpe_Ratio"] = sharpe_ratio

                # 환율 압력 지표 (고점/저점 대비)
                rolling_max = df["Close"].rolling(window=252).max()  # 1년 최고점
                rolling_min = df["Close"].rolling(window=252).min()  # 1년 최저점
                df["Position_in_Range"] = (df["Close"] - rolling_min) / (
                    rolling_max - rolling_min
                )

                self.logger.debug(f"{symbol}: 환율 지표 계산 완료")

        except Exception as e:
            self.logger.warning(f"{symbol}: 환율 지표 계산 중 오류 - {str(e)}")

        return df

    def collect_data(self, symbols: list[str]) -> dict[str, pd.DataFrame]:
        """
        여러 통화쌍 데이터 배치 수집

        Args:
            symbols: 수집할 통화쌍 심볼 리스트

        Returns:
            심볼별 데이터 딕셔너리
        """
        collected_data = {}
        total_symbols = len(symbols)

        self.logger.info(f"환율 데이터 배치 수집 시작: {total_symbols}개")

        # 배치별로 처리
        for i in range(0, total_symbols, self.batch_size):
            batch_symbols = symbols[i : i + self.batch_size]
            batch_num = i // self.batch_size + 1
            total_batches = (total_symbols + self.batch_size - 1) // self.batch_size

            self.logger.info(
                f"배치 {batch_num}/{total_batches} 처리 중: {len(batch_symbols)}개 통화쌍"
            )

            # 배치 내 각 심볼 처리
            for j, symbol in enumerate(batch_symbols):
                self.logger.debug(f"처리 중: {symbol} ({i+j+1}/{total_symbols})")

                # 재시도 로직과 함께 데이터 수집
                currency_data = self._retry_request(
                    self.collect_single_currency, symbol
                )

                if currency_data is not None and not currency_data.empty:
                    collected_data[symbol] = currency_data

                    # 개별 파일로 저장 (옵션)
                    if (
                        hasattr(self, "save_individual_files")
                        and self.save_individual_files
                    ):
                        file_path = (
                            self.data_path
                            / f"{symbol.replace('/', '_').replace('=X', '')}.csv"
                        )
                        self.save_to_file(currency_data, file_path, "csv")
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
            f"전체 환율 수집 완료: {successful_total}/{total_symbols}개 성공"
        )

        return collected_data

    def get_currency_info(self, symbol: str) -> dict[str, any] | None:
        """
        개별 통화쌍 정보 조회

        Args:
            symbol: 통화쌍 심볼

        Returns:
            통화쌍 정보 딕셔너리 또는 None
        """
        try:
            currencies_df = self.load_from_file(self.currency_list_path, "csv")

            if currencies_df is not None and not currencies_df.empty:
                # 다양한 방법으로 통화쌍 정보 조회
                currency_info = None

                if "Symbol" in currencies_df.columns:
                    currency_info = currencies_df[currencies_df["Symbol"] == symbol]

                if (currency_info is None or currency_info.empty) and len(
                    currencies_df.columns
                ) > 0:
                    first_col = currencies_df.columns[0]
                    try:
                        currency_info = currencies_df[
                            currencies_df[first_col] == symbol
                        ]
                    except:
                        pass

                if currency_info is not None and not currency_info.empty:
                    result = currency_info.iloc[0].to_dict()
                    self.logger.debug(f"{symbol}: 통화쌍 정보 조회 성공")
                    return result

            self.logger.warning(f"{symbol}: 통화쌍 정보 없음")
            return None

        except Exception as e:
            self.logger.error(f"{symbol}: 통화쌍 정보 조회 실패 - {str(e)}")
            return None

    def analyze_currency_performance(self, symbol: str) -> dict[str, any] | None:
        """
        환율 성과 분석

        Args:
            symbol: 분석할 통화쌍 심볼

        Returns:
            성과 분석 결과 딕셔너리
        """
        try:
            currency_data = self.collect_single_currency(symbol)

            if currency_data is None or currency_data.empty:
                return None

            analysis = {
                "symbol": symbol,
                "period": f"{currency_data.index.min().date()} ~ {currency_data.index.max().date()}",
                "total_days": len(currency_data),
            }

            if "Close" in currency_data.columns:
                rates = currency_data["Close"]

                # 기본 환율 지표
                analysis.update(
                    {
                        "start_rate": rates.iloc[0],
                        "end_rate": rates.iloc[-1],
                        "currency_return": (rates.iloc[-1] / rates.iloc[0] - 1) * 100,
                        "max_rate": rates.max(),
                        "min_rate": rates.min(),
                        "avg_rate": rates.mean(),
                    }
                )

                # 위험 지표
                if "Daily_Return" in currency_data.columns:
                    daily_returns = currency_data["Daily_Return"].dropna()
                    analysis.update(
                        {
                            "volatility": daily_returns.std()
                            * (252**0.5)
                            * 100,  # 연환산 변동성
                            "max_daily_gain": daily_returns.max() * 100,
                            "max_daily_loss": daily_returns.min() * 100,
                            "appreciation_days_ratio": (daily_returns > 0).sum()
                            / len(daily_returns)
                            * 100,
                        }
                    )

                # 환율 포지션 분석
                if "Position_in_Range" in currency_data.columns:
                    current_position = (
                        currency_data["Position_in_Range"].dropna().iloc[-1]
                        if not currency_data["Position_in_Range"].dropna().empty
                        else None
                    )
                    analysis["current_range_position"] = current_position

                # 기술적 지표
                if "RSI" in currency_data.columns:
                    current_rsi = (
                        currency_data["RSI"].dropna().iloc[-1]
                        if not currency_data["RSI"].dropna().empty
                        else None
                    )
                    analysis["current_rsi"] = current_rsi

                # 트렌드 분석
                if (
                    "Trend_Signal" in currency_data.columns
                    and "Currency_Strength" in currency_data.columns
                ):
                    current_trend = (
                        currency_data["Trend_Signal"].iloc[-1]
                        if len(currency_data) > 0
                        else 0
                    )
                    current_strength = (
                        currency_data["Currency_Strength"].dropna().iloc[-1]
                        if not currency_data["Currency_Strength"].dropna().empty
                        else None
                    )
                    analysis.update(
                        {
                            "trend_direction": (
                                "Appreciation" if current_trend else "Depreciation"
                            ),
                            "currency_strength": current_strength,
                        }
                    )

            self.logger.info(f"{symbol} 환율 성과 분석 완료")
            return analysis

        except Exception as e:
            self.logger.error(f"{symbol}: 환율 성과 분석 실패 - {str(e)}")
            return None

    def analyze_currency_correlation(self, symbols: list[str]) -> pd.DataFrame | None:
        """
        통화쌍 간 상관관계 분석

        Args:
            symbols: 분석할 통화쌍 심볼 리스트

        Returns:
            상관관계 매트릭스 DataFrame
        """
        try:
            self.logger.info(f"통화쌍 상관관계 분석 시작: {len(symbols)}개")

            # 모든 통화쌍 데이터 수집
            currency_returns = {}

            for symbol in symbols:
                data = self.collect_single_currency(symbol)
                if data is not None and "Daily_Return" in data.columns:
                    returns = data["Daily_Return"].dropna()
                    if len(returns) > 0:
                        currency_returns[symbol] = returns

            if len(currency_returns) < 2:
                self.logger.warning("상관관계 분석을 위한 충분한 데이터 없음")
                return None

            # 공통 날짜 기준으로 DataFrame 생성
            returns_df = pd.DataFrame(currency_returns)

            # 상관관계 계산
            correlation_matrix = returns_df.corr()

            self.logger.info("통화쌍 상관관계 분석 완료")
            return correlation_matrix

        except Exception as e:
            self.logger.error(f"통화쌍 상관관계 분석 실패: {str(e)}")
            return None

    def get_hedge_analysis(
        self, base_amount: float, target_currency: str, hedge_ratio: float = 1.0
    ) -> dict[str, any] | None:
        """
        환헤지 분석

        Args:
            base_amount: 기준 통화 금액
            target_currency: 대상 통화 (USD, EUR 등)
            hedge_ratio: 헤지 비율 (0.0~1.0)

        Returns:
            환헤지 분석 결과
        """
        try:
            # 해당 통화쌍 찾기
            hedge_symbol = None
            for pair_id, config in self.target_pairs.items():
                symbol = config["symbol"]
                if target_currency in symbol and self.base_currency in symbol:
                    hedge_symbol = symbol
                    break

            if not hedge_symbol:
                self.logger.error(
                    f"{target_currency}/{self.base_currency} 통화쌍을 찾을 수 없음"
                )
                return None

            # 환율 데이터 수집
            currency_data = self.collect_single_currency(hedge_symbol)

            if currency_data is None or currency_data.empty:
                return None

            if "Close" not in currency_data.columns:
                return None

            current_rate = currency_data["Close"].iloc[-1]

            # 환헤지 분석
            hedge_analysis = {
                "base_amount": base_amount,
                "target_currency": target_currency,
                "base_currency": self.base_currency,
                "current_rate": current_rate,
                "hedge_ratio": hedge_ratio,
                "hedged_amount": base_amount * hedge_ratio,
                "unhedged_amount": base_amount * (1 - hedge_ratio),
            }

            # 변환 금액 계산
            if f"{target_currency}{self.base_currency}" in hedge_symbol:
                # 예: USDKRW (USD를 KRW로 변환)
                converted_amount = base_amount * current_rate
                hedge_analysis["converted_amount"] = converted_amount
            else:
                # 예: KRWUSD (KRW를 USD로 변환)
                converted_amount = base_amount / current_rate
                hedge_analysis["converted_amount"] = converted_amount

            # 변동성 리스크 분석
            if "Volatility_20" in currency_data.columns:
                recent_volatility = (
                    currency_data["Volatility_20"].dropna().iloc[-1]
                    if not currency_data["Volatility_20"].dropna().empty
                    else None
                )
                if recent_volatility:
                    # 95% 신뢰구간 환율 리스크
                    risk_range = (
                        current_rate * recent_volatility * 1.96 * (21**0.5)
                    )  # 월간 리스크

                    hedge_analysis.update(
                        {
                            "monthly_volatility": recent_volatility * (21**0.5) * 100,
                            "risk_range_upper": current_rate + risk_range,
                            "risk_range_lower": current_rate - risk_range,
                            "value_at_risk_95": base_amount
                            * recent_volatility
                            * 1.96
                            * (21**0.5),
                        }
                    )

            self.logger.info(f"{target_currency} 환헤지 분석 완료")
            return hedge_analysis

        except Exception as e:
            self.logger.error(f"환헤지 분석 실패: {str(e)}")
            return None

    def get_carry_trade_analysis(
        self, currency_pairs: list[str]
    ) -> dict[str, dict] | None:
        """
        캐리 트레이드 분석

        Args:
            currency_pairs: 분석할 통화쌍 리스트

        Returns:
            캐리 트레이드 분석 결과
        """
        try:
            self.logger.info(f"캐리 트레이드 분석 시작: {len(currency_pairs)}개 통화쌍")

            carry_analysis = {}

            for symbol in currency_pairs:
                data = self.collect_single_currency(symbol)

                if data is None or "Close" not in data.columns:
                    continue

                # 기본 캐리 트레이드 분석
                analysis = {
                    "symbol": symbol,
                    "current_rate": data["Close"].iloc[-1],
                }

                # 수익률 및 위험 분석
                if "Daily_Return" in data.columns:
                    daily_returns = data["Daily_Return"].dropna()

                    if len(daily_returns) > 0:
                        # 연환산 수익률
                        annual_return = daily_returns.mean() * 252 * 100
                        annual_volatility = daily_returns.std() * (252**0.5) * 100

                        # 샤프 비율 (캐리 트레이드용)
                        sharpe_ratio = (
                            annual_return / annual_volatility
                            if annual_volatility > 0
                            else 0
                        )

                        analysis.update(
                            {
                                "annual_return": annual_return,
                                "annual_volatility": annual_volatility,
                                "sharpe_ratio": sharpe_ratio,
                                "return_volatility_ratio": (
                                    annual_return / annual_volatility
                                    if annual_volatility > 0
                                    else 0
                                ),
                            }
                        )

                # 트렌드 및 모멘텀 분석
                if "Currency_Strength" in data.columns:
                    current_strength = (
                        data["Currency_Strength"].dropna().iloc[-1]
                        if not data["Currency_Strength"].dropna().empty
                        else None
                    )
                    analysis["currency_momentum"] = current_strength

                # 드로우다운 분석 (캐리 트레이드 리스크)
                if len(data) > 60:  # 최소 2개월 데이터
                    cumulative_returns = (1 + data["Daily_Return"]).cumprod()
                    rolling_max = cumulative_returns.expanding().max()
                    drawdown = (cumulative_returns / rolling_max - 1) * 100
                    max_drawdown = drawdown.min()

                    analysis["max_drawdown"] = max_drawdown

                carry_analysis[symbol] = analysis

            # 캐리 트레이드 랭킹
            if carry_analysis:
                # 샤프 비율 기준 랭킹
                sorted_by_sharpe = sorted(
                    carry_analysis.items(),
                    key=lambda x: x[1].get("sharpe_ratio", 0),
                    reverse=True,
                )

                for i, (symbol, data) in enumerate(sorted_by_sharpe):
                    carry_analysis[symbol]["sharpe_rank"] = i + 1

                # 수익률/변동성 비율 기준 랭킹
                sorted_by_ratio = sorted(
                    carry_analysis.items(),
                    key=lambda x: x[1].get("return_volatility_ratio", 0),
                    reverse=True,
                )

                for i, (symbol, data) in enumerate(sorted_by_ratio):
                    carry_analysis[symbol]["ratio_rank"] = i + 1

            self.logger.info("캐리 트레이드 분석 완료")
            return carry_analysis

        except Exception as e:
            self.logger.error(f"캐리 트레이드 분석 실패: {str(e)}")
            return None

    def get_volatility_analysis(self, symbols: list[str]) -> dict[str, dict] | None:
        """
        환율 변동성 분석

        Args:
            symbols: 분석할 통화쌍 심볼 리스트

        Returns:
            변동성 분석 결과
        """
        try:
            self.logger.info(f"환율 변동성 분석 시작: {len(symbols)}개")

            volatility_analysis = {}

            for symbol in symbols:
                data = self.collect_single_currency(symbol)

                if data is None or "Daily_Return" not in data.columns:
                    continue

                daily_returns = data["Daily_Return"].dropna()

                if len(daily_returns) < 30:  # 최소 30일 데이터
                    continue

                analysis = {"symbol": symbol, "data_points": len(daily_returns)}

                # 기본 변동성 지표
                daily_vol = daily_returns.std()
                weekly_vol = daily_vol * (5**0.5)
                monthly_vol = daily_vol * (21**0.5)
                annual_vol = daily_vol * (252**0.5)

                analysis.update(
                    {
                        "daily_volatility": daily_vol * 100,
                        "weekly_volatility": weekly_vol * 100,
                        "monthly_volatility": monthly_vol * 100,
                        "annual_volatility": annual_vol * 100,
                    }
                )

                # 변동성 특성
                if len(daily_returns) >= 252:  # 1년 이상 데이터
                    # 롤링 변동성 추이
                    rolling_vol_252 = daily_returns.rolling(window=252).std() * (
                        252**0.5
                    )
                    current_vol = (
                        rolling_vol_252.iloc[-1]
                        if not rolling_vol_252.empty
                        else annual_vol
                    )
                    avg_vol = (
                        rolling_vol_252.mean()
                        if not rolling_vol_252.empty
                        else annual_vol
                    )

                    analysis.update(
                        {
                            "current_annual_vol": current_vol * 100,
                            "average_annual_vol": avg_vol * 100,
                            "vol_regime": (
                                "High"
                                if current_vol > avg_vol * 1.2
                                else (
                                    "Low" if current_vol < avg_vol * 0.8 else "Normal"
                                )
                            ),
                        }
                    )

                # 변동성 클러스터링 분석 (GARCH 효과)
                if len(daily_returns) >= 60:
                    abs_returns = abs(daily_returns)
                    vol_autocorr = abs_returns.autocorr(lag=1)  # 1일 지연 자기상관

                    analysis["volatility_clustering"] = vol_autocorr

                # 꼬리 위험 (tail risk) 분석
                percentile_5 = daily_returns.quantile(0.05)
                percentile_95 = daily_returns.quantile(0.95)
                var_95 = percentile_5  # Value at Risk (95%)

                analysis.update(
                    {
                        "var_95_daily": var_95 * 100,
                        "var_95_monthly": var_95 * (21**0.5) * 100,
                        "skewness": daily_returns.skew(),
                        "kurtosis": daily_returns.kurtosis(),
                    }
                )

                # 변동성 예측 (간단한 EWMA)
                if len(daily_returns) >= 30:
                    # 지수가중이동평균 (람다=0.94, RiskMetrics 기준)
                    ewma_vol = daily_returns.ewm(alpha=0.06).std().iloc[-1] * (252**0.5)
                    analysis["ewma_annual_vol"] = ewma_vol * 100

                volatility_analysis[symbol] = analysis

            self.logger.info("환율 변동성 분석 완료")
            return volatility_analysis

        except Exception as e:
            self.logger.error(f"환율 변동성 분석 실패: {str(e)}")
            return None
