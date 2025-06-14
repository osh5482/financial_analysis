"""
지수 데이터 수집기
- FinanceDataReader를 활용한 주요 지수 데이터 수집
- 한국(KOSPI, KOSDAQ), 미국(S&P500, NASDAQ, Dow Jones) 지수 지원
- 지수 특화 검증: 레벨 일관성, 변동성, 시장 대표성 분석
"""

import pandas as pd
import FinanceDataReader as fdr
from pathlib import Path
from datetime import datetime, timedelta

from .base_collector import BaseCollector
from config.settings import (
    MAJOR_INDICES,
    VALIDATION_CONFIG,
    DATA_PATHS,
    get_enabled_indices,
)


class IndexCollector(BaseCollector):
    """
    지수 데이터 수집기

    기능:
    - 주요 지수 데이터 자동 수집
    - 지수 레벨, 변화율, 변동성 분석
    - 시장 대표성 및 상관관계 분석
    - ETF 추적오차 분석을 위한 벤치마크 데이터 제공
    """

    def __init__(
        self,
        markets: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        include_inactive: bool = False,
        batch_size: int | None = None,
        **kwargs,
    ) -> None:
        """
        IndexCollector 초기화

        Args:
            markets: 수집할 시장 리스트 (KR, US 등, None시 모든 시장)
            start_date: 수집 시작일
            end_date: 수집 종료일
            include_inactive: 비활성 지수 포함 여부
            batch_size: 배치 처리 크기
            **kwargs: BaseCollector 추가 인자
        """
        super().__init__("IndexCollector", start_date, end_date, **kwargs)

        self.markets = markets or ["KR", "US"]
        self.include_inactive = include_inactive
        self.batch_size = batch_size or 20

        # 데이터 저장 경로 설정
        self.data_path = DATA_PATHS["indices"]
        self.index_list_path = self.data_path / "indices_list.csv"

        # 수집 대상 지수 필터링
        self.target_indices = self._get_target_indices()

        self.logger.info(f"수집 시장: {', '.join(self.markets)}")
        self.logger.info(f"수집 대상 지수: {len(self.target_indices)}개")
        self.logger.info(f"배치 크기: {self.batch_size}")

    def _get_target_indices(self) -> dict[str, dict]:
        """수집 대상 지수 필터링"""
        enabled_indices = get_enabled_indices()
        target_indices = {}

        for idx_name, idx_config in enabled_indices.items():
            if idx_config.get("market") in self.markets:
                target_indices[idx_name] = idx_config

        # 추가 주요 지수 (설정에 없는 경우)
        additional_indices = self._get_additional_indices()
        for idx_name, idx_config in additional_indices.items():
            if (
                idx_config.get("market") in self.markets
                and idx_name not in target_indices
            ):
                target_indices[idx_name] = idx_config

        return target_indices

    def _get_additional_indices(self) -> dict[str, dict]:
        """설정에 추가할 주요 지수들"""
        additional = {}

        if "KR" in self.markets:
            additional.update(
                {
                    "KOSPI_LARGE": {
                        "symbol": "KS200",
                        "name": "코스피 대형주",
                        "description": "코스피 대형주 지수",
                        "market": "KR",
                        "enabled": True,
                    },
                    "KOSDAQ_STAR": {
                        "symbol": "KQ150",
                        "name": "코스닥 150",
                        "description": "코스닥 우량기업 150개",
                        "market": "KR",
                        "enabled": True,
                    },
                }
            )

        if "US" in self.markets:
            additional.update(
                {
                    "RUSSELL_2000": {
                        "symbol": "RUT",
                        "name": "러셀 2000",
                        "description": "미국 소형주 지수",
                        "market": "US",
                        "enabled": True,
                    },
                    "VIX": {
                        "symbol": "VIX",
                        "name": "VIX 지수",
                        "description": "시장 변동성 지수",
                        "market": "US",
                        "enabled": True,
                    },
                    "NASDAQ_100": {
                        "symbol": "NDX",
                        "name": "나스닥 100",
                        "description": "나스닥 대형 기술주 100개",
                        "market": "US",
                        "enabled": True,
                    },
                }
            )

        return additional

    def get_symbol_list(self) -> list[str]:
        """
        지수 심볼 리스트 반환

        Returns:
            지수 심볼 리스트
        """
        try:
            self.logger.info("지수 리스트 수집 시작")

            symbols = []
            index_data = []

            for idx_name, idx_config in self.target_indices.items():
                symbol = idx_config["symbol"]
                symbols.append(symbol)

                # 지수 정보 저장
                index_data.append(
                    {
                        "Index_ID": idx_name,
                        "Symbol": symbol,
                        "Name": idx_config["name"],
                        "Description": idx_config["description"],
                        "Market": idx_config["market"],
                        "Enabled": idx_config.get("enabled", True),
                        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )

            # 지수 리스트 DataFrame 생성 및 저장
            indices_df = pd.DataFrame(index_data)
            self._save_index_list(indices_df)

            self.logger.info(f"지수 리스트 수집 완료: {len(symbols)}개")
            return symbols

        except Exception as e:
            self.logger.error(f"지수 리스트 수집 실패: {str(e)}")
            return self._load_cached_index_list()

    def _save_index_list(self, indices_df: pd.DataFrame) -> None:
        """지수 리스트를 파일로 저장"""
        try:
            self.save_to_file(indices_df, self.index_list_path, "csv")
            self.logger.info(f"지수 리스트 저장 완료: {self.index_list_path}")

        except Exception as e:
            self.logger.error(f"지수 리스트 저장 실패: {str(e)}")

    def _load_cached_index_list(self) -> list[str]:
        """캐시된 지수 리스트 로드"""
        try:
            indices_df = self.load_from_file(self.index_list_path, "csv")

            if indices_df is not None and not indices_df.empty:
                symbols = indices_df["Symbol"].tolist()
                symbols = [
                    str(symbol).strip() for symbol in symbols if pd.notna(symbol)
                ]

                self.logger.info(f"캐시된 지수 리스트 로드: {len(symbols)}개")
                return symbols

        except Exception as e:
            self.logger.error(f"캐시된 지수 리스트 로드 실패: {str(e)}")

        return []

    def collect_single_index(self, symbol: str) -> pd.DataFrame | None:
        """
        개별 지수 데이터 수집

        Args:
            symbol: 지수 심볼

        Returns:
            지수 데이터 DataFrame 또는 None
        """
        try:
            # FinanceDataReader로 데이터 수집
            data = fdr.DataReader(symbol, self.start_date, self.end_date)

            if data is None or data.empty:
                self.logger.warning(f"{symbol}: 지수 데이터 없음")
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

            # 지수 특화 데이터 검증
            if not self._validate_index_data(data, symbol):
                return None

            # 데이터 정제
            data = self.clean_dataframe(data, symbol)

            # 기본 검증 (지수는 거래량이 없을 수 있음)
            required_columns = ["Close"]  # 지수는 Close만 필수
            min_points = getattr(self, "min_data_points", None)
            if not self.validate_dataframe(data, symbol, required_columns, min_points):
                return None

            # 지수 특화 지표 계산
            data = self._calculate_index_metrics(data, symbol)

            self.logger.debug(f"{symbol}: 지수 데이터 수집 성공 ({len(data)}개 레코드)")
            return data

        except Exception as e:
            self.logger.error(f"{symbol}: 지수 데이터 수집 실패 - {str(e)}")
            return None

    def _validate_index_data(self, df: pd.DataFrame, symbol: str) -> bool:
        """
        지수 특화 데이터 검증

        Args:
            df: 검증할 DataFrame
            symbol: 지수 심볼

        Returns:
            검증 통과 여부
        """
        try:
            # Close 가격 필수 확인
            if "Close" not in df.columns:
                self.logger.error(f"{symbol}: Close 컬럼 누락")
                return False

            # 지수 레벨 검증 (음수 불가)
            if (df["Close"] <= 0).any():
                self.logger.error(f"{symbol}: 음수 또는 0 지수 레벨 존재")
                return False

            # OHLC가 있는 경우 일관성 검증
            if all(col in df.columns for col in ["Open", "High", "Low", "Close"]):
                invalid_levels = (
                    (df["High"] < df["Low"])
                    | (df["High"] < df["Open"])
                    | (df["High"] < df["Close"])
                    | (df["Low"] > df["Open"])
                    | (df["Low"] > df["Close"])
                )

                if invalid_levels.any():
                    invalid_count = invalid_levels.sum()
                    self.logger.warning(
                        f"{symbol}: 지수 레벨 일관성 오류 {invalid_count}건"
                    )

                    if invalid_count / len(df) > 0.05:
                        return False

            # 지수 변동성 검증 (극단적 변동 체크)
            if "Close" in df.columns and len(df) > 1:
                daily_changes = df["Close"].pct_change().dropna()

                # 일일 변동이 50% 이상인 경우 (지수로는 비정상)
                extreme_changes = (abs(daily_changes) > 0.5).sum()
                if extreme_changes > 0:
                    self.logger.warning(
                        f"{symbol}: 극단적 일일 변동 {extreme_changes}건"
                    )

                    # VIX 같은 변동성 지수는 예외
                    if symbol.upper() not in ["VIX", "VKOSPI"] and extreme_changes > 2:
                        return False

            # 지수 연속성 검증 (갭 체크)
            if "Close" in df.columns and len(df) > 1:
                # 전일 대비 갭이 20% 이상인 경우
                daily_gaps = abs(df["Close"].pct_change())
                large_gaps = (daily_gaps > 0.2).sum()

                if large_gaps > 1:  # 1회는 허용 (특수 상황)
                    self.logger.warning(f"{symbol}: 큰 갭 {large_gaps}건")

            return True

        except Exception as e:
            self.logger.error(f"{symbol}: 지수 데이터 검증 중 오류 - {str(e)}")
            return False

    def _calculate_index_metrics(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        지수 특화 지표 계산

        Args:
            df: 원본 DataFrame
            symbol: 지수 심볼

        Returns:
            지표가 추가된 DataFrame
        """
        try:
            if "Close" in df.columns and len(df) > 1:
                # 일일 수익률
                df["Daily_Return"] = df["Close"].pct_change()

                # 일일 변화량 (절대값)
                df["Daily_Change"] = df["Close"].diff()

                # 이동평균 (트렌드 분석용)
                df["MA_5"] = df["Close"].rolling(window=5).mean()
                df["MA_20"] = df["Close"].rolling(window=20).mean()
                df["MA_60"] = df["Close"].rolling(window=60).mean()
                df["MA_200"] = df["Close"].rolling(window=200).mean()

                # 변동성 지표
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

                # 최고점 대비 하락률 (Drawdown)
                rolling_max = df["Close"].expanding().max()
                df["Drawdown"] = (df["Close"] / rolling_max - 1) * 100

                # 지수 성과 지표 (연간화)
                if len(df) >= 252:  # 1년 이상 데이터
                    # 연환산 수익률
                    annual_return = (
                        (df["Close"].iloc[-1] / df["Close"].iloc[-252]) ** (252 / 252)
                    ) - 1
                    df.loc[df.index[-1], "Annual_Return"] = annual_return

                    # 연환산 변동성
                    annual_volatility = df["Daily_Return"].std() * (252**0.5)
                    df.loc[df.index[-1], "Annual_Volatility"] = annual_volatility

                    # 샤프 비율 (무위험 수익률 2% 가정)
                    if annual_volatility > 0:
                        sharpe_ratio = (annual_return - 0.02) / annual_volatility
                        df.loc[df.index[-1], "Sharpe_Ratio"] = sharpe_ratio

                # 추세 지표 (단순한 트렌드 방향)
                if "MA_20" in df.columns and "MA_60" in df.columns:
                    df["Trend_Short"] = (df["MA_20"] > df["MA_60"]).astype(
                        int
                    )  # 단기 상승 추세
                    df["Trend_Long"] = (df["Close"] > df["MA_200"]).astype(
                        int
                    )  # 장기 상승 추세

                self.logger.debug(f"{symbol}: 지수 지표 계산 완료")

        except Exception as e:
            self.logger.warning(f"{symbol}: 지수 지표 계산 중 오류 - {str(e)}")

        return df

    def collect_data(self, symbols: list[str]) -> dict[str, pd.DataFrame]:
        """
        여러 지수 데이터 배치 수집

        Args:
            symbols: 수집할 지수 심볼 리스트

        Returns:
            심볼별 데이터 딕셔너리
        """
        collected_data = {}
        total_symbols = len(symbols)

        self.logger.info(f"지수 데이터 배치 수집 시작: {total_symbols}개")

        # 배치별로 처리
        for i in range(0, total_symbols, self.batch_size):
            batch_symbols = symbols[i : i + self.batch_size]
            batch_num = i // self.batch_size + 1
            total_batches = (total_symbols + self.batch_size - 1) // self.batch_size

            self.logger.info(
                f"배치 {batch_num}/{total_batches} 처리 중: {len(batch_symbols)}개 지수"
            )

            # 배치 내 각 심볼 처리
            for j, symbol in enumerate(batch_symbols):
                self.logger.debug(f"처리 중: {symbol} ({i+j+1}/{total_symbols})")

                # 재시도 로직과 함께 데이터 수집
                index_data = self._retry_request(self.collect_single_index, symbol)

                if index_data is not None and not index_data.empty:
                    collected_data[symbol] = index_data

                    # 개별 파일로 저장 (옵션)
                    if (
                        hasattr(self, "save_individual_files")
                        and self.save_individual_files
                    ):
                        file_path = self.data_path / f"{symbol}.csv"
                        self.save_to_file(index_data, file_path, "csv")
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
            f"전체 지수 수집 완료: {successful_total}/{total_symbols}개 성공"
        )

        return collected_data

    def get_index_info(self, symbol: str) -> dict[str, any] | None:
        """
        개별 지수 정보 조회

        Args:
            symbol: 지수 심볼

        Returns:
            지수 정보 딕셔너리 또는 None
        """
        try:
            indices_df = self.load_from_file(self.index_list_path, "csv")

            if indices_df is not None and not indices_df.empty:
                # 다양한 방법으로 지수 정보 조회
                index_info = None

                if "Symbol" in indices_df.columns:
                    index_info = indices_df[indices_df["Symbol"] == symbol]

                if (index_info is None or index_info.empty) and len(
                    indices_df.columns
                ) > 0:
                    first_col = indices_df.columns[0]
                    try:
                        index_info = indices_df[indices_df[first_col] == symbol]
                    except:
                        pass

                if index_info is not None and not index_info.empty:
                    result = index_info.iloc[0].to_dict()
                    self.logger.debug(f"{symbol}: 지수 정보 조회 성공")
                    return result

            self.logger.warning(f"{symbol}: 지수 정보 없음")
            return None

        except Exception as e:
            self.logger.error(f"{symbol}: 지수 정보 조회 실패 - {str(e)}")
            return None

    def analyze_index_performance(self, symbol: str) -> dict[str, any] | None:
        """
        지수 성과 분석

        Args:
            symbol: 분석할 지수 심볼

        Returns:
            성과 분석 결과 딕셔너리
        """
        try:
            index_data = self.collect_single_index(symbol)

            if index_data is None or index_data.empty:
                return None

            analysis = {
                "symbol": symbol,
                "period": f"{index_data.index.min().date()} ~ {index_data.index.max().date()}",
                "total_days": len(index_data),
            }

            if "Close" in index_data.columns:
                close_levels = index_data["Close"]

                # 기본 성과 지표
                analysis.update(
                    {
                        "start_level": close_levels.iloc[0],
                        "end_level": close_levels.iloc[-1],
                        "total_return": (
                            close_levels.iloc[-1] / close_levels.iloc[0] - 1
                        )
                        * 100,
                        "max_level": close_levels.max(),
                        "min_level": close_levels.min(),
                        "avg_level": close_levels.mean(),
                    }
                )

                # 위험 지표
                if "Daily_Return" in index_data.columns:
                    daily_returns = index_data["Daily_Return"].dropna()
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

                # 최대 낙폭 (Drawdown)
                if "Drawdown" in index_data.columns:
                    analysis["max_drawdown"] = index_data["Drawdown"].min()

                # 기술적 지표
                if "RSI" in index_data.columns:
                    current_rsi = (
                        index_data["RSI"].dropna().iloc[-1]
                        if not index_data["RSI"].dropna().empty
                        else None
                    )
                    analysis["current_rsi"] = current_rsi

                # 트렌드 분석
                if (
                    "Trend_Short" in index_data.columns
                    and "Trend_Long" in index_data.columns
                ):
                    current_short_trend = (
                        index_data["Trend_Short"].iloc[-1] if len(index_data) > 0 else 0
                    )
                    current_long_trend = (
                        index_data["Trend_Long"].iloc[-1] if len(index_data) > 0 else 0
                    )
                    analysis.update(
                        {
                            "short_term_trend": "Up" if current_short_trend else "Down",
                            "long_term_trend": "Up" if current_long_trend else "Down",
                        }
                    )

            self.logger.info(f"{symbol} 지수 성과 분석 완료")
            return analysis

        except Exception as e:
            self.logger.error(f"{symbol}: 지수 성과 분석 실패 - {str(e)}")
            return None

    def analyze_market_correlation(self, symbols: list[str]) -> pd.DataFrame | None:
        """
        지수 간 상관관계 분석

        Args:
            symbols: 분석할 지수 심볼 리스트

        Returns:
            상관관계 매트릭스 DataFrame
        """
        try:
            self.logger.info(f"지수 상관관계 분석 시작: {len(symbols)}개")

            # 모든 지수 데이터 수집
            index_returns = {}

            for symbol in symbols:
                data = self.collect_single_index(symbol)
                if data is not None and "Daily_Return" in data.columns:
                    returns = data["Daily_Return"].dropna()
                    if len(returns) > 0:
                        index_returns[symbol] = returns

            if len(index_returns) < 2:
                self.logger.warning("상관관계 분석을 위한 충분한 데이터 없음")
                return None

            # 공통 날짜 기준으로 DataFrame 생성
            returns_df = pd.DataFrame(index_returns)

            # 상관관계 계산
            correlation_matrix = returns_df.corr()

            self.logger.info("지수 상관관계 분석 완료")
            return correlation_matrix

        except Exception as e:
            self.logger.error(f"지수 상관관계 분석 실패: {str(e)}")
            return None

    def get_market_summary(self, market: str = "KR") -> dict[str, any]:
        """
        시장별 지수 요약 정보

        Args:
            market: 시장 코드 (KR, US)

        Returns:
            시장 요약 정보 딕셔너리
        """
        try:
            market_indices = {
                k: v
                for k, v in self.target_indices.items()
                if v.get("market") == market
            }

            if not market_indices:
                return {}

            market_data = {}

            for idx_name, idx_config in market_indices.items():
                symbol = idx_config["symbol"]
                performance = self.analyze_index_performance(symbol)

                if performance:
                    market_data[idx_name] = {
                        "name": idx_config["name"],
                        "symbol": symbol,
                        "return": performance.get("total_return", 0),
                        "volatility": performance.get("volatility", 0),
                        "max_drawdown": performance.get("max_drawdown", 0),
                        "trend": performance.get("short_term_trend", "Unknown"),
                    }

            # 시장 평균 계산
            if market_data:
                avg_return = sum(data["return"] for data in market_data.values()) / len(
                    market_data
                )
                avg_volatility = sum(
                    data["volatility"] for data in market_data.values()
                ) / len(market_data)

                summary = {
                    "market": market,
                    "indices_count": len(market_data),
                    "average_return": avg_return,
                    "average_volatility": avg_volatility,
                    "indices_data": market_data,
                }

                self.logger.info(f"{market} 시장 요약 완료")
                return summary

            return {}

        except Exception as e:
            self.logger.error(f"{market} 시장 요약 실패: {str(e)}")
            return {}
