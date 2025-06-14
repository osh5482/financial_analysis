"""
로거 유틸리티 모듈
- 중앙화된 로거 설정 및 초기화
- 성능 측정 데코레이터
- 컨텍스트 매니저를 통한 작업 단위 로깅
"""

import time
import functools
from contextlib import contextmanager
from loguru import logger

from config.logging_config import LOGGING_CONFIG, DEVELOPMENT_CONFIG, PERFORMANCE_FORMAT


class LoggerSetup:
    """로거 설정 및 초기화를 담당하는 클래스"""

    _initialized = False

    @classmethod
    def initialize(cls, development_mode: bool = False) -> None:
        """
        로거 초기화

        Args:
            development_mode: 개발 모드 여부 (True시 더 상세한 로그 출력)
        """
        if cls._initialized:
            return

        # 기본 핸들러 제거
        logger.remove()

        # 설정에 따라 핸들러 추가
        config = DEVELOPMENT_CONFIG if development_mode else LOGGING_CONFIG

        for handler_name, handler_config in config.items():
            logger.add(**handler_config)

        # 개발 모드가 아닐 때만 프로덕션 핸들러들 추가
        if not development_mode:
            for handler_name, handler_config in LOGGING_CONFIG.items():
                if handler_name not in config:
                    logger.add(**handler_config)

        cls._initialized = True
        logger.info("로거 초기화 완료", extra={"development_mode": development_mode})

    @classmethod
    def get_logger(cls, name: str) -> any:
        """
        이름을 가진 로거 반환

        Args:
            name: 로거 이름 (보통 모듈명)

        Returns:
            설정된 로거 인스턴스
        """
        if not cls._initialized:
            cls.initialize()

        return logger.bind(name=name)


def timing_logger(operation_name: str = None, log_level: str = "INFO"):
    """
    함수 실행 시간을 측정하고 로깅하는 데코레이터

    Args:
        operation_name: 작업 이름 (미지정시 함수명 사용)
        log_level: 로그 레벨
    """

    def decorator(func: callable) -> callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation_name or func.__name__
            start_time = time.time()

            # 작업 시작 로그
            logger.info(f"작업 시작: {op_name}")

            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time

                # 성공 로그
                logger.bind(
                    operation=op_name, duration=duration, status="SUCCESS"
                ).info(f"작업 완료: {op_name}")

                return result

            except Exception as e:
                duration = time.time() - start_time

                # 실패 로그
                logger.bind(
                    operation=op_name, duration=duration, status="FAILED", error=str(e)
                ).error(f"작업 실패: {op_name} - {str(e)}")

                raise

        return wrapper

    return decorator


@contextmanager
def log_operation(operation_name: str, logger_instance: any = None, **extra_context):
    """
    작업 단위 로깅을 위한 컨텍스트 매니저

    Args:
        operation_name: 작업 이름
        logger_instance: 사용할 로거 (미지정시 기본 로거)
        **extra_context: 추가 컨텍스트 정보
    """
    log = logger_instance or logger
    start_time = time.time()

    # 작업 시작 로그
    log.info(f"[{operation_name}] 시작", extra=extra_context)

    try:
        yield log
        duration = time.time() - start_time

        # 성공 로그
        log.info(
            f"[{operation_name}] 완료 (소요시간: {duration:.2f}초)",
            extra={**extra_context, "duration": duration, "status": "SUCCESS"},
        )

    except Exception as e:
        duration = time.time() - start_time

        # 실패 로그
        log.error(
            f"[{operation_name}] 실패 (소요시간: {duration:.2f}초): {str(e)}",
            extra={
                **extra_context,
                "duration": duration,
                "status": "FAILED",
                "error": str(e),
            },
        )
        raise


def log_dataframe_info(
    df, df_name: str = "DataFrame", logger_instance: any = None
) -> None:
    """
    DataFrame 정보를 로깅하는 유틸리티 함수

    Args:
        df: pandas DataFrame
        df_name: DataFrame 이름
        logger_instance: 사용할 로거
    """
    log = logger_instance or logger

    if df is None or df.empty:
        log.warning(f"{df_name}: 빈 데이터프레임")
        return

    log.info(
        f"{df_name} 정보: 행수={len(df)}, 열수={len(df.columns)}, "
        f"메모리사용량={df.memory_usage(deep=True).sum() / 1024 / 1024:.2f}MB"
    )

    # 결측치 정보
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        log.warning(f"{df_name} 결측치: {null_counts[null_counts > 0].to_dict()}")


# 전역 로거 초기화 함수
def setup_logging(development_mode: bool = False) -> any:
    """
    프로젝트 전체에서 사용할 로거 초기화

    Args:
        development_mode: 개발 모드 여부

    Returns:
        초기화된 로거
    """
    LoggerSetup.initialize(development_mode)
    return LoggerSetup.get_logger(__name__)


# 기본 로거 인스턴스 (즉시 사용 가능)
default_logger = setup_logging()
