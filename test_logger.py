"""
로거 설정 테스트 스크립트
"""

import pandas as pd
from src.utils.logger import (
    setup_logging,
    timing_logger,
    log_operation,
    log_dataframe_info,
)

# 로거 초기화 (개발 모드)
logger = setup_logging(development_mode=True)


def test_basic_logging():
    """기본 로깅 테스트"""
    logger.info("기본 정보 로그 테스트")
    logger.debug("디버그 로그 테스트")
    logger.warning("경고 로그 테스트")
    logger.error("에러 로그 테스트")


@timing_logger("데이터 처리 작업")
def test_timing_decorator():
    """타이밍 데코레이터 테스트"""
    import time

    time.sleep(1)  # 1초 대기
    logger.info("데이터 처리 작업 완료")
    return "처리 결과"


def test_context_manager():
    """컨텍스트 매니저 테스트"""
    with log_operation("컨텍스트 매니저 테스트", extra_param="테스트값") as log:
        log.info("컨텍스트 내부에서 작업 수행")
        import time

        time.sleep(0.5)


def test_dataframe_logging():
    """DataFrame 로깅 테스트"""
    # 테스트용 DataFrame 생성
    df = pd.DataFrame(
        {"A": [1, 2, None, 4], "B": ["a", "b", "c", "d"], "C": [1.1, 2.2, 3.3, 4.4]}
    )

    log_dataframe_info(df, "테스트 데이터프레임")

    # 빈 DataFrame 테스트
    empty_df = pd.DataFrame()
    log_dataframe_info(empty_df, "빈 데이터프레임")


def test_error_handling():
    """에러 처리 테스트"""
    try:
        with log_operation("에러 발생 테스트"):
            raise ValueError("의도적인 테스트 에러")
    except ValueError:
        logger.info("에러가 정상적으로 처리되었습니다")


if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("로거 테스트 시작")
    logger.info("=" * 50)

    # 각 테스트 실행
    test_basic_logging()

    result = test_timing_decorator()
    logger.info(f"타이밍 테스트 결과: {result}")

    test_context_manager()
    test_dataframe_logging()
    test_error_handling()

    logger.info("=" * 50)
    logger.info("로거 테스트 완료")
    logger.info("=" * 50)
