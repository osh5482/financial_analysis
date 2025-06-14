"""
로깅 설정 모듈
- Loguru를 사용한 중앙화된 로깅 설정
- 파일 로테이션, 레벨별 분리, JSON 포맷 지원
"""

import sys
from pathlib import Path

from loguru import logger

# 프로젝트 루트 디렉토리 경로
PROJECT_ROOT = Path(__file__).parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"

# 로그 디렉토리 생성
LOGS_DIR.mkdir(exist_ok=True)

# 로깅 설정 딕셔너리
LOGGING_CONFIG: dict[str, any] = {
    # 콘솔 출력 설정
    "console": {
        "sink": sys.stdout,
        "level": "INFO",
        "format": "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>",
        "colorize": True,
        "backtrace": True,
        "diagnose": True,
    },
    # 일반 로그 파일 (INFO 이상)
    "file_info": {
        "sink": LOGS_DIR / "app_{time:YYYY-MM-DD}.log",
        "level": "INFO",
        "format": "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
        "rotation": "1 day",
        "retention": "30 days",
        "compression": "zip",
        "encoding": "utf-8",
        "backtrace": True,
        "diagnose": True,
    },
    # 에러 로그 파일 (ERROR 이상)
    "file_error": {
        "sink": LOGS_DIR / "error_{time:YYYY-MM-DD}.log",
        "level": "ERROR",
        "format": "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
        "rotation": "1 day",
        "retention": "90 days",
        "compression": "zip",
        "encoding": "utf-8",
        "backtrace": True,
        "diagnose": True,
    },
    # 데이터 수집 전용 로그 파일
    "file_data_collection": {
        "sink": LOGS_DIR / "data_collection_{time:YYYY-MM-DD}.log",
        "level": "DEBUG",
        "format": "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
        "rotation": "1 day",
        "retention": "7 days",
        "compression": "zip",
        "encoding": "utf-8",
        "filter": lambda record: "data_collector" in record["name"],
        "backtrace": True,
        "diagnose": True,
    },
    # JSON 형태의 구조화된 로그 (분석용)
    "file_json": {
        "sink": LOGS_DIR / "structured_{time:YYYY-MM-DD}.json",
        "level": "INFO",
        "format": "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {name}:{function}:{line} | {message}",
        "serialize": True,
        "rotation": "1 day",
        "retention": "30 days",
        "compression": "zip",
        "encoding": "utf-8",
    },
}

# 개발 환경 설정
DEVELOPMENT_CONFIG: dict[str, any] = {
    "console_debug": {
        "sink": sys.stdout,
        "level": "DEBUG",
        "format": "<green>{time:HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan> | "
        "<level>{message}</level>",
        "colorize": True,
    }
}

# 성능 측정용 로그 포맷
PERFORMANCE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | PERF | {name}:{function} | "
    "Operation: {extra[operation]} | Duration: {extra[duration]:.4f}s | "
    "Status: {extra[status]} | {message}"
)
