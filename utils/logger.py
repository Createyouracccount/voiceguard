"""
VoiceGuard AI - 로깅 설정 유틸리티
체계적인 로그 관리
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    console_output: bool = True,
    file_output: bool = True
):
    """로깅 시스템 설정"""
    
    # 로그 레벨 설정
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # 로그 포맷터
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 루트 로거 설정
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # 기존 핸들러 제거
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 콘솔 출력
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        root_logger.addHandler(console_handler)
    
    # 파일 출력
    if file_output:
        # 로그 디렉토리 생성
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # 로그 파일명 생성
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d")
            log_file = log_dir / f"voiceguard_{timestamp}.log"
        else:
            log_file = log_dir / log_file
        
        # 로테이팅 파일 핸들러
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        root_logger.addHandler(file_handler)
    
    # 외부 라이브러리 로그 레벨 조정
    _configure_external_loggers()
    
    # 시작 로그
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("VoiceGuard AI 로깅 시스템 시작")
    logger.info(f"로그 레벨: {log_level}")
    if file_output:
        logger.info(f"로그 파일: {log_file}")
    logger.info("=" * 60)

def _configure_external_loggers():
    """외부 라이브러리 로거 설정"""
    
    # 시끄러운 라이브러리들 조용히 하기
    external_loggers = {
        'urllib3': logging.WARNING,
        'requests': logging.WARNING,
        'google': logging.WARNING,
        'grpc': logging.ERROR,
        'elevenlabs': logging.WARNING,
        'openai': logging.WARNING,
        'anthropic': logging.WARNING,
        'langchain': logging.WARNING,
        'httpx': logging.WARNING,
        'asyncio': logging.WARNING
    }
    
    for logger_name, level in external_loggers.items():
        logging.getLogger(logger_name).setLevel(level)

class VoiceGuardLogger:
    """VoiceGuard 전용 로거 클래스"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.name = name
    
    def debug(self, message: str, **kwargs):
        """디버그 로그"""
        self.logger.debug(self._format_message(message, **kwargs))
    
    def info(self, message: str, **kwargs):
        """정보 로그"""
        self.logger.info(self._format_message(message, **kwargs))
    
    def warning(self, message: str, **kwargs):
        """경고 로그"""
        self.logger.warning(self._format_message(message, **kwargs))
    
    def error(self, message: str, **kwargs):
        """오류 로그"""
        self.logger.error(self._format_message(message, **kwargs))
    
    def critical(self, message: str, **kwargs):
        """치명적 오류 로그"""
        self.logger.critical(self._format_message(message, **kwargs))
    
    def _format_message(self, message: str, **kwargs) -> str:
        """메시지 포맷팅"""
        if kwargs:
            extra_info = " | ".join(f"{k}={v}" for k, v in kwargs.items())
            return f"{message} | {extra_info}"
        return message
    
    def log_performance(self, operation: str, duration: float, **metrics):
        """성능 로그"""
        perf_info = f"PERF | {operation} | {duration:.3f}s"
        if metrics:
            perf_info += " | " + " | ".join(f"{k}={v}" for k, v in metrics.items())
        self.logger.info(perf_info)
    
    def log_user_action(self, action: str, **context):
        """사용자 행동 로그"""
        action_info = f"USER | {action}"
        if context:
            action_info += " | " + " | ".join(f"{k}={v}" for k, v in context.items())
        self.logger.info(action_info)
    
    def log_detection(self, text: str, risk_score: float, scam_type: str, **details):
        """탐지 결과 로그"""
        detection_info = f"DETECT | risk={risk_score:.2f} | type={scam_type} | text={text[:50]}..."
        if details:
            detection_info += " | " + " | ".join(f"{k}={v}" for k, v in details.items())
        self.logger.info(detection_info)

def get_logger(name: str) -> VoiceGuardLogger:
    """VoiceGuard 로거 인스턴스 반환"""
    return VoiceGuardLogger(name)

def log_system_info():
    """시스템 정보 로깅"""
    import platform
    import sys
    
    logger = get_logger("system")
    
    logger.info("시스템 정보:")
    logger.info(f"  OS: {platform.system()} {platform.release()}")
    logger.info(f"  Python: {sys.version}")
    logger.info(f"  Architecture: {platform.machine()}")
    
    # 메모리 정보 (선택사항)
    try:
        import psutil
        memory = psutil.virtual_memory()
        logger.info(f"  Memory: {memory.total // (1024**3)}GB total, {memory.available // (1024**3)}GB available")
    except ImportError:
        logger.debug("psutil not available - memory info skipped")

class PerformanceLogger:
    """성능 측정 및 로깅"""
    
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.logger = get_logger("performance")
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.debug(f"{self.operation_name} 시작")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = (datetime.now() - self.start_time).total_seconds()
            
            if exc_type is None:
                self.logger.log_performance(self.operation_name, duration, status="success")
            else:
                self.logger.log_performance(self.operation_name, duration, status="failed", error=str(exc_val))

def setup_debug_logging():
    """디버그 모드 로깅 설정"""
    setup_logging(
        log_level="DEBUG",
        console_output=True,
        file_output=True
    )
    
    # 시스템 정보 출력
    log_system_info()

def setup_production_logging():
    """프로덕션 모드 로깅 설정"""
    setup_logging(
        log_level="INFO",
        console_output=False,  # 콘솔 출력 최소화
        file_output=True
    )

def cleanup_old_logs(days_to_keep: int = 7):
    """오래된 로그 파일 정리"""
    
    logger = get_logger("cleanup")
    log_dir = Path("logs")
    
    if not log_dir.exists():
        return
    
    cutoff_time = datetime.now().timestamp() - (days_to_keep * 24 * 3600)
    removed_count = 0
    
    for log_file in log_dir.glob("*.log*"):
        if log_file.stat().st_mtime < cutoff_time:
            try:
                log_file.unlink()
                removed_count += 1
            except Exception as e:
                logger.warning(f"로그 파일 삭제 실패: {log_file} - {e}")
    
    if removed_count > 0:
        logger.info(f"오래된 로그 파일 {removed_count}개 삭제됨")

# 편의 함수들
def debug(message: str, **kwargs):
    """전역 디버그 로그"""
    get_logger("global").debug(message, **kwargs)

def info(message: str, **kwargs):
    """전역 정보 로그"""
    get_logger("global").info(message, **kwargs)

def warning(message: str, **kwargs):
    """전역 경고 로그"""
    get_logger("global").warning(message, **kwargs)

def error(message: str, **kwargs):
    """전역 오류 로그"""
    get_logger("global").error(message, **kwargs)