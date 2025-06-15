"""
VoiceGuard AI - 유틸리티 패키지
공통 유틸리티 함수 및 헬퍼 클래스
"""

from .logger import (
    setup_logging,
    get_logger,
    VoiceGuardLogger,
    PerformanceLogger,
    log_system_info,
    cleanup_old_logs
)

from .validators import (
    validate_environment,
    validate_dependencies,
    check_api_connectivity,
    generate_setup_instructions,
    quick_health_check
)

# 시스템 초기화 헬퍼
def initialize_system(debug: bool = False) -> dict:
    """시스템 전체 초기화"""
    
    try:
        # 로깅 설정
        if debug:
            setup_logging(log_level="DEBUG")
        else:
            setup_logging(log_level="INFO")
        
        logger = get_logger("system")
        logger.info("시스템 초기화 시작")
        
        # 환경 검증
        env_result = validate_environment()
        if not env_result['valid']:
            logger.error("환경 검증 실패")
            return {
                "success": False,
                "stage": "environment_validation",
                "errors": env_result['errors']
            }
        
        # 의존성 확인
        dep_result = validate_dependencies()
        if not dep_result['all_required_available']:
            logger.error("필수 의존성 누락")
            return {
                "success": False,
                "stage": "dependency_check", 
                "errors": dep_result['missing_required']
            }
        
        # API 연결성 확인
        api_result = check_api_connectivity()
        if not any(api_result.values()):
            logger.warning("API 연결 불가 - 제한된 기능으로 동작")
        
        logger.info("시스템 초기화 완료")
        
        return {
            "success": True,
            "environment": env_result,
            "dependencies": dep_result,
            "api_connectivity": api_result
        }
        
    except Exception as e:
        return {
            "success": False,
            "stage": "initialization",
            "error": str(e)
        }

# 시스템 상태 확인
def check_system_status() -> dict:
    """시스템 전체 상태 확인"""
    
    try:
        logger = get_logger("system")
        
        # 빠른 헬스 체크
        health_ok = quick_health_check()
        
        # API 연결성
        api_status = check_api_connectivity()
        
        # 시스템 리소스 (psutil 사용 가능시)
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            resources = {
                "cpu_usage": f"{cpu_percent}%",
                "memory_usage": f"{memory.percent}%",
                "disk_usage": f"{(disk.used / disk.total * 100):.1f}%",
                "available_memory": f"{memory.available // (1024**3)}GB"
            }
        except ImportError:
            resources = {"status": "psutil not available"}
        
        return {
            "overall_health": health_ok,
            "api_connectivity": api_status,
            "system_resources": resources,
            "timestamp": "현재시간"  # datetime.now().isoformat()
        }
        
    except Exception as e:
        logger = get_logger("system")
        logger.error(f"시스템 상태 확인 실패: {e}")
        
        return {
            "overall_health": False,
            "error": str(e)
        }

# 시스템 정리
def cleanup_system():
    """시스템 정리 및 종료"""
    
    try:
        logger = get_logger("system")
        logger.info("시스템 정리 시작")
        
        # 오래된 로그 정리
        cleanup_old_logs(days_to_keep=7)
        
        # 기타 정리 작업
        # (향후 추가될 정리 로직)
        
        logger.info("시스템 정리 완료")
        
    except Exception as e:
        # 정리 중 오류가 발생해도 시스템 종료는 계속 진행
        print(f"정리 중 오류 (무시하고 계속): {e}")

# 디버그 정보 수집
def collect_debug_info() -> dict:
    """디버그용 시스템 정보 수집"""
    
    import platform
    import sys
    import os
    
    debug_info = {
        "system": {
            "platform": platform.system(),
            "platform_release": platform.release(),
            "platform_version": platform.version(),
            "architecture": platform.machine(),
            "processor": platform.processor()
        },
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "path": sys.path[:3]  # 처음 3개만
        },
        "environment": {
            "working_directory": os.getcwd(),
            "environment_vars": {
                key: "***" if "API" in key or "SECRET" in key or "PASSWORD" in key 
                else value for key, value in os.environ.items() 
                if key.startswith(("VOICE", "GOOGLE", "ELEVEN", "RETURN", "DEBUG", "LOG"))
            }
        }
    }
    
    # 패키지 정보 (일부만)
    try:
        import pkg_resources
        installed_packages = {pkg.project_name: pkg.version 
                            for pkg in pkg_resources.working_set 
                            if pkg.project_name in ['google-generativeai', 'elevenlabs', 'pyaudio']}
        debug_info["packages"] = installed_packages
    except:
        debug_info["packages"] = "정보 수집 실패"
    
    return debug_info

# 성능 모니터링 헬퍼
class SystemMonitor:
    """시스템 성능 모니터링"""
    
    def __init__(self):
        self.logger = get_logger("monitor")
        self.start_time = None
        self.metrics = {}
    
    def start_monitoring(self):
        """모니터링 시작"""
        import time
        self.start_time = time.time()
        self.logger.info("성능 모니터링 시작")
    
    def record_metric(self, name: str, value: float):
        """메트릭 기록"""
        self.metrics[name] = value
        self.logger.debug(f"메트릭 기록: {name}={value}")
    
    def get_runtime(self) -> float:
        """실행 시간 반환"""
        if self.start_time:
            import time
            return time.time() - self.start_time
        return 0.0
    
    def get_summary(self) -> dict:
        """성능 요약"""
        return {
            "runtime_seconds": self.get_runtime(),
            "metrics": self.metrics.copy(),
            "status": "monitoring"
        }

__all__ = [
    # 로깅 관련
    'setup_logging',
    'get_logger', 
    'VoiceGuardLogger',
    'PerformanceLogger',
    'log_system_info',
    'cleanup_old_logs',
    
    # 검증 관련
    'validate_environment',
    'validate_dependencies',
    'check_api_connectivity',
    'generate_setup_instructions',
    'quick_health_check',
    
    # 시스템 관리
    'initialize_system',
    'check_system_status',
    'cleanup_system',
    'collect_debug_info',
    'SystemMonitor'
]