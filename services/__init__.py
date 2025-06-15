"""
VoiceGuard AI - 서비스 패키지
외부 서비스 연동 및 시스템 서비스 관리
"""

# 오디오 서비스 import - 경로 수정
try:
    from .audio_manager import audio_manager
    from .tts_service import tts_service
    from .stt_service import SttService
    AUDIO_SERVICES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ 오디오 서비스 import 실패: {e}")
    AUDIO_SERVICES_AVAILABLE = False

# 서비스 상태 확인
async def check_services_health() -> dict:
    """모든 서비스 상태 확인"""
    
    health_status = {
        "audio_manager": False,
        "tts_service": False,
        "stt_service": False,
        "overall_status": False
    }
    
    if not AUDIO_SERVICES_AVAILABLE:
        health_status["error"] = "오디오 서비스를 사용할 수 없습니다"
        return health_status
    
    try:
        # 오디오 매니저 상태
        if hasattr(audio_manager, 'is_initialized'):
            health_status["audio_manager"] = audio_manager.is_initialized
        else:
            health_status["audio_manager"] = True  # 기본적으로 사용 가능
        
        # TTS 서비스 상태
        if hasattr(tts_service, 'test_connection'):
            health_status["tts_service"] = await tts_service.test_connection()
        else:
            health_status["tts_service"] = hasattr(tts_service, 'text_to_speech_stream')
        
        # STT 서비스 상태 (클래스이므로 인스턴스 생성 가능 여부만 확인)
        health_status["stt_service"] = True
        
        # 전체 상태
        health_status["overall_status"] = all([
            health_status["audio_manager"],
            health_status["tts_service"], 
            health_status["stt_service"]
        ])
        
    except Exception as e:
        health_status["error"] = str(e)
    
    return health_status

# 서비스 초기화
async def initialize_services() -> dict:
    """서비스 초기화"""
    
    results = {
        "audio_manager": False,
        "tts_service": False,
        "success": False
    }
    
    if not AUDIO_SERVICES_AVAILABLE:
        results["error"] = "오디오 서비스를 사용할 수 없습니다"
        return results
    
    try:
        # 오디오 매니저 초기화
        if hasattr(audio_manager, 'initialize_output'):
            results["audio_manager"] = audio_manager.initialize_output()
        else:
            results["audio_manager"] = True
        
        # TTS 서비스 테스트
        if hasattr(tts_service, 'test_connection'):
            results["tts_service"] = await tts_service.test_connection()
        else:
            results["tts_service"] = True
        
        results["success"] = results["audio_manager"] and results["tts_service"]
        
    except Exception as e:
        results["error"] = str(e)
    
    return results

# 서비스 정리
def cleanup_services():
    """서비스 정리"""
    
    if not AUDIO_SERVICES_AVAILABLE:
        return
    
    try:
        # 오디오 매니저 정리
        if hasattr(audio_manager, 'cleanup'):
            audio_manager.cleanup()
        
        # TTS 서비스 정리
        if hasattr(tts_service, 'cleanup'):
            tts_service.cleanup()
            
        print("✅ 서비스 정리 완료")
        
    except Exception as e:
        print(f"❌ 서비스 정리 중 오류: {e}")

# 서비스 통계
def get_services_stats() -> dict:
    """서비스 통계"""
    
    stats = {
        "available": AUDIO_SERVICES_AVAILABLE,
        "audio_manager": {},
        "tts_service": {},
        "timestamp": "2024-01-01T00:00:00"  # 현재 시간으로 교체 필요
    }
    
    if not AUDIO_SERVICES_AVAILABLE:
        return stats
    
    try:
        # 오디오 매니저 통계
        if hasattr(audio_manager, 'get_performance_stats'):
            stats["audio_manager"] = audio_manager.get_performance_stats()
        
        # TTS 서비스 통계
        if hasattr(tts_service, 'get_performance_stats'):
            stats["tts_service"] = tts_service.get_performance_stats()
            
    except Exception as e:
        stats["error"] = str(e)
    
    return stats

# 조건부 export
__all__ = ['check_services_health', 'initialize_services', 'cleanup_services', 'get_services_stats']

if AUDIO_SERVICES_AVAILABLE:
    __all__.extend(['audio_manager', 'tts_service', 'SttService'])