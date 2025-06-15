"""
VoiceGuard AI - 설정 패키지
시스템 전반의 설정 및 구성 관리
"""

from .settings import (
    settings,
    ai_config, 
    detection_thresholds,
    scam_config,
    voice_config,
    monitoring_config,
    integration_config
)

# 설정 검증 함수
def validate_config() -> dict:
    """설정 유효성 검증"""
    
    errors = []
    warnings = []
    
    # API 키 검증
    if not settings.GOOGLE_API_KEY:
        errors.append("GOOGLE_API_KEY가 설정되지 않음")
    
    if not settings.ELEVENLABS_API_KEY:
        warnings.append("ELEVENLABS_API_KEY가 설정되지 않음 - TTS 기능 제한")
    
    if not settings.RETURNZERO_CLIENT_ID or not settings.RETURNZERO_CLIENT_SECRET:
        warnings.append("ReturnZero API 설정되지 않음 - STT 기능 제한")
    
    # 임계값 검증
    if not (0 < detection_thresholds.medium_risk < detection_thresholds.high_risk < detection_thresholds.critical_risk <= 1.0):
        errors.append("탐지 임계값이 올바르지 않음")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings
    }

# 설정 요약 함수
def get_config_summary() -> dict:
    """현재 설정 요약"""
    
    return {
        "api_keys": {
            "google_api": bool(settings.GOOGLE_API_KEY),
            "elevenlabs_api": bool(settings.ELEVENLABS_API_KEY),
            "returnzero_api": bool(settings.RETURNZERO_CLIENT_ID and settings.RETURNZERO_CLIENT_SECRET),
            "openai_api": bool(settings.OPENAI_API_KEY),
            "anthropic_api": bool(settings.ANTHROPIC_API_KEY)
        },
        "detection_thresholds": {
            "medium": detection_thresholds.medium_risk,
            "high": detection_thresholds.high_risk, 
            "critical": detection_thresholds.critical_risk
        },
        "scam_categories": len(scam_config.SCAM_CATEGORIES),
        "voice_config": {
            "sample_rate": voice_config.SAMPLE_RATE,
            "channels": voice_config.CHANNELS,
            "stt_model": voice_config.STT_MODEL
        },
        "system": {
            "debug": settings.DEBUG,
            "log_level": settings.LOG_LEVEL
        }
    }

__all__ = [
    'settings',
    'ai_config',
    'detection_thresholds', 
    'scam_config',
    'voice_config',
    'monitoring_config',
    'integration_config',
    'validate_config',
    'get_config_summary'
]