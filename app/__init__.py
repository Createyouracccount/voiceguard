"""
VoiceGuard AI - 애플리케이션 패키지
메인 애플리케이션 및 운영 모드
"""

try:
    from .app import VoiceGuardApp
    APP_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ VoiceGuardApp import 실패: {e}")
    APP_AVAILABLE = False
    VoiceGuardApp = None

__all__ = [
    'VoiceGuardApp',
    'APP_AVAILABLE'
]