#!/usr/bin/env python3
"""
VoiceGuard AI - 메인 진입점
보이스피싱 종합 대응 시스템
"""

import asyncio
import sys
import signal
from pathlib import Path

# 프로젝트 루트를 패스에 추가
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.app import VoiceGuardApp
from utils.logger import setup_logging
from config.settings import settings

def setup_signal_handlers(app):
    """시그널 핸들러 설정"""
    def signal_handler(signum, frame):
        print(f"\n📶 종료 신호 수신 ({signum})")
        app.shutdown()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, signal_handler)

async def main():
    """메인 함수"""
    
    # 로깅 설정
    setup_logging()
    
    print("🛡️ VoiceGuard AI 시작 중...")
    
    try:
        # 애플리케이션 생성 및 초기화
        app = VoiceGuardApp()
        
        # 시그널 핸들러 설정
        setup_signal_handlers(app)
        
        # 애플리케이션 실행
        await app.run()
        
    except KeyboardInterrupt:
        print("\n👋 사용자에 의해 종료되었습니다.")
    except Exception as e:
        print(f"❌ 시스템 오류: {e}")
        if settings.DEBUG:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    finally:
        print("✅ VoiceGuard AI가 안전하게 종료되었습니다.")

if __name__ == "__main__":
    try:
        # Windows 환경 최적화
        if sys.platform.startswith('win'):
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        
        # 메인 실행
        asyncio.run(main())
        
    except Exception as e:
        print(f"💥 치명적 오류: {e}")
        sys.exit(1)