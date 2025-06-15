#!/usr/bin/env python3
"""
VoiceGuard AI - 실제 작동하는 간소화 버전
Gemini API 기반 보이스피싱 탐지 시스템
"""

import asyncio
import logging
import signal
import sys
import os
from pathlib import Path
from datetime import datetime

# 프로젝트 루트를 패스에 추가
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import settings
from services.conversation_manager import HighPerformanceConversationManager

# 간단하고 효율적인 로깅 설정
def setup_logging():
    """간단한 로깅 설정"""
    
    # 기본 로깅 설정
    log_level = getattr(logging, settings.LOG_LEVEL, logging.INFO)
    
    # 포맷터
    formatter = logging.Formatter(
        '%(asctime)s | %(name)-15s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    
    # 루트 로거 설정
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(console_handler)
    
    # 외부 라이브러리 로그 레벨 조정
    logging.getLogger('elevenlabs').setLevel(logging.WARNING)
    logging.getLogger('google').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('grpc').setLevel(logging.ERROR)

setup_logging()
logger = logging.getLogger(__name__)

class VoiceGuardSimpleApp:
    """간소화된 VoiceGuard 애플리케이션"""
    
    def __init__(self):
        self.conversation_manager = None
        self.is_running = False
        self.start_time = None
        
        # 필수 설정 검증
        self._validate_configuration()
    
    def _validate_configuration(self):
        """필수 설정 검증"""
        
        # Google API 키 확인 (필수)
        if not settings.GOOGLE_API_KEY:
            logger.error("❌ GOOGLE_API_KEY가 설정되지 않았습니다!")
            logger.info("📝 .env 파일에 GOOGLE_API_KEY=your_key_here를 추가하세요")
            sys.exit(1)
        
        # STT 설정 확인 (선택사항)
        if not settings.RETURNZERO_CLIENT_ID or not settings.RETURNZERO_CLIENT_SECRET:
            logger.warning("⚠️ ReturnZero STT API 키가 없습니다. 더미 입력을 사용합니다.")
        
        # TTS 설정 확인 (선택사항)
        if not settings.ELEVENLABS_API_KEY:
            logger.warning("⚠️ ElevenLabs TTS API 키가 없습니다. TTS가 제한될 수 있습니다.")
        
        logger.info("✅ 기본 설정 검증 완료")
    
    async def initialize(self):
        """애플리케이션 초기화"""
        
        logger.info("=" * 60)
        logger.info("🛡️  VoiceGuard AI - 간소화 버전")
        logger.info("🧠 Gemini 기반 보이스피싱 탐지 시스템")
        logger.info("=" * 60)
        
        self.start_time = datetime.now()
        
        try:
            # 대화 매니저 생성
            self.conversation_manager = HighPerformanceConversationManager(
                client_id=settings.RETURNZERO_CLIENT_ID or "dummy",
                client_secret=settings.RETURNZERO_CLIENT_SECRET or "dummy"
            )
            
            # 콜백 함수 설정 (UI 연동)
            self.conversation_manager.set_callbacks(
                on_user_speech=self._on_user_speech,
                on_ai_response=self._on_ai_response,
                on_state_change=self._on_state_change
            )
            
            init_time = (datetime.now() - self.start_time).total_seconds()
            logger.info(f"✅ 초기화 완료 ({init_time:.2f}초)")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 초기화 실패: {e}")
            return False
    
    async def run(self):
        """애플리케이션 실행"""
        
        if not await self.initialize():
            logger.error("❌ 애플리케이션 시작 실패")
            return
        
        self.is_running = True
        
        try:
            logger.info("🚀 시스템 시작")
            logger.info("💡 종료하려면 Ctrl+C를 누르세요")
            
            if settings.DEBUG:
                logger.info("🐛 디버그 모드: 'status' 입력으로 상태 확인 가능")
                self._setup_debug_mode()
            
            logger.info("-" * 60)
            
            # 시그널 핸들러 설정
            self._setup_signal_handlers()
            
            # 대화 시작
            await self.conversation_manager.start_conversation()
            
        except KeyboardInterrupt:
            logger.info("\n🛑 사용자 종료 요청")
        except Exception as e:
            logger.error(f"❌ 실행 중 오류: {e}")
        finally:
            await self.cleanup()
    
    def _setup_signal_handlers(self):
        """시그널 핸들러 설정"""
        
        def signal_handler(signum, frame):
            logger.info(f"\n📶 종료 신호 수신 ({signum})")
            if self.conversation_manager:
                self.conversation_manager.is_running = False
            self.is_running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, signal_handler)
    
    def _setup_debug_mode(self):
        """디버그 모드 설정"""
        
        def debug_input_worker():
            """디버그 입력 워커"""
            while self.is_running:
                try:
                    cmd = input().strip().lower()
                    
                    if cmd == 'status':
                        if self.conversation_manager:
                            status = self.conversation_manager.get_conversation_status()
                            metrics = self.conversation_manager.get_performance_metrics()
                            
                            print("\n📊 === 시스템 상태 ===")
                            print(f"   대화 상태: {status['state']}")
                            print(f"   실행 시간: {status['runtime_seconds']:.1f}초")
                            print(f"   총 턴: {status['total_turns']}")
                            print(f"   평균 응답시간: {status['avg_response_time']:.3f}초")
                            print(f"   LLM 비용: ${metrics['llm']['total_cost']:.4f}")
                            print(f"   남은 예산: ${metrics['llm']['remaining_budget']:.2f}")
                            print("=" * 25)
                    
                    elif cmd == 'help':
                        print("\n💡 디버그 명령어:")
                        print("   status - 시스템 상태 확인")
                        print("   help   - 도움말")
                        print()
                    
                except (EOFError, KeyboardInterrupt):
                    break
                except Exception as e:
                    logger.error(f"디버그 입력 오류: {e}")
        
        import threading
        debug_thread = threading.Thread(target=debug_input_worker, daemon=True)
        debug_thread.start()
    
    def _on_user_speech(self, text: str):
        """사용자 음성 콜백"""
        # 간결한 출력
        display_text = text[:60] + "..." if len(text) > 60 else text
        print(f"\n👤 사용자: {display_text}")
    
    def _on_ai_response(self, response: str):
        """AI 응답 콜백"""
        # 간결한 출력 (첫 줄만)
        first_line = response.split('\n')[0]
        display_response = first_line[:80] + "..." if len(first_line) > 80 else first_line
        print(f"\n🤖 VoiceGuard: {display_response}")
        
        # 상세 로그는 파일에만
        logger.info(f"AI 응답 생성 완료 ({len(response)}자)")
    
    def _on_state_change(self, old_state, new_state):
        """상태 변경 콜백"""
        
        if settings.DEBUG:
            state_icons = {
                "idle": "💤",
                "listening": "👂", 
                "processing": "🧠",
                "speaking": "🗣️",
                "error": "❌"
            }
            
            old_icon = state_icons.get(old_state.value, "❓")
            new_icon = state_icons.get(new_state.value, "❓")
            
            print(f"\n{old_icon} → {new_icon} ({new_state.value})")
    
    async def cleanup(self):
        """리소스 정리"""
        
        logger.info("🧹 애플리케이션 종료 중...")
        
        try:
            self.is_running = False
            
            # 대화 매니저 정리
            if self.conversation_manager:
                await self.conversation_manager.cleanup()
            
            # 실행 시간 계산
            if self.start_time:
                total_runtime = (datetime.now() - self.start_time).total_seconds()
                logger.info(f"📈 총 실행 시간: {total_runtime/60:.1f}분")
            
            logger.info("✅ 정리 완료. 안전하게 종료되었습니다.")
            
        except Exception as e:
            logger.error(f"정리 중 오류: {e}")

async def main():
    """메인 함수"""
    
    # 환경 변수 확인
    required_vars = ["GOOGLE_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print("❌ 필수 환경 변수가 설정되지 않았습니다:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\n📝 .env 파일을 생성하고 다음을 추가하세요:")
        print("GOOGLE_API_KEY=your_google_api_key_here")
        print("RETURNZERO_CLIENT_ID=your_returnzero_id (선택)")
        print("RETURNZERO_CLIENT_SECRET=your_returnzero_secret (선택)")
        print("ELEVENLABS_API_KEY=your_elevenlabs_key (선택)")
        sys.exit(1)
    
    # 애플리케이션 실행
    app = VoiceGuardSimpleApp()
    await app.run()

if __name__ == "__main__":
    try:
        # 이벤트 루프 정책 최적화 (Windows)
        if sys.platform.startswith('win'):
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        
        # 애플리케이션 실행
        asyncio.run(main())
        
    except KeyboardInterrupt:
        print("\n👋 안전하게 종료되었습니다.")
    except Exception as e:
        logger.error(f"치명적 오류: {e}")
        sys.exit(1)