"""
VoiceGuard AI - 메인 애플리케이션 클래스
모든 모드와 서비스를 통합 관리
"""

import asyncio
import logging
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum

from app.modes import (
    PreventionMode, DetectionMode, PostIncidentMode, ConsultationMode
)
from core.llm_manager import llm_manager
from services.audio_manager import audio_manager  # 경로 수정
from services.tts_service import tts_service      # 경로 수정
from config.settings import settings
from utils.validators import validate_environment

logger = logging.getLogger(__name__)

class AppState(Enum):
    """애플리케이션 상태"""
    INITIALIZING = "initializing"
    MODE_SELECTION = "mode_selection"
    RUNNING = "running"
    SHUTTING_DOWN = "shutting_down"
    ERROR = "error"

class VoiceGuardApp:
    """VoiceGuard AI 메인 애플리케이션"""
    
    def __init__(self):
        self.state = AppState.INITIALIZING
        self.current_mode = None
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.start_time = None
        self.is_running = False
        
        # 모드 클래스들
        self.modes = {
            'prevention': PreventionMode,
            'detection': DetectionMode,
            'post_incident': PostIncidentMode,
            'consultation': ConsultationMode
        }
        
        # 핵심 서비스들
        self.llm_manager = llm_manager
        self.audio_manager = audio_manager
        self.tts_service = tts_service
        
        logger.info("VoiceGuard 애플리케이션 초기화")
    
    async def run(self):
        """애플리케이션 메인 실행"""
        
        self.start_time = datetime.now()
        
        try:
            # 1. 환경 검증
            if not await self._validate_environment():
                raise RuntimeError("환경 검증 실패")
            
            # 2. 서비스 초기화
            if not await self._initialize_services():
                raise RuntimeError("서비스 초기화 실패")
            
            # 3. 환영 메시지
            await self._show_welcome()
            
            # 4. 모드 선택
            selected_mode = await self._select_mode()
            
            # 5. 선택된 모드 실행
            await self._run_mode(selected_mode)
            
        except Exception as e:
            logger.error(f"애플리케이션 실행 중 오류: {e}")
            self.state = AppState.ERROR
            raise
        finally:
            await self._cleanup()
    
    async def _validate_environment(self) -> bool:
        """환경 검증"""
        
        logger.info("🔍 환경 검증 중...")
        
        try:
            # 필수 환경변수 확인
            validation_result = validate_environment()
            
            if not validation_result['valid']:
                logger.error(f"환경 검증 실패: {validation_result['errors']}")
                print("❌ 환경 설정 오류:")
                for error in validation_result['errors']:
                    print(f"   - {error}")
                return False
            
            # 경고사항 출력
            if validation_result['warnings']:
                print("⚠️ 주의사항:")
                for warning in validation_result['warnings']:
                    print(f"   - {warning}")
            
            logger.info("✅ 환경 검증 완료")
            return True
            
        except Exception as e:
            logger.error(f"환경 검증 중 오류: {e}")
            return False
    
    async def _initialize_services(self) -> bool:
        """핵심 서비스 초기화"""
        
        logger.info("🚀 서비스 초기화 중...")
        
        try:
            # 1. LLM 상태 확인
            health_status = await self.llm_manager.health_check()
            if not any(health_status.values()):
                logger.error("LLM 서비스 연결 실패")
                return False
            
            logger.info(f"✅ LLM 상태: {list(health_status.keys())}")
            
            # 2. 오디오 서비스 초기화
            if not self.audio_manager.initialize_output():
                logger.warning("⚠️ 오디오 출력 초기화 실패 (계속 진행)")
            else:
                logger.info("✅ 오디오 서비스 초기화")
            
            # 3. TTS 연결 테스트
            if await self.tts_service.test_connection():
                logger.info("✅ TTS 서비스 연결")
            else:
                logger.warning("⚠️ TTS 서비스 연결 실패 (계속 진행)")
            
            logger.info("🎉 모든 서비스 초기화 완료")
            return True
            
        except Exception as e:
            logger.error(f"서비스 초기화 중 오류: {e}")
            return False
    
    async def _show_welcome(self):
        """환영 메시지 표시"""
        
        welcome_text = """
🛡️  VoiceGuard AI에 오신 것을 환영합니다!

보이스피싱으로부터 안전을 지키는 AI 시스템입니다.
필요한 서비스를 선택해주세요.
        """.strip()
        
        print("=" * 60)
        print(welcome_text)
        print("=" * 60)
        
        # TTS로도 환영 메시지 (선택사항)
        if settings.DEBUG:
            try:
                await self._speak("VoiceGuard AI에 오신 것을 환영합니다!")
            except:
                pass  # TTS 실패해도 계속 진행
    
    async def _select_mode(self) -> str:
        """모드 선택 UI"""
        
        self.state = AppState.MODE_SELECTION
        
        mode_descriptions = {
            'prevention': '🎓 예방 교육 - 보이스피싱 수법 학습 및 대응 훈련',
            'detection': '🔍 실시간 탐지 - 의심스러운 통화 내용 분석',
            'post_incident': '🚨 사후 대처 - 피해 발생 후 해야 할 일들',
            'consultation': '💬 상담 문의 - 보이스피싱 관련 질문 답변'
        }
        
        print("\n📋 서비스 선택:")
        for i, (mode_key, description) in enumerate(mode_descriptions.items(), 1):
            print(f"{i}. {description}")
        
        while True:
            try:
                print("\n원하시는 서비스 번호를 입력하세요 (1-4): ", end="")
                choice = input().strip()
                
                if choice in ['1', '2', '3', '4']:
                    mode_keys = list(mode_descriptions.keys())
                    selected_mode = mode_keys[int(choice) - 1]
                    
                    print(f"\n✅ '{mode_descriptions[selected_mode]}' 모드를 선택하셨습니다.")
                    return selected_mode
                else:
                    print("❌ 1-4 사이의 숫자를 입력해주세요.")
                    
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"❌ 입력 오류: {e}")
    
    async def _run_mode(self, mode_name: str):
        """선택된 모드 실행"""
        
        self.state = AppState.RUNNING
        self.is_running = True
        
        try:
            # 모드 클래스 인스턴스 생성
            mode_class = self.modes[mode_name]
            self.current_mode = mode_class(
                llm_manager=self.llm_manager,
                audio_manager=self.audio_manager,
                tts_service=self.tts_service,
                session_id=self.session_id
            )
            
            logger.info(f"🎯 {mode_name} 모드 시작")
            
            # 모드 실행
            await self.current_mode.run()
            
        except Exception as e:
            logger.error(f"모드 실행 중 오류: {e}")
            raise
        finally:
            self.is_running = False
    
    async def _speak(self, text: str):
        """TTS 음성 출력 (공통 메서드)"""
        
        try:
            audio_stream = self.tts_service.text_to_speech_stream(text)
            await self.audio_manager.play_audio_stream(audio_stream)
        except Exception as e:
            logger.warning(f"TTS 출력 실패: {e}")
    
    def shutdown(self):
        """애플리케이션 종료"""
        
        logger.info("🛑 애플리케이션 종료 시작")
        self.state = AppState.SHUTTING_DOWN
        self.is_running = False
        
        # 현재 모드 종료
        if self.current_mode and hasattr(self.current_mode, 'stop'):
            self.current_mode.stop()
    
    async def _cleanup(self):
        """리소스 정리"""
        
        logger.info("🧹 리소스 정리 중...")
        
        try:
            # 현재 모드 정리
            if self.current_mode and hasattr(self.current_mode, 'cleanup'):
                await self.current_mode.cleanup()
            
            # 오디오 매니저 정리
            if hasattr(self.audio_manager, 'cleanup'):
                self.audio_manager.cleanup()
            
            # TTS 서비스 정리
            if hasattr(self.tts_service, 'cleanup'):
                self.tts_service.cleanup()
            
            # 실행 시간 출력
            if self.start_time:
                runtime = datetime.now() - self.start_time
                logger.info(f"📈 총 실행 시간: {runtime}")
            
            logger.info("✅ 정리 완료")
            
        except Exception as e:
            logger.error(f"정리 중 오류: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """애플리케이션 상태 조회"""
        
        runtime = 0
        if self.start_time:
            runtime = (datetime.now() - self.start_time).total_seconds()
        
        return {
            "state": self.state.value,
            "session_id": self.session_id,
            "is_running": self.is_running,
            "current_mode": self.current_mode.__class__.__name__ if self.current_mode else None,
            "runtime_seconds": runtime,
            "start_time": self.start_time.isoformat() if self.start_time else None
        }