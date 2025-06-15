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
        """업데이트된 모드 선택 UI"""
        
        self.state = AppState.MODE_SELECTION
        
        # 더 자세한 모드 설명
        mode_descriptions = {
            'prevention': {
                'title': '🎓 예방 교육',
                'subtitle': '보이스피싱 수법 학습 및 대응 훈련',
                'features': [
                    '• 8가지 주요 사기 수법 학습',
                    '• 실전 시나리오 대응 훈련', 
                    '• 지식 확인 퀴즈',
                    '• 개인별 학습 진도 관리'
                ],
                'recommended_for': '보이스피싱에 대해 배우고 싶은 분'
            },
            'detection': {
                'title': '🔍 실시간 탐지',
                'subtitle': '의심스러운 통화 내용 실시간 분석',
                'features': [
                    '• AI 기반 실시간 위험도 분석',
                    '• 8가지 사기 유형 자동 분류',
                    '• 즉시 경고 및 대응 방법 안내',
                    '• 높은 정확도의 패턴 인식'
                ],
                'recommended_for': '지금 의심스러운 통화를 받고 있는 분'
            },
            'post_incident': {
                'title': '🚨 사후 대처',
                'subtitle': '피해 발생 후 금융감독원 기준 체계적 대응',
                'features': [
                    '• 금융감독원 공식 절차 가이드',
                    '• 단계별 체크리스트 제공',
                    '• 피해금 환급 신청 안내',
                    '• 명의도용 확인 및 차단',
                    '• 개인정보 보호 조치'
                ],
                'recommended_for': '이미 보이스피싱 피해를 당한 분'
            },
            'consultation': {
                'title': '💬 상담 문의',
                'subtitle': '보이스피싱 관련 질문 답변',
                'features': [
                    '• 자주 묻는 질문 답변',
                    '• 상황별 대처법 안내',
                    '• 관련 기관 연락처 제공',
                    '• 예방 수칙 및 팁'
                ],
                'recommended_for': '보이스피싱에 대해 궁금한 점이 있는 분'
            }
        }
        
        print("\n" + "="*80)
        print("🛡️ VoiceGuard AI 서비스 선택")
        print("="*80)
        
        # 긴급 상황 안내
        print("\n🚨 긴급상황이신가요?")
        print("   💰 돈을 송금했거나 → 3번 '사후 대처' 선택")
        print("   📞 지금 의심스러운 통화 중 → 2번 '실시간 탐지' 선택")
        print("   📞 긴급신고: 112 (경찰), 1332 (금융감독원)")
        
        print("\n📋 서비스 상세 안내:")
        
        for i, (mode_key, info) in enumerate(mode_descriptions.items(), 1):
            print(f"\n{i}. {info['title']}")
            print(f"   {info['subtitle']}")
            print(f"   👤 추천대상: {info['recommended_for']}")
            print(f"   ✨ 주요기능:")
            for feature in info['features']:
                print(f"      {feature}")
        
        print("\n" + "="*80)
        
        while True:
            try:
                print("\n원하시는 서비스 번호를 입력하세요 (1-4): ", end="")
                choice = input().strip()
                
                if choice in ['1', '2', '3', '4']:
                    mode_keys = list(mode_descriptions.keys())
                    selected_mode = mode_keys[int(choice) - 1]
                    selected_info = mode_descriptions[selected_mode]
                    
                    print(f"\n✅ '{selected_info['title']}' 모드를 선택하셨습니다.")
                    
                    # 선택 확인
                    if selected_mode == 'post_incident':
                        print("\n⚠️ 사후대처 모드 안내:")
                        print("   이 모드는 이미 피해를 당한 분들을 위한 것입니다.")
                        print("   금융감독원 공식 절차에 따라 단계별로 안내해드립니다.")
                        confirm = input("   계속 진행하시겠습니까? (y/n): ").strip().lower()
                        if confirm not in ['y', 'yes', '예', 'ㅇ']:
                            continue
                    
                    elif selected_mode == 'detection':
                        print("\n💡 실시간 탐지 모드 안내:")
                        print("   의심스러운 대화 내용을 텍스트로 입력하시면")
                        print("   AI가 실시간으로 분석하여 위험도를 알려드립니다.")
                    
                    return selected_mode
                    
                else:
                    print("❌ 1-4 사이의 숫자를 입력해주세요.")
                    
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"❌ 입력 오류: {e}")
    
    def _show_emergency_help(self):
        """긴급 상황 도움말 표시"""
        
        print("\n" + "🚨" * 20)
        print("긴급 상황 대처법")
        print("🚨" * 20)
        
        print("""
📞 즉시 연락할 곳:
• 112 (경찰청) - 보이스피싱 신고 및 수사의뢰
• 1332 (금융감독원) - 금융피해 신고 및 상담
• 해당 은행 고객센터 - 지급정지 신청

⚡ 상황별 즉시 대응:

💰 돈을 송금한 경우:
1. 즉시 112 신고
2. 1332 금융감독원 신고
3. 해당 은행에 지급정지 신청
4. VoiceGuard '사후대처' 모드 이용

📱 개인정보를 알려준 경우:
1. 관련 금융기관에 즉시 연락
2. 계좌/카드 사용정지 요청
3. 비밀번호 전체 변경
4. VoiceGuard '사후대처' 모드 이용

📲 앱을 설치한 경우:
1. 즉시 휴대폰 네트워크 차단
2. 휴대폰 완전 초기화
3. 통신사 고객센터 방문
4. 모든 금융앱 재설치

🛡️ 절대 하지 말 것:
• 사기범과 계속 연락
• 추가 개인정보 제공
• 더 이상의 송금
• 의심스러운 링크 클릭
""")
        
        input("\n이해했으면 Enter를 눌러 서비스 선택으로 돌아가세요...")
    
    def _recommend_mode_by_keywords(self, user_input: str) -> str:
        """키워드 기반 모드 추천"""
        
        user_input_lower = user_input.lower()
        
        # 긴급 상황 키워드
        emergency_keywords = ['돈을', '송금', '이체', '당했', '속았', '피해']
        if any(keyword in user_input_lower for keyword in emergency_keywords):
            return 'post_incident'
        
        # 실시간 상황 키워드  
        realtime_keywords = ['지금', '전화', '통화중', '말하고있', '의심스러운']
        if any(keyword in user_input_lower for keyword in realtime_keywords):
            return 'detection'
        
        # 학습 관련 키워드
        learning_keywords = ['배우고', '공부', '학습', '알고싶', '예방']
        if any(keyword in user_input_lower for keyword in learning_keywords):
            return 'prevention'
        
        # 상담 관련 키워드
        consultation_keywords = ['궁금', '질문', '문의', '상담']
        if any(keyword in user_input_lower for keyword in consultation_keywords):
            return 'consultation'
        
        return 'prevention'  # 기본값
    
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