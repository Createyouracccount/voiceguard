# # services/conversation_manager.py

# import asyncio
# import logging
# import threading
# import time
# import queue
# from datetime import datetime
# import numpy as np
# from typing import Optional, Dict, Any, Callable
# from enum import Enum

# # AI 에이전트 시스템 import
# from agents import coordinator_agent, TaskType, TaskPriority

# # 기존 서비스 및 설정 import
# from services.tts_service import tts_service
# from services.audio_manager import audio_manager
# from config.settings import settings

# logger = logging.getLogger(__name__)

# # 이전 코드와 호환성을 위해 SttService를 그대로 사용합니다.
# # 만약 stream_stt.py를 직접 사용하신다면 이 부분을 맞게 수정해야 합니다.
# try:
#     from .stt_service import SttService
# except ImportError:
#     logger.error("SttService를 import할 수 없습니다. services/stt_service.py 파일이 올바르게 구성되었는지 확인하세요.")
#     # 임시 클래스로 대체하여 프로그램이 시작될 수 있도록 함
#     class SttService:
#         def __init__(self, *args, **kwargs): pass
#         def start(self): pass
#         def stop(self): pass


# class ConversationState(Enum):
#     IDLE = "idle"
#     LISTENING = "listening"
#     PROCESSING = "processing"
#     SPEAKING = "speaking"
#     ERROR = "error"

# class ConversationManager:
#     """
#     고성능 대화 관리자와 멀티 에이전트 시스템을 통합한 최종 버전.
#     - 기존의 정교한 오디오/침묵 제어 기능 유지
#     - 분석 로직을 LangGraph에서 CoordinatorAgent로 교체
#     """
    
#     def __init__(self, client_id: str, client_secret: str):
#         # 1. 컴포넌트 초기화
#         self.stt_service = SttService(client_id, client_secret, self._on_final_transcript)
#         self.coordinator = coordinator_agent # LangGraph 대신 CoordinatorAgent 사용
#         self.tts_service = tts_service
#         self.audio_manager = audio_manager
        
#         # 2. 상태 관리
#         self.state = ConversationState.IDLE
#         self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
#         # 3. 제어 플래그
#         self.is_running = False
#         self.is_processing = False
        
#         # 4. STT 결과 큐
#         self.stt_queue = asyncio.Queue(maxsize=10)
        
#         # 5. 콜백 함수
#         self.callbacks = {'on_state_change': None, 'on_ai_response': None}
        
#         # 6. 성능 통계 (기존 로직 유지)
#         self.performance_stats = {
#             'conversation_start_time': None, 'total_turns': 0, 'avg_response_time': 0.0,
#         }
#         self.response_times = []
#         self.max_response_times = 50

#         # 7. 침묵 감지 및 오디오 모니터링 (기존 로직 유지)
#         self.last_activity_time = time.time()
#         self.silence_timeout = settings.SILENCE_TIMEOUT

#     def _on_final_transcript(self, text: str):
#         """SttService로부터 최종 텍스트를 받아 큐에 추가하는 콜백"""
#         if text:
#             logger.info(f"STT 최종 결과: {text}")
#             try:
#                 # 비동기 루프에 안전하게 데이터 전달
#                 asyncio.get_running_loop().call_soon_threadsafe(self.stt_queue.put_nowait, text)
#                 self.last_activity_time = time.time()
#             except Exception as e:
#                 logger.error(f"STT 결과를 큐에 추가하는 중 오류: {e}")

#     async def start(self):
#         """대화 시스템 시작"""
#         if not await self._initialize():
#             logger.error("초기화 실패, 대화 시스템을 시작할 수 없습니다.")
#             return

#         self.is_running = True
#         self.performance_stats['conversation_start_time'] = datetime.now()
#         logger.info("대화 시스템 시작됨. 종료하려면 Ctrl+C를 누르세요.")

#         await self._speak("안녕하세요! 보이스피싱 AI 대응 시스템입니다. 도움이 필요하시면 말씀해주세요.")
        
#         main_loop_task = asyncio.create_task(self._main_loop())
#         await main_loop_task

#     async def _initialize(self) -> bool:
#         """필수 서비스 초기화"""
#         try:
#             # 1. STT 서비스 시작 (백그라운드 스레드)
#             self.stt_service.start()
#             logger.info("✅ STT 서비스 시작됨")

#             # 2. 오디오 출력 초기화
#             self.audio_manager.initialize_output()
#             logger.info("✅ 오디오 출력 초기화 완료")
            
#             # 3. CoordinatorAgent 시작 (비동기 루프)
#             await self.coordinator.start()
#             logger.info("✅ Coordinator 에이전트 시작됨")

#             return True
#         except Exception as e:
#             logger.critical(f"초기화 중 심각한 오류 발생: {e}", exc_info=True)
#             return False

#     async def _main_loop(self):
#         """메인 이벤트 루프"""
#         while self.is_running:
#             try:
#                 user_input = await asyncio.wait_for(self.stt_queue.get(), timeout=1.0)
#                 if user_input and not self.is_processing:
#                     await self._process_user_input(user_input)
#             except asyncio.TimeoutError:
#                 # 타임아웃 시 침묵 감지
#                 if self.state == ConversationState.LISTENING and (time.time() - self.last_activity_time > self.silence_timeout):
#                     logger.info(f"{self.silence_timeout}초 이상 침묵 감지됨.")
#                     self.last_activity_time = time.time() # 타임아웃 반복 방지
#                     await self._process_user_input("... (침묵) ...")
#             except Exception as e:
#                 logger.error(f"메인 루프 오류: {e}", exc_info=True)
#                 await asyncio.sleep(1)

#     async def _process_user_input(self, text: str):
#         """사용자 입력을 받아 에이전트 시스템에 전달하고 응답을 처리"""
#         start_time = time.time()
#         self.is_processing = True
#         self._set_state(ConversationState.PROCESSING)
        
#         logger.info(f"👤 사용자: {text}")

#         try:
#             # CoordinatorAgent에게 작업 제출 (LangGraph 호출을 대체)
#             task_id = await self.coordinator.submit_task(
#                 task_type=TaskType.DETECTION,  # 초기 탐지부터 시작
#                 data={"text": text, "context": {"session_id": self.session_id}},
#                 priority=TaskPriority.HIGH
#             )
#             logger.info(f"Coordinator에게 작업 제출: {task_id}")

#             # 작업 완료 대기 및 결과 처리
#             response_text = await self._wait_for_task_completion(task_id)
            
#             if response_text:
#                 await self._speak(response_text)
#             else:
#                 await self._speak("분석 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.")
            
#             # 성능 통계 업데이트
#             processing_time = time.time() - start_time
#             self._update_performance_stats(processing_time)

#         except Exception as e:
#             logger.error(f"입력 처리 중 오류 발생: {e}", exc_info=True)
#             await self._speak("죄송합니다. 내부 오류가 발생했습니다.")
#         finally:
#             self.is_processing = False
#             self._set_state(ConversationState.LISTENING)
#             self.last_activity_time = time.time()

#     async def _wait_for_task_completion(self, task_id: str) -> Optional[str]:
#         """Coordinator 작업이 완료될 때까지 대기하고 최종 응답을 반환"""
#         for _ in range(30):  # 최대 30초 대기
#             task_status = self.coordinator.get_task_status(task_id)
#             if task_status and task_status["status"] in ["completed", "failed"]:
#                 if task_status["status"] == "failed":
#                     logger.error(f"작업 실패: {task_id}, 오류: {task_status['error']}")
#                     return "분석 중 내부 오류가 발생했습니다."
                
#                 final_result = task_status.get("result", {})
#                 # ResponseAgent의 결과 형식에 맞춰 메시지 추출
#                 user_message = final_result.get("user_message", "분석이 완료되었지만, 전달할 메시지가 없습니다.")
#                 logger.info(f"AI 응답 생성: {user_message[:100]}...")
#                 return user_message
            
#             await asyncio.sleep(1)
        
#         logger.warning(f"작업 {task_id} 시간 초과")
#         return "분석 시간이 초과되었습니다. 다시 한 번 말씀해주시겠어요?"

#     async def _speak(self, text: str):
#         """TTS를 통해 음성 출력"""
#         if not text: return

#         self._set_state(ConversationState.SPEAKING)
#         if self.callbacks.get('on_ai_response'):
#             self.callbacks['on_ai_response'](text)
            
#         try:
#             audio_stream = self.tts_service.text_to_speech_stream(text)
#             await self.audio_manager.play_audio_stream(audio_stream)
#         except Exception as e:
#             logger.error(f"TTS 음성 출력 중 오류 발생: {e}")
        
#         self.last_activity_time = time.time()
#         self._set_state(ConversationState.LISTENING)
        
#     def _set_state(self, new_state: ConversationState):
#         """상태 변경 및 콜백 호출"""
#         if self.state != new_state:
#             old_state = self.state
#             self.state = new_state
#             logger.info(f"상태 변경: {old_state.value} -> {new_state.value}")
#             if self.callbacks.get('on_state_change'):
#                 self.callbacks['on_state_change'](old_state, new_state)

#     def _update_performance_stats(self, processing_time: float):
#         """성능 통계 업데이트"""
#         self.response_times.append(processing_time)
#         if len(self.response_times) > self.max_response_times:
#             self.response_times.pop(0)
        
#         if self.response_times:
#             self.performance_stats['avg_response_time'] = sum(self.response_times) / len(self.response_times)
        
#         self.performance_stats['total_turns'] += 1
#         logger.debug(f"응답 시간: {processing_time:.3f}초, 평균: {self.performance_stats['avg_response_time']:.3f}초")

#     async def cleanup(self):
#         """리소스 정리"""
#         self.is_running = False
#         self.stt_service.stop()
#         await self.coordinator.stop()
#         self.audio_manager.cleanup()
#         logger.info("대화 관리자 정리 완료.")


"""
간소화된 대화 관리자 - 실제 작동 가능한 버전
복잡한 에이전트 시스템 대신 직접적인 Gemini 호출
"""

import asyncio
import logging
import threading
import time
import queue
from datetime import datetime
import numpy as np
from typing import Optional, Dict, Any, Callable
from enum import Enum

# 핵심 서비스들만 import
from services.tts_service import tts_service
from services.audio_manager import audio_manager
from core.llm_manager import llm_manager
from config.settings import settings

logger = logging.getLogger(__name__)

# 간단한 STT 서비스 (ReturnZero 대신 간단한 구현)
try:
    from .stt_service import SttService
except ImportError:
    logger.warning("SttService를 찾을 수 없습니다. 더미 구현을 사용합니다.")
    
    class SttService:
        """더미 STT 서비스 (테스트용)"""
        def __init__(self, client_id: str, client_secret: str, callback: Callable):
            self.callback = callback
            self.is_running = False
            
        def start(self):
            logger.info("더미 STT 서비스 시작 (실제 마이크 입력 없음)")
            self.is_running = True
            
            # 테스트용 입력을 시뮬레이션
            def simulate_input():
                import time
                time.sleep(3)
                if self.is_running:
                    self.callback("안녕하세요, 테스트 입력입니다.")
                    
            threading.Thread(target=simulate_input, daemon=True).start()
            
        def stop(self):
            self.is_running = False

class ConversationState(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    ERROR = "error"

class HighPerformanceConversationManager:
    """
    고성능 대화 관리자 - 실제 작동 버전
    복잡한 멀티 에이전트 대신 직접 Gemini 호출
    """
    
    def __init__(self, client_id: str, client_secret: str):
        # 1. 핵심 컴포넌트 초기화
        self.stt_service = SttService(client_id, client_secret, self._on_final_transcript)
        self.llm_manager = llm_manager  # Gemini 전용 매니저
        self.tts_service = tts_service
        self.audio_manager = audio_manager
        
        # 2. 상태 관리
        self.state = ConversationState.IDLE
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 3. 제어 플래그
        self.is_running = False
        self.is_processing = False
        
        # 4. STT 결과 큐
        self.stt_queue = asyncio.Queue(maxsize=10)
        
        # 5. 콜백 함수들
        self.callbacks = {
            'on_state_change': None,
            'on_ai_response': None,
            'on_user_speech': None
        }
        
        # 6. 성능 통계
        self.performance_stats = {
            'conversation_start_time': None,
            'total_turns': 0,
            'avg_response_time': 0.0,
            'successful_detections': 0,
            'total_detections': 0,
            'tts_success_rate': 1.0
        }
        self.response_times = []
        self.max_response_times = 50

        # 7. 침묵 감지
        self.last_activity_time = time.time()
        self.silence_timeout = settings.SILENCE_TIMEOUT
        
        logger.info("고성능 대화 관리자 초기화 완료")

    def set_callbacks(self, on_user_speech=None, on_ai_response=None, on_state_change=None):
        """콜백 함수 설정"""
        if on_user_speech:
            self.callbacks['on_user_speech'] = on_user_speech
        if on_ai_response:
            self.callbacks['on_ai_response'] = on_ai_response
        if on_state_change:
            self.callbacks['on_state_change'] = on_state_change

    def _on_final_transcript(self, text: str):
        """STT 결과 콜백"""
        if text and text.strip():
            logger.info(f"STT 결과: {text}")
            try:
                # 비동기 루프에 안전하게 전달
                asyncio.get_running_loop().call_soon_threadsafe(
                    self.stt_queue.put_nowait, text.strip()
                )
                self.last_activity_time = time.time()
                
                # 사용자 입력 콜백 호출
                if self.callbacks['on_user_speech']:
                    self.callbacks['on_user_speech'](text)
                    
            except Exception as e:
                logger.error(f"STT 결과 처리 오류: {e}")

    async def start_conversation(self):
        """대화 시작"""
        if not await self._initialize_services():
            logger.error("서비스 초기화 실패")
            return

        self.is_running = True
        self.performance_stats['conversation_start_time'] = datetime.now()
        
        logger.info("🚀 고성능 대화 시스템 시작")
        
        # 환영 메시지
        await self._speak("안녕하세요! VoiceGuard AI 보이스피싱 대응 시스템입니다. 의심스러운 통화 내용을 말씀해주시면 분석해드리겠습니다.")
        
        # 메인 루프 시작
        await self._main_conversation_loop()

    async def _initialize_services(self) -> bool:
        """서비스 초기화"""
        try:
            # 1. Gemini LLM 연결 테스트
            health_status = await self.llm_manager.health_check()
            if not any(health_status.values()):
                logger.error("Gemini 모델 연결 실패")
                return False
            
            logger.info(f"✅ Gemini 모델 상태: {health_status}")
            
            # 2. STT 서비스 시작
            self.stt_service.start()
            logger.info("✅ STT 서비스 시작")

            # 3. 오디오 출력 초기화
            if self.audio_manager.initialize_output():
                logger.info("✅ 오디오 출력 초기화 완료")
            else:
                logger.warning("⚠️ 오디오 출력 초기화 실패 (계속 진행)")
            
            # 4. TTS 연결 테스트
            if await self.tts_service.test_connection():
                logger.info("✅ TTS 서비스 연결 완료")
            else:
                logger.warning("⚠️ TTS 서비스 연결 실패 (계속 진행)")

            return True
            
        except Exception as e:
            logger.error(f"서비스 초기화 중 오류: {e}")
            return False

    async def _main_conversation_loop(self):
        """메인 대화 루프"""
        self._set_state(ConversationState.LISTENING)
        
        while self.is_running:
            try:
                # STT 결과 대기 (타임아웃 1초)
                user_input = await asyncio.wait_for(
                    self.stt_queue.get(), 
                    timeout=1.0
                )
                
                if user_input and not self.is_processing:
                    await self._process_user_input(user_input)
                    
            except asyncio.TimeoutError:
                # 침묵 감지 처리
                current_time = time.time()
                if (self.state == ConversationState.LISTENING and 
                    current_time - self.last_activity_time > self.silence_timeout):
                    
                    logger.info(f"{self.silence_timeout}초 침묵 감지")
                    await self._handle_silence_timeout()
                    
            except Exception as e:
                logger.error(f"대화 루프 오류: {e}")
                await asyncio.sleep(1)

    async def _handle_silence_timeout(self):
        """침묵 타임아웃 처리"""
        self.last_activity_time = time.time()  # 중복 처리 방지
        
        await self._speak("대화가 없으신 것 같네요. 도움이 필요하시면 언제든 말씀해주세요.")

    async def _process_user_input(self, text: str):
        """사용자 입력 처리 - 간소화된 버전"""
        start_time = time.time()
        self.is_processing = True
        self._set_state(ConversationState.PROCESSING)
        
        logger.info(f"👤 사용자 입력 처리: {text[:50]}...")

        try:
            # 1. Gemini로 직접 분석 (에이전트 시스템 우회)
            analysis_result = await self.llm_manager.analyze_scam_risk(
                text=text,
                context={
                    "session_id": self.session_id,
                    "call_duration": int(time.time() - self.performance_stats['conversation_start_time'].timestamp()),
                    "caller_info": "시뮬레이션"
                }
            )
            
            # 2. 분석 결과를 기반으로 응답 생성
            response_text = await self._generate_response_from_analysis(analysis_result)
            
            # 3. 음성 출력
            if response_text:
                await self._speak(response_text)
            else:
                await self._speak("분석 중 문제가 발생했습니다. 다시 말씀해주시겠어요?")
            
            # 4. 통계 업데이트
            processing_time = time.time() - start_time
            self._update_performance_stats(processing_time, analysis_result)

        except Exception as e:
            logger.error(f"입력 처리 중 오류: {e}")
            await self._speak("죄송합니다. 시스템 오류가 발생했습니다. 다시 시도해주세요.")
        
        finally:
            self.is_processing = False
            self._set_state(ConversationState.LISTENING)
            self.last_activity_time = time.time()

    async def _generate_response_from_analysis(self, analysis_result) -> str:
        """분석 결과를 바탕으로 응답 생성"""
        
        try:
            risk_level = analysis_result.risk_level
            metadata = analysis_result.metadata
            risk_score = metadata.get("risk_score", 0.0)
            scam_type = metadata.get("scam_type", "unknown")
            key_indicators = metadata.get("key_indicators", [])
            immediate_action = metadata.get("immediate_action", False)
            
            logger.info(f"분석 결과 - 위험도: {risk_level.value}, 점수: {risk_score:.2f}, 유형: {scam_type}")
            
            # 위험도별 맞춤 응답 생성
            if risk_level.value == "매우 위험":
                response = f"""🚨 매우 위험한 보이스피싱이 감지되었습니다!

분석 결과:
- 사기 유형: {scam_type}
- 위험도: {risk_score:.1%}
- 주요 위험 요소: {', '.join(key_indicators[:3])}

즉시 해야 할 일:
1. 지금 당장 통화를 끊으세요
2. 절대 개인정보나 금융정보를 제공하지 마세요
3. 112 또는 금융감독원(1332)에 신고하세요

이런 종류의 사기는 매우 정교하니 절대 속지 마세요!"""

            elif risk_level.value == "위험":
                response = f"""⚠️ 보이스피싱 위험이 높게 감지되었습니다.

분석 결과:
- 사기 유형: {scam_type}
- 위험도: {risk_score:.1%}
- 의심 요소: {', '.join(key_indicators[:3])}

권장 사항:
1. 통화를 중단하고 직접 해당 기관에 확인하세요
2. 급하게 결정하지 마세요
3. 가족이나 지인과 상의해보세요

정말 급한 일이라면 공식 홈페이지에서 연락처를 찾아 직접 전화하세요."""

            elif risk_level.value == "주의":
                response = f"""🔍 주의가 필요한 통화로 분석되었습니다.

분석 결과:
- 추정 유형: {scam_type}
- 위험도: {risk_score:.1%}
- 주의 요소: {', '.join(key_indicators[:2])}

확인사항:
1. 발신번호가 공식 번호인지 확인하세요
2. 요구하는 정보가 합리적인지 생각해보세요
3. 의심스럽다면 직접 기관에 문의하세요

혹시 모르니 조심하는 것이 좋겠습니다."""

            else:  # 낮음
                response = f"""✅ 상대적으로 안전한 통화로 분석되었습니다.

분석 결과:
- 위험도: {risk_score:.1%}
- 특이사항: {', '.join(key_indicators) if key_indicators else '없음'}

하지만 여전히 주의사항:
1. 개인정보는 신중하게 제공하세요
2. 금융 관련 요청이 있다면 다시 한번 확인하세요
3. 이상하다 싶으면 언제든 문의하세요

안전한 통화 되세요!"""

            return response
            
        except Exception as e:
            logger.error(f"응답 생성 오류: {e}")
            return "분석은 완료되었지만 응답 생성 중 오류가 발생했습니다. 의심스러운 통화라면 즉시 끊으시기 바랍니다."

    async def _speak(self, text: str):
        """TTS를 통한 음성 출력"""
        if not text:
            return

        self._set_state(ConversationState.SPEAKING)
        
        # AI 응답 콜백 호출
        if self.callbacks['on_ai_response']:
            self.callbacks['on_ai_response'](text)
            
        try:
            # TTS 스트리밍
            audio_stream = self.tts_service.text_to_speech_stream(text)
            await self.audio_manager.play_audio_stream(audio_stream)
            
            # TTS 성공 통계
            self.performance_stats['tts_success_rate'] = (
                self.performance_stats['tts_success_rate'] * 0.9 + 0.1
            )
            
        except Exception as e:
            logger.error(f"TTS 출력 오류: {e}")
            # TTS 실패 통계
            self.performance_stats['tts_success_rate'] = (
                self.performance_stats['tts_success_rate'] * 0.9
            )
        
        self.last_activity_time = time.time()
        self._set_state(ConversationState.LISTENING)
        
    def _set_state(self, new_state: ConversationState):
        """상태 변경 및 콜백"""
        if self.state != new_state:
            old_state = self.state
            self.state = new_state
            
            # 상태 변경 콜백 호출
            if self.callbacks['on_state_change']:
                self.callbacks['on_state_change'](old_state, new_state)

    def _update_performance_stats(self, processing_time: float, analysis_result):
        """성능 통계 업데이트"""
        
        # 응답 시간 추가
        self.response_times.append(processing_time)
        if len(self.response_times) > self.max_response_times:
            self.response_times.pop(0)
        
        # 평균 응답 시간 계산
        if self.response_times:
            self.performance_stats['avg_response_time'] = sum(self.response_times) / len(self.response_times)
        
        # 턴 수 증가
        self.performance_stats['total_turns'] += 1
        
        # 탐지 통계
        self.performance_stats['total_detections'] += 1
        if analysis_result.risk_level.value in ["위험", "매우 위험"]:
            self.performance_stats['successful_detections'] += 1
        
        logger.debug(f"응답 시간: {processing_time:.3f}초, 평균: {self.performance_stats['avg_response_time']:.3f}초")

    def get_conversation_status(self) -> Dict[str, Any]:
        """대화 상태 조회"""
        
        runtime = 0
        if self.performance_stats['conversation_start_time']:
            runtime = (datetime.now() - self.performance_stats['conversation_start_time']).total_seconds()
        
        return {
            "state": self.state.value,
            "session_id": self.session_id,
            "is_running": self.is_running,
            "is_processing": self.is_processing,
            "runtime_seconds": runtime,
            "queue_size": self.stt_queue.qsize(),
            **self.performance_stats
        }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """성능 메트릭 조회"""
        
        # LLM 통계 가져오기
        llm_stats = self.llm_manager.get_performance_stats()
        
        # 오디오 통계 가져오기
        audio_stats = self.audio_manager.get_performance_stats() if hasattr(self.audio_manager, 'get_performance_stats') else {}
        
        # TTS 통계 가져오기
        tts_stats = self.tts_service.get_performance_stats() if hasattr(self.tts_service, 'get_performance_stats') else {}
        
        return {
            "conversation": self.get_conversation_status(),
            "llm": llm_stats,
            "audio": audio_stats,
            "tts": tts_stats,
            "overall_health": {
                "all_systems_ok": self.is_running and not self.is_processing,
                "last_activity": self.last_activity_time,
                "silence_timeout": self.silence_timeout
            }
        }

    def get_audio_status(self) -> Dict[str, Any]:
        """오디오 상태 조회 (디버깅용)"""
        
        audio_status = {
            "audio_manager_initialized": hasattr(self.audio_manager, 'is_initialized'),
            "is_playing": self.audio_manager.is_audio_playing() if hasattr(self.audio_manager, 'is_audio_playing') else False,
            "tts_enabled": self.tts_service.is_enabled if hasattr(self.tts_service, 'is_enabled') else True,
            "stt_running": self.stt_service.is_running if hasattr(self.stt_service, 'is_running') else False
        }
        
        return audio_status

    async def cleanup(self):
        """리소스 정리"""
        logger.info("🧹 대화 관리자 정리 시작...")
        
        try:
            self.is_running = False
            
            # STT 서비스 정리
            if hasattr(self.stt_service, 'stop'):
                self.stt_service.stop()
            
            # 오디오 매니저 정리
            if hasattr(self.audio_manager, 'cleanup'):
                self.audio_manager.cleanup()
            
            # 큐 정리
            while not self.stt_queue.empty():
                try:
                    self.stt_queue.get_nowait()
                except:
                    break
            
            # 최종 통계 출력
            final_stats = self.get_conversation_status()
            logger.info(f"📊 최종 통계: 총 {final_stats['total_turns']}턴, "
                       f"평균 응답시간: {final_stats['avg_response_time']:.3f}초")
            
            logger.info("✅ 대화 관리자 정리 완료")
            
        except Exception as e:
            logger.error(f"정리 중 오류: {e}")

# 하위 호환성을 위한 별칭
ConversationManager = HighPerformanceConversationManager