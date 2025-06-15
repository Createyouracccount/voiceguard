# services/conversation_manager.py

import asyncio
import logging
import threading
import time
import queue
from datetime import datetime
import numpy as np
from typing import Optional, Dict, Any, Callable
from enum import Enum

# AI 에이전트 시스템 import
from agents import coordinator_agent, TaskType, TaskPriority

# 기존 서비스 및 설정 import
from services.tts_service import tts_service
from services.audio_manager import audio_manager
from config.settings import settings

logger = logging.getLogger(__name__)

# 이전 코드와 호환성을 위해 SttService를 그대로 사용합니다.
# 만약 stream_stt.py를 직접 사용하신다면 이 부분을 맞게 수정해야 합니다.
try:
    from .stt_service import SttService
except ImportError:
    logger.error("SttService를 import할 수 없습니다. services/stt_service.py 파일이 올바르게 구성되었는지 확인하세요.")
    # 임시 클래스로 대체하여 프로그램이 시작될 수 있도록 함
    class SttService:
        def __init__(self, *args, **kwargs): pass
        def start(self): pass
        def stop(self): pass


class ConversationState(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    ERROR = "error"

class ConversationManager:
    """
    고성능 대화 관리자와 멀티 에이전트 시스템을 통합한 최종 버전.
    - 기존의 정교한 오디오/침묵 제어 기능 유지
    - 분석 로직을 LangGraph에서 CoordinatorAgent로 교체
    """
    
    def __init__(self, client_id: str, client_secret: str):
        # 1. 컴포넌트 초기화
        self.stt_service = SttService(client_id, client_secret, self._on_final_transcript)
        self.coordinator = coordinator_agent # LangGraph 대신 CoordinatorAgent 사용
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
        
        # 5. 콜백 함수
        self.callbacks = {'on_state_change': None, 'on_ai_response': None}
        
        # 6. 성능 통계 (기존 로직 유지)
        self.performance_stats = {
            'conversation_start_time': None, 'total_turns': 0, 'avg_response_time': 0.0,
        }
        self.response_times = []
        self.max_response_times = 50

        # 7. 침묵 감지 및 오디오 모니터링 (기존 로직 유지)
        self.last_activity_time = time.time()
        self.silence_timeout = settings.SILENCE_TIMEOUT

    def _on_final_transcript(self, text: str):
        """SttService로부터 최종 텍스트를 받아 큐에 추가하는 콜백"""
        if text:
            logger.info(f"STT 최종 결과: {text}")
            try:
                # 비동기 루프에 안전하게 데이터 전달
                asyncio.get_running_loop().call_soon_threadsafe(self.stt_queue.put_nowait, text)
                self.last_activity_time = time.time()
            except Exception as e:
                logger.error(f"STT 결과를 큐에 추가하는 중 오류: {e}")

    async def start(self):
        """대화 시스템 시작"""
        if not await self._initialize():
            logger.error("초기화 실패, 대화 시스템을 시작할 수 없습니다.")
            return

        self.is_running = True
        self.performance_stats['conversation_start_time'] = datetime.now()
        logger.info("대화 시스템 시작됨. 종료하려면 Ctrl+C를 누르세요.")

        await self._speak("안녕하세요! 보이스피싱 AI 대응 시스템입니다. 도움이 필요하시면 말씀해주세요.")
        
        main_loop_task = asyncio.create_task(self._main_loop())
        await main_loop_task

    async def _initialize(self) -> bool:
        """필수 서비스 초기화"""
        try:
            # 1. STT 서비스 시작 (백그라운드 스레드)
            self.stt_service.start()
            logger.info("✅ STT 서비스 시작됨")

            # 2. 오디오 출력 초기화
            self.audio_manager.initialize_output()
            logger.info("✅ 오디오 출력 초기화 완료")
            
            # 3. CoordinatorAgent 시작 (비동기 루프)
            await self.coordinator.start()
            logger.info("✅ Coordinator 에이전트 시작됨")

            return True
        except Exception as e:
            logger.critical(f"초기화 중 심각한 오류 발생: {e}", exc_info=True)
            return False

    async def _main_loop(self):
        """메인 이벤트 루프"""
        while self.is_running:
            try:
                user_input = await asyncio.wait_for(self.stt_queue.get(), timeout=1.0)
                if user_input and not self.is_processing:
                    await self._process_user_input(user_input)
            except asyncio.TimeoutError:
                # 타임아웃 시 침묵 감지
                if self.state == ConversationState.LISTENING and (time.time() - self.last_activity_time > self.silence_timeout):
                    logger.info(f"{self.silence_timeout}초 이상 침묵 감지됨.")
                    self.last_activity_time = time.time() # 타임아웃 반복 방지
                    await self._process_user_input("... (침묵) ...")
            except Exception as e:
                logger.error(f"메인 루프 오류: {e}", exc_info=True)
                await asyncio.sleep(1)

    async def _process_user_input(self, text: str):
        """사용자 입력을 받아 에이전트 시스템에 전달하고 응답을 처리"""
        start_time = time.time()
        self.is_processing = True
        self._set_state(ConversationState.PROCESSING)
        
        logger.info(f"👤 사용자: {text}")

        try:
            # CoordinatorAgent에게 작업 제출 (LangGraph 호출을 대체)
            task_id = await self.coordinator.submit_task(
                task_type=TaskType.DETECTION,  # 초기 탐지부터 시작
                data={"text": text, "context": {"session_id": self.session_id}},
                priority=TaskPriority.HIGH
            )
            logger.info(f"Coordinator에게 작업 제출: {task_id}")

            # 작업 완료 대기 및 결과 처리
            response_text = await self._wait_for_task_completion(task_id)
            
            if response_text:
                await self._speak(response_text)
            else:
                await self._speak("분석 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.")
            
            # 성능 통계 업데이트
            processing_time = time.time() - start_time
            self._update_performance_stats(processing_time)

        except Exception as e:
            logger.error(f"입력 처리 중 오류 발생: {e}", exc_info=True)
            await self._speak("죄송합니다. 내부 오류가 발생했습니다.")
        finally:
            self.is_processing = False
            self._set_state(ConversationState.LISTENING)
            self.last_activity_time = time.time()

    async def _wait_for_task_completion(self, task_id: str) -> Optional[str]:
        """Coordinator 작업이 완료될 때까지 대기하고 최종 응답을 반환"""
        for _ in range(30):  # 최대 30초 대기
            task_status = self.coordinator.get_task_status(task_id)
            if task_status and task_status["status"] in ["completed", "failed"]:
                if task_status["status"] == "failed":
                    logger.error(f"작업 실패: {task_id}, 오류: {task_status['error']}")
                    return "분석 중 내부 오류가 발생했습니다."
                
                final_result = task_status.get("result", {})
                # ResponseAgent의 결과 형식에 맞춰 메시지 추출
                user_message = final_result.get("user_message", "분석이 완료되었지만, 전달할 메시지가 없습니다.")
                logger.info(f"AI 응답 생성: {user_message[:100]}...")
                return user_message
            
            await asyncio.sleep(1)
        
        logger.warning(f"작업 {task_id} 시간 초과")
        return "분석 시간이 초과되었습니다. 다시 한 번 말씀해주시겠어요?"

    async def _speak(self, text: str):
        """TTS를 통해 음성 출력"""
        if not text: return

        self._set_state(ConversationState.SPEAKING)
        if self.callbacks.get('on_ai_response'):
            self.callbacks['on_ai_response'](text)
            
        try:
            audio_stream = self.tts_service.text_to_speech_stream(text)
            await self.audio_manager.play_audio_stream(audio_stream)
        except Exception as e:
            logger.error(f"TTS 음성 출력 중 오류 발생: {e}")
        
        self.last_activity_time = time.time()
        self._set_state(ConversationState.LISTENING)
        
    def _set_state(self, new_state: ConversationState):
        """상태 변경 및 콜백 호출"""
        if self.state != new_state:
            old_state = self.state
            self.state = new_state
            logger.info(f"상태 변경: {old_state.value} -> {new_state.value}")
            if self.callbacks.get('on_state_change'):
                self.callbacks['on_state_change'](old_state, new_state)

    def _update_performance_stats(self, processing_time: float):
        """성능 통계 업데이트"""
        self.response_times.append(processing_time)
        if len(self.response_times) > self.max_response_times:
            self.response_times.pop(0)
        
        if self.response_times:
            self.performance_stats['avg_response_time'] = sum(self.response_times) / len(self.response_times)
        
        self.performance_stats['total_turns'] += 1
        logger.debug(f"응답 시간: {processing_time:.3f}초, 평균: {self.performance_stats['avg_response_time']:.3f}초")

    async def cleanup(self):
        """리소스 정리"""
        self.is_running = False
        self.stt_service.stop()
        await self.coordinator.stop()
        self.audio_manager.cleanup()
        logger.info("대화 관리자 정리 완료.")