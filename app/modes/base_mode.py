"""
VoiceGuard AI - 기본 모드 클래스
모든 운영 모드의 기본 인터페이스
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

class ModeState(Enum):
    """모드 상태"""
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"

class BaseMode(ABC):
    """모든 운영 모드의 기본 클래스"""
    
    def __init__(self, llm_manager, audio_manager, tts_service, session_id: str):
        self.llm_manager = llm_manager
        self.audio_manager = audio_manager
        self.tts_service = tts_service
        self.session_id = session_id
        
        # 모드 상태
        self.state = ModeState.INITIALIZING
        self.is_running = False
        self.start_time = None
        
        # 성능 통계
        self.stats = {
            'total_interactions': 0,
            'successful_interactions': 0,
            'avg_response_time': 0.0,
            'mode_specific_stats': {}
        }
        
        # 콜백 함수들
        self.callbacks = {}
        
        # 설정
        self.config = self._load_mode_config()
        
        logger.info(f"{self.__class__.__name__} 모드 초기화")
    
    @property
    @abstractmethod
    def mode_name(self) -> str:
        """모드 이름 (구현 필수)"""
        pass
    
    @property
    @abstractmethod
    def mode_description(self) -> str:
        """모드 설명 (구현 필수)"""
        pass
    
    @abstractmethod
    def _load_mode_config(self) -> Dict[str, Any]:
        """모드별 설정 로드 (구현 필수)"""
        pass
    
    @abstractmethod
    async def _initialize_mode(self) -> bool:
        """모드별 초기화 로직 (구현 필수)"""
        pass
    
    @abstractmethod
    async def _run_mode_logic(self):
        """모드별 메인 로직 (구현 필수)"""
        pass
    
    async def run(self):
        """모드 실행 (공통 로직)"""
        
        self.start_time = datetime.now()
        self.state = ModeState.INITIALIZING
        
        try:
            logger.info(f"🎯 {self.mode_name} 모드 시작")
            print(f"\n🎯 {self.mode_description}")
            print("-" * 50)
            
            # 1. 모드별 초기화
            if not await self._initialize_mode():
                raise RuntimeError("모드 초기화 실패")
            
            self.state = ModeState.READY
            
            # 2. 모드 시작 안내
            await self._announce_mode_start()
            
            # 3. 메인 로직 실행
            self.state = ModeState.RUNNING
            self.is_running = True
            
            await self._run_mode_logic()
            
        except KeyboardInterrupt:
            logger.info("사용자에 의해 모드 중단됨")
            print("\n🛑 모드를 중단합니다...")
        except Exception as e:
            logger.error(f"모드 실행 중 오류: {e}")
            self.state = ModeState.ERROR
            raise
        finally:
            await self._stop()
    
    async def _announce_mode_start(self):
        """모드 시작 안내"""
        
        start_message = f"{self.mode_name} 모드를 시작합니다."
        print(f"🚀 {start_message}")
        
        # TTS 안내 (선택사항)
        try:
            await self._speak(start_message)
        except:
            pass  # TTS 실패해도 계속 진행
    
    async def _speak(self, text: str):
        """TTS 음성 출력"""
        
        try:
            audio_stream = self.tts_service.text_to_speech_stream(text)
            await self.audio_manager.play_audio_stream(audio_stream)
        except Exception as e:
            logger.warning(f"TTS 출력 실패: {e}")
    
    def stop(self):
        """모드 중지"""
        
        logger.info(f"{self.mode_name} 모드 중지 요청")
        self.state = ModeState.STOPPING
        self.is_running = False
    
    async def _stop(self):
        """모드 중지 처리"""
        
        try:
            self.state = ModeState.STOPPING
            self.is_running = False
            
            # 모드별 정리 작업
            await self._cleanup_mode()
            
            # 최종 통계 출력
            await self._print_final_stats()
            
            self.state = ModeState.STOPPED
            logger.info(f"{self.mode_name} 모드 종료 완료")
            
        except Exception as e:
            logger.error(f"모드 중지 중 오류: {e}")
    
    async def _cleanup_mode(self):
        """모드별 정리 작업 (서브클래스에서 오버라이드)"""
        pass
    
    async def _print_final_stats(self):
        """최종 통계 출력"""
        
        runtime = 0
        if self.start_time:
            runtime = (datetime.now() - self.start_time).total_seconds()
        
        print(f"\n📊 {self.mode_name} 통계:")
        print(f"   실행 시간: {runtime/60:.1f}분")
        print(f"   총 상호작용: {self.stats['total_interactions']}")
        print(f"   성공률: {self._calculate_success_rate():.1%}")
        
        # 모드별 추가 통계
        if self.stats['mode_specific_stats']:
            print("   모드별 통계:")
            for key, value in self.stats['mode_specific_stats'].items():
                print(f"     {key}: {value}")
    
    def _calculate_success_rate(self) -> float:
        """성공률 계산"""
        
        if self.stats['total_interactions'] == 0:
            return 0.0
        
        return self.stats['successful_interactions'] / self.stats['total_interactions']
    
    def _update_stats(self, success: bool = True, **kwargs):
        """통계 업데이트"""
        
        self.stats['total_interactions'] += 1
        
        if success:
            self.stats['successful_interactions'] += 1
        
        # 모드별 통계 업데이트
        for key, value in kwargs.items():
            self.stats['mode_specific_stats'][key] = value
    
    def set_callback(self, event: str, callback: Callable):
        """콜백 함수 설정"""
        self.callbacks[event] = callback
    
    def _trigger_callback(self, event: str, *args, **kwargs):
        """콜백 함수 호출"""
        
        if event in self.callbacks:
            try:
                self.callbacks[event](*args, **kwargs)
            except Exception as e:
                logger.error(f"콜백 실행 오류 ({event}): {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """모드 상태 조회"""
        
        runtime = 0
        if self.start_time:
            runtime = (datetime.now() - self.start_time).total_seconds()
        
        return {
            "mode_name": self.mode_name,
            "state": self.state.value,
            "is_running": self.is_running,
            "session_id": self.session_id,
            "runtime_seconds": runtime,
            "stats": self.stats.copy(),
            "config": self.config.copy()
        }
    
    async def pause(self):
        """모드 일시정지"""
        if self.state == ModeState.RUNNING:
            self.state = ModeState.PAUSED
            logger.info(f"{self.mode_name} 모드 일시정지")
    
    async def resume(self):
        """모드 재개"""
        if self.state == ModeState.PAUSED:
            self.state = ModeState.RUNNING
            logger.info(f"{self.mode_name} 모드 재개")
    
    def is_active(self) -> bool:
        """모드 활성 상태 확인"""
        return self.state in [ModeState.RUNNING, ModeState.READY]