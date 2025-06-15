"""
VoiceGuard AI - 간단한 STT 서비스
복잡한 gRPC 대신 텍스트 입력으로 대체
"""

import logging
import threading
import time
from typing import Callable

logger = logging.getLogger(__name__)

class SimpleSttService:
    """간단한 STT 서비스 (텍스트 입력 기반)"""
    
    def __init__(self, client_id: str, client_secret: str, transcript_callback: Callable):
        self.client_id = client_id
        self.client_secret = client_secret
        self.transcript_callback = transcript_callback
        self.is_running = False
        self.thread = None
        
        logger.info("간단한 STT 서비스 초기화")
    
    def start(self):
        """STT 서비스 시작"""
        if self.is_running:
            logger.warning("STT 서비스가 이미 실행 중입니다")
            return
        
        self.is_running = True
        self.thread = threading.Thread(target=self._input_worker, daemon=True)
        self.thread.start()
        
        logger.info("간단한 STT 서비스 시작됨 (텍스트 입력 모드)")
        print("🎤 음성 대신 텍스트로 입력하세요.")
        print("💡 '종료'를 입력하면 분석을 마칩니다.")
    
    def _input_worker(self):
        """텍스트 입력 워커"""
        while self.is_running:
            try:
                # 논블로킹 입력 시뮬레이션
                time.sleep(0.1)
                # 실제로는 여기서 텍스트 입력을 받지 않고
                # detection_mode에서 직접 처리하도록 함
                
            except Exception as e:
                logger.error(f"입력 워커 오류: {e}")
                break
    
    def stop(self):
        """STT 서비스 중지"""
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1)
        logger.info("간단한 STT 서비스 중지됨")
    
    def simulate_input(self, text: str):
        """입력 시뮬레이션 (테스트용)"""
        if self.is_running and text:
            self.transcript_callback(text)

# 호환성을 위한 별칭
SttService = SimpleSttService