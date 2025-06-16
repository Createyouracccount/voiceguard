"""
VoiceGuard AI - 단순화된 명확한 시스템
실제 사용되는 AI 기능만 명확하게 구현
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional
from enum import Enum

from core.llm_manager import llm_manager
from services.tts_service import tts_service
from services.audio_manager import audio_manager

logger = logging.getLogger(__name__)

class SimplifiedConversationManager:
    """명확하고 단순한 AI 대화 관리자"""
    
    def __init__(self):
        # 핵심 AI 컴포넌트
        self.llm_manager = llm_manager  # Gemini 기반 AI
        self.tts_service = tts_service
        self.audio_manager = audio_manager
        
        # 상태 관리
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.is_running = False
        self.stt_queue = asyncio.Queue(maxsize=10)
        
        # AI 사용 현황 명확히 추적
        self.ai_usage = {
            'total_ai_calls': 0,
            'gemini_calls': 0,
            'successful_analyses': 0,
            'avg_ai_response_time': 0.0,
            'ai_cost_estimate': 0.0
        }
        
        # 더미 STT
        self.stt_service = self._create_test_stt()
        
        logger.info("🤖 Simplified AI System 초기화")
        logger.info("   AI Engine: Google Gemini (LLM Manager)")
        logger.info("   Audio: ElevenLabs TTS + PyAudio")
        logger.info("   Analysis: 직접 Gemini 호출")

    def _create_test_stt(self):
        """테스트용 STT"""
        class TestSTT:
            def __init__(self, callback):
                self.callback = callback
                self.is_running = False
            
            def start(self):
                self.is_running = True
                logger.info("🎤 테스트 STT 시작")
                
                # 테스트 시나리오
                test_scenarios = [
                    "안녕하세요, 금융감독원 조사과입니다.",
                    "고객님 계좌에서 의심거래가 발견되었습니다.",
                    "안전을 위해 임시 계좌로 이체해주세요.",
                    "지금 즉시 처리하지 않으면 계좌가 동결됩니다."
                ]
                
                async def simulate():
                    await asyncio.sleep(3)
                    for i, scenario in enumerate(test_scenarios):
                        if self.is_running:
                            logger.info(f"🎭 시나리오 {i+1}: {scenario}")
                            self.callback(scenario)
                            await asyncio.sleep(8)  # 8초 간격
                
                asyncio.create_task(simulate())
            
            def stop(self):
                self.is_running = False
        
        return TestSTT(self._on_speech_input)

    def _on_speech_input(self, text: str):
        """STT 입력 처리"""
        if text and text.strip():
            try:
                loop = asyncio.get_running_loop()
                loop.call_soon_threadsafe(self.stt_queue.put_nowait, text.strip())
            except Exception as e:
                logger.error(f"STT 처리 오류: {e}")

    async def start_conversation(self):
        """대화 시작"""
        
        logger.info("🚀 AI 시스템 초기화...")
        
        # 1. Gemini 연결 확인
        health_status = await self.llm_manager.health_check()
        active_models = [model for model, status in health_status.items() if status]
        
        if not active_models:
            logger.error("❌ Gemini 모델 연결 실패")
            return False
        
        logger.info(f"✅ Gemini 연결 성공: {active_models}")
        
        # 2. 오디오 시스템
        self.audio_manager.initialize_output()
        
        # 3. TTS 테스트
        tts_ok = await self.tts_service.test_connection()
        logger.info(f"🔊 TTS 상태: {'OK' if tts_ok else 'FAIL'}")
        
        # 4. STT 시작
        self.stt_service.start()
        
        self.is_running = True
        
        # 시작 메시지 (AI 사용 명시)
        await self._speak_with_ai_info(
            "안녕하세요! VoiceGuard AI입니다. "
            "저는 Google Gemini 인공지능을 사용하여 보이스피싱을 분석합니다. "
            "의심스러운 대화를 말씀해주시면 AI가 실시간으로 위험도를 평가해드립니다."
        )
        
        # 메인 루프
        await self._main_conversation_loop()

    async def _main_conversation_loop(self):
        """메인 대화 루프"""
        
        while self.is_running:
            try:
                # 사용자 입력 대기
                user_input = await asyncio.wait_for(
                    self.stt_queue.get(), 
                    timeout=2.0
                )
                
                if user_input:
                    await self._process_with_ai_analysis(user_input)
                    
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"대화 루프 오류: {e}")

    async def _process_with_ai_analysis(self, text: str):
        """AI 분석 처리 (명확한 단계별 로깅)"""
        
        ai_start_time = time.time()
        
        logger.info("=" * 60)
        logger.info(f"🤖 AI 분석 시작")
        logger.info(f"📝 입력 텍스트: {text}")
        logger.info(f"🧠 AI 엔진: Google Gemini (via LLM Manager)")
        
        try:
            # === AI 분석 실행 ===
            logger.info("⚡ Gemini API 호출 중...")
            
            analysis_result = await self.llm_manager.analyze_scam_risk(
                text=text,
                context={
                    "session_id": self.session_id,
                    "timestamp": datetime.now().isoformat(),
                    "analysis_mode": "real_time_detection"
                }
            )
            
            ai_duration = time.time() - ai_start_time
            
            # === AI 결과 로깅 ===
            logger.info(f"✅ AI 분석 완료 ({ai_duration:.2f}초)")
            logger.info(f"🎯 사용된 모델: {analysis_result.model_used}")
            logger.info(f"💰 예상 비용: ${analysis_result.cost_estimate:.4f}")
            logger.info(f"🔍 위험도: {analysis_result.metadata['risk_score']:.1%}")
            logger.info(f"📊 신뢰도: {analysis_result.confidence:.1%}")
            logger.info(f"⚠️ 사기 유형: {analysis_result.metadata.get('scam_type', 'unknown')}")
            
            # === 통계 업데이트 ===
            self._update_ai_usage_stats(ai_duration, analysis_result)
            
            # === 사용자 응답 생성 ===
            response_text = self._create_clear_ai_response(analysis_result, ai_duration)
            
            # === 음성 출력 ===
            await self._speak_with_ai_info(response_text)
            
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"❌ AI 분석 실패: {e}")
            await self._speak_with_ai_info(
                "죄송합니다. AI 분석 중 오류가 발생했습니다. "
                "Gemini 연결 상태를 확인하고 다시 시도해주세요."
            )

    def _create_clear_ai_response(self, analysis_result, processing_time: float) -> str:
        """명확한 AI 분석 결과 응답 생성"""
        
        metadata = analysis_result.metadata
        risk_score = metadata.get('risk_score', 0.0)
        scam_type = metadata.get('scam_type', 'unknown')
        key_indicators = metadata.get('key_indicators', [])
        
        # AI 사용 정보 명시
        response = f"""🤖 Google Gemini AI 분석 결과
(처리시간: {processing_time:.1f}초, 모델: {analysis_result.model_used})

📊 위험도 평가: {risk_score:.1%}
🎯 추정 사기 유형: {scam_type}
🔍 AI 신뢰도: {analysis_result.confidence:.1%}

💡 AI가 탐지한 위험 요소:"""

        # 탐지된 지표들
        for i, indicator in enumerate(key_indicators[:5], 1):
            response += f"\n   {i}. {indicator}"

        # 위험도별 권장사항
        if risk_score >= 0.8:
            response += """

🚨 AI 판정: 매우 높은 위험
• 즉시 통화 중단 권장
• 112 또는 1332 신고 필요
• 절대 개인정보 제공 금지"""

        elif risk_score >= 0.6:
            response += """

⚠️ AI 판정: 높은 위험
• 통화 중단하고 직접 확인 필요
• 상대방 신원 재확인 권장
• 급한 결정 피하시기 바랍니다"""

        elif risk_score >= 0.4:
            response += """

🔍 AI 판정: 주의 필요
• 상대방 요구사항 신중히 검토
• 개인정보 제공 전 재확인
• 의심스러우면 공식 경로 이용"""

        else:
            response += """

✅ AI 판정: 상대적으로 안전
• 현재까지는 위험 요소 적음
• 여전히 개인정보는 신중하게
• 이상한 요구 시 즉시 확인"""

        return response

    async def _speak_with_ai_info(self, text: str):
        """AI 사용 정보와 함께 TTS 출력"""
        
        try:
            # TTS 시작 로깅
            logger.info(f"🔊 TTS 시작: {text[:50]}...")
            
            tts_start = time.time()
            audio_stream = self.tts_service.text_to_speech_stream(text)
            await self.audio_manager.play_audio_stream(audio_stream)
            
            tts_duration = time.time() - tts_start
            logger.info(f"✅ TTS 완료 ({tts_duration:.1f}초)")
            
        except Exception as e:
            logger.error(f"TTS 오류: {e}")

    def _update_ai_usage_stats(self, processing_time: float, analysis_result):
        """AI 사용 통계 업데이트"""
        
        self.ai_usage['total_ai_calls'] += 1
        self.ai_usage['gemini_calls'] += 1
        self.ai_usage['successful_analyses'] += 1
        
        # 평균 응답 시간
        current_avg = self.ai_usage['avg_ai_response_time']
        call_count = self.ai_usage['total_ai_calls']
        self.ai_usage['avg_ai_response_time'] = (
            current_avg * (call_count - 1) + processing_time
        ) / call_count
        
        # 비용 누적
        self.ai_usage['ai_cost_estimate'] += analysis_result.cost_estimate
        
        # 실시간 통계 로깅
        logger.info(f"📈 AI 사용 통계: 총 {self.ai_usage['total_ai_calls']}회, "
                   f"평균 {self.ai_usage['avg_ai_response_time']:.2f}초, "
                   f"총 비용 ${self.ai_usage['ai_cost_estimate']:.4f}")

    def get_ai_usage_report(self) -> Dict[str, Any]:
        """AI 사용 현황 리포트"""
        
        # Gemini 모델 상태
        gemini_stats = self.llm_manager.get_performance_stats()
        
        return {
            "session_info": {
                "session_id": self.session_id,
                "is_running": self.is_running,
                "start_time": getattr(self, 'start_time', 'unknown')
            },
            "ai_usage": self.ai_usage.copy(),
            "gemini_manager": {
                "active_models": list(self.llm_manager.models.keys()),
                "total_calls": gemini_stats.get('total_calls', 0),
                "total_cost": gemini_stats.get('total_cost', 0.0),
                "remaining_budget": gemini_stats.get('remaining_budget', 0.0)
            },
            "system_components": {
                "llm_manager": "✅ Active (Gemini)",
                "tts_service": "✅ Active (ElevenLabs)",
                "audio_manager": "✅ Active (PyAudio)",
                "stt_service": "🧪 Test Mode"
            },
            "ai_effectiveness": {
                "success_rate": (
                    self.ai_usage['successful_analyses'] / 
                    max(1, self.ai_usage['total_ai_calls']) * 100
                ),
                "avg_response_time": self.ai_usage['avg_ai_response_time'],
                "cost_per_analysis": (
                    self.ai_usage['ai_cost_estimate'] / 
                    max(1, self.ai_usage['successful_analyses'])
                )
            }
        }

    async def show_ai_demo(self):
        """AI 시스템 데모"""
        
        logger.info("🎭 AI 시스템 데모 시작")
        
        await self._speak_with_ai_info(
            "AI 시스템 데모를 시작합니다. "
            "곧 보이스피싱 시나리오들이 자동으로 입력되어 "
            "Gemini AI의 실시간 분석을 확인할 수 있습니다."
        )
        
        # 데모 시나리오 실행
        demo_scenarios = [
            {
                "text": "안녕하세요, 금융감독원입니다. 고객님 계좌 점검이 필요합니다.",
                "expected_risk": "high",
                "description": "기관사칭 패턴"
            },
            {
                "text": "저금리 대출 가능합니다. 지금 앱만 설치하시면 됩니다.",
                "expected_risk": "medium",
                "description": "대출사기 + 악성앱"
            },
            {
                "text": "아들이 사고났어요! 병원비가 급해요!",
                "expected_risk": "critical",
                "description": "납치협박 패턴"
            }
        ]
        
        for i, scenario in enumerate(demo_scenarios, 1):
            await self._speak_with_ai_info(f"데모 시나리오 {i}: {scenario['description']}")
            await asyncio.sleep(2)
            
            logger.info(f"🎭 데모 {i}: {scenario['text']}")
            await self._process_with_ai_analysis(scenario['text'])
            
            await asyncio.sleep(3)
        
        # 최종 리포트
        report = self.get_ai_usage_report()
        await self._speak_with_ai_info(
            f"데모 완료! 총 {report['ai_usage']['total_ai_calls']}회 AI 분석, "
            f"평균 {report['ai_effectiveness']['avg_response_time']:.1f}초 소요되었습니다."
        )

    async def cleanup(self):
        """시스템 정리"""
        
        logger.info("🧹 시스템 정리 중...")
        
        self.is_running = False
        
        try:
            # STT 정리
            if hasattr(self.stt_service, 'stop'):
                self.stt_service.stop()
            
            # 오디오 정리
            self.audio_manager.cleanup()
            
            # 최종 AI 사용 리포트
            final_report = self.get_ai_usage_report()
            
            logger.info("📊 최종 AI 사용 리포트:")
            logger.info(f"   총 AI 호출: {final_report['ai_usage']['total_ai_calls']}")
            logger.info(f"   성공률: {final_report['ai_effectiveness']['success_rate']:.1f}%")
            logger.info(f"   총 비용: ${final_report['ai_usage']['ai_cost_estimate']:.4f}")
            logger.info(f"   평균 응답시간: {final_report['ai_effectiveness']['avg_response_time']:.2f}초")
            
            logger.info("✅ 시스템 정리 완료")
            
        except Exception as e:
            logger.error(f"정리 중 오류: {e}")

# 간단한 테스트 함수
async def test_ai_system():
    """AI 시스템 테스트"""
    
    print("🤖 VoiceGuard AI System Test")
    print("=" * 50)
    
    manager = SimplifiedConversationManager()
    
    try:
        # AI 데모 실행
        await manager.show_ai_demo()
        
        # 상태 리포트
        report = manager.get_ai_usage_report()
        print("\n📊 AI 시스템 상태:")
        for component, status in report['system_components'].items():
            print(f"   {component}: {status}")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
    finally:
        await manager.cleanup()

if __name__ == "__main__":
    asyncio.run(test_ai_system())