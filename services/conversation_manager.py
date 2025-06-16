"""
VoiceGuard AI - AI 시스템 완전 활용 대화 관리자
모든 구현된 Agent와 LangChain을 실제로 사용
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
from enum import Enum

# 구현된 모든 AI 컴포넌트들을 실제 사용
from agents.coordinator_agent import CoordinatorAgent, TaskType, TaskPriority
from agents.detection_agent import DetectionAgent
from agents.analysis_agent import AnalysisAgent
from agents.response_agent import ResponseAgent
from langchain_workflows.detection_chain import DetectionChain
from core.llm_manager import llm_manager
from services.tts_service import tts_service
from services.audio_manager import audio_manager

logger = logging.getLogger(__name__)

class ConversationState(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    DETECTING = "detecting"
    ANALYZING = "analyzing"
    RESPONDING = "responding"
    SPEAKING = "speaking"
    ERROR = "error"

class EnhancedConversationManager:
    """AI 시스템을 완전히 활용하는 대화 관리자"""
    
    def __init__(self, client_id: str, client_secret: str):
        # 1. 기본 서비스들
        self.llm_manager = llm_manager
        self.tts_service = tts_service
        self.audio_manager = audio_manager
        
        # 2. AI 에이전트 시스템 활성화
        self.detection_agent = DetectionAgent()
        self.analysis_agent = AnalysisAgent()
        self.response_agent = ResponseAgent()
        
        # 3. 코디네이터 에이전트 (멀티에이전트 조정)
        self.coordinator = CoordinatorAgent(
            self.detection_agent,
            self.analysis_agent,
            self.response_agent
        )
        
        # 4. LangChain 워크플로우
        self.detection_chain = DetectionChain()
        
        # 5. 상태 관리
        self.state = ConversationState.IDLE
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.is_running = False
        self.stt_queue = asyncio.Queue(maxsize=10)
        
        # 6. AI 활용 통계
        self.ai_stats = {
            'agent_calls': 0,
            'langchain_calls': 0,
            'coordinator_tasks': 0,
            'avg_agent_time': 0.0,
            'detection_accuracy': 0.0,
            'response_quality': 0.0
        }
        
        # 7. 더미 STT (실제 구현 대체)
        self.stt_service = self._create_dummy_stt(client_id, client_secret)
        
        logger.info("🤖 Enhanced AI Conversation Manager 초기화 완료")
        logger.info(f"   활성 에이전트: Detection, Analysis, Response")
        logger.info(f"   LangChain 워크플로우: Detection Chain")
        logger.info(f"   코디네이터: Multi-Agent 조정 활성화")

    def _create_dummy_stt(self, client_id: str, client_secret: str):
        """더미 STT 서비스"""
        class DummySTT:
            def __init__(self, client_id, client_secret, callback):
                self.callback = callback
                self.is_running = False
            
            def start(self):
                logger.info("🎤 더미 STT 시작 (데모 입력 시뮬레이션)")
                self.is_running = True
                
                # 시뮬레이션 입력들
                demo_inputs = [
                    "안녕하세요, 금융감독원입니다. 고객님 계좌에 문제가 발생했습니다.",
                    "저는 검찰청 수사관입니다. 긴급한 사건으로 연락드렸습니다.",
                    "저금리 대출 승인이 가능합니다. 지금 바로 앱을 설치해주세요.",
                    "아들이 사고났어요! 병원비가 급히 필요해요!"
                ]
                
                async def simulate():
                    for i, text in enumerate(demo_inputs):
                        if self.is_running:
                            await asyncio.sleep(5)  # 5초 간격
                            logger.info(f"🎤 시뮬레이션 입력 {i+1}: {text[:30]}...")
                            self.callback(text)
                
                asyncio.create_task(simulate())
            
            def stop(self):
                self.is_running = False
        
        return DummySTT(client_id, client_secret, self._on_speech_detected)

    def _on_speech_detected(self, text: str):
        """STT 콜백 - 큐에 추가"""
        if text and text.strip():
            try:
                loop = asyncio.get_running_loop()
                loop.call_soon_threadsafe(self.stt_queue.put_nowait, text.strip())
            except Exception as e:
                logger.error(f"STT 결과 처리 오류: {e}")

    async def start_conversation(self):
        """AI 시스템을 완전 활용한 대화 시작"""
        
        logger.info("🚀 AI 시스템 초기화 중...")
        
        # 1. 코디네이터 시작 (멀티에이전트 조정)
        await self.coordinator.start()
        logger.info("✅ 멀티에이전트 코디네이터 활성화")
        
        # 2. 오디오 시스템 초기화
        self.audio_manager.initialize_output()
        
        # 3. STT 시작
        self.stt_service.start()
        
        self.is_running = True
        logger.info("🤖 Enhanced AI 대화 시스템 시작")
        
        # 환영 메시지
        await self._speak("안녕하세요! VoiceGuard AI 고급 분석 시스템입니다. 저는 여러 AI 에이전트와 LangChain을 활용하여 정교한 보이스피싱 분석을 제공합니다.")
        
        # 메인 루프
        await self._enhanced_conversation_loop()

    async def _enhanced_conversation_loop(self):
        """AI 완전 활용 대화 루프"""
        
        self._set_state(ConversationState.LISTENING)
        
        while self.is_running:
            try:
                # STT 입력 대기
                user_input = await asyncio.wait_for(
                    self.stt_queue.get(), 
                    timeout=2.0
                )
                
                if user_input:
                    await self._process_with_full_ai_pipeline(user_input)
                    
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"AI 대화 루프 오류: {e}")

    async def _process_with_full_ai_pipeline(self, text: str):
        """전체 AI 파이프라인 활용 처리"""
        
        start_time = time.time()
        logger.info(f"🤖 AI 파이프라인 시작: {text[:50]}...")
        
        try:
            # === STAGE 1: 멀티에이전트 병렬 분석 ===
            self._set_state(ConversationState.DETECTING)
            
            # 코디네이터에게 탐지 작업 제출
            detection_task_id = await self.coordinator.submit_task(
                task_type=TaskType.DETECTION,
                data={"text": text, "context": {"session_id": self.session_id}},
                priority=TaskPriority.HIGH
            )
            
            logger.info(f"📡 코디네이터 탐지 작업 제출: {detection_task_id}")
            self.ai_stats['coordinator_tasks'] += 1
            
            # === STAGE 2: LangChain 워크플로우 병렬 실행 ===
            langchain_task = asyncio.create_task(
                self._run_langchain_analysis(text)
            )
            
            # === STAGE 3: 결과 수집 및 통합 ===
            self._set_state(ConversationState.ANALYZING)
            
            # 코디네이터 결과 대기
            coordinator_result = await self._wait_for_coordinator_result(detection_task_id)
            
            # LangChain 결과 대기
            langchain_result = await langchain_task
            
            # === STAGE 4: 결과 통합 및 응답 생성 ===
            self._set_state(ConversationState.RESPONDING)
            
            final_analysis = await self._integrate_ai_results(
                coordinator_result, 
                langchain_result,
                text
            )
            
            # === STAGE 5: 고급 응답 생성 ===
            response_text = await self._generate_enhanced_response(final_analysis)
            
            # === STAGE 6: 음성 출력 ===
            if response_text:
                await self._speak(response_text)
            
            # 성능 통계 업데이트
            processing_time = time.time() - start_time
            self._update_ai_stats(processing_time, final_analysis)
            
            logger.info(f"🎯 AI 파이프라인 완료: {processing_time:.2f}초")
            
        except Exception as e:
            logger.error(f"AI 파이프라인 오류: {e}")
            await self._speak("AI 시스템 처리 중 오류가 발생했습니다. 다시 말씀해주세요.")

    async def _run_langchain_analysis(self, text: str) -> Dict[str, Any]:
        """LangChain 워크플로우 실행"""
        
        logger.info("🔗 LangChain 워크플로우 시작")
        
        try:
            # 포괄적 사기 분석 (LangChain)
            langchain_result = await self.detection_chain.analyze_scam_comprehensive(
                text=text,
                context={"session_id": self.session_id}
            )
            
            self.ai_stats['langchain_calls'] += 1
            logger.info("✅ LangChain 분석 완료")
            
            return langchain_result
            
        except Exception as e:
            logger.error(f"LangChain 분석 오류: {e}")
            return {"error": str(e), "source": "langchain"}

    async def _wait_for_coordinator_result(self, task_id: str) -> Dict[str, Any]:
        """코디네이터 결과 대기"""
        
        for attempt in range(30):  # 30초 대기
            task_status = self.coordinator.get_task_status(task_id)
            
            if task_status:
                if task_status["status"] == "completed":
                    logger.info("✅ 코디네이터 분석 완료")
                    return task_status.get("result", {})
                elif task_status["status"] == "failed":
                    logger.error(f"❌ 코디네이터 분석 실패: {task_status.get('error')}")
                    return {"error": task_status.get('error'), "source": "coordinator"}
            
            await asyncio.sleep(1)
        
        logger.warning("⏰ 코디네이터 작업 타임아웃")
        return {"error": "timeout", "source": "coordinator"}

    async def _integrate_ai_results(self, coordinator_result: Dict, langchain_result: Dict, original_text: str) -> Dict[str, Any]:
        """AI 결과 통합 분석"""
        
        logger.info("🧠 AI 결과 통합 중...")
        
        # 결과 수집
        results = []
        
        if "error" not in coordinator_result:
            results.append({
                "source": "multi_agent_coordinator",
                "data": coordinator_result,
                "weight": 0.6  # 멀티에이전트에 높은 가중치
            })
        
        if "error" not in langchain_result:
            results.append({
                "source": "langchain_workflow", 
                "data": langchain_result,
                "weight": 0.4  # LangChain에 보조 가중치
            })
        
        if not results:
            logger.warning("모든 AI 시스템 실패 - 폴백 분석 사용")
            return await self._fallback_analysis(original_text)
        
        # 가중 평균으로 최종 위험도 계산
        total_weight = sum(r["weight"] for r in results)
        weighted_risk = 0.0
        
        all_indicators = set()
        scam_types = []
        
        for result in results:
            data = result["data"]
            weight = result["weight"]
            
            # 위험도 추출
            risk_score = self._extract_risk_score(data)
            weighted_risk += risk_score * weight
            
            # 지표 수집
            indicators = self._extract_indicators(data)
            all_indicators.update(indicators)
            
            # 사기 유형 수집
            scam_type = self._extract_scam_type(data)
            if scam_type:
                scam_types.append(scam_type)
        
        final_risk = weighted_risk / total_weight
        
        return {
            "final_risk_score": final_risk,
            "risk_level": self._determine_risk_level(final_risk),
            "primary_scam_type": scam_types[0] if scam_types else "unknown",
            "all_indicators": list(all_indicators),
            "ai_sources": [r["source"] for r in results],
            "analysis_quality": len(results) / 2.0,  # 두 시스템 모두 성공하면 1.0
            "processing_details": {
                "coordinator_used": any(r["source"] == "multi_agent_coordinator" for r in results),
                "langchain_used": any(r["source"] == "langchain_workflow" for r in results),
                "integration_method": "weighted_ensemble"
            }
        }

    def _extract_risk_score(self, data: Dict) -> float:
        """데이터에서 위험도 점수 추출"""
        return (
            data.get("risk_score", 0.0) or
            data.get("final_risk_score", 0.0) or
            data.get("score", 0.0) or
            0.5  # 기본값
        )

    def _extract_indicators(self, data: Dict) -> List[str]:
        """데이터에서 위험 지표 추출"""
        return (
            data.get("key_indicators", []) or
            data.get("all_indicators", []) or
            data.get("indicators", []) or
            data.get("detected_patterns", []) or
            []
        )

    def _extract_scam_type(self, data: Dict) -> Optional[str]:
        """데이터에서 사기 유형 추출"""
        return (
            data.get("scam_type") or
            data.get("primary_scam_type") or
            data.get("type") or
            data.get("category")
        )

    def _determine_risk_level(self, risk_score: float) -> str:
        """위험도 레벨 결정"""
        if risk_score >= 0.8:
            return "critical"
        elif risk_score >= 0.6:
            return "high"
        elif risk_score >= 0.4:
            return "medium"
        else:
            return "low"

    async def _fallback_analysis(self, text: str) -> Dict[str, Any]:
        """폴백 분석 (모든 AI 시스템 실패 시)"""
        
        logger.info("🔄 폴백 분석 실행")
        
        # 기본 LLM 직접 호출
        try:
            result = await self.llm_manager.analyze_scam_risk(text=text)
            
            return {
                "final_risk_score": result.metadata.get("risk_score", 0.5),
                "risk_level": "medium",
                "primary_scam_type": result.metadata.get("scam_type", "unknown"),
                "all_indicators": ["폴백_분석"],
                "ai_sources": ["direct_llm"],
                "analysis_quality": 0.3,
                "processing_details": {
                    "fallback_used": True,
                    "reason": "ai_systems_failed"
                }
            }
        except Exception as e:
            logger.error(f"폴백 분석도 실패: {e}")
            return {
                "final_risk_score": 0.5,
                "risk_level": "unknown",
                "primary_scam_type": "analysis_failed",
                "all_indicators": ["시스템_오류"],
                "ai_sources": [],
                "analysis_quality": 0.0,
                "processing_details": {"total_failure": True}
            }

    async def _generate_enhanced_response(self, analysis: Dict[str, Any]) -> str:
        """AI 분석 결과 기반 고급 응답 생성"""
        
        risk_score = analysis["final_risk_score"]
        risk_level = analysis["risk_level"]
        scam_type = analysis["primary_scam_type"]
        indicators = analysis["all_indicators"]
        ai_sources = analysis["ai_sources"]
        quality = analysis["analysis_quality"]
        
        # AI 시스템 활용도 표시
        ai_info = []
        if "multi_agent_coordinator" in ai_sources:
            ai_info.append("멀티에이전트 시스템")
        if "langchain_workflow" in ai_sources:
            ai_info.append("LangChain 워크플로우")
        if "direct_llm" in ai_sources:
            ai_info.append("Gemini 직접 분석")
        
        response = f"""🤖 AI 종합 분석 결과 (활용 시스템: {', '.join(ai_info)})

📊 위험도 평가:
• 최종 위험도: {risk_score:.1%} ({risk_level})
• 추정 사기 유형: {scam_type}
• 분석 품질: {quality:.1%}

🔍 탐지된 위험 요소:
{chr(10).join(f'• {indicator}' for indicator in indicators[:5])}

💡 AI 분석 상세:
• 사용된 AI 시스템: {len(ai_sources)}개
• 분석 방법: {analysis['processing_details'].get('integration_method', '단일 분석')}"""

        # 위험도별 권장사항
        if risk_level == "critical":
            response += """

🚨 즉시 대응 필요:
1. 지금 당장 통화를 끊으세요
2. 절대 개인정보를 제공하지 마세요  
3. 112(경찰) 또는 1332(금융감독원)에 신고하세요
4. 가족에게 상황을 알리세요"""

        elif risk_level == "high":
            response += """

⚠️ 높은 위험 감지:
1. 통화를 중단하고 직접 기관에 확인하세요
2. 급하게 결정하지 마세요
3. 의심스러운 요구는 거절하세요"""

        elif risk_level == "medium":
            response += """

🔍 주의 필요:
1. 상대방 신원을 다시 한번 확인하세요
2. 개인정보 제공 전 신중히 판단하세요
3. 의심되면 공식 경로로 확인하세요"""

        else:
            response += """

✅ 상대적으로 안전:
1. 여전히 개인정보는 신중하게 제공하세요
2. 이상한 요구가 있다면 즉시 확인하세요"""

        return response

    async def _speak(self, text: str):
        """TTS 음성 출력"""
        self._set_state(ConversationState.SPEAKING)
        
        try:
            audio_stream = self.tts_service.text_to_speech_stream(text)
            await self.audio_manager.play_audio_stream(audio_stream)
        except Exception as e:
            logger.error(f"TTS 오류: {e}")
        
        self._set_state(ConversationState.LISTENING)

    def _set_state(self, new_state: ConversationState):
        """상태 변경"""
        if self.state != new_state:
            old_state = self.state
            self.state = new_state
            logger.debug(f"상태 변경: {old_state.value} -> {new_state.value}")

    def _update_ai_stats(self, processing_time: float, analysis: Dict[str, Any]):
        """AI 통계 업데이트"""
        
        self.ai_stats['agent_calls'] += 1
        
        # 평균 처리 시간
        current_avg = self.ai_stats['avg_agent_time']
        call_count = self.ai_stats['agent_calls']
        self.ai_stats['avg_agent_time'] = (current_avg * (call_count - 1) + processing_time) / call_count
        
        # 분석 품질
        quality = analysis.get('analysis_quality', 0.5)
        self.ai_stats['response_quality'] = (self.ai_stats['response_quality'] + quality) / 2
        
        logger.info(f"📈 AI 성능: 처리시간 {processing_time:.2f}초, 품질 {quality:.1%}")

    def get_ai_status(self) -> Dict[str, Any]:
        """AI 시스템 상태 조회"""
        
        return {
            "ai_systems_active": {
                "coordinator": hasattr(self, 'coordinator'),
                "detection_agent": hasattr(self, 'detection_agent'),
                "analysis_agent": hasattr(self, 'analysis_agent'),
                "response_agent": hasattr(self, 'response_agent'),
                "langchain_detection": hasattr(self, 'detection_chain'),
                "llm_manager": hasattr(self, 'llm_manager')
            },
            "performance_stats": self.ai_stats.copy(),
            "session_info": {
                "session_id": self.session_id,
                "state": self.state.value,
                "is_running": self.is_running
            }
        }

    async def cleanup(self):
        """시스템 정리"""
        logger.info("🧹 Enhanced AI 시스템 정리 중...")
        
        self.is_running = False
        
        try:
            if hasattr(self, 'coordinator'):
                await self.coordinator.stop()
            
            if hasattr(self, 'stt_service'):
                self.stt_service.stop()
            
            if hasattr(self, 'audio_manager'):
                self.audio_manager.cleanup()
            
            # 최종 AI 통계 출력
            stats = self.get_ai_status()
            logger.info("🤖 AI 시스템 최종 통계:")
            logger.info(f"   에이전트 호출: {stats['performance_stats']['agent_calls']}")
            logger.info(f"   LangChain 호출: {stats['performance_stats']['langchain_calls']}")  
            logger.info(f"   코디네이터 작업: {stats['performance_stats']['coordinator_tasks']}")
            logger.info(f"   평균 처리시간: {stats['performance_stats']['avg_agent_time']:.2f}초")
            
            logger.info("✅ Enhanced AI 시스템 정리 완료")
            
        except Exception as e:
            logger.error(f"정리 중 오류: {e}")