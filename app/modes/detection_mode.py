"""
VoiceGuard AI - 실시간 탐지 모드
의심스러운 통화 내용을 실시간으로 분석하고 경고
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from .base_mode import BaseMode, ModeState

# 조건부 import
try:
    from services.simple_stt_service import SttService
except ImportError:
    try:
        from services.stt_service import SttService
    except ImportError:
        # 더미 STT 서비스
        class SttService:
            def __init__(self, *args, **kwargs):
                self.is_running = False
            def start(self):
                print("🎤 STT 서비스를 사용할 수 없습니다. 텍스트 입력으로 진행합니다.")
            def stop(self):
                pass

try:
    from core.analyzer import VoicePhishingAnalyzer
except ImportError:
    # 더미 분석기
    class VoicePhishingAnalyzer:
        def __init__(self, llm_manager):
            self.llm_manager = llm_manager
        async def analyze_text(self, text, context=None):
            return {
                "risk_score": 0.3,
                "risk_level": "낮음",
                "scam_type": "테스트",
                "key_indicators": ["테스트"],
                "recommendation": "정상적인 테스트 결과입니다."
            }

try:
    from config.settings import settings
except ImportError:
    # 더미 설정
    class Settings:
        RETURNZERO_CLIENT_ID = "demo"
        RETURNZERO_CLIENT_SECRET = "demo"
    settings = Settings()

logger = logging.getLogger(__name__)

class DetectionMode(BaseMode):
    """실시간 탐지 모드"""
    
    @property
    def mode_name(self) -> str:
        return "실시간 탐지"
    
    @property
    def mode_description(self) -> str:
        return "의심스러운 통화 내용을 실시간으로 분석하여 보이스피싱을 탐지합니다"
    
    def _load_mode_config(self) -> Dict[str, Any]:
        """탐지 모드 설정"""
        return {
            'analysis_threshold': 0.3,
            'real_time_alerts': True,
            'auto_record': False,
            'sensitivity_level': 'medium',
            'max_analysis_length': 1000
        }
    
    async def _initialize_mode(self) -> bool:
        """탐지 모드 초기화"""
        
        try:
            # STT 서비스 초기화
            self.stt_service = SttService(
                client_id=settings.RETURNZERO_CLIENT_ID or "demo",
                client_secret=settings.RETURNZERO_CLIENT_SECRET or "demo",
                transcript_callback=self._on_speech_detected
            )
            
            # 분석 엔진 초기화
            self.analyzer = VoicePhishingAnalyzer(self.llm_manager)
            
            # 분석 큐 및 상태
            self.analysis_queue = asyncio.Queue(maxsize=10)
            self.current_conversation = []
            self.last_analysis_time = datetime.now()
            
            logger.info("✅ 실시간 탐지 모드 초기화 완료")
            return True
            
        except Exception as e:
            logger.error(f"탐지 모드 초기화 실패: {e}")
            return False
    
    async def _run_mode_logic(self):
        """실시간 탐지 메인 로직"""
        
        print("🎤 실시간 보이스피싱 탐지 모드")
        print("💡 의심스러운 통화 내용을 입력해주세요.")
        print("💡 '종료'를 입력하면 분석을 마칩니다.")
        print("-" * 50)
        
        # STT 서비스 시작 (사용 가능한 경우)
        try:
            self.stt_service.start()
        except:
            print("🎤 음성 인식 대신 텍스트 입력을 사용합니다.")
        
        # 분석 워커 시작
        analysis_task = asyncio.create_task(self._analysis_worker())
        
        try:
            # 메인 루프 - 텍스트 입력 받기
            while self.is_running:
                try:
                    # 비동기적으로 사용자 입력 받기
                    print("\n📝 분석할 내용을 입력하세요: ", end="", flush=True)
                    user_input = await asyncio.to_thread(input)
                    
                    if user_input.strip():
                        # 종료 명령 확인
                        if any(keyword in user_input.lower() for keyword in ['종료', '끝', '중단', '그만', 'exit', 'quit']):
                            print(f"\n🛑 종료 명령: '{user_input}'")
                            break
                        
                        # 분석 수행
                        await self._process_user_input(user_input.strip())
                
                except (EOFError, KeyboardInterrupt):
                    break
                except Exception as e:
                    logger.error(f"입력 처리 오류: {e}")
                    print(f"❌ 입력 오류: {e}")
            
        except Exception as e:
            logger.error(f"탐지 모드 실행 오류: {e}")
        finally:
            # 정리
            try:
                self.stt_service.stop()
            except:
                pass
            analysis_task.cancel()
            
            try:
                await analysis_task
            except asyncio.CancelledError:
                pass
    
    async def _process_user_input(self, text: str):
        """사용자 입력 직접 처리"""
        if not text or not text.strip():
            return
        
        timestamp = datetime.now()
        
        # 분석 큐에 추가
        try:
            self.analysis_queue.put_nowait({
                'text': text,
                'timestamp': timestamp
            })
            
            # 현재 대화에 추가
            self.current_conversation.append({
                'text': text,
                'timestamp': timestamp
            })
            
            # 대화 길이 제한
            if len(self.current_conversation) > 20:
                self.current_conversation.pop(0)
            
            print(f"\n👤 입력: {text}")
            
        except asyncio.QueueFull:
            logger.warning("분석 큐가 가득참 - 이전 분석 대기 중")
            print("⚠️ 분석 중입니다. 잠시 기다려주세요.")
    
    def _on_speech_detected(self, text: str):
        """STT 결과 콜백"""
        
        if not text or not text.strip():
            return
        
        text = text.strip()
        
        # 종료 명령 확인
        if any(keyword in text.lower() for keyword in ['종료', '끝', '중단', '그만']):
            print(f"\n🛑 종료 명령 감지: '{text}'")
            self.stop()
            return
        
        # 분석 큐에 추가
        try:
            timestamp = datetime.now()
            self.analysis_queue.put_nowait({
                'text': text,
                'timestamp': timestamp
            })
            
            # 현재 대화에 추가
            self.current_conversation.append({
                'text': text,
                'timestamp': timestamp
            })
            
            # 대화 길이 제한
            if len(self.current_conversation) > 20:
                self.current_conversation.pop(0)
            
            print(f"\n👤 입력: {text}")
            
        except asyncio.QueueFull:
            logger.warning("분석 큐가 가득참 - 이전 분석 대기 중")
    
    async def _analysis_worker(self):
        """분석 워커 - 백그라운드에서 지속적으로 분석"""
        
        while self.is_running:
            try:
                # 분석할 데이터 대기
                speech_data = await asyncio.wait_for(
                    self.analysis_queue.get(),
                    timeout=1.0
                )
                
                # 분석 수행
                await self._analyze_speech(speech_data)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"분석 워커 오류: {e}")
                await asyncio.sleep(1)
    
    async def _analyze_speech(self, speech_data: Dict[str, Any]):
        """음성 데이터 분석"""
        
        start_time = datetime.now()
        text = speech_data['text']
        
        try:
            print(f"🧠 분석 중... ", end="", flush=True)
            
            # LLM 기반 분석
            analysis_result = await self.analyzer.analyze_text(
                text=text,
                context={
                    'conversation_history': self.current_conversation[-5:],  # 최근 5개
                    'session_id': self.session_id,
                    'timestamp': speech_data['timestamp'].isoformat()
                }
            )
            
            # 분석 시간 계산
            analysis_time = (datetime.now() - start_time).total_seconds()
            print(f"완료 ({analysis_time:.2f}초)")
            
            # 결과 출력
            await self._display_analysis_result(analysis_result, text)
            
            # 통계 업데이트
            self._update_stats(
                success=True,
                last_risk_score=analysis_result.get('risk_score', 0),
                analysis_time=analysis_time
            )
            
        except Exception as e:
            logger.error(f"분석 실패: {e}")
            print(f"❌ 분석 실패: {e}")
            self._update_stats(success=False)
    
    async def _display_analysis_result(self, result: Dict[str, Any], original_text: str):
        """분석 결과 표시"""
        
        risk_score = result.get('risk_score', 0)
        risk_level = result.get('risk_level', '낮음')
        scam_type = result.get('scam_type', '알 수 없음')
        
        # 위험도에 따른 아이콘 및 색상
        if risk_score >= 0.8:
            icon = "🚨"
            level_text = "매우 위험"
        elif risk_score >= 0.6:
            icon = "⚠️"
            level_text = "위험"
        elif risk_score >= 0.4:
            icon = "🔍"
            level_text = "주의 필요"
        else:
            icon = "✅"
            level_text = "안전"
        
        print(f"\n{icon} 분석 결과:")
        print(f"   위험도: {level_text} ({risk_score:.1%})")
        print(f"   추정 유형: {scam_type}")
        
        # 주요 지표 출력
        indicators = result.get('key_indicators', [])
        if indicators:
            print(f"   주요 지표: {', '.join(indicators[:3])}")
        
        # 권장사항
        recommendation = result.get('recommendation', '')
        if recommendation:
            print(f"   권장사항: {recommendation}")
        
        # 높은 위험도일 때 음성 경고
        if risk_score >= 0.7:
            await self._voice_alert(risk_score, scam_type)
        
        print("-" * 50)
    
    async def _voice_alert(self, risk_score: float, scam_type: str):
        """음성 경고"""
        
        try:
            if risk_score >= 0.8:
                alert_text = f"위험! {scam_type} 의심됩니다. 즉시 통화를 중단하세요!"
            else:
                alert_text = f"주의하세요. {scam_type} 가능성이 있습니다."
            
            print(f"🔊 음성 경고: {alert_text}")
            await self._speak(alert_text)
            
        except Exception as e:
            logger.warning(f"음성 경고 실패: {e}")
    
    def _should_stop(self) -> bool:
        """중지 조건 확인"""
        
        # 사용자가 명시적으로 중지 요청
        if not self.is_running:
            return True
        
        # 너무 오랫동안 입력이 없으면 안내 메시지
        time_since_last = (datetime.now() - self.last_analysis_time).total_seconds()
        if time_since_last > 120:  # 2분
            print("\n💡 음성 입력이 없습니다. 계속하려면 말씀해주세요.")
            self.last_analysis_time = datetime.now()
        
        return False
    
    async def _cleanup_mode(self):
        """탐지 모드 정리"""
        
        try:
            # STT 서비스 정리
            if hasattr(self, 'stt_service'):
                self.stt_service.stop()
            
            # 분석 큐 정리
            while not self.analysis_queue.empty():
                try:
                    self.analysis_queue.get_nowait()
                except:
                    break
            
            logger.info("탐지 모드 정리 완료")
            
        except Exception as e:
            logger.error(f"탐지 모드 정리 오류: {e}")
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """대화 요약 조회"""
        
        total_inputs = len(self.current_conversation)
        
        if total_inputs == 0:
            return {"message": "분석된 대화가 없습니다."}
        
        return {
            "total_inputs": total_inputs,
            "session_duration": (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
            "latest_inputs": [item['text'] for item in self.current_conversation[-3:]],
            "analysis_count": self.stats['total_interactions']
        }