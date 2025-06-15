"""
VoiceGuard AI - STT (Speech-to-Text) 처리 엔진
실시간 음성 인식으로 보이스피싱 탐지 지원
"""

import asyncio
import io
import wave
import logging
from typing import Optional, List, Dict, Any
import time
from dataclasses import dataclass

import openai
import whisper
import torch
import numpy as np
from pydub import AudioSegment
from pydub.effects import normalize

from config.settings import voice_config, monitoring_config

logger = logging.getLogger(__name__)

@dataclass
class TranscriptionResult:
    """음성 인식 결과"""
    text: str
    confidence: float
    processing_time: float
    language: str
    segments: List[Dict[str, Any]]
    audio_duration: float

class STTEngine:
    """실시간 STT 처리 엔진"""
    
    def __init__(self):
        self.whisper_model = None
        self.openai_client = openai.AsyncOpenAI()
        self.processing_stats = {
            "total_requests": 0,
            "successful_transcriptions": 0,
            "failed_transcriptions": 0,
            "avg_processing_time": 0.0,
            "total_processing_time": 0.0
        }
        
        # 초기화
        self._initialize_models()
        logger.info("STT 엔진 초기화 완료")
    
    def _initialize_models(self):
        """모델 초기화"""
        try:
            # Whisper 로컬 모델 로드 (오프라인 처리용)
            if torch.cuda.is_available():
                self.whisper_model = whisper.load_model(
                    voice_config.STT_MODEL.replace("-v3", ""), 
                    device="cuda"
                )
                logger.info("Whisper 모델 GPU 로드 완료")
            else:
                self.whisper_model = whisper.load_model(
                    voice_config.STT_MODEL.replace("-v3", ""),
                    device="cpu"
                )
                logger.info("Whisper 모델 CPU 로드 완료")
                
        except Exception as e:
            logger.error(f"Whisper 모델 로드 실패: {e}")
            self.whisper_model = None
    
    async def transcribe_chunk(self, 
                              audio_data: bytes, 
                              use_cloud: bool = False) -> Optional[str]:
        """오디오 청크를 텍스트로 변환"""
        
        start_time = time.time()
        self.processing_stats["total_requests"] += 1
        
        try:
            # 1. 오디오 전처리
            processed_audio = self._preprocess_audio(audio_data)
            if processed_audio is None:
                return None
            
            # 2. STT 처리 선택
            if use_cloud and hasattr(self, 'openai_client'):
                text = await self._transcribe_with_openai(processed_audio)
            else:
                text = await self._transcribe_with_whisper(processed_audio)
            
            # 3. 후처리
            cleaned_text = self._postprocess_text(text)
            
            # 4. 통계 업데이트
            processing_time = time.time() - start_time
            self._update_stats(processing_time, success=True)
            
            # 5. 성능 체크
            if processing_time > monitoring_config.TARGET_RESPONSE_TIME:
                logger.warning(f"STT 처리 시간 초과: {processing_time:.2f}초")
            
            return cleaned_text
            
        except Exception as e:
            self._update_stats(time.time() - start_time, success=False)
            logger.error(f"STT 처리 실패: {e}")
            return None
    
    def _preprocess_audio(self, audio_data: bytes) -> Optional[np.ndarray]:
        """오디오 전처리"""
        try:
            # 1. 바이트 데이터를 AudioSegment로 변환
            audio_segment = AudioSegment.from_raw(
                io.BytesIO(audio_data),
                sample_width=2,  # 16-bit
                frame_rate=voice_config.SAMPLE_RATE,
                channels=1
            )
            
            # 2. 너무 짧은 오디오 필터링
            if len(audio_segment) < 500:  # 0.5초 미만
                return None
            
            # 3. 오디오 정규화 및 노이즈 감소
            normalized_audio = normalize(audio_segment)
            
            # 4. 샘플레이트 조정
            if normalized_audio.frame_rate != voice_config.SAMPLE_RATE:
                normalized_audio = normalized_audio.set_frame_rate(voice_config.SAMPLE_RATE)
            
            # 5. NumPy 배열로 변환
            audio_array = np.array(normalized_audio.get_array_of_samples(), dtype=np.float32)
            audio_array = audio_array / 32768.0  # 정규화
            
            return audio_array
            
        except Exception as e:
            logger.error(f"오디오 전처리 실패: {e}")
            return None
    
    async def _transcribe_with_whisper(self, audio_array: np.ndarray) -> str:
        """Whisper 로컬 모델로 음성 인식"""
        if self.whisper_model is None:
            raise Exception("Whisper 모델이 로드되지 않음")
        
        try:
            # 비동기 처리를 위해 스레드풀 사용
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                self.whisper_model.transcribe, 
                audio_array,
                {"language": voice_config.STT_LANGUAGE, "fp16": False}
            )
            
            return result.get("text", "").strip()
            
        except Exception as e:
            logger.error(f"Whisper 음성 인식 실패: {e}")
            raise
    
    async def _transcribe_with_openai(self, audio_array: np.ndarray) -> str:
        """OpenAI Whisper API로 음성 인식"""
        try:
            # NumPy 배열을 WAV 파일로 변환
            audio_bytes = self._numpy_to_wav(audio_array)
            
            # OpenAI API 호출
            transcript = await self.openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=("audio.wav", audio_bytes, "audio/wav"),
                language=voice_config.STT_LANGUAGE
            )
            
            return transcript.text.strip()
            
        except Exception as e:
            logger.error(f"OpenAI STT 실패: {e}")
            raise
    
    def _numpy_to_wav(self, audio_array: np.ndarray) -> bytes:
        """NumPy 배열을 WAV 바이트로 변환"""
        # 16-bit PCM으로 변환
        audio_int16 = (audio_array * 32767).astype(np.int16)
        
        # WAV 파일 생성
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # 모노
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(voice_config.SAMPLE_RATE)
            wav_file.writeframes(audio_int16.tobytes())
        
        wav_buffer.seek(0)
        return wav_buffer.read()
    
    def _postprocess_text(self, text: str) -> str:
        """텍스트 후처리"""
        if not text:
            return ""
        
        # 1. 기본 정리
        cleaned = text.strip()
        
        # 2. 한국어 특수 처리
        # 숫자 정규화 (일, 이, 삼 -> 1, 2, 3)
        number_map = {
            "일": "1", "이": "2", "삼": "3", "사": "4", "오": "5",
            "육": "6", "칠": "7", "팔": "8", "구": "9", "십": "10"
        }
        
        for korean, arabic in number_map.items():
            cleaned = cleaned.replace(korean, arabic)
        
        # 3. 사기 관련 키워드 정규화
        keyword_map = {
            "금감원": "금융감독원",
            "검찰": "검찰청", 
            "경찰": "경찰서",
            "국세": "국세청",
            "송금": "송금",
            "계좌": "계좌",
            "대출": "대출"
        }
        
        for short, full in keyword_map.items():
            cleaned = cleaned.replace(short, full)
        
        # 4. 불필요한 문자 제거
        import re
        cleaned = re.sub(r'[^\w\s가-힣]', '', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        return cleaned.strip()
    
    def _update_stats(self, processing_time: float, success: bool):
        """통계 업데이트"""
        if success:
            self.processing_stats["successful_transcriptions"] += 1
        else:
            self.processing_stats["failed_transcriptions"] += 1
        
        self.processing_stats["total_processing_time"] += processing_time
        self.processing_stats["avg_processing_time"] = (
            self.processing_stats["total_processing_time"] / 
            self.processing_stats["total_requests"]
        )
    
    async def transcribe_full_audio(self, 
                                   audio_data: bytes,
                                   detailed: bool = False) -> TranscriptionResult:
        """전체 오디오 상세 분석"""
        
        start_time = time.time()
        
        try:
            # 1. 오디오 전처리
            processed_audio = self._preprocess_audio(audio_data)
            if processed_audio is None:
                raise Exception("오디오 전처리 실패")
            
            # 2. 상세 분석 (세그먼트 포함)
            if self.whisper_model is None:
                raise Exception("Whisper 모델이 로드되지 않음")
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self.whisper_model.transcribe,
                processed_audio,
                {
                    "language": voice_config.STT_LANGUAGE,
                    "word_timestamps": detailed,
                    "fp16": False
                }
            )
            
            # 3. 결과 구성
            processing_time = time.time() - start_time
            audio_duration = len(processed_audio) / voice_config.SAMPLE_RATE
            
            return TranscriptionResult(
                text=result.get("text", "").strip(),
                confidence=self._calculate_confidence(result),
                processing_time=processing_time,
                language=result.get("language", voice_config.STT_LANGUAGE),
                segments=result.get("segments", []),
                audio_duration=audio_duration
            )
            
        except Exception as e:
            logger.error(f"전체 오디오 분석 실패: {e}")
            raise
    
    def _calculate_confidence(self, whisper_result: Dict) -> float:
        """Whisper 결과에서 신뢰도 계산"""
        segments = whisper_result.get("segments", [])
        if not segments:
            return 0.5  # 기본값
        
        # 각 세그먼트의 평균 확률 계산
        total_prob = 0.0
        total_tokens = 0
        
        for segment in segments:
            tokens = segment.get("tokens", [])
            if tokens:
                avg_logprob = segment.get("avg_logprob", -1.0)
                # 로그 확률을 확률로 변환
                prob = np.exp(avg_logprob) if avg_logprob > -10 else 0.1
                total_prob += prob * len(tokens)
                total_tokens += len(tokens)
        
        if total_tokens == 0:
            return 0.5
        
        return min(total_prob / total_tokens, 1.0)
    
    async def detect_voice_activity(self, audio_data: bytes) -> bool:
        """음성 활동 감지 (VAD - Voice Activity Detection)"""
        try:
            processed_audio = self._preprocess_audio(audio_data)
            if processed_audio is None:
                return False
            
            # 간단한 에너지 기반 VAD
            energy = np.sqrt(np.mean(processed_audio ** 2))
            silence_threshold = 0.01  # 조정 가능한 임계값
            
            return energy > silence_threshold
            
        except Exception as e:
            logger.error(f"음성 활동 감지 실패: {e}")
            return False
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계 반환"""
        return {
            **self.processing_stats,
            "success_rate": (
                self.processing_stats["successful_transcriptions"] / 
                max(1, self.processing_stats["total_requests"])
            ),
            "model_info": {
                "whisper_loaded": self.whisper_model is not None,
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "model_name": voice_config.STT_MODEL
            }
        }
    
    async def health_check(self) -> Dict[str, bool]:
        """STT 엔진 상태 확인"""
        status = {
            "whisper_model": self.whisper_model is not None,
            "openai_api": False,
            "audio_processing": True
        }
        
        # OpenAI API 테스트
        try:
            # 간단한 테스트 오디오로 API 확인
            test_audio = np.zeros(16000, dtype=np.float32)  # 1초 무음
            test_wav = self._numpy_to_wav(test_audio)
            
            await self.openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=("test.wav", test_wav[:1000], "audio/wav"),  # 작은 샘플만
                language="ko"
            )
            status["openai_api"] = True
        except Exception:
            status["openai_api"] = False
        
        return status
    
    def cleanup(self):
        """리소스 정리"""
        if self.whisper_model is not None:
            del self.whisper_model
            self.whisper_model = None
            
        # GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("STT 엔진 리소스 정리 완료")