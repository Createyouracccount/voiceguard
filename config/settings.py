# import os
# from pathlib import Path
# from dotenv import load_dotenv
# from enum import Enum

# # 프로젝트 루트의 .env 파일 로드
# BASE_DIR = Path(__file__).resolve().parent.parent
# dotenv_path = BASE_DIR / '.env'
# load_dotenv(dotenv_path)

# # --- AI 모델 및 위험도 설정 ---
# class RiskLevel(Enum):
#     LOW = "낮음"
#     MEDIUM = "주의"
#     HIGH = "위험"
#     CRITICAL = "매우 위험"

# class ModelTier(Enum):
#     FAST = "fast"
#     BALANCED = "balanced"
#     ACCURATE = "accurate"

# class AIConfig:
#     class GPT4:
#         temperature = 0.3
#         max_tokens = 2048
#     class GPT35_TURBO:
#         temperature = 0.1
#         max_tokens = 1024
#     class CLAUDE:
#         temperature = 0.2
#         max_tokens = 2048

# # --- 탐지 임계값 ---
# class DetectionThresholds:
#     medium_risk = 0.4
#     high_risk = 0.7
#     critical_risk = 0.85

# # --- 전체 설정 클래스 ---
# class Settings:
#     """애플리케이션 설정"""
#     # LLM API 키
#     OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
#     GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
#     LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")

#     # STT 설정 (ReturnZero)
#     RETURNZERO_CLIENT_ID = os.getenv("RETURNZERO_CLIENT_ID", "")
#     RETURNZERO_CLIENT_SECRET = os.getenv("RETURNZERO_CLIENT_SECRET", "")

#     # TTS 설정 (ElevenLabs)
#     ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "").strip()
#     TTS_VOICE_ID = os.getenv("TTS_VOICE_ID", "uyVNoMrnUku1dZyVEXwD").strip()
#     TTS_MODEL = os.getenv("TTS_MODEL", "eleven_multilingual_v2").strip()

#     # 오디오 설정
#     SAMPLE_RATE = 16000
#     CHUNK_SIZE = int(SAMPLE_RATE / 10)  # 100ms
#     CHANNELS = 1

#     # 시스템 설정
#     DEBUG = os.getenv("DEBUG", "True").lower() == "true"
#     LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
#     # 대화 설정
#     SESSION_TIMEOUT = 1800  # 30분
#     SILENCE_TIMEOUT = 10.0 # 10초 침묵 시 반응

# # 인스턴스 생성
# settings = Settings()
# ai_config = AIConfig()
# detection_thresholds = DetectionThresholds()

import os
from pathlib import Path
from enum import Enum
from dotenv import load_dotenv

# .env 파일 로드
BASE_DIR = Path(__file__).resolve().parent.parent
dotenv_path = BASE_DIR / '.env'
load_dotenv(dotenv_path)

# --- AI 모델 및 위험도 관련 설정 클래스들 (LLMManager가 필요로 함) ---

class RiskLevel(Enum):
    LOW = "낮음"
    MEDIUM = "주의"
    HIGH = "위험"
    CRITICAL = "매우 위험"

class ModelTier(Enum):
    FAST = "fast"
    BALANCED = "balanced"
    ACCURATE = "accurate"

class AIConfig:
    class GPT4:
        temperature = 0.1
        max_tokens = 2048
    class GPT35_TURBO:
        temperature = 0.1
        max_tokens = 1024
    class GEMINI:
        temperature = 0.1
        max_tokens = 2048
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

class DetectionThresholds:
    medium_risk = 0.4
    high_risk = 0.7
    critical_risk = 0.85

# --- 기존의 하드웨어 및 서비스 관련 설정 ---

class Settings:
    """애플리케이션 설정"""
    # LLM API 키
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
    
    # STT 설정 (ReturnZero)
    RETURNZERO_CLIENT_ID = os.getenv("RETURNZERO_CLIENT_ID", "")
    RETURNZERO_CLIENT_SECRET = os.getenv("RETURNZERO_CLIENT_SECRET", "")
    
    # TTS 설정 (ElevenLabs)
    ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "").strip()
    TTS_VOICE_ID = os.getenv("TTS_VOICE_ID", "uyVNoMrnUku1dZyVEXwD").strip()
    TTS_MODEL = os.getenv("TTS_MODEL", "eleven_multilingual_v2").strip()
    
    # 오디오 설정
    SAMPLE_RATE = 16000
    CHUNK_SIZE = int(SAMPLE_RATE / 10)  # 100ms
    CHANNELS = 1
    
    # 시스템 설정
    DEBUG = os.getenv("DEBUG", "True").lower() == "true"
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # 대화 설정
    SESSION_TIMEOUT = 1800  # 30분
    SILENCE_TIMEOUT = 10.0 # 10초 침묵 시 반응

# 설정 클래스들을 인스턴스화하여 다른 파일에서 쉽게 가져다 쓸 수 있도록 함
settings = Settings()
ai_config = AIConfig()
detection_thresholds = DetectionThresholds()