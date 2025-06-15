# import os
# from pathlib import Path
# from enum import Enum
# from dotenv import load_dotenv

# # .env 파일 로드
# BASE_DIR = Path(__file__).resolve().parent.parent
# dotenv_path = BASE_DIR / '.env'
# load_dotenv(dotenv_path)

# # --- AI 모델 및 위험도 관련 설정 클래스들 (LLMManager가 필요로 함) ---

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
#         temperature = 0.1
#         max_tokens = 2048
#     class GPT35_TURBO:
#         temperature = 0.1
#         max_tokens = 1024
#     class GEMINI:
#         temperature = 0.1
#         max_tokens = 2048
#     GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# class DetectionThresholds:
#     medium_risk = 0.4
#     high_risk = 0.7
#     critical_risk = 0.85

# # --- 기존의 하드웨어 및 서비스 관련 설정 ---

# class Settings:
#     """애플리케이션 설정"""
#     # LLM API 키
#     OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
#     GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
    
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

# # 설정 클래스들을 인스턴스화하여 다른 파일에서 쉽게 가져다 쓸 수 있도록 함
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

# --- AI 모델 및 위험도 관련 설정 클래스들 ---

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
        cost_per_1k_tokens = 0.03
    class GPT35_TURBO:
        temperature = 0.1
        max_tokens = 1024
        cost_per_1k_tokens = 0.001
    class GEMINI:
        temperature = 0.1
        max_tokens = 2048
        cost_per_1k_tokens = 0.012
    class CLAUDE:
        cost_per_1k_tokens = 0.024

class DetectionThresholds:
    medium_risk = 0.4
    high_risk = 0.7
    critical_risk = 0.85

# --- 사기 설정 추가 ---
class ScamConfig:
    """사기 유형별 설정"""
    
    # 8가지 주요 사기 유형
    SCAM_CATEGORIES = {
        "대포통장": {
            "keywords": ["통장", "계좌", "대여", "명의", "신분증", "카드"],
            "weight": 0.8,
            "description": "통장, 카드 명의 대여 관련"
        },
        "대포폰": {
            "keywords": ["휴대폰", "개통", "유심", "명의", "통신사"],
            "weight": 0.7,
            "description": "휴대폰, 유심 개통 관련"
        },
        "악성앱": {
            "keywords": ["앱", "설치", "다운로드", "권한", "허용", "업데이트"],
            "weight": 0.85,
            "description": "앱 설치, 권한 허용 관련"
        },
        "미끼문자": {
            "keywords": ["정부지원금", "환급", "당첨", "코로나", "재난지원금"],
            "weight": 0.6,
            "description": "정부지원금, 환급, 당첨 관련"
        },
        "기관사칭": {
            "keywords": ["금융감독원", "검찰청", "경찰서", "국세청", "수사", "조사"],
            "weight": 0.9,
            "description": "금융감독원, 검찰청, 경찰서 사칭"
        },
        "납치협박": {
            "keywords": ["납치", "유괴", "사고", "응급실", "위험", "죽는다"],
            "weight": 1.0,
            "description": "납치, 사고, 응급실 언급"
        },
        "대출사기": {
            "keywords": ["대출", "저금리", "무담보", "승인", "융자"],
            "weight": 0.7,
            "description": "저금리, 무담보 대출 제안"
        },
        "가상자산": {
            "keywords": ["비트코인", "코인", "가상화폐", "투자", "수익", "거래소"],
            "weight": 0.75,
            "description": "비트코인, 투자 수익 제안"
        }
    }
    
    # 대면편취 지표 (급증 추세)
    FACE_TO_FACE_INDICATORS = [
        "만나서", "직접", "현장", "카페", "현금", "와서", "가져와"
    ]
    
    # 개입 규칙
    intervention_rules = {
        "high_pressure": "즉시 통화를 중단하고 112에 신고하세요.",
        "complex_deception": "이 통화는 정교한 사기일 가능성이 높습니다. 통화를 끊고 관련 기관에 직접 확인하세요."
    }

class VoiceConfig:
    """음성 관련 설정"""
    SAMPLE_RATE = 16000
    CHUNK_SIZE = int(SAMPLE_RATE / 10)  # 100ms
    CHANNELS = 1
    STT_MODEL = "whisper-large-v3"
    STT_LANGUAGE = "ko"

class MonitoringConfig:
    """모니터링 설정"""
    ENABLE_LANGSMITH = False  # 일단 비활성화
    LANGSMITH_PROJECT = "voiceguard-ai"
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
    TARGET_RESPONSE_TIME = 3.0  # 3초
    TARGET_ACCURACY = 0.85
    ALERT_WEBHOOK_URL = os.getenv("ALERT_WEBHOOK_URL", "")

class IntegrationConfig:
    """통합 설정"""
    pass

# --- 기존의 하드웨어 및 서비스 관련 설정 ---

class Settings:
    """애플리케이션 설정"""
    # LLM API 키
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")  # 추가
    
    # STT 설정 (ReturnZero)
    RETURNZERO_CLIENT_ID = os.getenv("RETURNZERO_CLIENT_ID", "")
    RETURNZERO_CLIENT_SECRET = os.getenv("RETURNZERO_CLIENT_SECRET", "")
    
    # TTS 설정 (ElevenLabs)
    ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "").strip()
    TTS_VOICE_ID = os.getenv("TTS_VOICE_ID", "uyVNoMrnUku1dZyVEXwD").strip()
    TTS_MODEL = os.getenv("TTS_MODEL", "eleven_multilingual_v2").strip()
    TTS_OPTIMIZE_LATENCY = True
    
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

# 설정 클래스들을 인스턴스화
settings = Settings()
ai_config = AIConfig()
detection_thresholds = DetectionThresholds()
scam_config = ScamConfig()
voice_config = VoiceConfig()
monitoring_config = MonitoringConfig()
integration_config = IntegrationConfig()