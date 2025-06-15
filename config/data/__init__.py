"""
VoiceGuard AI - 데이터 패키지
교육 컨텐츠 및 설정 데이터
"""

try:
    from .education_content import EDUCATION_SCENARIOS, QUIZ_QUESTIONS, LEARNING_PATH, ACHIEVEMENTS
except ImportError:
    # 기본 데이터 제공
    EDUCATION_SCENARIOS = []
    QUIZ_QUESTIONS = []
    LEARNING_PATH = {}
    ACHIEVEMENTS = {}

__all__ = [
    'EDUCATION_SCENARIOS',
    'QUIZ_QUESTIONS', 
    'LEARNING_PATH',
    'ACHIEVEMENTS'
]