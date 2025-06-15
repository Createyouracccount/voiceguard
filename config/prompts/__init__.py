"""
VoiceGuard AI - 프롬프트 템플릿 패키지
AI 모델용 전문 프롬프트 모음
"""

from .detection_prompts import (
    DETECTION_PROMPTS,
    get_detection_prompt,
    create_adaptive_prompt
)

from .prevention_prompts import (
    PREVENTION_PROMPTS,
    get_prevention_prompt,
    create_lesson_content,
    create_quiz_prompt
)

# 프롬프트 타입 상수
class PromptTypes:
    """프롬프트 타입 상수"""
    
    # 탐지용 프롬프트
    BASIC_DETECTION = "basic"
    DETAILED_DETECTION = "detailed"
    EMERGENCY_DETECTION = "emergency"
    CONTEXTUAL_DETECTION = "contextual"
    SCAM_SPECIFIC = "scam_specific"
    
    # 교육용 프롬프트
    BASIC_EDUCATION = "basic_education"
    SCENARIO_TRAINING = "scenario_training"
    QUIZ_GENERATION = "quiz_generation"
    PERSONALIZED = "personalized"
    FAMILY_EDUCATION = "family_education"
    ADVANCED = "advanced"
    TRAUMA_CARE = "trauma_care"

# 통합 프롬프트 관리자
class PromptManager:
    """프롬프트 통합 관리 클래스"""
    
    def __init__(self):
        self.detection_prompts = DETECTION_PROMPTS
        self.prevention_prompts = PREVENTION_PROMPTS
    
    def get_prompt(self, category: str, prompt_type: str, **kwargs) -> str:
        """카테고리와 타입에 따른 프롬프트 반환"""
        
        if category == "detection":
            return get_detection_prompt(prompt_type, kwargs.get("scam_type"))
        elif category == "prevention":
            return get_prevention_prompt(prompt_type, kwargs.get("user_profile"))
        else:
            raise ValueError(f"Unknown prompt category: {category}. Available categories: detection, prevention")