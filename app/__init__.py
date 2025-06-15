"""
VoiceGuard AI - 운영 모드 패키지 (업데이트)
모든 운영 모드를 통합 관리 - 완전한 사후대처 모드 포함
"""

from app.modes.base_mode import BaseMode, ModeState
from app.modes.detection_mode import DetectionMode
from app.modes.prevention_mode import PreventionMode
from app.modes.post_incident_mode import PostIncidentMode  # 완전 구현된 버전 import

# 간소화된 상담 모드 (기존 유지)
class ConsultationMode(BaseMode):
    """상담 모드 (간소화 버전)"""
    
    @property
    def mode_name(self) -> str:
        return "상담 문의"
    
    @property
    def mode_description(self) -> str:
        return "보이스피싱 관련 질문에 답변드립니다"
    
    def _load_mode_config(self):
        return {'interactive_qa': True, 'knowledge_base': True}
    
    async def _initialize_mode(self) -> bool:
        print("💬 상담 모드를 시작합니다.")
        return True
    
    async def _run_mode_logic(self):
        print("""
💬 VoiceGuard 상담 서비스

자주 묻는 질문:

Q: 보이스피싱인지 어떻게 확인하나요?
A: 다음 특징이 있으면 의심하세요:
   - 전화로 개인정보/금융정보 요구
   - 긴급하게 돈을 요구
   - 공공기관 사칭
   - 앱 설치 요구

Q: 의심스러운 전화를 받았을 때 대처법은?
A: 1) 즉시 통화 끊기
   2) 해당 기관에 직접 확인
   3) 의심되면 112 신고

Q: 가족이 당했다고 하는데 진짜인가요?
A: 먼저 침착하게 가족에게 직접 연락해보세요.
   실제 응급상황이라면 112를 통해 확인 가능합니다.

Q: 개인정보를 알려줬는데 어떻게 하나요?
A: 즉시 관련 금융기관에 연락하여 계좌 모니터링을 
   강화하고, 필요시 계좌 변경을 고려하세요.

Q: 악성 앱을 설치했는데 어떻게 하나요?
A: 1) 즉시 휴대폰 초기화 또는 통신사 고객센터 방문
   2) 모든 금융 앱 재설치
   3) 비밀번호 전체 변경
   4) 공동인증서 재발급

Q: 돈을 송금했는데 되돌릴 수 있나요?
A: 1) 즉시 112 신고
   2) 1332 (금융감독원) 신고
   3) 해당 은행에 지급정지 신청
   4) 전기통신금융사기 피해 특별법에 따라 환급 가능

🔄 향후 AI 챗봇 기능이 추가될 예정입니다.
        """)
        
        input("\n계속하려면 Enter를 누르세요...")

# 모드 팩토리 (업데이트)
MODE_REGISTRY = {
    'prevention': PreventionMode,
    'detection': DetectionMode,
    'post_incident': PostIncidentMode,  # 완전 구현된 버전
    'consultation': ConsultationMode
}

def get_mode_class(mode_name: str):
    """모드 이름으로 클래스 반환"""
    return MODE_REGISTRY.get(mode_name)

def get_available_modes():
    """사용 가능한 모드 목록 반환"""
    return list(MODE_REGISTRY.keys())

def get_mode_info():
    """모드 정보 반환"""
    mode_descriptions = {
        'prevention': '🎓 예방 교육 - 보이스피싱 수법 학습 및 대응 훈련',
        'detection': '🔍 실시간 탐지 - 의심스러운 통화 내용 분석',
        'post_incident': '🚨 사후 대처 - 피해 발생 후 금융감독원 기준 체계적 대응',
        'consultation': '💬 상담 문의 - 보이스피싱 관련 질문 답변'
    }
    
    info = {}
    for name, mode_class in MODE_REGISTRY.items():
        info[name] = {
            'class': mode_class.__name__,
            'description': mode_descriptions[name],
            'available': True,
            'features': _get_mode_features(name)
        }
    return info

def _get_mode_features(mode_name: str) -> list:
    """모드별 특징 반환"""
    features = {
        'prevention': [
            '보이스피싱 수법 학습',
            '실전 시나리오 훈련',
            '지식 퀴즈',
            '개인별 진도 관리'
        ],
        'detection': [
            '실시간 텍스트 분석',
            'AI 기반 위험도 평가',
            '8가지 사기 유형 탐지',
            '즉시 경고 및 대응 안내'
        ],
        'post_incident': [
            '금융감독원 공식 절차',
            '단계별 체크리스트',
            '피해금 환급 안내',
            '명의도용 확인',
            '개인정보 보호 조치'
        ],
        'consultation': [
            '자주 묻는 질문 답변',
            '상황별 대처법 안내',
            '관련 기관 연락처',
            '예방 수칙 제공'
        ]
    }
    return features.get(mode_name, [])

# 모드 추천 시스템
def recommend_mode_for_situation(situation: str) -> str:
    """상황에 따른 모드 추천"""
    
    situation_lower = situation.lower()
    
    # 피해를 이미 당한 경우
    if any(keyword in situation_lower for keyword in 
           ['당했', '송금했', '속았', '피해', '이체했', '빼앗겼']):
        return 'post_incident'
    
    # 의심스러운 통화를 받고 있는 경우
    elif any(keyword in situation_lower for keyword in 
             ['지금', '전화', '말하고있', '통화중', '확인해달라']):
        return 'detection'
    
    # 학습하고 싶은 경우
    elif any(keyword in situation_lower for keyword in 
             ['배우고', '공부', '학습', '알고싶', '훈련']):
        return 'prevention'
    
    # 질문이 있는 경우
    elif any(keyword in situation_lower for keyword in 
             ['궁금', '질문', '문의', '물어보고']):
        return 'consultation'
    
    # 기본값
    else:
        return 'prevention'

def get_emergency_guidance() -> str:
    """긴급 상황 가이드"""
    return """
🚨 긴급 상황 대처법

📞 즉시 연락할 곳:
• 112 (경찰청) - 보이스피싱 신고
• 1332 (금융감독원) - 금융피해 신고
• 해당 은행 고객센터 - 지급정지 신청

⚡ 즉시 해야 할 일:
1. 통화 중이라면 즉시 끊기
2. 송금했다면 은행에 지급정지 신청
3. 개인정보 제공했다면 관련 기관에 신고
4. 앱 설치했다면 휴대폰 초기화

🛡️ 절대 하지 말 것:
• 추가 개인정보 제공
• 더 이상의 송금
• 의심스러운 링크 클릭
• 사기범과의 계속 연락

💡 VoiceGuard 사후대처 모드를 이용하여
   체계적인 대응 절차를 따라하세요!
"""

__all__ = [
    'BaseMode', 
    'ModeState',
    'DetectionMode', 
    'PreventionMode', 
    'PostIncidentMode',  # 완전 구현된 버전
    'ConsultationMode',
    'MODE_REGISTRY',
    'get_mode_class',
    'get_available_modes',
    'get_mode_info',
    'recommend_mode_for_situation',
    'get_emergency_guidance'
]