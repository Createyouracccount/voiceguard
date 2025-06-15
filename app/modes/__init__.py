"""
VoiceGuard AI - 운영 모드 패키지
모든 운영 모드를 통합 관리
"""

from .base_mode import BaseMode, ModeState
from .detection_mode import DetectionMode
from .prevention_mode import PreventionMode

# 추가 모드들 (향후 구현)
class PostIncidentMode(BaseMode):
    """사후대처 모드 (간소화 버전)"""
    
    @property
    def mode_name(self) -> str:
        return "사후 대처"
    
    @property
    def mode_description(self) -> str:
        return "보이스피싱 피해 발생 후 대응 방안을 안내합니다"
    
    def _load_mode_config(self):
        return {'emergency_mode': True, 'auto_report': True}
    
    async def _initialize_mode(self) -> bool:
        print("🚨 사후대처 모드는 현재 개발 중입니다.")
        return True
    
    async def _run_mode_logic(self):
        print("""
🆘 보이스피싱 피해 대응 가이드

즉시 해야 할 일:
1. 📞 112 (경찰) 신고
2. 📞 1332 (금융감독원) 신고  
3. 📞 해당 은행 고객센터 연락
4. 💳 관련 계좌/카드 사용 정지
5. 📋 피해 내역 정리 및 증거 수집

법적 대응:
- 피해신고서 작성
- 관련 서류 준비
- 전문가 상담 (필요시)

심리적 지원:
- 가족/지인과 상의
- 전문 상담 서비스 이용
- 2차 피해 방지

🔄 이 기능은 향후 더 자세히 구현될 예정입니다.
        """)
        
        input("\n계속하려면 Enter를 누르세요...")

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
        print("💬 상담 모드는 현재 개발 중입니다.")
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

🔄 향후 AI 챗봇 기능이 추가될 예정입니다.
        """)
        
        input("\n계속하려면 Enter를 누르세요...")

# 모드 팩토리
MODE_REGISTRY = {
    'prevention': PreventionMode,
    'detection': DetectionMode,
    'post_incident': PostIncidentMode,
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
    info = {}
    for name, mode_class in MODE_REGISTRY.items():
        # 임시 인스턴스로 정보 추출 (실제로는 초기화 안함)
        info[name] = {
            'class': mode_class.__name__,
            'description': f"{name} 모드",
            'available': True
        }
    return info

__all__ = [
    'BaseMode', 
    'ModeState',
    'DetectionMode', 
    'PreventionMode', 
    'PostIncidentMode', 
    'ConsultationMode',
    'MODE_REGISTRY',
    'get_mode_class',
    'get_available_modes',
    'get_mode_info'
]