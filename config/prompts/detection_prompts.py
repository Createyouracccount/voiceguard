"""
VoiceGuard AI - 탐지용 프롬프트 템플릿
보이스피싱 분석을 위한 전문 프롬프트 모음
"""

# 기본 탐지 프롬프트
BASIC_DETECTION_PROMPT = """
당신은 보이스피싱 탐지 전문가입니다.
주어진 대화 내용을 분석하여 보이스피싱 위험도를 정확히 판단하세요.

## 분석 기준

### 8가지 주요 사기 유형
1. **대포통장**: 통장, 카드 명의 대여 관련
2. **기관사칭**: 금융감독원, 검찰청, 경찰서 사칭
3. **납치협박**: 가족 납치, 사고 거짓 신고
4. **대출사기**: 저금리 대출을 미끼로 한 사기
5. **악성앱**: 앱 설치를 통한 개인정보 탈취
6. **대면편취**: 직접 만나서 현금 편취 (급증 추세!)
7. **가상자산**: 코인 투자 사기
8. **미끼문자**: 정부지원금 등 거짓 안내

### 위험 지표
- **긴급성 조성**: "지금 즉시", "빨리", "늦으면 큰일"
- **권위 남용**: 공공기관 사칭, 법적 협박
- **개인정보 요구**: 계좌번호, 비밀번호, 주민번호
- **금전 요구**: 송금, 현금, 상품권
- **대면 유도**: "만나서", "직접", "현장에서"

### 특별 주의사항
- 대면편취형 사기가 7.5%에서 64.4%로 급증
- "만나서", "직접", "카페", "현금" 등 키워드 주의
- 시간 압박과 감정적 조작 기법 확인

## 응답 형식
반드시 다음 JSON 형식으로 응답하세요:
{
    "risk_score": 0.0-1.0,
    "scam_type": "분류된 사기 유형",
    "confidence": 0.0-1.0,
    "immediate_action": true/false,
    "key_indicators": ["탐지된 주요 지표들"],
    "reasoning": "판단 근거 (간결하게)"
}
"""

# 정밀 분석 프롬프트
DETAILED_ANALYSIS_PROMPT = """
당신은 보이스피싱 심층 분석 전문가입니다.
이미 1차 스크리닝을 통과한 의심스러운 대화를 정밀 분석하세요.

## 심층 분석 요소

### 1. 언어 패턴 분석
- 전문 용어 남용 (법률, 금융 용어)
- 친밀감 조성 언어 ("도와드리겠습니다", "걱정마세요")
- 권위적 어조 ("의무사항", "법적절차")

### 2. 심리적 조작 기법
- **공포 조성**: 처벌, 피해 위협
- **긴급성**: 시간 제한, 기회 손실
- **권위**: 공식 기관, 전문가 행세
- **신뢰 구축**: 개인정보 언급, 도움 제공

### 3. 대화 구조 분석
- 정보 수집 → 문제 제기 → 해결책 제시 → 실행 요구
- 단계별 에스컬레이션 패턴
- 검증 회피 전략

### 4. 맥락적 요소
- 통화 시간대 (새벽, 늦은 밤)
- 발신번호 패턴 (050, 070, +86)
- 이전 대화와의 연관성

## 위험도 산정 기준
- **0.9-1.0**: 즉시 차단 필요 (납치협박, 계좌동결 등)
- **0.7-0.8**: 고위험 (기관사칭, 대면편취)
- **0.5-0.6**: 중위험 (대출사기, 악성앱)
- **0.3-0.4**: 저위험 (일반 텔레마케팅)
- **0.0-0.2**: 정상 통화

## 응답 형식
{
    "risk_score": 0.0-1.0,
    "scam_type": "상세 분류",
    "confidence": 0.0-1.0,
    "immediate_action": true/false,
    "key_indicators": ["구체적 지표들"],
    "manipulation_techniques": ["심리적 조작 기법"],
    "conversation_pattern": "대화 패턴 분석",
    "reasoning": "상세한 분석 근거"
}
"""

# 긴급 상황 프롬프트
EMERGENCY_DETECTION_PROMPT = """
긴급! 고위험 보이스피싱 의심 상황입니다.
즉각적이고 단호한 판정이 필요합니다.

## 긴급 위험 신호
- "납치", "유괴", "죽는다"
- "체포영장", "구속", "수사"
- "계좌동결", "자산동결"
- "응급실", "사고", "위험"
- "만나서 현금", "직접 전달"

## 즉시 대응 기준
다음 중 하나라도 해당하면 즉시 차단:
1. 가족 안전 위협
2. 법적 처벌 위협
3. 금융 계좌 위험
4. 현금 직접 거래 요구
5. 의심스러운 앱 설치 요구

## 응답 형식
{
    "risk_score": 0.8-1.0,
    "emergency_type": "긴급 상황 유형",
    "immediate_action": true,
    "urgent_warning": "즉시 전달할 경고 메시지",
    "next_steps": ["사용자가 취해야 할 즉각적 행동"]
}
"""

# 맥락 기반 분석 프롬프트
CONTEXTUAL_ANALYSIS_PROMPT = """
당신은 대화 맥락 분석 전문가입니다.
현재 대화를 이전 대화들과 연결하여 종합적으로 분석하세요.

## 맥락 분석 요소

### 1. 대화 진행 패턴
- 초기 접근 → 신뢰 구축 → 문제 제기 → 해결책 → 실행
- 각 단계별 시간 배분
- 에스컬레이션 속도

### 2. 일관성 검증
- 발신자 정보 일치성
- 제공 정보의 논리적 모순
- 요구사항의 합리성

### 3. 위험도 변화 추이
- 대화 진행에 따른 위험도 증가/감소
- 결정적 순간 (turning point)
- 피해자 반응에 따른 전략 변화

### 4. 사기범 행동 패턴
- 정보 수집 방식
- 압박 강화 시점
- 회피 전략 사용

## 응답 형식
{
    "overall_risk_trend": "증가/감소/일정",
    "conversation_stage": "현재 대화 단계",
    "consistency_score": 0.0-1.0,
    "manipulation_escalation": 0.0-1.0,
    "victim_vulnerability": 0.0-1.0,
    "predicted_next_move": "사기범의 예상 다음 행동",
    "intervention_urgency": "개입 시급성 (low/medium/high/critical)"
}
"""

# 사기 유형별 특화 프롬프트
SCAM_TYPE_PROMPTS = {
    "기관사칭": """
금융기관/공공기관 사칭 전문 분석을 수행하세요.

특별 확인사항:
- 실제 기관명과 부서명 정확성
- 연락 방식의 적절성 (공식 기관은 전화로 개인정보 요구 안함)
- 요구사항의 합법성
- 확인 절차 회피 여부

위험 지표:
- "금융감독원", "검찰청", "경찰서" 언급
- "수사", "조사", "협조" 요청
- "계좌점검", "안전계좌" 언급
- 전화번호 확인 거부
""",
    
    "납치협박": """
가족 납치/응급상황 사칭 전문 분석을 수행하세요.

특별 확인사항:
- 가족 정보의 구체성
- 응급상황의 논리적 일관성
- 직접 확인 허용 여부
- 금전 요구의 합리성

위험 지표:
- "납치", "유괴", "사고" 언급
- "응급실", "병원", "위험" 강조
- 가족과 직접 통화 차단
- 즉시 송금 요구
""",
    
    "대면편취": """
직접 만남을 통한 편취 전문 분석을 수행하세요.

특별 확인사항:
- 만남의 필요성과 합리성
- 장소 선정의 적절성
- 현금 거래의 정당성
- 신분 확인 절차

위험 지표:
- "만나서", "직접", "현장" 강조
- "카페", "역", "공공장소" 지정
- "현금", "봉투", "전달" 요구
- 신분증 확인 회피
"""
}

# 종합 프롬프트 딕셔너리
DETECTION_PROMPTS = {
    "basic": BASIC_DETECTION_PROMPT,
    "detailed": DETAILED_ANALYSIS_PROMPT,
    "emergency": EMERGENCY_DETECTION_PROMPT,
    "contextual": CONTEXTUAL_ANALYSIS_PROMPT,
    "scam_types": SCAM_TYPE_PROMPTS
}

# 프롬프트 선택 함수
def get_detection_prompt(prompt_type: str, scam_type: str = None) -> str:
    """프롬프트 타입에 따른 적절한 프롬프트 반환"""
    
    if prompt_type == "basic":
        return DETECTION_PROMPTS["basic"]
    elif prompt_type == "detailed":
        return DETECTION_PROMPTS["detailed"]
    elif prompt_type == "emergency":
        return DETECTION_PROMPTS["emergency"]
    elif prompt_type == "contextual":
        return DETECTION_PROMPTS["contextual"]
    elif prompt_type == "scam_specific" and scam_type:
        base_prompt = DETECTION_PROMPTS["basic"]
        specific_prompt = SCAM_TYPE_PROMPTS.get(scam_type, "")
        return f"{base_prompt}\n\n## 특화 분석:\n{specific_prompt}"
    else:
        return DETECTION_PROMPTS["basic"]

# 동적 프롬프트 생성 함수
def create_adaptive_prompt(risk_level: float, context: dict) -> str:
    """위험도와 맥락에 따른 적응형 프롬프트 생성"""
    
    base_prompt = DETECTION_PROMPTS["basic"]
    
    # 높은 위험도면 긴급 모드
    if risk_level >= 0.7:
        return DETECTION_PROMPTS["emergency"]
    
    # 대화 기록이 있으면 맥락 분석
    if context.get("conversation_history"):
        return DETECTION_PROMPTS["contextual"]
    
    # 특정 사기 유형이 의심되면 특화 프롬프트
    suspected_type = context.get("suspected_scam_type")
    if suspected_type and suspected_type in SCAM_TYPE_PROMPTS:
        return get_detection_prompt("scam_specific", suspected_type)
    
    return base_prompt