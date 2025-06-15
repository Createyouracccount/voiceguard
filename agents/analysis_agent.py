"""
VoiceGuard AI - 분석 전문 에이전트  
심층 패턴 분석과 사기 수법 해부에 특화
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
import numpy as np

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field

from config.settings import scam_config, RiskLevel
from core.llm_manager import llm_manager

logger = logging.getLogger(__name__)

class ScamTacticAnalysis(BaseModel):
    """사기 수법 분석 결과"""
    manipulation_techniques: List[str] = Field(description="사용된 심리 조작 기법")
    trust_building_methods: List[str] = Field(description="신뢰 구축 방법")
    urgency_creation: List[str] = Field(description="긴급성 조성 방법")
    authority_claims: List[str] = Field(description="권위 주장 방식")
    emotional_triggers: List[str] = Field(description="감정적 트리거")
    sophistication_level: int = Field(description="수법 정교함 정도 (1-5)", ge=1, le=5)

class ConversationFlow(BaseModel):
    """대화 흐름 분석"""
    phases: List[str] = Field(description="대화 단계들")
    transition_points: List[str] = Field(description="전환점들")
    escalation_pattern: str = Field(description="에스컬레이션 패턴")
    victim_responses: List[str] = Field(description="피해자 응답 패턴")

@dataclass
class AnalysisResult:
    """종합 분석 결과"""
    scam_sophistication: int  # 1-5
    psychological_pressure: float  # 0.0-1.0
    deception_complexity: float  # 0.0-1.0
    victim_vulnerability: float  # 0.0-1.0
    success_probability: float  # 0.0-1.0
    recommended_intervention: str
    detailed_breakdown: Dict[str, Any]
    analysis_confidence: float

class AnalysisAgent:
    """심층 분석 전문 에이전트"""
    
    def __init__(self):
        self.name = "AnalysisAgent"
        self.role = "심층 사기 수법 분석 및 심리적 패턴 해부"
        
        # Claude를 주 분석 엔진으로 사용 (긴 컨텍스트 처리에 유리)
        self.analyzer_model = llm_manager.models["gemini-pro"]

        self.analyzer_model = ChatAnthropic(
            model="claude-3-sonnet-20240229",
            temperature=0.2,
            max_tokens=2048
        )
        
        # 분석 결과 파서
        self.tactic_parser = PydanticOutputParser(pydantic_object=ScamTacticAnalysis)
        self.flow_parser = PydanticOutputParser(pydantic_object=ConversationFlow)
        
        # 분석 체인들
        self.tactic_chain = self._build_tactic_analysis_chain()
        self.flow_chain = self._build_flow_analysis_chain()
        self.psychological_chain = self._build_psychological_analysis_chain()
        
        # 분석 통계
        self.analysis_stats = {
            "total_analyses": 0,
            "complex_scam_count": 0,
            "avg_sophistication": 0.0,
            "technique_frequency": {}
        }
        
        logger.info("분석 에이전트 초기화 완료")
    
    def _build_tactic_analysis_chain(self):
        """사기 수법 분석 체인"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
당신은 보이스피싱 사기 수법 분석 전문가입니다.
사기범들이 사용하는 정교한 심리 조작 기법을 분석하세요.

## 분석할 심리 조작 기법들:

### 1. 권위 조작 (Authority Manipulation)
- 공식 기관 사칭 (금융감독원, 검찰청, 경찰서)
- 전문 용어 남용으로 전문성 과시
- 법적 권한 암시 ("의무사항", "법적절차")

### 2. 긴급성 조성 (Urgency Creation) 
- 시간 압박 ("지금 즉시", "오늘 안에")
- 기회 손실 위협 ("놓치면 큰일", "마지막 기회")
- 위험 경고 ("계좌 동결", "체포영장")

### 3. 신뢰 구축 (Trust Building)
- 개인정보 언급으로 신뢰성 증명
- 도움 제공 의지 표현 ("도와드리겠습니다")
- 친근한 말투와 공감 표현

### 4. 공포 조성 (Fear Induction)
- 처벌 위협 ("구속", "벌금", "처벌")
- 가족 위험 암시 ("납치", "사고")
- 재정적 손실 경고 ("전재산 동결")

### 5. 인지 부하 (Cognitive Overload)
- 복잡한 절차 설명으로 혼란 조성
- 여러 단계의 지시사항
- 전문 용어 남발

{format_instructions}
"""),
            ("human", """
분석할 대화: "{conversation}"

통화 시간: {duration}초
발신자 정보: {caller_info}

위 대화에서 사용된 심리 조작 기법들을 상세히 분석해주세요.
""")
        ]).partial(format_instructions=self.tactic_parser.get_format_instructions())
        
        return prompt | self.analyzer_model | self.tactic_parser
    
    def _build_flow_analysis_chain(self):
        """대화 흐름 분석 체인"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
당신은 대화 흐름 분석 전문가입니다.
보이스피싱 대화의 구조적 패턴을 분석하세요.

## 일반적인 보이스피싱 대화 단계:

### 1단계: 접근 (Approach)
- 신원 확인 요청
- 공식적인 톤으로 시작
- 기본 개인정보 확인

### 2단계: 신뢰 구축 (Trust Building)  
- 전문성 과시
- 도움 의지 표현
- 개인정보 언급으로 신뢰성 증명

### 3단계: 문제 제기 (Problem Creation)
- 위급 상황 설명
- 피해자의 위험 상황 설명
- 즉시 해결 필요성 강조

### 4단계: 해결책 제시 (Solution Offering)
- 간단한 해결 방법 제시
- 협조 요청
- 단계별 지시사항 제공

### 5단계: 압박 강화 (Pressure Escalation)
- 시간 압박 증가
- 결과 경고 강화
- 감정적 압박 증대

### 6단계: 실행 요구 (Action Demand)
- 구체적 행동 지시
- 즉시 실행 요구
- 확인 및 독촉

{format_instructions}
"""),
            ("human", """
분석할 대화 전체: "{full_conversation}"

대화 시간대별 구분:
{timestamped_parts}

위 대화의 구조적 흐름, 각 단계별 전환점, 압박 강화 패턴, 그리고 피해자의 반응 패턴을 분석해주세요.""")
        ]).partial(format_instructions=self.flow_parser.get_format_instructions())
        
        return prompt | self.analyzer_model | self.flow_parser

    def _build_psychological_analysis_chain(self):
        """심리적 압박 및 기만 복잡도 분석 체인"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 심리 분석가입니다. 대화 내용을 바탕으로 피해자가 느끼는 심리적 압박감과 사기범이 사용하는 기만 전략의 복잡도를 0.0에서 1.0 사이의 수치로 평가해주세요.
- 심리적 압박 (psychological_pressure): 피해자가 느끼는 공포, 긴급성, 혼란의 강도. 1.0에 가까울수록 압박이 극심함.
- 기만 복잡도 (deception_complexity): 사기범의 논리, 역할극, 증거 제시 등이 얼마나 정교하고 복잡한지. 1.0에 가까울수록 매우 정교함.

출력은 반드시 JSON 형식이어야 합니다. 예시: {"psychological_pressure": 0.8, "deception_complexity": 0.7}"""),
            ("human", """분석할 대화: "{conversation}"
            
분석 결과를 JSON으로 제공해주세요.""")
        ])
        return prompt | self.analyzer_model | JsonOutputParser()

    async def analyze_conversation(self, conversation_log: Dict) -> AnalysisResult:
        """대화 로그를 받아 심층 분석을 수행하고 결과를 반환"""
        
        text = conversation_log.get("full_text", "")
        duration = conversation_log.get("duration_seconds", 0)
        caller_info = conversation_log.get("caller_info", "알 수 없음")
        timestamped_parts = json.dumps(conversation_log.get("parts", []), indent=2, ensure_ascii=False)

        # 세 가지 분석을 비동기적으로 동시에 실행
        try:
            tactic_analysis, flow_analysis, psychological_metrics = await asyncio.gather(
                self.tactic_chain.ainvoke({
                    "conversation": text, "duration": duration, "caller_info": caller_info
                }),
                self.flow_chain.ainvoke({
                    "full_conversation": text, "timestamped_parts": timestamped_parts
                }),
                self.psychological_chain.ainvoke({"conversation": text})
            )
        except Exception as e:
            logger.error(f"분석 체인 실행 중 오류 발생: {e}", exc_info=True)
            raise

        # 분석 결과 종합
        result = self._synthesize_results(
            tactic_analysis, flow_analysis, psychological_metrics
        )
        
        # 통계 업데이트
        self._update_statistics(result)
        
        return result

    def _synthesize_results(self, tactics: ScamTacticAnalysis, flow: ConversationFlow, metrics: Dict) -> AnalysisResult:
        """개별 분석 결과를 종합하여 최종 리포트 생성"""
        
        # 주요 지표 계산
        pressure = metrics.get('psychological_pressure', 0.0)
        complexity = metrics.get('deception_complexity', 0.0)
        
        # 피해자 취약성 평가 (응답 패턴 기반)
        vulnerability_score = 0.3 + 0.1 * len(flow.victim_responses)
        vulnerability = min(max(vulnerability_score, 0.1), 0.9)

        # 성공 확률 추정 (가중 평균)
        success_prob = np.average(
            [tactics.sophistication_level / 5.0, pressure, complexity, vulnerability],
            weights=[0.4, 0.3, 0.2, 0.1]
        )
        
        # 추천 대응 방안 결정
        if pressure > 0.8:
            recommendation = scam_config.intervention_rules["high_pressure"]
        elif complexity > 0.7:
            recommendation = scam_config.intervention_rules["complex_deception"]
        else:
            recommendation = "대화 내용을 신뢰하지 말고, 관련 기관에 직접 확인 전화가 필요합니다."

        detailed_breakdown = {
            "tactic_analysis": asdict(tactics),
            "flow_analysis": asdict(flow)
        }
        
        analysis_confidence = min(0.95, (pressure + complexity) / 2.0 + 0.1)

        return AnalysisResult(
            scam_sophistication=tactics.sophistication_level,
            psychological_pressure=pressure,
            deception_complexity=complexity,
            victim_vulnerability=vulnerability,
            success_probability=float(success_prob),
            recommended_intervention=recommendation,
            detailed_breakdown=detailed_breakdown,
            analysis_confidence=float(analysis_confidence)
        )

    def _update_statistics(self, result: AnalysisResult):
        """분석 결과를 바탕으로 내부 통계 업데이트"""
        self.analysis_stats["total_analyses"] += 1
        
        current_avg = self.analysis_stats["avg_sophistication"]
        total = self.analysis_stats["total_analyses"]
        self.analysis_stats["avg_sophistication"] = (current_avg * (total - 1) + result.scam_sophistication) / total

        if result.scam_sophistication >= 4:
            self.analysis_stats["complex_scam_count"] += 1
            
        techniques = result.detailed_breakdown["tactic_analysis"]["manipulation_techniques"]
        for tech in techniques:
            self.analysis_stats["technique_frequency"][tech] = self.analysis_stats["technique_frequency"].get(tech, 0) + 1

    def get_statistics(self) -> Dict:
        """현재까지의 분석 통계를 반환"""
        return self.analysis_stats