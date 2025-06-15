"""
VoiceGuard AI - Response Agent
보이스피싱 대응 전략 수립 전문 에이전트
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from core.llm_manager import llm_manager
from config.settings import (
    scam_config, detection_thresholds, RiskLevel,
    integration_config
)
from monitoring.langsmith_tracker import tracker

logger = logging.getLogger(__name__)

class InterventionStrategy(BaseModel):
    """개입 전략"""
    action_type: str = Field(description="개입 유형 (warn/block/report)")
    message_to_user: str = Field(description="사용자에게 전달할 메시지")
    confidence: float = Field(description="전략 신뢰도", ge=0.0, le=1.0)
    urgency: str = Field(description="긴급도 (low/medium/high/critical)")
    
class ReportingInfo(BaseModel):
    """신고 정보"""
    should_report: bool = Field(description="신고 필요 여부")
    agencies: List[str] = Field(description="신고할 기관 목록")
    evidence_summary: str = Field(description="증거 요약")
    report_template: str = Field(description="신고서 템플릿")

class UserGuidance(BaseModel):
    """사용자 안내"""
    immediate_actions: List[str] = Field(description="즉시 취해야 할 행동")
    verification_steps: List[str] = Field(description="확인 절차")
    safety_tips: List[str] = Field(description="안전 수칙")
    support_resources: List[str] = Field(description="지원 리소스")

class ResponseStrategy(BaseModel):
    """종합 대응 전략"""
    intervention: InterventionStrategy
    reporting: ReportingInfo
    guidance: UserGuidance
    follow_up_required: bool = Field(description="후속 조치 필요 여부")
    estimated_prevention_rate: float = Field(description="예상 예방률", ge=0.0, le=1.0)

class ResponseAgent:
    """대응 전략 수립 에이전트"""
    
    def __init__(self):
        self.name = "ResponseAgent"
        self.strategy_parser = PydanticOutputParser(pydantic_object=ResponseStrategy)
        
        # 대응 프롬프트
        self.response_prompt = self._build_response_prompt()
        self.emergency_prompt = self._build_emergency_prompt()
        
        # 대응 템플릿
        self.response_templates = self._load_response_templates()
        
        # 통계
        self.stats = {
            "total_responses": 0,
            "successful_interventions": 0,
            "reports_generated": 0
        }
        
        logger.info("Response Agent 초기화 완료")
    
    def _build_response_prompt(self) -> ChatPromptTemplate:
        """대응 전략 프롬프트"""
        return ChatPromptTemplate.from_messages([
            ("system", """
당신은 보이스피싱 대응 전문가입니다.
탐지된 위협에 대한 효과적인 대응 전략을 수립하세요.

## 대응 원칙
1. **사용자 안전 최우선**: 금전적, 정신적 피해 방지
2. **명확한 지시**: 혼란 없는 구체적 행동 지침
3. **신속한 대응**: 골든타임 내 개입
4. **증거 보존**: 향후 신고를 위한 기록
5. **심리적 지원**: 패닉 방지 및 안정화

## 대응 수준
- **경고 (warn)**: 주의 메시지, 확인 절차 안내
- **차단 (block)**: 즉시 통화 종료, 번호 차단
- **신고 (report)**: 관계 기관 자동 신고

## 사기 유형별 특화 대응
- 기관사칭: 직접 확인 강조, 공식 연락처 제공
- 납치협박: 112 신고, 가족 확인 절차
- 대출사기: 금융감독원 확인, 정식 절차 안내
- 대면편취: 절대 만나지 말 것, 현금 거래 금지

{format_instructions}
"""),
            ("human", """
분석 결과:
- 위험도: {risk_level}
- 위험 점수: {risk_score}
- 사기 유형: {scam_type}
- 주요 증거: {evidence}
- 긴급도: {urgency}

이 상황에 대한 최적의 대응 전략을 수립해주세요.
""")
        ]).partial(format_instructions=self.strategy_parser.get_format_instructions())
    
    def _build_emergency_prompt(self) -> ChatPromptTemplate:
        """긴급 대응 프롬프트"""
        return ChatPromptTemplate.from_messages([
            ("system", """
긴급 상황입니다. 즉각적이고 단호한 대응이 필요합니다.

## 긴급 대응 지침
1. 즉시 통화 종료 권고
2. 112 또는 관련 기관 신고
3. 추가 피해 방지 조치
4. 명확하고 강력한 경고 메시지

한국어로 간단명료하게 응답하세요.
"""),
            ("human", "긴급: {situation}\n\n즉시 전달할 경고 메시지를 생성하세요.")
        ])
    
    def _load_response_templates(self) -> Dict[str, Dict[str, str]]:
        """대응 템플릿 로드"""
        return {
            "기관사칭": {
                "warning": "⚠️ 주의: 금융감독원이나 검찰은 전화로 개인정보를 요구하지 않습니다.",
                "action": "통화를 끊고 해당 기관에 직접 확인하세요.",
                "contact": "금융감독원: 1332, 검찰청: 1301"
            },
            "납치협박": {
                "warning": "🚨 위험: 납치 협박 의심됩니다.",
                "action": "즉시 112에 신고하고 가족에게 직접 연락하세요.",
                "contact": "긴급신고: 112"
            },
            "대출사기": {
                "warning": "⚠️ 주의: 정식 금융기관은 선입금을 요구하지 않습니다.",
                "action": "금융감독원에서 정식 등록된 업체인지 확인하세요.",
                "contact": "금융감독원: 1332"
            },
            "대면편취": {
                "warning": "🚫 경고: 현금을 들고 만나는 것은 매우 위험합니다.",
                "action": "절대 만나지 마시고 경찰에 신고하세요.",
                "contact": "경찰: 112"
            }
        }
    
    @tracker.track_detection
    async def process_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """작업 처리 메인 메서드"""
        
        risk_level = task_data.get("risk_level", "medium")
        risk_score = task_data.get("risk_score", 0.5)
        scam_type = task_data.get("scam_type", "unknown")
        evidence = task_data.get("evidence", [])
        
        start_time = datetime.now()
        
        try:
            # 1. 긴급도 판단
            urgency = self._determine_urgency(risk_score, scam_type)
            
            # 2. 긴급 상황 처리
            if urgency == "critical":
                return await self._handle_emergency(task_data)
            
            # 3. 일반 대응 전략 수립
            strategy = await self._generate_response_strategy(
                risk_level, risk_score, scam_type, evidence, urgency
            )
            
            # 4. 사용자 메시지 생성
            user_message = self._create_user_message(strategy, scam_type)
            
            # 5. 신고 필요시 신고서 준비
            if strategy.reporting.should_report:
                report_data = self._prepare_report(task_data, strategy)
            else:
                report_data = None
            
            # 6. 통계 업데이트
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_stats(strategy)
            
            return {
                "agent": self.name,
                "timestamp": datetime.now().isoformat(),
                "strategy": strategy.dict(),
                "user_message": user_message,
                "report_data": report_data,
                "processing_time": processing_time,
                "action_required": strategy.intervention.action_type,
                "follow_up_required": strategy.follow_up_required
            }
            
        except Exception as e:
            logger.error(f"Response Agent 작업 처리 실패: {e}")
            return self._create_error_response(str(e))
    
    def _determine_urgency(self, risk_score: float, scam_type: str) -> str:
        """긴급도 결정"""
        
        # 납치협박은 항상 최고 긴급도
        if scam_type == "납치협박":
            return "critical"
        
        # 위험도 기반 긴급도
        if risk_score >= detection_thresholds.critical_risk:
            return "critical"
        elif risk_score >= detection_thresholds.high_risk:
            return "high"
        elif risk_score >= detection_thresholds.medium_risk:
            return "medium"
        else:
            return "low"
    
    async def _handle_emergency(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """긴급 상황 처리"""
        
        scam_type = task_data.get("scam_type", "unknown")
        situation = f"{scam_type} 유형의 고위험 보이스피싱 탐지"
        
        # 긴급 메시지 생성
        chain = self.emergency_prompt | llm_manager.models["gpt-3.5-turbo"]
        
        try:
            response = await chain.ainvoke({"situation": situation})
            emergency_message = response.content
        except:
            # 폴백 메시지
            emergency_message = "🚨 위험! 즉시 통화를 끊으세요! 보이스피싱이 의심됩니다."
        
        # 긴급 대응 전략
        strategy = ResponseStrategy(
            intervention=InterventionStrategy(
                action_type="block",
                message_to_user=emergency_message,
                confidence=0.95,
                urgency="critical"
            ),
            reporting=ReportingInfo(
                should_report=True,
                agencies=["경찰청", "금융감독원"],
                evidence_summary=json.dumps(task_data.get("evidence", []), ensure_ascii=False),
                report_template=self._generate_emergency_report(task_data)
            ),
            guidance=UserGuidance(
                immediate_actions=[
                    "즉시 통화를 끊으세요",
                    "112에 신고하세요",
                    "계좌 이체를 중단하세요"
                ],
                verification_steps=[],
                safety_tips=["절대 개인정보를 제공하지 마세요"],
                support_resources=["112", "1332", "118"]
            ),
            follow_up_required=True,
            estimated_prevention_rate=0.9
        )
        
        return {
            "agent": self.name,
            "timestamp": datetime.now().isoformat(),
            "emergency": True,
            "strategy": strategy.dict(),
            "user_message": emergency_message,
            "action_required": "block",
            "processing_time": 0.1  # 긴급 처리
        }
    
    async def _generate_response_strategy(self,
                                        risk_level: str,
                                        risk_score: float,
                                        scam_type: str,
                                        evidence: List[str],
                                        urgency: str) -> ResponseStrategy:
        """대응 전략 생성"""
        
        chain = self.response_prompt | llm_manager.models["gpt-4"] | self.strategy_parser
        
        try:
            strategy = await chain.ainvoke({
                "risk_level": risk_level,
                "risk_score": risk_score,
                "scam_type": scam_type,
                "evidence": ", ".join(evidence[:5]),  # 상위 5개 증거
                "urgency": urgency
            })
            
            return strategy
            
        except Exception as e:
            logger.error(f"전략 생성 실패: {e}")
            # 폴백 전략
            return self._create_fallback_strategy(risk_level, scam_type)
    
    def _create_user_message(self, strategy: ResponseStrategy, scam_type: str) -> str:
        """사용자 메시지 생성"""
        
        # 템플릿 기반 메시지
        if scam_type in self.response_templates:
            template = self.response_templates[scam_type]
            base_message = f"{template['warning']}\n\n{template['action']}\n\n{template['contact']}"
        else:
            base_message = strategy.intervention.message_to_user
        
        # 추가 안내사항
        if strategy.guidance.immediate_actions:
            base_message += "\n\n📋 즉시 행동사항:"
            for action in strategy.guidance.immediate_actions[:3]:
                base_message += f"\n• {action}"
        
        return base_message
    
    def _prepare_report(self, task_data: Dict[str, Any], 
                       strategy: ResponseStrategy) -> Dict[str, Any]:
        """신고서 준비"""
        
        return {
            "report_id": f"report_{datetime.now().timestamp()}",
            "timestamp": datetime.now().isoformat(),
            "scam_type": task_data.get("scam_type"),
            "risk_score": task_data.get("risk_score"),
            "evidence": task_data.get("evidence", []),
            "agencies": strategy.reporting.agencies,
            "template": strategy.reporting.report_template,
            "status": "prepared"
        }
    
    def _generate_emergency_report(self, task_data: Dict[str, Any]) -> str:
        """긴급 신고서 생성"""
        
        return f"""
[긴급 보이스피싱 신고]
발생시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
사기유형: {task_data.get('scam_type', '미상')}
위험도: 매우 높음
주요 증거: {', '.join(task_data.get('evidence', [])[:3])}
"""
    
    def _create_fallback_strategy(self, risk_level: str, scam_type: str) -> ResponseStrategy:
        """폴백 전략"""
        
        action_type = "block" if risk_level in ["high", "critical"] else "warn"
        
        return ResponseStrategy(
            intervention=InterventionStrategy(
                action_type=action_type,
                message_to_user="보이스피싱이 의심됩니다. 통화를 중단하고 확인하세요.",
                confidence=0.7,
                urgency=risk_level
            ),
            reporting=ReportingInfo(
                should_report=risk_level in ["high", "critical"],
                agencies=["경찰청"],
                evidence_summary="자동 생성 실패",
                report_template="표준 신고 템플릿"
            ),
            guidance=UserGuidance(
                immediate_actions=["통화 중단", "번호 확인", "기관 문의"],
                verification_steps=["발신번호 확인", "내용 확인"],
                safety_tips=["개인정보 보호", "송금 중단"],
                support_resources=["112", "1332"]
            ),
            follow_up_required=True,
            estimated_prevention_rate=0.7
        )
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """오류 응답 생성"""
        
        return {
            "agent": self.name,
            "timestamp": datetime.now().isoformat(),
            "error": error_message,
            "action_required": "warn",
            "user_message": "시스템 오류가 발생했습니다. 주의하시고 의심스러운 통화는 즉시 끊으세요.",
            "follow_up_required": True
        }
    
    def _update_stats(self, strategy: ResponseStrategy):
        """통계 업데이트"""
        
        self.stats["total_responses"] += 1
        
        if strategy.reporting.should_report:
            self.stats["reports_generated"] += 1
        
        if strategy.intervention.action_type in ["block", "report"]:
            self.stats["successful_interventions"] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """통계 반환"""
        
        return {
            **self.stats,
            "intervention_rate": (
                self.stats["successful_interventions"] / 
                max(1, self.stats["total_responses"])
            )
        }

# 전역 인스턴스는 __init__.py에서 생성됨