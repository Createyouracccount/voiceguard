"""
VoiceGuard AI - LangChain 기반 탐지 워크플로우
복잡한 사기 패턴을 체계적으로 분석하는 다단계 체인
"""

from typing import Dict, List, Any, Optional, Tuple
import asyncio
from datetime import datetime
import logging

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from pydantic import BaseModel, Field

from core.llm_manager import llm_manager
from config.settings import scam_config, detection_thresholds, RiskLevel


from langchain_core.prompts import ChatPromptTemplate, PromptTemplate







logger = logging.getLogger(__name__)

class ScamAnalysisResult(BaseModel):
    """사기 분석 결과 모델"""
    scam_type: str = Field(description="분류된 사기 유형")
    risk_score: float = Field(description="위험도 점수 (0.0-1.0)", ge=0.0, le=1.0)
    confidence: float = Field(description="분석 신뢰도 (0.0-1.0)", ge=0.0, le=1.0)
    immediate_action: bool = Field(description="즉시 대응 필요 여부")
    key_indicators: List[str] = Field(description="탐지된 주요 지표들")
    reasoning: str = Field(description="판단 근거")
    suggested_responses: List[str] = Field(description="권장 대응 방안")

class CallContextAnalysis(BaseModel):
    """통화 맥락 분석 결과"""
    conversation_flow: str = Field(description="대화 흐름 분석")
    emotional_state: str = Field(description="감정 상태 분석")
    urgency_level: int = Field(description="긴급도 (1-5)", ge=1, le=5)
    trust_building_attempts: List[str] = Field(description="신뢰 구축 시도")
    manipulation_tactics: List[str] = Field(description="조작 전술")

class DetectionChain:
    """LangChain 기반 사기 탐지 체인"""
    
    def __init__(self):
        self.scam_parser = PydanticOutputParser(pydantic_object=ScamAnalysisResult)
        self.context_parser = PydanticOutputParser(pydantic_object=CallContextAnalysis)
        self.embeddings = OpenAIEmbeddings()
        
        # 사기 패턴 벡터 데이터베이스 초기화
        self.pattern_vectorstore = self._initialize_pattern_db()
        
        # 체인 구성
        self.detection_chain = self._build_detection_chain()
        self.context_chain = self._build_context_chain()
        self.verification_chain = self._build_verification_chain()
        
        logger.info("LangChain 탐지 체인 초기화 완료")
    
    def _initialize_pattern_db(self) -> FAISS:
        """사기 패턴 벡터 데이터베이스 초기화"""
        
        # 문서에서 확인된 실제 사기 패턴들을 문서화
        scam_patterns = [
            Document(
                page_content="금융감독원을 사칭하여 계좌 점검이 필요하다며 개인정보를 요구하는 수법",
                metadata={"type": "기관사칭", "risk": "high", "keywords": ["금융감독원", "계좌점검", "개인정보"]}
            ),
            Document(
                page_content="검찰청 직원을 사칭하여 수사 관련이라며 계좌 동결을 언급하는 수법",
                metadata={"type": "기관사칭", "risk": "critical", "keywords": ["검찰청", "수사", "계좌동결"]}
            ),
            Document(
                page_content="저금리 대출을 미끼로 앱 설치를 유도하여 개인정보를 탈취하는 수법",
                metadata={"type": "대출사기", "risk": "high", "keywords": ["저금리", "대출", "앱설치"]}
            ),
            Document(
                page_content="납치나 사고를 빙자하여 즉시 돈을 송금하라고 협박하는 수법",
                metadata={"type": "납치협박", "risk": "critical", "keywords": ["납치", "사고", "응급실", "송금"]}
            ),
            Document(
                page_content="정부지원금이나 환급금을 미끼로 개인정보를 요구하는 수법",
                metadata={"type": "미끼문자", "risk": "medium", "keywords": ["정부지원금", "환급", "당첨"]}
            ),
            Document(
                page_content="만나서 직접 현금을 전달하라고 요구하는 대면편취형 수법",
                metadata={"type": "대면편취", "risk": "high", "keywords": ["만나서", "직접", "현금", "카페"]}
            ),
            Document(
                page_content="가상자산 투자를 빙자하여 투자금을 편취하는 수법",
                metadata={"type": "가상자산", "risk": "high", "keywords": ["비트코인", "투자", "수익", "거래소"]}
            ),
            Document(
                page_content="대포통장 개설을 위해 신분증과 통장을 요구하는 수법",
                metadata={"type": "대포통장", "risk": "high", "keywords": ["통장", "신분증", "명의", "대여"]}
            )
        ]
        
        # 벡터 스토어 생성
        try:
            vectorstore = FAISS.from_documents(scam_patterns, self.embeddings)
            logger.info(f"사기 패턴 벡터 DB 초기화 완료: {len(scam_patterns)}개 패턴")
            return vectorstore
        except Exception as e:
            logger.error(f"벡터 DB 초기화 실패: {e}")
            return None
    
    def _build_detection_chain(self):
        """사기 탐지 체인 구성"""
        
        # 시스템 프롬프트
        system_prompt = """
당신은 보이스피싱 탐지 전문가입니다. 
실제 문서에서 확인된 바와 같이, 대면편취형 사기가 7.5%에서 64.4%로 급증했습니다.
다음 8가지 주요 사기 유형을 기준으로 분석하세요:

1. **대포통장**: 통장, 카드 명의 대여 관련
2. **대포폰**: 휴대폰, 유심 개통 관련  
3. **악성앱**: 앱 설치, 권한 허용 관련
4. **미끼문자**: 정부지원금, 환급, 당첨 관련
5. **기관사칭**: 금융감독원, 검찰청, 경찰서 사칭
6. **납치협박**: 납치, 사고, 응급실 언급
7. **대출사기**: 저금리, 무담보 대출 제안
8. **가상자산**: 비트코인, 투자 수익 제안

## 특별 주의사항
- 대면편취형 키워드: "만나서", "직접", "현장", "카페", "현금"
- 긴급성을 조성하는 언어: "지금 즉시", "빨리", "늦으면"
- 신뢰성을 강조하는 언어: "공식", "정부", "법적"

{format_instructions}
"""
        
        human_prompt = """
분석할 텍스트: "{text}"

통화 컨텍스트:
- 통화 시간: {call_duration}초
- 발신자 정보: {caller_info}
- 이전 대화: {previous_context}
- 유사 패턴: {similar_patterns}

위 정보를 종합하여 보이스피싱 위험도를 분석해주세요.
"""
        
        detection_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
        ]).partial(format_instructions=self.scam_parser.get_format_instructions())
        
        # 체인 구성
        def get_similar_patterns(inputs):
            """유사 패턴 검색"""
            if self.pattern_vectorstore is None:
                return "패턴 DB 없음"
            
            try:
                similar_docs = self.pattern_vectorstore.similarity_search(
                    inputs["text"], k=3
                )
                return "\n".join([doc.page_content for doc in similar_docs])
            except Exception as e:
                logger.error(f"유사 패턴 검색 실패: {e}")
                return "검색 실패"
        
        chain = (
            RunnablePassthrough.assign(
                similar_patterns=RunnableLambda(get_similar_patterns)
            )
            | detection_prompt
            | llm_manager.models["gpt-4"]
            | self.scam_parser
        )
        
        return chain
    
    def _build_context_chain(self):
        """대화 맥락 분석 체인"""
        
        context_prompt = ChatPromptTemplate.from_messages([
            ("system", """
당신은 심리 조작 패턴 분석 전문가입니다.
보이스피싱범들이 사용하는 심리적 조작 기법을 분석하세요:

1. **신뢰 구축**: 공식 기관 사칭, 전문 용어 사용
2. **긴급성 조성**: "지금 즉시", "늦으면 큰일", 시간 압박
3. **공포 조성**: "계좌 동결", "체포영장", "수사 대상"
4. **권위 암시**: "법적 절차", "의무 사항", "정부 지시"
5. **친밀감 형성**: "도와드리겠습니다", "걱정 마세요"

{format_instructions}
"""),
            ("human", """
대화 내용: "{conversation}"
통화 시간: {duration}초
이전 대화들: {history}

위 대화에서 심리적 조작 패턴을 분석해주세요.
""")
        ]).partial(format_instructions=self.context_parser.get_format_instructions())
        
        return (
            context_prompt
            | llm_manager.models["gemini-1.5-flash"]
            | self.context_parser
        )
    
    def _build_verification_chain(self):
        """검증 체인 - 2차 검증을 위한 다른 모델 사용"""
        
        verification_prompt = ChatPromptTemplate.from_messages([
            ("system", """
당신은 1차 분석 결과를 검증하는 전문가입니다.
다음 분석 결과가 적절한지 검증하고 수정 사항을 제안하세요:

검증 기준:
1. 위험도 점수가 증거와 일치하는가?
2. 사기 유형 분류가 정확한가?
3. 즉시 대응 필요성 판단이 적절한가?
4. 놓친 중요한 지표는 없는가?

응답 형식: JSON
{
    "verified_risk_score": 0.0-1.0,
    "verified_scam_type": "사기유형",
    "verification_confidence": 0.0-1.0,
    "modifications": ["수정사항들"],
    "additional_indicators": ["추가지표들"]
}
"""),
            ("human", """
원본 텍스트: "{original_text}"

1차 분석 결과:
- 사기 유형: {scam_type}
- 위험도: {risk_score}
- 신뢰도: {confidence}
- 주요 지표: {indicators}
- 판단 근거: {reasoning}

이 분석 결과를 검증해주세요.
""")
        ])
        
        return (
            verification_prompt
            | llm_manager.models["gpt-3.5-turbo"]
            | JsonOutputParser()
        )
    
    async def analyze_scam_comprehensive(self, 
                                       text: str,
                                       context: Dict[str, Any] = None) -> Dict[str, Any]:
        """포괄적 사기 분석"""
        
        context = context or {}
        
        try:
            # 1. 기본 탐지 분석
            detection_input = {
                "text": text,
                "call_duration": context.get("call_duration", 0),
                "caller_info": context.get("caller_info", "알 수 없음"),
                "previous_context": str(context.get("previous_transcripts", [])),
            }
            
            detection_result = await self.detection_chain.ainvoke(detection_input)
            
            # 2. 대화 맥락 분석 (병렬 처리)
            context_input = {
                "conversation": text,
                "duration": context.get("call_duration", 0),
                "history": context.get("previous_transcripts", [])
            }
            
            context_task = asyncio.create_task(
                self.context_chain.ainvoke(context_input)
            )
            
            # 3. 고위험으로 판정된 경우 2차 검증 실행
            verification_result = None
            if detection_result.risk_score >= detection_thresholds.high_risk:
                verification_input = {
                    "original_text": text,
                    "scam_type": detection_result.scam_type,
                    "risk_score": detection_result.risk_score,
                    "confidence": detection_result.confidence,
                    "indicators": detection_result.key_indicators,
                    "reasoning": detection_result.reasoning
                }
                
                verification_task = asyncio.create_task(
                    self.verification_chain.ainvoke(verification_input)
                )
                verification_result = await verification_task
            
            # 4. 맥락 분석 결과 대기
            context_result = await context_task
            
            # 5. 결과 통합
            final_result = self._integrate_results(
                detection_result, 
                context_result, 
                verification_result
            )
            
            return final_result
            
        except Exception as e:
            logger.error(f"포괄적 사기 분석 실패: {e}")
            raise
    
    def _integrate_results(self, 
                          detection: ScamAnalysisResult,
                          context: CallContextAnalysis,
                          verification: Optional[Dict] = None) -> Dict[str, Any]:
        """분석 결과들을 통합"""
        
        # 기본 결과
        result = {
            "scam_type": detection.scam_type,
            "risk_score": detection.risk_score,
            "confidence": detection.confidence,
            "immediate_action": detection.immediate_action,
            "key_indicators": detection.key_indicators,
            "reasoning": detection.reasoning,
            "suggested_responses": detection.suggested_responses,
            
            # 맥락 분석 결과 추가
            "conversation_analysis": {
                "flow": context.conversation_flow,
                "emotional_state": context.emotional_state,
                "urgency_level": context.urgency_level,
                "trust_building_attempts": context.trust_building_attempts,
                "manipulation_tactics": context.manipulation_tactics
            }
        }
        
        # 검증 결과가 있으면 반영
        if verification:
            # 검증된 위험도가 더 신뢰할 만하면 업데이트
            if verification.get("verification_confidence", 0) > 0.8:
                result["risk_score"] = verification.get("verified_risk_score", result["risk_score"])
                result["scam_type"] = verification.get("verified_scam_type", result["scam_type"])
                
                # 추가 지표들 병합
                additional_indicators = verification.get("additional_indicators", [])
                result["key_indicators"].extend(additional_indicators)
                result["key_indicators"] = list(set(result["key_indicators"]))  # 중복 제거
                
                # 수정 사항 기록
                result["verification"] = {
                    "performed": True,
                    "confidence": verification.get("verification_confidence"),
                    "modifications": verification.get("modifications", [])
                }
        
        # 최종 위험도 레벨 결정
        if result["risk_score"] >= detection_thresholds.critical_risk:
            result["risk_level"] = "critical"
        elif result["risk_score"] >= detection_thresholds.high_risk:
            result["risk_level"] = "high"
        elif result["risk_score"] >= detection_thresholds.medium_risk:
            result["risk_level"] = "medium"
        else:
            result["risk_level"] = "low"
        
        return result
    
    async def quick_risk_assessment(self, text: str) -> float:
        """빠른 위험도 평가 (키워드 + 간단한 LLM)"""
        
        # 1. 키워드 기반 빠른 스크리닝
        keyword_risk = self._calculate_keyword_risk(text)
        
        # 2. 높은 위험도가 감지되면 LLM으로 확인
        if keyword_risk >= 0.6:
            simple_prompt = ChatPromptTemplate.from_messages([
                ("system", "당신은 보이스피싱 탐지 전문가입니다. 주어진 텍스트의 사기 위험도를 0.0-1.0으로 평가하세요. 숫자만 응답하세요."),
                ("human", "텍스트: {text}")
            ])
            
            try:
                chain = simple_prompt | llm_manager.models["gpt-3.5-turbo"]
                result = await chain.ainvoke({"text": text})
                
                # 응답에서 숫자 추출
                import re
                numbers = re.findall(r'0\.\d+|1\.0|1|0', result.content)
                if numbers:
                    llm_risk = float(numbers[0])
                    return max(keyword_risk, llm_risk)
            except Exception as e:
                logger.error(f"빠른 LLM 평가 실패: {e}")
        
        return keyword_risk
    
    def _calculate_keyword_risk(self, text: str) -> float:
        """키워드 기반 위험도 계산"""
        text_lower = text.lower()
        total_risk = 0.0
        
        # 각 사기 유형별 키워드 점수 계산
        for category, config in scam_config.SCAM_CATEGORIES.items():
            category_score = 0.0
            found_keywords = []
            
            for keyword in config["keywords"]:
                if keyword in text_lower:
                    category_score += 0.2
                    found_keywords.append(keyword)
            
            # 여러 키워드가 함께 나타나면 가중치 증가
            if len(found_keywords) >= 2:
                category_score *= 1.3
            
            # 카테고리별 가중치 적용
            weighted_score = min(category_score, 1.0) * config["weight"]
            total_risk = max(total_risk, weighted_score)
        
        # 대면편취형 특별 처리
        face_to_face_indicators = 0
        for indicator in scam_config.FACE_TO_FACE_INDICATORS:
            if indicator in text_lower:
                face_to_face_indicators += 1
        
        if face_to_face_indicators >= 2:
            total_risk = max(total_risk, 0.8)  # 대면편취형 고위험
        
        return min(total_risk, 1.0)
    
    async def analyze_conversation_trend(self, 
                                       conversation_history: List[str]) -> Dict[str, Any]:
        """대화 전체 흐름 분석"""
        
        if len(conversation_history) < 2:
            return {"trend": "insufficient_data"}
        
        trend_prompt = ChatPromptTemplate.from_messages([
            ("system", """
당신은 보이스피싱 대화 패턴 분석 전문가입니다.
대화의 시간적 흐름을 분석하여 사기 가능성의 변화를 평가하세요.

분석 요소:
1. 위험도 변화 추이 (증가/감소/일정)
2. 조작 전술의 진화
3. 피해자 반응 변화
4. 결정적 순간 (turning point)

응답 형식: JSON
{
    "risk_trend": "increasing/decreasing/stable",
    "peak_risk_moment": "대화 중 가장 위험한 순간",
    "manipulation_evolution": ["조작 전술 변화"],
    "victim_resistance": "저항도 (low/medium/high)",
    "predicted_outcome": "예상 결과",
    "intervention_points": ["개입하기 좋은 시점들"]
}
"""),
            ("human", """
대화 기록 (시간 순):
{conversation_history}

이 대화의 흐름을 분석해주세요.
""")
        ])
        
        try:
            conversation_text = "\n".join([
                f"[{i+1}] {conv}" for i, conv in enumerate(conversation_history)
            ])
            
            chain = trend_prompt | llm_manager.models["gemini-1.5-flash"] | JsonOutputParser()
            result = await chain.ainvoke({"conversation_history": conversation_text})
            
            return result
            
        except Exception as e:
            logger.error(f"대화 흐름 분석 실패: {e}")
            return {"trend": "analysis_failed", "error": str(e)}
    
    async def generate_intervention_script(self, 
                                         scam_type: str, 
                                         risk_level: str) -> List[str]:
        """사기 유형별 개입 스크립트 생성"""
        
        script_prompt = ChatPromptTemplate.from_messages([
            ("system", """
당신은 보이스피싱 개입 전문가입니다.
사기 유형과 위험도에 따른 효과적인 개입 멘트를 생성하세요.

개입 원칙:
1. 즉시 의심하게 만들기
2. 구체적인 행동 지침 제공
3. 감정적 안정화
4. 검증 방법 제시

사기 유형별 특화 대응:
- 기관사칭: "진짜 {기관명}에서는 이런 식으로 연락하지 않습니다"
- 납치협박: "침착하세요. 먼저 직접 확인해보세요"
- 대출사기: "정식 금융기관은 먼저 돈을 요구하지 않습니다"
- 대면편취: "만나기 전에 반드시 확인하세요"

응답 형식: 문자열 리스트
"""),
            ("human", """
사기 유형: {scam_type}
위험도: {risk_level}

이 상황에 적합한 개입 멘트 3-5개를 생성해주세요.
""")
        ])
        
        try:
            chain = script_prompt | llm_manager.models["gpt-4"]
            result = await chain.ainvoke({
                "scam_type": scam_type,
                "risk_level": risk_level
            })
            
            # 응답에서 리스트 추출
            content = result.content
            scripts = []
            
            # 번호나 불릿 포인트로 구분된 문장들 추출
            import re
            patterns = [
                r'\d+\.\s*(.+?)(?=\d+\.|$)',  # 1. 2. 3. 형식
                r'[-•]\s*(.+?)(?=[-•]|$)',    # - • 형식
                r'"([^"]+)"',                  # 따옴표 형식
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, content, re.DOTALL)
                if matches:
                    scripts = [match.strip() for match in matches if match.strip()]
                    break
            
            # 패턴 매칭 실패시 기본 스크립트
            if not scripts:
                scripts = self._get_default_scripts(scam_type, risk_level)
            
            return scripts[:5]  # 최대 5개
            
        except Exception as e:
            logger.error(f"개입 스크립트 생성 실패: {e}")
            return self._get_default_scripts(scam_type, risk_level)
    
    def _get_default_scripts(self, scam_type: str, risk_level: str) -> List[str]:
        """기본 개입 스크립트"""
        
        base_scripts = [
            "잠깐, 이 전화가 의심스럽습니다. 통화를 끊고 다시 생각해보세요.",
            "진짜 기관에서는 이런 식으로 연락하지 않습니다. 직접 확인해보세요.",
            "급하게 결정하지 마세요. 가족이나 지인에게 먼저 상의해보세요."
        ]
        
        # 사기 유형별 특화 스크립트
        type_specific = {
            "기관사칭": [
                "금융감독원이나 검찰청에서 전화로 개인정보를 요구하지 않습니다.",
                "해당 기관에 직접 전화해서 확인해보세요."
            ],
            "납치협박": [
                "침착하세요. 가족에게 직접 연락해서 확인해보세요.",
                "진짜 응급상황이라면 112에 신고하세요."
            ],
            "대출사기": [
                "정식 금융기관은 먼저 수수료를 요구하지 않습니다.",
                "금융감독원 홈페이지에서 등록된 업체인지 확인하세요."
            ],
            "대면편취": [
                "만나기 전에 반드시 정식 기관에 확인하세요.",
                "현금을 들고 만나는 것은 매우 위험합니다."
            ]
        }
        
        scripts = base_scripts.copy()
        if scam_type in type_specific:
            scripts.extend(type_specific[scam_type])
        
        return scripts
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """체인 성능 통계"""
        return {
            "detection_chain": "active",
            "context_chain": "active", 
            "verification_chain": "active",
            "pattern_db_size": self.pattern_vectorstore.index.ntotal if self.pattern_vectorstore else 0,
            "supported_scam_types": list(scam_config.SCAM_CATEGORIES.keys())
        }

# 전역 탐지 체인 인스턴스
detection_chain = DetectionChain()