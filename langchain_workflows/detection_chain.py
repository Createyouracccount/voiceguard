"""
VoiceGuard AI - 통합 탐지 에이전트
빠른 패턴 매칭 + 심층 LangChain 분석 결합
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import re
import json

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document
from pydantic import BaseModel, Field

from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.callbacks.manager import get_openai_callback

from core.llm_manager import llm_manager
from config.settings import (
    scam_config, detection_thresholds, RiskLevel,
    voice_config, ai_config
)
from monitoring.langsmith_tracker import tracker

logger = logging.getLogger(__name__)

class QuickDetectionResult(BaseModel):
    """빠른 탐지 결과"""
    is_suspicious: bool = Field(description="의심스러운 통화 여부")
    risk_score: float = Field(description="위험도 점수 (0.0-1.0)", ge=0.0, le=1.0)
    detected_patterns: List[str] = Field(description="탐지된 패턴들")
    scam_category: Optional[str] = Field(description="추정 사기 유형")
    confidence: float = Field(description="탐지 신뢰도", ge=0.0, le=1.0)
    requires_deep_analysis: bool = Field(description="심층 분석 필요 여부")

class ScamAnalysisResult(BaseModel):
    """심층 사기 분석 결과"""
    risk_score: float = Field(description="위험도 점수", ge=0.0, le=1.0)
    scam_type: str = Field(description="사기 유형")
    indicators: List[str] = Field(description="위험 지표들")
    psychological_tactics: List[str] = Field(description="심리적 조작 기법")
    urgency_level: str = Field(description="긴급도 레벨")
    recommendation: str = Field(description="대응 권장사항")

class CallContextAnalysis(BaseModel):
    """대화 맥락 분석 결과"""
    trust_building: float = Field(description="신뢰 구축 점수", ge=0.0, le=1.0)
    urgency_creation: float = Field(description="긴급성 조성 점수", ge=0.0, le=1.0)
    fear_induction: float = Field(description="공포 조성 점수", ge=0.0, le=1.0)
    authority_abuse: float = Field(description="권위 악용 점수", ge=0.0, le=1.0)
    manipulation_patterns: List[str] = Field(description="조작 패턴들")

class PatternMatcher:
    """키워드 기반 패턴 매칭 엔진"""
    
    def __init__(self):
        self.compiled_patterns = self._compile_patterns()
        self.keyword_weights = self._initialize_keyword_weights()
        
    def _compile_patterns(self) -> Dict[str, List[re.Pattern]]:
        """정규식 패턴 컴파일"""
        patterns = {}
        
        # 각 사기 유형별 패턴 컴파일
        for category, config in scam_config.SCAM_CATEGORIES.items():
            category_patterns = []
            
            for keyword in config["keywords"]:
                pattern = re.compile(
                    rf'\b{re.escape(keyword)}\b',
                    re.IGNORECASE
                )
                category_patterns.append(pattern)
            
            patterns[category] = category_patterns
        
        # 긴급성 패턴
        patterns["urgency"] = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in [
                r'지금\s*즉시',
                r'빨리',
                r'늦으면',
                r'시간\s*없',
                r'서둘러'
            ]
        ]
        
        # 대면 유도 패턴 (급증 추세 반영)
        patterns["face_to_face"] = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in [
                r'만나서',
                r'직접',
                r'현장',
                r'카페',
                r'현금.*전달'
            ]
        ]
        
        return patterns
    
    def _initialize_keyword_weights(self) -> Dict[str, float]:
        """키워드별 가중치 초기화"""
        return {
            # 고위험 키워드 (즉시 차단)
            "납치": 0.95, "유괴": 0.95, "죽는다": 0.90,
            "계좌동결": 0.88, "체포영장": 0.90, "구속": 0.85,
            
            # 기관사칭 키워드
            "금융감독원": 0.80, "검찰청": 0.85, "경찰서": 0.75,
            "수사": 0.70, "조사": 0.65,
            
            # 대면편취 키워드 (급증 추세 반영)
            "만나서": 0.75, "직접": 0.70, "현장": 0.68,
            "카페": 0.65, "현금": 0.72,
            
            # 일반 사기 키워드
            "대출": 0.50, "저금리": 0.60, "정부지원금": 0.65,
            "환급": 0.55, "당첨": 0.58,
            
            # 기술적 키워드
            "앱설치": 0.80, "권한": 0.75, "다운로드": 0.65,
            "업데이트": 0.60
        }
    
    def quick_scan(self, text: str) -> Dict[str, Any]:
        """빠른 키워드 스캔"""
        
        matched_patterns = []
        matched_keywords = []
        category_scores = {}
        
        # 각 카테고리별 매칭
        for category, patterns in self.compiled_patterns.items():
            matches = 0
            
            for pattern in patterns:
                if pattern.search(text):
                    matches += 1
                    matched_keywords.append(pattern.pattern)
            
            if matches > 0:
                # 카테고리별 점수 계산
                if category in scam_config.SCAM_CATEGORIES:
                    weight = scam_config.SCAM_CATEGORIES[category]["weight"]
                    score = min(matches * 0.2 * weight, 1.0)
                else:
                    score = min(matches * 0.15, 1.0)
                
                category_scores[category] = score
                matched_patterns.append(category)
        
        # 최종 위험도 점수
        risk_score = max(category_scores.values()) if category_scores else 0.0
        
        # 특별 보정: 여러 카테고리가 동시에 매칭되면 위험도 증가
        if len(matched_patterns) >= 2:
            risk_score = min(risk_score * 1.3, 1.0)
        
        return {
            "risk_score": risk_score,
            "patterns": matched_patterns,
            "matched_keywords": list(set(matched_keywords)),
            "category_scores": category_scores
        }

class DetectionChain:
    """LangChain 기반 심층 분석 체인 (Gemini 전용)"""
    
    def __init__(self):
        self.scam_parser = PydanticOutputParser(pydantic_object=ScamAnalysisResult)
        self.context_parser = PydanticOutputParser(pydantic_object=CallContextAnalysis)
        
        # Google Embeddings 사용
        try:
            from config.settings import settings
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=settings.GOOGLE_API_KEY
            )
            logger.info("✅ Google Embeddings 초기화 성공")
        except Exception as e:
            logger.warning(f"Google Embeddings 초기화 실패: {e}")
            self.embeddings = None
        
        # 사기 패턴 벡터 데이터베이스 초기화
        self.pattern_vectorstore = self._initialize_pattern_db()
        
        # 체인 구성
        self.detection_chain = self._build_detection_chain()
        self.context_chain = self._build_context_chain()
        self.verification_chain = self._build_verification_chain()
    
    def _initialize_pattern_db(self) -> Optional[FAISS]:
        """사기 패턴 벡터 데이터베이스 초기화"""
        
        if not self.embeddings:
            return None
        
        scam_patterns = [
            Document(
                page_content="금융감독원을 사칭하여 계좌 점검이 필요하다며 개인정보를 요구하는 수법",
                metadata={"type": "기관사칭", "risk": "high", "keywords": ["금융감독원", "계좌점검"]}
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
                page_content="만나서 직접 현금을 전달하라고 요구하는 대면편취형 수법",
                metadata={"type": "대면편취", "risk": "high", "keywords": ["만나서", "직접", "현금"]}
            )
        ]
        
        try:
            vectorstore = FAISS.from_documents(scam_patterns, self.embeddings)
            logger.info(f"사기 패턴 벡터 DB 초기화 완료: {len(scam_patterns)}개 패턴")
            return vectorstore
        except Exception as e:
            logger.error(f"벡터 DB 초기화 실패: {e}")
            return None
    
    def _build_detection_chain(self):
        """사기 탐지 체인 구성"""
        
        system_prompt = """
당신은 보이스피싱 탐지 전문가입니다. 
대면편취형 사기가 급증하고 있습니다 (7.5% → 64.4%).

주요 사기 유형 8가지:
1. 대포통장 2. 대포폰 3. 악성앱 4. 미끼문자
5. 기관사칭 6. 납치협박 7. 대출사기 8. 가상자산

특별 주의: "만나서", "직접", "현장", "카페", "현금" (대면편취)

{format_instructions}
"""
        
        human_prompt = """
분석 텍스트: "{text}"
컨텍스트: {context}
유사 패턴: {similar_patterns}

보이스피싱 위험도를 분석해주세요.
"""
        
        detection_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
        ]).partial(format_instructions=self.scam_parser.get_format_instructions())
        
        def get_similar_patterns(inputs):
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
        
        # Gemini 모델 사용
        try:
            gemini_model = llm_manager.get_model("gemini-1.5-pro")
        except:
            try:
                gemini_model = llm_manager.get_model("gemini-2.0-flash")
            except:
                # 폴백: 사용 가능한 첫 번째 모델
                available_models = llm_manager.get_available_models()
                gemini_model = llm_manager.get_model(available_models[0])
        
        chain = (
            RunnablePassthrough.assign(
                similar_patterns=RunnableLambda(get_similar_patterns)
            )
            | detection_prompt
            | gemini_model
            | self.scam_parser
        )
        
        return chain
    
    def _build_context_chain(self):
        """대화 맥락 분석 체인"""
        
        context_prompt = ChatPromptTemplate.from_messages([
            ("system", """
당신은 심리 조작 패턴 분석 전문가입니다.
보이스피싱범들의 심리적 조작 기법을 분석하세요:

1. 신뢰 구축: 공식 기관 사칭, 전문 용어
2. 긴급성 조성: "지금 즉시", 시간 압박
3. 공포 조성: "계좌 동결", "체포영장"
4. 권위 암시: "법적 절차", "의무 사항"
5. 친밀감 형성: "도와드리겠습니다"

{format_instructions}
"""),
            ("human", """
대화: "{conversation}"
시간: {duration}초
이력: {history}

심리적 조작 패턴을 분석해주세요.
""")
        ]).partial(format_instructions=self.context_parser.get_format_instructions())
        
        try:
            gemini_model = llm_manager.get_model("gemini-1.5-flash")
        except:
            try:
                gemini_model = llm_manager.get_model("gemini-2.0-flash")
            except:
                available_models = llm_manager.get_available_models()
                gemini_model = llm_manager.get_model(available_models[0])
        
        return (
            context_prompt
            | gemini_model
            | self.context_parser
        )
    
    def _build_verification_chain(self):
        """검증 체인"""
        
        verification_prompt = ChatPromptTemplate.from_messages([
            ("system", """
1차 분석 결과를 검증하는 전문가입니다.
다음을 확인하세요:

1. 위험도 점수가 증거와 일치하는가?
2. 사기 유형 분류가 정확한가?
3. 즉시 대응 필요성이 적절한가?

JSON 응답:
{
    "verified_risk_score": 0.0-1.0,
    "verified_scam_type": "사기유형",
    "verification_confidence": 0.0-1.0,
    "modifications": ["수정사항"],
    "additional_indicators": ["추가지표"]
}
"""),
            ("human", """
원본: "{original_text}"
1차 결과: 유형={scam_type}, 위험도={risk_score}, 지표={indicators}

검증해주세요.
""")
        ])
        
        try:
            gemini_model = llm_manager.get_model("gemini-1.5-flash")
        except:
            try:
                gemini_model = llm_manager.get_model("gemini-2.0-flash")
            except:
                available_models = llm_manager.get_available_models()
                gemini_model = llm_manager.get_model(available_models[0])
        
        return (
            verification_prompt
            | gemini_model
            | JsonOutputParser()
        )

class DetectionAgent:
    """통합 탐지 에이전트 (빠른 스캔 + 심층 분석)"""
    
    def __init__(self):
        self.name = "DetectionAgent"
        self.role = "통합 사기 탐지 및 심층 분석"
        
        # 컴포넌트 초기화
        self.pattern_matcher = PatternMatcher()
        self.detection_chain = DetectionChain()
        
        # 파서 초기화
        self.detection_parser = PydanticOutputParser(pydantic_object=QuickDetectionResult)
        
        # 통계
        self.stats = {
            "total_detections": 0,
            "high_risk_detections": 0,
            "avg_detection_time": 0.0,
            "true_positives": 0,
            "false_positives": 0
        }
        
        logger.info("통합 DetectionAgent 초기화 완료")
    
    @tracker.track_detection
    async def process_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """작업 처리 메인 메서드"""
        
        text = task_data.get("text", "")
        context = task_data.get("context", {})
        
        start_time = datetime.now()
        
        try:
            # 1. 빠른 키워드 스크리닝
            keyword_result = self.pattern_matcher.quick_scan(text)
            
            # 2. 위험도가 낮으면 빠른 종료
            if keyword_result["risk_score"] < 0.2:
                return self._create_low_risk_result(text, keyword_result)
            
            # 3. 심층 LangChain 분석 (높은 위험도만)
            if keyword_result["risk_score"] >= 0.5:
                deep_result = await self._perform_deep_analysis(text, context)
                langchain_result = deep_result
            else:
                langchain_result = None
            
            # 4. 결과 통합
            final_result = self._integrate_results(
                keyword_result, 
                langchain_result,
                context
            )
            
            # 5. 통계 업데이트
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_stats(processing_time, final_result)
            
            return final_result
            
        except Exception as e:
            logger.error(f"Detection Agent 작업 처리 실패: {e}")
            return self._create_error_result(str(e))
    
    async def _perform_deep_analysis(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """LangChain 기반 심층 분석"""
        
        try:
            # 병렬로 심층 분석과 컨텍스트 분석 실행
            detection_task = self.detection_chain.detection_chain.ainvoke({
                "text": text,
                "context": json.dumps(context, ensure_ascii=False)
            })
            
            context_task = self.detection_chain.context_chain.ainvoke({
                "conversation": text,
                "duration": context.get("call_duration", 0),
                "history": json.dumps(context.get("risk_history", []))
            })
            
            # 결과 수집
            detection_result, context_result = await asyncio.gather(
                detection_task, context_task, return_exceptions=True
            )
            
            # 검증 단계
            if isinstance(detection_result, ScamAnalysisResult):
                verification_result = await self.detection_chain.verification_chain.ainvoke({
                    "original_text": text,
                    "scam_type": detection_result.scam_type,
                    "risk_score": detection_result.risk_score,
                    "indicators": json.dumps(detection_result.indicators)
                })
            else:
                verification_result = None
            
            return {
                "detection": detection_result,
                "context": context_result,
                "verification": verification_result
            }
            
        except Exception as e:
            logger.error(f"심층 분석 실패: {e}")
            return None
    
    def _integrate_results(self,
                          keyword_result: Dict[str, Any],
                          langchain_result: Optional[Dict[str, Any]],
                          context: Dict[str, Any]) -> Dict[str, Any]:
        """결과 통합"""
        
        # 기본 위험도는 키워드 결과
        base_risk_score = keyword_result["risk_score"]
        
        # LangChain 결과가 있으면 조합
        if langchain_result and isinstance(langchain_result.get("detection"), ScamAnalysisResult):
            detection = langchain_result["detection"]
            verification = langchain_result.get("verification")
            
            # 검증된 점수가 있으면 사용
            if verification and "verified_risk_score" in verification:
                llm_score = verification["verified_risk_score"]
                scam_type = verification.get("verified_scam_type", detection.scam_type)
            else:
                llm_score = detection.risk_score
                scam_type = detection.scam_type
            
            # 가중 평균 (키워드 30%, LLM 70%)
            final_risk_score = base_risk_score * 0.3 + llm_score * 0.7
            
            # 증거 수집
            evidence = keyword_result.get("matched_keywords", [])
            evidence.extend(detection.indicators)
            
            psychological_tactics = detection.psychological_tactics
            recommendation = detection.recommendation
            
        else:
            # 키워드 결과만 사용
            final_risk_score = base_risk_score
            scam_type = self._determine_scam_type_from_keywords(keyword_result)
            evidence = keyword_result.get("matched_keywords", [])
            psychological_tactics = []
            recommendation = self._generate_basic_recommendation(final_risk_score)
        
        # 컨텍스트 기반 조정
        context_adjustment = self._analyze_basic_context(context)
        final_risk_score = min(final_risk_score + context_adjustment, 1.0)
        
        return {
            "agent": self.name,
            "timestamp": datetime.now().isoformat(),
            "risk_score": final_risk_score,
            "risk_level": self._determine_risk_level(final_risk_score),
            "is_suspicious": final_risk_score >= detection_thresholds.medium_risk,
            "scam_type": scam_type,
            "confidence": self._calculate_confidence(keyword_result, langchain_result),
            "evidence": list(set(evidence)),
            "psychological_tactics": psychological_tactics,
            "detected_patterns": {
                "keyword_patterns": keyword_result.get("patterns", []),
                "category_scores": keyword_result.get("category_scores", {})
            },
            "requires_deep_analysis": final_risk_score >= 0.7,
            "recommendation": recommendation,
            "immediate_alert": self._should_immediate_alert(final_risk_score, evidence)
        }
    
    def _determine_scam_type_from_keywords(self, keyword_result: Dict[str, Any]) -> str:
        """키워드 결과에서 사기 유형 추정"""
        category_scores = keyword_result.get("category_scores", {})
        if category_scores:
            return max(category_scores.keys(), key=lambda k: category_scores[k])
        return "unknown"
    
    def _analyze_basic_context(self, context: Dict[str, Any]) -> float:
        """기본 컨텍스트 분석"""
        adjustment = 0.0
        
        # 통화 시간
        call_duration = context.get("call_duration", 0)
        if call_duration > 300:  # 5분 이상
            adjustment += 0.1
        
        # 발신자 정보
        caller_info = context.get("caller_info", {})
        caller_number = caller_info.get("number", "")
        suspicious_patterns = ["050", "070", "+86", "+82-50"]
        if any(pattern in caller_number for pattern in suspicious_patterns):
            adjustment += 0.15
        
        return min(adjustment, 0.3)
    
    def _calculate_confidence(self, 
                             keyword_result: Dict[str, Any], 
                             langchain_result: Optional[Dict[str, Any]]) -> float:
        """신뢰도 계산"""
        base_confidence = 0.6
        
        # 키워드 매칭 수
        keyword_count = len(keyword_result.get("matched_keywords", []))
        keyword_boost = min(keyword_count * 0.1, 0.2)
        
        # LangChain 분석이 있으면 신뢰도 증가
        langchain_boost = 0.2 if langchain_result else 0.0
        
        return min(base_confidence + keyword_boost + langchain_boost, 1.0)
    
    def _determine_risk_level(self, risk_score: float) -> str:
        """위험도 레벨 결정"""
        if risk_score >= detection_thresholds.critical_risk:
            return RiskLevel.CRITICAL.value
        elif risk_score >= detection_thresholds.high_risk:
            return RiskLevel.HIGH.value
        elif risk_score >= detection_thresholds.medium_risk:
            return RiskLevel.MEDIUM.value
        else:
            return RiskLevel.LOW.value
    
    def _generate_basic_recommendation(self, risk_score: float) -> str:
        """기본 권장 사항 생성"""
        if risk_score >= detection_thresholds.critical_risk:
            return "즉시 통화 차단 및 신고 권장"
        elif risk_score >= detection_thresholds.high_risk:
            return "경고 메시지 표시 및 추가 검증 필요"
        elif risk_score >= detection_thresholds.medium_risk:
            return "주의 메시지 표시 및 모니터링 강화"
        else:
            return "정상 통화로 판단되나 지속 모니터링"
    
    def _should_immediate_alert(self, risk_score: float, evidence: List[str]) -> bool:
        """즉시 알림 필요성 판단"""
        # 높은 위험도
        if risk_score >= detection_thresholds.critical_risk:
            return True
        
        # 위험 키워드
        critical_keywords = ["납치", "유괴", "죽는다", "체포영장", "계좌동결"]
        if any(keyword in evidence for keyword in critical_keywords):
            return True
        
        return False
    
    def _create_low_risk_result(self, text: str, keyword_result: Dict[str, Any]) -> Dict[str, Any]:
        """저위험 결과 생성"""
        return {
            "agent": self.name,
            "timestamp": datetime.now().isoformat(),
            "risk_score": keyword_result["risk_score"],
            "risk_level": RiskLevel.LOW.value,
            "is_suspicious": False,
            "scam_type": None,
            "confidence": 0.9,
            "evidence": keyword_result.get("matched_keywords", []),
            "psychological_tactics": [],
            "detected_patterns": {
                "keyword_patterns": keyword_result.get("patterns", [])
            },
            "requires_deep_analysis": False,
            "recommendation": "정상 통화로 판단됨",
            "immediate_alert": False
        }
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """오류 결과 생성"""
        return {
            "agent": self.name,
            "timestamp": datetime.now().isoformat(),
            "error": error_message,
            "risk_score": 0.5,
            "risk_level": RiskLevel.MEDIUM.value,
            "is_suspicious": True,
            "requires_deep_analysis": True,
            "recommendation": "시스템 오류로 인한 수동 검토 필요",
            "immediate_alert": False
        }
    
    def _update_stats(self, processing_time: float, result: Dict[str, Any]):
        """통계 업데이트"""
        self.stats["total_detections"] += 1
        
        # 평균 처리 시간 업데이트
        self.stats["avg_detection_time"] = (
            self.stats["avg_detection_time"] * 0.9 + processing_time * 0.1
        )
        
        if result.get("risk_score", 0) >= detection_thresholds.high_risk:
            self.stats["high_risk_detections"] += 1
    
    async def quick_risk_assessment(self, text: str) -> Tuple[float, bool]:
        """초고속 위험도 평가 (응급용)"""
        
        # 극도로 빠른 키워드 체크만 수행
        critical_keywords = ["납치", "체포", "계좌동결", "죽는다", "응급실"]
        
        text_lower = text.lower()
        for keyword in critical_keywords:
            if keyword in text_lower:
                return 0.95, True  # 매우 높은 위험도, 즉시 알림
        
        # 기본 키워드 체크
        risky_keywords = ["금융감독원", "검찰청", "앱설치", "대출", "만나서"]
        risk_count = sum(1 for keyword in risky_keywords if keyword in text_lower)
        
        quick_score = min(risk_count * 0.2, 0.8)
        needs_alert = quick_score >= 0.6
        
        return quick_score, needs_alert
    
    def get_stats(self) -> Dict[str, Any]:
        """통계 반환"""
        return {
            **self.stats,
            "accuracy": (
                self.stats["true_positives"] / 
                max(1, self.stats["true_positives"] + self.stats["false_positives"])
            ),
            "high_risk_rate": (
                self.stats["high_risk_detections"] / 
                max(1, self.stats["total_detections"])
            )
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """성능 메트릭 반환"""
        return {
            "agent_name": self.name,
            "total_detections": self.stats["total_detections"],
            "high_risk_rate": (
                self.stats["high_risk_detections"] / 
                max(1, self.stats["total_detections"])
            ),
            "avg_processing_time": self.stats["avg_detection_time"],
            "loaded_patterns": len(self.pattern_matcher.compiled_patterns),
            "loaded_keywords": len(self.pattern_matcher.keyword_weights),
            "langchain_enabled": self.detection_chain is not None,
            "vector_db_enabled": self.detection_chain.pattern_vectorstore is not None
        }
    
    async def update_patterns(self, new_patterns: Dict[str, List[str]]):
        """새로운 패턴 동적 업데이트"""
        try:
            for category, patterns in new_patterns.items():
                if category not in self.pattern_matcher.compiled_patterns:
                    self.pattern_matcher.compiled_patterns[category] = []
                
                # 새 패턴들을 컴파일해서 추가
                compiled_new = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
                self.pattern_matcher.compiled_patterns[category].extend(compiled_new)
            
            logger.info(f"패턴 업데이트 완료: {len(new_patterns)}개 카테고리")
            
        except Exception as e:
            logger.error(f"패턴 업데이트 실패: {e}")
    
    def reset_stats(self):
        """통계 초기화"""
        self.stats = {
            "total_detections": 0,
            "high_risk_detections": 0,
            "avg_detection_time": 0.0,
            "true_positives": 0,
            "false_positives": 0
        }
        logger.info("탐지 에이전트 통계 초기화")


# 사용 예시 및 테스트 함수들
async def test_detection_agent():
    """DetectionAgent 테스트"""
    
    agent = DetectionAgent()
    
    # 테스트 케이스들
    test_cases = [
        {
            "text": "안녕하세요. 금융감독원 직원입니다. 고객님 계좌에 문제가 있어서 즉시 확인이 필요합니다.",
            "context": {"call_duration": 120, "caller_info": {"number": "050-1234-5678"}},
            "expected_high_risk": True
        },
        {
            "text": "저금리 대출 가능합니다. 지금 앱만 설치하시면 바로 승인됩니다.",
            "context": {"call_duration": 180},
            "expected_high_risk": True
        },
        {
            "text": "안녕하세요. 오늘 날씨가 좋네요. 어떻게 지내세요?",
            "context": {"call_duration": 30},
            "expected_high_risk": False
        },
        {
            "text": "아들이 사고났어요! 빨리 병원비가 필요해요. 지금 카페에서 만나요.",
            "context": {"call_duration": 45, "caller_info": {"number": "070-9999-8888"}},
            "expected_high_risk": True
        }
    ]
    
    print("=== DetectionAgent 테스트 시작 ===")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n테스트 {i}: {test_case['text'][:30]}...")
        
        # 빠른 평가
        quick_score, quick_alert = await agent.quick_risk_assessment(test_case["text"])
        print(f"빠른 평가: 위험도={quick_score:.2f}, 알림={quick_alert}")
        
        # 전체 분석
        result = await agent.process_task({
            "text": test_case["text"],
            "context": test_case["context"]
        })
        
        print(f"전체 분석: 위험도={result['risk_score']:.2f}, 유형={result.get('scam_type')}")
        print(f"권장사항: {result.get('recommendation')}")
        print(f"즉시알림: {result.get('immediate_alert')}")
        
        # 예상 결과와 비교
        is_high_risk = result['risk_score'] >= detection_thresholds.high_risk
        if is_high_risk == test_case["expected_high_risk"]:
            print("✅ 예상 결과와 일치")
        else:
            print("❌ 예상 결과와 불일치")
    
    # 성능 메트릭 출력
    print(f"\n=== 성능 메트릭 ===")
    metrics = agent.get_performance_metrics()
    for key, value in metrics.items():
        print(f"{key}: {value}")

# 실제 사용을 위한 팩토리 함수
def create_detection_agent() -> DetectionAgent:
    """DetectionAgent 인스턴스 생성"""
    return DetectionAgent()

# 전역 인스턴스 (필요시 사용)
detection_agent = None

def get_detection_agent() -> DetectionAgent:
    """전역 DetectionAgent 인스턴스 반환"""
    global detection_agent
    if detection_agent is None:
        detection_agent = DetectionAgent()
    return detection_agent


if __name__ == "__main__":
    # 테스트 실행
    import asyncio
    asyncio.run(test_detection_agent())