"""
VoiceGuard AI - Detection Agent
보이스피싱 초기 탐지 전문 에이전트
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import re
import json

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.callbacks import get_openai_callback
from pydantic import BaseModel, Field

from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # OpenAI 대신 Google 사용
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

class PatternDetectionResult(BaseModel):
    """패턴 탐지 결과"""
    pattern_type: str = Field(description="패턴 유형")
    match_score: float = Field(description="매칭 점수", ge=0.0, le=1.0)
    evidence: List[str] = Field(description="증거 텍스트")
    context: str = Field(description="패턴 컨텍스트")

class DetectionAgent:
    """탐지 전문 에이전트"""
    
    def __init__(self):
        self.name = "DetectionAgent"
        self.role = "심층 사기 수법 분석 및 심리적 패턴 해부"
        
        try:
            # gemini-1.5-pro 또는 사용 가능한 첫 번째 모델 사용
            available_models = llm_manager.get_available_models()
            if "gemini-1.5-pro" in available_models:
                self.analyzer_model = llm_manager.get_model("gemini-1.5-pro")
            elif "gemini-2.0-flash" in available_models:
                self.analyzer_model = llm_manager.get_model("gemini-2.0-flash")
            else:
                # 첫 번째 사용 가능한 모델 사용
                self.analyzer_model = llm_manager.get_model(available_models[0])
            
            logger.info(f"DetectionAgent 모델 설정: {available_models[0] if available_models else 'None'}")
            
        except Exception as e:
            logger.error(f"DetectionAgent 모델 설정 실패: {e}")
            # 폴백: LLM 매니저에서 직접 분석 메서드 사용
            self.analyzer_model = None
    
    def _build_quick_detection_prompt(self) -> ChatPromptTemplate:
        """빠른 탐지 프롬프트"""
        return ChatPromptTemplate.from_messages([
            ("system", """
당신은 보이스피싱 탐지 전문가입니다.
주어진 텍스트를 신속하게 분석하여 의심스러운 요소를 찾아내세요.

## 주요 탐지 포인트
1. **긴급성 조성**: "지금 즉시", "빨리", "늦으면 큰일"
2. **권위 악용**: 공공기관 사칭, 법적 협박
3. **금전 요구**: 송금, 계좌번호, 개인정보
4. **감정 조작**: 공포, 불안, 탐욕 유발
5. **대면 유도**: "만나서", "직접", "현장에서"

## 8가지 주요 사기 유형
1. 대포통장 (통장/카드 양도)
2. 대포폰 (휴대폰 개통)
3. 악성앱 (앱 설치 유도)
4. 미끼문자 (정부지원금, 환급)
5. 기관사칭 (금융감독원, 검찰청)
6. 납치협박 (가족 위협)
7. 대출사기 (저금리 대출)
8. 가상자산 (투자 유혹)

{format_instructions}
"""),
            ("human", """
분석할 텍스트: "{text}"

이 텍스트가 보이스피싱일 가능성을 평가해주세요.
빠른 판단이 중요합니다.
""")
        ]).partial(format_instructions=self.detection_parser.get_format_instructions())
    
    def _build_pattern_detection_prompt(self) -> ChatPromptTemplate:
        """패턴 탐지 프롬프트"""
        return ChatPromptTemplate.from_messages([
            ("system", """
당신은 보이스피싱 패턴 분석 전문가입니다.
특정 사기 패턴이 텍스트에 나타나는지 정밀하게 분석하세요.

## 분석 방법
1. 키워드 출현 빈도
2. 문맥상 의미
3. 화법 패턴
4. 설득 전략

{format_instructions}
"""),
            ("human", """
텍스트: "{text}"
찾을 패턴: "{pattern_type}"
패턴 설명: "{pattern_description}"

이 패턴이 텍스트에 나타나는지 분석해주세요.
""")
        ]).partial(format_instructions=self.pattern_parser.get_format_instructions())
    
    @tracker.track_detection
    async def process_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """작업 처리 메인 메서드"""
        
        text = task_data.get("text", "")
        context = task_data.get("context", {})
        
        start_time = datetime.now()
        
        try:
            # 1. 키워드 기반 빠른 스크리닝
            keyword_result = self.pattern_matcher.quick_scan(text)
            
            # 2. 위험도가 낮으면 빠른 종료
            if keyword_result["risk_score"] < 0.2:
                return self._create_low_risk_result(text, keyword_result)
            
            # 3. LLM 기반 정밀 탐지
            detection_result = await self._perform_llm_detection(text, context)
            
            # 4. 패턴 매칭 보강
            pattern_results = await self._detect_specific_patterns(text, detection_result)
            
            # 5. 결과 통합
            final_result = self._integrate_results(
                keyword_result, 
                detection_result, 
                pattern_results
            )
            
            # 6. 통계 업데이트
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_stats(processing_time, final_result)
            
            return final_result
            
        except Exception as e:
            logger.error(f"Detection Agent 작업 처리 실패: {e}")
            return self._create_error_result(str(e))
    
    async def _perform_llm_detection(self, 
                                   text: str, 
                                   context: Dict[str, Any]) -> QuickDetectionResult:
        """LLM 기반 탐지"""
        
        # 모델 선택 (빠른 탐지는 경제적 모델 사용)
        model = llm_manager.models["gpt-3.5-turbo"]
        
        chain = self.quick_detection_prompt | model | self.detection_parser
        
        try:
            with get_openai_callback() as cb:
                result = await chain.ainvoke({"text": text})
                
                # 비용 로깅
                logger.debug(f"탐지 비용: ${cb.total_cost:.4f}, 토큰: {cb.total_tokens}")
                
            return result
            
        except Exception as e:
            logger.error(f"LLM 탐지 실패: {e}")
            # 폴백 결과
            return QuickDetectionResult(
                is_suspicious=True,  # 안전을 위해 의심으로 처리
                risk_score=0.5,
                detected_patterns=["LLM 분석 실패"],
                scam_category=None,
                confidence=0.3,
                requires_deep_analysis=True
            )
    
    async def _detect_specific_patterns(self, 
                                      text: str,
                                      initial_result: QuickDetectionResult) -> List[PatternDetectionResult]:
        """특정 패턴 상세 탐지"""
        
        patterns_to_check = []
        
        # 초기 결과에 따라 확인할 패턴 결정
        if initial_result.scam_category:
            patterns_to_check.append(initial_result.scam_category)
        
        # 높은 위험도면 추가 패턴 확인
        if initial_result.risk_score >= 0.7:
            patterns_to_check.extend(["기관사칭", "납치협박", "대면편취"])
        
        # 병렬로 패턴 체크
        pattern_tasks = []
        for pattern in set(patterns_to_check):  # 중복 제거
            if pattern in scam_config.SCAM_CATEGORIES:
                task = self._check_single_pattern(text, pattern)
                pattern_tasks.append(task)
        
        if pattern_tasks:
            results = await asyncio.gather(*pattern_tasks, return_exceptions=True)
            return [r for r in results if isinstance(r, PatternDetectionResult)]
        
        return []
    
    async def _check_single_pattern(self, 
                                  text: str, 
                                  pattern_type: str) -> PatternDetectionResult:
        """단일 패턴 체크"""
        
        pattern_info = scam_config.SCAM_CATEGORIES.get(pattern_type, {})
        
        chain = self.pattern_detection_prompt | llm_manager.models["gpt-3.5-turbo"] | self.pattern_parser
        
        try:
            result = await chain.ainvoke({
                "text": text,
                "pattern_type": pattern_type,
                "pattern_description": f"키워드: {', '.join(pattern_info.get('keywords', []))}"
            })
            
            return result
            
        except Exception as e:
            logger.error(f"패턴 체크 실패 ({pattern_type}): {e}")
            return None
    
    def _integrate_results(self,
                          keyword_result: Dict[str, Any],
                          detection_result: QuickDetectionResult,
                          pattern_results: List[PatternDetectionResult]) -> Dict[str, Any]:
        """결과 통합"""
        
        # 위험도 점수 가중 평균
        risk_scores = [
            (keyword_result["risk_score"], 0.2),  # 키워드 20%
            (detection_result.risk_score, 0.5),    # LLM 50%
        ]
        
        # 패턴 결과 추가 (30%)
        if pattern_results:
            pattern_score = max(p.match_score for p in pattern_results)
            risk_scores.append((pattern_score, 0.3))
        else:
            # 패턴 결과 없으면 다른 가중치 조정
            risk_scores[0] = (risk_scores[0][0], 0.3)
            risk_scores[1] = (risk_scores[1][0], 0.7)
        
        final_risk_score = sum(score * weight for score, weight in risk_scores)
        
        # 증거 수집
        all_evidence = []
        all_evidence.extend(keyword_result.get("matched_keywords", []))
        all_evidence.extend(detection_result.detected_patterns)
        
        for pattern in pattern_results:
            all_evidence.extend(pattern.evidence)
        
        # 사기 유형 결정
        scam_type = detection_result.scam_category
        if not scam_type and pattern_results:
            # 가장 높은 점수의 패턴 유형 선택
            best_pattern = max(pattern_results, key=lambda x: x.match_score)
            scam_type = best_pattern.pattern_type
        
        # 신뢰도 계산
        confidence_factors = [
            detection_result.confidence,
            1.0 if len(all_evidence) >= 3 else 0.7,
            0.9 if pattern_results else 0.6
        ]
        final_confidence = sum(confidence_factors) / len(confidence_factors)
        
        return {
            "agent": self.name,
            "timestamp": datetime.now().isoformat(),
            "risk_score": min(final_risk_score, 1.0),
            "risk_level": self._determine_risk_level(final_risk_score),
            "is_suspicious": final_risk_score >= detection_thresholds.medium_risk,
            "scam_type": scam_type,
            "confidence": final_confidence,
            "evidence": list(set(all_evidence)),  # 중복 제거
            "detected_patterns": {
                "keyword_patterns": keyword_result.get("patterns", []),
                "llm_patterns": detection_result.detected_patterns,
                "specific_patterns": [
                    {
                        "type": p.pattern_type,
                        "score": p.match_score,
                        "evidence": p.evidence
                    }
                    for p in pattern_results
                ]
            },
            "requires_deep_analysis": detection_result.requires_deep_analysis or final_risk_score >= 0.7,
            "recommendation": self._generate_recommendation(final_risk_score, scam_type)
        }
    
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
    
    def _generate_recommendation(self, risk_score: float, scam_type: Optional[str]) -> str:
        """권장 사항 생성"""
        
        if risk_score >= detection_thresholds.critical_risk:
            return "즉시 통화 차단 및 신고 권장"
        elif risk_score >= detection_thresholds.high_risk:
            return "경고 메시지 표시 및 추가 검증 필요"
        elif risk_score >= detection_thresholds.medium_risk:
            return "주의 메시지 표시 및 모니터링 강화"
        else:
            return "정상 통화로 판단되나 지속 모니터링"
    
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
            "detected_patterns": {
                "keyword_patterns": keyword_result.get("patterns", [])
            },
            "requires_deep_analysis": False,
            "recommendation": "정상 통화로 판단됨"
        }
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """오류 결과 생성"""
        return {
            "agent": self.name,
            "timestamp": datetime.now().isoformat(),
            "error": error_message,
            "risk_score": 0.5,  # 중간 위험도로 설정 (안전을 위해)
            "risk_level": RiskLevel.MEDIUM.value,
            "is_suspicious": True,
            "requires_deep_analysis": True,
            "recommendation": "시스템 오류로 인한 수동 검토 필요"
        }
    
    def _update_stats(self, processing_time: float, result: Dict[str, Any]):
        """통계 업데이트"""
        self.stats["total_detections"] += 1
        
        # 평균 처리 시간 업데이트 (이동 평균)
        self.stats["avg_detection_time"] = (
            self.stats["avg_detection_time"] * 0.9 + processing_time * 0.1
        )
        
    def get_stats(self) -> Dict[str, Any]:
        """통계 반환"""
        return {
            **self.stats,
            "accuracy": (
                self.stats["true_positives"] / 
                max(1, self.stats["true_positives"] + self.stats["false_positives"])
            )
        }


class PatternMatcher:
    """키워드 기반 패턴 매칭 엔진"""
    
    def __init__(self):
        self.compiled_patterns = self._compile_patterns()
        
    def _compile_patterns(self) -> Dict[str, List[re.Pattern]]:
        """정규식 패턴 컴파일"""
        patterns = {}
        
        # 각 사기 유형별 패턴 컴파일
        for category, config in scam_config.SCAM_CATEGORIES.items():
            category_patterns = []
            
            for keyword in config["keywords"]:
                # 단어 경계를 포함한 패턴
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
        
        # 대면 유도 패턴
        patterns["face_to_face"] = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in scam_config.FACE_TO_FACE_INDICATORS
        ]
        
        return patterns
    
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