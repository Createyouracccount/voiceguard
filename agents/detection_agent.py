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
        self.detection_parser = PydanticOutputParser(pydantic_object=QuickDetectionResult)
        self.pattern_parser = PydanticOutputParser(pydantic_object=PatternDetectionResult)
        
        # 탐지 프롬프트
        self.quick_detection_prompt = self._build_quick_detection_prompt()
        self.pattern_detection_prompt = self._build_pattern_detection_prompt()
        
        # 패턴 매칭 엔진
        self.pattern_matcher = PatternMatcher()
        
        # 통계
        self.stats = {
            "total_detections": 0,
            "true_positives": 0,
            "false_positives": 0,
            "avg_detection_time": 0.0
        }
        
        logger.info("Detection Agent 초기화 완료")
    
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

# 전역 인스턴스는 __init__.py에서 생성됨



# """
# VoiceGuard AI - 탐지 전문 에이전트
# 실시간 패턴 매칭과 빠른 위험도 판정에 특화
# """

# import asyncio
# import time
# import re
# from typing import Dict, List, Optional, Tuple, Any
# from dataclasses import dataclass
# from datetime import datetime
# import logging

# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import JsonOutputParser
# from langchain_core.tools import Tool
# from langchain.agents import AgentExecutor, create_react_agent
# from langchain_openai import ChatOpenAI

# from config.settings import scam_config, detection_thresholds, RiskLevel
# from core.llm_manager import llm_manager

# logger = logging.getLogger(__name__)

# @dataclass
# class DetectionResult:
#     """탐지 결과"""
#     risk_score: float
#     detected_patterns: List[str]
#     scam_category: str
#     confidence: float
#     processing_time: float
#     immediate_alert: bool
#     detected_keywords: List[str]

# class DetectionAgent:
#     """실시간 사기 탐지 전문 에이전트"""
    
#     def __init__(self):
#         self.name = "DetectionAgent"
#         self.role = "실시간 사기 패턴 탐지 및 즉시 위험도 판정"
        
#         # 빠른 탐지를 위한 전처리된 패턴들
#         self.compiled_patterns = self._compile_detection_patterns()
#         self.keyword_weights = self._initialize_keyword_weights()
        
#         # 성능 통계
#         self.detection_stats = {
#             "total_detections": 0,
#             "high_risk_detections": 0,
#             "avg_processing_time": 0.0,
#             "pattern_hit_rate": {}
#         }
        
#         # 실시간 처리를 위한 경량 모델
#         self.fast_model = ChatOpenAI(
#             model="gpt-3.5-turbo",
#             temperature=0.1,
#             max_tokens=512  # 빠른 응답을 위해 토큰 제한
#         )
        
#         logger.info("탐지 에이전트 초기화 완료")
    
#     def _compile_detection_patterns(self) -> Dict[str, List]:
#         """정규식 패턴 미리 컴파일"""
#         patterns = {}
        
#         # 사기 유형별 정규식 패턴 정의
#         pattern_definitions = {
#             "대포통장": [
#                 r"(통장|계좌).*(대여|빌려|명의)",
#                 r"(신분증|주민등록증).*(복사|사진|전송)",
#                 r"(카드|체크카드).*(만들어|개설|발급)"
#             ],
#             "기관사칭": [
#                 r"(금융감독원|금감원).*(직원|담당자|조사관)",
#                 r"(검찰청|검찰|검사).*(수사|조사|영장)",
#                 r"(경찰서|경찰|형사).*(신고|체포|구속)",
#                 r"(국세청|세무서).*(체납|압류|조사)"
#             ],
#             "납치협박": [
#                 r"(납치|유괴).*(아들|딸|자녀|가족)",
#                 r"(사고|응급실|병원).*(위험|다쳤|입원)",
#                 r"(죽는다|죽겠다|위험하다).*(빨리|즉시|지금)"
#             ],
#             "악성앱": [
#                 r"(앱|어플|프로그램).*(설치|다운로드|받으)",
#                 r"(권한|허용|승인).*(필요|요청|설정)",
#                 r"(보안|업데이트|인증).*(앱|프로그램)"
#             ],
#             "대출사기": [
#                 r"(저금리|무담보|신용).*(대출|융자)",
#                 r"(승인|한도|가능).*(즉시|바로|지금)",
#                 r"(정부지원|특별|우대).*(대출|융자)"
#             ],
#             "대면편취": [
#                 r"(만나서|직접|현장).*(전달|받으|가져)",
#                 r"(카페|역|장소).*(오시면|와서|방문)",
#                 r"(현금|돈|봉투).*(준비|가져|전달)"
#             ],
#             "가상자산": [
#                 r"(비트코인|코인|가상화폐).*(투자|수익)",
#                 r"(거래소|지갑|계정).*(개설|등록)",
#                 r"(블록체인|채굴|디파이).*(투자|참여)"
#             ]
#         }
        
#         # 정규식 컴파일
#         for category, pattern_list in pattern_definitions.items():
#             patterns[category] = [re.compile(pattern, re.IGNORECASE) for pattern in pattern_list]
        
#         return patterns
    
#     def _initialize_keyword_weights(self) -> Dict[str, float]:
#         """키워드별 가중치 초기화"""
#         # 문서 분석 결과를 바탕으로 한 키워드 가중치
#         return {
#             # 고위험 키워드 (즉시 차단)
#             "납치": 0.95, "유괴": 0.95, "죽는다": 0.90,
#             "계좌동결": 0.88, "체포영장": 0.90, "구속": 0.85,
            
#             # 기관사칭 키워드
#             "금융감독원": 0.80, "검찰청": 0.85, "경찰서": 0.75,
#             "수사": 0.70, "조사": 0.65,
            
#             # 대면편취 키워드 (급증 추세 반영)
#             "만나서": 0.75, "직접": 0.70, "현장": 0.68,
#             "카페": 0.65, "현금": 0.72,
            
#             # 일반 사기 키워드
#             "대출": 0.50, "저금리": 0.60, "정부지원금": 0.65,
#             "환급": 0.55, "당첨": 0.58,
            
#             # 기술적 키워드
#             "앱설치": 0.80, "권한": 0.75, "다운로드": 0.65,
#             "업데이트": 0.60
#         }
    
#     async def detect_scam_patterns(self, 
#                                   text: str, 
#                                   context: Dict[str, Any] = None) -> DetectionResult:
#         """실시간 사기 패턴 탐지"""
        
#         start_time = time.time()
#         context = context or {}
        
#         try:
#             # 1. 빠른 키워드 스캔
#             keyword_score, detected_keywords = self._fast_keyword_scan(text)
            
#             # 2. 정규식 패턴 매칭
#             pattern_score, detected_patterns, category = self._pattern_matching(text)
            
#             # 3. 컨텍스트 기반 점수 조정
#             context_adjustment = self._analyze_context(text, context)
            
#             # 4. 최종 위험도 계산
#             final_score = self._calculate_final_score(
#                 keyword_score, pattern_score, context_adjustment
#             )
            
#             # 5. 즉시 알림 필요성 판단
#             immediate_alert = self._should_immediate_alert(
#                 final_score, detected_keywords, category
#             )
            
#             # 6. 신뢰도 계산
#             confidence = self._calculate_confidence(
#                 keyword_score, pattern_score, len(detected_keywords)
#             )
            
#             processing_time = time.time() - start_time
            
#             # 7. 통계 업데이트
#             self._update_stats(final_score, category, processing_time)
            
#             return DetectionResult(
#                 risk_score=final_score,
#                 detected_patterns=detected_patterns,
#                 scam_category=category,
#                 confidence=confidence,
#                 processing_time=processing_time,
#                 immediate_alert=immediate_alert,
#                 detected_keywords=detected_keywords
#             )
            
#         except Exception as e:
#             logger.error(f"탐지 에이전트 오류: {e}")
#             # 안전을 위해 중간 위험도로 반환
#             return DetectionResult(
#                 risk_score=0.5,
#                 detected_patterns=["오류발생"],
#                 scam_category="unknown",
#                 confidence=0.3,
#                 processing_time=time.time() - start_time,
#                 immediate_alert=False,
#                 detected_keywords=[]
#             )
    
#     def _fast_keyword_scan(self, text: str) -> Tuple[float, List[str]]:
#         """빠른 키워드 스캔"""
#         text_lower = text.lower()
#         detected_keywords = []
#         total_score = 0.0
        
#         for keyword, weight in self.keyword_weights.items():
#             if keyword in text_lower:
#                 detected_keywords.append(keyword)
#                 total_score += weight
        
#         # 정규화 (여러 키워드가 있어도 1.0을 넘지 않도록)
#         normalized_score = min(total_score / 2.0, 1.0)
        
#         return normalized_score, detected_keywords
    
#     def _pattern_matching(self, text: str) -> Tuple[float, List[str], str]:
#         """정규식 패턴 매칭"""
#         max_score = 0.0
#         detected_patterns = []
#         best_category = "unknown"
        
#         for category, patterns in self.compiled_patterns.items():
#             category_score = 0.0
#             category_patterns = []
            
#             for pattern in patterns:
#                 matches = pattern.findall(text)
#                 if matches:
#                     category_patterns.extend([f"{category}:{match}" for match in matches])
#                     category_score += 0.3  # 패턴당 0.3점
            
#             # 카테고리별 가중치 적용
#             category_weights = {
#                 "납치협박": 1.0,
#                 "기관사칭": 0.9,
#                 "대면편취": 0.8,  # 급증 추세 반영
#                 "악성앱": 0.85,
#                 "대포통장": 0.8,
#                 "대출사기": 0.7,
#                 "가상자산": 0.75
#             }
            
#             weighted_score = category_score * category_weights.get(category, 0.6)
            
#             if weighted_score > max_score:
#                 max_score = weighted_score
#                 detected_patterns = category_patterns
#                 best_category = category
        
#         return min(max_score, 1.0), detected_patterns, best_category
    
#     def _analyze_context(self, text: str, context: Dict[str, Any]) -> float:
#         """컨텍스트 기반 점수 조정"""
#         adjustment = 0.0
        
#         # 통화 시간 고려 (긴 통화일수록 위험)
#         call_duration = context.get("call_duration", 0)
#         if call_duration > 300:  # 5분 이상
#             adjustment += 0.1
#         elif call_duration > 600:  # 10분 이상
#             adjustment += 0.2
        
#         # 발신자 정보 고려
#         caller_info = context.get("caller_info", {})
#         caller_number = caller_info.get("number", "")
        
#         # 의심스러운 번호 패턴
#         suspicious_patterns = ["050", "070", "+86", "+82-50"]
#         if any(pattern in caller_number for pattern in suspicious_patterns):
#             adjustment += 0.15
        
#         # 시간대 고려 (새벽/늦은 밤)
#         current_hour = datetime.now().hour
#         if current_hour < 6 or current_hour > 22:
#             adjustment += 0.1
        
#         # 이전 위험도 트렌드
#         risk_history = context.get("risk_history", [])
#         if risk_history and len(risk_history) >= 2:
#             if all(risk > 0.5 for risk in risk_history[-2:]):
#                 adjustment += 0.15  # 지속적으로 위험한 대화
        
#         return min(adjustment, 0.5)  # 최대 0.5점 조정
    
#     def _calculate_final_score(self, 
#                               keyword_score: float, 
#                               pattern_score: float, 
#                               context_adjustment: float) -> float:
#         """최종 위험도 점수 계산"""
        
#         # 가중 평균 계산
#         base_score = (keyword_score * 0.4 + pattern_score * 0.6)
        
#         # 컨텍스트 조정 적용
#         final_score = base_score + context_adjustment
        
#         # 0.0 - 1.0 범위로 제한
#         return max(0.0, min(final_score, 1.0))
    
#     def _should_immediate_alert(self, 
#                                score: float, 
#                                keywords: List[str], 
#                                category: str) -> bool:
#         """즉시 알림 필요성 판단"""
        
#         # 1. 높은 위험도 점수
#         if score >= detection_thresholds.critical_risk:
#             return True
        
#         # 2. 위험 키워드 감지
#         critical_keywords = ["납치", "유괴", "죽는다", "체포영장", "계좌동결"]
#         if any(keyword in keywords for keyword in critical_keywords):
#             return True
        
#         # 3. 특정 카테고리의 고위험 패턴
#         if category in ["납치협박", "기관사칭"] and score >= 0.7:
#             return True
        
#         # 4. 악성앱 관련 즉시 차단 필요
#         if category == "악성앱" and score >= 0.8:
#             return True
        
#         return False
    
#     def _calculate_confidence(self, 
#                              keyword_score: float, 
#                              pattern_score: float, 
#                              keyword_count: int) -> float:
#         """분석 신뢰도 계산"""
        
#         # 기본 신뢰도
#         base_confidence = 0.5
        
#         # 키워드 점수가 높을수록 신뢰도 증가
#         keyword_boost = min(keyword_score * 0.3, 0.3)
        
#         # 패턴 매칭 점수가 높을수록 신뢰도 증가
#         pattern_boost = min(pattern_score * 0.2, 0.2)
        
#         # 다양한 키워드 감지 시 신뢰도 증가
#         diversity_boost = min(keyword_count * 0.05, 0.2)
        
#         confidence = base_confidence + keyword_boost + pattern_boost + diversity_boost
        
#         return min(confidence, 1.0)
    
#     def _update_stats(self, score: float, category: str, processing_time: float):
#         """통계 업데이트"""
#         self.detection_stats["total_detections"] += 1
        
#         if score >= detection_thresholds.high_risk:
#             self.detection_stats["high_risk_detections"] += 1
        
#         # 평균 처리 시간 업데이트
#         total_time = (self.detection_stats["avg_processing_time"] * 
#                      (self.detection_stats["total_detections"] - 1) + processing_time)
#         self.detection_stats["avg_processing_time"] = total_time / self.detection_stats["total_detections"]
        
#         # 패턴별 적중률 업데이트
#         if category not in self.detection_stats["pattern_hit_rate"]:
#             self.detection_stats["pattern_hit_rate"][category] = 0
#         self.detection_stats["pattern_hit_rate"][category] += 1
    
#     async def quick_risk_assessment(self, text: str) -> Tuple[float, bool]:
#         """초고속 위험도 평가 (응급용)"""
        
#         # 극도로 빠른 키워드 체크만 수행
#         critical_keywords = ["납치", "체포", "계좌동결", "죽는다", "응급실"]
        
#         text_lower = text.lower()
#         for keyword in critical_keywords:
#             if keyword in text_lower:
#                 return 0.95, True  # 매우 높은 위험도, 즉시 알림
        
#         # 기본 키워드 체크
#         risky_keywords = ["금융감독원", "검찰청", "앱설치", "대출", "만나서"]
#         risk_count = sum(1 for keyword in risky_keywords if keyword in text_lower)
        
#         quick_score = min(risk_count * 0.2, 0.8)
#         needs_alert = quick_score >= 0.6
        
#         return quick_score, needs_alert
    
#     def get_performance_metrics(self) -> Dict[str, Any]:
#         """성능 메트릭 반환"""
#         return {
#             "agent_name": self.name,
#             "total_detections": self.detection_stats["total_detections"],
#             "high_risk_rate": (
#                 self.detection_stats["high_risk_detections"] / 
#                 max(1, self.detection_stats["total_detections"])
#             ),
#             "avg_processing_time": self.detection_stats["avg_processing_time"],
#             "pattern_hit_rate": self.detection_stats["pattern_hit_rate"],
#             "loaded_patterns": len(self.compiled_patterns),
#             "loaded_keywords": len(self.keyword_weights)
#         }
    
#     async def update_patterns(self, new_patterns: Dict[str, List[str]]):
#         """새로운 패턴 동적 업데이트"""
#         try:
#             for category, patterns in new_patterns.items():
#                 if category not in self.compiled_patterns:
#                     self.compiled_patterns[category] = []
                
#                 # 새 패턴들을 컴파일해서 추가
#                 compiled_new = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
#                 self.compiled_patterns[category].extend(compiled_new)
            
#             logger.info(f"패턴 업데이트 완료: {len(new_patterns)}개 카테고리")
            
#         except Exception as e:
#             logger.error(f"패턴 업데이트 실패: {e}")
    
#     def reset_stats(self):
#         """통계 초기화"""
#         self.detection_stats = {
#             "total_detections": 0,
#             "high_risk_detections": 0,
#             "avg_processing_time": 0.0,
#             "pattern_hit_rate": {}
#         }
#         logger.info("탐지 에이전트 통계 초기화")