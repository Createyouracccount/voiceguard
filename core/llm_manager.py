"""
VoiceGuard AI - LLM 관리 엔진
비용 최적화와 정확도를 동시에 고려한 지능적 모델 라우팅
"""
import os
import asyncio
import time
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import json
import logging
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import (
    AIConfig, RiskLevel, ModelTier, detection_thresholds,
    # monitoring_config
)

logger = logging.getLogger(__name__)

@dataclass
class LLMResponse:
    """LLM 응답 표준 포맷"""
    content: str
    confidence: float
    processing_time: float
    model_used: str
    cost_estimate: float
    risk_level: RiskLevel
    metadata: Dict[str, Any]

class ModelSelector:
    """위험도와 비용을 고려한 지능적 모델 선택"""
    def __init__(self):
        # 2. 사용 통계(usage_stats) 수정
        self.usage_stats = {
            "gpt-4": {"calls": 0, "total_cost": 0.0, "avg_accuracy": 0.95},
            "gemini-pro": {"calls": 0, "total_cost": 0.0, "avg_accuracy": 0.90},
            "gpt-3.5-turbo": {"calls": 0, "total_cost": 0.0, "avg_accuracy": 0.80}
        }
        
        
    def select_model(self, risk_level: RiskLevel, 
                    context_length: int = 0, 
                    budget_remaining: float = 1000.0) -> str:
        """
        위험도, 컨텍스트 길이, 남은 예산을 고려한 모델 선택
        """
        
        # 1. 위험도 기반 초기 선택
        if risk_level == RiskLevel.CRITICAL:
            if budget_remaining > 50:  # 충분한 예산이 있으면 최고 모델
                return "gpt-4"
            else:
                return "claude-3-sonnet"  # 예산 부족시 차선책
                
        elif risk_level == RiskLevel.HIGH:
            if context_length > 1500 or budget_remaining > 30:
                return "claude-3-sonnet"
            else:
                return "gpt-3.5-turbo"
                
        elif risk_level == RiskLevel.MEDIUM:
            if context_length > 2000:
                return "gemini-pro" # claude-3-sonnet -> gemini-pro
            else:
                return "gpt-3.5-turbo"
                
        else:  # LOW risk
            return "gpt-3.5-turbo"
    
    def should_use_ensemble(self, risk_level: RiskLevel, 
                           confidence: float) -> bool:
        """앙상블 사용 여부 결정"""
        if risk_level == RiskLevel.CRITICAL:
            return True
        if risk_level == RiskLevel.HIGH and confidence < 0.8:
            return True
        return False

class LLMManager:
    """LLM 통합 관리 시스템"""
    
    def __init__(self):
        self.selector = ModelSelector()
        self._initialize_models()
        self.total_cost = 0.0
        self.call_count = 0
        
    def _initialize_models(self):
        """모델 인스턴스 초기화"""
        # 4. 모델 초기화 로직 수정
        self.models = {
            "gpt-4": ChatOpenAI(
                model="gpt-4",
                temperature=AIConfig.GPT4.temperature,
                max_tokens=AIConfig.GPT4.max_tokens
            ),
            "gpt-3.5-turbo": ChatOpenAI(
                model="gpt-3.5-turbo", 
                temperature=AIConfig.GPT35_TURBO.temperature,
                max_tokens=AIConfig.GPT35_TURBO.max_tokens
            ),
            # "claude-3-sonnet"을 "gemini-pro"로 교체
            "gemini-pro": ChatGoogleGenerativeAI(
                model="gemini-1.5-pro-flash",
                temperature=AIConfig.GEMINI.temperature,
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
        }
    
    # LLMManager 클래스 추가 메서드

    def get_model(self, model_name: str):
        """특정 모델 인스턴스 반환"""
        if model_name not in self.models:
            raise ValueError(f"모델 '{model_name}'을 찾을 수 없습니다. 사용 가능한 모델: {list(self.models.keys())}")
        return self.models[model_name]

    def get_available_models(self) -> List[str]:
        """사용 가능한 모델 목록 반환"""
        return list(self.models.keys())

    async def test_connection(self, model_name: str = None) -> Dict[str, Any]:
        """모델 연결 테스트"""
        if model_name:
            # 특정 모델 테스트
            try:
                model = self.get_model(model_name)
                response = await model.ainvoke("테스트 메시지입니다.")
                return {
                    "model": model_name,
                    "status": "success",
                    "response": response.content[:100] + "..." if len(response.content) > 100 else response.content
                }
            except Exception as e:
                return {
                    "model": model_name,
                    "status": "failed",
                    "error": str(e)
                }
        else:
            # 모든 모델 테스트
            results = {}
            for name in self.models.keys():
                results[name] = await self.test_connection(name)
            return results

    async def analyze_scam_risk(self, 
                               text: str, 
                               context: Dict[str, Any] = None,
                               force_model: Optional[str] = None) -> LLMResponse:
        """
        보이스피싱 위험도 분석 - 핵심 메서드
        """
        start_time = time.time()
        context = context or {}
        
        # 1. 초기 위험도 추정 (빠른 키워드 기반)
        initial_risk = self._estimate_initial_risk(text)
        
        # 2. 모델 선택
        if force_model:
            selected_model = force_model
        else:
            selected_model = self.selector.select_model(
                initial_risk, 
                len(text), 
                self._get_remaining_budget()
            )
        
        # 3. 프롬프트 구성
        prompt = self._build_analysis_prompt(text, context, initial_risk)
        
        # 4. LLM 분석 실행
        try:
            if self.selector.should_use_ensemble(initial_risk, 0.7):
                response = await self._ensemble_analysis(prompt, text)
            else:
                response = await self._single_model_analysis(
                    selected_model, prompt, text
                )
                
            processing_time = time.time() - start_time
            
            # 5. 응답 후처리 및 검증
            validated_response = self._validate_response(response, processing_time)
            
            # 6. 통계 업데이트
            self._update_usage_stats(selected_model, validated_response)
            
            return validated_response
            
        except Exception as e:
            logger.error(f"LLM 분석 중 오류: {e}")
            # 폴백 응답
            return self._create_fallback_response(text, initial_risk)
    
    def _estimate_initial_risk(self, text: str) -> RiskLevel:
        """빠른 키워드 기반 초기 위험도 추정"""
        text_lower = text.lower()
        
        # 고위험 키워드 체크
        critical_keywords = [
            "납치", "협박", "위험", "응급실", "사고났", "죽는다"
        ]
        high_risk_keywords = [
            "금융감독원", "검찰청", "경찰서", "계좌동결", "체포영장"
        ]
        medium_risk_keywords = [
            "대출", "저금리", "정부지원금", "환급", "당첨"
        ]
        
        if any(keyword in text_lower for keyword in critical_keywords):
            return RiskLevel.CRITICAL
        elif any(keyword in text_lower for keyword in high_risk_keywords):
            return RiskLevel.HIGH
        elif any(keyword in text_lower for keyword in medium_risk_keywords):
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _build_analysis_prompt(self, text: str, context: Dict, 
                              initial_risk: RiskLevel) -> ChatPromptTemplate:
        """맞춤형 분석 프롬프트 구성"""
        
        system_prompt = """
당신은 보이스피싱 탐지 전문가입니다. 다음 기준으로 분석해주세요:

## 분석 기준
1. **사기 유형 분류**: 대포통장, 대포폰, 악성앱, 미끼문자, 기관사칭, 납치협박, 대출사기, 가상자산
2. **위험도 계산**: 0.0(안전) ~ 1.0(매우 위험)
3. **확신도 평가**: 분석 결과에 대한 신뢰도
4. **즉시 대응 필요성**: 실시간 차단이 필요한지 여부

## 특별 주의사항
- 대면편취형 사기 급증 (문서 근거: 7.5% → 64.4%)
- "만나서", "직접", "현장" 등 대면 관련 키워드 주의
- 금융기관/공공기관 사칭 시 즉시 고위험 판정

## 응답 형식 (JSON)
{{
    "scam_type": "분류된 사기 유형",
    "risk_score": 0.0-1.0,
    "confidence": 0.0-1.0,
    "immediate_action": true/false,
    "key_indicators": ["탐지된 주요 지표들"],
    "reasoning": "판단 근거"
}}
"""
        
        human_prompt = f"""
분석할 텍스트: "{text}"

추가 컨텍스트:
- 통화 시간: {context.get('call_duration', 'N/A')}초
- 발신자 정보: {context.get('caller_info', '알 수 없음')}
- 초기 위험도 추정: {initial_risk.value}

위 정보를 종합하여 보이스피싱 위험도를 분석해주세요.
"""
        
        return ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ])
    
    async def _single_model_analysis(self, model_name: str, 
                                   prompt: ChatPromptTemplate, 
                                   text: str) -> Dict[str, Any]:
        """단일 모델 분석"""
        model = self.models[model_name]
        
        # JSON 파서 연결
        parser = JsonOutputParser()
        chain = prompt | model | parser
        
        try:
            result = await chain.ainvoke({"text": text})
            result["model_used"] = model_name
            return result
        except Exception as e:
            logger.error(f"모델 {model_name} 분석 실패: {e}")
            raise
    
    async def _ensemble_analysis(self, prompt: ChatPromptTemplate, 
                                text: str) -> Dict[str, Any]:
        """앙상블 분석 - 여러 모델 결과 종합"""
        
        # 병렬로 여러 모델 실행
        tasks = []
        models_to_use = ["gpt-4", "gemini-pro"]
        
        for model_name in models_to_use:
            task = self._single_model_analysis(model_name, prompt, text)
            tasks.append(task)
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 성공한 결과만 필터링
            valid_results = [r for r in results if not isinstance(r, Exception)]
            
            if not valid_results:
                raise Exception("모든 모델에서 분석 실패")
            
            # 앙상블 결과 계산
            ensemble_result = self._combine_ensemble_results(valid_results)
            ensemble_result["model_used"] = "ensemble_" + "_".join(models_to_use)
            
            return ensemble_result
            
        except Exception as e:
            logger.error(f"앙상블 분석 실패: {e}")
            # 단일 모델로 폴백
            return await self._single_model_analysis("gpt-3.5-turbo", prompt, text)
    
    def _combine_ensemble_results(self, results: List[Dict]) -> Dict[str, Any]:
        """앙상블 결과 통합"""
        if len(results) == 1:
            return results[0]
        
        # 위험도 점수 가중 평균 (높은 신뢰도에 더 큰 가중치)
        total_weight = sum(r.get("confidence", 0.5) for r in results)
        weighted_risk = sum(
            r.get("risk_score", 0.5) * r.get("confidence", 0.5) 
            for r in results
        ) / total_weight if total_weight > 0 else 0.5
        
        # 최고 위험도 기준으로 즉시 대응 여부 결정
        immediate_action = any(r.get("immediate_action", False) for r in results)
        
        # 가장 높은 신뢰도의 결과에서 사기 유형 선택
        best_result = max(results, key=lambda x: x.get("confidence", 0))
        
        # 모든 모델의 지표를 합집합으로 통합
        all_indicators = []
        for r in results:
            all_indicators.extend(r.get("key_indicators", []))
        unique_indicators = list(set(all_indicators))
        
        return {
            "scam_type": best_result.get("scam_type", "unknown"),
            "risk_score": min(weighted_risk * 1.1, 1.0),  # 앙상블은 약간 보수적으로
            "confidence": max(r.get("confidence", 0.5) for r in results),
            "immediate_action": immediate_action,
            "key_indicators": unique_indicators,
            "reasoning": f"앙상블 분석 ({len(results)}개 모델): " + 
                        "; ".join(r.get("reasoning", "") for r in results)
        }
    
    def _validate_response(self, response: Dict[str, Any], 
                          processing_time: float) -> LLMResponse:
        """응답 검증 및 표준화"""
        
        # 필수 필드 검증
        risk_score = max(0.0, min(1.0, response.get("risk_score", 0.5)))
        confidence = max(0.0, min(1.0, response.get("confidence", 0.5)))
        
        # 위험도 레벨 결정
        if risk_score >= detection_thresholds.critical_risk:
            risk_level = RiskLevel.CRITICAL
        elif risk_score >= detection_thresholds.high_risk:
            risk_level = RiskLevel.HIGH
        elif risk_score >= detection_thresholds.medium_risk:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW
        
        # 비용 추정
        model_used = response.get("model_used", "unknown")
        cost_estimate = self._estimate_cost(model_used, response)
        
        return LLMResponse(
            content=response.get("reasoning", ""),
            confidence=confidence,
            processing_time=processing_time,
            model_used=model_used,
            cost_estimate=cost_estimate,
            risk_level=risk_level,
            metadata={
                "scam_type": response.get("scam_type"),
                "risk_score": risk_score,
                "immediate_action": response.get("immediate_action", False),
                "key_indicators": response.get("key_indicators", []),
                "ensemble_used": "ensemble" in model_used
            }
        )
    
    def _estimate_cost(self, model_name: str, response: Dict) -> float:
        """비용 추정"""
        base_model = model_name.replace("ensemble_", "").split("_")[0]
        
        # 6. 비용 추정 맵 수정
        cost_map = {
            "gpt-4": 0.03,
            "gemini-pro": 0.012,
            "gpt-3.5-turbo": 0.001
        }
        
        base_cost = cost_map.get(base_model, 0.01)
        
        # 앙상블 사용시 비용 증가
        if "ensemble" in model_name:
            base_cost *= 2.5  # 여러 모델 사용으로 인한 비용 증가
        
        return base_cost
    
    def _create_fallback_response(self, text: str, 
                                 initial_risk: RiskLevel) -> LLMResponse:
        """LLM 실패시 폴백 응답"""
        
        # 키워드 기반 간단한 분석
        risk_score = 0.3
        if initial_risk == RiskLevel.CRITICAL:
            risk_score = 0.9
        elif initial_risk == RiskLevel.HIGH:
            risk_score = 0.7
        elif initial_risk == RiskLevel.MEDIUM:
            risk_score = 0.5
        
        return LLMResponse(
            content="LLM 분석 실패로 인한 키워드 기반 분석",
            confidence=0.6,
            processing_time=0.1,
            model_used="fallback_keyword",
            cost_estimate=0.0,
            risk_level=initial_risk,
            metadata={
                "scam_type": "unknown",
                "risk_score": risk_score,
                "immediate_action": initial_risk in [RiskLevel.HIGH, RiskLevel.CRITICAL],
                "key_indicators": ["폴백 분석"],
                "fallback_used": True
            }
        )
    
    def _update_usage_stats(self, model_name: str, response: LLMResponse):
        """사용 통계 업데이트"""
        base_model = model_name.replace("ensemble_", "").split("_")[0]
        
        if base_model in self.selector.usage_stats:
            stats = self.selector.usage_stats[base_model]
            stats["calls"] += 1
            stats["total_cost"] += response.cost_estimate
            
            # 정확도는 별도 피드백 시스템으로 업데이트 (향후 구현)
        
        self.total_cost += response.cost_estimate
        self.call_count += 1
    
    def _get_remaining_budget(self) -> float:
        """남은 예산 계산"""
        daily_budget = 1000.0  # 일일 예산 1000원
        return max(0, daily_budget - self.total_cost)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계 반환"""
        return {
            "total_calls": self.call_count,
            "total_cost": self.total_cost,
            "avg_cost_per_call": self.total_cost / max(1, self.call_count),
            "model_usage": self.selector.usage_stats,
            "remaining_budget": self._get_remaining_budget()
        }
    
    async def health_check(self) -> Dict[str, bool]:
        """모델 상태 확인"""
        health_status = {}
        
        for model_name, model in self.models.items():
            try:
                # 간단한 테스트 쿼리
                test_prompt = ChatPromptTemplate.from_messages([
                    HumanMessage(content="테스트")
                ])
                await (test_prompt | model).ainvoke({})
                health_status[model_name] = True
            except Exception:
                health_status[model_name] = False
                logger.warning(f"모델 {model_name} 상태 이상")
        
        return health_status

    

# 전역 LLM 매니저 인스턴스
llm_manager = LLMManager()


import os

# LangSmith 트래킹 완전 비활성화
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = ""
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["LANGCHAIN_PROJECT"] = ""