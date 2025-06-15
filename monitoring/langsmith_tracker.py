"""
VoiceGuard AI - LangSmith 모니터링 트래커
실시간 성능 추적 및 비용 모니터링
"""

import os
import time
import json
import logging
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import asyncio
from functools import wraps

from langsmith import Client
from langsmith.run_trees import RunTree
from langchain_core.callbacks import BaseCallbackHandler

from config.settings import monitoring_config, ai_config

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """성능 메트릭"""
    latency_ms: float
    tokens_used: int
    model_name: str
    cost_estimate: float
    accuracy_score: Optional[float] = None
    error_rate: float = 0.0

@dataclass
class DetectionMetrics:
    """탐지 관련 메트릭"""
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    
    @property
    def accuracy(self) -> float:
        total = self.true_positives + self.false_positives + self.true_negatives + self.false_negatives
        if total == 0:
            return 0.0
        correct = self.true_positives + self.true_negatives
        return correct / total
    
    @property
    def precision(self) -> float:
        if self.true_positives + self.false_positives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_positives)
    
    @property
    def recall(self) -> float:
        if self.true_positives + self.false_negatives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_negatives)
    
    @property
    def f1_score(self) -> float:
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)

class VoiceGuardCallbackHandler(BaseCallbackHandler):
    """LangChain 콜백 핸들러"""
    
    def __init__(self, run_name: str, metadata: Dict[str, Any] = None):
        self.run_name = run_name
        self.metadata = metadata or {}
        self.start_time = None
        self.tokens_used = 0
        
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs):
        """LLM 호출 시작"""
        self.start_time = time.time()
        logger.debug(f"LLM 시작: {self.run_name}")
        
    def on_llm_end(self, response, **kwargs):
        """LLM 호출 완료"""
        if self.start_time:
            latency = (time.time() - self.start_time) * 1000
            
            # 토큰 사용량 추정 (실제로는 response에서 가져와야 함)
            if hasattr(response, 'llm_output') and response.llm_output:
                self.tokens_used = response.llm_output.get('token_usage', {}).get('total_tokens', 0)
            
            logger.info(f"LLM 완료: {self.run_name}, 지연시간: {latency:.2f}ms, 토큰: {self.tokens_used}")
    
    def on_llm_error(self, error: Exception, **kwargs):
        """LLM 오류 발생"""
        logger.error(f"LLM 오류: {self.run_name}, 에러: {str(error)}")

class LangSmithTracker:
    """LangSmith 성능 추적 시스템"""
    
    def __init__(self):
        self.client = None
        self.project_name = monitoring_config.LANGSMITH_PROJECT
        self.enabled = monitoring_config.ENABLE_LANGSMITH
        
        # 메트릭 저장소
        self.performance_history: List[PerformanceMetrics] = []
        self.detection_metrics = DetectionMetrics()
        self.daily_costs = {}
        
        # 초기화
        if self.enabled:
            self._initialize_client()
    
    def _initialize_client(self):
        """LangSmith 클라이언트 초기화"""
        try:
            if monitoring_config.LANGSMITH_API_KEY:
                os.environ["LANGCHAIN_TRACING_V2"] = "true"
                os.environ["LANGCHAIN_API_KEY"] = monitoring_config.LANGSMITH_API_KEY
                os.environ["LANGCHAIN_PROJECT"] = self.project_name
                
                self.client = Client()
                logger.info("LangSmith 클라이언트 초기화 완료")
            else:
                logger.warning("LangSmith API 키가 없어 추적이 비활성화됩니다")
                self.enabled = False
                
        except Exception as e:
            logger.error(f"LangSmith 초기화 실패: {e}")
            self.enabled = False
    
    def track_detection(self, func: Callable) -> Callable:
        """탐지 함수 추적 데코레이터"""
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            if not self.enabled:
                return await func(*args, **kwargs)
            
            run_tree = RunTree(
                name=f"detection_{func.__name__}",
                run_type="chain",
                inputs={"args": str(args[:2]), "kwargs": str(list(kwargs.keys()))}
            )
            
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                latency = (time.time() - start_time) * 1000
                
                # 성능 메트릭 기록
                self._record_performance(
                    func_name=func.__name__,
                    latency_ms=latency,
                    success=True,
                    result=result
                )
                
                run_tree.end(outputs={"success": True, "latency_ms": latency})
                
                if self.client:
                    self.client.create_run(run_tree)
                
                return result
                
            except Exception as e:
                run_tree.end(error=str(e))
                
                if self.client:
                    self.client.create_run(run_tree)
                
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            if not self.enabled:
                return func(*args, **kwargs)
            
            # 동기 함수 버전 (비슷한 로직)
            return func(*args, **kwargs)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    def track_llm_call(self, model_name: str, run_name: str, 
                      metadata: Dict[str, Any] = None) -> VoiceGuardCallbackHandler:
        """LLM 호출 추적"""
        if not self.enabled:
            return VoiceGuardCallbackHandler(run_name, metadata)
        
        # 메타데이터에 모델 정보 추가
        enhanced_metadata = {
            "model": model_name,
            "timestamp": datetime.now().isoformat(),
            **(metadata or {})
        }
        
        return VoiceGuardCallbackHandler(run_name, enhanced_metadata)
    
    def _record_performance(self, func_name: str, latency_ms: float, 
                           success: bool, result: Any = None):
        """성능 메트릭 기록"""
        
        # 모델 정보 추출 (결과에서 가져오기)
        model_name = "unknown"
        tokens_used = 0
        
        if result and hasattr(result, 'model_used'):
            model_name = result.model_used
        
        # 비용 계산
        cost_estimate = self._calculate_cost(model_name, tokens_used)
        
        # 메트릭 저장
        metric = PerformanceMetrics(
            latency_ms=latency_ms,
            tokens_used=tokens_used,
            model_name=model_name,
            cost_estimate=cost_estimate,
            error_rate=0.0 if success else 1.0
        )
        
        self.performance_history.append(metric)
        
        # 일일 비용 업데이트
        today = datetime.now().strftime("%Y-%m-%d")
        if today not in self.daily_costs:
            self.daily_costs[today] = 0.0
        self.daily_costs[today] += cost_estimate
        
        # 메모리 관리 (최근 1000개만 유지)
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
    
    def _calculate_cost(self, model_name: str, tokens: int) -> float:
        """비용 계산"""
        cost_per_1k = {
            "gpt-4": ai_config.GPT4.cost_per_1k_tokens,
            "gpt-3.5-turbo": ai_config.GPT35_TURBO.cost_per_1k_tokens,
            "claude-3-sonnet": ai_config.CLAUDE.cost_per_1k_tokens
        }
        
        base_model = model_name.split("-")[0] if "-" in model_name else model_name
        rate = cost_per_1k.get(base_model, 0.01)  # 기본값
        
        return (tokens / 1000) * rate
    
    def update_detection_metrics(self, prediction: bool, actual: bool):
        """탐지 정확도 메트릭 업데이트"""
        if prediction and actual:
            self.detection_metrics.true_positives += 1
        elif prediction and not actual:
            self.detection_metrics.false_positives += 1
        elif not prediction and actual:
            self.detection_metrics.false_negatives += 1
        else:
            self.detection_metrics.true_negatives += 1
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """성능 요약 통계"""
        if not self.performance_history:
            return {
                "avg_latency_ms": 0,
                "total_calls": 0,
                "error_rate": 0,
                "total_cost": 0,
                "detection_accuracy": 0
            }
        
        total_calls = len(self.performance_history)
        avg_latency = sum(m.latency_ms for m in self.performance_history) / total_calls
        error_rate = sum(m.error_rate for m in self.performance_history) / total_calls
        total_cost = sum(m.cost_estimate for m in self.performance_history)
        
        return {
            "avg_latency_ms": round(avg_latency, 2),
            "total_calls": total_calls,
            "error_rate": round(error_rate, 4),
            "total_cost": round(total_cost, 2),
            "cost_today": round(self.daily_costs.get(datetime.now().strftime("%Y-%m-%d"), 0), 2),
            "detection_metrics": {
                "accuracy": round(self.detection_metrics.accuracy, 3),
                "precision": round(self.detection_metrics.precision, 3),
                "recall": round(self.detection_metrics.recall, 3),
                "f1_score": round(self.detection_metrics.f1_score, 3)
            },
            "model_usage": self._get_model_usage_stats()
        }
    
    def _get_model_usage_stats(self) -> Dict[str, Dict[str, Any]]:
        """모델별 사용 통계"""
        model_stats = {}
        
        for metric in self.performance_history:
            model = metric.model_name
            if model not in model_stats:
                model_stats[model] = {
                    "calls": 0,
                    "total_cost": 0,
                    "avg_latency": 0,
                    "total_latency": 0
                }
            
            model_stats[model]["calls"] += 1
            model_stats[model]["total_cost"] += metric.cost_estimate
            model_stats[model]["total_latency"] += metric.latency_ms
        
        # 평균 계산
        for model, stats in model_stats.items():
            if stats["calls"] > 0:
                stats["avg_latency"] = round(stats["total_latency"] / stats["calls"], 2)
            del stats["total_latency"]
            stats["total_cost"] = round(stats["total_cost"], 2)
        
        return model_stats
    
    def export_metrics(self, filepath: str):
        """메트릭을 파일로 내보내기"""
        try:
            metrics_data = {
                "timestamp": datetime.now().isoformat(),
                "summary": self.get_performance_summary(),
                "detection_metrics": asdict(self.detection_metrics),
                "daily_costs": self.daily_costs,
                "recent_performance": [
                    asdict(m) for m in self.performance_history[-100:]
                ]
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(metrics_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"메트릭 내보내기 완료: {filepath}")
            
        except Exception as e:
            logger.error(f"메트릭 내보내기 실패: {e}")
    
    def create_alert(self, alert_type: str, message: str, severity: str = "info"):
        """알림 생성"""
        alert_data = {
            "type": alert_type,
            "message": message,
            "severity": severity,
            "timestamp": datetime.now().isoformat(),
            "project": self.project_name
        }
        
        # LangSmith에 알림 기록
        if self.client and self.enabled:
            try:
                run_tree = RunTree(
                    name="alert",
                    run_type="tool",
                    inputs=alert_data
                )
                run_tree.end(outputs={"logged": True})
                self.client.create_run(run_tree)
            except Exception as e:
                logger.error(f"알림 기록 실패: {e}")
        
        # 중요 알림은 웹훅으로도 전송
        if severity in ["warning", "error", "critical"]:
            self._send_webhook_alert(alert_data)
    
    def _send_webhook_alert(self, alert_data: Dict[str, Any]):
        """웹훅으로 알림 전송"""
        if not monitoring_config.ALERT_WEBHOOK_URL:
            return
        
        # 실제 환경에서는 requests나 httpx 사용
        logger.info(f"웹훅 알림 전송: {alert_data}")
    
    def check_performance_thresholds(self):
        """성능 임계값 체크 및 알림"""
        summary = self.get_performance_summary()
        
        # 지연시간 체크
        if summary["avg_latency_ms"] > monitoring_config.TARGET_RESPONSE_TIME * 1000:
            self.create_alert(
                "performance",
                f"평균 응답 시간이 목표치를 초과했습니다: {summary['avg_latency_ms']}ms",
                "warning"
            )
        
        # 정확도 체크
        if summary["detection_metrics"]["accuracy"] < monitoring_config.TARGET_ACCURACY:
            self.create_alert(
                "accuracy",
                f"탐지 정확도가 목표치 미달입니다: {summary['detection_metrics']['accuracy']}",
                "critical"
            )
        
        # 일일 비용 체크
        if summary["cost_today"] > 100:  # $100 초과
            self.create_alert(
                "cost",
                f"일일 비용이 $100를 초과했습니다: ${summary['cost_today']}",
                "warning"
            )

# 전역 트래커 인스턴스
tracker = LangSmithTracker()