"""
VoiceGuard AI - Coordinator Agent
멀티 에이전트 시스템의 중앙 조정자
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import Document

from .detection_agent import DetectionAgent
from .analysis_agent import AnalysisAgent
from .response_agent import ResponseAgent

from core.llm_manager import llm_manager
from config.settings import RiskLevel, detection_thresholds, ai_config
from monitoring.langsmith_tracker import tracker

logger = logging.getLogger(__name__)

class TaskType(Enum):
    """작업 유형"""
    DETECTION = "detection"
    ANALYSIS = "analysis"
    RESPONSE = "response"
    VERIFICATION = "verification"
    ESCALATION = "escalation"

class TaskPriority(Enum):
    """작업 우선순위"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class Task:
    """에이전트 작업"""
    id: str
    type: TaskType
    priority: TaskPriority
    data: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)
    assigned_to: Optional[str] = None
    status: str = "pending"
    result: Optional[Any] = None
    error: Optional[str] = None

@dataclass
class AgentState:
    """에이전트 상태"""
    agent_id: str
    agent_type: str
    is_busy: bool = False
    current_task: Optional[str] = None
    completed_tasks: int = 0
    error_count: int = 0
    avg_processing_time: float = 0.0

class CoordinatorAgent:
    """멀티 에이전트 시스템 조정자"""
    
    def __init__(self, detection_agent: DetectionAgent, analysis_agent: AnalysisAgent, response_agent: ResponseAgent):
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.priority_queue: List[Task] = []
        self.agents: Dict[str, AgentState] = {}
        self.task_history: Dict[str, Task] = {}

        self.detection_agent = detection_agent
        self.analysis_agent = analysis_agent
        self.response_agent = response_agent
        
        # 에이전트 등록
        self._register_agents()
        
        # 조정 전략
        self.coordination_prompt = self._build_coordination_prompt()
        
        # 실행 상태
        self.is_running = False
        self.coordination_tasks = []

        # 에이전트와 태스크 타입 매핑
        self.agent_map = {
            TaskType.DETECTION: self.detection_agent,
            TaskType.ANALYSIS: self.analysis_agent,
            TaskType.RESPONSE: self.response_agent,
        }
        
        logger.info("Coordinator Agent 초기화 완료")
    
    def _register_agents(self):
        """에이전트 등록"""
        self.agents = {
            "detection_agent": AgentState(
                agent_id="detection_agent",
                agent_type="detection"
            ),
            "analysis_agent": AgentState(
                agent_id="analysis_agent",
                agent_type="analysis"
            ),
            "response_agent": AgentState(
                agent_id="response_agent",
                agent_type="response"
            )
        }
    
    def _build_coordination_prompt(self) -> ChatPromptTemplate:
        """조정 전략 프롬프트"""
        return ChatPromptTemplate.from_messages([
            SystemMessage(content="""
당신은 보이스피싱 탐지 시스템의 조정자입니다.
여러 전문 에이전트들을 효율적으로 조율하여 최적의 결과를 도출해야 합니다.

## 에이전트 역할
1. **Detection Agent**: 초기 위험 탐지 및 분류
2. **Analysis Agent**: 심층 분석 및 패턴 인식
3. **Response Agent**: 대응 전략 수립 및 실행

## 조정 원칙
1. **우선순위**: Critical > High > Medium > Low
2. **병렬 처리**: 독립적인 작업은 동시 실행
3. **순차 처리**: 의존성이 있는 작업은 순서대로
4. **부하 분산**: 에이전트 상태를 고려한 작업 할당
5. **실패 처리**: 실패 시 재시도 또는 대체 전략

## 의사결정 기준
- 위험도가 높을수록 더 많은 에이전트 투입
- 응답 시간이 중요한 경우 병렬 처리 우선
- 정확도가 중요한 경우 순차적 검증 수행
"""),
            HumanMessage(content="""
현재 상황:
- 대기 중인 작업: {pending_tasks}
- 활성 에이전트: {active_agents}
- 평균 처리 시간: {avg_processing_time}
- 현재 부하: {system_load}

새로운 작업:
- 유형: {task_type}
- 우선순위: {priority}
- 데이터: {task_data}

이 작업을 어떻게 처리해야 할까요? 다음 형식으로 응답하세요:
{{
    "assign_to": ["agent_id1", "agent_id2"],
    "processing_strategy": "parallel|sequential|hybrid",
    "estimated_time": seconds,
    "additional_resources": [],
    "reasoning": "판단 근거"
}}
""")
        ])
    
    async def start(self):
        """조정자 시작"""
        if self.is_running:
            logger.warning("Coordinator Agent가 이미 실행 중입니다")
            return
        
        self.is_running = True
        
        # 작업 처리 루프 시작
        self.coordination_tasks = [
            asyncio.create_task(self._process_task_queue()),
            asyncio.create_task(self._monitor_agents()),
            asyncio.create_task(self._optimize_performance())
        ]
        
        logger.info("Coordinator Agent 시작됨")
    
    async def stop(self):
        """조정자 중지"""
        self.is_running = False
        
        # 모든 작업 완료 대기
        for task in self.coordination_tasks:
            task.cancel()
        
        await asyncio.gather(*self.coordination_tasks, return_exceptions=True)
        
        logger.info("Coordinator Agent 중지됨")
    
    @tracker.track_detection
    async def submit_task(self, 
                         task_type: TaskType,
                         data: Dict[str, Any],
                         priority: Optional[TaskPriority] = None) -> str:
        """새 작업 제출"""
        
        # 우선순위 자동 결정
        if priority is None:
            priority = self._determine_priority(task_type, data)
        
        # 작업 생성
        task = Task(
            id=f"task_{datetime.now().timestamp()}",
            type=task_type,
            priority=priority,
            data=data
        )
        
        # 작업 저장
        self.task_history[task.id] = task
        
        # 우선순위에 따라 큐에 추가
        if priority == TaskPriority.CRITICAL:
            self.priority_queue.insert(0, task)
        else:
            await self.task_queue.put(task)
        
        logger.info(f"새 작업 제출: {task.id}, 유형: {task_type.value}, 우선순위: {priority.name}")
        
        return task.id
    
    def _determine_priority(self, task_type: TaskType, data: Dict[str, Any]) -> TaskPriority:
        """작업 우선순위 자동 결정"""
        
        # 위험도 기반 우선순위
        risk_score = data.get("risk_score", 0)
        
        if risk_score >= detection_thresholds.critical_risk:
            return TaskPriority.CRITICAL
        elif risk_score >= detection_thresholds.high_risk:
            return TaskPriority.HIGH
        elif risk_score >= detection_thresholds.medium_risk:
            return TaskPriority.MEDIUM
        else:
            return TaskPriority.LOW
    
    async def _process_task_queue(self):
        """작업 큐 처리"""
        while self.is_running:
            try:
                # 우선순위 큐 먼저 확인
                if self.priority_queue:
                    task = self.priority_queue.pop(0)
                else:
                    # 일반 큐에서 가져오기 (타임아웃 설정)
                    task = await asyncio.wait_for(
                        self.task_queue.get(), 
                        timeout=1.0
                    )
                
                # 작업 처리
                await self._process_task(task)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"작업 처리 중 오류: {e}")
    
    async def _process_task(self, task: Task):
        """개별 작업 처리"""
        try:
            # 작업 상태 업데이트
            task.status = "processing"
            
            # 최적 에이전트 선택
            strategy = await self._determine_strategy(task)
            
            # 에이전트 할당 및 실행
            if strategy["processing_strategy"] == "parallel":
                results = await self._execute_parallel(task, strategy["assign_to"])
            elif strategy["processing_strategy"] == "sequential":
                results = await self._execute_sequential(task, strategy["assign_to"])
            else:  # hybrid
                results = await self._execute_hybrid(task, strategy["assign_to"])
            
            # 결과 통합
            task.result = self._integrate_results(results)
            task.status = "completed"
            
            logger.info(f"작업 완료: {task.id}")
            
        except Exception as e:
            task.status = "failed"
            task.error = str(e)
            logger.error(f"작업 실패: {task.id}, 오류: {e}")
    
    async def _determine_strategy(self, task: Task) -> Dict[str, Any]:
        """작업 처리 전략 결정"""
        
        # 현재 시스템 상태 수집
        system_state = {
            "pending_tasks": self.task_queue.qsize() + len(self.priority_queue),
            "active_agents": sum(1 for a in self.agents.values() if not a.is_busy),
            "avg_processing_time": self._calculate_avg_processing_time(),
            "system_load": self._calculate_system_load()
        }
        
        # LLM을 통한 전략 결정
        try:
            chain = self.coordination_prompt | llm_manager.models["gpt-3.5-turbo"]
            
            response = await chain.ainvoke({
                **system_state,
                "task_type": task.type.value,
                "priority": task.priority.name,
                "task_data": json.dumps(task.data, ensure_ascii=False)
            })
            
            # 응답 파싱
            import json
            strategy = json.loads(response.content)
            
            return strategy
            
        except Exception as e:
            logger.error(f"전략 결정 실패: {e}")
            # 기본 전략
            return self._get_default_strategy(task)
    
    def _get_default_strategy(self, task: Task) -> Dict[str, Any]:
        """기본 처리 전략"""
        
        strategies = {
            TaskType.DETECTION: {
                "assign_to": ["detection_agent"],
                "processing_strategy": "sequential",
                "estimated_time": 3.0,
                "additional_resources": [],
                "reasoning": "기본 탐지 전략"
            },
            TaskType.ANALYSIS: {
                "assign_to": ["analysis_agent", "detection_agent"],
                "processing_strategy": "parallel",
                "estimated_time": 5.0,
                "additional_resources": [],
                "reasoning": "심층 분석을 위한 병렬 처리"
            },
            TaskType.RESPONSE: {
                "assign_to": ["response_agent"],
                "processing_strategy": "sequential",
                "estimated_time": 2.0,
                "additional_resources": [],
                "reasoning": "대응 전략 수립"
            }
        }
        
        return strategies.get(task.type, strategies[TaskType.DETECTION])
    
    async def _execute_parallel(self, task: Task, agent_ids: List[str]) -> List[Any]:
        """병렬 실행"""
        tasks = []
        
        for agent_id in agent_ids:
            if agent_id in self.agents:
                agent_task = self._execute_agent_task(agent_id, task)
                tasks.append(agent_task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 오류 필터링
        valid_results = [r for r in results if not isinstance(r, Exception)]
        
        return valid_results
    
    async def _execute_sequential(self, task: Task, agent_ids: List[str]) -> List[Any]:
        """순차 실행"""
        results = []
        
        for agent_id in agent_ids:
            if agent_id in self.agents:
                try:
                    result = await self._execute_agent_task(agent_id, task)
                    results.append(result)
                    
                    # 결과를 다음 에이전트에 전달
                    task.data["previous_result"] = result
                    
                except Exception as e:
                    logger.error(f"에이전트 {agent_id} 실행 실패: {e}")
                    break
        
        return results
    
    async def _execute_hybrid(self, task: Task, agent_ids: List[str]) -> List[Any]:
        """하이브리드 실행 (일부는 병렬, 일부는 순차)"""
        
        # 첫 번째 그룹은 병렬 실행
        if len(agent_ids) >= 2:
            parallel_agents = agent_ids[:2]
            sequential_agents = agent_ids[2:]
            
            # 병렬 실행
            parallel_results = await self._execute_parallel(task, parallel_agents)
            
            # 병렬 결과를 순차 실행에 전달
            task.data["parallel_results"] = parallel_results
            
            # 순차 실행
            sequential_results = await self._execute_sequential(task, sequential_agents)
            
            return parallel_results + sequential_results
        else:
            return await self._execute_sequential(task, agent_ids)
    
    async def _execute_agent_task(self, agent_id: str, task: Task) -> Any:
        """개별 에이전트 작업 실행"""
        
        agent_state = self.agents[agent_id]
        agent_instance = self.agent_map.get(task.type)

        if not agent_instance:
            raise ValueError(f"작업 유형에 맞는 에이전트를 찾을 수 없습니다: {task.type}")

        
        # 에이전트 상태 업데이트
        agent_state.is_busy = True
        agent_state.current_task = task.id
        
        start_time = datetime.now()
        
        try:
            # 주입된 에이전트 인스턴스의 작업 처리 메서드 호출
            result = await agent_instance.process_task(task.data)
            # 에이전트별 실행 로직
            if agent_id == "detection_agent":
                from . import detection_agent
                result = await detection_agent.process_task(task.data)
                
            elif agent_id == "analysis_agent":
                from . import analysis_agent
                result = await analysis_agent.process_task(task.data)
                
            elif agent_id == "response_agent":
                from . import response_agent
                result = await response_agent.process_task(task.data)
                
            else:
                raise ValueError(f"알 수 없는 에이전트: {agent_id}")
            
            # 성공 통계 업데이트
            agent_state.completed_tasks += 1
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # 평균 처리 시간 업데이트 (이동 평균)
            agent_state.avg_processing_time = (
                agent_state.avg_processing_time * 0.9 + processing_time * 0.1
            )
            
            return result
            
        except Exception as e:
            agent_state.error_count += 1
            logger.error(f"에이전트 {agent_id} 작업 실행 실패: {e}")
            raise
            
        finally:
            # 에이전트 상태 복원
            agent_state.is_busy = False
            agent_state.current_task = None
    
    def _integrate_results(self, results: List[Any]) -> Dict[str, Any]:
        """결과 통합"""
        
        if not results:
            return {"error": "결과 없음"}
        
        if len(results) == 1:
            return results[0]
        
        # 여러 결과 통합
        integrated = {
            "results": results,
            "consensus": self._find_consensus(results),
            "confidence": self._calculate_confidence(results),
            "timestamp": datetime.now().isoformat()
        }
        
        return integrated
    
    def _find_consensus(self, results: List[Any]) -> Any:
        """결과 간 합의 도출"""
        
        # 위험도 점수들의 평균
        risk_scores = []
        for result in results:
            if isinstance(result, dict) and "risk_score" in result:
                risk_scores.append(result["risk_score"])
        
        if risk_scores:
            return {"avg_risk_score": sum(risk_scores) / len(risk_scores)}
        
        return results[0] if results else None
    
    def _calculate_confidence(self, results: List[Any]) -> float:
        """결과 신뢰도 계산"""
        
        if len(results) <= 1:
            return 0.8
        
        # 결과 간 일치도 기반 신뢰도
        risk_scores = []
        for result in results:
            if isinstance(result, dict) and "risk_score" in result:
                risk_scores.append(result["risk_score"])
        
        if len(risk_scores) >= 2:
            # 표준편차가 낮을수록 신뢰도 높음
            import numpy as np
            std_dev = np.std(risk_scores)
            confidence = max(0.5, 1.0 - std_dev)
            return min(confidence, 1.0)
        
        return 0.8
    
    async def _monitor_agents(self):
        """에이전트 모니터링"""
        
        while self.is_running:
            try:
                # 30초마다 상태 확인
                await asyncio.sleep(30)
                
                # 에이전트 상태 로깅
                for agent_id, state in self.agents.items():
                    if state.error_count > 10:
                        logger.warning(
                            f"에이전트 {agent_id} 오류 다수 발생: {state.error_count}"
                        )
                
                # 성능 메트릭 수집
                metrics = self.get_performance_metrics()
                logger.info(f"시스템 메트릭: {metrics}")
                
            except Exception as e:
                logger.error(f"에이전트 모니터링 오류: {e}")
    
    async def _optimize_performance(self):
        """성능 최적화"""
        
        while self.is_running:
            try:
                # 5분마다 최적화
                await asyncio.sleep(300)
                
                # 부하 분산 최적화
                await self._rebalance_load()
                
                # 메모리 정리
                self._cleanup_old_tasks()
                
            except Exception as e:
                logger.error(f"성능 최적화 오류: {e}")
    
    async def _rebalance_load(self):
        """부하 재분산"""
        
        # 가장 바쁜 에이전트와 가장 한가한 에이전트 찾기
        busy_agents = sorted(
            self.agents.items(),
            key=lambda x: x[1].completed_tasks,
            reverse=True
        )
        
        if len(busy_agents) >= 2:
            busiest = busy_agents[0]
            idlest = busy_agents[-1]
            
            # 부하 차이가 크면 재분산 고려
            if busiest[1].completed_tasks > idlest[1].completed_tasks * 2:
                logger.info(
                    f"부하 재분산 필요: {busiest[0]}({busiest[1].completed_tasks}) "
                    f"-> {idlest[0]}({idlest[1].completed_tasks})"
                )
    
    def _cleanup_old_tasks(self):
        """오래된 작업 정리"""
        
        current_time = datetime.now()
        tasks_to_remove = []
        
        for task_id, task in self.task_history.items():
            # 1시간 이상 된 완료 작업 제거
            if task.status == "completed":
                age = (current_time - task.created_at).total_seconds()
                if age > 3600:
                    tasks_to_remove.append(task_id)
        
        for task_id in tasks_to_remove:
            del self.task_history[task_id]
        
        if tasks_to_remove:
            logger.info(f"{len(tasks_to_remove)}개의 오래된 작업 정리됨")
    
    def _calculate_avg_processing_time(self) -> float:
        """평균 처리 시간 계산"""
        
        times = [a.avg_processing_time for a in self.agents.values() if a.avg_processing_time > 0]
        return sum(times) / len(times) if times else 0.0
    
    def _calculate_system_load(self) -> float:
        """시스템 부하 계산"""
        
        total_agents = len(self.agents)
        busy_agents = sum(1 for a in self.agents.values() if a.is_busy)
        
        return busy_agents / total_agents if total_agents > 0 else 0.0
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """작업 상태 조회"""
        
        if task_id in self.task_history:
            task = self.task_history[task_id]
            return {
                "id": task.id,
                "type": task.type.value,
                "status": task.status,
                "priority": task.priority.name,
                "created_at": task.created_at.isoformat(),
                "result": task.result,
                "error": task.error
            }
        
        return None
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """성능 메트릭 조회"""
        
        return {
            "queue_size": self.task_queue.qsize(),
            "priority_queue_size": len(self.priority_queue),
            "total_tasks": len(self.task_history),
            "system_load": self._calculate_system_load(),
            "avg_processing_time": self._calculate_avg_processing_time(),
            "agent_stats": {
                agent_id: {
                    "is_busy": state.is_busy,
                    "completed_tasks": state.completed_tasks,
                    "error_count": state.error_count,
                    "avg_processing_time": state.avg_processing_time
                }
                for agent_id, state in self.agents.items()
            }
        }

# 전역 조정자 인스턴스는 __init__.py에서 생성됨