"""
VoiceGuard AI - Multi-Agent System
전문화된 AI 에이전트들이 협업하여 보이스피싱 탐지 및 대응
"""
from .detection_agent import DetectionAgent
from .analysis_agent import AnalysisAgent
from .response_agent import ResponseAgent
from .coordinator_agent import CoordinatorAgent

# 전역 에이전트 인스턴스 생성
# 각 에이전트는 시스템 전체에서 하나의 인스턴스만 유지 (싱글턴 패턴)
detection_agent = DetectionAgent()
analysis_agent = AnalysisAgent()
response_agent = ResponseAgent()
# Coordinator는 다른 에이전트들을 참조해야 하므로 마지막에 초기화
coordinator_agent = CoordinatorAgent(detection_agent, analysis_agent, response_agent)

__all__ = [
    'DetectionAgent',
    'AnalysisAgent',
    'ResponseAgent',
    'CoordinatorAgent',
    'detection_agent',
    'analysis_agent',
    'response_agent',
    'coordinator_agent'
]