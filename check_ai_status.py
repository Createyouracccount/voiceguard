#!/usr/bin/env python3
"""
VoiceGuard AI 시스템 상태 확인 도구
현재 AI가 제대로 활용되고 있는지 확인
"""

import asyncio
import sys
import os
from pathlib import Path

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

async def check_ai_status():
    """AI 시스템 상태 종합 확인"""
    
    print("🤖 VoiceGuard AI 시스템 상태 확인")
    print("=" * 60)
    
    status_report = {
        "환경설정": False,
        "LLM_Manager": False,
        "Agents": False,
        "LangChain": False,
        "실제_AI_사용": False
    }
    
    # 1. 환경설정 확인
    print("\n📋 1. 환경설정 확인")
    try:
        google_key = os.getenv('GOOGLE_API_KEY')
        if google_key and len(google_key) > 20:
            print("   ✅ GOOGLE_API_KEY 설정됨")
            status_report["환경설정"] = True
        else:
            print("   ❌ GOOGLE_API_KEY 누락 또는 잘못됨")
    except Exception as e:
        print(f"   ❌ 환경설정 오류: {e}")
    
    # 2. LLM Manager 확인
    print("\n🧠 2. LLM Manager 상태")
    try:
        from core.llm_manager import llm_manager
        
        # 모델 목록
        models = llm_manager.get_available_models()
        print(f"   📋 사용 가능한 모델: {models}")
        
        # 연결 테스트
        health = await llm_manager.health_check()
        healthy_models = [m for m, status in health.items() if status]
        
        if healthy_models:
            print(f"   ✅ 연결된 모델: {healthy_models}")
            status_report["LLM_Manager"] = True
        else:
            print("   ❌ 연결된 모델 없음")
            
        # 성능 통계
        stats = llm_manager.get_performance_stats()
        print(f"   📊 총 호출: {stats.get('total_calls', 0)}")
        print(f"   💰 총 비용: ${stats.get('total_cost', 0):.4f}")
        
    except Exception as e:
        print(f"   ❌ LLM Manager 오류: {e}")
    
    # 3. Agents 확인
    print("\n🤖 3. Agent 시스템 상태")
    try:
        from agents.detection_agent import DetectionAgent
        from agents.analysis_agent import AnalysisAgent
        from agents.response_agent import ResponseAgent
        
        detection_agent = DetectionAgent()
        print("   ✅ DetectionAgent 로드됨")
        
        analysis_agent = AnalysisAgent()
        print("   ✅ AnalysisAgent 로드됨")
        
        response_agent = ResponseAgent()
        print("   ✅ ResponseAgent 로드됨")
        
        status_report["Agents"] = True
        
    except Exception as e:
        print(f"   ❌ Agents 로드 실패: {e}")
    
    # 4. LangChain 확인
    print("\n🔗 4. LangChain 워크플로우 상태")
    try:
        from langchain_workflows.detection_chain import DetectionChain
        
        detection_chain = DetectionChain()
        print("   ✅ DetectionChain 로드됨")
        
        # 성능 통계
        chain_stats = detection_chain.get_performance_stats()
        print(f"   📊 체인 상태: {chain_stats}")
        
        status_report["LangChain"] = True
        
    except Exception as e:
        print(f"   ❌ LangChain 로드 실패: {e}")
    
    # 5. 실제 AI 테스트
    print("\n🧪 5. 실제 AI 분석 테스트")
    try:
        if status_report["LLM_Manager"]:
            test_text = "안녕하세요, 금융감독원입니다. 계좌 점검이 필요합니다."
            
            print(f"   📝 테스트 입력: {test_text}")
            print("   ⚡ AI 분석 중...")
            
            start_time = asyncio.get_event_loop().time()
            result = await llm_manager.analyze_scam_risk(text=test_text)
            end_time = asyncio.get_event_loop().time()
            
            print(f"   ✅ AI 분석 완료 ({end_time - start_time:.2f}초)")
            print(f"   🎯 위험도: {result.metadata.get('risk_score', 0):.1%}")
            print(f"   🧠 사용된 모델: {result.model_used}")
            print(f"   💰 비용: ${result.cost_estimate:.4f}")
            
            status_report["실제_AI_사용"] = True
        else:
            print("   ⏭️ LLM Manager 문제로 건너뜀")
            
    except Exception as e:
        print(f"   ❌ AI 테스트 실패: {e}")
    
    # 6. 현재 사용 패턴 분석
    print("\n📈 6. 현재 시스템 사용 패턴 분석")
    try:
        from services.conversation_manager_backup import ConversationManager
        
        # 실제 사용되는 클래스 확인
        print("   📋 ConversationManager 분석:")
        
        # 소스 코드에서 AI 사용 패턴 확인
        import inspect
        source = inspect.getsource(ConversationManager.__init__)
        
        if "coordinator_agent" in source:
            print("   ✅ CoordinatorAgent 활용됨")
        else:
            print("   ❌ CoordinatorAgent 활용 안됨")
            
        if "detection_chain" in source:
            print("   ✅ LangChain 활용됨")
        else:
            print("   ❌ LangChain 활용 안됨")
            
        if "llm_manager.analyze_scam_risk" in source:
            print("   ✅ 직접 LLM 호출 사용됨")
        else:
            print("   ❌ 직접 LLM 호출도 안됨")
            
    except Exception as e:
        print(f"   ❌ 사용 패턴 분석 실패: {e}")
    
    # 7. 최종 진단 및 권장사항
    print("\n🏥 7. 최종 진단")
    print("=" * 40)
    
    total_score = sum(status_report.values())
    max_score = len(status_report)
    
    print(f"전체 상태: {total_score}/{max_score} ({total_score/max_score*100:.1f}%)")
    
    for component, status in status_report.items():
        icon = "✅" if status else "❌"
        print(f"   {icon} {component}")
    
    print("\n💡 권장사항:")
    
    if not status_report["환경설정"]:
        print("   🔧 .env 파일에 GOOGLE_API_KEY 설정 필요")
    
    if not status_report["실제_AI_사용"]:
        print("   🤖 AI 시스템이 실제로 작동하지 않음")
        print("   📋 Option 1: Enhanced System으로 교체 (위 artifact)")
        print("   📋 Option 2: Simplified System으로 단순화")
    
    if status_report["Agents"] and status_report["LangChain"] and not status_report["실제_AI_사용"]:
        print("   ⚠️ 고급 AI 시스템이 구현되어 있지만 활용되지 않음")
        print("   💡 Enhanced Conversation Manager 적용 권장")
    
    if total_score < 3:
        print("   🚨 시스템 전면 재구성 필요")
    elif total_score < 4:
        print("   ⚠️ 일부 기능 수정 필요")
    else:
        print("   ✅ 시스템 상태 양호")

if __name__ == "__main__":
    asyncio.run(check_ai_status())