# test_llm.py (수정된 버전)
import asyncio
from core.llm_manager import llm_manager

async def main():
    print("LLM 매니저를 테스트합니다...")
    
    try:
        # 방법 1: health_check 메서드 사용 (권장)
        print("1. 모델 상태 확인 중...")
        health_status = await llm_manager.health_check()
        print(f"모델 상태: {health_status}")
        
        # 방법 2: 실제 보이스피싱 분석 테스트
        print("\n2. 보이스피싱 분석 테스트...")
        test_cases = [
            {
                "text": "안녕하세요, 정상적인 대화입니다.",
                "description": "정상 텍스트"
            },
            {
                "text": "금융감독원에서 전화드렸습니다. 계좌가 동결될 위험이 있으니 즉시 확인이 필요합니다.",
                "description": "의심스러운 텍스트"
            },
            {
                "text": "아들이 납치됐다! 빨리 돈을 보내지 않으면 위험하다!",
                "description": "고위험 텍스트"
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n테스트 {i}: {test_case['description']}")
            print(f"입력: {test_case['text']}")
            
            try:
                # analyze_scam_risk 메서드 사용
                result = await llm_manager.analyze_scam_risk(
                    text=test_case['text'],
                    context={"call_duration": 30, "caller_info": "unknown"}
                )
                
                print(f"결과:")
                print(f"  - 위험도: {result.risk_level.value}")
                print(f"  - 신뢰도: {result.confidence:.2f}")
                print(f"  - 사용 모델: {result.model_used}")
                print(f"  - 처리 시간: {result.processing_time:.2f}초")
                print(f"  - 사기 유형: {result.metadata.get('scam_type', 'N/A')}")
                print(f"  - 즉시 조치: {result.metadata.get('immediate_action', False)}")
                
            except Exception as e:
                print(f"  분석 실패: {e}")
        
        # 방법 3: 성능 통계 확인
        print("\n3. 성능 통계:")
        stats = llm_manager.get_performance_stats()
        print(f"총 호출 수: {stats['total_calls']}")
        print(f"총 비용: ${stats['total_cost']:.4f}")
        print(f"평균 비용: ${stats['avg_cost_per_call']:.4f}")
        print(f"남은 예산: ${stats['remaining_budget']:.2f}")
        
        print("\nLLM 연결 테스트 성공!")
        
    except Exception as e:
        print(f"LLM 연결 테스트 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # .env 로드
    from dotenv import load_dotenv
    load_dotenv()
    
    # 비동기 실행
    asyncio.run(main())