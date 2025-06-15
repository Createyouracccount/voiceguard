"""
VoiceGuard AI - 예방 교육 모드
보이스피싱 수법 학습 및 대응 훈련
"""

import asyncio
import logging
import random
from typing import Dict, Any, List, Optional
from datetime import datetime

from .base_mode import BaseMode, ModeState
# Import 경로 수정 - 상대경로로 변경
try:
    from config.data.education_content import EDUCATION_SCENARIOS, QUIZ_QUESTIONS
except ImportError:
    # 임시로 기본 데이터 사용
    EDUCATION_SCENARIOS = []
    QUIZ_QUESTIONS = []

try:
    from config.prompts.prevention_prompts import PREVENTION_PROMPTS
except ImportError:
    # 임시로 기본 프롬프트 사용
    PREVENTION_PROMPTS = {}

logger = logging.getLogger(__name__)

class PreventionMode(BaseMode):
    """예방 교육 모드"""
    
    @property
    def mode_name(self) -> str:
        return "예방 교육"
    
    @property
    def mode_description(self) -> str:
        return "보이스피싱 수법을 학습하고 대응 방법을 훈련합니다"
    
    def _load_mode_config(self) -> Dict[str, Any]:
        """예방 교육 모드 설정"""
        return {
            'interactive_mode': True,
            'voice_feedback': True,
            'quiz_enabled': True,
            'scenario_count': 5,
            'difficulty_level': 'beginner'
        }
    
    async def _initialize_mode(self) -> bool:
        """예방 교육 모드 초기화"""
        
        try:
            # 교육 진행 상태
            self.current_lesson = 0
            self.quiz_score = 0
            self.completed_scenarios = []
            
            # 사용자 진행 기록
            self.user_progress = {
                'lessons_completed': 0,
                'quiz_score': 0,
                'scenarios_practiced': 0,
                'knowledge_level': 'beginner'
            }
            
            logger.info("✅ 예방 교육 모드 초기화 완료")
            return True
            
        except Exception as e:
            logger.error(f"예방 교육 모드 초기화 실패: {e}")
            return False
    
    async def _run_mode_logic(self):
        """예방 교육 메인 로직"""
        
        await self._show_education_menu()
        
        while self.is_running:
            try:
                choice = await self._get_user_choice()
                
                if choice == 'learn':
                    await self._run_learning_session()
                elif choice == 'practice':
                    await self._run_practice_session()
                elif choice == 'quiz':
                    await self._run_quiz_session()
                elif choice == 'summary':
                    await self._show_progress_summary()
                elif choice == 'exit':
                    break
                else:
                    print("❌ 잘못된 선택입니다. 다시 선택해주세요.")
                
                # 메뉴로 돌아가기
                if self.is_running:
                    await self._show_education_menu()
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"교육 세션 오류: {e}")
                print(f"❌ 오류가 발생했습니다: {e}")
    
    async def _show_education_menu(self):
        """교육 메뉴 표시"""
        
        print(f"\n🎓 {self.mode_name} 메뉴")
        print("=" * 40)
        print("1. 📚 학습하기 - 보이스피싱 수법 알아보기")
        print("2. 🎭 연습하기 - 실제 시나리오로 대응 훈련")
        print("3. 📝 퀴즈풀기 - 학습 내용 점검")
        print("4. 📊 진행현황 - 학습 진도 확인")
        print("5. 🚪 종료하기")
        print("=" * 40)
    
    async def _get_user_choice(self) -> str:
        """사용자 선택 입력"""
        
        choice_map = {
            '1': 'learn',
            '2': 'practice', 
            '3': 'quiz',
            '4': 'summary',
            '5': 'exit'
        }
        
        while True:
            try:
                print("\n선택하세요 (1-5): ", end="")
                user_input = input().strip()
                
                if user_input in choice_map:
                    choice = choice_map[user_input]
                    print(f"✅ '{user_input}' 선택됨")
                    return choice
                else:
                    print("❌ 1-5 사이의 숫자를 입력해주세요.")
                    
            except (EOFError, KeyboardInterrupt):
                return 'exit'
    
    async def _run_learning_session(self):
        """학습 세션 실행"""
        
        print(f"\n📚 보이스피싱 학습 세션 시작")
        await self._speak("보이스피싱 학습을 시작합니다.")
        
        # 학습 주제들
        topics = [
            {
                'title': '🏦 기관사칭형',
                'content': '''
금융감독원, 검찰청, 경찰서 등을 사칭하여 신뢰를 얻는 수법입니다.

📋 주요 특징:
• "계좌가 범죄에 연루되었다"
• "수사에 협조해달라"  
• "안전계좌로 이체하라"
• 공식 기관은 전화로 개인정보를 요구하지 않습니다!

🛡️ 대응법:
1. 즉시 통화를 끊으세요
2. 해당 기관에 직접 전화하여 확인
3. 절대 개인정보나 금융정보를 제공하지 마세요
                '''
            },
            {
                'title': '👨‍👩‍👧‍👦 납치협박형', 
                'content': '''
가족이 납치되었다고 거짓말하여 돈을 요구하는 수법입니다.

📋 주요 특징:
• "아들/딸이 사고났다"
• "응급실에 있다"
• "즉시 돈을 보내달라"
• 시간적 압박을 가합니다

🛡️ 대응법:
1. 침착하게 행동하세요
2. 가족에게 직접 연락하여 확인
3. 112에 신고하세요
4. 절대 먼저 돈을 보내지 마세요
                '''
            },
            {
                'title': '💰 대출사기형',
                'content': '''
저금리 대출을 미끼로 수수료를 편취하는 수법입니다.

📋 주요 특징:
• "저금리 특별대출"
• "정부지원 대출"
• "선수수료 필요"
• "앱 설치 필요"

🛡️ 대응법:
1. 정식 금융기관은 선수수료를 요구하지 않습니다
2. 금융감독원 홈페이지에서 등록업체 확인
3. 의심스러운 앱 설치 금지
4. 1332(금융감독원)에 문의
                '''
            }
        ]
        
        # 순차적으로 학습 진행
        for i, topic in enumerate(topics, 1):
            print(f"\n--- {i}/{len(topics)} {topic['title']} ---")
            print(topic['content'])
            
            # 음성 설명
            summary = f"{topic['title']} 수법에 대해 설명드렸습니다."
            await self._speak(summary)
            
            # 다음으로 넘어가기
            if i < len(topics):
                input("\n계속하려면 Enter를 누르세요...")
        
        # 학습 완료
        self.user_progress['lessons_completed'] += 1
        self._update_stats(success=True, lessons_completed=self.user_progress['lessons_completed'])
        
        print(f"\n🎉 학습 세션 완료!")
        await self._speak("학습이 완료되었습니다. 수고하셨습니다!")
    
    async def _run_practice_session(self):
        """연습 세션 실행"""
        
        print(f"\n🎭 보이스피싱 대응 훈련 시작")
        await self._speak("실제 상황을 가정한 대응 훈련을 시작합니다.")
        
        # 시나리오 선택
        available_scenarios = [s for s in EDUCATION_SCENARIOS 
                             if s['id'] not in self.completed_scenarios]
        
        if not available_scenarios:
            print("🎊 모든 시나리오를 완료하셨습니다!")
            return
        
        scenario = random.choice(available_scenarios)
        
        print(f"\n📞 시나리오: {scenario['title']}")
        print(f"상황: {scenario['description']}")
        print("-" * 50)
        
        # 시나리오 대화 시뮬레이션
        for step in scenario['conversation']:
            if step['speaker'] == 'scammer':
                print(f"\n🎭 사기범: {step['message']}")
                await self._speak(f"사기범 역할: {step['message']}")
            else:
                print(f"\n💭 어떻게 대응하시겠습니까?")
                
                # 선택지 제공
                for i, option in enumerate(step['options'], 1):
                    print(f"{i}. {option['text']}")
                
                # 사용자 선택
                while True:
                    try:
                        choice = int(input("\n선택 (번호): ")) - 1
                        if 0 <= choice < len(step['options']):
                            selected = step['options'][choice]
                            print(f"\n✅ 선택: {selected['text']}")
                            
                            # 피드백
                            if selected['correct']:
                                print(f"🎉 정답! {selected['feedback']}")
                                await self._speak("훌륭한 대응입니다!")
                            else:
                                print(f"❌ 아쉽네요. {selected['feedback']}")
                                await self._speak("다시 생각해보세요.")
                            
                            break
                        else:
                            print("❌ 올바른 번호를 선택해주세요.")
                    except ValueError:
                        print("❌ 숫자를 입력해주세요.")
        
        # 시나리오 완료
        self.completed_scenarios.append(scenario['id'])
        self.user_progress['scenarios_practiced'] += 1
        self._update_stats(success=True, scenarios_practiced=self.user_progress['scenarios_practiced'])
        
        print(f"\n🏆 시나리오 완료!")
        await self._speak("훈련이 완료되었습니다.")
    
    async def _run_quiz_session(self):
        """퀴즈 세션 실행"""
        
        print(f"\n📝 보이스피싱 지식 퀴즈")
        await self._speak("보이스피싱 지식을 확인하는 퀴즈를 시작합니다.")
        
        # 퀴즈 문제 선택 (5문제)
        selected_questions = random.sample(QUIZ_QUESTIONS, min(5, len(QUIZ_QUESTIONS)))
        correct_answers = 0
        
        for i, question in enumerate(selected_questions, 1):
            print(f"\n--- 문제 {i}/{len(selected_questions)} ---")
            print(question['question'])
            
            # 선택지 출력
            for j, option in enumerate(question['options'], 1):
                print(f"{j}. {option}")
            
            # 답변 입력
            while True:
                try:
                    answer = int(input("\n정답 번호: ")) - 1
                    if 0 <= answer < len(question['options']):
                        if answer == question['correct_answer']:
                            print("🎉 정답!")
                            await self._speak("정답입니다!")
                            correct_answers += 1
                        else:
                            correct_option = question['options'][question['correct_answer']]
                            print(f"❌ 틀렸습니다. 정답: {correct_option}")
                            await self._speak("틀렸습니다.")
                        
                        # 해설
                        if question.get('explanation'):
                            print(f"💡 해설: {question['explanation']}")
                        
                        break
                    else:
                        print("❌ 올바른 번호를 선택해주세요.")
                except ValueError:
                    print("❌ 숫자를 입력해주세요.")
        
        # 퀴즈 결과
        score = (correct_answers / len(selected_questions)) * 100
        self.user_progress['quiz_score'] = max(self.user_progress['quiz_score'], score)
        
        print(f"\n📊 퀴즈 결과:")
        print(f"정답: {correct_answers}/{len(selected_questions)} ({score:.0f}점)")
        
        if score >= 80:
            message = "우수합니다! 보이스피싱에 대한 이해도가 높네요."
            await self._speak("우수한 결과입니다!")
        elif score >= 60:
            message = "양호합니다. 조금 더 학습하시면 완벽해질 것 같아요."
            await self._speak("양호한 결과입니다.")
        else:
            message = "더 많은 학습이 필요합니다. 학습 메뉴를 다시 이용해보세요."
            await self._speak("추가 학습이 필요합니다.")
        
        print(f"💬 {message}")
        
        self._update_stats(success=True, quiz_score=score)
    
    async def _show_progress_summary(self):
        """진행 현황 요약"""
        
        print(f"\n📊 {self.user_progress['knowledge_level'].upper()} 학습자 진행 현황")
        print("=" * 40)
        print(f"📚 완료한 학습: {self.user_progress['lessons_completed']}회")
        print(f"🎭 연습한 시나리오: {self.user_progress['scenarios_practiced']}개") 
        print(f"📝 최고 퀴즈 점수: {self.user_progress['quiz_score']:.0f}점")
        print(f"🕒 총 학습 시간: {self._get_study_time()}")
        
        # 추천 사항
        recommendations = self._get_recommendations()
        if recommendations:
            print(f"\n💡 추천 사항:")
            for rec in recommendations:
                print(f"   • {rec}")
        
        # 음성 요약
        summary_text = f"현재까지 {self.user_progress['lessons_completed']}회 학습하고 {self.user_progress['scenarios_practiced']}개 시나리오를 연습하셨습니다."
        await self._speak(summary_text)
    
    def _get_study_time(self) -> str:
        """학습 시간 계산"""
        if self.start_time:
            delta = datetime.now() - self.start_time
            minutes = int(delta.total_seconds() / 60)
            return f"{minutes}분"
        return "0분"
    
    def _get_recommendations(self) -> List[str]:
        """개인화된 추천 사항"""
        recommendations = []
        
        if self.user_progress['lessons_completed'] == 0:
            recommendations.append("먼저 '학습하기'로 기본 지식을 쌓아보세요")
        
        if self.user_progress['scenarios_practiced'] < 3:
            recommendations.append("실전 대응력 향상을 위해 '연습하기'를 더 해보세요")
        
        if self.user_progress['quiz_score'] < 80:
            recommendations.append("'퀴즈풀기'로 학습 내용을 점검해보세요")
        
        if len(self.completed_scenarios) >= 3 and self.user_progress['quiz_score'] >= 80:
            recommendations.append("훌륭합니다! 가족들에게도 보이스피싱 예방법을 알려주세요")
        
        return recommendations
    
    async def _cleanup_mode(self):
        """예방 교육 모드 정리"""
        
        try:
            # 학습 진행 상황 저장 (향후 데이터베이스 연동 시 사용)
            final_progress = {
                'session_id': self.session_id,
                'completion_time': datetime.now(),
                'final_stats': self.user_progress.copy()
            }
            
            logger.info(f"예방 교육 세션 완료: {final_progress}")
            
        except Exception as e:
            logger.error(f"예방 교육 모드 정리 오류: {e}")
    
    def get_learning_progress(self) -> Dict[str, Any]:
        """학습 진도 조회"""
        
        total_scenarios = len(EDUCATION_SCENARIOS)
        total_questions = len(QUIZ_QUESTIONS)
        
        return {
            "user_progress": self.user_progress.copy(),
            "completion_rates": {
                "scenarios": len(self.completed_scenarios) / total_scenarios if total_scenarios > 0 else 0,
                "lessons": min(self.user_progress['lessons_completed'] / 3, 1.0),  # 3개 주요 레슨
                "quiz": self.user_progress['quiz_score'] / 100
            },
            "study_time_minutes": int((datetime.now() - self.start_time).total_seconds() / 60) if self.start_time else 0,
            "recommendations": self._get_recommendations()
        }