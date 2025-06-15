"""
VoiceGuard AI - 교육 컨텐츠 데이터
보이스피싱 예방 교육용 시나리오와 퀴즈
"""

# 교육 시나리오 데이터
EDUCATION_SCENARIOS = [
    {
        'id': 'bank_impersonation_1',
        'title': '🏦 은행 직원 사칭',
        'category': '기관사칭',
        'difficulty': 'beginner',
        'description': '은행 직원을 사칭하여 계좌 점검이 필요하다고 접근하는 상황',
        'conversation': [
            {
                'speaker': 'scammer',
                'message': '안녕하세요, OO은행 보안팀 김과장입니다. 고객님 계좌에 의심스러운 거래가 발견되어 연락드렸습니다.'
            },
            {
                'speaker': 'user',
                'options': [
                    {
                        'text': '어떤 거래인가요? 자세히 알려주세요.',
                        'correct': False,
                        'feedback': '개인정보를 더 노출시킬 수 있습니다. 먼저 상대방을 확인해야 합니다.'
                    },
                    {
                        'text': '확인해보겠습니다. 은행에 직접 전화드리겠습니다.',
                        'correct': True,
                        'feedback': '정확한 대응입니다! 항상 공식 채널로 확인하세요.'
                    },
                    {
                        'text': '계좌번호를 알려드릴게요.',
                        'correct': False,
                        'feedback': '절대 금지! 전화로 계좌정보를 요구하는 것은 사기입니다.'
                    }
                ]
            }
        ]
    },
    {
        'id': 'police_impersonation_1',
        'title': '👮 경찰서 사칭',
        'category': '기관사칭',
        'difficulty': 'intermediate',
        'description': '경찰관을 사칭하여 수사 협조를 요청하는 상황',
        'conversation': [
            {
                'speaker': 'scammer',
                'message': '안녕하세요, 서울지방경찰청 수사과입니다. 고객님과 관련된 사건이 접수되어 수사에 협조해주셔야 합니다.'
            },
            {
                'speaker': 'user',
                'options': [
                    {
                        'text': '무슨 사건인지 자세히 설명해주세요.',
                        'correct': False,
                        'feedback': '사기범에게 더 많은 정보를 제공할 기회를 줍니다.'
                    },
                    {
                        'text': '신분증을 확인하고 싶습니다. 직접 만날 수 있나요?',
                        'correct': False,
                        'feedback': '대면 만남은 위험할 수 있습니다. 공식 채널로 확인하세요.'
                    },
                    {
                        'text': '경찰서에 직접 방문하겠습니다. 주소 알려주세요.',
                        'correct': True,
                        'feedback': '훌륭합니다! 공식 기관 방문이 가장 안전한 방법입니다.'
                    }
                ]
            }
        ]
    },
    {
        'id': 'family_emergency_1',
        'title': '👨‍👩‍👧‍👦 가족 응급상황',
        'category': '납치협박',
        'difficulty': 'advanced',
        'description': '자녀가 사고를 당했다며 긴급히 돈을 요구하는 상황',
        'conversation': [
            {
                'speaker': 'scammer',
                'message': '어머니! 큰일났어요! 제가 교통사고를 내서 지금 병원에 있는데, 치료비가 급히 필요해요!'
            },
            {
                'speaker': 'user',
                'options': [
                    {
                        'text': '얼마나 필요해? 어느 병원이야?',
                        'correct': False,
                        'feedback': '서둘러 반응하면 사기에 당하기 쉽습니다.'
                    },
                    {
                        'text': '잠깐, 네 목소리가 좀 이상한데? 정말 너야?',
                        'correct': True,
                        'feedback': '훌륭한 판단력입니다! 의심하는 것이 중요합니다.'
                    },
                    {
                        'text': '알겠어, 지금 돈을 보낼게!',
                        'correct': False,
                        'feedback': '매우 위험합니다! 먼저 확인해야 합니다.'
                    }
                ]
            }
        ]
    },
    {
        'id': 'loan_scam_1',
        'title': '💰 저금리 대출 제안',
        'category': '대출사기',
        'difficulty': 'beginner',
        'description': '정부지원 저금리 대출을 미끼로 수수료를 요구하는 상황',
        'conversation': [
            {
                'speaker': 'scammer',
                'message': '안녕하세요! 정부지원 특별 저금리 대출 상품을 안내드리려고 연락했습니다. 연 1.5% 금리로 최대 3천만원까지 가능합니다!'
            },
            {
                'speaker': 'user',
                'options': [
                    {
                        'text': '정말요? 어떤 조건이 있나요?',
                        'correct': False,
                        'feedback': '관심을 보이면 더 정교한 사기에 빠질 수 있습니다.'
                    },
                    {
                        'text': '금융감독원에 등록된 업체인가요? 등록번호를 알려주세요.',
                        'correct': True,
                        'feedback': '완벽합니다! 정식 등록업체 확인은 필수입니다.'
                    },
                    {
                        'text': '선수수료는 얼마나 되나요?',
                        'correct': False,
                        'feedback': '정식 금융기관은 선수수료를 요구하지 않습니다!'
                    }
                ]
            }
        ]
    },
    {
        'id': 'app_installation_1',
        'title': '📱 악성 앱 설치 유도',
        'category': '악성앱',
        'difficulty': 'intermediate',
        'description': '대출 진행을 위해 앱 설치를 요구하는 상황',
        'conversation': [
            {
                'speaker': 'scammer',
                'message': '대출 승인을 위해서는 저희 전용 앱을 설치하셔야 합니다. 문자로 링크를 보내드릴게요.'
            },
            {
                'speaker': 'user',
                'options': [
                    {
                        'text': '앱스토어에서 직접 다운받겠습니다.',
                        'correct': True,
                        'feedback': '현명한 선택입니다! 항상 공식 스토어를 이용하세요.'
                    },
                    {
                        'text': '링크로 바로 설치하겠습니다.',
                        'correct': False,
                        'feedback': '매우 위험합니다! 출처불명 앱은 절대 설치하지 마세요.'
                    },
                    {
                        'text': '앱 이름이 뭔가요? 검색해볼게요.',
                        'correct': True,
                        'feedback': '좋은 접근입니다! 정식 앱인지 확인하는 것이 중요합니다.'
                    }
                ]
            }
        ]
    }
]

# 퀴즈 문제 데이터
QUIZ_QUESTIONS = [
    {
        'id': 'q001',
        'category': '기본 지식',
        'difficulty': 'easy',
        'question': '금융감독원에서 전화로 개인 계좌번호를 물어볼 수 있다?',
        'options': ['예, 가능하다', '아니오, 절대 하지 않는다', '경우에 따라 다르다'],
        'correct_answer': 1,
        'explanation': '금융감독원을 포함한 모든 공공기관은 전화로 개인 금융정보를 요구하지 않습니다.'
    },
    {
        'id': 'q002',
        'category': '대응 방법',
        'difficulty': 'easy',
        'question': '의심스러운 전화를 받았을 때 가장 먼저 해야 할 일은?',
        'options': ['상대방 말을 끝까지 듣기', '즉시 통화 끊기', '가족에게 물어보기'],
        'correct_answer': 1,
        'explanation': '의심스러운 순간 즉시 통화를 끊고, 공식 채널로 확인하는 것이 가장 안전합니다.'
    },
    {
        'id': 'q003',
        'category': '사기 수법',
        'difficulty': 'medium',
        'question': '대면편취형 보이스피싱의 특징이 아닌 것은?',
        'options': ['카페에서 만나자고 함', '현금을 직접 전달하라고 함', '전화로만 모든 것을 처리함'],
        'correct_answer': 2,
        'explanation': '대면편취형은 직접 만나서 현금이나 카드를 받아가는 것이 특징입니다.'
    },
    {
        'id': 'q004',
        'category': '법률 지식',
        'difficulty': 'medium',
        'question': '보이스피싱 피해를 당했을 때 신고할 곳이 아닌 것은?',
        'options': ['112 (경찰)', '1332 (금융감독원)', '1588-7080 (우체국)'],
        'correct_answer': 2,
        'explanation': '보이스피싱 신고는 112(경찰), 1332(금융감독원), 118(인터넷신고센터)에 할 수 있습니다.'
    },
    {
        'id': 'q005',
        'category': '기술적 보안',
        'difficulty': 'hard',
        'question': '악성 앱 설치를 방지하는 가장 효과적인 방법은?',
        'options': ['출처불명 앱 설치 금지', '안티바이러스 설치', '정기적인 폰 초기화'],
        'correct_answer': 0,
        'explanation': '공식 앱스토어 외의 출처에서 앱을 설치하지 않는 것이 가장 기본적이고 효과적인 방법입니다.'
    },
    {
        'id': 'q006',
        'category': '심리적 대응',
        'difficulty': 'hard',
        'question': '사기범이 사용하는 심리적 압박 기법이 아닌 것은?',
        'options': ['긴급성 조성', '권위 남용', '충분한 검토 시간 제공'],
        'correct_answer': 2,
        'explanation': '사기범은 피해자가 충분히 생각할 시간을 주지 않고, 긴급하게 결정하도록 압박합니다.'
    },
    {
        'id': 'q007',
        'category': '예방 수칙',
        'difficulty': 'easy',
        'question': '가족 납치 협박 전화를 받았을 때 올바른 대응은?',
        'options': ['즉시 돈을 준비한다', '가족에게 직접 연락해본다', '협박자와 협상한다'],
        'correct_answer': 1,
        'explanation': '가족에게 직접 연락하여 안전을 확인하는 것이 가장 중요합니다.'
    },
    {
        'id': 'q008',
        'category': '사기 수법',
        'difficulty': 'medium',
        'question': '정부지원금 사기의 일반적인 특징은?',
        'options': ['복잡한 서류 요구', '개인정보 요구', '오랜 대기 시간'],
        'correct_answer': 1,
        'explanation': '정부지원금 사기는 주민번호, 계좌번호 등 개인정보를 요구하는 것이 특징입니다.'
    },
    {
        'id': 'q009',
        'category': '기술적 보안',
        'difficulty': 'hard',
        'question': 'OTP(일회용 비밀번호)를 타인에게 알려주면 안 되는 이유는?',
        'options': ['법적 처벌을 받음', '계정 해킹이 가능함', '휴대폰이 고장남'],
        'correct_answer': 1,
        'explanation': 'OTP는 금융거래 인증용이므로 타인에게 알려주면 계정이 해킹당할 수 있습니다.'
    },
    {
        'id': 'q010',
        'category': '대응 방법',
        'difficulty': 'medium',
        'question': '보이스피싱 의심 전화를 받았을 때 녹음하는 것이 좋은 이유는?',
        'options': ['재미있어서', '증거 자료로 활용', '다른 사람에게 자랑하려고'],
        'correct_answer': 1,
        'explanation': '통화 녹음은 신고 시 중요한 증거 자료로 활용될 수 있습니다.'
    }
]

# 교육 진도에 따른 추천 컨텐츠
LEARNING_PATH = {
    'beginner': {
        'recommended_scenarios': ['bank_impersonation_1', 'loan_scam_1'],
        'required_quiz_score': 60,
        'next_level': 'intermediate'
    },
    'intermediate': {
        'recommended_scenarios': ['police_impersonation_1', 'app_installation_1'],
        'required_quiz_score': 75,
        'next_level': 'advanced'
    },
    'advanced': {
        'recommended_scenarios': ['family_emergency_1'],
        'required_quiz_score': 85,
        'next_level': 'expert'
    },
    'expert': {
        'recommended_scenarios': [],  # 모든 시나리오 완료
        'required_quiz_score': 90,
        'next_level': None
    }
}

# 성취 배지 시스템
ACHIEVEMENTS = {
    'first_lesson': {
        'name': '🎓 첫 학습 완료',
        'description': '첫 번째 학습을 완료했습니다!',
        'condition': lambda progress: progress['lessons_completed'] >= 1
    },
    'scenario_master': {
        'name': '🎭 시나리오 마스터',
        'description': '5개 이상의 시나리오를 완료했습니다!',
        'condition': lambda progress: progress['scenarios_practiced'] >= 5
    },
    'quiz_expert': {
        'name': '📝 퀴즈 전문가',
        'description': '퀴즈에서 90점 이상을 획득했습니다!',
        'condition': lambda progress: progress['quiz_score'] >= 90
    },
    'prevention_champion': {
        'name': '🏆 예방 챔피언',
        'description': '모든 교육 과정을 완료했습니다!',
        'condition': lambda progress: (
            progress['lessons_completed'] >= 3 and
            progress['scenarios_practiced'] >= 5 and
            progress['quiz_score'] >= 80
        )
    }
}