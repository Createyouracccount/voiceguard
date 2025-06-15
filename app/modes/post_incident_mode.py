"""
VoiceGuard AI - 사후 대처 모드 (완전 구현)
금융감독원 공식 보이스피싱 피해 대처방법 기반
"""

import asyncio
import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from enum import Enum

from .base_mode import BaseMode, ModeState

logger = logging.getLogger(__name__)

class DamageType(Enum):
    """피해 유형"""
    FINANCIAL_TRANSFER = "금전_이체"
    PERSONAL_INFO_LEAK = "개인정보_유출"
    MALICIOUS_APP = "악성앱_설치"
    ACCOUNT_OPENING = "계좌_개설"
    PHONE_OPENING = "휴대폰_개통"
    CARD_MISUSE = "카드_오남용"

class RecoveryStage(Enum):
    """회복 단계"""
    DAMAGE_ASSESSMENT = "피해_평가"
    IMMEDIATE_RESPONSE = "즉시_대응"
    EVIDENCE_COLLECTION = "증거_수집"
    PERSONAL_INFO_PROTECTION = "개인정보_보호"
    LEGAL_PROCEDURES = "법적_절차"
    FINANCIAL_RECOVERY = "피해금_환급"
    PREVENTION_SETUP = "재발_방지"

class PostIncidentMode(BaseMode):
    """사후 대처 모드 - 금융감독원 기준 완전 구현"""
    
    @property
    def mode_name(self) -> str:
        return "사후 대처"
    
    @property
    def mode_description(self) -> str:
        return "보이스피싱 피해 발생 후 금융감독원 기준 체계적 대응 및 회복을 지원합니다"
    
    def _load_mode_config(self) -> Dict[str, Any]:
        """사후 대처 모드 설정"""
        return {
            'emergency_mode': True,
            'step_by_step_guide': True,
            'progress_tracking': True,
            'legal_compliance': True,
            'official_procedures': True
        }
    
    async def _initialize_mode(self) -> bool:
        """사후 대처 모드 초기화"""
        
        try:
            # 피해 상황 데이터
            self.incident_data = {
                'timestamp': datetime.now(),
                'damage_types': [],
                'financial_loss': 0,
                'transferred_accounts': [],
                'evidence_collected': [],
                'current_stage': RecoveryStage.DAMAGE_ASSESSMENT,
                'completed_steps': set()
            }
            
            # 금융감독원 공식 체크리스트
            self.official_checklist = {
                'immediate_actions': {
                    'police_report_112': {
                        'name': '경찰청(112) 신고',
                        'description': '피해 사실 즉시 신고',
                        'completed': False,
                        'required': True
                    },
                    'fss_report_1332': {
                        'name': '금융감독원(1332) 신고',
                        'description': '금융 피해 신고',
                        'completed': False,
                        'required': True
                    },
                    'bank_contact': {
                        'name': '송금/입금 은행 고객센터 연락',
                        'description': '해당 금융회사에 지급정지 신청',
                        'completed': False,
                        'required': True
                    }
                },
                'personal_info_protection': {
                    'cert_reset': {
                        'name': '공동인증서 초기화/재발급',
                        'description': '기존 인증서 삭제 후 재발급',
                        'completed': False,
                        'required': True
                    },
                    'malware_removal': {
                        'name': '악성앱 삭제/단말기 초기화',
                        'description': '통신사 고객센터 방문 또는 초기화',
                        'completed': False,
                        'required': True
                    },
                    'personal_info_registration': {
                        'name': '개인정보 노출사실 등록',
                        'description': 'pd.fss.or.kr에서 등록',
                        'completed': False,
                        'required': True
                    }
                },
                'verification_steps': {
                    'account_check': {
                        'name': '계좌 개설 여부 조회',
                        'description': 'www.payinfo.or.kr에서 확인',
                        'completed': False,
                        'required': True
                    },
                    'phone_check': {
                        'name': '휴대폰 개설 여부 조회',
                        'description': 'www.msafer.or.kr에서 확인',
                        'completed': False,
                        'required': True
                    }
                },
                'legal_procedures': {
                    'incident_report': {
                        'name': '사건사고사실확인원 발급',
                        'description': '경찰서 또는 사이버수사대 방문',
                        'completed': False,
                        'required': True
                    },
                    'damage_claim': {
                        'name': '피해금 환급 신청',
                        'description': '금융회사 영업점 제출 (3일 이내)',
                        'completed': False,
                        'required': True
                    }
                }
            }
            
            # 공식 웹사이트 링크
            self.official_links = {
                '개인정보노출자_사고예방시스템': 'https://pd.fss.or.kr',
                '계좌정보통합관리서비스': 'https://www.payinfo.or.kr',
                '명의도용방지서비스': 'https://www.msafer.or.kr',
                '금융감독원_보이스피싱지킴이': 'https://www.fss.or.kr'
            }
            
            logger.info("✅ 사후대처 모드 초기화 완료")
            return True
            
        except Exception as e:
            logger.error(f"사후대처 모드 초기화 실패: {e}")
            return False
    
    async def _run_mode_logic(self):
        """사후 대처 메인 로직"""
        
        print("🚨 보이스피싱 피해 사후대처 시스템")
        print("📋 금융감독원 공식 절차에 따라 단계별로 안내해드립니다")
        print("=" * 60)
        
        # 1. 피해 상황 평가
        await self._assess_damage()
        
        # 2. 단계별 대응 진행
        while self._has_remaining_steps():
            await self._show_current_stage()
            await self._execute_current_stage()
            
            if not await self._ask_continue():
                break
        
        # 3. 최종 요약 및 추가 안내
        await self._show_final_summary()
    
    async def _assess_damage(self):
        """피해 상황 평가"""
        
        print("\n🔍 STEP 1: 피해 상황 평가")
        print("-" * 30)
        
        # 피해 유형 확인
        print("어떤 유형의 피해를 당하셨나요? (여러 개 선택 가능)")
        print("1. 💰 돈을 송금/이체했다")
        print("2. 📱 개인정보(신분증, 계좌번호 등)를 알려줬다")
        print("3. 📲 의심스러운 링크를 클릭하거나 앱을 설치했다")
        print("4. 💳 카드 정보를 제공했다")
        print("5. 📞 기타 피해")
        
        while True:
            try:
                damage_input = input("\n피해 유형 번호를 입력하세요 (예: 1,2,3): ").strip()
                if damage_input:
                    damage_numbers = [int(x.strip()) for x in damage_input.split(',') if x.strip().isdigit()]
                    if damage_numbers and all(1 <= num <= 5 for num in damage_numbers):
                        break
                print("❌ 올바른 번호를 입력해주세요 (1-5)")
            except ValueError:
                print("❌ 숫자를 입력해주세요")
        
        # 피해 유형 매핑
        damage_mapping = {
            1: DamageType.FINANCIAL_TRANSFER,
            2: DamageType.PERSONAL_INFO_LEAK,
            3: DamageType.MALICIOUS_APP,
            4: DamageType.CARD_MISUSE,
            5: DamageType.PERSONAL_INFO_LEAK  # 기타는 개인정보 유출로 분류
        }
        
        for num in damage_numbers:
            self.incident_data['damage_types'].append(damage_mapping[num])
        
        # 금전 피해 확인
        if DamageType.FINANCIAL_TRANSFER in self.incident_data['damage_types']:
            while True:
                try:
                    amount = input("\n💰 송금한 금액을 입력하세요 (원): ").strip()
                    if amount.isdigit():
                        self.incident_data['financial_loss'] = int(amount)
                        break
                    print("❌ 숫자만 입력해주세요")
                except ValueError:
                    print("❌ 올바른 금액을 입력해주세요")
        
        # 피해 시간 확인
        print("\n⏰ 피해 발생 시간:")
        print("1. 방금 전 (1시간 이내)")
        print("2. 오늘 (24시간 이내)")
        print("3. 어제 (48시간 이내)")
        print("4. 3일 이상 전")
        
        time_choice = input("시간을 선택하세요 (1-4): ").strip()
        
        if time_choice == "1":
            self.incident_data['urgency'] = 'CRITICAL'
        elif time_choice == "2":
            self.incident_data['urgency'] = 'HIGH'
        elif time_choice == "3":
            self.incident_data['urgency'] = 'MEDIUM'
        else:
            self.incident_data['urgency'] = 'LOW'
        
        print(f"\n✅ 피해 상황 평가 완료")
        print(f"   피해 유형: {len(self.incident_data['damage_types'])}개")
        if self.incident_data['financial_loss'] > 0:
            print(f"   금전 피해: {self.incident_data['financial_loss']:,}원")
        print(f"   긴급도: {self.incident_data['urgency']}")
    
    async def _show_current_stage(self):
        """현재 단계 표시"""
        
        stage = self.incident_data['current_stage']
        
        if stage == RecoveryStage.IMMEDIATE_RESPONSE:
            print("\n🚨 STEP 2: 즉시 대응 조치")
            print("=" * 40)
            print("⚠️ 우선 다음 연락을 즉시 하셔야 합니다:")
            self._show_checklist_category('immediate_actions')
            
        elif stage == RecoveryStage.PERSONAL_INFO_PROTECTION:
            print("\n🛡️ STEP 3: 개인정보 보호 조치")
            print("=" * 40)
            print("📱 개인정보가 유출된 경우 다음 조치를 취하세요:")
            self._show_checklist_category('personal_info_protection')
            
        elif stage == RecoveryStage.EVIDENCE_COLLECTION:
            print("\n🔍 STEP 4: 명의도용 확인")
            print("=" * 40)
            print("🔐 본인 명의로 개설된 계좌/휴대폰을 확인하세요:")
            self._show_checklist_category('verification_steps')
            
        elif stage == RecoveryStage.LEGAL_PROCEDURES:
            print("\n⚖️ STEP 5: 법적 절차 및 피해금 환급")
            print("=" * 40)
            print("📄 공식 서류 발급 및 환급 신청:")
            self._show_checklist_category('legal_procedures')
    
    def _show_checklist_category(self, category: str):
        """체크리스트 카테고리 표시"""
        
        items = self.official_checklist[category]
        
        for i, (key, item) in enumerate(items.items(), 1):
            status = "✅" if item['completed'] else "☐"
            required = "⭐" if item['required'] else ""
            
            print(f"\n{i}. {status} {required} {item['name']}")
            print(f"   📝 {item['description']}")
            
            # 상세 안내 추가
            if not item['completed']:
                self._show_detailed_guidance(key)
    
    def _show_detailed_guidance(self, action_key: str):
        """상세 안내 표시"""
        
        guidance = {
            'police_report_112': """
   📞 전화: 112
   💬 말할 내용: "보이스피싱 피해를 당했습니다. 지급정지 신청이 필요합니다."
   📋 준비사항: 사기범 계좌번호, 송금 시간, 금액""",
            
            'fss_report_1332': """
   📞 전화: 1332
   💬 목적: 금융 피해 신고 및 상담
   📋 준비사항: 피해 내용, 관련 은행명""",
            
            'bank_contact': """
   📞 해당 은행 고객센터 전화
   💬 요청사항: "보이스피싱 피해로 지급정지 신청합니다"
   ⏰ 중요: 가능한 빨리 (24시간 이내 권장)""",
            
            'cert_reset': """
   🏛️ 방문: 은행 또는 인증기관
   📋 준비물: 신분증
   💡 순서: 기존 인증서 폐기 → 새 인증서 발급""",
            
            'malware_removal': """
   📱 방법1: 휴대폰 완전 초기화
   🏪 방법2: 통신사 고객센터 방문하여 악성앱 삭제
   ⚠️ 주의: 초기화 전 중요 데이터 백업""",
            
            'personal_info_registration': """
   🌐 웹사이트: pd.fss.or.kr
   📱 인증: 휴대폰 본인인증
   📋 효과: 신규 계좌개설, 카드발급 제한""",
            
            'account_check': """
   🌐 웹사이트: www.payinfo.or.kr
   🔐 로그인: 공동인증서 + 휴대폰 인증
   🔍 확인: '내계좌한눈에'에서 모든 계좌 조회
   ⚠️ 의심계좌 발견 시 즉시 해당 은행 신고""",
            
            'phone_check': """
   🌐 웹사이트: www.msafer.or.kr
   🔐 로그인: 공동인증서 또는 카카오페이
   🔍 확인: '가입사실현황조회'
   🛡️ 설정: '가입제한서비스'로 신규개통 차단""",
            
            'incident_report': """
   🏛️ 방문: 가까운 경찰서 또는 사이버수사대
   📄 발급: 사건사고사실확인원
   📋 준비물: 신분증, 피해 관련 자료""",
            
            'damage_claim': """
   🏛️ 방문: 지급정지 신청한 금융회사 영업점
   📄 제출서류: 
      - 사건사고사실확인원
      - 피해구제신청서
      - 신분증 사본
   ⏰ 기한: 지급정지 신청일로부터 3일 이내 (영업일 기준)"""
        }
        
        if action_key in guidance:
            print(guidance[action_key])
    
    async def _execute_current_stage(self):
        """현재 단계 실행"""
        
        stage = self.incident_data['current_stage']
        
        if stage == RecoveryStage.DAMAGE_ASSESSMENT:
            self.incident_data['current_stage'] = RecoveryStage.IMMEDIATE_RESPONSE
            
        elif stage == RecoveryStage.IMMEDIATE_RESPONSE:
            await self._handle_immediate_actions()
            if self._needs_personal_info_protection():
                self.incident_data['current_stage'] = RecoveryStage.PERSONAL_INFO_PROTECTION
            else:
                self.incident_data['current_stage'] = RecoveryStage.EVIDENCE_COLLECTION
                
        elif stage == RecoveryStage.PERSONAL_INFO_PROTECTION:
            await self._handle_personal_info_protection()
            self.incident_data['current_stage'] = RecoveryStage.EVIDENCE_COLLECTION
            
        elif stage == RecoveryStage.EVIDENCE_COLLECTION:
            await self._handle_verification_steps()
            self.incident_data['current_stage'] = RecoveryStage.LEGAL_PROCEDURES
            
        elif stage == RecoveryStage.LEGAL_PROCEDURES:
            await self._handle_legal_procedures()
            self.incident_data['current_stage'] = RecoveryStage.FINANCIAL_RECOVERY
    
    async def _handle_immediate_actions(self):
        """즉시 대응 조치 처리"""
        
        print(f"\n{'='*50}")
        print("🚨 즉시 대응이 필요합니다!")
        print("💡 다음 순서로 전화하시기 바랍니다:")
        print(f"{'='*50}")
        
        actions = self.official_checklist['immediate_actions']
        
        for key, action in actions.items():
            if not action['completed']:
                print(f"\n📞 {action['name']}")
                self._show_detailed_guidance(key)
                
                completed = input(f"\n✅ '{action['name']}'를 완료하셨나요? (y/n): ").strip().lower()
                if completed in ['y', 'yes', '예', 'ㅇ']:
                    action['completed'] = True
                    self.incident_data['completed_steps'].add(key)
                    print(f"✅ {action['name']} 완료 체크됨")
                else:
                    print(f"⏰ {action['name']}는 가능한 빨리 완료해주세요")
    
    async def _handle_personal_info_protection(self):
        """개인정보 보호 조치 처리"""
        
        print(f"\n{'='*50}")
        print("🛡️ 개인정보 보호 조치")
        print("📱 해킹 피해 확산 방지를 위한 필수 조치입니다")
        print(f"{'='*50}")
        
        actions = self.official_checklist['personal_info_protection']
        
        for key, action in actions.items():
            if not action['completed']:
                print(f"\n🔒 {action['name']}")
                self._show_detailed_guidance(key)
                
                completed = input(f"\n✅ '{action['name']}'를 완료하셨나요? (y/n): ").strip().lower()
                if completed in ['y', 'yes', '예', 'ㅇ']:
                    action['completed'] = True
                    self.incident_data['completed_steps'].add(key)
                    print(f"✅ {action['name']} 완료 체크됨")
    
    async def _handle_verification_steps(self):
        """명의도용 확인 단계 처리"""
        
        print(f"\n{'='*50}")
        print("🔍 명의도용 확인")
        print("🔐 본인 명의로 무단 개설된 계좌/휴대폰을 찾아보겠습니다")
        print(f"{'='*50}")
        
        actions = self.official_checklist['verification_steps']
        
        for key, action in actions.items():
            if not action['completed']:
                print(f"\n🔍 {action['name']}")
                self._show_detailed_guidance(key)
                
                # 실제 확인 결과 입력
                if key == 'account_check':
                    suspicious = input("\n⚠️ 의심스러운 계좌가 발견되었나요? (y/n): ").strip().lower()
                    if suspicious in ['y', 'yes', '예', 'ㅇ']:
                        print("🚨 즉시 해당 은행에 신고하고 계좌 정지를 요청하세요!")
                        
                elif key == 'phone_check':
                    suspicious = input("\n⚠️ 의심스러운 휴대폰이 발견되었나요? (y/n): ").strip().lower()
                    if suspicious in ['y', 'yes', '예', 'ㅇ']:
                        print("🚨 즉시 해당 통신사에 회선 해지를 요청하세요!")
                
                completed = input(f"\n✅ '{action['name']}'를 완료하셨나요? (y/n): ").strip().lower()
                if completed in ['y', 'yes', '예', 'ㅇ']:
                    action['completed'] = True
                    self.incident_data['completed_steps'].add(key)
                    print(f"✅ {action['name']} 완료 체크됨")
    
    async def _handle_legal_procedures(self):
        """법적 절차 처리"""
        
        print(f"\n{'='*50}")
        print("⚖️ 법적 절차 및 피해금 환급")
        print("📄 공식 서류를 통한 피해금 환급 신청")
        print(f"{'='*50}")
        
        # 사건사고사실확인원 발급 안내
        if DamageType.FINANCIAL_TRANSFER in self.incident_data['damage_types']:
            print("\n💰 금전 피해가 있으므로 피해금 환급 절차를 진행합니다")
            print("📋 전기통신금융사기 피해 방지 및 피해금 환급에 관한 특별법에 따라")
            print("   소송 없이 신속하게 피해금을 환급받을 수 있습니다")
        
        actions = self.official_checklist['legal_procedures']
        
        for key, action in actions.items():
            if not action['completed']:
                print(f"\n📄 {action['name']}")
                self._show_detailed_guidance(key)
                
                if key == 'damage_claim':
                    # 3일 기한 강조
                    print("\n⏰ 중요: 지급정지 신청일로부터 3일 이내(영업일 기준)에 제출해야 합니다!")
                    deadline = datetime.now() + timedelta(days=3)
                    print(f"📅 예상 마감일: {deadline.strftime('%Y년 %m월 %d일')}")
                
                completed = input(f"\n✅ '{action['name']}'를 완료하셨나요? (y/n): ").strip().lower()
                if completed in ['y', 'yes', '예', 'ㅇ']:
                    action['completed'] = True
                    self.incident_data['completed_steps'].add(key)
                    print(f"✅ {action['name']} 완료 체크됨")
    
    def _needs_personal_info_protection(self) -> bool:
        """개인정보 보호 조치 필요 여부"""
        return (DamageType.PERSONAL_INFO_LEAK in self.incident_data['damage_types'] or
                DamageType.MALICIOUS_APP in self.incident_data['damage_types'])
    
    def _has_remaining_steps(self) -> bool:
        """남은 단계가 있는지 확인"""
        return self.incident_data['current_stage'] != RecoveryStage.FINANCIAL_RECOVERY
    
    async def _ask_continue(self) -> bool:
        """계속 진행 여부 확인"""
        
        print("\n" + "="*50)
        choice = input("다음 단계로 진행하시겠습니까? (y/n): ").strip().lower()
        return choice in ['y', 'yes', '예', 'ㅇ']
    
    async def _show_final_summary(self):
        """최종 요약 표시"""
        
        print(f"\n{'='*60}")
        print("📊 사후대처 진행 상황 요약")
        print(f"{'='*60}")
        
        total_steps = 0
        completed_steps = 0
        
        for category, actions in self.official_checklist.items():
            print(f"\n📋 {category.replace('_', ' ').title()}:")
            for action in actions.values():
                status = "✅" if action['completed'] else "❌"
                print(f"   {status} {action['name']}")
                total_steps += 1
                if action['completed']:
                    completed_steps += 1
        
        completion_rate = (completed_steps / total_steps * 100) if total_steps > 0 else 0
        print(f"\n📈 전체 진행률: {completed_steps}/{total_steps} ({completion_rate:.1f}%)")
        
        # 중요 안내사항
        print(f"\n{'='*60}")
        print("🔔 중요 안내사항")
        print(f"{'='*60}")
        
        print("""
📞 추가 상담 및 문의:
   • 금융감독원: 1332
   • 경찰청: 112
   • 사이버수사대: 지역별 상이

🌐 유용한 웹사이트:
   • 개인정보노출자 사고예방: pd.fss.or.kr
   • 계좌정보통합관리: www.payinfo.or.kr  
   • 명의도용방지: www.msafer.or.kr
   • 금융감독원 보이스피싱지킴이: www.fss.or.kr

⚠️ 추가 주의사항:
   • 피해금 환급 절차는 2-3주 정도 소요될 수 있습니다
   • 의심스러운 연락이 다시 오면 즉시 차단하세요
   • 개인정보는 절대 전화로 제공하지 마세요
   • 정기적으로 본인 명의 계좌/휴대폰을 확인하세요

💡 재발 방지 팁:
   • 공식 기관은 전화로 개인정보를 요구하지 않습니다
   • 의심스러운 링크는 절대 클릭하지 마세요
   • 급하다고 하더라도 직접 확인하는 습관을 기르세요
   • 가족들에게도 보이스피싱 수법을 공유하세요
""")
        
        print(f"{'='*60}")
        print("🎯 사후대처 과정이 완료되었습니다.")
        print("💪 빠른 회복을 위해 남은 절차들도 꼭 완료해주세요!")
        print(f"{'='*60}")
    
    async def _cleanup_mode(self):
        """사후대처 모드 정리"""
        
        try:
            # 진행 상황 저장 (향후 확장)
            summary = {
                'session_id': self.session_id,
                'completion_time': datetime.now(),
                'damage_types': [dt.value for dt in self.incident_data['damage_types']],
                'completed_steps': list(self.incident_data['completed_steps']),
                'financial_loss': self.incident_data['financial_loss']
            }
            
            logger.info(f"사후대처 세션 완료: {summary}")
            
        except Exception as e:
            logger.error(f"사후대처 모드 정리 오류: {e}")