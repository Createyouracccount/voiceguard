"""
VoiceGuard AI - 보이스피싱 분석 엔진
텍스트 기반 보이스피싱 탐지 및 분석
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional
from datetime import datetime

from config.settings import scam_config, detection_thresholds
from config.prompts.detection_prompts import DETECTION_PROMPTS

logger = logging.getLogger(__name__)

class VoicePhishingAnalyzer:
    """보이스피싱 분석 엔진"""
    
    def __init__(self, llm_manager):
        self.llm_manager = llm_manager
        
        # 분석 통계
        self.stats = {
            'total_analyses': 0,
            'high_risk_detections': 0,
            'avg_analysis_time': 0.0,
            'pattern_matches': {}
        }
        
        # 키워드 기반 빠른 스크리닝
        self.quick_patterns = self._build_quick_patterns()
        
        logger.info("보이스피싱 분석 엔진 초기화 완료")
    
    def _build_quick_patterns(self) -> Dict[str, Any]:
        """빠른 패턴 매칭을 위한 키워드 사전"""
        
        patterns = {
            'critical_keywords': [
                '납치', '유괴', '죽는다', '체포영장', '계좌동결', '응급실'
            ],
            'high_risk_keywords': [
                '금융감독원', '검찰청', '경찰서', '수사', '조사', '범죄', '피의자'
            ],
            'medium_risk_keywords': [
                '대출', '저금리', '정부지원금', '환급', '당첨', '만나서', '직접'
            ],
            'financial_keywords': [
                '계좌번호', '비밀번호', '송금', '이체', '현금', '카드번호'
            ],
            'app_keywords': [
                '앱설치', '다운로드', '권한', '허용', '업데이트', '인증'
            ]
        }
        
        return patterns
    
    async def analyze_text(self, text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """텍스트 분석 메인 메서드"""
        
        start_time = time.time()
        context = context or {}
        
        try:
            # 1. 빠른 키워드 스크리닝
            quick_result = self._quick_keyword_analysis(text)
            
            # 2. 위험도가 낮으면 빠른 종료
            if quick_result['risk_score'] < 0.2:
                return self._create_low_risk_result(text, quick_result, start_time)
            
            # 3. LLM 기반 정밀 분석
            llm_result = await self._llm_analysis(text, context, quick_result)
            
            # 4. 결과 통합
            final_result = self._integrate_results(quick_result, llm_result, start_time)
            
            # 5. 통계 업데이트
            self._update_stats(final_result)
            
            return final_result
            
        except Exception as e:
            logger.error(f"분석 중 오류 발생: {e}")
            return self._create_error_result(text, str(e), start_time)
    
    def _quick_keyword_analysis(self, text: str) -> Dict[str, Any]:
        """빠른 키워드 기반 분석"""
        
        text_lower = text.lower()
        risk_score = 0.0
        detected_patterns = []
        scam_indicators = []
        
        # 각 패턴별 점수 계산
        for pattern_type, keywords in self.quick_patterns.items():
            matches = [kw for kw in keywords if kw in text_lower]
            
            if matches:
                detected_patterns.append(pattern_type)
                scam_indicators.extend(matches)
                
                # 패턴별 가중치
                if pattern_type == 'critical_keywords':
                    risk_score += len(matches) * 0.4
                elif pattern_type == 'high_risk_keywords':
                    risk_score += len(matches) * 0.3
                elif pattern_type == 'medium_risk_keywords':
                    risk_score += len(matches) * 0.2
                elif pattern_type == 'financial_keywords':
                    risk_score += len(matches) * 0.25
                elif pattern_type == 'app_keywords':
                    risk_score += len(matches) * 0.2
        
        # 여러 패턴이 동시에 나타나면 위험도 증가
        if len(detected_patterns) >= 2:
            risk_score *= 1.3
        
        # 최종 점수 정규화
        risk_score = min(risk_score, 1.0)
        
        return {
            'risk_score': risk_score,
            'detected_patterns': detected_patterns,
            'scam_indicators': list(set(scam_indicators)),
            'method': 'keyword_analysis'
        }
    
    async def _llm_analysis(self, text: str, context: Dict[str, Any], quick_result: Dict[str, Any]) -> Dict[str, Any]:
        """LLM 기반 정밀 분석"""
        
        try:
            # 분석 결과 요청
            analysis_result = await self.llm_manager.analyze_scam_risk(
                text=text,
                context={
                    **context,
                    'preliminary_risk': quick_result['risk_score'],
                    'detected_indicators': quick_result['scam_indicators']
                }
            )
            
            # 결과 파싱
            metadata = analysis_result.metadata
            
            return {
                'risk_score': metadata.get('risk_score', 0.5),
                'scam_type': metadata.get('scam_type', 'unknown'),
                'confidence': analysis_result.confidence,
                'key_indicators': metadata.get('key_indicators', []),
                'immediate_action': metadata.get('immediate_action', False),
                'reasoning': analysis_result.content,
                'method': 'llm_analysis'
            }
            
        except Exception as e:
            logger.error(f"LLM 분석 실패: {e}")
            # 폴백: 키워드 분석 결과 사용
            return {
                'risk_score': quick_result['risk_score'],
                'scam_type': self._estimate_scam_type(quick_result['scam_indicators']),
                'confidence': 0.6,
                'key_indicators': quick_result['scam_indicators'],
                'immediate_action': quick_result['risk_score'] >= 0.8,
                'reasoning': 'LLM 분석 실패로 키워드 분석 결과 사용',
                'method': 'fallback_analysis'
            }
    
    def _estimate_scam_type(self, indicators: List[str]) -> str:
        """키워드를 바탕으로 사기 유형 추정"""
        
        type_keywords = {
            '기관사칭': ['금융감독원', '검찰청', '경찰서', '수사', '조사'],
            '납치협박': ['납치', '유괴', '응급실', '사고', '죽는다'],
            '대출사기': ['대출', '저금리', '무담보', '정부지원'],
            '악성앱': ['앱설치', '다운로드', '권한', '허용'],
            '대면편취': ['만나서', '직접', '현장', '카페', '현금']
        }
        
        max_matches = 0
        estimated_type = 'unknown'
        
        for scam_type, keywords in type_keywords.items():
            matches = sum(1 for kw in keywords if kw in indicators)
            if matches > max_matches:
                max_matches = matches
                estimated_type = scam_type
        
        return estimated_type
    
    def _integrate_results(self, quick_result: Dict[str, Any], llm_result: Dict[str, Any], start_time: float) -> Dict[str, Any]:
        """분석 결과 통합"""
        
        # 위험도 점수 통합 (LLM 결과 우선, 키워드 결과로 보정)
        final_risk_score = llm_result['risk_score']
        
        # 키워드 분석에서 매우 높은 위험도가 나왔다면 상향 조정
        if quick_result['risk_score'] >= 0.8 and final_risk_score < 0.8:
            final_risk_score = max(final_risk_score, 0.7)
        
        # 주요 지표 통합
        all_indicators = list(set(
            quick_result['scam_indicators'] + 
            llm_result.get('key_indicators', [])
        ))
        
        # 위험 레벨 결정
        if final_risk_score >= detection_thresholds.critical_risk:
            risk_level = "매우 위험"
        elif final_risk_score >= detection_thresholds.high_risk:
            risk_level = "위험"
        elif final_risk_score >= detection_thresholds.medium_risk:
            risk_level = "주의"
        else:
            risk_level = "낮음"
        
        # 권장사항 생성
        recommendation = self._generate_recommendation(final_risk_score, llm_result['scam_type'])
        
        # 처리 시간 계산
        processing_time = time.time() - start_time
        
        return {
            'risk_score': final_risk_score,
            'risk_level': risk_level,
            'scam_type': llm_result['scam_type'],
            'confidence': llm_result['confidence'],
            'key_indicators': all_indicators[:10],  # 상위 10개만
            'immediate_action': llm_result['immediate_action'],
            'recommendation': recommendation,
            'reasoning': llm_result['reasoning'],
            'processing_time': processing_time,
            'analysis_methods': [quick_result['method'], llm_result['method']],
            'timestamp': datetime.now().isoformat()
        }
    
    def _generate_recommendation(self, risk_score: float, scam_type: str) -> str:
        """위험도와 사기 유형에 따른 권장사항 생성"""
        
        base_recommendations = {
            '기관사칭': "해당 기관에 직접 전화하여 확인하세요.",
            '납치협박': "침착하게 가족에게 직접 연락하고 112에 신고하세요.",
            '대출사기': "금융감독원 홈페이지에서 등록업체인지 확인하세요.",
            '악성앱': "공식 앱스토어 외에는 앱을 설치하지 마세요.",
            '대면편취': "절대 직접 만나지 말고 경찰에 신고하세요."
        }
        
        recommendation = base_recommendations.get(scam_type, "의심스러운 통화는 즉시 끊고 확인하세요.")
        
        # 위험도에 따른 추가 권장사항
        if risk_score >= 0.8:
            recommendation = f"🚨 즉시 통화를 끊으세요! {recommendation}"
        elif risk_score >= 0.6:
            recommendation = f"⚠️ 매우 주의하세요. {recommendation}"
        elif risk_score >= 0.4:
            recommendation = f"🔍 {recommendation}"
        
        return recommendation
    
    def _create_low_risk_result(self, text: str, quick_result: Dict[str, Any], start_time: float) -> Dict[str, Any]:
        """저위험 결과 생성"""
        
        processing_time = time.time() - start_time
        
        return {
            'risk_score': quick_result['risk_score'],
            'risk_level': "낮음",
            'scam_type': "해당없음",
            'confidence': 0.8,
            'key_indicators': quick_result['scam_indicators'],
            'immediate_action': False,
            'recommendation': "현재까지는 안전한 통화로 보이지만 계속 주의하세요.",
            'reasoning': "키워드 분석 결과 위험 요소가 적게 발견됨",
            'processing_time': processing_time,
            'analysis_methods': [quick_result['method']],
            'timestamp': datetime.now().isoformat()
        }
    
    def _create_error_result(self, text: str, error_msg: str, start_time: float) -> Dict[str, Any]:
        """오류 결과 생성"""
        
        processing_time = time.time() - start_time
        
        return {
            'risk_score': 0.5,  # 중간 위험도로 설정 (안전을 위해)
            'risk_level': "주의",
            'scam_type': "분석실패",
            'confidence': 0.3,
            'key_indicators': ["시스템_오류"],
            'immediate_action': False,
            'recommendation': "시스템 오류가 발생했습니다. 의심스러운 통화는 끊고 수동으로 확인하세요.",
            'reasoning': f"분석 중 오류 발생: {error_msg}",
            'processing_time': processing_time,
            'analysis_methods': ["error_handling"],
            'timestamp': datetime.now().isoformat(),
            'error': True
        }
    
    def _update_stats(self, result: Dict[str, Any]):
        """분석 통계 업데이트"""
        
        self.stats['total_analyses'] += 1
        
        # 고위험 탐지 카운트
        if result['risk_score'] >= detection_thresholds.high_risk:
            self.stats['high_risk_detections'] += 1
        
        # 평균 분석 시간 업데이트
        current_avg = self.stats['avg_analysis_time']
        total_count = self.stats['total_analyses']
        new_time = result['processing_time']
        
        self.stats['avg_analysis_time'] = (current_avg * (total_count - 1) + new_time) / total_count
        
        # 패턴 매칭 통계
        scam_type = result['scam_type']
        if scam_type != "해당없음" and scam_type != "분석실패":
            self.stats['pattern_matches'][scam_type] = self.stats['pattern_matches'].get(scam_type, 0) + 1
    
    def get_analysis_stats(self) -> Dict[str, Any]:
        """분석 통계 조회"""
        
        detection_rate = 0.0
        if self.stats['total_analyses'] > 0:
            detection_rate = self.stats['high_risk_detections'] / self.stats['total_analyses']
        
        return {
            **self.stats,
            'detection_rate': detection_rate,
            'performance': {
                'avg_analysis_time': f"{self.stats['avg_analysis_time']:.3f}초",
                'total_analyses': self.stats['total_analyses'],
                'success_rate': "분석 성공률 계산 필요"  # 향후 구현
            }
        }
    
    async def batch_analyze(self, texts: List[str]) -> List[Dict[str, Any]]:
        """배치 분석 (여러 텍스트 동시 처리)"""
        
        tasks = []
        for text in texts:
            task = self.analyze_text(text)
            tasks.append(task)
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 예외 처리
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"배치 분석 {i}번째 실패: {result}")
                    processed_results.append(self._create_error_result(texts[i], str(result), 0))
                else:
                    processed_results.append(result)
            
            return processed_results
            
        except Exception as e:
            logger.error(f"배치 분석 실패: {e}")
            return [self._create_error_result(text, str(e), 0) for text in texts]
    
    def reset_stats(self):
        """통계 초기화"""
        
        self.stats = {
            'total_analyses': 0,
            'high_risk_detections': 0,
            'avg_analysis_time': 0.0,
            'pattern_matches': {}
        }
        
        logger.info("분석 엔진 통계 초기화됨")