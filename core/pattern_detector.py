# core/pattern_detector.py

import json
import logging
from pathlib import Path
from typing import Dict, List, Any

class PatternDetector:
    def __init__(self, patterns_path: Path):
        self.patterns = self._load_patterns(patterns_path)
        logging.info(f"사기 패턴 로드 완료. {len(self.patterns)}개 카테고리.")

    def _load_patterns(self, patterns_path: Path) -> Dict[str, Any]:
        try:
            with open(patterns_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logging.error(f"패턴 파일 로드 실패: {e}")
            return {}

    def detect(self, text: str) -> Dict[str, Any]:
        detected_patterns = {}
        total_score = 0
        
        for p_type, p_data in self.patterns.items():
            found_keywords = [kw for kw in p_data.get("keywords", []) if kw in text]
            found_phrases = [ph for ph in p_data.get("phrases", []) if ph in text]
            
            if found_keywords or found_phrases:
                score = len(found_keywords) + len(found_phrases) * 1.5
                total_score += score * p_data.get("weight", 1.0)
                detected_patterns[p_type] = {
                    "keywords": found_keywords,
                    "phrases": found_phrases,
                    "score": score,
                    "weight": p_data.get("weight", 1.0)
                }

        return {"detected_patterns": detected_patterns, "pattern_score": total_score}