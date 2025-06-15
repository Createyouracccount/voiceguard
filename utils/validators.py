"""
VoiceGuard AI - 환경 검증 유틸리티
시스템 실행 전 필수 조건 확인
"""

import os
import logging
from typing import Dict, List, Any
from pathlib import Path

logger = logging.getLogger(__name__)

def validate_environment() -> Dict[str, Any]:
    """환경 검증 종합 함수"""
    
    errors = []
    warnings = []
    
    # 1. 필수 환경변수 확인
    env_check = validate_environment_variables()
    errors.extend(env_check['errors'])
    warnings.extend(env_check['warnings'])
    
    # 2. 파일 시스템 확인
    file_check = validate_file_system()
    errors.extend(file_check['errors'])
    warnings.extend(file_check['warnings'])
    
    # 3. 시스템 리소스 확인
    resource_check = validate_system_resources()
    warnings.extend(resource_check['warnings'])
    
    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings,
        'summary': f"검증 완료 - 오류: {len(errors)}개, 경고: {len(warnings)}개"
    }

def validate_environment_variables() -> Dict[str, List[str]]:
    """환경변수 검증"""
    
    errors = []
    warnings = []
    
    # 필수 환경변수
    required_vars = {
        'GOOGLE_API_KEY': 'Google Gemini API 키가 필요합니다',
    }
    
    # 선택 환경변수
    optional_vars = {
        'RETURNZERO_CLIENT_ID': 'STT 서비스 이용 시 필요 (ReturnZero)',
        'RETURNZERO_CLIENT_SECRET': 'STT 서비스 이용 시 필요 (ReturnZero)',
        'ELEVENLABS_API_KEY': 'TTS 서비스 이용 시 필요 (ElevenLabs)',
        'OPENAI_API_KEY': '향후 OpenAI 모델 사용 시 필요',
        'ANTHROPIC_API_KEY': '향후 Claude 모델 사용 시 필요'
    }
    
    # 필수 환경변수 확인
    for var_name, description in required_vars.items():
        if not os.getenv(var_name):
            errors.append(f"{var_name} 환경변수가 설정되지 않음 - {description}")
        else:
            # API 키 형식 간단 검증
            value = os.getenv(var_name)
            if len(value) < 10:
                warnings.append(f"{var_name} 값이 너무 짧습니다 (API 키가 맞나요?)")
    
    # 선택 환경변수 확인
    for var_name, description in optional_vars.items():
        if not os.getenv(var_name):
            warnings.append(f"{var_name} 미설정 - {description}")
    
    return {'errors': errors, 'warnings': warnings}

def validate_file_system() -> Dict[str, List[str]]:
    """파일 시스템 검증"""
    
    errors = []
    warnings = []
    
    # 프로젝트 루트
    project_root = Path(__file__).parent.parent
    
    # 필수 디렉토리 확인 - audio 폴더 제거
    required_dirs = [
        'app',
        'app/modes',
        'core',
        'services',
        'config',
        'config/data',
        'utils'
    ]
    
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if not full_path.exists():
            errors.append(f"필수 디렉토리 없음: {dir_path}")
        elif not full_path.is_dir():
            errors.append(f"디렉토리가 아님: {dir_path}")
    
    # 필수 파일 확인 - audio 폴더 경로 수정
    required_files = [
        'app/app.py',
        'core/llm_manager.py', 
        'core/analyzer.py',
        'services/tts_service.py',  # audio 폴더 제거
        'config/settings.py'
    ]
    
    for file_path in required_files:
        full_path = project_root / file_path
        if not full_path.exists():
            errors.append(f"필수 파일 없음: {file_path}")
        elif not full_path.is_file():
            errors.append(f"파일이 아님: {file_path}")
    
    # 쓰기 권한 확인
    temp_file = project_root / 'temp_write_test.txt'
    try:
        with open(temp_file, 'w') as f:
            f.write('test')
        temp_file.unlink()  # 삭제
    except Exception as e:
        warnings.append(f"프로젝트 디렉토리 쓰기 권한 없음: {e}")
    
    return {'errors': errors, 'warnings': warnings}

def validate_system_resources() -> Dict[str, List[str]]:
    """시스템 리소스 검증"""
    
    warnings = []
    
    try:
        import psutil
        
        # 메모리 확인
        memory = psutil.virtual_memory()
        if memory.available < 1024 * 1024 * 1024:  # 1GB
            warnings.append("사용 가능한 메모리가 1GB 미만입니다")
        
        # CPU 확인
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > 90:
            warnings.append("CPU 사용률이 90% 이상입니다")
        
        # 디스크 확인
        disk = psutil.disk_usage('/')
        if disk.free < 1024 * 1024 * 1024:  # 1GB
            warnings.append("사용 가능한 디스크 공간이 1GB 미만입니다")
            
    except ImportError:
        warnings.append("psutil 패키지가 설치되지 않아 시스템 리소스를 확인할 수 없습니다")
    except Exception as e:
        warnings.append(f"시스템 리소스 확인 중 오류: {e}")
    
    return {'warnings': warnings}

def validate_dependencies() -> Dict[str, Any]:
    """Python 패키지 의존성 검증"""
    
    required_packages = {
        'asyncio': '비동기 처리',
        'logging': '로깅',
        'pathlib': '파일 경로 처리',
        'datetime': '시간 처리',
        'enum': '열거형'
    }
    
    optional_packages = {
        'google-generativeai': 'Google Gemini API',
        'openai': 'OpenAI API (향후)',
        'anthropic': 'Anthropic Claude API (향후)',
        'elevenlabs': 'ElevenLabs TTS',
        'pyaudio': '오디오 입출력',
        'grpc': 'gRPC 통신 (STT)',
        'psutil': '시스템 모니터링'
    }
    
    missing_required = []
    missing_optional = []
    
    # 필수 패키지 확인
    for package, description in required_packages.items():
        try:
            __import__(package)
        except ImportError:
            missing_required.append(f"{package} - {description}")
    
    # 선택 패키지 확인
    for package, description in optional_packages.items():
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_optional.append(f"{package} - {description}")
    
    return {
        'missing_required': missing_required,
        'missing_optional': missing_optional,
        'all_required_available': len(missing_required) == 0
    }

def check_api_connectivity() -> Dict[str, bool]:
    """API 연결성 확인 (간단한 버전)"""
    
    connectivity = {}
    
    # Google API 키 형식 확인
    google_key = os.getenv('GOOGLE_API_KEY')
    if google_key and len(google_key) > 20:
        connectivity['google_api'] = True
    else:
        connectivity['google_api'] = False
    
    # ElevenLabs API 키 형식 확인
    elevenlabs_key = os.getenv('ELEVENLABS_API_KEY')
    if elevenlabs_key and len(elevenlabs_key) > 20:
        connectivity['elevenlabs_api'] = True
    else:
        connectivity['elevenlabs_api'] = False
    
    # ReturnZero 자격증명 확인
    rz_id = os.getenv('RETURNZERO_CLIENT_ID')
    rz_secret = os.getenv('RETURNZERO_CLIENT_SECRET')
    if rz_id and rz_secret:
        connectivity['returnzero_api'] = True
    else:
        connectivity['returnzero_api'] = False
    
    return connectivity

def generate_setup_instructions(validation_result: Dict[str, Any]) -> str:
    """설정 가이드 생성"""
    
    instructions = []
    
    if validation_result['errors']:
        instructions.append("🔧 필수 설정:")
        for error in validation_result['errors']:
            instructions.append(f"   ❌ {error}")
    
    if validation_result['warnings']:
        instructions.append("\n⚠️ 권장 설정:")
        for warning in validation_result['warnings']:
            instructions.append(f"   ⚠️ {warning}")
    
    # .env 파일 템플릿 제안
    if any('환경변수' in error for error in validation_result['errors']):
        instructions.append("""
📝 .env 파일 생성 예시:
```
# 필수 설정
GOOGLE_API_KEY=your_google_api_key_here

# 선택 설정 (기능별 필요시)
RETURNZERO_CLIENT_ID=your_returnzero_id
RETURNZERO_CLIENT_SECRET=your_returnzero_secret
ELEVENLABS_API_KEY=your_elevenlabs_key

# 디버그 설정
DEBUG=True
LOG_LEVEL=INFO
```
""")
    
    return "\n".join(instructions)

def quick_health_check() -> bool:
    """빠른 상태 확인"""
    
    # 최소한의 실행 조건만 확인
    google_key = os.getenv('GOOGLE_API_KEY')
    
    if not google_key:
        return False
    
    if len(google_key) < 10:
        return False
    
    return True