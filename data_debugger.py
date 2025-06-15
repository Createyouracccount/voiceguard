# debug_config.py - 설정 확인용 파일
import os
from dotenv import load_dotenv

def check_configuration():
    """설정 상태 확인"""
    load_dotenv()
    
    print("=== 환경변수 확인 ===")
    
    # OpenAI API 키 확인
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        print(f"✓ OPENAI_API_KEY: ...{openai_key[-4:]}")
    else:
        print("✗ OPENAI_API_KEY: 설정되지 않음")
    
    # Google API 키 확인
    google_key = os.getenv("GOOGLE_API_KEY")
    if google_key:
        print(f"✓ GOOGLE_API_KEY: ...{google_key[-4:]}")
    else:
        print("✗ GOOGLE_API_KEY: 설정되지 않음")
    
    # LangSmith 설정 확인
    langsmith_tracing = os.getenv("LANGCHAIN_TRACING_V2", "false")
    print(f"LangSmith 트래킹: {langsmith_tracing}")
    
    return openai_key is not None, google_key is not None

if __name__ == "__main__":
    check_configuration()