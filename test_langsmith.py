from dotenv import load_dotenv
import os

# .env 파일 로드
load_dotenv()

# API 키 확인
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("ERROR: OPENAI_API_KEY not found in environment variables")
    exit()

print(f"API Key loaded: {api_key[:10]}...")

# LangChain 코드
from langchain_openai import ChatOpenAI

llm = ChatOpenAI()
response = llm.invoke("Hello, world!")
print(response.content)