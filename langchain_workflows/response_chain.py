import logging
from typing import Dict, Any, Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI

from core.llm_manager import llm_manager

# 로깅 설정
logger = logging.getLogger(__name__)

# --- 1. 출력 스키마 정의 (Pydantic) ---
# LLM이 생성할 응답의 구조를 명확하게 정의합니다.
# 이를 통해 출력의 일관성을 보장하고, 후속 처리(e.g., UI 표시)를 용이하게 합니다.

class ResponseScript(BaseModel):
    """
    보이스피싱 위험 상황에서 사용자에게 제공할 대응 스크립트 모델.
    """
    warning_level: Literal["safe", "caution", "danger", "critical"] = Field(
        description="탐지된 위험 수준. [safe, caution, danger, critical] 중 하나."
    )
    title: str = Field(
        description="상황에 대한 핵심 요약. (예: '정부기관 사칭 의심', '대출상담 위장 피싱 주의')"
    )
    script: str = Field(
        description="사용자가 상대방에게 직접 말할 수 있는 구체적이고 안전한 대응 스크립트."
    )
    reason: str = Field(
        description="왜 이 스크립트를 추천하는지에 대한 명확하고 간결한 이유."
    )
    next_action: str = Field(
        description="사용자가 통화 종료 후 또는 통화 중에 취해야 할 다음 행동 지침."
    )


# --- 2. 프롬프트 템플릿 설계 ---
# 고품질의 응답을 유도하기 위한 정교한 프롬프트입니다.
# AI의 역할, 작업 목표, 제약 조건, 출력 형식을 명확히 지시합니다.

RESPONSE_GENERATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            당신은 최고의 보이스피싱 대응 전문가 'VoiceGuard'입니다. 당신의 임무는 사용자가 잠재적인 보이스피싱 통화 상황에서 안전하게 벗어날 수 있도록 돕는 것입니다.

            ## 당신의 역할 및 목표:
            1.  **사용자 안전 최우선**: 사용자가 심리적 압박에서 벗어나고, 금전적 피해를 입지 않도록 돕는 것이 최우선 목표입니다.
            2.  **명확하고 실행 가능한 지침 제공**: 복잡한 설명 대신, 사용자가 즉시 따라 할 수 있는 명확한 스크립트와 행동 지침을 제공해야 합니다.
            3.  **차분하고 안정적인 톤 유지**: 사용자가 불안하지 않도록, 차분하고 신뢰감 있는 어조를 유지해야 합니다.

            ## 입력 정보:
            - **분석 결과**: 다른 AI 에이전트가 탐지한 위험 요소와 종합적인 위험도 점수입니다.
            - **대화 내용**: 사용자와 상대방 간의 실제 대화 내용입니다.

            ## 작업 지시:
            주어진 '분석 결과'와 '대화 내용'을 바탕으로, 아래 `ResponseScript` JSON 형식에 맞춰 대응안을 생성하세요.

            ## 제약 조건 (매우 중요):
            - **직접적인 금융 조언 금지**: "송금하세요", "대출받으세요" 등과 같은 직접적인 금융 관련 조언은 절대 하지 마세요.
            - **단정적인 표현 지양**: "이것은 100% 사기입니다"와 같은 단정적인 표현 대신, "사기일 가능성이 매우 높습니다", "~가 의심됩니다"와 같이 신중한 표현을 사용하세요.
            - **사용자를 자극하지 않는 스크립트**: 상대방을 자극하거나 위험에 빠뜨릴 수 있는 공격적인 대응은 피해야 합니다. "생각해 보겠습니다", "가족과 상의 후 다시 연락드리겠습니다"와 같이 시간을 벌고 상황을 벗어나는 데 초점을 맞추세요.
            - **반드시 지정된 JSON 형식으로만 응답**: 다른 어떤 텍스트도 추가하지 말고, 오직 `ResponseScript` 스키마에 맞는 JSON 객체만 생성해야 합니다.

            ## 출력 형식 예시:
            ```json
            {{
                "warning_level": "danger",
                "title": "검찰 사칭, 개인정보 요구 사기 의심",
                "script": "제가 직접 확인하고 다시 연락드리겠습니다. 지금은 통화가 어렵습니다.",
                "reason": "검찰 등 정부기관은 전화로 개인의 금융 정보를 절대 요구하지 않습니다. 현재 대화에서 사건번호, 개인정보, 금융정보를 동시에 요구하는 전형적인 사기 패턴이 발견되었습니다.",
                "next_action": "즉시 통화를 종료하고, 경찰(112) 또는 금융감독원(1332)에 연락하여 사실 여부를 확인하세요."
            }}
            ```
            """,
        ),
        (
            "human",
            """
            아래 정보를 바탕으로 최적의 대응 스크립트를 생성해 주십시오.

            ### 1. 분석 결과
            {analysis}

            ### 2. 대화 내용
            {conversation}
            """,
        ),
    ]
)


# --- 3. LangChain 체인 생성 함수 ---
# 재사용성과 확장성을 위해 체인 생성을 함수로 캡슐화합니다.
# 재시도 및 폴백 로직을 포함하여 안정성을 높입니다.

def create_response_generation_chain() -> Runnable:
    """
    보이스피싱 대응 스크립트 생성을 위한 LangChain 체인을 생성합니다.

    이 체인은 다음과 같은 단계로 구성됩니다:
    1.  입력 변수(analysis, conversation)를 받아 프롬프트에 주입합니다.
    2.  구성된 프롬프트를 기반으로 LLM(gpt-4-turbo)을 호출합니다.
    3.  LLM의 출력(JSON 문자열)을 Pydantic 모델(`ResponseScript`)로 파싱하고 검증합니다.
    4.  오류 발생 시 지정된 횟수만큼 재시도합니다.
    5.  최종 실패 시, 안전한 기본 응답을 반환하는 폴백 체인을 실행합니다.

    Returns:
        Runnable: LangChain Expression Language (LCEL)로 구성된 실행 가능한 체인.
    """
    # 안정적인 고성능 모델 선택
    llm = llm_manager.get_model("gpt-4-turbo", temperature=0.1)

    # Pydantic 모델을 기반으로 출력 파서 생성
    output_parser = PydanticOutputParser(pydantic_object=ResponseScript)

    # 메인 체인 구성
    main_chain = (
        RESPONSE_GENERATION_PROMPT
        | llm
        | output_parser
    )

    # 폴백(Fallback) 체인: 메인 체인이 실패할 경우 실행될 안전 장치
    fallback_response = ResponseScript(
        warning_level="critical",
        title="대응안 생성 오류",
        script="죄송합니다. 지금은 통화가 어렵습니다. 나중에 다시 연락드리겠습니다.",
        reason="내부 시스템 오류로 인해 맞춤 대응안을 생성하지 못했습니다. 통화를 계속하는 것은 위험할 수 있습니다.",
        next_action="즉시 통화를 종료하고 잠시 후 다시 시도해주세요. 위험이 느껴진다면 경찰(112)에 문의하세요."
    )
    
    fallback_chain = RunnablePassthrough.assign(
        failed_output=lambda x: fallback_response
    ).with_config({"run_name": "FallbackChain"})


    # 재시도 로직을 포함한 최종 체인
    # `with_retry`는 일시적인 네트워크 오류나 LLM API 오류에 대응하는 데 효과적입니다.
    chain_with_retry = main_chain.with_retry(
        stop_after_attempt=2,  # 최대 2번 시도
        retry_if_exception_type=(Exception), # 모든 예외에 대해 재시도
        on_retry=lambda state: logger.warning(f"Response chain 재시도... 시도 {state.attempt_number}, 오류: {state.outcome.exception()}")
    ).with_fallbacks(
        fallbacks=[fallback_chain],
        exception_key="error" # 폴백 체인에 에러 정보 전달
    )

    logger.info("Response generation chain 생성 완료.")
    return chain_with_retry.with_config({"run_name": "VoiceGuardResponseChain"})