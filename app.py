import os

import openai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from text.sentiment_analysis import classify_emotion, generate_chat_response, load_models_and_data
from dotenv import load_dotenv

load_dotenv()

# 환경 변수에서 OpenAI API 키 불러오기
# 로컬에서 실행 시, 반드시 .env 등에 OPENAI_API_KEY를 세팅해야 함
openai.api_key = os.getenv('OPENAI_SECRET_KEY')
if openai.api_key is None:
    raise RuntimeError("OPENAI_API_KEY 환경변수가 설정되어 있지 않습니다.")

# FastAPI 앱 객체 생성
app = FastAPI(
    title="Hear-Bear ChatGPT API",
    description="프론트엔드에서 온 문장을 감정 분류 후 ChatGPT에 전달하여 답변을 반환하는 API",
    version="1.0"
)


# 요청(Request) 및 응답(Response) 모델 정의
class ChatRequest(BaseModel):
    message: str = Field(description="프론트엔드로부터 전달받은 새로운 사용자 메시지")
    # history: List[str] = Field(default_factory=list, description="(선택 사항) 이전 대화 내역을 리스트로 전달. (없으면 빈 리스트로)")
    # tfidf_threshold: float = Field(0.75, description="TF-IDF 신뢰도 임계값 (0~1 사이)")

class ChatResponse(BaseModel):
    sentiment: str = Field(..., description="분류된 감정 레이블(예: '행복', '슬픔', '분노' 등)")
    confidence: float = Field(..., description="분류 확률 또는 SBERT-KNN fallback 신뢰도")
    bot_reply: str = Field(..., description="ChatGPT가 반환한 답변 메시지")
    # history: List[str] = Field(description="응답 후 업데이트된 대화 내역(추가된 사용자 메시지와 봇의 답변 포함)", default_factory=list)


# OpenAI의 일반적인 Example response
# [
#     {
#         "index": 0,
#         "message": {
#             "role": "assistant",
#             "content": "Under the soft glow of the moon, Luna the unicorn danced through fields of twinkling stardust, leaving trails of dreams for every child asleep.",
#             "refusal": null
#         },
#         "logprobs": null,
#         "finish_reason": "stop"
#     }
# ]
# 이 중에서, chat에 필요한 데이터는 index, content

@app.on_event("startup")
async def on_startup():
    """
    서버가 시작될 때 단 한 번만 실행됩니다.
    여기서 무거운 모델 로드 및 데이터 로드를 수행합니다.
    """
    print(">> 서버 시작 시, 감정 분류 모델과 SBERT 임베딩을 한 번만 로드합니다.")
    load_models_and_data()
    print(">> 모델 로드 완료!!")

@app.get("/ping")
async def ping():
    return {"pong": True}

# POST /chat 엔드포인트 정의
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    1) request.message → classify_emotion 으로 감정 분류
    2) 분류 결과(sentiment, confidence)와 request.history (이전 대화들) 을
       generate_chat_response 에 전달 → ChatGPT 답변 받기
    3) 최종 JSON 형태로 응답
    """
    FIXED_TFIDF_THRESHOLD = 0.75
    print(request.message)

    # 1) 감정 분류
    try:
        cls_res = classify_emotion(request.message, tfidf_threshold=FIXED_TFIDF_THRESHOLD)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"감정 분류 중 오류 발생: {str(e)}")

    sentiment = cls_res["sentiment"]
    confidence = cls_res["confidence"]

    # 2) ChatGPT 응답 생성
    #    - 대화 이력(history)에 현재 사용자 메시지를 추가하고,
    #      generate_chat_response에 넘겨줍니다.
    # new_history = request.history.copy()
    # new_history.append(request.message)  # 사용자 발화 추가

    try:
        bot_reply = generate_chat_response(request.message, sentiment)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ChatGPT 호출 중 오류 발생: {str(e)}")

    # 3) 봇의 답변을 내역에 추가
    # new_history.append(bot_reply)

    return ChatResponse(
        sentiment=sentiment,
        confidence=confidence,
        bot_reply=bot_reply,
        # history=new_history
    )