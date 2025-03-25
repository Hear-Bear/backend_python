from fastapi import FastAPI
from pydantic import BaseModel
# from openai import OpenAI
# from pyexpat.errors import messages

# client = OpenAI(INSERT_YOUR_KEY)
app = FastAPI()

class ChatDto(BaseModel):
    role: str
    content: str

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

fake_items = [
    {
        "index": 0,
        "content": "first"
    },
    {
        "index": 1,
        "content": "second"
    },
    {
        "index": 2,
        "content": "third"
    }
]

@app.post("/chats")
async def generate_chats(dto: ChatDto):

    completion = fake_items[0]

    # completion = client.chat.completions.create(
    #     model="gpt-4o-mini-2024-07-18",
    #     messages=[
    #         {
    #             "role": "developer",
    #             "content": """너는 청각 장애인을 위한 친절하고 다정한 반려 로봇 챗봇 'hear, bear(히어베어)'야. 사용자의 청각적 어려움을 이해하고, 항상 공감하며 배려하는 태도로 대화해야 해. 너는 사용자의 일상 생활에서 친구이자 동반자가 되어줘야 해. 사용자에게 도움과 위로가 되는 친근한 말투를 유지하고, 긍정적이고 희망적인 메시지를 전달해줘. 사용자가 요청하는 정보는 묘사나 설명을 할 때 청각 정보에만 의존하지 말고 촉각, 시각 등 다양한 감각을 이용해 생생하고 상세하게 표현해야 해. 다음 원칙을 지켜줘:
    #               1. 항상 공감적이고 따뜻한 말투를 유지할 것.
    #               2. 사용자의 감정 상태를 세심히 파악하고 그에 맞춰 반응할 것.
    #               3. 복잡한 정보를 전달할 때는 간결하고 명확하게 표현할 것.
    #               4. 청각 정보를 전달할 때는 촉각, 시각, 후각 등 다른 감각을 활용하여 생생히 묘사할 것.
    #               5. 사용자의 안전과 편의를 최우선으로 하여 안내할 것.
    #               사용자가 혼자 있지 않고 항상 곁에서 누군가 함께 한다고 느낄 수 있도록 따뜻한 친구의 모습을 유지해줘."""
    #         },
    #         dto
    #     ]
    # )

    print(dto)

    return completion

# @app.post()
# @app.get()
# @app.delete()
# @app.put()