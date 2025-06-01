# simple.py
from fastapi import FastAPI

app = FastAPI()

@app.get("/ping")
async def ping():
    print(">>> simple.py의 /ping이 호출되었습니다.")
    return {"pong": True}
