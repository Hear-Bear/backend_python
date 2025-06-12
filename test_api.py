from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import os
import threading
import subprocess
from pathlib import Path
from pydantic import BaseModel
import json 
import signal
import time
import httpx
import firebase_admin
from firebase_admin import messaging, credentials


app = FastAPI()

# ROS 노드 정의
class RosPublisherNode(Node):
    def __init__(self):
        super().__init__('fastapi_ros_node')
        self.command_publisher = self.create_publisher(String, 'command', 10)
        self.pixel_goal_publisher = self.create_publisher(String, 'pixel_goal', 10)
        self.reexplore_publisher = self.create_publisher(String, 'reexplore', 10)

    def publish_command(self, text):
        msg = String()
        msg.data = text
        self.command_publisher.publish(msg)
        self.get_logger().info(f"Published command: {text}")

    def publish_pixel_goal(self, px, py):
        msg = String()
        msg.data = f"{px},{py}"
        self.pixel_goal_publisher.publish(msg)
        self.get_logger().info(f"Published pixel goal: {px},{py}")
    
    def publish_reexplore(self):
        msg = String()
        self.reexplore_publisher.publish(msg)
        self.get_logger().info(f"Published reexplore")

class NoiseCoordinate(BaseModel):
    name: str
    x: float
    y: float

# ROS 노드 전역 인스턴스 생성
ros_node = None

def ros_spin_thread(node):
    rclpy.spin(node)

@app.on_event("startup")
def startup_event():
    global ros_node
    rclpy.init()
    ros_node = RosPublisherNode()
    threading.Thread(target=ros_spin_thread, args=(ros_node,), daemon=True).start()

@app.on_event("shutdown")
def shutdown_event():
    global ros_node
    ros_node.destroy_node()
    rclpy.shutdown()

@app.get("/")
def hello():
    return {"message": "Hello world"}

listener_proc = None  # 전역 listener 프로세스 참조

@app.post("/get-command")
async def receive_test(request: Request):
    global listener_proc

    # 1) JSON 파싱
    try:
        data = await request.json()
        command = data.get("command")
        if not command:
            raise ValueError("`command` 필드가 없습니다.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    print(f"[FastAPI] 받은 명령어: {command}")

    # subprocess.Popen(["ros2", "run", "hearbear", "command_listener.py"])

    # 4) Listener가 spin 시작할 시간을 잠깐 줌
    time.sleep(0.5)

    # 5) 퍼블리시
    ros_node.publish_command(command)

    return {"status": "ok", "command": command}

def launch_ros2_mapping():
    subprocess.Popen(["ros2", "launch", "hearbear", "cartographer_mapping.launch.py"])

@app.get("/get-map")
def get_map():
    image_path = "/home/nvidia/turtlebot3_ws/map/map.png"
    
    if os.path.exists(image_path):
        return FileResponse(image_path, media_type="image/png")
    else:
        return {"error": "Image not found"}

@app.post('/pixel-to-navi')
async def navigate(payload: dict):
    try:
        px = int(payload.get('px'))
        py = int(payload.get('py'))
    except (ValueError, TypeError):
        return {"error": "Invalid 'px' or 'py' in payload. Must be integers."}

    ros_node.publish_pixel_goal(px, py)

    return {'status': 'pixel goal sent', 'px': px, 'py': py}

@app.post("/request_reexplore")
async def request_reexplore():
    ros_node.publish_reexplore()
    return {"status": "재탐색 요청이 전달되었습니다."}

class DestinationItem(BaseModel):
    category: str
    px: float
    py: float

DEST_PATH = Path("/home/nvidia/turtlebot3_ws/dest_list/destination.json")
@app.post("/save-destination")
async def save_destination(destinations: list[DestinationItem]):
    try:
        # 상위 디렉토리 생성
        DEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        # 리스트를 dict로 변환 후 저장
        data_to_save = [dest.dict() for dest in destinations]
        with open(DEST_PATH, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=2)
    except Exception as e:
        return {"error": f"File write failed: {str(e)}"}

    return {
        "message": "Destination saved successfully",
        "path": str(DEST_PATH),
        "count": len(destinations)
    }


tokens = []
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)

class TokenModel(BaseModel):
    token: str

@app.post("/register_token")
def register_token(data: TokenModel):
    tokens.append(data.token)
    print("등록된 토큰: ", data.token)
    return {"status":"received"}

def send_notification():
    message = messaging.Message(
        notification=messaging.Notification(
            title="알림 제목",
            body="알림 내용"
        ),
        token="사용자_FCM_토큰"
    )
    response = messaging.send(message)
    print("푸시 전송 성공:", response)

class PushRequest(BaseModel):
    title: str
    body: str

@app.post("/send_push")
def send_push(data: PushRequest):
    if not tokens:
        return {"error": "No token registered yet"}

    message = messaging.Message(
        notification=messaging.Notification(
            title=data.title,
            body=data.body
        ),
        token=tokens[-1]
    )
    response = messaging.send(message)
    print("푸시 전송 성공:", response)
    return {"status": "success", "response": response}

