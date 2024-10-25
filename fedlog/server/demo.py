"""fedserver 联邦学习客户端服务
负责从中央服务器获取模型并进行本地训练
"""
import sys
sys.path.append("..")
from fastapi import FastAPI

# 创建 FastAPI 应用实例
app = FastAPI()

# 定义一个简单的 GET 路由
@app.get("/")
async def root():
    return {"message": "Welcome to FastAPI!"}

# 定义一个带路径参数的路由
@app.get("/hello/{name}")
async def greet_user(name: str):
    return {"message": f"Hello, {name}!"}

# 定义一个带查询参数的路由
@app.get("/items/")
async def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}

# 启动命令（如果以脚本形式运行该文件）
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
