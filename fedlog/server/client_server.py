"""fedserver 联邦学习客户端服务
负责从中央服务器获取模型并进行本地训练
"""
import sys
sys.path.append("..")

from config import init_config, get_config
from fastapi import FastAPI

from handler.demo import MnistClientApp, ServerInfo
from utils.tensor import TensorData

# 创建 FastAPI 应用实例
init_config()
app = FastAPI()

# 定义一个简单的 GET 路由
@app.get("/")
async def root():
    return {"message": "Welcome to client!"}

@app.post("/mnist/demo/forward/")
async def mnist_output_forward(input: TensorData):
    return MnistClientApp.get_instance().forward_output(input)


@app.get("/mnist/demo/start/")
async def start_job():
    return MnistClientApp.get_instance().start_sl_job()


# 启动命令（如果以脚本形式运行该文件）
if __name__ == "__main__":
    conf = get_config()
    
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=conf.server_config.client_port)
