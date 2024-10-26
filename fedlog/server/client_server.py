"""fedserver 联邦学习客户端服务
负责从中央服务器获取模型并进行本地训练
"""
import sys
sys.path.append("..")

from config import init_config, get_config
from fastapi import FastAPI

from handler.mnist_demo.handler import MnistClientApp
from handler.mnist_demo.bean import FedModel
from utils.tensor import TensorData
from utils.monitor import ProcessMonitor, Report

# 创建 FastAPI 应用实例
init_config()
app = FastAPI()

# 定义一个简单的 GET 路由
@app.get("/")
async def root():
    return {"message": "Welcome to client!"}

@app.post("/mnist/demo/forward")
async def mnist_output_forward(input: TensorData):
    return MnistClientApp.get_instance().forward_output(input)


@app.get("/mnist/demo/start")
async def start_mnist_job(mode: str = "fl", local_epoch: int = 5, batch: int = 1024):
    if mode == "fl":
        res =  MnistClientApp.get_instance().start_fl_job(local_epoch, batch)
    else:
        res = MnistClientApp.get_instance().start_sl_job(local_epoch, batch)
    ProcessMonitor.get_instance().start_monitor()
    return res
@app.get("/report")
async def get_report():
    return ProcessMonitor.get_instance().stop_monitor()


@app.get("/mnist/demo/get_model")
async def get_mnist_model(mode: str = "fl"):
    return MnistClientApp.get_instance().get_model(mode)


@app.post("/mnist/demo/send_model")
async def send_mnist_model(model: FedModel):
    return MnistClientApp.get_instance().send_model(model)


# 启动命令（如果以脚本形式运行该文件）
if __name__ == "__main__":
    conf = get_config()
    
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=conf.server_config.client_port, log_level="info", timeout_keep_alive=300)
