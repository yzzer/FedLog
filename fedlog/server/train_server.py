"""fedserver 联邦学习训练服务
承接部分模型层，承担主体模型的训练，减轻客户端的负载
"""

import sys

sys.path.append("..")

from fastapi import FastAPI, Request
from config import init_config, get_config
from handler.mnist_demo.handler import MnistTrainServerApp
from handler.mnist_demo.bean import FedModel
from utils.tensor import TensorData

# 创建 FastAPI 应用实例
init_config()
app = FastAPI()


# 定义一个简单的 GET 路由
@app.get("/")
async def root():
    return {"message": "Welcome to train server!"}


@app.get("/ping")
async def ping():
    return {"message": "ping successfully"}


@app.get("/mnist/demo/model")
async def mnist_model():
    return MnistTrainServerApp.get_instance().get_model()

@app.post("/mnist/demo/send_model")
async def send_mnist_model(model: FedModel):
    return MnistTrainServerApp.get_instance().send_model(model)


@app.post("/mnist/demo/forward")
async def mnist_forward(req: Request):
    return MnistTrainServerApp.get_instance().forward(req.body())


# 启动命令（如果以脚本形式运行该文件）
if __name__ == "__main__":
    conf = get_config()
    
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=conf.server_config.server_port, log_level="info", timeout_keep_alive=300)
