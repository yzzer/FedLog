"""fedserver 联邦学习参数汇总服务
从各客户端收集参数，通过联邦学习算法合并模型并分发
"""
import sys
sys.path.append("..")

from config import get_config, init_config
from fastapi import FastAPI

# 创建 FastAPI 应用实例
init_config()
app = FastAPI()

# 定义一个简单的 GET 路由
@app.get("/")
async def root():
    return {"message": "Welcome to fed server!"}


@app.get("/ping")
async def ping():
    return {"message": "ping successfully"}


# 启动命令（如果以脚本形式运行该文件）
if __name__ == "__main__":
    
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=get_config().fedserver.port)
