import os
from pydantic import BaseModel

class Config(BaseModel):
    config_path: str
    pids: list = []
    log_dir: str = "logs"
    client_num: int = 10
    mode: str = "start"  # start / stop
    part: str = "client"
    


def get_config() -> Config:
    import argparse

    # 创建解析器对象
    parser = argparse.ArgumentParser(description="命令行参数示例")

    # 添加命令行参数
    parser.add_argument("--config", required=False, default="")
    parser.add_argument("--log-dir", default="logs", required=False)
    parser.add_argument("--client-num", type=int, default=3, required=False)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--part", required=False)
    

    # 解析命令行参数
    args = parser.parse_args()
    
    return Config(config_path=args.config, log_dir=args.log_dir, client_num=args.client_num, mode=args.mode, part=args.part)


def start_client_nohup(config_path: str, log_dir: str, client_id: int):
    import subprocess
    import sys

    command = f"nohup {sys.executable} client_server.py --config {config_path} --id {client_id} > {log_dir}/client-{client_id}.log 2>&1 & echo $!"
    # 使用 subprocess.Popen 启动服务并将其置于后台
    result = subprocess.check_output(command, shell=True)
    return int(result.decode().strip())

def start_trainer_nohup(config_path: str, log_dir: str, client_id: int):
    import subprocess
    import sys

    command = f"nohup {sys.executable} train_server.py --config {config_path}  --id {client_id} > {log_dir}/trainer-{client_id}.log 2>&1 & echo $!"
    # 使用 subprocess.Popen 启动服务并将其置于后台
    result = subprocess.check_output(command, shell=True)
    return int(result.decode().strip())


def main():
    conf = get_config()
    import json

    if conf.mode == "start":
        if os.path.exists(conf.log_dir):
            import shutil
            shutil.rmtree(conf.log_dir)
        os.makedirs(conf.log_dir)
        
        for i in range(conf.client_num):
            if conf.part == "client":
                pid = start_client_nohup(conf.config_path, conf.log_dir, i)
                conf.pids.append(pid)
                print(f"start client {i} pid: {pid}")
            else:
                pid = start_trainer_nohup(conf.config_path, conf.log_dir, i)
                conf.pids.append(pid)
                print(f"start trainer {i} pid: {pid}")
        json.dump(conf.model_dump(), open("./cache.json", "w"))
    else:
        json_conf = json.load(open("./cache.json", "r"))
        conf = Config(**json_conf)
        for pid in conf.pids:
            os.kill(pid, 9)
            print(f"kill client {pid}")
        conf.pids = []
        json.dump(conf.model_dump(), open("./cache.json", "w"))
    
if __name__ == "__main__":
    main()