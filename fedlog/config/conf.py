from pydantic import BaseModel
from typing import Dict
from config.log import init_log_config

_config_instance = None

class FedServer(BaseModel):
    host: str = "127.0.0.1"
    port: int = 16844

class ServerGroup(BaseModel):
    client_host: str
    client_port: int
    server_host: str
    server_port: int
    id: int
    weight: float = 1.0


class Config(BaseModel):
    fedserver: FedServer = FedServer()
    server_id: int = 0
    server_config: ServerGroup = None
    server_groups: Dict[int, ServerGroup] = {}

    
    

def get_config() -> Config:
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance
    
    
    
def parse_server_config():
    import argparse

    # 创建解析器对象
    parser = argparse.ArgumentParser(description="命令行参数示例")

    # 添加命令行参数
    parser.add_argument("--config", required=True)
    parser.add_argument("--id", required=True)
    
    args = parser.parse_args()
    config_path = args.config
    server_id = int(args.id)
    
    import yaml
    # 读取 YAML 文件并将其转换为字典
    with open(config_path, 'r') as file:
        data = yaml.safe_load(file)
        conf_instance = get_config()
        conf_instance.server_id = server_id
        
        # fedserver
        conf_instance.fedserver.host = data['fedserver']['host']
        conf_instance.fedserver.port = data['fedserver']['port']
        
        for item_id, items in data['servers'].items():
            server_group = ServerGroup(client_host=items["client_host"],
                                       client_port=items["client_port"], 
                                       server_host=items["server_host"],
                                       server_port=items["server_port"],
                                       weight=items["weight"],
                                       id=item_id)
            conf_instance.server_groups[item_id] = server_group
        conf_instance.server_config = conf_instance.server_groups[server_id]
    

def init_config():
    init_log_config()
    parse_server_config()
    
    
if __name__ == "__main__":
    parse_server_config()