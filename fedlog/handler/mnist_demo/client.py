import logging
import requests
import torch
import torch.nn as nn

from models.mnist_demo import MnistModel
from handler.mnist_demo.bean import ServerInfo, FedModel
from config import get_config
from utils.tensor import (
    TensorData,
    tensor_to_base64,
    base64_to_tensor,
    load_state_from_base64,
    base64_to_state,
    model_to_base64
)
from utils.monitor import Report

conn_session = None

def get_session():
    global conn_session
    if conn_session is None:
        conn_session = requests.Session()
    return conn_session


class TrainService:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.address = f"http://{self.host}:{self.port}"
        self.session = get_session()

    def ping(self):
        url = self.address + "/ping"
        resp = self.session.get(url)
        assert resp.status_code == 200

    def forward(self, input):
        """
        传递输入, 并从远程训练服务器返回梯度
        """
        tensor_data = tensor_to_base64(input)
        url = self.address + "/mnist/demo/forward"
        resp = self.session.post(url=url, data=tensor_data)
        assert resp.status_code == 200
        grad_tensor_data = resp.content
        grad = base64_to_tensor(grad_tensor_data, None)
        return grad

    def load_model(self, model: nn.Module):
        url = self.address + "/mnist/demo/model"
        resp = self.session.get(url)
        assert resp.status_code == 200
        load_state_from_base64(resp.text, model)
        
    def send_model(self, model: MnistModel):
        model_do = FedModel(
            main_model_base64=model_to_base64(model.main_model),
            type="sl"
        )
        url = self.address + "/mnist/demo/send_model"
        resp = self.session.post(url=url, data=model_do.model_dump_json())
        assert resp.status_code == 200
    
    def get_model(self):
        url = self.address + "/mnist/demo/model"
        resp = self.session.get(url)
        assert resp.status_code == 200
        return base64_to_state(resp.text)


class ClientSevice:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.address = f"http://{self.host}:{self.port}"
        self.session = get_session()
        

    def forward(self, input: torch.Tensor):
        """
        传递输入, 并从客户端服务器返回梯度
        """
        tensor_data = tensor_to_base64(input)
        url = self.address + "/mnist/demo/forward"
        resp = self.session.post(url=url, data=tensor_data)
        assert resp.status_code == 200
        grad_tensor_data = resp.content
        grad = base64_to_tensor(grad_tensor_data, None)
        return grad
    
    
    def send_model(self, model: MnistModel, mode: str):
        if mode == "fl":
            model_do = FedModel(
                main_model_base64=model_to_base64(model),
                type=mode
            )
        else:
            model_do = FedModel(
                input_model_base64=model_to_base64(model.input_model),
                output_model_base64=model_to_base64(model.output_model),
                type=mode
            )
        url = self.address + "/mnist/demo/send_model"
        resp = self.session.post(url=url, data=model_do.model_dump_json())
        assert resp.status_code == 200

        
    
    def start_job(self, mode, local_epoch=3, global_epoch=10, batch=512):
        url = self.address + "/mnist/demo/start"
        resp = self.session.get(url=url, params={"mode": mode, "local_epoch": local_epoch, "global_epoch": global_epoch, "batch": batch})
        assert resp.status_code == 200
        return resp.json()
    
    def get_report(self) -> Report:
        url = self.address + "/report"
        resp = self.session.get(url)
        assert resp.status_code == 200
        return Report(**resp.json())


class MnistFedService:   
    def __init__(self):
        conf = get_config()
        self.host = conf.fedserver.host
        self.port = conf.fedserver.port
        self.address = f"http://{self.host}:{self.port}"
        self.session = get_session()
        
    def collect_model(self, model: MnistModel, type: str):
        if type == "sl":
            model_do = FedModel(
                input_model_base64=model_to_base64(model.input_model),
                output_model_base64=model_to_base64(model.output_model),
                type=type,
            )
        else:
            model_do = FedModel(
                main_model_base64=model_to_base64(model),
                type=type
            )
        url = self.address + "/mnist/demo/collect_model"
        resp = self.session.post(url=url, data=model_do.model_dump_json())
        assert resp.status_code == 200