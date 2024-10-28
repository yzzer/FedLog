import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from handler.mnist_demo.bean import ServerInfo, FedModel
from handler.mnist_demo.client import *
from threading import Lock
from utils.fedavg import fedavg

from models.mnist_demo import MnistModel
from config import get_config
from concurrent.futures import ThreadPoolExecutor


from utils.tensor import (
    TensorData,
    tensor_to_base64,
    base64_to_tensor,
    load_state_from_base64,
    model_to_base64,
    base64_to_state
)



class MnistTrainServerApp:
    app = None

    def __init__(self):
        self.model: MnistModel = MnistModel()
        self.client_server: ServerInfo = ServerInfo(
            host=get_config().server_config.client_host,
            port=get_config().server_config.client_port,
        )

        self.optimizer: optim.SGD = None
        self.criterion = nn.CrossEntropyLoss()

        self.client: ClientSevice = ClientSevice(
            host=get_config().server_config.client_host,
            port=get_config().server_config.client_port,
        )
        self.optimizer = optim.Adam(self.model.main_model.parameters(), lr=0.001)

    @staticmethod
    def get_instance():
        if MnistTrainServerApp.app is None:
            MnistTrainServerApp.app = MnistTrainServerApp()
        return MnistTrainServerApp.app

    def forward(self, tensor: TensorData) -> bytes:
        self.optimizer.zero_grad()
        
        # get grad
        input = base64_to_tensor(tensor, shape=None)
        input.requires_grad_()
        output = self.model.main_model(input)
        grad = self.client.forward(output)

        # backward
        output.backward(grad)
        self.optimizer.step()
        grad = input.grad
        return tensor_to_base64(grad)

    def get_model(self) -> str:
        return model_to_base64(self.model.main_model)
    
    def send_model(self, model: FedModel):
        load_state_from_base64(model.main_model_base64, self.model.main_model)


class MnistFedApp:
    app = None
    
    def __init__(self):
        self.local_main_models = []
        self.local_input_models = []
        self.local_output_models = []
        self.mode = None
        self.now_global_epoch = 0
        self.target_global_epoch = 10
        
        self.global_model = MnistModel()
        
        self.lock = Lock()
        self.start_time = 0
        
        transform = transforms.Compose([transforms.ToTensor()])
        test_dataset = datasets.MNIST(
            root="../../data", train=False, download=True, transform=transform
        )
        
        self.test_loader = DataLoader(
            dataset=test_dataset, batch_size=512, shuffle=False
        )
        
        self.clients = [
            ClientSevice(group.client_host, group.client_port) for group in get_config().server_groups.values()
        ]
        self.trainers = [
            TrainService(group.server_host, group.server_port) for group in get_config().server_groups.values()
        ]
        self.weights = [
            group.weight for group in get_config().server_groups.values()
        ]
        
        self.workers = ThreadPoolExecutor(max_workers=20)
        
    @staticmethod
    def get_instance():
        if MnistFedApp.app is None:
            MnistFedApp.app = MnistFedApp()
        return MnistFedApp.app
    
    def _eval(self):
        self.global_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                outputs = self.global_model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        logging.info(
            f"gloabl_epoch={self.now_global_epoch} Accuracy of the model on the test images: {100 * correct / total:.2f}%"
        )
    
    def _collect_main_models(self):
        self.local_main_models = [None] * len(self.clients)
        def bc_main(train: TrainService, idx: int):
            self.local_main_models[idx] = train.get_model()
            return True
        if not all(self.workers.map(bc_main, self.trainers, range(len(self.trainers)))):
            raise Exception("get model failed")
        logging.info(f"collected models from trainers")
    
    def _broadcast_model(self):
        if self.mode == "fl":
            def bc_fl(client: ClientSevice):
                client.send_model(self.global_model, self.mode)
                return True
            if not all(self.workers.map(bc_fl, self.clients)):
                raise Exception("send model failed")
        else:
            def bc_sl(client: ClientSevice, trainer: TrainService):
                client.send_model(self.global_model, self.mode)
                trainer.send_model(self.global_model)    
                return True
            if not all(self.workers.map(bc_sl, self.clients, self.trainers)):
                raise Exception("send model failed")
        logging.info(f"send models to clients in {self.mode} mode")
        
    def _collect_report(self):
        def get_report(client: ClientSevice):
            return client.get_report()
        reports = self.workers.map(get_report, self.clients)
        from utils.monitor import MonitorReportor
        return MonitorReportor.merge_report(list(reports))
        
            
    def start_job(self, mode="fl", local_epoch=3, global_epoch=10, batch=512):
        self.mode = mode
        
        import time
        self.start_time = time.time()
        self.now_global_epoch = 0
        self.target_global_epoch = global_epoch
        self.global_model = MnistModel()
        
        def start_job(client: ClientSevice):
            client.start_job(mode, local_epoch, global_epoch, batch=batch)
            return True
        if not all(self.workers.map(start_job, self.clients)):
            raise Exception("start job failed")
        logging.info("send job to clients")
        self._broadcast_model()   
        return {
            "status": "started",
            "client_num": len(self.clients),
            "global_epoch": global_epoch,
            "local_epoch": local_epoch,
            "mode": mode,
        }    
      
    def collect_model(self, model: FedModel):
        local_input_model = None
        local_output_model = None
        local_main_model = None
        
        if model.input_model_base64 != "":
            local_input_model = base64_to_state(model.input_model_base64)
        if model.output_model_base64 != "":
            local_output_model = base64_to_state(model.output_model_base64)
        if model.main_model_base64 != "":
            local_main_model = base64_to_state(model.main_model_base64)
        
        def merge(): 
            with self.lock:
                if model.type == "fl":
                    self.local_main_models.append(local_main_model)
                else:
                    if local_input_model is not None:
                        self.local_input_models.append(local_input_model)
                    if local_output_model is not None:
                        self.local_output_models.append(local_output_model)
                    if local_main_model is not None:
                        self.local_main_models.append(local_main_model)
                finish_global_epoch = False
                if self.mode == "fl" and len(self.local_main_models) == len(self.clients):
                    fedavg(self.global_model, self.local_main_models, self.weights)
                    finish_global_epoch = True
                elif self.mode == "sl" and len(self.local_input_models) == len(self.clients) \
                    and len(self.local_output_models) == len(self.clients):
                    self._collect_main_models()
                    fedavg(self.global_model.input_model, self.local_input_models, self.weights)
                    fedavg(self.global_model.output_model, self.local_output_models, self.weights)
                    fedavg(self.global_model.main_model, self.local_main_models, self.weights)
                    finish_global_epoch = True
                    
                if finish_global_epoch:
                    self.now_global_epoch += 1
                    self._eval()
                    if self.now_global_epoch < self.target_global_epoch:
                        self._broadcast_model()
                    else:
                        import time
                        logging.info(f"job cost time: {time.time() - self.start_time} s")
                        logging.info(f"client monitor report: {self._collect_report()}")
                    self.local_main_models.clear()
                    self.local_input_models.clear()
                    self.local_output_models.clear()
                    self.global_model = MnistModel()
                    
                    
        import threading
        thread = threading.Thread(target=merge)
        thread.start()


class MnistClientApp:
    app = None

    def __init__(self):
        self.model: MnistModel = None
        self.train_server: ServerInfo = ServerInfo()
        self.client_server: ServerInfo = ServerInfo()

        self.input_optimizer: optim.Adam = None
        self.output_optimizer: optim.Adam = None
        self.optimizer: optim.Adam = None

        self.criterion = nn.CrossEntropyLoss()

        self.trainer: TrainService = None
        self.fed: MnistFedService = MnistFedService()

        # dataset
        self.train_loader: DataLoader = None
        self.test_loader: DataLoader = None
        self.label_cache = None

        self.epoch = 20
        self.batch = 512
        self.now_epoch = 0

    @staticmethod
    def get_instance():
        if MnistClientApp.app is None:
            MnistClientApp.app = MnistClientApp()
        return MnistClientApp.app
            
    def prepare_env(self, epoch=5, batch=512):
        conf = get_config()
        self.train_server.host = conf.server_config.server_host
        self.train_server.port = conf.server_config.server_port
        self.client_server.host = conf.server_config.client_host
        self.client_server.port = conf.server_config.client_port
        
        # dataset
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = datasets.MNIST(
            root="../../data", train=True, download=True, transform=transform
        )
        self.train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
        
        self.epoch = epoch
        self.batch = batch
    
    def send_model(self, model_do: FedModel):
        # 通过广播模型触发训练
        if model_do.type == "fl":
            load_state_from_base64(model_do.main_model_base64, self.model)
        else:
            load_state_from_base64(model_do.input_model_base64, self.model.input_model)
            load_state_from_base64(model_do.output_model_base64, self.model.output_model)
                 
        # 后台线程启动训练
        import threading

        thread = threading.Thread(target=self.start_fl_train if model_do.type == "fl" else self.start_sl_train)
        thread.start()
        return {
            "status": "started",
        }
        
    def get_model(self, type):
        if type == "fl":
            return FedModel(model_base64=model_to_base64(self.model), type=type)
        else:
            return FedModel(
                input_model_base64=model_to_base64(self.model.input_model),
                output_model_base64=model_to_base64(self.model.output_model))
    
    def start_fl_train(self):
        import time
        start_time = time.time()
        for epoch in range(self.epoch):
            for inputs, labels in self.train_loader:
                self.optimizer.zero_grad()
                output = self.model(inputs)
                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()
            logging.info(f"loss = {loss.item():.4f} in epoch {epoch}")

        self.fed.collect_model(self.model, "fl")
        end_time = time.time()
        logging.info("train time: {}".format(end_time - start_time))
        import gc
        gc.collect()
    
    def start_fl_job(self, local_epoch=5, batch=512):
        self.prepare_env()   
        self.epoch = local_epoch
        self.batch = batch
        self.model = MnistModel()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        logging.info("fl env prepared")
        
    def start_sl_train(self):
        import time
        start_time = time.time()
        for epoch in range(self.epoch):
            self.now_epoch = epoch
            for inputs, labels in self.train_loader:
                self.label_cache = labels
                self.input_optimizer.zero_grad()

                output = self.model.input_model(inputs)

                grad = self.trainer.forward(output)
                output.backward(grad)
                self.input_optimizer.step()
        self.fed.collect_model(self.model, "sl")
        end_time = time.time()
        logging.info("train time: {}".format(end_time - start_time))  
        import gc
        gc.collect()     

    def start_sl_job(self, local_epoch=5, batch=512) -> dict:
        self.prepare_env(epoch=local_epoch, batch=batch)
        
        # 将client的信息发送给server
        server = TrainService(self.train_server.host, self.train_server.port)
        server.ping()
        self.trainer = server
        self.model = MnistModel()
        
        del self.model.main_model
        import gc
        gc.collect()
        
        # train
        self.input_optimizer = optim.Adam(self.model.input_model.parameters(), lr=0.001)
        self.output_optimizer = optim.Adam(self.model.output_model.parameters(), lr=0.001)

        logging.info("sl env prepared")

    def forward_output(self, input: bytes) -> TensorData:
        self.output_optimizer.zero_grad()
        input = base64_to_tensor(input, None)
        input.requires_grad_()

        output = self.model.output_model(input)
        loss = self.criterion(output, self.label_cache)

        # backward
        loss.backward()
        self.output_optimizer.step()

        grad = input.grad
        grad = tensor_to_base64(grad)

        logging.info(f"epoch = {self.now_epoch} loss = {loss.item():.4f}")
        return grad
