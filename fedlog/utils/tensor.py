import torch
import base64
import pickle
import numpy as np
from pydantic import BaseModel


# 定义一个用于序列化/反序列化的模型
class TensorData(BaseModel):
    tensor: str  # Base64 序列化的 Tensor 数据
    shape: list  # Tensor 的形状


# Tensor 序列化为 JSON（Base64 格式）
def tensor_to_base64(tensor: torch.Tensor) -> str:
    # 将 PyTorch Tensor 转换为 NumPy 数组
    numpy_tensor = tensor.detach().numpy()
    # 将 NumPy 数组转换为字节流
    byte_tensor = numpy_tensor.tobytes()
    # 将字节流编码为 Base64 字符串
    base64_tensor = base64.b64encode(byte_tensor)
    return base64_tensor


# 将 Base64 解码并反序列化为 Tensor
def base64_to_tensor(base64_str: str, shape: list) -> torch.Tensor:
    # 将 Base64 字符串解码为字节流
    byte_tensor = base64.b64decode(base64_str)
    # 将字节流转换为 NumPy 数组
    numpy_tensor = np.frombuffer(byte_tensor, dtype=np.float32).reshape(shape)
    # 将 NumPy 数组转换为 PyTorch Tensor
    tensor = torch.from_numpy(numpy_tensor)
    return tensor


def model_to_base64(model: torch.nn.Module) -> str:
    return base64.b64encode(pickle.dumps(model.state_dict()))


def load_state_from_base64(base64_str: str, model: torch.nn.Module):
    model.load_state_dict(pickle.loads(base64.b64decode(base64_str)))
