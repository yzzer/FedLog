import torch
import base64
import pickle
import numpy as np
import io
from pydantic import BaseModel

buffer = None

# 定义一个用于序列化/反序列化的模型
class TensorData(BaseModel):
    tensor: bytes  # Base64 序列化的 Tensor 数据
    shape: list  # Tensor 的形状


def get_buffer() -> io.BytesIO:
    global buffer
    if buffer is None:
        buffer = io.BytesIO()
    return buffer


# Tensor 序列化为 JSON（Base64 格式）
def tensor_to_base64(tensor: torch.Tensor) -> bytes:
    ts = tensor.detach()
    bf = get_buffer()
    bf.seek(0)  # 移动指针到开头
    bf.truncate(0)  # 清空缓冲区
    torch.save(ts, bf, pickle_protocol=4)
    return bf.getvalue()


# 将 Base64 解码并反序列化为 Tensor
def base64_to_tensor(base64_str: bytes, shape: list) -> torch.Tensor:
    bf = get_buffer()
    bf.seek(0)  # 移动指针到开头
    bf.truncate(0)  # 清空缓冲区
    bf.write(base64_str)
    bf.seek(0)
    tensor = torch.load(bf, map_location="cpu")
    return tensor


def model_to_base64(model: torch.nn.Module) -> str:
    return base64.b64encode(pickle.dumps(model.state_dict()))


def load_state_from_base64(base64_str: str, model: torch.nn.Module):
    model.load_state_dict(pickle.loads(base64.b64decode(base64_str)))

def base64_to_state(base64_str: str) -> dict:
    return pickle.loads(base64.b64decode(base64_str))

if __name__ == "__main__":
    a = torch.rand(100, 100)
    b = tensor_to_base64(a)
    c = base64_to_tensor(b, a.shape)
    print(a)
    print(c)
    print(torch.allclose(a, c))