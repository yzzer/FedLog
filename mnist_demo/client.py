import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader

# Client 端模型
class ClientModel(nn.Module):
    def __init__(self):
        super(ClientModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        return x

def client_process():
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'

    dist.init_process_group("gloo", rank=1, world_size=2)

    model = ClientModel()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # 使用 MNIST 数据集
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=False)

    for epoch in range(20):  # 模拟5个epoch
        for inputs, _ in train_loader:
            optimizer.zero_grad()

            # 前向传播
            output = model(inputs)

            # 发送中间激活值到 Server 端
            dist.send(tensor=output, dst=0)

            # 接收来自 Server 的梯度
            grad = torch.zeros_like(output)
            dist.recv(tensor=grad, src=0)

            # 反向传播
            output.backward(grad)
            optimizer.step()
            
    torch.save(model.state_dict(), 'client_model.pth')
    dist.destroy_process_group()

if __name__ == "__main__":
    client_process()
