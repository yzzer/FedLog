import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader

# Server 端模型
class ServerModel(nn.Module):
    def __init__(self):
        super(ServerModel, self).__init__()
        self.fc1 = nn.Linear(32 * 14 * 14, 10)  # 假设输入是经过Client处理的32通道14x14图像
    
    def forward(self, x):
        x = x.view(-1, 32 * 14 * 14)  # 展开
        x = self.fc1(x)
        return x

def evaluate(model, input_tensor, target):
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        accuracy = (predicted == target).float().mean().item()  # 计算准确率
        return predicted, accuracy

def server_process():
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'

    dist.init_process_group("gloo", rank=0, world_size=2)

    model = ServerModel()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失

    # 使用 MNIST 数据集
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=False)

    for epoch in range(20):  # 模拟5个epoch
        for inputs, labels in train_loader:  # 使用训练集的标签
            # 接收来自 Client 的中间激活值
            input_tensor = torch.zeros(len(inputs), 32, 14, 14, requires_grad=True)  # 假设输入大小
            dist.recv(tensor=input_tensor, src=1)

            # 前向传播
            output = model(input_tensor)
            loss = criterion(output, labels)  # 使用训练集标签计算损失

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 将梯度发送回 Client
            if input_tensor.grad is not None:
                dist.send(tensor=input_tensor.grad, dst=1)

            # 评估
            predictions, accuracy = evaluate(model, input_tensor, labels)  # 使用真实标签
            print(f'Epoch {epoch + 1}, Predictions: {predictions}, Accuracy: {accuracy:.4f}')
            
    torch.save(model.state_dict(), 'server_model.pth')
    dist.destroy_process_group()

if __name__ == "__main__":
    server_process()
