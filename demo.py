import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset

# Client 端模型
class ClientModel(nn.Module):
    def __init__(self):
        super(ClientModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        return x

# Server 端模型
class ServerModel(nn.Module):
    def __init__(self):
        super(ServerModel, self).__init__()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = torch.relu(self.conv2(x))
        x = x.view(-1, 64 * 14 * 14)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Client 端执行前向传播并传递中间激活值
def run_client(client_model, client_optimizer, pipe, dataloader):
    for inputs, _ in dataloader:
        client_optimizer.zero_grad()
        # Client 端前向传播
        client_output = client_model(inputs)
        # 将输出传递给 Server 端
        pipe.send(client_output)
        # 接收来自 Server 端的梯度
        client_grad = pipe.recv()
        # 将接收到的梯度赋值给 Client 端输出
        client_output.backward(client_grad)
        # 更新 Client 端的参数
        client_optimizer.step()

# Server 端接收激活值，计算损失并回传梯度
def run_server(server_model, server_optimizer, criterion, pipe, labels):
    while True:
        server_optimizer.zero_grad()
        # 接收来自 Client 端的输出（即中间激活值）
        client_output = pipe.recv()
        # Server 端继续前向传播
        server_output = server_model(client_output)
        # 计算损失
        loss = criterion(server_output, labels)
        # 反向传播
        loss.backward()
        # 将 Server 端传递回 Client 端的梯度发回 Client
        pipe.send(client_output.grad)
        # 更新 Server 端的参数
        server_optimizer.step()
        print(f"Loss: {loss.item():.4f}")

# 主进程，用于启动 Client 和 Server 端进程
def main():
    # 假数据集
    data = torch.randn(100, 1, 28, 28)  # 100 个样本, 每个样本 1个通道，28x28图像
    labels = torch.randint(0, 10, (100,))  # 100 个标签（0-9）
    dataset = TensorDataset(data, labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 创建 Client 和 Server 模型
    client_model = ClientModel()
    server_model = ServerModel()

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    client_optimizer = optim.SGD(client_model.parameters(), lr=0.01)
    server_optimizer = optim.SGD(server_model.parameters(), lr=0.01)

    # 使用 Pipe 进行进程间通信
    parent_conn, child_conn = mp.Pipe()

    # 启动 Client 和 Server 进程
    client_process = mp.Process(target=run_client, args=(client_model, client_optimizer, parent_conn, dataloader))
    server_process = mp.Process(target=run_server, args=(server_model, server_optimizer, criterion, child_conn, labels))

    client_process.start()
    server_process.start()

    client_process.join()
    server_process.join()

if __name__ == "__main__":
    mp.set_start_method('fork')  # Windows 下需要用 'spawn'，Linux 下可用 'fork'
    main()
