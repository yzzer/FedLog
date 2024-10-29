import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader

from models.base import InputModel, OuputModel, FedBaseModel

class MnistInputModel(InputModel):
    def __init__(self):
        super(MnistInputModel, self).__init__()
        self.input_layer = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=1)  # 输出: 16x28x28
        self.conv1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)  # 输入: 16x28x28, 输出: 32x28x28
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        x = self.pool(nn.ReLU()(self.input_layer(x)))
        x = self.pool(nn.ReLU()(self.conv1(x)))
        return x
        # return nn.ReLU()(self.input_layer(x))
    

class MnistOutputModel(OuputModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        return self.fc2(x)


class MnistMainModel(nn.Module):
    def __init__(self):
        super(MnistMainModel, self).__init__()
         # 卷积层1
        # self.conv1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)  # 输入: 16x28x28, 输出: 32x28x28
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 池化层
        
        # 卷积层2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)  # 输入: 32x28x28, 输出: 64x28x28
        
        # 自适应平均池化层
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))  # 输出: 64x7x7
        
        # 全连接层
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # 输入: 64*7*7, 输出: 128
        
    def forward(self, x):
        # x = self.pool(nn.ReLU()(self.conv1(x)))  # 第一个卷积层 + 激活 + 池化
        x = self.pool(nn.ReLU()(self.conv2(x)))  # 第二个卷积层 + 激活 + 池化
        
        x = self.adaptive_pool(x)  # 自适应平均池化层
        x = x.view(-1, 64 * 7 * 7)  # 展平
        x = nn.ReLU()(self.fc1(x))  # 第一个全连接层 + 激活
        return x
    
    
class MnistModel(FedBaseModel):
    
    def __init__(self):
        super().__init__(MnistInputModel(), MnistOutputModel(), MnistMainModel())
        
def evaluate(model, input_tensor, target):
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        accuracy = (predicted == target).float().mean().item()  # 计算准确率
        return predicted, accuracy


def train_and_eval():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 超参数
    batch_size = 512
    learning_rate = 0.001
    num_epochs = 15

    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # 标准化
    ])

    # 加载 MNIST 数据集
    train_dataset = datasets.MNIST(root='../../data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='../../data', train=False, download=True, transform=transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型、损失函数和优化器
    model = MnistModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            # output = model.input_model(images)
            # print(output.shape)
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # 测试模型
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f'Accuracy of the model on the test images: {100 * correct / total:.2f}%')

if __name__ == "__main__":
    train_and_eval()