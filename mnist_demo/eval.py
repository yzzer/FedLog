import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader

from client import ClientModel
from server import ServerModel

def evaluate(model, data_loader):
    model.eval()  # 设置模型为评估模式
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = total_correct / total_samples
    return accuracy




# Server 端模型
class MergedModel(nn.Module):
    def __init__(self):
        super(MergedModel, self).__init__()
        self.client_model = ClientModel()
        self.server_model = ServerModel()
    
    def forward(self, x):
        return self.server_model(self.client_model(x))
    
    def load(self, client_path, server_path):
        self.client_model.load_state_dict(torch.load(client_path, weights_only=True))
        self.server_model.load_state_dict(torch.load(server_path, weights_only=True))
    
def eval():
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    model = MergedModel()
    model.load('client_model.pth', 'server_model.pth')
    acc = evaluate(model, test_loader)
    print(f"accuracy = {acc:.4f}")
    
if __name__ == "__main__":
    eval() 