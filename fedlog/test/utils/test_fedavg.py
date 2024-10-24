import torch.nn as nn
import torch
import unittest

import sys
sys.path.append("../..")

from utils.fedavg import fedavg

# 定义一个简单的全连接网络
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # 展平
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

# 单元测试
class TestFedAvg(unittest.TestCase):

    def setUp(self):
        # 初始化全局模型
        self.global_model = SimpleNN()
        
        # 创建客户端模型的 state_dict
        self.client_state_dicts = []
        for i in range(3):  # 假设有三个客户端
            local_model = SimpleNN()
            # 随机初始化权重
            for param in local_model.parameters():
                param.data = torch.randn_like(param)
            self.client_state_dicts.append(local_model.state_dict())
        
        # 设置客户端权重
        self.client_weights = [1, 1, 1]

    def test_fedavg(self):
        # 记录全局模型的状态
        from copy import deepcopy
        initial_global_state_dict = deepcopy(self.global_model.state_dict())
        
        # 使用 FedAvg 更新全局模型
        fedavg(self.global_model, self.client_state_dicts, self.client_weights)
        
        # 检查更新后的全局模型参数是否符合预期
        updated_global_state_dict = self.global_model.state_dict()

        # 验证参数是否更新
        for param_name in initial_global_state_dict:
            initial_value = initial_global_state_dict[param_name].clone()  # 克隆初始值以便比较
            updated_value = updated_global_state_dict[param_name]
            
            self.assertFalse(torch.equal(initial_value, updated_value), 
                             f"Global model parameter '{param_name}' should be updated.")

        # 验证全局模型参数的加权平均是否正确
        for param_name in updated_global_state_dict:
            weighted_sum = sum(state_dict[param_name] * weight for state_dict, weight in zip(self.client_state_dicts, self.client_weights))
            expected_value = weighted_sum / sum(self.client_weights)
            self.assertTrue(torch.allclose(updated_global_state_dict[param_name], expected_value),
                            f"Global model parameter '{param_name}' is not equal to the expected value after FedAvg.")

if __name__ == '__main__':
    unittest.main()
