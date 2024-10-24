import torch

# FedAvg 算法实现
def fedavg(global_model, client_state_dicts, client_weights):
    # 计算全局模型参数的平均值
    with torch.no_grad():
        global_state_dict = global_model.state_dict()
        for param in global_state_dict:
            global_state_dict[param].zero_()

        for state_dict, weight in zip(client_state_dicts, client_weights):
            for param_name, param_value in state_dict.items():
                global_state_dict[param_name].add_(param_value.data * weight)

        for param_name in global_state_dict:
            global_state_dict[param_name].div_(sum(client_weights))

        global_model.load_state_dict(global_state_dict)
