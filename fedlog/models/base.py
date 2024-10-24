
import torch.nn as nn


class InputModel(nn.Module):
    
    def __init__(self, *args, **kwargs):
        super(InputModel, self).__init__()
        

class OuputModel(nn.Module):
    
    def __init__(self, *args, **kwargs):
        super(OuputModel, self).__init__(*args, **kwargs)
        
class MainModel(nn.Module):
    
    def __init__(self, *args, **kwargs):
        super(MainModel, self).__init__(*args, **kwargs)
        
        
class FedBaseModel(nn.Module):
    
    def __init__(self, input_model: InputModel, output_model: OuputModel, main_model: MainModel, *args, **kwargs):
        super(FedBaseModel, self).__init__(*args, **kwargs)
        self.input_model = input_model
        self.output_model = output_model
        self.main_model = main_model
        
    def forward(self, x):
        after_input = self.input_model(x)
        after_main = self.main_model(after_input)
        output = self.output_model(after_main)
        return output