import importlib



import torch.nn as nn
import importlib


#骨干整合模型
class BackboneCombinedModel(nn.Module):
    def __init__(self, model_config):
        super(BackboneCombinedModel, self).__init__()
        self.models = nn.ModuleList()
        for model_name, config in model_config.items():

            self.models.append(config)



    def forward(self, data):
        x,edge_index=data.x,data.edge_index
        for model in self.models:

            x = model(x,edge_index)
            # 添加其他逻辑，根据你的需要

        return  x



#头部整合模型
class HeadCombinedModel(nn.Module):
    def __init__(self, model_config):
        super(HeadCombinedModel, self).__init__()
        self.models = nn.ModuleList()
        for model_name, config in model_config.items():

            self.models.append(config)



    def forward(self, data):

        for model in self.models:

            data = model(data)
            # 添加其他逻辑，根据你的需要

        return  data
