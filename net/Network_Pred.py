import torch
from imports.vit import *
import torch.nn.functional as F
import torch.nn as nn



class Network_phenotype_prediction(torch.nn.Module):
    def __init__(self, indim, outdim):
        '''

        :param indim: (int) node feature dimension
        :param ratio: (float) pooling ratio in (0,1)
        :param nclass: (int)  number of classes
        :param k: (int) number of communities
        :param R: (int) number of ROIs
        '''
        super(Network_phenotype_prediction, self).__init__()
        self.indim = indim
        self.outdim = outdim

        self.input_layer = nn.Linear(self.indim, self.outdim)







    def forward(self, x):

        regression_output = self.input_layer(x)
        regression_params = self.input_layer

        return regression_output, regression_params




class RegressionNetwork(nn.Module):
    def __init__(self, indim, outdim):
        super(RegressionNetwork, self).__init__()

        self.indim = indim
        self.outdim = outdim
        self.input_layer = nn.Linear(self.indim, self.outdim, bias=False)

    def forward(self, x):

        regression_output = self.input_layer(x)
        regression_params = self.input_layer

        return regression_output  , regression_params


# 创建多任务回归预测网络
class MultiTaskRegressionNetwork(nn.Module):
    def __init__(self, num_tasks):
        super(MultiTaskRegressionNetwork, self).__init__()

        self.num_tasks = num_tasks
        self.regression_networks = nn.ModuleList([RegressionNetwork(1024,1) for _ in range(num_tasks)])

    def forward(self, x):
        regression_outputs = []
        regression_params = []

        for i in range(self.num_tasks):
            output , params= self.regression_networks[i](x[: ,i,:])  #

            regression_outputs.append(output)
            regression_params.append(params.weight.data)

        return regression_outputs , regression_params