import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random

class CustomNeuron:
    def __init__(self, input_size, index:int) -> None:
        self.weights = torch.randn(input_size, requires_grad=True) # 树突 TODO need to dynamic
        self.bias = torch.randn(1, requires_grad=True)
        self.lifetime = random.randint(50, 100)  # 随机设定一个生命周期
        self.index = index
        self.age = 0

    def destroy(self) -> int :
        return self.index
    
    def copy(self) -> 'CustomNeuron':
        newNeuron:CustomNeuron = CustomNeuron.__init__()
        newNeuron.weights = self.weights
        newNeuron.bias = self.bias
        newNeuron.index = self.index
        self.age += 2
        newNeuron.age = self.age
        return newNeuron

    def forward(self, x):
        return x @ self.weights + self.bias

    def step(self) -> Intger:
        self.age += 1
        if self.age > self.lifetime:
            return self.destroy() 
        return -1 # -1 is meaning its alive
    
