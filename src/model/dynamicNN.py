import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random

class CustomNeuron:
    def __init__(self, input_size):
        self.weights = torch.randn(input_size, requires_grad=True)
        self.bias = torch.randn(1, requires_grad=True)
        self.lifetime = random.randint(50, 100)  # 随机设定一个生命周期
        self.age = 0

    def forward(self, x):
        return x @ self.weights + self.bias

    def step(self):
        self.age += 1
        if self.age > self.lifetime:
            return False  # 返回False表示该神经元已经死亡
        return True

class CustomLayer(nn.Module):
    def __init__(self, input_size, initial_neurons):
        super(CustomLayer, self).__init__()
        self.neurons = [CustomNeuron(input_size) for _ in range(initial_neurons)]
    
    def forward(self, x):
        outputs = [neuron.forward(x) for neuron in self.neurons if neuron.step()]
        self.neurons = [neuron for neuron in self.neurons if neuron.age <= neuron.lifetime]  # 移除死亡神经元
        return torch.stack(outputs, dim=-1).sum(dim=-1)  # 将输出堆叠并求和

    def add_neuron(self, input_size):
        self.neurons.append(CustomNeuron(input_size))

    def prune_neuron(self):
        if self.neurons:
            self.neurons.pop(random.randint(0, len(self.neurons) - 1))

class DynamicNN(nn.Module):
    def __init__(self, input_size, initial_neurons, output_size):
        super(DynamicNN, self).__init__()
        self.custom_layer = CustomLayer(input_size, initial_neurons)
        self.output_layer = nn.Linear(initial_neurons, output_size)
    
    def forward(self, x):
        x = self.custom_layer(x)
        x = self.output_layer(x)
        return x

# 创建动态神经网络
input_size = 784
initial_neurons = 10
output_size = 10
dynamic_model = DynamicNN(input_size, initial_neurons, output_size)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(dynamic_model.parameters(), lr=0.001)

# 示例训练循环
def train_dynamic(model, train_loader, criterion, optimizer, epochs=5):
    for epoch in range(epochs):
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        # 动态调整神经元
        if epoch % 2 == 0:  # 每两个epoch增加一个神经元
            model.custom_layer.add_neuron(input_size)
        elif epoch % 3 == 0:  # 每三个epoch剪掉一个神经元
            model.custom_layer.prune_neuron()

# 训练示例（假设train_loader已经定义）
# train_dynamic(dynamic_model, train_loader, criterion, optimizer)
