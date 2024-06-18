import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DynamicNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DynamicNN, self).__init__()
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_size)])
        self.output_layer = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        x = self.output_layer(x)
        return x
    
    def add_neuron(self, layer_index):
        layer = self.hidden_layers[layer_index]
        new_layer = nn.Linear(layer.in_features, layer.out_features + 1)
        new_layer.weight.data[:,:-1] = layer.weight.data
        new_layer.bias.data[:-1] = layer.bias.data
        self.hidden_layers[layer_index] = new_layer
    
    def prune_neuron(self, layer_index, neuron_index):
        layer = self.hidden_layers[layer_index]
        new_layer = nn.Linear(layer.in_features, layer.out_features - 1)
        new_layer.weight.data = torch.cat((layer.weight.data[:neuron_index], layer.weight.data[neuron_index+1:]), dim=0)
        new_layer.bias.data = torch.cat((layer.bias.data[:neuron_index], layer.bias.data[neuron_index+1:]), dim=0)
        self.hidden_layers[layer_index] = new_layer


# parameter
input_size = 784  # size of the data
hidden_size = 128
output_size = 10  # size of the result class


# create dynamic Neural Networks
dynamic_model = DynamicNN(input_size, hidden_size, output_size)

# add the layer of the dynamic_model
dynamic_model.add_neuron(0)

# Reduce neurons 0 to 5
dynamic_model.prune_neuron(0, 5)


# train
def train_dynamic(model, train_loader, criterion, optimizer, epochs=5):
    for epoch in range(epochs):
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        # 动态调整神经元
        if epoch % 2 == 0:  # 每两个epoch调整一次
            model.add_neuron(0)
        elif epoch % 3 == 0:
            model.prune_neuron(0, 5)

