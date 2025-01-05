#voladity forecasting of BlackRock stock

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

#simple neural network
class NeuralNetwork(nn.Module):
    
    def __init__(self, inputs, outputs):
        super(NeuralNetwork, self).__init__()
        self.layer = nn.Linear(inputs, 100)
        self.layer2 = nn.Linear(100, outputs)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.layer(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

class Normalize:
    
    def __init__(self):
        self.minx = None
        self.maxx = None
        
    def normalize(self, x):
        x = np.array(x)
        self.minx = min(x)
        self.maxx = max(x)
        return (x - self.minx) / (self.maxx - self.minx)
    
    def denormalize(self, x):
        return x * (self.maxx - self.minx) + self.minx

def BuildDataSet(percent):
    window = 150
    result = []
    for i in range(window, len(percent)):
        hold = percent[i - window : i]
        result.append(np.std(hold))
    return result

def BuildTrainTest(vol):
    window = 100
    output = 30
    inputs = []
    outputs = []
    
    for i in range(window, len(vol) - output + 1):
        curr_input = vol[i - window : i]
        curr_output = vol[i : i + output]
        inputs.append(curr_input)
        outputs.append(curr_output)
        
    IN = [torch.tensor(i, dtype = torch.float32) for i in inputs]
    OUT = [torch.tensor(i, dtype = torch.float32) for i in outputs]
    TEST = [torch.tensor(vol[-window:], dtype = torch.float32)]
    return torch.stack(IN), torch.stack(OUT), torch.stack((TEST))

normal = Normalize()
model = NeuralNetwork(100,30)

data = pd.read_csv('Volatility_Data.csv')[::-1]

close = data['adjClose'].values

delta = close[1:] / close[:-1] - 1.0

Volatility = BuildDataSet(delta)
nVol = normal.normalize(Volatility)

TIN, TOUT, TTEST = BuildTrainTest(nVol)

#using Adams optimizer

learning_rate = 0.001

optimizer = optim.Adam(model.parameters(), lr = learning_rate)
criterion = nn.MSELoss()

epochs = 500
for epoch in range(epochs):
    output = model(TIN)
    loss = criterion(output, TOUT)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(epoch, loss.item())

with torch.no_grad():
    testout = model(TTEST)

#print(testout.numpy()[0])

nvolatility = testout.numpy()[0]
predicted_volatility = normal.denormalize(nvolatility)

print(predicted_volatility)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.set_title("Volatility Forecast for BlackRock")
ax.set_xlabel("Time")
ax.set_ylabel("Volatility")

kVolatility = Volatility[-100:]

ux = list(range(len(kVolatility)))
px = list(range(len(kVolatility), len(kVolatility) + len(predicted_volatility)))

ax.plot(ux, kVolatility, color = 'red')
ax.plot(px, predicted_volatility, color = 'limegreen')

plt.show()
