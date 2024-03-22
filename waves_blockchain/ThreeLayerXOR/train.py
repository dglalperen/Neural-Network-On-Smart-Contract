import torch
from torch import nn
from torch.optim import Adam
from ThreeLayerXORNet import ThreeLayerXORNet

# Initialize the network
model = ThreeLayerXORNet()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=0.1)

# XOR input and output
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float)
Y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float)

# Training loop
for epoch in range(10000):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, Y)
    loss.backward()
    optimizer.step()

    if epoch % 1000 == 0:
        print(f'Epoch [{epoch}/10000] Loss: {loss.item()}')

# Save the trained model
torch.save(model.state_dict(), 'three_layer_xor_net.pth')
